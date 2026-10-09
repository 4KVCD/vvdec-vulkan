/* -----------------------------------------------------------------------------
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or
other Intellectual Property Rights other than the copyrights concerning
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2018-2026, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The VVdeC Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

------------------------------------------------------------------------------------------- */

/** \file     GpuParse.cpp
    \brief    the GPU's parse of the slice data (shaders/parse.slang): its inputs, the comparison with the CPU's
*/

#include "GpuParse.h"
#include "GpuContext.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Contexts.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Rom.h"
#include "CommonLib/Slice.h"

#include <cstdlib>
#include <cstring>

namespace vvdec
{
namespace gpu
{

// the context sets' first contexts (CTX_...) and their count, as parse.slang has them
#include "shaders/parsectx.slang"

namespace
{
// parse.slang's tables (keep in step with it)
enum PicWord : int
{
  kPicWidthInCtus  = 0,
  kPicHeightInCtus = 1,
  kPicLog2Ctu      = 2,
  kPicWidth        = 3,
  kPicHeight       = 4,
  kPicChromaFormat = 5,
  kPicQpBdOffset   = 6,
  kPicLog2MinCb    = 7,
  kPicMaxTbSize    = 8,
  kPicWpp          = 9,
  kPicLog2MaxTs    = 10,
  kPicTools        = 11,
  kPicMaxMergeCand = 12,
  kPicMaxGeoCand   = 13,
  kPicMaxIbcCand   = 14,
  kPicUseDqp       = 15,
  kPicCqpListLen   = 16,
  kPicTrRange      = 17,
  kPicCuSlots      = 18,
  kPicTuSlots      = 19,
  kPicTileCols     = 20,
  kPicCqpOffsets   = 24,
  kPicChromaQpMap  = 160,
};

enum Tool : int
{
  kToolSbt        = 1 << 0,
  kToolAffine     = 1 << 1,
  kToolAffineType = 1 << 2,
  kToolAmvr       = 1 << 3,
  kToolAffineAmvr = 1 << 4,
  kToolBcw        = 1 << 5,
  kToolCiip       = 1 << 6,
  kToolGeo        = 1 << 7,
  kToolMmvd       = 1 << 8,
  kToolSmvd       = 1 << 9,
  kToolIsp        = 1 << 10,
  kToolMrl        = 1 << 11,
  kToolMip        = 1 << 12,
  kToolLfnst      = 1 << 13,
  kToolMtsIntra   = 1 << 14,
  kToolMtsInter   = 1 << 15,
  kToolLmChroma   = 1 << 16,
  kToolIbc        = 1 << 17,
  kToolAct        = 1 << 18,
  kToolBdpcm      = 1 << 19,
  kToolTs         = 1 << 20,
  kToolJointCbCr  = 1 << 21,
  kToolMts        = 1 << 22,
};

enum SliceWord : int
{
  kSlType          = 0,
  kSlQp            = 1,
  kSlFlags         = 2,
  kSlNumRef        = 3,
  kSlSymRef        = 4,
  kSlMaxAffMerge   = 5,
  kSlAlf           = 6,
  kSlSubdiv        = 7,
  kSlChromaQpDelta = 8,
  kSlWp            = 11,
  kSlPartLuma      = 12,
  kSlPartChroma    = 13,
};

enum SliceFlag : uint32_t
{
  kSlSaoLuma     = 1 << 0,
  kSlSaoChroma   = 1 << 1,
  kSlAlfY        = 1 << 2,
  kSlAlfCb       = 1 << 3,
  kSlAlfCr       = 1 << 4,
  kSlCcAlfCb     = 1 << 5,
  kSlCcAlfCr     = 1 << 6,
  kSlDepQuant    = 1 << 7,
  kSlSignHiding  = 1 << 8,
  kSlTsResOff    = 1 << 9,
  kSlChromaQpAdj = 1 << 10,
  kSlMvdL1Zero   = 1 << 11,
  kSlBiDirPred   = 1 << 12,
  kSlCheckLDC    = 1 << 13,
  kSlDualTree    = 1 << 14,
};

enum CtuFlag : uint32_t
{
  kCtuRestart   = 1,
  kCtuSyncLoad  = 2,
  kCtuSyncSave  = 4,
  kCtuTerminate = 8,
  kCtuWaitAbove = 16,
};

constexpr size_t kSliceStride = kParseSliceWords + 2 * CTX_COUNT;

// Partitioner::initCtu's limits of a channel (GDR_ADJ): maxBTD | maxBtSize << 8 | maxTtSize << 16 | minQtSize << 24
uint32_t partitionLimits( const Slice& slice, const SPS& sps, const PicHeader& ph, ChannelType ch )
{
  const int valIdx = slice.isIntra() ? ( slice.getSPS()->getUseDualITree() ? int( ch ) << 1 : 0 ) : 1;
  unsigned  maxBTD, maxBt, maxTt, minQt;
  if( ph.getSplitConsOverrideFlag() )
  {
    maxBTD = ph.getMaxMTTHierarchyDepth( slice.getSliceType(), ch );
    maxBt  = ph.getMaxBTSize( slice.getSliceType(), ch );
    maxTt  = ph.getMaxTTSize( slice.getSliceType(), ch );
    minQt  = ph.getMinQTSize( slice.getSliceType(), ch );
  }
  else
  {
    maxBTD = sps.getMaxMTTHierarchyDepths()[valIdx];
    maxBt  = sps.getMaxBTSizes()[valIdx];
    maxTt  = sps.getMaxTTSizes()[valIdx];
    minQt  = sps.getMinQTSizes()[valIdx];
  }
  CHECK( maxBTD > 255 || maxBt > 255 || maxTt > 255 || minQt > 255, "partition limits out of range" );
  return maxBTD | maxBt << 8 | maxTt << 16 | minQt << 24;
}
}   // namespace

bool parseShadowRequested()
{
  static const bool on = Options::fromEnvironment().parse;
  return on;
}

bool buildParseInput( const CodingStructure& cs, ParseInput& in )
{
  const PreCalcValues& pcv  = *cs.pcv;
  const SPS&           sps  = *cs.sps;
  const PPS&           pps  = *cs.pps;
  const Picture&       pic  = *cs.picture;
  const int            ctus = int( pcv.sizeInCtus );
  const bool           wpp  = sps.getEntropyCodingSyncEnabledFlag();
  const bool           entryPoints = sps.getEntryPointsPresentFlag();
  const ChromaFormat   chFmt = sps.getChromaFormatIdc();

  in.bits.clear();
  in.streams.clear();
  in.numStreams = 0;
  in.ctuList.clear();
  in.ctuInfo.assign( 2 * size_t( ctus ), 0u );
  in.slices.assign( kSliceStride * pic.slices.size(), 0u );
  in.pic.assign( kParsePicWords, 0 );

  // a CTU's records at most: its luma's 4x4 blocks, its chroma's of 16 samples (the chroma tree's CUs)
  const int ctuSize  = int( pcv.maxCUWidth );
  const int chromaW  = chFmt == CHROMA_400 ? 0 : ctuSize >> getChannelTypeScaleX( CH_C, chFmt );
  const int chromaH  = chFmt == CHROMA_400 ? 0 : ctuSize >> getChannelTypeScaleY( CH_C, chFmt );
  in.cuSlots         = uint32_t( ctuSize * ctuSize / 16 + chromaW * chromaH / 16 );
  in.tuSlots         = in.cuSlots;
  in.wppSlots        = uint32_t( pcv.heightInCtus * pps.getNumTileColumns() );
  in.contexts        = CTX_COUNT;

  // the CTUs' slices and tiles
  for( int ctu = 0; ctu < ctus; ctu++ )
  {
    const int x             = ctu % int( pcv.widthInCtus );
    const int y             = ctu / int( pcv.widthInCtus );
    in.ctuInfo[2 * ctu + 1] = uint32_t( pps.getTileIdx( uint32_t( x ), uint32_t( y ) ) ) | ( pps.ctuIsTileColBd( x ) ? 1u << 16 : 0u );
  }
  for( size_t si = 0; si < pic.slices.size(); si++ )
  {
    const Slice* slice = pic.slices[si];
    for( uint32_t i = 0; i < slice->getNumCtuInSlice(); i++ )
    {
      in.ctuInfo[2 * slice->getCtuAddrInSlice( int( i ) )] = uint32_t( si );
    }
  }
  auto sameSliceTile = [&]( int a, int b ) {
    return in.ctuInfo[2 * a] == in.ctuInfo[2 * b] && ( in.ctuInfo[2 * a + 1] & 0xffff ) == ( in.ctuInfo[2 * b + 1] & 0xffff );
  };

  for( size_t si = 0; si < pic.slices.size(); si++ )
  {
    Slice*                      slice = pic.slices[si];
    const std::vector<uint8_t>& data  = slice->parseTaskParams.gpuSliceData;
    if( data.empty() )
    {
      return false;
    }
    const PicHeader& ph          = *slice->getPicHeader();
    const uint32_t   sliceOffset = uint32_t( in.bits.size() );
    in.bits.insert( in.bits.end(), data.begin(), data.end() );

    // the slice's words
    uint32_t* w      = in.slices.data() + kSliceStride * si;
    w[kSlType]       = uint32_t( slice->getSliceType() );
    w[kSlQp]         = uint32_t( slice->getSliceQp() );
    uint32_t flags   = 0;
    flags |= slice->getSaoEnabledFlag( CHANNEL_TYPE_LUMA ) ? kSlSaoLuma : 0;
    flags |= slice->getSaoEnabledFlag( CHANNEL_TYPE_CHROMA ) ? kSlSaoChroma : 0;
    flags |= slice->getAlfEnabledFlag( COMPONENT_Y ) ? kSlAlfY : 0;
    flags |= slice->getAlfEnabledFlag( COMPONENT_Cb ) ? kSlAlfCb : 0;
    flags |= slice->getAlfEnabledFlag( COMPONENT_Cr ) ? kSlAlfCr : 0;
    flags |= slice->getCcAlfEnabledFlag( 1 ) ? kSlCcAlfCb : 0;
    flags |= slice->getCcAlfEnabledFlag( 2 ) ? kSlCcAlfCr : 0;
    flags |= slice->getDepQuantEnabledFlag() ? kSlDepQuant : 0;
    flags |= slice->getSignDataHidingEnabledFlag() ? kSlSignHiding : 0;
    flags |= slice->getTSResidualCodingDisabledFlag() ? kSlTsResOff : 0;
    flags |= slice->getUseChromaQpAdj() ? kSlChromaQpAdj : 0;
    flags |= ph.getMvdL1ZeroFlag() ? kSlMvdL1Zero : 0;
    flags |= slice->getBiDirPred() ? kSlBiDirPred : 0;
    flags |= slice->getCheckLDC() ? kSlCheckLDC : 0;
    flags |= slice->isIntra() && sps.getUseDualITree() ? kSlDualTree : 0;
    w[kSlFlags]       = flags;
    w[kSlNumRef]      = uint32_t( slice->getNumRefIdx( REF_PIC_LIST_0 ) ) | uint32_t( slice->getNumRefIdx( REF_PIC_LIST_1 ) ) << 8;
    w[kSlSymRef]      = ( uint32_t( slice->getSymRefIdx( 0 ) ) & 0xff ) | ( uint32_t( slice->getSymRefIdx( 1 ) ) & 0xff ) << 8;
    w[kSlMaxAffMerge] = ph.getMaxNumAffineMergeCand();
    uint32_t alf      = uint32_t( slice->getNumAlfAps() ) & 0xff;
    if( slice->getAlfEnabledFlag( COMPONENT_Cb ) || slice->getAlfEnabledFlag( COMPONENT_Cr ) )
    {
      const APS* aps = slice->getAlfAPS( slice->getAlfApsIdChroma() );
      CHECK( aps == nullptr, "APS not initialized" );
      alf |= ( uint32_t( aps->getAlfAPSParam().numAlternativesChroma ) & 0xff ) << 8;
    }
    for( int comp = 1; comp <= 2; comp++ )
    {
      if( slice->getCcAlfEnabledFlag( comp ) )
      {
        const APS* aps = slice->getAlfAPS( comp == 1 ? slice->getCcAlfCbApsId() : slice->getCcAlfCrApsId() );
        CHECK( aps == nullptr, "APS not initialized" );
        alf |= ( uint32_t( aps->getCcAlfAPSParam().ccAlfFilterCount[comp - 1] ) & 0xff ) << ( comp == 1 ? 16 : 24 );
      }
    }
    w[kSlAlf]               = alf;
    w[kSlSubdiv]            = slice->getCuQpDeltaSubdiv() | slice->getCuChromaQpOffsetSubdiv() << 8;
    w[kSlChromaQpDelta]     = uint32_t( slice->getSliceChromaQpDelta( COMPONENT_Cb ) );
    w[kSlChromaQpDelta + 1] = uint32_t( slice->getSliceChromaQpDelta( COMPONENT_Cr ) );
    w[kSlChromaQpDelta + 2] = uint32_t( slice->getSliceChromaQpDelta( JOINT_CbCr ) );
    uint32_t wp             = 0;
    for( int list = 0; list < 2; list++ )
    {
      for( int refIdx = 0; refIdx < slice->getNumRefIdx( RefPicList( list ) ) && refIdx < 16; refIdx++ )
      {
        const WPScalingParam* p = nullptr;
        slice->getWpScaling( RefPicList( list ), refIdx, p );
        if( p && ( p[COMPONENT_Y].bPresentFlag || p[COMPONENT_Cb].bPresentFlag || p[COMPONENT_Cr].bPresentFlag ) )
        {
          wp |= 1u << ( 16 * list + refIdx );
        }
      }
    }
    w[kSlWp]          = wp;
    w[kSlPartLuma]    = partitionLimits( *slice, sps, ph, CHANNEL_TYPE_LUMA );
    w[kSlPartChroma]  = slice->isIntra() && sps.getUseDualITree() ? partitionLimits( *slice, sps, ph, CHANNEL_TYPE_CHROMA ) : 0u;
    // the initial contexts (CABACReader::initCtxModels)
    SliceType initType = slice->getSliceType();
    if( slice->getPPS()->getCabacInitPresentFlag() && slice->getCabacInitFlag() )
    {
      initType = initType == P_SLICE ? B_SLICE : ( initType == B_SLICE ? P_SLICE : initType );
    }
    Ctx ctx;
    ctx.init( slice->getSliceQp(), int( initType ) );
    for( int k = 0; k < CTX_COUNT; k++ )
    {
      w[kParseSliceWords + k]             = ctx[k].gpuState();
      w[kParseSliceWords + CTX_COUNT + k] = ctx[k].gpuRates();
    }

    // the substreams (DecSlice::parseSlice)
    const uint32_t numSubstreams = slice->getNumberOfSubstreamSizes() + 1;
    uint32_t       streamOffset  = sliceOffset;
    uint32_t       subStream     = 0;
    auto           streamSize    = [&]( uint32_t s ) {
      if( s + 1 < numSubstreams )
      {
        return slice->getSubstreamSize( s );
      }
      return uint32_t( sliceOffset + data.size() - streamOffset );
    };
    auto beginStream = [&]() {
      const size_t at = in.streams.size();
      in.streams.resize( at + kParseStreamWords, 0u );
      in.streams[at + 0] = streamOffset;
      in.streams[at + 1] = streamSize( subStream );
      in.streams[at + 2] = uint32_t( in.ctuList.size() / 2 );
      in.streams[at + 4] = uint32_t( si );
      in.numStreams++;
    };
    auto endStream = [&]() {
      const size_t at    = in.streams.size() - kParseStreamWords;
      in.streams[at + 3] = uint32_t( in.ctuList.size() / 2 ) - in.streams[at + 2];
      CHECK( in.streams[at + 0] + in.streams[at + 1] > sliceOffset + data.size(), "a substream past its slice's data" );
    };
    beginStream();
    const int widthInCtus = int( pcv.widthInCtus );
    for( uint32_t ctuIdx = 0; ctuIdx < slice->getNumCtuInSlice(); ctuIdx++ )
    {
      const int      ctuRs     = int( slice->getCtuAddrInSlice( int( ctuIdx ) ) );
      const int      x         = ctuRs % widthInCtus;
      const int      y         = ctuRs / widthInCtus;
      const uint32_t tileCol   = pps.ctuToTileCol( x );
      const uint32_t tileRow   = pps.ctuToTileRow( y );
      const int      tileX0    = int( pps.getTileColumnBd( tileCol ) );
      const int      tileY0    = int( pps.getTileRowBd( tileRow ) );
      const int      tileW     = int( pps.getTileColumnWidth( tileCol ) );
      const int      tileH     = int( pps.getTileRowHeight( tileRow ) );
      const bool     aboveSame = y > 0 && sameSliceTile( ctuRs - widthInCtus, ctuRs );
      uint32_t       flags     = 0;
      if( x == tileX0 && y == tileY0 )
      {
        flags |= ctuIdx != 0 ? kCtuRestart : 0;
      }
      else if( x == tileX0 && wpp )
      {
        flags |= ctuIdx != 0 ? kCtuRestart : 0;
        flags |= aboveSame ? kCtuSyncLoad : 0;
      }
      if( x == tileX0 && wpp )
      {
        flags |= kCtuSyncSave;
      }
      if( wpp && aboveSame )
      {
        flags |= kCtuWaitAbove;
      }
      const bool last = ctuIdx + 1 == slice->getNumCtuInSlice();
      const bool ends = !last && x + 1 == tileX0 + tileW && ( y + 1 == tileY0 + tileH || wpp );
      if( ends && !entryPoints )
      {
        flags |= kCtuTerminate;
      }
      const uint32_t slot = uint32_t( y ) * uint32_t( pps.getNumTileColumns() ) + tileCol;
      in.ctuList.push_back( uint32_t( ctuRs ) );
      in.ctuList.push_back( flags | slot << 8 );
      if( ends && entryPoints )
      {
        endStream();
        streamOffset += streamSize( subStream );
        subStream++;
        CHECK( subStream >= numSubstreams, "more substreams than entry points" );
        beginStream();
      }
    }
    endStream();
  }

  // the picture's
  int32_t* p           = in.pic.data();
  p[kPicWidthInCtus]   = int32_t( pcv.widthInCtus );
  p[kPicHeightInCtus]  = int32_t( pcv.heightInCtus );
  p[kPicLog2Ctu]       = int32_t( getLog2( pcv.maxCUWidth ) );
  p[kPicWidth]         = int32_t( pcv.lumaWidth );
  p[kPicHeight]        = int32_t( pcv.lumaHeight );
  p[kPicChromaFormat]  = int32_t( chFmt );
  p[kPicQpBdOffset]    = sps.getQpBDOffset();
  p[kPicLog2MinCb]     = int32_t( sps.getLog2MinCodingBlockSize() );
  p[kPicMaxTbSize]     = int32_t( sps.getMaxTbSize() );
  p[kPicWpp]           = wpp ? 1 : 0;
  p[kPicLog2MaxTs]     = int32_t( sps.getLog2MaxTransformSkipBlockSize() );
  int tools            = 0;
  tools |= sps.getUseSBT() ? kToolSbt : 0;
  tools |= sps.getUseAffine() ? kToolAffine : 0;
  tools |= sps.getUseAffineType() ? kToolAffineType : 0;
  tools |= sps.getAMVREnabledFlag() ? kToolAmvr : 0;
  tools |= sps.getAffineAmvrEnabledFlag() ? kToolAffineAmvr : 0;
  tools |= sps.getUseBcw() ? kToolBcw : 0;
  tools |= sps.getUseCiip() ? kToolCiip : 0;
  tools |= sps.getUseGeo() ? kToolGeo : 0;
  tools |= sps.getUseMMVD() ? kToolMmvd : 0;
  tools |= sps.getUseSMVD() ? kToolSmvd : 0;
  tools |= sps.getUseISP() ? kToolIsp : 0;
  tools |= sps.getUseMRL() ? kToolMrl : 0;
  tools |= sps.getUseMIP() ? kToolMip : 0;
  tools |= sps.getUseLFNST() ? kToolLfnst : 0;
  tools |= sps.getUseIntraMTS() ? kToolMtsIntra : 0;
  tools |= sps.getUseInterMTS() ? kToolMtsInter : 0;
  tools |= sps.getUseLMChroma() ? kToolLmChroma : 0;
  tools |= sps.getIBCFlag() ? kToolIbc : 0;
  tools |= sps.getUseColorTrans() ? kToolAct : 0;
  tools |= sps.getBDPCMEnabledFlag() ? kToolBdpcm : 0;
  tools |= sps.getTransformSkipEnabledFlag() ? kToolTs : 0;
  tools |= sps.getJointCbCrEnabledFlag() ? kToolJointCbCr : 0;
  tools |= sps.getUseMTS() ? kToolMts : 0;
  p[kPicTools]        = tools;
  p[kPicMaxMergeCand] = int32_t( sps.getMaxNumMergeCand() );
  p[kPicMaxGeoCand]   = int32_t( sps.getMaxNumGeoCand() );
  p[kPicMaxIbcCand]   = int32_t( sps.getMaxNumIBCMergeCand() );
  p[kPicUseDqp]       = pps.getUseDQP() ? 1 : 0;
  p[kPicCqpListLen]   = pps.getChromaQpOffsetListLen();
  p[kPicTrRange]      = sps.getMaxLog2TrDynamicRange( CHANNEL_TYPE_LUMA ) | sps.getMaxLog2TrDynamicRange( CHANNEL_TYPE_CHROMA ) << 8;
  p[kPicCuSlots]      = int32_t( in.cuSlots );
  p[kPicTuSlots]      = int32_t( in.tuSlots );
  p[kPicTileCols]     = int32_t( pps.getNumTileColumns() );
  if( chFmt != CHROMA_400 )
  {
    const ComponentID comps[3] = { COMPONENT_Cb, COMPONENT_Cr, JOINT_CbCr };
    for( int c = 0; c < 3; c++ )
    {
      p[kPicCqpOffsets + c] = pps.getQpOffset( comps[c] );
      for( int i = 1; i <= pps.getChromaQpOffsetListLen() && i <= 6; i++ )
      {
        p[kPicCqpOffsets + 3 + ( i - 1 ) * 3 + c] = pps.getChromaQpOffsetListEntry( i ).get( comps[c] );
      }
      const int qpBdOffset = sps.getQpBDOffset();
      CHECK( MAX_QP + 1 + qpBdOffset > 128, "the chroma QP mapping does not fit" );
      // (joint Cb-Cr's table only with it: vvdec makes it then)
      for( int qp = -qpBdOffset; qp <= MAX_QP && ( c < 2 || sps.getJointCbCrEnabledFlag() ); qp++ )
      {
        p[kPicChromaQpMap + c * 128 + qp + qpBdOffset] = sps.getMappedChromaQpValue( comps[c], qp );
      }
    }
  }
  return true;
}

static int scanTabOffset( int log2W, int log2H )
{
  return ( ( 1 << log2W ) - 1 ) * 31 + ( 1 << log2W ) * ( ( 1 << log2H ) - 1 );
}

void writeParseScans( uint32_t* out )
{
  memset( out, 0, kParseScanTab * sizeof( uint32_t ) );
  for( int log2W = 0; log2W <= 4; log2W++ )
  {
    for( int log2H = 0; log2H <= 4; log2H++ )
    {
      const int       n     = 1 << ( log2W + log2H );
      const uint16_t* plain = g_scanOrder[SCAN_UNGROUPED][log2W][log2H];
      uint32_t*       t     = out + scanTabOffset( log2W, log2H );
      for( int i = 0; plain && i < n; i++ )
      {
        t[i] |= plain[i];
        t[plain[i]] |= uint32_t( i ) << 16;
      }
    }
  }
}

// parse.slang's grouped scans (the groups' diagonal scan, each group's) against vvdec's g_scanOrder[SCAN_GROUPED_4x4],
// over the positions the parse reaches (the zero-out's groups)
static bool checkParseScans()
{
  std::vector<uint32_t> tab( kParseScanTab );
  writeParseScans( tab.data() );
  int bad = 0;
  for( int log2W = 0; log2W <= 6; log2W++ )
  {
    for( int log2H = 0; log2H <= 6; log2H++ )
    {
      const uint16_t* grouped = g_scanOrder[SCAN_GROUPED_4x4][log2W][log2H];
      if( !grouped )
      {
        continue;
      }
      const int cgW = g_log2SbbSize[log2W][log2H][0];
      const int cgH = g_log2SbbSize[log2W][log2H][1];
      const int wig = std::min( 32, 1 << log2W ) >> cgW;
      const int hig = std::min( 32, 1 << log2H ) >> cgH;
      for( int scanPos = 0; scanPos < wig * hig << ( cgW + cgH ); scanPos++ )
      {
        const int cg    = int( tab[scanTabOffset( getLog2( wig ), getLog2( hig ) ) + ( scanPos >> ( cgW + cgH ) )] & 0xffff );
        const int inner = int( tab[scanTabOffset( cgW, cgH ) + ( scanPos & ( ( 1 << ( cgW + cgH ) ) - 1 ) )] & 0xffff );
        const int cgX   = cg % wig;
        const int cgY   = cg / wig;
        const int pos   = ( ( cgY << cgH ) + ( inner >> cgW ) ) * ( 1 << log2W ) + ( cgX << cgW ) + ( inner & ( ( 1 << cgW ) - 1 ) );
        if( pos != grouped[scanPos] && bad++ < 4 )
        {
          msg( WARNING, "GPU parse: the scan of %dx%d at %d: %d, vvdec's %d\n", 1 << log2W, 1 << log2H, scanPos, pos, grouped[scanPos] );
        }
      }
    }
  }
  return bad == 0;
}

bool checkParseContexts()
{
  struct Set
  {
    const char* name;
    unsigned    vvdec;
    int         gpu;
  };
  const Set sets[] = {
    { "SplitFlag", Ctx::SplitFlag(), CTX_SplitFlag },
    { "SplitQtFlag", Ctx::SplitQtFlag(), CTX_SplitQtFlag },
    { "SplitHvFlag", Ctx::SplitHvFlag(), CTX_SplitHvFlag },
    { "Split12Flag", Ctx::Split12Flag(), CTX_Split12Flag },
    { "ModeConsFlag", Ctx::ModeConsFlag(), CTX_ModeConsFlag },
    { "SkipFlag", Ctx::SkipFlag(), CTX_SkipFlag },
    { "MergeFlag", Ctx::MergeFlag(), CTX_MergeFlag },
    { "RegularMergeFlag", Ctx::RegularMergeFlag(), CTX_RegularMergeFlag },
    { "MergeIdx", Ctx::MergeIdx(), CTX_MergeIdx },
    { "MmvdFlag", Ctx::MmvdFlag(), CTX_MmvdFlag },
    { "MmvdMergeIdx", Ctx::MmvdMergeIdx(), CTX_MmvdMergeIdx },
    { "MmvdStepMvpIdx", Ctx::MmvdStepMvpIdx(), CTX_MmvdStepMvpIdx },
    { "PredMode", Ctx::PredMode(), CTX_PredMode },
    { "MultiRefLineIdx", Ctx::MultiRefLineIdx(), CTX_MultiRefLineIdx },
    { "IPredMode[0]", Ctx::IPredMode[0](), CTX_IPredMode_0 },
    { "IPredMode[1]", Ctx::IPredMode[1](), CTX_IPredMode_1 },
    { "IntraLumaPlanarFlag", Ctx::IntraLumaPlanarFlag(), CTX_IntraLumaPlanarFlag },
    { "CclmModeFlag", Ctx::CclmModeFlag(), CTX_CclmModeFlag },
    { "CclmModeIdx", Ctx::CclmModeIdx(), CTX_CclmModeIdx },
    { "MipFlag", Ctx::MipFlag(), CTX_MipFlag },
    { "DeltaQP", Ctx::DeltaQP(), CTX_DeltaQP },
    { "InterDir", Ctx::InterDir(), CTX_InterDir },
    { "RefPic", Ctx::RefPic(), CTX_RefPic },
    { "SubblockMergeFlag", Ctx::SubblockMergeFlag(), CTX_SubblockMergeFlag },
    { "AffineFlag", Ctx::AffineFlag(), CTX_AffineFlag },
    { "AffineType", Ctx::AffineType(), CTX_AffineType },
    { "AffMergeIdx", Ctx::AffMergeIdx(), CTX_AffMergeIdx },
    { "BcwIdx", Ctx::BcwIdx(), CTX_BcwIdx },
    { "Mvd", Ctx::Mvd(), CTX_Mvd },
    { "BDPCMMode", Ctx::BDPCMMode(), CTX_BDPCMMode },
    { "QtRootCbf", Ctx::QtRootCbf(), CTX_QtRootCbf },
    { "ACTFlag", Ctx::ACTFlag(), CTX_ACTFlag },
    { "QtCbf[0]", Ctx::QtCbf[0](), CTX_QtCbf_0 },
    { "QtCbf[1]", Ctx::QtCbf[1](), CTX_QtCbf_1 },
    { "QtCbf[2]", Ctx::QtCbf[2](), CTX_QtCbf_2 },
    { "SigCoeffGroup[0]", Ctx::SigCoeffGroup[0](), CTX_SigCoeffGroup_0 },
    { "SigCoeffGroup[1]", Ctx::SigCoeffGroup[1](), CTX_SigCoeffGroup_1 },
    { "SigFlag[0]", Ctx::SigFlag[0](), CTX_SigFlag_0 },
    { "SigFlag[1]", Ctx::SigFlag[1](), CTX_SigFlag_1 },
    { "SigFlag[2]", Ctx::SigFlag[2](), CTX_SigFlag_2 },
    { "SigFlag[3]", Ctx::SigFlag[3](), CTX_SigFlag_3 },
    { "SigFlag[4]", Ctx::SigFlag[4](), CTX_SigFlag_4 },
    { "SigFlag[5]", Ctx::SigFlag[5](), CTX_SigFlag_5 },
    { "ParFlag[0]", Ctx::ParFlag[0](), CTX_ParFlag_0 },
    { "ParFlag[1]", Ctx::ParFlag[1](), CTX_ParFlag_1 },
    { "GtxFlag[0]", Ctx::GtxFlag[0](), CTX_GtxFlag_0 },
    { "GtxFlag[1]", Ctx::GtxFlag[1](), CTX_GtxFlag_1 },
    { "GtxFlag[2]", Ctx::GtxFlag[2](), CTX_GtxFlag_2 },
    { "GtxFlag[3]", Ctx::GtxFlag[3](), CTX_GtxFlag_3 },
    { "LastX[0]", Ctx::LastX[0](), CTX_LastX_0 },
    { "LastX[1]", Ctx::LastX[1](), CTX_LastX_1 },
    { "LastY[0]", Ctx::LastY[0](), CTX_LastY_0 },
    { "LastY[1]", Ctx::LastY[1](), CTX_LastY_1 },
    { "MVPIdx", Ctx::MVPIdx(), CTX_MVPIdx },
    { "SmvdFlag", Ctx::SmvdFlag(), CTX_SmvdFlag },
    { "SaoMergeFlag", Ctx::SaoMergeFlag(), CTX_SaoMergeFlag },
    { "SaoTypeIdx", Ctx::SaoTypeIdx(), CTX_SaoTypeIdx },
    { "LFNSTIdx", Ctx::LFNSTIdx(), CTX_LFNSTIdx },
    { "MTSIndex", Ctx::MTSIndex(), CTX_MTSIndex },
    { "ISPMode", Ctx::ISPMode(), CTX_ISPMode },
    { "SbtFlag", Ctx::SbtFlag(), CTX_SbtFlag },
    { "SbtQuadFlag", Ctx::SbtQuadFlag(), CTX_SbtQuadFlag },
    { "SbtHorFlag", Ctx::SbtHorFlag(), CTX_SbtHorFlag },
    { "SbtPosFlag", Ctx::SbtPosFlag(), CTX_SbtPosFlag },
    { "ChromaQpAdjFlag", Ctx::ChromaQpAdjFlag(), CTX_ChromaQpAdjFlag },
    { "ChromaQpAdjIdc", Ctx::ChromaQpAdjIdc(), CTX_ChromaQpAdjIdc },
    { "ImvFlag", Ctx::ImvFlag(), CTX_ImvFlag },
    { "ctbAlfFlag", Ctx::ctbAlfFlag(), CTX_ctbAlfFlag },
    { "ctbAlfAlternative", Ctx::ctbAlfAlternative(), CTX_ctbAlfAlternative },
    { "AlfUseTemporalFilt", Ctx::AlfUseTemporalFilt(), CTX_AlfUseTemporalFilt },
    { "CcAlfFilterControlFlag", Ctx::CcAlfFilterControlFlag(), CTX_CcAlfFilterControlFlag },
    { "CiipFlag", Ctx::CiipFlag(), CTX_CiipFlag },
    { "IBCFlag", Ctx::IBCFlag(), CTX_IBCFlag },
    { "JointCbCrFlag", Ctx::JointCbCrFlag(), CTX_JointCbCrFlag },
    { "TsSigCoeffGroup", Ctx::TsSigCoeffGroup(), CTX_TsSigCoeffGroup },
    { "TsSigFlag", Ctx::TsSigFlag(), CTX_TsSigFlag },
    { "TsParFlag", Ctx::TsParFlag(), CTX_TsParFlag },
    { "TsGtxFlag", Ctx::TsGtxFlag(), CTX_TsGtxFlag },
    { "TsLrg1Flag", Ctx::TsLrg1Flag(), CTX_TsLrg1Flag },
    { "TsResidualSign", Ctx::TsResidualSign(), CTX_TsResidualSign },
  };
  bool ok = checkParseScans() && Ctx::NumberOfContexts == unsigned( CTX_COUNT );
  if( !ok )
  {
    msg( WARNING, "GPU parse: %u contexts, parsectx.slang has %d\n", Ctx::NumberOfContexts, CTX_COUNT );
  }
  for( const Set& s: sets )
  {
    if( s.vvdec != unsigned( s.gpu ) )
    {
      msg( WARNING, "GPU parse: context set %s at %u, parsectx.slang has %d\n", s.name, s.vvdec, s.gpu );
      ok = false;
    }
  }
  return ok;
}

void captureSaoAlf( const CodingStructure& cs, int ctuRsAddr, uint32_t out[7] )
{
  const CtuData& ctu = cs.getCtuData( ctuRsAddr );
  for( int comp = 0; comp < 3; comp++ )
  {
    const SAOOffset& o = ctu.saoParam[comp];
    out[2 * comp]      = uint32_t( o.modeIdc ) | uint32_t( o.typeIdc ) << 4 | uint32_t( o.typeAuxInfo ) << 8;
    uint32_t offs      = 0;
    if( o.modeIdc == SAO_MODE_NEW )
    {
      int v[4];
      if( o.typeIdc == SAO_TYPE_START_BO )
      {
        for( int k = 0; k < 4; k++ )
        {
          v[k] = o.offset[( o.typeAuxInfo + k ) % MAX_NUM_SAO_CLASSES];
        }
      }
      else
      {
        v[0] = o.offset[0];
        v[1] = o.offset[1];
        v[2] = o.offset[3];
        v[3] = o.offset[4];
      }
      for( int k = 0; k < 4; k++ )
      {
        offs |= ( uint32_t( v[k] ) & 0xff ) << ( 8 * k );
      }
    }
    out[2 * comp + 1] = offs;
  }
  const CtuAlfData& a = ctu.alfParam;
  out[6] = uint32_t( a.alfCtuEnableFlag[0] ? 1 : 0 ) | uint32_t( a.alfCtuEnableFlag[1] ? 2 : 0 ) | uint32_t( a.alfCtuEnableFlag[2] ? 4 : 0 )
           | uint32_t( a.alfCtbFilterIndex ) << 3 | uint32_t( a.alfCtuAlternative[0] ) << 8 | uint32_t( a.alfCtuAlternative[1] ) << 12
           | uint32_t( a.ccAlfFilterControl[0] ) << 16 | uint32_t( a.ccAlfFilterControl[1] ) << 20;
}

void compareParse( const CodingStructure& cs, const uint32_t* gpuCus, const uint32_t* gpuTus, const uint32_t* gpuCtus, const uint32_t* gpuCtuOut,
                   const uint32_t* gpuTotals, const std::vector<std::vector<CuRecord>>& cpuCus, const std::vector<std::vector<TuRecord>>& cpuTus,
                   const std::vector<uint32_t>& cpuSaoAlf, const uint32_t* levels, uint32_t errors, ParseStats& stats )
{
  const PreCalcValues& pcv  = *cs.pcv;
  const int            ctus = int( pcv.sizeInCtus );
  const int            poc  = cs.picture->poc;
  stats.pictures++;
  if( errors != 0 )
  {
    stats.errors++;
    if( stats.reported < 20 )
    {
      stats.reported++;
      msg( WARNING, "GPU parse: POC %d: %u substreams gave up\n", poc, errors );
    }
  }
  std::vector<CtuRecord> cpuCtus( static_cast<size_t>( ctus ) );
  writeCtuRecords( cs, cpuCus, cpuTus, cpuCtus.data() );
  for( int ctu = 0; ctu < ctus; ctu++ )
  {
    // the CTU's record
    stats.ctusCompared++;
    bool same = memcmp( cpuCtus[ctu].w, gpuCtus + 4 * ctu, 4 * sizeof( uint32_t ) ) == 0;
    // its SAO and ALF parameters
    if( !cpuSaoAlf.empty() )
    {
      same = same && memcmp( cpuSaoAlf.data() + 7 * ctu, gpuCtuOut + kParseCtuOutWords * ctu + 2, 7 * sizeof( uint32_t ) ) == 0;
    }
    if( !same )
    {
      stats.ctusDiffer++;
      if( stats.reported < 20 )
      {
        stats.reported++;
        const uint32_t* g = gpuCtus + 4 * ctu;
        msg( WARNING, "GPU parse: POC %d CTU %d: CPU %08x %08x %08x %08x, GPU %08x %08x %08x %08x\n", poc, ctu, cpuCtus[ctu].w[0], cpuCtus[ctu].w[1],
             cpuCtus[ctu].w[2], cpuCtus[ctu].w[3], g[0], g[1], g[2], g[3] );
        if( !cpuSaoAlf.empty() )
        {
          const uint32_t* c = cpuSaoAlf.data() + 7 * ctu;
          const uint32_t* o = gpuCtuOut + kParseCtuOutWords * ctu + 2;
          msg( WARNING, "  SAO/ALF CPU %08x %08x %08x %08x %08x %08x %08x, GPU %08x %08x %08x %08x %08x %08x %08x\n", c[0], c[1], c[2], c[3], c[4], c[5],
               c[6], o[0], o[1], o[2], o[3], o[4], o[5], o[6] );
        }
      }
    }
    // its CUs and TUs, when the GPU's has as many
    const uint32_t first   = gpuCtus[4 * ctu];
    const uint32_t count   = gpuCtus[4 * ctu + 1] & 0xffff;
    const uint32_t firstTu = gpuCtus[4 * ctu + 3];
    const auto&    cus     = cpuCus[ctu];
    const auto&    tus     = cpuTus[ctu];
    stats.cusCompared += int64_t( cus.size() );
    stats.tusCompared += int64_t( tus.size() );
    if( count != cus.size() || uint32_t( gpuCtuOut[kParseCtuOutWords * ctu + 1] ) != tus.size() )
    {
      stats.cusDiffer += int64_t( cus.size() );
      stats.tusDiffer += int64_t( tus.size() );
      continue;
    }
    for( size_t i = 0; i < cus.size(); i++ )
    {
      const uint32_t* g = gpuCus + 32 * ( first + i );
      if( memcmp( cus[i].w, g, sizeof( CuRecord ) ) != 0 )
      {
        stats.cusDiffer++;
        if( stats.reported < 20 )
        {
          stats.reported++;
          int w = 0;
          while( w < 32 && cus[i].w[w] == g[w] )
          {
            w++;
          }
          msg( WARNING, "GPU parse: POC %d CTU %d CU %d (%d,%d %dx%d, chroma (%d,%d)): word %d CPU %08x GPU %08x\n", poc, ctu, int( i ),
               int( cus[i].w[0] & 0xffff ), int( cus[i].w[0] >> 16 ), int( cus[i].w[1] & 0xff ), int( ( cus[i].w[1] >> 8 ) & 0xff ),
               int( cus[i].w[19] & 0xffff ), int( cus[i].w[19] >> 16 ), w, w < 32 ? cus[i].w[w] : 0u, w < 32 ? g[w] : 0u );
        }
      }
    }
    for( size_t i = 0; i < tus.size(); i++ )
    {
      const uint32_t* g = gpuTus + 16 * ( firstTu + i );
      if( memcmp( tus[i].w, g, sizeof( TuRecord ) ) != 0 )
      {
        stats.tusDiffer++;
        if( stats.reported < 20 )
        {
          stats.reported++;
          int w = 0;
          while( w < 16 && tus[i].w[w] == g[w] )
          {
            w++;
          }
          msg( WARNING, "GPU parse: POC %d CTU %d TU %d (%d,%d %dx%d) of CU %u: word %d CPU %08x GPU %08x\n", poc, ctu, int( i ),
               int( tus[i].w[0] & 0xffff ), int( tus[i].w[0] >> 16 ), int( tus[i].w[1] & 0xff ), int( ( tus[i].w[1] >> 8 ) & 0xff ), tus[i].w[8], w,
               w < 16 ? tus[i].w[w] : 0u, w < 16 ? g[w] : 0u );
        }
      }
    }
  }
  (void) gpuTotals;
  stats.levelsCompared += levels[0];
  stats.levelsDiffer += levels[1];
  if( levels[1] != 0 && stats.reported < 20 )
  {
    stats.reported++;
    const uint32_t  t = levels[2] - 1;
    const uint32_t* g = gpuTus + 16 * t;
    msg( WARNING, "GPU parse: POC %d levels: %u of %u differ, the first in TU %u (%d,%d %dx%d) component %u\n", poc, levels[1], levels[0], t,
         int( g[0] & 0xffff ), int( g[0] >> 16 ), int( g[1] & 0xff ), int( ( g[1] >> 8 ) & 0xff ), levels[3] );
  }
}

}   // namespace gpu
}   // namespace vvdec
