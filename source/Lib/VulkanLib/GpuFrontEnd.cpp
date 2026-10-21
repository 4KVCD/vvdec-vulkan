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
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


------------------------------------------------------------------------------------------- */

/** \file     GpuFrontEnd.cpp
    \brief    The records the CPU's parse writes for the GPU's front end, and the tables its shaders read (see
              GpuFrontEnd.h)
*/

#include "GpuFrontEnd.h"
#include "GpuPictureMemory.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/LoopFilter.h"
#include "CommonLib/MotionInfo.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Rom.h"
#include "CommonLib/Slice.h"
#include "CommonLib/Unit.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace vvdec
{
namespace gpu
{

// Returns the CU record's flags (CuRecord::Flags) of cu.
static uint32_t cuFlags( const CodingUnit& cu )
{
  uint32_t flags = 0;
  flags |= cu.mergeFlag() ? CuRecord::kMerge : 0;
  flags |= cu.skip() ? CuRecord::kSkip : 0;
  flags |= cu.mmvdFlag() ? CuRecord::kMmvd : 0;
  flags |= cu.affineFlag() ? CuRecord::kAffine : 0;
  flags |= cu.geoFlag() ? CuRecord::kGeo : 0;
  flags |= cu.ciipFlag() ? CuRecord::kCiip : 0;
  flags |= cu.chType() == CHANNEL_TYPE_CHROMA ? CuRecord::kChroma : 0;
  flags |= cu.affineType() == AFFINEMODEL_6PARAM ? CuRecord::kAffine6 : 0;
  return flags;
}

// Returns the record of cu, a CU of CTU ctuRsAddr. Word 16, the range of the CU's TUs, is left for writeCuRecords.
static CuRecord cuRecord( const CodingStructure& cs, const CodingUnit& cu, int ctuRsAddr )
{
  CuRecord       r{};
  const Position pos  = cu.lumaPos();
  const Size     size = cu.lumaSize();
  r.w[0] = uint32_t( pos.x ) | uint32_t( pos.y ) << 16;
  r.w[1] = uint32_t( size.width ) | uint32_t( size.height ) << 8 | uint32_t( cu.predMode() ) << 16 | cuFlags( cu ) << 18;
  r.w[2] = uint32_t( cu.mergeIdx() ) | uint32_t( cu.mmvdIdx & 63 ) << 3 | uint32_t( cu.geoSplitDir & 63 ) << 9 | uint32_t( cu.geoMergeIdx0() ) << 15
           | uint32_t( cu.geoMergeIdx1() ) << 18 | uint32_t( cu.imv() ) << 21 | uint32_t( cu.BcwIdx() ) << 23 | uint32_t( cu.interDir() ) << 26
           | uint32_t( cu.mvpIdx[0] & 1 ) << 28 | uint32_t( cu.mvpIdx[1] & 1 ) << 29;
  r.w[3] = uint32_t( uint8_t( cu.refIdx[0] ) ) | uint32_t( uint8_t( cu.refIdx[1] ) ) << 8 | uint32_t( uint8_t( cu.qp ) ) << 16
           | uint32_t( uint8_t( cu.chromaQpAdj ) ) << 24;
  for( int list = 0; list < 2; list++ )
  {
    for( int k = 0; k < 3; k++ )
    {
      r.w[4 + 6 * list + 2 * k]     = uint32_t( cu.mv[list][k].getHor() );
      r.w[4 + 6 * list + 2 * k + 1] = uint32_t( cu.mv[list][k].getVer() );
    }
  }
  r.w[17] = uint32_t( cu.chType() ) | uint32_t( cu.treeType() ) << 1 | uint32_t( cu.modeType() ) << 3 | uint32_t( cu.ispMode() ) << 5
            | uint32_t( cu.lfnstIdx() ) << 7 | ( cu.mipFlag() ? 1u << 9 : 0u ) | ( cu.mipTransposedFlag() ? 1u << 10 : 0u )
            | uint32_t( cu.multiRefIdx() ) << 11 | ( cu.colorTransform() ? 1u << 13 : 0u ) | uint32_t( cu.bdpcmMode() ) << 14
            | uint32_t( cu.bdpcmModeChroma() ) << 16 | ( cu.rootCbf() ? 1u << 18 : 0u ) | uint32_t( cu.sbtInfo() ) << 19
            | ( cu.planeCbf( 0 ) ? 1u << 27 : 0u ) | ( cu.planeCbf( 1 ) ? 1u << 28 : 0u ) | ( cu.planeCbf( 2 ) ? 1u << 29 : 0u );
  r.w[18] = uint32_t( uint8_t( cu.intraDir[0] ) ) | uint32_t( uint8_t( cu.intraDir[1] ) ) << 8 | uint32_t( cu.qtDepth ) << 16
            | uint32_t( cu.depth ) << 20;
  if( cu.blocks.size() > 1 && cu.blocks[COMPONENT_Cb].valid() )
  {
    const CompArea& b = cu.blocks[COMPONENT_Cb];
    r.w[19]           = uint32_t( b.x ) | uint32_t( b.y ) << 16;
    r.w[20]           = uint32_t( b.width ) | uint32_t( b.height ) << 8 | 1u << 16;
  }
  r.w[20] |= cu.blocks[COMPONENT_Y].valid() ? 1u << 17 : 0u;
  // Where the CU's DMVR refinements start among its CTU's (set for luma CUs of 8x8 and more), and the neighbours that
  // the parse found available.
  const ptrdiff_t dmvrOff = cu.mvdL0SubPuOff - ptrdiff_t( ctuRsAddr ) * ptrdiff_t( cs.pcv->num8x8CtuBlks );
  r.w[21] = ( dmvrOff >= 0 && dmvrOff < 0x10000 ? uint32_t( dmvrOff ) : 0u ) | ( cu.left ? 1u << 16 : 0u ) | ( cu.above ? 1u << 17 : 0u )
            | ( cu.smvdMode() ? 1u << 18 : 0u );
  return r;
}

// Returns the record of tu, a TU of the CU of index cuIndex in its CTU.
static TuRecord tuRecord( const TransformUnit& tu, uint32_t cuIndex )
{
  TuRecord t{};
  if( tu.blocks[COMPONENT_Y].valid() )
  {
    const CompArea& b = tu.blocks[COMPONENT_Y];
    t.w[0]            = uint32_t( b.x ) | uint32_t( b.y ) << 16;
    t.w[1]            = uint32_t( b.width ) | uint32_t( b.height ) << 8 | 1u << 16;
  }
  if( tu.blocks.size() > 1 && tu.blocks[COMPONENT_Cb].valid() )
  {
    const CompArea& b = tu.blocks[COMPONENT_Cb];
    t.w[2]            = uint32_t( b.x ) | uint32_t( b.y ) << 16;
    t.w[3]            = uint32_t( b.width ) | uint32_t( b.height ) << 8 | 1u << 16;
  }
  t.w[4] = uint32_t( tu.cbf ) | uint32_t( tu.jointCbCr ) << 3 | uint32_t( tu.mtsIdx( 0 ) ) << 5 | uint32_t( tu.mtsIdx( 1 ) ) << 8
           | uint32_t( tu.mtsIdx( 2 ) ) << 9 | uint32_t( tu.chType() ) << 10;
  t.w[5] = uint32_t( uint8_t( tu.chromaQp[0] ) ) | uint32_t( uint8_t( tu.chromaQp[1] ) ) << 8;
  t.w[6] = uint32_t( tu.maxScanPosX[0] ) | uint32_t( tu.maxScanPosX[1] ) << 8 | uint32_t( tu.maxScanPosX[2] ) << 16;
  t.w[7] = uint32_t( tu.maxScanPosY[0] ) | uint32_t( tu.maxScanPosY[1] ) << 8 | uint32_t( tu.maxScanPosY[2] ) << 16;
  t.w[8] = cuIndex;
  return t;
}

void writeCuRecords( CodingStructure& cs, int ctuRsAddr, std::vector<CuRecord>& out, std::vector<TuRecord>& tus )
{
  for( const CodingUnit& cu: cs.traverseCUs( ctuRsAddr ) )
  {
    CuRecord       r       = cuRecord( cs, cu, ctuRsAddr );
    const uint32_t firstTu = uint32_t( tus.size() );
    for( const TransformUnit* tu = &cu.firstTU; tu; tu = tu->next )   // (the CU's last TU ends the chain)
    {
      tus.push_back( tuRecord( *tu, uint32_t( out.size() ) ) );
    }
    const uint32_t numTus = uint32_t( tus.size() ) - firstTu;
    CHECK( numTus > 0xffff, "too many TUs in a CU" );
    r.w[16] = firstTu | numTus << 16;
    out.push_back( r );
  }
}

// Returns the index of slice among its picture's slices.
static uint32_t sliceIndex( const CodingStructure& cs, const Slice* slice )
{
  const auto& slices = cs.picture->slices;
  for( size_t i = 0; i < slices.size(); i++ )
  {
    if( slices[i] == slice )
    {
      return uint32_t( i );
    }
  }
  CHECK( true, "a CTU's slice not among its picture's" );
  return 0;
}

void writeCtuRecords( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, const std::vector<std::vector<TuRecord>>& tus,
                      CtuRecord* out )
{
  const PPS& pps     = *cs.pps;
  uint32_t   first   = 0;
  uint32_t   firstTu = 0;
  for( int ctu = 0; ctu < (int) cs.pcv->sizeInCtus; ctu++ )
  {
    const int  ctuX  = ctu % cs.pcv->widthInCtus;
    const int  ctuY  = ctu / cs.pcv->widthInCtus;
    const auto count = uint32_t( cus[ctu].size() );
    CHECK( count > 0xffff, "too many CUs in a CTU" );
    out[ctu].w[0] = first;
    out[ctu].w[1] = count | sliceIndex( cs, cs.getCtuData( ctu ).slice ) << 16;
    out[ctu].w[2] = ( pps.ctuIsTileColBd( ctuX ) ? 1u : 0u ) | pps.getTileIdx( uint32_t( ctuX ), uint32_t( ctuY ) ) << 16;
    out[ctu].w[3] = firstTu;
    first += count;
    firstTu += uint32_t( tus[ctu].size() );
  }
}

size_t sliceTableWords( const CodingStructure& cs )
{
  return size_t( cs.pcv->sizeInCtus ) + kSliceRecordWords * cs.picture->slices.size();
}

const Picture* colocatedPicture( const Slice& slice )
{
  if( slice.isIntra() )
  {
    return nullptr;
  }
  const RefPicList colList = RefPicList( slice.isInterB() ? 1 - slice.getColFromL0Flag() : 0 );
  if( int( slice.getColRefIdx() ) >= slice.getNumRefIdx( colList ) )
  {
    return nullptr;
  }
  return slice.getRefPic( colList, int( slice.getColRefIdx() ) );
}

// The words of a slice's record in the slice table (mider.slang reads them)
enum SliceRecordWord : size_t
{
  kSlicePoc            = 0,
  kSliceFlags          = 1,    // the reference counts, the slice's flags and the colocated reference index
  kSliceMaxCands       = 2,    // the maximum IBC, GPM and affine merge candidates, a byte each
  kSliceIndependentIdx = 3,
  kSliceLongTermUsed   = 4,    // bit 16 * list + refIdx: the reference is used as a long-term one
  kSliceLongTermRpl    = 5,    // bit 16 * list + refIdx: the reference picture list marks it long-term
  kSliceColMotion      = 6,    // 2 words: the colocated picture's motion
  kSliceColTable       = 8,    // 2 words: the colocated picture's slice table
  kSliceRefPocs        = 16,   // [16 * list + refIdx]: the reference pictures' POCs
};

void writeSliceTable( const CodingStructure& cs, uint32_t* table )
{
  const auto& slices = cs.picture->slices;
  const int   ctus   = (int) cs.pcv->sizeInCtus;
  for( int ctu = 0; ctu < ctus; ctu++ )
  {
    table[ctu] = sliceIndex( cs, cs.getCtuData( ctu ).slice );
  }
  for( size_t i = 0; i < slices.size(); i++ )
  {
    const Slice& s   = *slices[i];
    uint32_t*    rec = table + ctus + kSliceRecordWords * i;
    memset( rec, 0, kSliceRecordWords * sizeof( uint32_t ) );

    // The colocated picture, when its motion and its slice table are within the GPU's reach.
    const Picture* col    = colocatedPicture( s );
    const bool     hasCol = col && col->cs && col->cs->m_colMiAllocation && col->cs->m_gpuSliceAllocation;

    rec[kSlicePoc]   = uint32_t( s.getPOC() );
    rec[kSliceFlags] = uint32_t( s.getNumRefIdx( REF_PIC_LIST_0 ) ) | uint32_t( s.getNumRefIdx( REF_PIC_LIST_1 ) ) << 8 | ( s.isInterB() ? 1u << 16 : 0u )
                       | ( s.getCheckLDC() ? 1u << 17 : 0u ) | ( s.getColFromL0Flag() ? 1u << 18 : 0u )
                       | ( s.getPicHeader()->getEnableTMVPFlag() ? 1u << 19 : 0u ) | ( s.getPicHeader()->getDisFracMMVD() ? 1u << 20 : 0u )
                       | ( hasCol ? 1u << 21 : 0u ) | ( s.isIRAP() ? 1u << 22 : 0u ) | ( s.getDeblockingFilterDisable() ? 1u << 23 : 0u )
                       | ( s.getColRefIdx() & 15 ) << 24;
    rec[kSliceMaxCands] =
      s.getPicHeader()->getMaxNumIBCMergeCand() | s.getPicHeader()->getMaxNumGeoCand() << 8 | s.getPicHeader()->getMaxNumAffineMergeCand() << 16;
    rec[kSliceIndependentIdx] = s.getIndependentSliceIdx();
    for( int list = 0; list < 2; list++ )
    {
      for( int refIdx = 0; refIdx < s.getNumRefIdx( RefPicList( list ) ) && refIdx < 16; refIdx++ )
      {
        rec[kSliceLongTermUsed] |= s.getIsUsedAsLongTerm( list, refIdx ) ? 1u << ( 16 * list + refIdx ) : 0u;
        rec[kSliceLongTermRpl] |= s.getRPL( RefPicList( list ) )->isRefPicLongterm( refIdx ) ? 1u << ( 16 * list + refIdx ) : 0u;
        rec[kSliceRefPocs + 16 * list + refIdx] = uint32_t( s.getRefPOC( RefPicList( list ), refIdx ) );
      }
    }
    if( hasCol )
    {
      const VkDeviceAddress motion   = hostAllocationAddress( col->cs->m_colMiAllocation );
      const VkDeviceAddress colTable = hostAllocationAddress( col->cs->m_gpuSliceAllocation );
      rec[kSliceColMotion]     = uint32_t( motion );
      rec[kSliceColMotion + 1] = uint32_t( motion >> 32 );
      rec[kSliceColTable]      = uint32_t( colTable );
      rec[kSliceColTable + 1]  = uint32_t( colTable >> 32 );
    }
  }
}

// Returns the number of pieces that a block side of n samples splits into when it is cut into powers of two (one per
// bit set: SBT's 3/4 sides split into two).
static size_t pieces( uint32_t n )
{
  size_t count = 0;
  for( ; n; n &= n - 1 )
  {
    count++;
  }
  return count;
}

// Returns the number of bands of rows that a job of the given size is cut into (IntraJob::bands).
static size_t bandsOf( uint32_t width, uint32_t height )
{
  return size_t( IntraJob::bands( int( width ), int( height ) ) );
}

FrontCapacities frontCapacities( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, const std::vector<std::vector<TuRecord>>& tus )
{
  FrontCapacities caps;
  const bool      chroma = isChromaEnabled( cs.pcv->chrFormat );
  const int       sx     = chroma ? getComponentScaleX( COMPONENT_Cb, cs.pcv->chrFormat ) : 0;
  const int       sy     = chroma ? getComponentScaleY( COMPONENT_Cb, cs.pcv->chrFormat ) : 0;
  for( size_t ctu = 0; ctu < cus.size(); ctu++ )
  {
    for( const CuRecord& r: cus[ctu] )
    {
      const uint32_t mode    = r.predMode();
      const uint32_t flags   = r.flags();
      const uint32_t w       = r.width();
      const uint32_t h       = r.height();
      const uint32_t firstTu = r.firstTu();
      const uint32_t numTus  = r.numTus();
      const size_t   tiles16 = size_t( ( w + 15 ) / 16 ) * ( ( h + 15 ) / 16 );
      if( mode == MODE_INTER && r.hasLuma() )
      {
        // GPM's tiles, the regular tiles, the affine pairs or SbTMVP's sub-PUs (the subblock merge CUs are affine
        // ones).
        caps.inter += flags & CuRecord::kAffine ? std::max<size_t>( 2 * tiles16, size_t( std::max<uint32_t>( w / 8, 1 ) ) * std::max<uint32_t>( h / 8, 1 ) )
                                                : tiles16;
        if( flags & CuRecord::kCiip )
        {
          caps.intra += 3;
          caps.bands += bandsOf( w, h ) + 2 * bandsOf( w >> sx, h >> sy );
        }
        // The edges of the affine subblock grid
        if( flags & CuRecord::kAffine )
        {
          caps.edges += 2 * ( size_t( w / 8 ) * ( h / 4 ) + size_t( h / 8 ) * ( w / 4 ) );
        }
      }
      for( uint32_t t = firstTu; t < firstTu + numTus; t++ )
      {
        const TuRecord& tu    = tus[ctu][t];
        const bool      luma  = tu.hasLuma();
        const bool      chr   = tu.hasChroma();
        const uint32_t  lw    = tu.width();
        const uint32_t  lh    = tu.height();
        const uint32_t  cw    = tu.chromaWidth();
        const uint32_t  ch    = tu.chromaHeight();
        const int       comps = ( luma ? 1 : 0 ) + ( chr ? 2 : 0 );
        // Per component, a residual job or the pieces of zeros. Then an intra or IBC job per component, or LMCS's
        // chroma pieces.
        caps.trafo += ( luma ? pieces( lw ) * pieces( lh ) : 0 ) + ( chr ? 2 * pieces( cw ) * pieces( ch ) : 0 );
        if( mode != MODE_INTER || r.rootCbf() )
        {
          caps.intra += size_t( comps ) + ( chr ? 2 * pieces( cw ) * pieces( ch ) : 0 );
          caps.bands += ( luma ? bandsOf( lw, lh ) : 0 ) + ( chr ? 2 * bandsOf( cw, ch ) + 2 * pieces( cw ) * pieces( ch ) * bandsOf( cw, ch ) : 0 );
        }
        // The edges along the TU's left and top sides, luma's and chroma's (on the luma grid)
        caps.edges += ( luma ? lw / 4 + lh / 4 : 0 ) + ( chr ? ( cw << sx ) / 4 + ( ch << sy ) / 4 : 0 );
      }
    }
  }
  caps.trafo += kFrontCapacitySpare;
  caps.inter += kFrontCapacitySpare;
  caps.intra += kFrontCapacitySpare;
  caps.bands += kFrontCapacitySpare;
  caps.edges += kFrontCapacitySpare;
  return caps;
}

// Where each of jobs.slang's T_* tables starts among the job tables' words
enum JobTable : size_t
{
  kTableLfnstLut       = 0,      // g_lfnstLut
  kTableChroma422Modes = 128,    // g_chroma422IntraAngleMappingTable
  kTableGeoParams      = 256,    // per GPM split: g_GeoParams' angle | distance << 8
  kTableGeoMask        = 320,    // per GPM angle: g_angle2mask
  kTableGeoMirror      = 352,    // per GPM angle: g_angle2mirror
  kTableGeoOffsets     = 384,    // per GPM split, log2 height and log2 width: g_weightOffset's x | y << 16
  kTableBcwWeights     = 1408,   // per BCW index: list 1's weight
};

void writeJobTables( uint32_t* out )
{
  memset( out, 0, kJobTableWords * sizeof( uint32_t ) );
  for( int i = 0; i < NUM_INTRA_MODE + NUM_EXT_LUMA_MODE - 1; i++ )
  {
    out[kTableLfnstLut + i] = g_lfnstLut[i];
  }
  for( int i = 0; i < NUM_INTRA_MODE; i++ )
  {
    out[kTableChroma422Modes + i] = g_chroma422IntraAngleMappingTable[i];
  }
  for( int split = 0; split < GEO_NUM_PARTITION_MODE; split++ )
  {
    out[kTableGeoParams + split] = uint32_t( g_GeoParams[split][0] ) | uint32_t( g_GeoParams[split][1] ) << 8;
    for( int h = 0; h < GEO_NUM_CU_SIZE; h++ )
    {
      for( int w = 0; w < GEO_NUM_CU_SIZE; w++ )
      {
        out[kTableGeoOffsets + ( split * 4 + h ) * 4 + w] =
          uint32_t( uint16_t( g_weightOffset[split][h][w][0] ) ) | uint32_t( uint16_t( g_weightOffset[split][h][w][1] ) ) << 16;
      }
    }
  }
  for( int a = 0; a < GEO_NUM_ANGLES; a++ )
  {
    out[kTableGeoMask + a]   = uint32_t( int32_t( g_angle2mask[a] ) );
    out[kTableGeoMirror + a] = uint32_t( int32_t( g_angle2mirror[a] ) );
  }
  for( int i = 0; i < BCW_NUM; i++ )
  {
    out[kTableBcwWeights + i] = uint32_t( int32_t( getBcwWeight( g_BcwInternBcw[i], REF_PIC_LIST_1 ) ) );
  }
}

// Where the reference pictures' words start in a slice's kJobSliceWords words (see GpuFrontEnd.h)
static constexpr size_t kJobSliceRefInfo = 32;
static constexpr size_t kJobSliceWeights = 64;

void writeJobSlices( const CodingStructure& cs, const RefTable& refs, uint32_t* out )
{
  const auto& slices = cs.picture->slices;
  const PPS&  pps    = *cs.pps;
  for( size_t i = 0; i < slices.size(); i++ )
  {
    const Slice& s   = *slices[i];
    uint32_t*    rec = out + kJobSliceWords * i;
    memset( rec, 0, kJobSliceWords * sizeof( uint32_t ) );
    rec[0] = uint32_t( s.getSliceType() ) | ( s.getLmcsEnabledFlag() ? 4u : 0u ) | ( s.getDepQuantEnabledFlag() ? 8u : 0u )
             | ( s.getExplicitScalingListUsed() ? 16u : 0u );
    for( int list = 0; list < 2; list++ )
    {
      for( int refIdx = 0; refIdx < s.getNumRefIdx( RefPicList( list ) ) && refIdx < 16; refIdx++ )
      {
        const Picture* ref = s.getRefPic( RefPicList( list ), refIdx );
        const WPScalingParam* wp = nullptr;
        s.getWpScaling( RefPicList( list ), refIdx, wp );
        const bool sameSize = ref->cs->pps->getPicWidthInLumaSamples() == pps.getPicWidthInLumaSamples()
                              && ref->cs->pps->getPicHeightInLumaSamples() == pps.getPicHeightInLumaSamples();
        uint32_t info = uint32_t( refs.slot( ref ) & 0xff ) | ( ref->isRefScaled( &pps ) ? 1u << 8 : 0u ) | ( sameSize ? 1u << 9 : 0u );
        for( int c = 0; c < MAX_NUM_COMPONENT; c++ )
        {
          info |= wp[c].bPresentFlag ? 1u << ( 10 + c ) : 0u;
          rec[kJobSliceWeights + ( 16 * list + refIdx ) * 3 + c] = uint32_t( uint16_t( int16_t( wp[c].iWeight ) ) )
                                                                   | uint32_t( uint8_t( int8_t( wp[c].iOffset ) ) ) << 16
                                                                   | ( wp[c].uiLog2WeightDenom & 15 ) << 24;
        }
        rec[kJobSliceRefInfo + 16 * list + refIdx] = info;
      }
    }
  }
}

}   // namespace gpu
}   // namespace vvdec
