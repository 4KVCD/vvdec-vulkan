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

/** \file     GpuReconFilters.cpp
    \brief    GpuRecon's in-loop filters: LMCS's inverse mapping, deblocking, SAO and ALF (see GpuRecon.h)
*/

#include "GpuRecon.h"

#ifdef _MSC_VER
#  include <intrin.h>
#endif

#include "vvdec_gpu/GpuShaderIds.h"

#include "CommonLib/AdaptiveLoopFilter.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/LoopFilter.h"
#include "CommonLib/Picture.h"
#include "CommonLib/SampleAdaptiveOffset.h"
#include "CommonLib/Slice.h"

#include <cstring>

namespace vvdec
{
namespace gpu
{

// ====================================================================================================================
// LMCS
// ====================================================================================================================

// The push constants of lmcs.slang
struct LmcsPush
{
  VkDeviceAddress plane, lut, ctus;
  int32_t         stride, width, height, ctuLog2, widthInCtus;
};

void GpuRecon::recordLmcs( const CodingStructure& cs, const Planes& planes, VkDeviceAddress ctus )
{
  const PreCalcValues& pcv = *cs.pcv;
  LmcsPush             push{};
  push.plane       = planes.addr[COMPONENT_Y];
  push.lut         = m_lmcs.addr + m_lmcsCount * sizeof( int16_t );   // the inverse mapping
  push.ctus        = ctus;
  push.stride      = planes.stride[COMPONENT_Y];
  push.width       = pcv.lumaWidth;
  push.height      = pcv.lumaHeight;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  CHECK( pcv.lumaWidth % 4 != 0, "Vulkan: LMCS takes 4 luma samples at a time" );
  // one workgroup per 256 samples of a row
  dispatch( kShader_lmcs, push, uint32_t( ( pcv.lumaWidth + 255 ) / 256 ), uint32_t( pcv.lumaHeight ) );
  mark( "lmcs" );
}

// ====================================================================================================================
// Deblocking
// ====================================================================================================================

// The push constants of deblock.slang
struct DeblockPush
{
  VkDeviceAddress pic0, pic1, ctus, edges;
  int32_t         stride, width, height, ctuLog2, widthInCtus, bitDepth, dir, scaleX, scaleY, count;
  int32_t         ladfNum, ladfOffset[5], ladfBound[4];
  VkDeviceAddress countAt;
};
static_assert( sizeof( DeblockPush ) == 120, "DeblockPush: the layout of deblock.slang's Params" );
// deblock.slang's workgroups: a thread per edge, 64 threads each
static constexpr int kDeblockEdgesPerGroup = 64;

// the index of the lowest bit set in v (v must not be 0)
static inline int lowestBit( uint32_t v )
{
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward( &index, v );
  return int( index );
#else
  return __builtin_ctz( v );
#endif
}

void GpuRecon::listEdges( const CodingStructure& cs, int ctuRsAddr, EdgeLists& edges )
{
  edges.clear();
  const PreCalcValues& pcv = *cs.pcv;
  const CtuData&       ctu = cs.getCtuData( ctuRsAddr );
  if( ctu.slice->getDeblockingFilterDisable() )
  {
    return;
  }
  const int  n      = int( pcv.maxCUWidth ) >> 2;   // 4x4 blocks across a CTU (at most 32, so a row fits the bits of an int)
  const int  x0     = ( ctuRsAddr % int( pcv.widthInCtus ) ) * int( pcv.maxCUWidth );
  const int  y0     = ( ctuRsAddr / int( pcv.widthInCtus ) ) * int( pcv.maxCUHeight );
  const bool chroma = isChromaEnabled( pcv.chrFormat );
  const int  sx     = chroma ? getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) : 0;
  const int  sy     = chroma ? getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) : 0;
  const int  cols   = std::min( n, ( int( pcv.lumaWidth ) - x0 + 3 ) >> 2 );
  const int  rows   = std::min( n, ( int( pcv.lumaHeight ) - y0 + 3 ) >> 2 );
  for( int dir = 0; dir < 2; dir++ )
  {
    const LoopFilterParam* lfp = ctu.lfParam[dir];
    // The chroma segments on the luma grid: across the edge every 8 chroma samples, along it as many lines as a
    // 4x4 luma block has.
    const int across = dir == 0 ? ( ( 8 << sx ) / 4 * ( 4 >> sx ) ) << sx : ( ( 8 << sy ) / 4 * ( 4 >> sy ) ) << sy;
    const int gridX  = dir == 0 ? across : ( 4 >> sx ) << sx;
    const int gridY  = dir == 0 ? ( 4 >> sy ) << sy : across;
    CHECKD( ( gridX & ( gridX - 1 ) ) || ( gridY & ( gridY - 1 ) ), "the chroma grid is a power of 2" );
    // The 4x4 blocks inside the picture, row by row. (No divisions: this runs for every CTU, between its motion and
    // its jobs.)
    for( int r = 0; r < rows; r++ )
    {
      const LoopFilterParam* row      = lfp + r * n;
      const int              ly       = y0 + r * 4;
      const bool             chromaOn = chroma && ( ly & ( gridY - 1 ) ) == 0;
      // the row's blocks with an edge, as bits (no branch per block: most have no edge)
      uint32_t any = 0;
      for( int c = 0; c < cols; c++ )
      {
        any |= uint32_t( row[c].bs != 0 ) << c;
      }
      while( any )
      {
        const int              c  = lowestBit( any );
        any                      &= any - 1;
        const LoopFilterParam& p  = row[c];
        const int              lx = x0 + c * 4;
        if( p.bs & 3 )
        {
          edges.lists[dir].push_back(
            { uint32_t( lx ) | uint32_t( ly ) << 16, uint32_t( uint8_t( p.qp[0] ) ) | uint32_t( p.bs ) << 8 | uint32_t( p.sideMaxFiltLength ) << 16 } );
        }
        if( chromaOn && ( p.bs & 0x3c ) && ( lx & ( gridX - 1 ) ) == 0 )
        {
          edges.lists[2 + dir].push_back( { uint32_t( lx >> sx ) | uint32_t( ly >> sy ) << 16,
                                            uint32_t( uint8_t( p.qp[1] ) ) | uint32_t( uint8_t( p.qp[2] ) ) << 8 | uint32_t( p.bs ) << 16
                                              | uint32_t( p.flags ) << 24 } );
        }
      }
    }
  }
}

// The deblocking parameters: per CTU, 8 bytes (whether its slice disables deblocking, then the slice's beta and tc
// offsets of luma, Cb and Cr)
size_t GpuRecon::deblockParamSize( const CodingStructure& cs )
{
  return alignUp( size_t( cs.pcv->sizeInCtus ) * 8, 256 );
}

bool GpuRecon::prepareDeblock( const CodingStructure& cs, uint8_t* base )
{
  bool any = false;
  for( int ctuIdx = 0; ctuIdx < (int) cs.pcv->sizeInCtus; ctuIdx++ )
  {
    const Slice& slice = *cs.getCtuData( ctuIdx ).slice;
    int8_t*      ctu   = reinterpret_cast<int8_t*>( base + ctuIdx * 8 );
    ctu[0] = slice.getDeblockingFilterDisable() ? 1 : 0;
    ctu[1] = int8_t( slice.getDeblockingFilterBetaOffsetDiv2() );
    ctu[2] = int8_t( slice.getDeblockingFilterTcOffsetDiv2() );
    ctu[3] = int8_t( slice.getDeblockingFilterCbBetaOffsetDiv2() );
    ctu[4] = int8_t( slice.getDeblockingFilterCbTcOffsetDiv2() );
    ctu[5] = int8_t( slice.getDeblockingFilterCrBetaOffsetDiv2() );
    ctu[6] = int8_t( slice.getDeblockingFilterCrTcOffsetDiv2() );
    ctu[7] = 0;
    any |= !slice.getDeblockingFilterDisable();
  }
  return any;
}

void GpuRecon::uploadCpuEdges( const std::vector<EdgeLists>& edges, size_t first[5] )
{
  // the four lists, one after the other, of all the CTUs
  first[0] = 0;
  for( int l = 0; l < 4; l++ )
  {
    first[l + 1] = first[l];
    for( const EdgeLists& ctu: edges )
    {
      first[l + 1] += ctu.lists[l].size();
    }
  }
  if( m_cpuJobs.edges.size < first[4] * sizeof( Edge ) )
  {
    m_ctx->free( m_cpuJobs.edges );
    m_cpuJobs.edges = m_ctx->alloc( std::max<size_t>( first[4], 1024 ) * 2 * sizeof( Edge ), MemKind::Upload );
  }
  Edge* all = reinterpret_cast<Edge*>( m_cpuJobs.edges.mapped );
  for( int l = 0; l < 4; l++ )
  {
    for( const EdgeLists& ctu: edges )
    {
      memcpy( all, ctu.lists[l].data(), ctu.lists[l].size() * sizeof( Edge ) );
      all += ctu.lists[l].size();
    }
  }
}

// deblock.slang's push constants for the picture: the CTUs' parameters, the picture's geometry and bit depth, and LADF's
// intervals. The planes, the direction and the edge list are set per dispatch.
static void deblockPush( const CodingStructure& cs, VkDeviceAddress params, DeblockPush& push )
{
  const PreCalcValues& pcv    = *cs.pcv;
  const SPS&           sps    = *cs.sps;
  const bool           chroma = isChromaEnabled( pcv.chrFormat );
  push             = DeblockPush{};
  push.ctus        = params;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  push.bitDepth    = sps.getBitDepth();
  push.scaleX      = chroma ? getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) : 0;
  push.scaleY      = chroma ? getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) : 0;
  // LADF (luma-adaptive deblocking): its intervals' QP offsets and lower bounds
  if( sps.getLadfEnabled() )
  {
    push.ladfNum = sps.getLadfNumIntervals();
    for( int k = 0; k < push.ladfNum; k++ )
    {
      push.ladfOffset[k] = sps.getLadfQpOffset( k );
      if( k > 0 )
      {
        push.ladfBound[k - 1] = sps.getLadfIntervalLowerBound( k );
      }
    }
  }
}

// Dispatches shader on one edge list, a thread per edge: with the GPU front end, its list of pass (counted on the GPU);
// otherwise the CPU front end's list number list, as uploadCpuEdges laid the lists out (first: each list's first edge,
// then the total).
void GpuRecon::dispatchEdgeList( int shader, FrontPass pass, const size_t first[5], int list, DeblockPush& push )
{
  push.edges   = m_useFrontEnd ? m_frontEnd.edges.addr : m_cpuJobs.edges.addr + first[list] * sizeof( Edge );
  push.count   = m_useFrontEnd ? -1 : int32_t( first[list + 1] - first[list] );
  push.countAt = frontEndCount( pass );
  if( m_useFrontEnd )
  {
    bind( shader );
    pushConstants( push );
    dispatchIndirect( pass );
  }
  else if( push.count > 0 )
  {
    bind( shader );
    pushConstants( push );
    dispatchGroups( uint32_t( ( push.count + kDeblockEdgesPerGroup - 1 ) / kDeblockEdgesPerGroup ) );
  }
}

void GpuRecon::recordDeblock( const CodingStructure& cs, const Planes& planes, VkDeviceAddress params, const std::vector<EdgeLists>& edges )
{
  const PreCalcValues& pcv    = *cs.pcv;
  const bool           chroma = isChromaEnabled( pcv.chrFormat );
  // Luma's filters load and store 4 samples at a time, from multiples of 4, so the luma rows must be 8-byte aligned.
  // The chroma filters take both chroma planes with one stride.
  CHECK_FATAL( planes.addr[COMPONENT_Y] % 8 != 0 || planes.stride[COMPONENT_Y] % 4 != 0, "Vulkan: the luma rows are not 8-byte aligned" );
  CHECK( chroma && planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );

  // The CPU front end's edges are uploaded (the GPU front end's are in its buffer, with their counts on the GPU).
  size_t first[5] = { 0, 0, 0, 0, 0 };
  if( !m_useFrontEnd )
  {
    uploadCpuEdges( edges, first );
  }
  DeblockPush push;
  deblockPush( cs, params, push );
  // The lists: luma's vertical and horizontal edges (lists 0 and 1), then chroma's (lists 2 and 3).
  for( int dir = 0; dir < 2; dir++ )
  {
    if( dir == 1 )
    {
      computeBarrier();   // the horizontal edges are filtered after the vertical ones
    }
    push.dir    = dir;
    push.pic0   = planes.addr[COMPONENT_Y];
    push.pic1   = 0;
    push.stride = planes.stride[COMPONENT_Y];
    push.width  = pcv.lumaWidth;
    push.height = pcv.lumaHeight;
    dispatchEdgeList( dir == 0 ? kShader_deblock_luma_ver : kShader_deblock_luma_hor, FrontPass( kPassDeblockLumaVer + dir ), first, dir, push );
    if( chroma )
    {
      push.pic0   = planes.addr[COMPONENT_Cb];
      push.pic1   = planes.addr[COMPONENT_Cr];
      push.stride = planes.stride[COMPONENT_Cb];
      push.width  = pcv.lumaWidth >> push.scaleX;
      push.height = pcv.lumaHeight >> push.scaleY;
      dispatchEdgeList( kShader_deblock_chroma, FrontPass( kPassDeblockChromaVer + dir ), first, 2 + dir, push );
    }
    mark( dir == 0 ? "deblock ver" : "deblock hor" );
  }
}

// ====================================================================================================================
// SAO and ALF
// ====================================================================================================================

// The push constants of alf.slang
struct AlfPush
{
  VkDeviceAddress src0, dst0, src1, dst1, srcLuma, ctus, coeff, clip, ccCoeff;
  int32_t         srcStride, dstStride, lumaStride, width, height, lumaWidth, lumaHeight, ctuLog2, widthInCtus, bitDepth, scaleX, scaleY;
  int32_t         numVbVer, numVbHor, vbVer[3], vbHor[3];
  int32_t         aligned, pass;
};

// The push constants of sao.slang
struct SaoPush
{
  VkDeviceAddress plane0, plane1, plane2, ctus, list;
  int32_t         count;
  VkDeviceAddress ring;
  int32_t         stride0, stride12, width, height, ctuWidth, ctuHeight, widthInCtus, bitDepth, scaleX, scaleY, ringSlot, pass, aligned;
  int32_t         numVbVer, numVbHor, vbVer[3], vbHor[3];
};

// The picture's virtual boundaries (in luma samples) into the push constants of ALF or SAO
template<class Push>
static void setVirtualBoundaries( const PicHeader& picHeader, Push& push )
{
  push.numVbVer = push.numVbHor = 0;
  if( picHeader.getVirtualBoundariesPresentFlag() )
  {
    push.numVbVer = int32_t( picHeader.getNumVerVirtualBoundaries() );
    push.numVbHor = int32_t( picHeader.getNumHorVirtualBoundaries() );
    CHECK( push.numVbVer > 3 || push.numVbHor > 3, "too many virtual boundaries" );
    for( int i = 0; i < push.numVbVer; i++ )
    {
      push.vbVer[i] = int32_t( picHeader.getVirtualBoundariesPosX( i ) );
    }
    for( int i = 0; i < push.numVbHor; i++ )
    {
      push.vbHor[i] = int32_t( picHeader.getVirtualBoundariesPosY( i ) );
    }
  }
}

// SAO's parameters: per CTU and component, 8 int16_t (sao.slang): the type (-1: off), the band position, the
// neighbours' availability, then the four offsets. Then the list of the CTUs' components whose SAO is on, the ones
// sao.slang filters: per entry a uint32_t, the CTU's raster index | the component << 24.
static size_t saoListOffset( const CodingStructure& cs )
{
  return size_t( cs.pcv->sizeInCtus ) * 3 * 8 * sizeof( int16_t );
}

size_t GpuRecon::saoParamSize( const CodingStructure& cs )
{
  return alignUp( saoListOffset( cs ) + size_t( cs.pcv->sizeInCtus ) * 3 * sizeof( uint32_t ), 256 );
}

bool GpuRecon::prepareSao( CodingStructure& cs, SampleAdaptiveOffset& sao, uint8_t* base )
{
  const int numCtus = (int) cs.pcv->sizeInCtus;
  const int numComp = (int) getNumberValidComponents( cs.pcv->chrFormat );
  int16_t*  ctus    = reinterpret_cast<int16_t*>( base );
  uint32_t* list    = reinterpret_cast<uint32_t*>( base + saoListOffset( cs ) );
  uint32_t  entries = 0;

  // in raster order: a merged CTU copies the parameters of its left or above neighbour
  for( int ctuIdx = 0; ctuIdx < numCtus; ctuIdx++ )
  {
    sao.reconstructCtuParam( cs, ctuIdx );
  }

  for( int ctuIdx = 0; ctuIdx < numCtus; ctuIdx++ )
  {
    const SAOBlkParam& param = cs.getCtuData( ctuIdx ).saoParam;
    int16_t            avail = -1;   // worked out when a component of the CTU uses SAO
    for( int comp = 0; comp < 3; comp++ )
    {
      int16_t*         ctu    = ctus + ( ctuIdx * 3 + comp ) * 8;
      const SAOOffset& offset = param[comp];
      memset( ctu, 0, 8 * sizeof( int16_t ) );
      ctu[0] = -1;
      if( comp >= numComp || offset.modeIdc == SAO_MODE_OFF )
      {
        continue;
      }
      CHECK( offset.modeIdc != SAO_MODE_NEW || offset.typeIdc < SAO_TYPE_START_EO || offset.typeIdc > SAO_TYPE_BO, "unresolved SAO parameters" );
      if( avail < 0 )
      {
        avail = sao.neighbourAvailability( cs, ctuIdx );
      }
      list[entries++] = uint32_t( ctuIdx ) | uint32_t( comp ) << 24;
      ctu[0]          = int16_t( offset.typeIdc );
      ctu[2] = avail;
      if( offset.typeIdc == SAO_TYPE_BO )
      {
        ctu[1] = int16_t( offset.typeAuxInfo );
        for( int i = 0; i < 4; i++ )
        {
          ctu[4 + i] = int16_t( offset.offset[( offset.typeAuxInfo + i ) % NUM_SAO_BO_CLASSES] );
        }
      }
      else
      {
        ctu[4] = int16_t( offset.offset[0] );
        ctu[5] = int16_t( offset.offset[1] );
        ctu[6] = int16_t( offset.offset[3] );
        ctu[7] = int16_t( offset.offset[4] );
      }
    }
  }
  m_prepared.saoEntries = entries;
  return entries > 0;
}

void GpuRecon::recordSao( const CodingStructure& cs, const Planes& planes, VkDeviceAddress params )
{
  const PreCalcValues& pcv = *cs.pcv;
  // Per list entry, a ring: the rows above and below the CTU (its width and 2 samples more each), the columns left and
  // right of it
  const int    ringSlot = 2 * ( int( pcv.maxCUWidth ) + 2 ) + 2 * int( pcv.maxCUHeight );
  const size_t ringSize = size_t( m_prepared.saoEntries ) * size_t( ringSlot ) * sizeof( int16_t );
  m_ctx->ensure( m_saoRing, ringSize, MemKind::Device );
  SaoPush push{};
  push.plane0       = planes.addr[COMPONENT_Y];
  push.plane1       = planes.addr[COMPONENT_Cb];
  push.plane2       = planes.addr[COMPONENT_Cr];
  push.ctus         = params;
  push.list         = params + saoListOffset( cs );
  push.count        = int32_t( m_prepared.saoEntries );
  push.ring         = m_saoRing.addr;
  push.stride0      = planes.stride[COMPONENT_Y];
  push.stride12     = planes.stride[COMPONENT_Cb];
  push.width        = pcv.lumaWidth;
  push.height       = pcv.lumaHeight;
  push.ctuWidth     = pcv.maxCUWidth;
  push.ctuHeight    = pcv.maxCUHeight;
  push.widthInCtus  = pcv.widthInCtus;
  push.bitDepth     = cs.sps->getBitDepth();
  push.scaleX       = getComponentScaleX( COMPONENT_Cb, pcv.chrFormat );
  push.scaleY       = getComponentScaleY( COMPONENT_Cb, pcv.chrFormat );
  push.ringSlot     = ringSlot;
  setVirtualBoundaries( *cs.picHeader, push );
  // the planes whose groups of 4 samples the shader may load and store whole (8-byte aligned)
  for( int comp = 0; comp < (int) getNumberValidComponents( pcv.chrFormat ); comp++ )
  {
    if( planes.addr[comp] % 8 == 0 && planes.stride[comp] % 4 == 0 )
    {
      push.aligned |= 1 << comp;
    }
  }
  CHECK( getNumberValidComponents( pcv.chrFormat ) > 1 && planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );
  // pass 0 saves the rings, pass 1 filters the CTUs in place: a workgroup per entry each
  bind( kShader_sao );
  push.pass = 0;
  pushConstants( push );
  dispatchJobs( m_prepared.saoEntries );
  computeBarrier();
  push.pass = 1;
  pushConstants( push );
  dispatchJobs( m_prepared.saoEntries );
  mark( "sao" );
}

// ALF's parameters: the filter tables (luma sets [16 fixed + 8 APS][transpose][class][13], chroma
// [APS][alternative][7], CC-ALF [APS][component][filter][8]), then 8 bytes per CTU (alf.slang)
constexpr size_t kAlfSetSize     = MAX_NUM_ALF_TRANSPOSE_ID * MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF;
constexpr size_t kAlfChromaSize  = ALF_CTB_MAX_NUM_APS * MAX_NUM_ALF_ALTERNATIVES_CHROMA * MAX_NUM_ALF_CHROMA_COEFF;
constexpr size_t kAlfCcSize      = ALF_CTB_MAX_NUM_APS * 2 * MAX_NUM_CC_ALF_FILTERS * MAX_NUM_CC_ALF_CHROMA_COEFF;
constexpr size_t kAlfLumaCoeff   = 0;
constexpr size_t kAlfLumaClip    = kAlfLumaCoeff + NUM_TOTAL_FILTER_SETS * kAlfSetSize * sizeof( int16_t );
constexpr size_t kAlfChromaCoeff = kAlfLumaClip + NUM_TOTAL_FILTER_SETS * kAlfSetSize * sizeof( int16_t );
constexpr size_t kAlfChromaClip  = kAlfChromaCoeff + kAlfChromaSize * sizeof( int16_t );
constexpr size_t kAlfCcCoeff     = kAlfChromaClip + kAlfChromaSize * sizeof( int16_t );
constexpr size_t kAlfCtus        = alignUp( kAlfCcCoeff + kAlfCcSize * sizeof( int16_t ), 256 );
static_assert( kAlfSetSize == 4 * 25 * 13, "the layout of shaders/alf.slang" );

size_t GpuRecon::alfParamSize( const CodingStructure& cs )
{
  return alignUp( kAlfCtus + size_t( cs.pcv->sizeInCtus ) * 8, 256 );
}

// The filter tables at base: the fixed luma sets, then the luma, chroma and CC-ALF filters of the APSs the slices
// use, each at its APS's place
static void writeAlfFilterTables( const CodingStructure& cs, const AdaptiveLoopFilter& alfFilter, uint8_t* base )
{
  int16_t* lumaCoeff   = reinterpret_cast<int16_t*>( base + kAlfLumaCoeff );
  int16_t* lumaClip    = reinterpret_cast<int16_t*>( base + kAlfLumaClip );
  int16_t* chromaCoeff = reinterpret_cast<int16_t*>( base + kAlfChromaCoeff );
  int16_t* chromaClip  = reinterpret_cast<int16_t*>( base + kAlfChromaClip );
  int16_t* ccCoeff     = reinterpret_cast<int16_t*>( base + kAlfCcCoeff );
  for( int set = 0; set < NUM_FIXED_FILTER_SETS; set++ )
  {
    memcpy( lumaCoeff + set * kAlfSetSize, alfFilter.getFixedFilterSetCoeff( set ), kAlfSetSize * sizeof( int16_t ) );
    memcpy( lumaClip  + set * kAlfSetSize, alfFilter.getClipDefault(),               kAlfSetSize * sizeof( int16_t ) );
  }
  for( const Slice* slice: cs.picture->slices )
  {
    if( slice->getAlfEnabledFlag( COMPONENT_Y ) )
    {
      for( int i = 0; i < slice->getNumAlfAps(); i++ )
      {
        const int  apsId = slice->getAlfApsIdsLuma()[i];
        const APS* aps   = slice->getAlfAPS( apsId );
        CHECK( aps == nullptr, "invalid APS" );
        const AlfSliceParam& alfParam = aps->getAlfAPSParam();
        memcpy( lumaCoeff + ( NUM_FIXED_FILTER_SETS + apsId ) * kAlfSetSize, alfParam.lumaCoeffFinal, kAlfSetSize * sizeof( int16_t ) );
        memcpy( lumaClip  + ( NUM_FIXED_FILTER_SETS + apsId ) * kAlfSetSize, alfParam.lumaClippFinal, kAlfSetSize * sizeof( int16_t ) );
      }
    }
    if( slice->getAlfEnabledFlag( COMPONENT_Cb ) || slice->getAlfEnabledFlag( COMPONENT_Cr ) )
    {
      const int  apsId = slice->getAlfApsIdChroma();
      const APS* aps   = slice->getAlfAPS( apsId );
      CHECK( aps == nullptr, "invalid APS" );
      const AlfSliceParam& alfParam = aps->getAlfAPSParam();
      constexpr size_t     perAps   = MAX_NUM_ALF_ALTERNATIVES_CHROMA * MAX_NUM_ALF_CHROMA_COEFF;
      memcpy( chromaCoeff + apsId * perAps, alfParam.chromaCoeff,    perAps * sizeof( int16_t ) );
      memcpy( chromaClip  + apsId * perAps, alfParam.chrmClippFinal, perAps * sizeof( int16_t ) );
    }
    for( int comp = 1; comp < 3; comp++ )
    {
      if( slice->getCcAlfEnabledFlag( comp ) )
      {
        const int  apsId = comp == 1 ? slice->getCcAlfCbApsId() : slice->getCcAlfCrApsId();
        const APS* aps   = slice->getAlfAPS( apsId );
        CHECK( aps == nullptr, "invalid APS" );
        constexpr size_t perComp = MAX_NUM_CC_ALF_FILTERS * MAX_NUM_CC_ALF_CHROMA_COEFF;
        memcpy( ccCoeff + ( apsId * 2 + comp - 1 ) * perComp, aps->getCcAlfAPSParam().ccAlfCoeff[comp - 1], perComp * sizeof( int16_t ) );
      }
    }
  }
}

// CTU ctuIdx's 8 bytes at ctu: the luma filter set (255: off), the chroma APS, each chroma component's alternative
// (255: off), each chroma component's CC-ALF filter (255: off), then the CTU's clipped sides (virtual boundaries on
// them; slice, tile and subpicture boundaries without loop filtering across) and its raster-slice corners. Returns
// whether such a boundary clips the CTU or virtual boundaries split it (the BOUNDS shaders are needed then).
static bool writeAlfCtu( const CodingStructure& cs, int ctuIdx, uint8_t* ctu )
{
  const PreCalcValues& pcv     = *cs.pcv;
  const CtuData&       ctuData = cs.getCtuData( ctuIdx );
  const Slice*         slice   = ctuData.slice;
  const CtuAlfData&    alf     = ctuData.alfParam;
  ctu[0] = 255;
  if( alf.alfCtuEnableFlag[COMPONENT_Y] )
  {
    const int index = alf.alfCtbFilterIndex;
    ctu[0] = uint8_t( index < NUM_FIXED_FILTER_SETS ? index : NUM_FIXED_FILTER_SETS + slice->getAlfApsIdsLuma()[index - NUM_FIXED_FILTER_SETS] );
  }
  ctu[1] = uint8_t( slice->getAlfApsIdChroma() & ( ALF_CTB_MAX_NUM_APS - 1 ) );
  ctu[2] = alf.alfCtuEnableFlag[COMPONENT_Cb] ? alf.alfCtuAlternative[0] : 255;
  ctu[3] = alf.alfCtuEnableFlag[COMPONENT_Cr] ? alf.alfCtuAlternative[1] : 255;
  ctu[4] = slice->getCcAlfEnabledFlag( 1 ) && alf.ccAlfFilterControl[0] ? uint8_t( slice->getCcAlfCbApsId() * 4 + alf.ccAlfFilterControl[0] - 1 ) : 255;
  ctu[5] = slice->getCcAlfEnabledFlag( 2 ) && alf.ccAlfFilterControl[1] ? uint8_t( slice->getCcAlfCrApsId() * 4 + alf.ccAlfFilterControl[1] - 1 ) : 255;
  const Position pos( ( ctuIdx % pcv.widthInCtus ) * pcv.maxCUWidth, ( ctuIdx / pcv.widthInCtus ) * pcv.maxCUHeight );
  const Area     area( pos, Size( std::min( pcv.maxCUWidth, pcv.lumaWidth - pos.x ), std::min( pcv.maxCUHeight, pcv.lumaHeight - pos.y ) ) );
  bool           clipTop, clipBottom, clipLeft, clipRight;
  int            numHor, numVer, horPos[3], verPos[3], rasterPad;
  const bool     bounds = AdaptiveLoopFilter::isClipOrCrossedByVirtualBoundaries( cs, area, clipTop, clipBottom, clipLeft, clipRight, numHor, numVer, horPos,
                                                                                  verPos, rasterPad );
  ctu[6] = uint8_t( int( clipTop ) | int( clipBottom ) << 1 | int( clipLeft ) << 2 | int( clipRight ) << 3 | rasterPad << 4 );
  ctu[7] = 0;
  return bounds;
}

bool GpuRecon::prepareAlf( const CodingStructure& cs, const AdaptiveLoopFilter& alfFilter, uint8_t* base )
{
  writeAlfFilterTables( cs, alfFilter, base );
  bool bounds = false;
  for( int ctuIdx = 0; ctuIdx < (int) cs.pcv->sizeInCtus; ctuIdx++ )
  {
    bounds |= writeAlfCtu( cs, ctuIdx, base + kAlfCtus + ctuIdx * 8 );
  }
  return bounds;
}

// alf.slang's workgroups along a side of samples samples: a workgroup takes 32x32 samples (8x8 threads of 4x4)
static uint32_t alfGroups( int samples )
{
  return uint32_t( ( ( samples + 3 ) / 4 + 7 ) / 8 );
}

// Luma (alf.slang loads and stores the rows 8 bytes at a time, from multiples of 4 samples)
void GpuRecon::recordAlfLuma( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params, AlfPush& push )
{
  const PreCalcValues& pcv = *cs.pcv;
  CHECK_FATAL( src.addr[COMPONENT_Y] % 8 != 0 || src.stride[COMPONENT_Y] % 4 != 0 || dst.addr[COMPONENT_Y] % 8 != 0 || dst.stride[COMPONENT_Y] % 4 != 0,
               "Vulkan: the luma rows are not 8-byte aligned" );
  push.src0      = src.addr[COMPONENT_Y];
  push.dst0      = dst.addr[COMPONENT_Y];
  push.coeff     = params + kAlfLumaCoeff;
  push.clip      = params + kAlfLumaClip;
  push.srcStride = src.stride[COMPONENT_Y];
  push.dstStride = dst.stride[COMPONENT_Y];
  push.width     = pcv.lumaWidth;
  push.height    = pcv.lumaHeight;
  dispatch( m_prepared.alfBounds ? kShader_alf_luma_bounds : kShader_alf_luma, push, alfGroups( push.width ), alfGroups( push.height ) );
  mark( "alf luma" );
}

// Chroma, ALF and CC-ALF. Pass 0 filters the blocks of the CTUs with ALF or CC-ALF on a chroma component into the
// scratch planes (dst); pass 1 copies those blocks back into the twin (src), where the others stay (the caller swaps
// only the luma planes).
void GpuRecon::recordAlfChroma( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params, AlfPush& push )
{
  const PreCalcValues& pcv = *cs.pcv;
  CHECK( src.stride[COMPONENT_Cb] != src.stride[COMPONENT_Cr] || dst.stride[COMPONENT_Cb] != dst.stride[COMPONENT_Cr],
         "the chroma planes differ in stride" );
  push.src0      = src.addr[COMPONENT_Cb];
  push.dst0      = dst.addr[COMPONENT_Cb];
  push.src1      = src.addr[COMPONENT_Cr];
  push.dst1      = dst.addr[COMPONENT_Cr];
  push.srcLuma   = src.addr[COMPONENT_Y];
  push.coeff     = params + kAlfChromaCoeff;
  push.clip      = params + kAlfChromaClip;
  push.ccCoeff   = params + kAlfCcCoeff;
  push.srcStride = src.stride[COMPONENT_Cb];
  push.dstStride = dst.stride[COMPONENT_Cb];
  push.scaleX    = getComponentScaleX( COMPONENT_Cb, pcv.chrFormat );
  push.scaleY    = getComponentScaleY( COMPONENT_Cb, pcv.chrFormat );
  push.width     = pcv.lumaWidth >> push.scaleX;
  push.height    = pcv.lumaHeight >> push.scaleY;
  // the chroma blocks' rows read and written 8 bytes at a time where all the planes' rows start 8-byte aligned
  push.aligned = 1;
  for( const Planes* p: { &src, &dst } )
  {
    for( const ComponentID comp: { COMPONENT_Cb, COMPONENT_Cr } )
    {
      push.aligned &= p->addr[comp] % 8 == 0 && p->stride[comp] % 4 == 0 ? 1 : 0;
    }
  }
  const uint32_t groupsX = alfGroups( push.width );
  const uint32_t groupsY = alfGroups( push.height );
  push.pass              = 0;
  dispatch( m_prepared.alfBounds ? kShader_alf_chroma_bounds : kShader_alf_chroma, push, groupsX, groupsY );
  computeBarrier();
  push.pass = 1;
  std::swap( push.src0, push.dst0 );
  std::swap( push.src1, push.dst1 );
  std::swap( push.srcStride, push.dstStride );
  dispatch( kShader_alf_chroma, push, groupsX, groupsY );
  mark( "alf chroma" );
}

void GpuRecon::recordAlf( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params )
{
  const PreCalcValues& pcv = *cs.pcv;
  AlfPush              push{};
  push.ctus        = params + kAlfCtus;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  push.bitDepth    = cs.sps->getBitDepth();
  push.lumaWidth   = pcv.lumaWidth;
  push.lumaHeight  = pcv.lumaHeight;
  push.lumaStride  = src.stride[COMPONENT_Y];
  setVirtualBoundaries( *cs.picHeader, push );
  // (with boundaries, m_prepared.alfBounds from prepareAlf: the BOUNDS variants)
  recordAlfLuma( cs, src, dst, params, push );
  if( isChromaEnabled( pcv.chrFormat ) )
  {
    recordAlfChroma( cs, src, dst, params, push );
  }
}

}   // namespace gpu
}   // namespace vvdec
