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
    \brief    the parse's CUs as the GPU's front end takes them (motion derivation, shaders/mider.slang)
*/

#include "GpuFrontEnd.h"
#include "GpuRecon.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/LoopFilter.h"
#include "CommonLib/MotionInfo.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Slice.h"
#include "CommonLib/Unit.h"

#include <cstring>

namespace vvdec
{
namespace gpu
{

void writeCuRecords( CodingStructure& cs, int ctuRsAddr, std::vector<CuRecord>& out, std::vector<TuRecord>& tus )
{
  for( const CodingUnit& cu: cs.traverseCUs( ctuRsAddr ) )
  {
    CuRecord       r{};
    const Position pos  = cu.lumaPos();
    const Size     size = cu.lumaSize();
    uint32_t       flags = 0;
    flags |= cu.mergeFlag() ? CuRecord::kMerge : 0;
    flags |= cu.skip() ? CuRecord::kSkip : 0;
    flags |= cu.mmvdFlag() ? CuRecord::kMmvd : 0;
    flags |= cu.affineFlag() ? CuRecord::kAffine : 0;
    flags |= cu.geoFlag() ? CuRecord::kGeo : 0;
    flags |= cu.ciipFlag() ? CuRecord::kCiip : 0;
    flags |= cu.chType() == CHANNEL_TYPE_CHROMA ? CuRecord::kChroma : 0;
    flags |= cu.affineType() == AFFINEMODEL_6PARAM ? CuRecord::kAffine6 : 0;
    r.w[0] = uint32_t( pos.x ) | uint32_t( pos.y ) << 16;
    r.w[1] = uint32_t( size.width ) | uint32_t( size.height ) << 8 | uint32_t( cu.predMode() ) << 16 | flags << 18;
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
    // its TUs
    const uint32_t firstTu = uint32_t( tus.size() );
    for( const TransformUnit* next = &cu.firstTU; next; next = next->next )   // (the CU's last TU ends the chain)
    {
      const TransformUnit& tu = *next;
      TuRecord             t{};
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
      t.w[8] = uint32_t( out.size() );   // (the CU's index in its CTU's)
      tus.push_back( t );
    }
    const uint32_t numTus = uint32_t( tus.size() ) - firstTu;
    CHECK( numTus > 0xffff, "too many TUs in a CU" );
    r.w[16] = firstTu | numTus << 16;
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
    out.push_back( r );
  }
}

// the index of a slice among its picture's
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
  return size_t( cs.pcv->sizeInCtus ) + 64 * cs.picture->slices.size();
}

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
    uint32_t*    rec = table + ctus + 64 * i;
    memset( rec, 0, 64 * sizeof( uint32_t ) );

    // the colocated picture, with its motion and slice table on the GPU's reach
    const Picture* col = nullptr;
    if( !s.isIntra() )
    {
      const RefPicList colList = RefPicList( s.isInterB() ? 1 - s.getColFromL0Flag() : 0 );
      if( int( s.getColRefIdx() ) < s.getNumRefIdx( colList ) )
      {
        col = s.getRefPic( colList, int( s.getColRefIdx() ) );
      }
    }
    const bool hasCol = col && col->cs && col->cs->m_colMiAllocation && col->cs->m_gpuSliceAllocation;

    rec[0] = uint32_t( s.getPOC() );
    rec[1] = uint32_t( s.getNumRefIdx( REF_PIC_LIST_0 ) ) | uint32_t( s.getNumRefIdx( REF_PIC_LIST_1 ) ) << 8 | ( s.isInterB() ? 1u << 16 : 0u )
             | ( s.getCheckLDC() ? 1u << 17 : 0u ) | ( s.getColFromL0Flag() ? 1u << 18 : 0u )
             | ( s.getPicHeader()->getEnableTMVPFlag() ? 1u << 19 : 0u ) | ( s.getPicHeader()->getDisFracMMVD() ? 1u << 20 : 0u )
             | ( hasCol ? 1u << 21 : 0u ) | ( s.isIRAP() ? 1u << 22 : 0u ) | ( s.getDeblockingFilterDisable() ? 1u << 23 : 0u )
             | ( s.getColRefIdx() & 15 ) << 24;
    rec[2] = s.getPicHeader()->getMaxNumIBCMergeCand() | s.getPicHeader()->getMaxNumGeoCand() << 8 | s.getPicHeader()->getMaxNumAffineMergeCand() << 16;
    rec[3] = s.getIndependentSliceIdx();
    for( int list = 0; list < 2; list++ )
    {
      for( int refIdx = 0; refIdx < s.getNumRefIdx( RefPicList( list ) ) && refIdx < 16; refIdx++ )
      {
        rec[4] |= s.getIsUsedAsLongTerm( list, refIdx ) ? 1u << ( 16 * list + refIdx ) : 0u;
        rec[5] |= s.getRPL( RefPicList( list ) )->isRefPicLongterm( refIdx ) ? 1u << ( 16 * list + refIdx ) : 0u;
        rec[16 + 16 * list + refIdx] = uint32_t( s.getRefPOC( RefPicList( list ), refIdx ) );
      }
    }
    if( hasCol )
    {
      const VkDeviceAddress motion = hostAllocationAddress( col->cs->m_colMiAllocation );
      const VkDeviceAddress table  = hostAllocationAddress( col->cs->m_gpuSliceAllocation );
      rec[6] = uint32_t( motion );
      rec[7] = uint32_t( motion >> 32 );
      rec[8] = uint32_t( table );
      rec[9] = uint32_t( table >> 32 );
    }
  }
}

void compareMotion( const CodingStructure& cs, const int32_t* gpu, const MotionInfo* cpu, MiderStats& stats )
{
  const PreCalcValues& pcv     = *cs.pcv;
  const int            ctuSize = int( pcv.maxCUWidth );
  const int            size4   = ctuSize >> 2;
  stats.pictures++;
  for( int ctu = 0; ctu < (int) pcv.sizeInCtus; ctu++ )
  {
    const int x0 = ( ctu % pcv.widthInCtus ) * ctuSize;
    const int y0 = ( ctu / pcv.widthInCtus ) * ctuSize;
    for( int y = 0; y < size4; y++ )
    {
      for( int x = 0; x < size4; x++ )
      {
        if( x0 + 4 * x >= int( pcv.lumaWidth ) || y0 + 4 * y >= int( pcv.lumaHeight ) )
        {
          continue;
        }
        const int         at   = ctu * size4 * size4 + y * size4 + x;
        const int32_t*    g    = gpu + at * 5;
        const MotionInfo& c    = cpu[at];
        const uint32_t    w    = uint32_t( g[4] );
        const int         kind = int( ( w >> 16 ) & 0xff );
        stats.compared++;
        const int ref[2] = { int( int8_t( w & 0xff ) ), int( int8_t( ( w >> 8 ) & 0xff ) ) };
        bool      same   = true;
        for( int l = 0; l < 2; l++ )
        {
          const int cref = c.miRefIdx[l] < 0 ? -1 : c.miRefIdx[l];
          same &= ref[l] == cref;
          // (IBC: the block vector, without a reference index)
          if( ref[l] >= 0 || ( kind == 3 && l == 0 ) )
          {
            same &= g[2 * l] == c.mv[l].getHor() && g[2 * l + 1] == c.mv[l].getVer();
          }
        }
        if( !same )
        {
          stats.mismatched++;
          if( stats.reported < 20 )
          {
            stats.reported++;
            msg( WARNING,
                 "GPU motion derivation: POC %d at (%d,%d) kind %d: GPU ref %d,%d mv (%d,%d) (%d,%d), CPU ref %d,%d mv (%d,%d) (%d,%d)\n",
                 cs.picture->poc, x0 + 4 * x, y0 + 4 * y, kind, ref[0], ref[1], g[0], g[1], g[2], g[3], c.miRefIdx[0], c.miRefIdx[1],
                 c.mv[0].getHor(), c.mv[0].getVer(), c.mv[1].getHor(), c.mv[1].getVer() );
            // the CU there as the CPU left it
            if( const CodingUnit* cu = cs.getCU( Position( x0 + 4 * x, y0 + 4 * y ), CHANNEL_TYPE_LUMA ) )
            {
              msg( WARNING,
                   "  CU (%d,%d) %dx%d mode %d merge %d idx %d mmvd %d (%d) affine %d type %d geo %d ciip %d mrgType %d interDir %d refIdx %d,%d "
                   "mvp %d,%d imv %d bcw %d slice %d\n",
                   cu->lx(), cu->ly(), cu->lwidth(), cu->lheight(), cu->predMode(), cu->mergeFlag(), cu->mergeIdx(), cu->mmvdFlag(), cu->mmvdIdx,
                   cu->affineFlag(), cu->affineType(), cu->geoFlag(), cu->ciipFlag(), cu->mergeType(), cu->interDir(), cu->refIdx[0], cu->refIdx[1],
                   cu->mvpIdx[0], cu->mvpIdx[1], cu->imv(), cu->BcwIdx(), cu->slice->getIndependentSliceIdx() );
            }
          }
        }
      }
    }
  }
}

void compareLfp( const CodingStructure& cs, const uint32_t* gpu, const LoopFilterParam* cpu, MiderStats& stats )
{
  const PreCalcValues& pcv = *cs.pcv;
  const size_t         n   = size_t( pcv.sizeInCtus ) * 2 * pcv.num4x4CtuBlks;
  const int            per = int( pcv.num4x4CtuBlks );
  const int            s4  = int( pcv.maxCUWidth ) >> 2;
  for( size_t i = 0; i < n; i++ )
  {
    const LoopFilterParam& c = cpu[i];
    const uint32_t         a = gpu[2 * i];
    const uint32_t         b = gpu[2 * i + 1];
    stats.lfpCompared++;
    stats.lfpEdges += c.bs != 0 ? 1 : 0;
    const bool same = int8_t( a & 0xff ) == c.qp[0] && int8_t( ( a >> 8 ) & 0xff ) == c.qp[1] && int8_t( ( a >> 16 ) & 0xff ) == c.qp[2]
                      && uint8_t( a >> 24 ) == c.bs && uint8_t( b & 0xff ) == c.sideMaxFiltLength && uint8_t( ( b >> 8 ) & 0xff ) == c.flags;
    if( !same )
    {
      stats.lfpMismatched++;
      if( stats.lfpReported < 20 )
      {
        stats.lfpReported++;
        const int ctu = int( i / ( 2 * per ) );
        const int dir = int( ( i / per ) & 1 );
        const int pos = int( i % per );
        const int x   = ( ctu % int( pcv.widthInCtus ) ) * int( pcv.maxCUWidth ) + ( pos % s4 ) * 4;
        const int y   = ( ctu / int( pcv.widthInCtus ) ) * int( pcv.maxCUHeight ) + ( pos / s4 ) * 4;
        msg( WARNING,
             "GPU deblocking parameters: POC %d at (%d,%d) dir %d: GPU qp %d,%d,%d bs %02x len %02x flags %02x, CPU qp %d,%d,%d bs %02x len %02x flags %02x\n",
             cs.picture->poc, x, y, dir, int8_t( a & 0xff ), int8_t( ( a >> 8 ) & 0xff ), int8_t( ( a >> 16 ) & 0xff ), a >> 24, b & 0xff,
             ( b >> 8 ) & 0xff, c.qp[0], c.qp[1], c.qp[2], c.bs, c.sideMaxFiltLength, c.flags );
      }
    }
  }
}

}   // namespace gpu
}   // namespace vvdec
