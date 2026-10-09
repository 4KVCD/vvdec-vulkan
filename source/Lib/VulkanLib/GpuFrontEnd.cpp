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
#include "CommonLib/MotionInfo.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Slice.h"
#include "CommonLib/Unit.h"

#include <cstring>

namespace vvdec
{
namespace gpu
{

void writeCuRecords( CodingStructure& cs, int ctuRsAddr, std::vector<CuRecord>& out )
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
    r.w[3] = uint32_t( uint8_t( cu.refIdx[0] ) ) | uint32_t( uint8_t( cu.refIdx[1] ) ) << 8;
    for( int list = 0; list < 2; list++ )
    {
      for( int k = 0; k < 3; k++ )
      {
        r.w[4 + 6 * list + 2 * k]     = uint32_t( cu.mv[list][k].getHor() );
        r.w[4 + 6 * list + 2 * k + 1] = uint32_t( cu.mv[list][k].getVer() );
      }
    }
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

void writeCtuRecords( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, CtuRecord* out )
{
  const PPS& pps   = *cs.pps;
  uint32_t   first = 0;
  for( int ctu = 0; ctu < (int) cs.pcv->sizeInCtus; ctu++ )
  {
    const int  ctuX  = ctu % cs.pcv->widthInCtus;
    const int  ctuY  = ctu / cs.pcv->widthInCtus;
    const auto count = uint32_t( cus[ctu].size() );
    CHECK( count > 0xffff, "too many CUs in a CTU" );
    out[ctu].w[0] = first;
    out[ctu].w[1] = count | sliceIndex( cs, cs.getCtuData( ctu ).slice ) << 16;
    out[ctu].w[2] = ( pps.ctuIsTileColBd( ctuX ) ? 1u : 0u ) | pps.getTileIdx( uint32_t( ctuX ), uint32_t( ctuY ) ) << 16;
    out[ctu].w[3] = 0;
    first += count;
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
             | ( hasCol ? 1u << 21 : 0u ) | ( s.isIRAP() ? 1u << 22 : 0u ) | ( s.getColRefIdx() & 15 ) << 24;
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

}   // namespace gpu
}   // namespace vvdec
