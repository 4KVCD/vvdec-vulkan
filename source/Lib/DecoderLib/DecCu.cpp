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

/** \file     DecCu.cpp
    \brief    CU decoder class
*/

#include "DecCu.h"

#include "CommonLib/InterPrediction.h"
#include "CommonLib/IntraPrediction.h"
#include "CommonLib/UnitTools.h"
#include "CommonLib/TimeProfiler.h"

#include "CommonLib/dtrace_buffer.h"

namespace vvdec
{

//! \ingroup DecoderLib
//! \{

void DecCu::TaskDeriveCtuMotionInfo( CodingStructure &cs, const int ctuRsAddr, const UnitArea &ctuArea, MotionHist& hist )
{
  PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_CONTROL_PARSE_DERIVE_LL, cs, CH_L );

  const unsigned  ctuXPosInCtus  = ctuRsAddr % cs.pcv->widthInCtus;
  const unsigned  tileColIdx     = cs.pps->ctuToTileCol( ctuXPosInCtus );
  const unsigned  tileXPosInCtus = cs.pps->getTileColumnBd( tileColIdx );

  if( ctuXPosInCtus == tileXPosInCtus )
  {
    hist.motionLut.   resize(0);
    hist.motionLutIbc.resize(0);
  }
  
  for( auto &currCU : cs.traverseCUs( ctuRsAddr ) )
  {
    CHECK( !ctuArea.blocks[currCU.chType()].contains( currCU.blocks[currCU.chType()] ),
           "Traversing CU at (" << currCU.blocks[currCU.chType()].x << "," << currCU.blocks[currCU.chType()].y
           << ") outside of the CTU at (" << ctuArea.blocks[currCU.chType()].x << "," << ctuArea.blocks[currCU.chType()].y << ")!"
    );

#if JVET_O1170_CHECK_BV_AT_DECODER
    if( currCU.Y().valid() && currCU.sps->getIBCFlag() )
    {
      const int vSize = cs.sps->getMaxCUHeight() > 64 ? 64 : cs.sps->getMaxCUHeight();
      if( ( currCU.Y().x % vSize ) == 0 && ( currCU.Y().y % vSize ) == 0 )
      {
        m_pcInterPred->resetVPDUforIBC( cs.pcv->chrFormat, cs.sps->getMaxCUHeight(), vSize, currCU.Y().x  + g_IBCBufferSize / cs.sps->getMaxCUHeight() / 2, currCU.Y().y );
      }
    }
#endif

    if( !CU::isIntra( currCU ) )
    {
      xDeriveCUMV( currCU, hist );
    }
    else
    {
      MotionBuf mb = currCU.getMotionBuf();
      mb.memset( MI_NOT_VALID );
    }
  }
}

void DecCu::TaskTrafoCtu( CodingStructure &cs, const int ctuRsAddr, const UnitArea &ctuArea )
{
  PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_ITRANS_REC, cs, CH_L );

  for( auto &currCU : cs.traverseCUs( ctuRsAddr ) )
  {
    CHECK( !ctuArea.blocks[currCU.chType()].contains( currCU.blocks[currCU.chType()] ), "Should never happen!" );

    if( currCU.rootCbf() )
    {
      reconstructResi( currCU );
    }
  }
}

void DecCu::TaskInterCtu( CodingStructure &cs, const int ctuRsAddr, const UnitArea &ctuArea )
{
  PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_CONTROL_PARSE_DERIVE_LL, cs, CH_L );

  for( auto &currCU: cs.traverseCUs( ctuRsAddr ) )
  {
    CHECK( !ctuArea.blocks[currCU.chType()].contains( currCU.blocks[currCU.chType()] ), "Should never happen!" );

    if( !CU::isIntra( currCU ) && !CU::isIBC( currCU ) )
    {
      predAndReco( currCU, false );
    }
  }
}

void DecCu::TaskCriticalIntraKernel( CodingStructure &cs, const int ctuRsAddr, const UnitArea &ctuArea )
{
  PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_CONTROL_PARSE_DERIVE_LL, cs, CH_L );
  const CtuData &ctuData = cs.getCtuData( ctuRsAddr );

  for( auto &currCU : cs.traverseCUs( ctuRsAddr ) )
  {
    CHECK( !ctuArea.blocks[currCU.chType()].contains( currCU.blocks[currCU.chType()] ), "Should never happen!" );

    if( CU::isIntra( currCU ) || currCU.ciipFlag() || CU::isIBC( currCU ) )
    {
      predAndReco( currCU, true );
    }
    else if( currCU.rootCbf() )
    {
      finishLMCSAndReco( currCU );
    }

    if( cs.sps->getIBCFlag() && cs.hasIbcBlock[ctuData.lineIdx] )
    {
      cs.fillIBCbuffer( currCU, ctuData.lineIdx );
    }
  }
}

// DMVR's refined motion of a CU where TMVP takes it, on the 8x8 grid: in the
// motion at( position ) returns, a MotionInfo or a ColocatedMotionInfo
template<typename At>
static void storeDmvrMotion( const CodingStructure &cs, const CodingUnit &cu, At at )
{
  const int dy = std::min<int>( cu.lumaSize().height, DMVR_SUBCU_HEIGHT );
  const int dx = std::min<int>( cu.lumaSize().width,  DMVR_SUBCU_WIDTH );

  static constexpr unsigned scale = 4 * AMVP_DECIMATION_FACTOR;
  static constexpr unsigned mask  = scale - 1;

  const Position puPos = cu.lumaPos();
  const Mv mv0 = cu.mv[0][0];
  const Mv mv1 = cu.mv[1][0];

  for( int y = puPos.y, num = 0; y < ( puPos.y + cu.lumaSize().height ); y = y + dy )
  {
    for( int x = puPos.x; x < ( puPos.x + cu.lumaSize().width ); x = x + dx, num++ )
    {
      const Mv subPuMv0 = mv0 + cs.m_dmvrMvCache[cu.mvdL0SubPuOff + num];
      const Mv subPuMv1 = mv1 - cs.m_dmvrMvCache[cu.mvdL0SubPuOff + num];

      int y2 = ( ( y - 1 ) & ~mask ) + scale;

      for( ; y2 < y + dy; y2 += scale )
      {
        int x2 = ( ( x - 1 ) & ~mask ) + scale;

        for( ; x2 < x + dx; x2 += scale )
        {
          auto& mi = at( Position{ x2, y2 } );

          mi.mv[0] = subPuMv0;
          mi.mv[1] = subPuMv1;
        }
      }
    }
  }
}

// the motion of a CTU for TMVP in the next pictures: its colocated motion
static void storeColMotion( CodingStructure &cs, const int ctuRsAddr )
{
  CtuData &ctuData = cs.getCtuData( ctuRsAddr );

  const int size4x4 = (int)cs.get4x4MapStride();

  const MotionInfo          *src = ctuData.motion;
        ColocatedMotionInfo *dst = ctuData.colMotion;

  // ctuDta.colMotion should already be set to '0'
  //memset( dst, 0, sizeof( ctuData.colMotion ) );

  // skip every second source line
  for( int y = 0; y < size4x4; y += 2, src += size4x4 )
  {
    for( int x = 0; x < size4x4; x += 2, src += 2, dst++ )
    {
      //if( src->isInter() )
      {
        *dst = *src;
      }
    }
  }
}

void DecCu::TaskFinishMotionInfo( CodingStructure &cs, const int ctuRsAddr, const int col, const int row )
{
  PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_CONTROL_PARSE_DERIVE_LL, cs, CH_L );

  // first, finish DMVR motion

  UnitArea    ctuArea = getCtuArea( cs, col, row, true );
  MotionBuf   mb      = cs.getMotionBuf( ctuArea.Y() );
  MotionInfo* orgPtr  = mb.buf;

  for( CodingUnit &cu : cs.traverseCUs( ctuRsAddr ) )
  {
    CHECKD( !ctuArea.blocks[cu.chType()].contains( cu.blocks[cu.chType()] ), "Should never happen!" );

    if( isLuma( cu.chType() ) && cu.dmvrCondition() )
    {
      storeDmvrMotion( cs, cu, [&]( Position pos ) -> MotionInfo& { return orgPtr[cs.inCtuPos( pos, CH_L )]; } );
    }
  }

  storeColMotion( cs, ctuRsAddr );
}
#if VVDEC_VULKAN
void DecCu::TaskColMotion( CodingStructure &cs, const int ctuRsAddr )
{
  PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_CONTROL_PARSE_DERIVE_LL, cs, CH_L );

  storeColMotion( cs, ctuRsAddr );
}

void DecCu::TaskGpuCtu( CodingStructure& cs, const int ctuRsAddr, const UnitArea& ctuArea, const gpu::CtuJobs& jobs )
{
  PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_CONTROL_PARSE_DERIVE_LL, cs, CH_L );

  for( auto &currCU : cs.traverseCUs( ctuRsAddr ) )
  {
    CHECK( !ctuArea.blocks[currCU.chType()].contains( currCU.blocks[currCU.chType()] ), "Should never happen!" );

    if( jobs.trafo && currCU.rootCbf() )
    {
      xGpuTrafoJobs( currCU, *jobs.trafo );
    }

    if( CU::isIntra( currCU ) )
    {
      xGpuIntraJobs( currCU, *jobs.intra, *jobs.waves );
    }
    else if( CU::isIBC( currCU ) )
    {
      xGpuIbcJobs( currCU, *jobs.intra, *jobs.waves );
    }
    else if( jobs.inter )
    {
      xGpuInterJobs( currCU, *jobs.inter, *jobs.refs, jobs.trafo, *jobs.intra, *jobs.waves );
    }
    else if( currCU.rootCbf() )
    {
      finishLMCSAndReco( currCU );
    }
  }
}

// LMCS's chroma residual scaling of a chroma transform block whose luma is at
// lumaPos (Reshape::calculateChromaAdjVpduNei): the scale comes from the
// average luma left of and above the CU at the top-left of the VPDU holding
// lumaPos (64 x 64, the CTU when smaller), min( 64, CTU size ) samples a
// side; returns the highest wave of those samples
static int gpuLmcsChroma( const CodingUnit& cu, const Position lumaPos, gpu::IntraJob& job, const gpu::WaveMap& waves )
{
  const CodingStructure& cs      = *cu.cs;
  const int              ctuSize = cs.sps->getCTUSize();
  const int              count   = std::min( 64, ctuSize );
  const int              mask    = ctuSize == 128 ? ~63 : ~( ctuSize - 1 );
  const Position         vpdu( lumaPos.x & mask, lumaPos.y & mask );
  const CodingUnit*      first   = cs.getCU( vpdu, CHANNEL_TYPE_LUMA );
  const CodingUnit*      above   = cs.getCURestricted( first->lumaPos().offset( 0, -1 ), *first, CHANNEL_TYPE_LUMA, first->ly() == vpdu.y ? first : first->above );
  const CodingUnit*      left    = cs.getCURestricted( first->lumaPos().offset( -1, 0 ), *first, CHANNEL_TYPE_LUMA, first->lx() == vpdu.x ? first : first->left );
  const Position         at      = first->lumaPos();
  job.setLmcs( at.x, at.y, left != nullptr, above != nullptr, getLog2( count ) );
  int wave = 0;
  if( left )
  {
    wave = std::max( wave, waves.maxIn( COMPONENT_Y, at.x - 1, at.y, 1, count ) );
  }
  if( above )
  {
    wave = std::max( wave, waves.maxIn( COMPONENT_Y, at.x, at.y - 1, count, 1 ) );
  }
  return wave;
}

void DecCu::xGpuIntraJobs( CodingUnit& cu, std::vector<gpu::IntraJob>& jobs, gpu::WaveMap& waves )
{
  // the transform blocks and components in the order predAndReco takes them:
  // the reuse of the neighbourhood across a CU's components depends on it
  for( TransformUnit &tu : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
  {
    for( const CompArea &area : tu.blocks )
    {
      if( !area.valid() )
      {
        continue;
      }

      const ComponentID compID = area.compID();
      const ChannelType chType = toChannelType( compID );
      const uint32_t    mode   = PU::getFinalIntraMode( cu, chType );
      const int         bdpcm  = isLuma( compID ) ? cu.bdpcmMode() : cu.bdpcmModeChroma();

      int kind   = gpu::IntraJob::kAngular;
      int flags  = 0;
      int jobMode = int( mode );
      if( bdpcm )
      {
        kind = gpu::IntraJob::kBdpcm;
      }
      else if( CU::isMIP( cu, chType ) )
      {
        // MIP: luma's mode, or (4:4:4 chroma in DM) the co-located luma's
        kind                    = gpu::IntraJob::kMip;
        const CodingUnit& luma  = isLuma( compID ) ? cu : PU::getCoLocatedLumaPU( cu );
        jobMode                 = luma.intraDir[CHANNEL_TYPE_LUMA];
        flags                  |= luma.mipTransposedFlag() ? gpu::IntraJob::kMipTransposed : 0;
      }
      else if( !isLuma( compID ) && PU::isLMCMode( mode ) )
      {
        kind = gpu::IntraJob::kCclm;
      }

      // ISP: a luma sub-partition, predicted from references
      // initIntraPatternChTypeISP fills for the whole CU (as a block of its
      // size) once, then from the sub-partition before
      const bool isp = isLuma( compID ) && cu.ispMode() != NOT_INTRA_SUBPARTITIONS;

      int units[3];
      if( isp )
      {
        m_pcIntraPred->getNeighborUnits( cu.firstTU, cu.Y(), units );
      }
      else
      {
        m_pcIntraPred->getNeighborUnits( tu, area, units );
      }

      if( TU::getCbf( tu, compID ) || ( isChroma( compID ) && tu.jointCbCr ) )
      {
        flags |= gpu::IntraJob::kResidual;
      }
      if( kind == gpu::IntraJob::kAngular && isLuma( compID ) && !isp && IntraPrediction::useFilteredIntraRefSamples( compID, cu, tu ) )
      {
        flags |= gpu::IntraJob::kFiltered;
      }

      // the reference line (MRL: luma's lines further out)
      const int refLine = isLuma( compID ) ? cu.multiRefIdx() : 0;

      gpu::IntraJob job;
      job.set( area.x, area.y, area.width, area.height, compID, kind,
               kind == gpu::IntraJob::kAngular || kind == gpu::IntraJob::kMip ? jobMode : 0, flags | ( isp ? gpu::IntraJob::kIsp : 0 ), bdpcm,
               refLine );
      job.setNeighbors( units[0], units[1], units[2] );

      int wave = 0;
      if( isp )
      {
        // the block predicted: a vertical split's sub-partitions 1 or 2
        // samples wide share a block 4 wide (its first predicts it)
        const CompArea& cuArea = cu.Y();
        CompArea        pred   = area;
        if( CU::isPredRegDiffFromTB( cu, compID ) )
        {
          pred.x     = cuArea.x + ( ( area.x - cuArea.x ) & ~( PRED_REG_MIN_WIDTH - 1 ) );
          pred.width = PRED_REG_MIN_WIDTH;
        }
        const Position         posLT      = pred.pos();
        const CodingStructure& cs         = *cu.cs;
        const bool             leftAvail  = nullptr != cs.getCURestricted( posLT.offset( -1, 0 ), cu, CH_L, posLT.x == cu.lx() ? cu.left : &cu );
        const bool             aboveAvail = nullptr != cs.getCURestricted( posLT.offset( 0, -1 ), cu, CH_L, posLT.y == cu.ly() ? cu.left : &cu );
        job.setIsp( cuArea.x, cuArea.y, cuArea.width, cuArea.height, cu.ispMode(), leftAvail, aboveAvail );

        // what it reads: the CU's references for the first, else the row above
        // and the column to the left, the CU's width (height) longer than the
        // block's (and what the first read, before it)
        const bool first   = pred.x == cuArea.x && pred.y == cuArea.y;
        const int  x       = first ? cuArea.x : pred.x;
        const int  y       = first ? cuArea.y : pred.y;
        const int  topLen  = first ? 2 * cuArea.width : cuArea.width + pred.width;
        const int  leftLen = first ? 2 * cuArea.height : cuArea.height + pred.height;
        wave = std::max( waves.maxIn( compID, x - 1, y - 1, topLen + 1, 1 ), waves.maxIn( compID, x - 1, y, 1, leftLen ) );
      }
      else
      {
        // what the block reads: its references (the row above with the
        // above-right samples, the column to the left with the below-left
        // ones, refLine further out), for CCLM also the luma at and around the
        // block
        wave = std::max( waves.maxIn( compID, area.x - 1 - refLine, area.y - 1 - refLine, 2 * area.width + 1 + refLine, 1 ),
                         waves.maxIn( compID, area.x - 1 - refLine, area.y - refLine, 1, 2 * area.height + refLine ) );
      }
      // LMCS: the chroma residual scaled (predAndReco's doChrScale), by the
      // luma around the VPDU of the block's luma
      if( isChroma( compID ) && cu.slice->getLmcsEnabledFlag() && cu.cs->picHeader->getLmcsChromaResidualScaleFlag() && area.area() > 4
          && ( TU::getCbf( tu, compID ) || tu.jointCbCr ) )
      {
        const Position lumaPos = tu.Y().valid() ? tu.Y().pos() : recalcPosition( cu.chromaFormat, tu.chType(), CHANNEL_TYPE_LUMA, tu.blocks[tu.chType()].pos() );
        wave = std::max( wave, gpuLmcsChroma( cu, lumaPos, job, waves ) );
      }
      if( kind == gpu::IntraJob::kCclm )
      {
        const IntraPrediction::LmAvailability lm = IntraPrediction::getLmAvailability( cu, area );
        job.setCclm( lm.lumaAbove, lm.lumaLeft, lm.above, lm.left, lm.actualTop, lm.actualLeft );

        const int sx = getComponentScaleX( compID, cu.chromaFormat );
        const int sy = getComponentScaleY( compID, cu.chromaFormat );
        const int lx = area.x << sx, ly = area.y << sy, lw = area.width << sx, lh = area.height << sy;
        wave = std::max( wave, waves.maxIn( COMPONENT_Y, lx - 4, ly - 4, 2 * lw + 8, 2 * lh + 8 ) );
      }
      job.setWave( wave + 1 );
      waves.set( compID, area.x, area.y, area.width, area.height, wave + 1 );
      jobs.push_back( job );
    }
  }
}

void DecCu::xGpuIbcJobs( CodingUnit& cu, std::vector<gpu::IntraJob>& jobs, gpu::WaveMap& waves )
{
  Mv bv = cu.mv[REF_PIC_LIST_0][0];
  bv.changePrecision( MV_PRECISION_INTERNAL, MV_PRECISION_INT );

  for( TransformUnit &tu : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
  {
    for( const CompArea &area : tu.blocks )
    {
      // the components of the CU, as predAndReco takes them: a skipped CU's
      // transform units (addEmptyTUs) span the chroma of a luma-tree CU too
      if( !area.valid() || !cu.blocks[area.compID()].valid() )
      {
        continue;
      }

      const ComponentID compID = area.compID();
      const int         bvx    = isLuma( compID ) ? bv.hor : bv.hor >> getComponentScaleX( compID, cu.chromaFormat );
      const int         bvy    = isLuma( compID ) ? bv.ver : bv.ver >> getComponentScaleY( compID, cu.chromaFormat );

      int flags = 0;
      if( TU::getCbf( tu, compID ) || ( isChroma( compID ) && tu.jointCbCr ) )
      {
        flags |= gpu::IntraJob::kResidual;
      }

      gpu::IntraJob job;
      job.set( area.x, area.y, area.width, area.height, compID, gpu::IntraJob::kIbc, 0, flags, 0 );
      job.setBlockVector( bvx, bvy );

      int wave = waves.maxIn( compID, area.x + bvx, area.y + bvy, area.width, area.height );
      // LMCS: the chroma residual scaled (finishLMCSAndReco), by the luma
      // around the VPDU of the transform unit's luma
      if( isChroma( compID ) && cu.rootCbf() && cu.cs->picHeader->getLmcsEnabledFlag() && cu.cs->picHeader->getLmcsChromaResidualScaleFlag()
          && cu.slice->getLmcsEnabledFlag() && area.area() > 4 && ( TU::getCbf( tu, compID ) || tu.jointCbCr ) )
      {
        wave = std::max( wave, gpuLmcsChroma( cu, tu.blocks[COMPONENT_Y].pos(), job, waves ) );
      }
      wave += 1;
      job.setWave( wave );
      waves.set( compID, area.x, area.y, area.width, area.height, wave );
      jobs.push_back( job );
    }
  }
}

// weighted prediction (xPredInterBi): P slices under the PPS's weighted
// prediction, B slices under its bi-prediction one (BCW's weights left at
// their default), not GPM, nor DMVR or BDOF (whose references have no
// weights: xProcessDMVR, xSubPuBio average without them): the lists'
// weights, offsets and denominators (getWpScaling), for the job (flags: the
// lists the job predicts from, DMVR, BDOF)
static void gpuWp( const CodingUnit& cu, const int8_t refIdx[2], int flags, gpu::InterJob& job )
{
  const Slice& slice = *cu.slice;
  const PPS&   pps   = *cu.pps;
  if( cu.geoFlag() || ( flags & ( gpu::InterJob::kDmvr | gpu::InterJob::kBdof ) )
      || !( ( pps.getWPBiPred() && slice.getSliceType() == B_SLICE && cu.BcwIdx() == BCW_DEFAULT ) || ( pps.getUseWP() && slice.getSliceType() == P_SLICE ) ) )
  {
    return;
  }
  int weight[2][3] = {}, offset[2][3] = {}, log2Denom[3] = {};
  int n = 0;
  for( int list = 0; list < 2; list++ )
  {
    if( !( flags & ( list == 0 ? gpu::InterJob::kL0 : gpu::InterJob::kL1 ) ) )
    {
      continue;
    }
    const WPScalingParam* wp = nullptr;
    slice.getWpScaling( RefPicList( list ), refIdx[list], wp );
    for( int c = 0; c < getNumberValidComponents( cu.chromaFormat ); c++ )
    {
      weight[n][c] = wp[c].iWeight;
      offset[n][c] = wp[c].iOffset;
      log2Denom[c] = int( wp[c].uiLog2WeightDenom );
    }
    n++;
  }
  job.setWp( weight, offset, log2Denom, n == 2 );
}

// motionCompensation's bioApplied: whether BDOF refines the CU's bi-prediction
static bool gpuBdof( const CodingUnit& cu )
{
  const Slice& slice = *cu.slice;
  const PPS&   pps   = *cu.pps;
  if( !cu.sps->getUseBIO() || cu.cs->picHeader->getDisBdofFlag() || cu.affineFlag() || cu.ciipFlag() || cu.smvdMode()
      || ( cu.sps->getUseBcw() && cu.BcwIdx() != BCW_DEFAULT ) || cu.refIdx[0] < 0 || cu.refIdx[1] < 0 )
  {
    return false;
  }
  const WPScalingParam* wp0 = nullptr;
  const WPScalingParam* wp1 = nullptr;
  slice.getWpScaling( REF_PIC_LIST_0, cu.refIdx[0], wp0 );
  slice.getWpScaling( REF_PIC_LIST_1, cu.refIdx[1], wp1 );
  const bool wp = ( wp0[COMPONENT_Y].bPresentFlag || wp0[COMPONENT_Cb].bPresentFlag || wp0[COMPONENT_Cr].bPresentFlag || wp1[COMPONENT_Y].bPresentFlag
                    || wp1[COMPONENT_Cb].bPresentFlag || wp1[COMPONENT_Cr].bPresentFlag )
                  && slice.getSliceType() == B_SLICE;
  if( wp || ( pps.getUseWP() && slice.getSliceType() == P_SLICE ) || !PU::isBiPredFromDifferentDirEqDistPoc( cu ) || cu.Y().height < 8
      || cu.Y().width < 8 || cu.Y().area() < 128 )
  {
    return false;
  }
  return !slice.getRefPic( REF_PIC_LIST_0, cu.refIdx[0] )->isRefScaled( cu.pps ) && !slice.getRefPic( REF_PIC_LIST_1, cu.refIdx[1] )->isRefScaled( cu.pps );
}

void DecCu::xGpuInterJobs( CodingUnit& cu, std::vector<gpu::InterJob>& jobs, const gpu::RefTable& refs, std::vector<gpu::TrafoJob>* trafoJobs,
                           std::vector<gpu::IntraJob>& intraJobs, gpu::WaveMap& waves )
{
  const Slice& slice = *cu.slice;
  // what all the CU's blocks share: AMVR's half-sample filter, LMCS, the
  // residual, CIIP
  int flags = 0;
  if( cu.imv() == IMV_HPEL )
  {
    flags |= gpu::InterJob::kHalfPel;
  }

  // LMCS: the luma prediction forward-mapped (rspBufFwd); with its chroma
  // residual scaling, a chroma component with a residual is finished by
  // intra jobs once the luma its scale comes from is there
  // (finishLMCSAndReco): inter keeps its prediction aside
  const PicHeader& picHeader  = *cu.cs->picHeader;
  const bool       lmcs       = slice.getLmcsEnabledFlag() && picHeader.getLmcsEnabledFlag();
  const bool       lmcsChroma = lmcs && picHeader.getLmcsChromaResidualScaleFlag() && cu.rootCbf();
  if( lmcs )
  {
    flags |= gpu::InterJob::kLumaMap;
  }

  // the residual: the GPU adds it where a component has one; inside such a
  // component, the transform blocks without one are cleared, by the GPU when
  // it makes the residuals (trafoJobs), else here
  if( cu.rootCbf() )
  {
    for( int c = 0; c < getNumberValidComponents( cu.chromaFormat ); c++ )
    {
      const ComponentID compID = ComponentID( c );
      if( !cu.planeCbf( compID ) )
      {
        continue;
      }
      if( !cu.ciipFlag() )
      {
        flags |= lmcsChroma && isChroma( compID ) ? gpu::InterJob::kDeferCb << ( c - 1 ) : gpu::InterJob::kResidualY << c;
      }
      for( TransformUnit &tu : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
      {
        if( tu.blocks[compID].valid() && !TU::getCbf( tu, compID ) && !( isChroma( compID ) && tu.jointCbCr ) )
        {
          const CompArea& block = tu.blocks[compID];
          if( trafoJobs )
          {
            // in blocks of powers of two (SBT's transform units without the
            // residual are 3/4 of the CU: 12, 24, 48 luma samples)
            for( int y = 0; y < int( block.height ); y += 1 << getLog2( block.height - y ) )
            {
              for( int x = 0; x < int( block.width ); x += 1 << getLog2( block.width - x ) )
              {
                gpu::TrafoJob job;
                job.set( block.x + x, block.y + y, getLog2( block.width - x ), getLog2( block.height - y ), compID, 0, 0, gpu::TrafoJob::kZero, 0,
                         0, 0, 0, 0, 0 );
                trafoJobs->push_back( job );
              }
            }
          }
          else
          {
            cu.cs->getRecoBuf( block ).fill( 0 );
          }
        }
      }
    }
  }

  // CIIP: inter's prediction kept aside, all of it, for the blend with
  // planar's (xGpuCiipJobs)
  if( cu.ciipFlag() )
  {
    flags |= gpu::InterJob::kDeferY | gpu::InterJob::kDeferCb | gpu::InterJob::kDeferCb << 1;
  }

  // the blocks motionCompensation predicts with one motion each: the CU, or
  // SbTMVP's sub-PUs joined where their motion is the same (xSubPuMC), each
  // with its own clip of the motion, in tiles of up to 16 x 16 (DMVR's
  // subblocks)
  const auto mcBlock = [&]( int bx, int by, int bw, int bh, const int8_t refIdx[2], const Mv mv[2], bool subPu )
  {
    int bflags = flags;
    int ref[2] = { 0, 0 };
    for( int list = 0; list < 2; list++ )
    {
      if( refIdx[list] >= 0 )
      {
        bflags    |= list == 0 ? gpu::InterJob::kL0 : gpu::InterJob::kL1;
        ref[list]  = refs.slot( slice.getRefPic( RefPicList( list ), refIdx[list] ) );
      }
    }
    // as motionCompensation chooses: BDOF and DMVR (not for sub-PUs), else
    // the same motion twice is a uni-prediction (xCheckIdenticalMotion), else
    // BCW's weights (not with CIIP)
    const bool dmvr = !subPu && PU::checkDMVRCondition( cu );
    const bool bi   = refIdx[0] >= 0 && refIdx[1] >= 0;
    if( !subPu && gpuBdof( cu ) )
    {
      bflags |= gpu::InterJob::kBdof;
    }
    if( dmvr )
    {
      cu.setDmvrCondition( true );
      bflags |= gpu::InterJob::kDmvr;
    }
    else if( bi && slice.isInterB() && !cu.pps->getWPBiPred()
             && slice.getRefPOC( REF_PIC_LIST_0, refIdx[0] ) == slice.getRefPOC( REF_PIC_LIST_1, refIdx[1] ) && mv[0] == mv[1] )
    {
      bflags &= ~gpu::InterJob::kL1;
    }
    const bool bcw = ( bflags & gpu::InterJob::kL1 ) && ( bflags & gpu::InterJob::kL0 ) && cu.BcwIdx() != BCW_DEFAULT && !cu.ciipFlag();

    const int tileW = std::min( bw, DMVR_SUBCU_WIDTH );
    const int tileH = std::min( bh, DMVR_SUBCU_HEIGHT );
    int       num   = 0;
    for( int y = by; y < by + bh; y += tileH )
    {
      for( int x = bx; x < bx + bw; x += tileW, num++ )
      {
        gpu::InterJob job;
        job.set( x, y, tileW, tileH, bflags, mv[0].hor, mv[0].ver, mv[1].hor, mv[1].ver, bx, by, ref[0], ref[1],
                 dmvr ? int( cu.mvdL0SubPuOff ) + num : 0 );
        if( bcw )
        {
          job.setBcw( getBcwWeight( g_BcwInternBcw[cu.BcwIdx()], REF_PIC_LIST_1 ) );
        }
        gpuWp( cu, refIdx, bflags, job );
        jobs.push_back( job );
      }
    }
  };

  if( cu.geoFlag() )
  {
    // motionCompensationGeo: each partition a uni-prediction (14-bit), from
    // its own list's reference and motion, then blended by the mask of the
    // split's angle
    const int     dir0    = cu.interDirrefIdxGeo0() >> 4;
    const int     dir1    = cu.interDirrefIdxGeo1() >> 4;
    const int     ref0    = refs.slot( slice.getRefPic( RefPicList( dir0 - 1 ), cu.interDirrefIdxGeo0() & 15 ) );
    const int     ref1    = refs.slot( slice.getRefPic( RefPicList( dir1 - 1 ), cu.interDirrefIdxGeo1() & 15 ) );
    const Mv&     mv0     = cu.mv[0][1];
    const Mv&     mv1     = cu.mv[1][1];
    const uint8_t split   = cu.geoSplitDir;
    const int     angle   = g_GeoParams[split][0];
    const int     wIdx    = getLog2( cu.lwidth() ) - GEO_MIN_CU_LOG2;
    const int     hIdx    = getLog2( cu.lheight() ) - GEO_MIN_CU_LOG2;
    for( int y = cu.ly(); y < cu.ly() + int( cu.lheight() ); y += 16 )
    {
      for( int x = cu.lx(); x < cu.lx() + int( cu.lwidth() ); x += 16 )
      {
        gpu::InterJob job;
        job.set( x, y, std::min<int>( 16, cu.lwidth() ), std::min<int>( 16, cu.lheight() ), flags | gpu::InterJob::kL0 | gpu::InterJob::kL1,
                 mv0.hor, mv0.ver, mv1.hor, mv1.ver, cu.lx(), cu.ly(), ref0, ref1, 0 );
        job.setGeo( g_angle2mask[angle], g_angle2mirror[angle], g_weightOffset[split][hIdx][wIdx][0], g_weightOffset[split][hIdx][wIdx][1] );
        jobs.push_back( job );
      }
    }
  }
  // (SbTMVP's CUs are subblock merge CUs: affine ones, by their flag)
  else if( cu.affineFlag() && cu.mergeType() != MRG_TYPE_SUBPU_ATMVP )
  {
    // xPredAffineBlk: 4x4 subblocks with motion of their own, refined by PROF;
    // the same motion twice is a uni-prediction (xCheckIdenticalMotion)
    int aflags = flags;
    int ref[2] = { 0, 0 };
    for( int list = 0; list < 2; list++ )
    {
      if( cu.refIdx[list] >= 0 )
      {
        aflags    |= list == 0 ? gpu::InterJob::kL0 : gpu::InterJob::kL1;
        ref[list]  = refs.slot( slice.getRefPic( RefPicList( list ), cu.refIdx[list] ) );
      }
    }
    if( cu.refIdx[0] >= 0 && cu.refIdx[1] >= 0 && slice.isInterB() && !cu.pps->getWPBiPred()
        && slice.getRefPOC( REF_PIC_LIST_0, cu.refIdx[0] ) == slice.getRefPOC( REF_PIC_LIST_1, cu.refIdx[1] ) && cu.mv[0][0] == cu.mv[1][0]
        && cu.mv[0][1] == cu.mv[1][1] && ( cu.affineType() == AFFINEMODEL_4PARAM || cu.mv[0][2] == cu.mv[1][2] ) )
    {
      aflags &= ~gpu::InterJob::kL1;
    }
    const bool bcw = ( aflags & gpu::InterJob::kL0 ) && ( aflags & gpu::InterJob::kL1 ) && cu.BcwIdx() != BCW_DEFAULT;

    bool prof[2]      = { false, false };
    bool spread[2]    = { false, false };
    int  params[2][6] = {};
    for( int list = 0; list < 2; list++ )
    {
      if( cu.refIdx[list] < 0 )
      {
        continue;
      }
      const int shift = MAX_CU_DEPTH;
      const Mv& affLT = cu.mv[list][0];
      const Mv& affRT = cu.mv[list][1];
      const Mv& affLB = cu.mv[list][2];
      const int dHorX = ( affRT - affLT ).getHor() * ( 1 << ( shift - getLog2( cu.lwidth() ) ) );
      const int dHorY = ( affRT - affLT ).getVer() * ( 1 << ( shift - getLog2( cu.lwidth() ) ) );
      const int dVerX = cu.affineType() == AFFINEMODEL_6PARAM ? ( affLB - affLT ).getHor() * ( 1 << ( shift - getLog2( cu.lheight() ) ) ) : -dHorY;
      const int dVerY = cu.affineType() == AFFINEMODEL_6PARAM ? ( affLB - affLT ).getVer() * ( 1 << ( shift - getLog2( cu.lheight() ) ) ) : dHorX;
      spread[list] = InterPrediction::isSubblockVectorSpreadOverLimit( dHorX, dHorY, dVerX, dVerY, cu.interDir() );
      prof[list]   = cu.sps->getUsePROF() && !cu.cs->picHeader->getDisProfFlag()
                   && !( ( cu.affineType() == AFFINEMODEL_6PARAM && affLT == affRT && affLT == affLB ) || ( cu.affineType() == AFFINEMODEL_4PARAM && affLT == affRT ) )
                   && !spread[list] && !slice.getRefPic( RefPicList( list ), cu.refIdx[list] )->isRefScaled( cu.pps );
      const int p[6] = { affLT.getHor() * ( 1 << shift ), affLT.getVer() * ( 1 << shift ), dHorX, dHorY, dVerX, dVerY };
      std::copy( p, p + 6, params[list] );
    }

    for( int y = cu.ly(); y < cu.ly() + int( cu.lheight() ); y += 16 )
    {
      for( int x = cu.lx(); x < cu.lx() + int( cu.lwidth() ); x += 16 )
      {
        gpu::InterJob job, next;
        job.set( x, y, std::min<int>( 16, cu.lwidth() ), std::min<int>( 16, cu.lheight() ), aflags | gpu::InterJob::kAffine, 0, 0, 0, 0, cu.lx(),
                 cu.ly(), ref[0], ref[1], 0 );
        if( bcw )
        {
          job.setBcw( getBcwWeight( g_BcwInternBcw[cu.BcwIdx()], REF_PIC_LIST_1 ) );
        }
        const int8_t refIdx[2] = { int8_t( cu.refIdx[0] ), int8_t( cu.refIdx[1] ) };
        gpuWp( cu, refIdx, aflags, job );
        next.set( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 );
        gpu::InterJob::setAffine( job, next, cu.lwidth(), cu.lheight(), prof, spread, params );
        jobs.push_back( job );
        jobs.push_back( next );
      }
    }
  }
  else if( cu.mergeType() == MRG_TYPE_SUBPU_ATMVP )
  {
    const Position puPos       = cu.lumaPos();
    const Size     puSize      = cu.lumaSize();
    const int      numPartLine = std::max<SizeType>( puSize.width >> ATMVP_SUB_BLOCK_SIZE, 1u );
    const int      numPartCol  = std::max<SizeType>( puSize.height >> ATMVP_SUB_BLOCK_SIZE, 1u );
    const int      puHeight    = numPartCol == 1 ? puSize.height : 1 << ATMVP_SUB_BLOCK_SIZE;
    const int      puWidth     = numPartLine == 1 ? puSize.width : 1 << ATMVP_SUB_BLOCK_SIZE;
    const bool     verMC       = puSize.height > puSize.width;
    const int      fstStart    = !verMC ? puPos.y : puPos.x;
    const int      secStart    = !verMC ? puPos.x : puPos.y;
    const int      fstEnd      = !verMC ? puPos.y + puSize.height : puPos.x + puSize.width;
    const int      secEnd      = !verMC ? puPos.x + puSize.width : puPos.y + puSize.height;
    const int      fstStep     = !verMC ? puHeight : puWidth;
    const int      secStep     = !verMC ? puWidth : puHeight;
    const bool     scaled      = slice.getRefPic( REF_PIC_LIST_0, 0 )->isRefScaled( cu.pps )
                                 || ( slice.getSliceType() == B_SLICE ? slice.getRefPic( REF_PIC_LIST_1, 0 )->isRefScaled( cu.pps ) : false );
    for( int fstDim = fstStart; fstDim < fstEnd; fstDim += fstStep )
    {
      for( int secDim = secStart; secDim < secEnd; secDim += secStep )
      {
        int               x     = !verMC ? secDim : fstDim;
        int               y     = !verMC ? fstDim : secDim;
        const MotionInfo& curMi = cu.getMotionInfo( Position{ x, y } );
        int               length = secStep;
        int               later  = secDim + secStep;
        while( later < secEnd )
        {
          const MotionInfo& laterMi = !verMC ? cu.getMotionInfo( Position{ later, fstDim } ) : cu.getMotionInfo( Position{ fstDim, later } );
          if( !scaled && laterMi == curMi )
          {
            length += secStep;
          }
          else
          {
            break;
          }
          later += secStep;
        }
        int dx = !verMC ? length : puWidth;
        int dy = !verMC ? puHeight : length;
        if( !verMC && ( dx & 15 ) && dx > 16 )
        {
          const int dxPart = dx & ~15;
          mcBlock( x, y, dxPart, dy, curMi.miRefIdx, curMi.mv, true );
          x  += dxPart;
          dx -= dxPart;
        }
        else if( verMC && ( dy & 15 ) && dy > 16 )
        {
          const int dyPart = dy & ~15;
          mcBlock( x, y, dx, dyPart, curMi.miRefIdx, curMi.mv, true );
          y  += dyPart;
          dy -= dyPart;
        }
        mcBlock( x, y, dx, dy, curMi.miRefIdx, curMi.mv, true );
        secDim = later - secStep;
      }
    }
  }
  else
  {
    const int8_t refIdx[2] = { int8_t( cu.refIdx[0] ), int8_t( cu.refIdx[1] ) };
    const Mv     mv[2]     = { cu.mv[0][0], cu.mv[1][0] };
    mcBlock( cu.lx(), cu.ly(), cu.lwidth(), cu.lheight(), refIdx, mv, false );
  }

  if( cu.ciipFlag() )
  {
    xGpuCiipJobs( cu, intraJobs, waves, lmcsChroma );
  }
  // the chroma kept aside: each transform block's prediction plus its
  // residual, scaled where finishLMCSAndReco scales it
  else if( flags & ( gpu::InterJob::kDeferCb | gpu::InterJob::kDeferCb << 1 ) )
  {
    for( TransformUnit &tu : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
    {
      for( int c = COMPONENT_Cb; c <= COMPONENT_Cr; c++ )
      {
        const ComponentID compID = ComponentID( c );
        const CompArea&   block  = tu.blocks[compID];
        if( !( flags & ( gpu::InterJob::kDeferCb << ( c - 1 ) ) ) || !block.valid() )
        {
          continue;
        }
        // (in blocks of powers of two: SBT's transform units without the
        // residual are 3/4 of the CU, the scaled ones powers of two)
        const bool scaled = ( TU::getCbf( tu, compID ) || tu.jointCbCr ) && block.area() > 4;
        for( int y = 0; y < int( block.height ); y += 1 << getLog2( block.height - y ) )
        {
          for( int x = 0; x < int( block.width ); x += 1 << getLog2( block.width - x ) )
          {
            const int     bw = 1 << getLog2( block.width - x );
            const int     bh = 1 << getLog2( block.height - y );
            gpu::IntraJob job;
            job.set( block.x + x, block.y + y, bw, bh, compID, gpu::IntraJob::kLmcs, 0, gpu::IntraJob::kResidual, 0 );
            int wave = 0;
            if( scaled )
            {
              wave = gpuLmcsChroma( cu, tu.blocks[COMPONENT_Y].pos(), job, waves );
            }
            job.setWave( wave + 1 );
            waves.set( compID, block.x + x, block.y + y, bw, bh, wave + 1 );
            intraJobs.push_back( job );
          }
        }
      }
    }
  }
}

void DecCu::xGpuCiipJobs( CodingUnit& cu, std::vector<gpu::IntraJob>& jobs, gpu::WaveMap& waves, bool lmcsChroma )
{
  // predBlendIntraCiip: planar's weight from whether the CUs left of the CU's
  // bottom-left sample and above its top-right one are intra
  const CodingUnit* cuLeft     = cu.cs->getCURestricted( cu.Y().bottomLeft().offset( -1, 0 ), cu, CHANNEL_TYPE_LUMA, cu.left );
  const CodingUnit* cuAbove    = cu.cs->getCURestricted( cu.Y().topRight().offset( 0, -1 ), cu, CHANNEL_TYPE_LUMA, cu.above );
  const int         wIntra     = 3 - !( cuLeft && CU::isIntra( *cuLeft ) ) - !( cuAbove && CU::isIntra( *cuAbove ) );
  const bool        blendChroma = isChromaEnabled( cu.chromaFormat ) && cu.chromaSize().width > 2;

  // a job a component of the whole CU, as predBlendIntraCiip predicts it
  // (planar, from the CU's references; luma's [1 2 1]-filtered where
  // useFilteredIntraRefSamples says), in its order: the reuse of the
  // neighbourhood across the CU's components depends on it
  for( int c = 0; c < getNumberValidComponents( cu.chromaFormat ); c++ )
  {
    const ComponentID compID = ComponentID( c );
    const CompArea&   area   = cu.blocks[compID];
    int               units[3];
    m_pcIntraPred->getNeighborUnits( cu.firstTU, area, units );

    int flags = 0;
    if( cu.planeCbf( compID ) )
    {
      flags |= gpu::IntraJob::kResidual;
    }
    if( isLuma( compID ) && IntraPrediction::useFilteredIntraRefSamples( COMPONENT_Y, cu, cu ) )
    {
      flags |= gpu::IntraJob::kFiltered;
    }
    gpu::IntraJob job;
    job.set( area.x, area.y, area.width, area.height, compID, gpu::IntraJob::kCiip, PLANAR_IDX, flags, 0 );
    job.setNeighbors( units[0], units[1], units[2] );
    job.setCiip( isLuma( compID ) || blendChroma ? wIntra : 0 );

    int wave = std::max( waves.maxIn( compID, area.x - 1, area.y - 1, 2 * area.width + 1, 1 ),
                         waves.maxIn( compID, area.x - 1, area.y, 1, 2 * area.height ) );
    // LMCS: the chroma residual scaled (finishLMCSAndReco: a CU of CIIP lies
    // in one VPDU, its transform units of the same size)
    if( isChroma( compID ) && lmcsChroma && cu.firstTU.blocks[compID].area() > 4 )
    {
      bool residual = false;
      for( const TransformUnit &tu : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
      {
        residual |= TU::getCbf( tu, compID ) || tu.jointCbCr;
      }
      if( residual )
      {
        wave = std::max( wave, gpuLmcsChroma( cu, cu.firstTU.blocks[COMPONENT_Y].pos(), job, waves ) );
      }
    }
    job.setWave( wave + 1 );
    waves.set( compID, area.x, area.y, area.width, area.height, wave + 1 );
    jobs.push_back( job );
  }
}

void DecCu::xGpuTrafoJobs( CodingUnit& cu, std::vector<gpu::TrafoJob>& jobs )
{
  // the transform blocks reconstructResi takes, with the parameters TrQuant uses
  for( TransformUnit &tu : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
  {
    for( const auto &area : tu.blocks )
    {
      if( !area.valid() )
      {
        continue;
      }

      ComponentID compID = area.compID();
      int         ict    = 0;
      if( tu.jointCbCr && isChroma( compID ) )
      {
        if( compID != COMPONENT_Cb )
        {
          continue;   // with Cb's job
        }
        compID = ( tu.jointCbCr >> 1 ) ? COMPONENT_Cb : COMPONENT_Cr;
        ict    = TU::getICTMode( tu, cu.cs->picHeader->getJointCbCrSignFlag() );
      }
      else if( !TU::getCbf( tu, compID ) )
      {
        continue;
      }

      const CompArea& block = tu.blocks[compID];
      const QpParam   cQP( tu, compID );
      const bool      ts    = tu.mtsIdx( compID ) == MTS_SKIP;
      // the slice's explicit scaling list, where Quant::dequant applies it: not
      // with transform skip, nor where the SPS leaves LFNST's blocks or ACT's
      // colour space out
      const SPS&                sps          = *cu.sps;
      const bool                lfnstApplied = cu.lfnstIdx() > 0 && ( CU::isSepTree( cu ) || isLuma( compID ) );
      const bool                scalingList  = cu.slice->getExplicitScalingListUsed() && !ts && !( lfnstApplied && sps.getDisableScalingMatrixForLfnstBlks() )
                                               && !( sps.getScalingMatrixForAlternativeColourSpaceDisabledFlag()
                                                     && sps.getScalingMatrixDesignatedColourSpaceFlag() == cu.colorTransform() );
      const Quant::DequantParam param        = Quant::dequantParam( tu, compID, cQP, scalingList );
      CHECK( param.bdpcm && !ts, "a residual the GPU cannot reconstruct" );

      int typeHor = DCT2;
      int typeVer = DCT2;
      int flags   = 0;
      if( ts )
      {
        flags |= gpu::TrafoJob::kTransformSkip;
      }
      else
      {
        m_pcTrQuant->getTrTypes( tu, compID, typeHor, typeVer );
      }
      if( param.bdpcm )
      {
        flags |= ( isLuma( compID ) ? cu.bdpcmMode() : cu.bdpcmModeChroma() ) == 1 ? gpu::TrafoJob::kBdpcmHor : gpu::TrafoJob::kBdpcmVer;
      }

      gpu::TrafoJob job;
      job.set( block.x, block.y, getLog2( block.width ), getLog2( block.height ), compID, typeHor, typeVer, flags, ict, param.maxX, param.maxY,
               param.scale, param.rightShift, param.inputMaximum );
      // LFNST, as TrQuant::xInvLfnst picks its set and orientation (luma's
      // only, but in a separate tree)
      if( cu.lfnstIdx() && cu.sps->getUseLFNST() && !ts && ( CU::isSepTree( cu ) || isLuma( compID ) ) )
      {
        const ChannelType chType    = toChannelType( compID );
        uint32_t          intraMode = CU::isMIP( cu, chType )                ? PLANAR_IDX
                                      : PU::isLMCMode( cu.intraDir[chType] ) ? PU::getCoLocatedIntraLumaMode( cu )
                                                                             : PU::getFinalIntraMode( cu, chType );
        intraMode = m_pcTrQuant->getLFNSTIntraMode( PU::getWideAngIntraMode( tu, intraMode, compID ) );
        job.setLfnst( cu.lfnstIdx(), g_lfnstLut[intraMode], m_pcTrQuant->getTransposeFlag( intraMode ), block.width >= 8 && block.height >= 8,
                      ( block.width == 4 && block.height == 4 ) || ( block.width == 8 && block.height == 8 ) );
      }
      if( scalingList )
      {
        // the matrix of the block's size and of its list (Quant's
        // getScalingListType: intra, else inter, by component)
        const int list = ( cu.predMode() == MODE_INTRA ? 0 : MAX_NUM_COMPONENT ) + compID;
        job.setScalingList( int( m_pcTrQuant->getDequantCoeff( list, getLog2( block.width ), getLog2( block.height ) ) - m_pcTrQuant->getDequantCoeffBuf() ) );
      }
      jobs.push_back( job );
    }
  }
}
#endif

// ====================================================================================================================
// Constructor / destructor / create / destroy
// ====================================================================================================================

DecCu::DecCu()
{
  create();
}

DecCu::~DecCu()
{
  destroy();
}

void DecCu::init( IntraPrediction* pcIntra, InterPrediction* pcInter, Reshape *pcReshape, TrQuant *pcTrQuant )
{
  m_pcIntraPred  = pcIntra;
  m_pcInterPred  = pcInter;
  m_pcTrQuant    = pcTrQuant;
  m_pcReshape    = pcReshape;
}

void DecCu::create()
{
}

void DecCu::destroy()
{
}

// ====================================================================================================================
// Public member functions
// ====================================================================================================================

void DecCu::predAndReco( CodingUnit& cu, bool doCiipIntra )
{
  PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_ITRANS_REC, *cu.cs, CH_L );
  CodingStructure &cs = *cu.cs;
  
  if( CU::isIntra( cu ) )
  {
    if( cu.colorTransform() )
    {
      xIntraRecACT( cu );
      return;
    }

    for( TransformUnit &tu : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
    {
      for( const CompArea &area : tu.blocks )
      {
        if( !area.valid() )
        {
          continue;
        }

        const ComponentID compID = area.compID();
        const ChannelType chType = toChannelType( compID );

        PelBuf piPred;
        {
          PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_INTRAPRED, cs, compID );

          if( cu.planeCbf( compID ) )
          {
            if( chType == CHANNEL_TYPE_LUMA )
            {
              Position pos = Position( tu.Y().x - cu.lumaPos().x, tu.Y().y - cu.lumaPos().y );
              piPred = cs.getPredBuf( cu ).Y().subBuf( pos, tu.lumaSize() );
            }
            else
            {
              piPred = cs.getPredBuf( cu ).Cb().subBuf( Position( 0, 0 ), tu.chromaSize() );
            }
          }
          else
          {
            piPred = cs.getRecoBuf( area );
          }

          const uint32_t uiChFinalMode  = PU::getFinalIntraMode( cu, chType );

          if( CU::isMIP( cu, chType ) )
          {
            m_pcIntraPred->initIntraPatternChType( tu, area );
            m_pcIntraPred->initIntraMip( cu, area );
            m_pcIntraPred->predIntraMip( compID, piPred, cu );
          }
          else if( compID != COMPONENT_Y && PU::isLMCMode( uiChFinalMode ) )
          {
            m_pcIntraPred->initIntraPatternChType( tu, area );
            m_pcIntraPred->xGetLumaRecPixels( cu, area );
            m_pcIntraPred->predIntraChromaLM( compID, piPred, cu, area, uiChFinalMode );
          }
          else
          {
            const bool predRegDiffFromTB = CU::isPredRegDiffFromTB( *tu.cu, compID );
            const bool firstTBInPredReg  = CU::isFirstTBInPredReg( *tu.cu, compID, area );
            CompArea areaPredReg( COMPONENT_Y, area );

            const bool bUseFilteredPredictions = isLuma( compID ) && cu.ispMode() == NOT_INTRA_SUBPARTITIONS && IntraPrediction::useFilteredIntraRefSamples( compID, cu, tu );

            if( tu.cu->ispMode() && isLuma( compID ) )
            {
              PelBuf piReco = cs.getRecoBuf( area );
              if( predRegDiffFromTB )
              {
                if( firstTBInPredReg )
                {
                  CU::adjustPredArea( areaPredReg );
                  m_pcIntraPred->initIntraPatternChTypeISP( *tu.cu, areaPredReg, piReco );
                }
              }
              else
              {
                m_pcIntraPred->initIntraPatternChTypeISP( *tu.cu, area, piReco );
              }
            }
            else
            {
              m_pcIntraPred->initIntraPatternChType( tu, area, bUseFilteredPredictions );
            }

            if( predRegDiffFromTB )
            {
              if( firstTBInPredReg )
              {
                Position pos( areaPredReg.x - cu.lumaPos().x, areaPredReg.y - cu.lumaPos().y );
                piPred      = cs.getPredBuf( cu ).Y().subBuf( pos, areaPredReg.size() );

                m_pcIntraPred->predIntraAng( compID, piPred, cu, bUseFilteredPredictions );

                piPred.width = area.width;
              }
            }
            else
              m_pcIntraPred->predIntraAng( compID, piPred, cu, bUseFilteredPredictions );
          }
        }

        //===== inverse transform =====
        PelBuf piReco = cs.getRecoBuf( area );
        PelBuf piResi = cs.getRecoBuf( area );

        const Slice &slice      = *cu.slice;
        const bool   doChrScale = isChroma( compID )
                                  && slice.getLmcsEnabledFlag()
                                  && slice.getPicHeader()->getLmcsChromaResidualScaleFlag()
                                  && tu.blocks[compID].area() > 4
                                  && ( TU::getCbf( tu, compID ) || tu.jointCbCr )
                                  ;

        if( doChrScale )
        {
          const Area area       = tu.Y().valid() ? tu.Y() : Area(recalcPosition(tu.chromaFormat, tu.chType(), CHANNEL_TYPE_LUMA, tu.blocks[tu.chType()].pos()), recalcSize(tu.chromaFormat, tu.chType(), CHANNEL_TYPE_LUMA, tu.blocks[tu.chType()].size()));
          int chromaResScaleInv = m_pcReshape->calculateChromaAdjVpduNei( tu, area.pos() );
          piReco.scaleSignal( chromaResScaleInv, slice.clpRng( compID ) );
        }

        if( TU::getCbf( tu, compID ) || ( isChroma( compID ) && tu.jointCbCr ) )
        {
          piReco.reconstruct( piPred, piResi, slice.clpRng( compID ) );
        }
        else if( cu.planeCbf( compID ) )
        {
          piReco.copyFrom( piPred );
        }
      }
    }
  }
  else
  {
    const UnitArea& cuArea = cu;

    PelUnitBuf predBuf = cu.rootCbf() ? cs.getPredBuf( cu ) : cs.getRecoBuf( cuArea );
    PelUnitBuf recoBuf = cs.getRecoBuf( cuArea );

    // CBF in at least one channel, but no TU split
    if( cu.rootCbf() )
    {
      for( const auto &blk : cu.blocks )
      {
        if( blk.valid() && !cu.planeCbf( blk.compID() ) )
        {
          predBuf.bufs[blk.compID()] = recoBuf.bufs[blk.compID()];
        }
      }
    }

    if( cu.geoFlag() )
    {
      m_pcInterPred->motionCompensationGeo( cu, predBuf );
    }
    else
    {
      // inter prediction
      CHECK( CU::isIBC( cu ) && cu.ciipFlag(),       "IBC and CIIP cannot be used together"   );
      CHECK( CU::isIBC( cu ) && cu.affineFlag(),     "IBC and AFFINE cannot be used together" );
      CHECK( CU::isIBC( cu ) && cu.geoFlag(),        "IBC and GEO cannot be used together"    );
      CHECK( CU::isIBC( cu ) && cu.mmvdFlag(),       "IBC and MMVD cannot be used together"     );

      if( !CU::isIBC( cu ) && !doCiipIntra )
      {
        m_pcInterPred->motionCompensation( cu, predBuf );
      }
      else if( CU::isIBC( cu ) )
      {
        const bool luma   = cu.Y ().valid();
        const bool chroma = isChromaEnabled(cu.chromaFormat) && cu.Cb().valid();
        m_pcInterPred->motionCompensation( cu, predBuf, luma, chroma );
      }
    }

    if( cu.ciipFlag() && doCiipIntra )
    {
      m_pcIntraPred->predBlendIntraCiip( predBuf, cu );
    }

    DTRACE    ( g_trace_ctx, D_TMP, "pred " );
    DTRACE_CRC( g_trace_ctx, D_TMP, *cu.cs, predBuf, &cu.Y() );

    if( cu.rootCbf() )
    {
      if( cu.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag( *cu.slice ) && !doCiipIntra )
      {
         m_pcReshape->rspBufFwd( predBuf.Y() );
      }

      if( ( cu.ciipFlag() && doCiipIntra ) || CU::isIBC( cu ) )
      {
        finishLMCSAndReco( cu );
      }
    }
    else
    {
      if( cu.slice->getLmcsEnabledFlag() && m_pcReshape->getCTUFlag( *cu.slice ) && !CU::isIBC( cu ) )
      {
        m_pcReshape->rspBufFwd( predBuf.Y() );
      }
    }

    DTRACE    ( g_trace_ctx, D_TMP, "reco " );
    DTRACE_CRC( g_trace_ctx, D_TMP, *cu.cs, cu.cs->getRecoBuf( cuArea ), &cu.Y() );
  }
}

void DecCu::finishLMCSAndReco( CodingUnit &cu )
{
  PROFILER_SCOPE_AND_STAGE_EXT( 1, g_timeProfiler, P_ITRANS_REC, *cu.cs, CH_L );
  CodingStructure &cs = *cu.cs;

  const uint32_t uiNumVaildComp = getNumberValidComponents( cu.chromaFormat );
  const bool     doCS           = cs.picHeader->getLmcsEnabledFlag() && cs.picHeader->getLmcsChromaResidualScaleFlag() && cu.slice->getLmcsEnabledFlag();
  const PelUnitBuf predUnitBuf  = cs.getPredBuf( cu );

  for( auto& currTU : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
  {
    int chromaResScaleInv = 0;

    for( uint32_t ch = 0; ch < uiNumVaildComp; ch++ )
    {
      const ComponentID compID = ComponentID( ch );
        
      if( doCS )
      {
        if( isLuma( compID ) )
        {
          chromaResScaleInv = m_pcReshape->calculateChromaAdjVpduNei( currTU, currTU.blocks[COMPONENT_Y] );
        }
        else if( ( TU::getCbf( currTU, compID ) || currTU.jointCbCr ) && currTU.blocks[compID].area() > 4 )
        {
          const CompArea &area = currTU.blocks[compID];
          PelBuf resiBuf       = cs.getRecoBuf( area );
          resiBuf.scaleSignal( chromaResScaleInv, cu.slice->clpRng( compID ) );
        }
      }

      if( currTU.blocks[compID].valid() )
      {
        CPelBuf predBuf = predUnitBuf.bufs[compID];
        OFFSET( predBuf.buf, predBuf.stride, currTU.blocks[compID].x - cu.blocks[compID].x, currTU.blocks[compID].y - cu.blocks[compID].y );
        //CPelBuf resiBuf = cs.getRecoBuf( currTU.blocks[compID] );
         PelBuf recoBuf = cs.getRecoBuf( currTU.blocks[compID] );
         predBuf.width  = recoBuf.width;
         predBuf.height = recoBuf.height;

        if( TU::getCbf( currTU, compID ) || ( isChroma( compID ) && currTU.jointCbCr ) )
        {
          recoBuf.reconstruct( predBuf, recoBuf, cu.slice->clpRngs() );
        }
        else if( cu.planeCbf( compID ) )
        {
          recoBuf.copyFrom( predBuf );
        }
      }
    }
  }
}

void DecCu::reconstructResi( CodingUnit &cu )
{
  CodingStructure &cs = *cu.cs;

  for( TransformUnit &tu : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
  {
    for( const auto &area : tu.blocks )
    {
      if( !area.valid() )
      {
        continue;
      }

      const ComponentID compID = area.compID();

      //===== inverse transform =====
      PelBuf piResi = cs.getRecoBuf( area );

      if( tu.jointCbCr && isChroma( compID ) )
      {
        if( compID == COMPONENT_Cb )
        {
          PelBuf resiCr = cs.getRecoBuf( tu.blocks[ COMPONENT_Cr ] );
          if( tu.jointCbCr >> 1 )
          {
            QpParam qpCb( tu, compID );
            m_pcTrQuant->invTransformNxN( tu, COMPONENT_Cb, piResi, qpCb );
          }
          else
          {
            QpParam qpCr( tu, COMPONENT_Cr );
            m_pcTrQuant->invTransformNxN( tu, COMPONENT_Cr, resiCr, qpCr );
          }
          m_pcTrQuant->invTransformICT( tu, piResi, resiCr );
        }
      }
      else if( TU::getCbf( tu, compID ) )
      {
        QpParam cQP( tu, compID );
        m_pcTrQuant->invTransformNxN( tu, compID, piResi, cQP );
      }
    }
  }
}

// ====================================================================================================================
// Protected member functions
// ====================================================================================================================

/** Function for deriving reconstructed luma/chroma samples of a PCM mode CU.
* \param pcCU pointer to current CU
* \param uiPartIdx part index
* \param piPCM pointer to PCM code arrays
* \param piReco pointer to reconstructed sample arrays
* \param uiStride stride of reconstructed sample arrays
* \param uiWidth CU width
* \param uiHeight CU height
* \param compID colour component ID
* \returns void
*/

void DecCu::xIntraRecACT( CodingUnit &cu )
{
  CodingStructure &cs  = *cu.cs;
  const Slice &slice   = *cu.slice;

  for( TransformUnit &tu : TUTraverser( &cu.firstTU, cu.lastTU->next ) )
  {
    cs.getRecoBuf( tu ).colorSpaceConvert( cs.getRecoBuf( tu ), slice.clpRng( COMPONENT_Y ) );

    for( const CompArea &area : tu.blocks )
    {
      const ComponentID compID = area.compID();

      if( compID == COMPONENT_Y )
      {
        continue;
      }
      //===== inverse transform =====
      PelBuf piReco = cs.getRecoBuf( area );

      const bool   doChrScale = isChroma( compID )
                                && slice.getLmcsEnabledFlag() 
                                && slice.getPicHeader()->getLmcsChromaResidualScaleFlag()
                                && tu.blocks[compID].area() > 4
                                && ( TU::getCbf( tu, compID ) || tu.jointCbCr )
                                ;

      if( doChrScale )
      {
        const Area area       = tu.Y().valid() ? tu.Y() : Area(recalcPosition(tu.chromaFormat, tu.chType(), CHANNEL_TYPE_LUMA, tu.blocks[tu.chType()].pos()), recalcSize(tu.chromaFormat, tu.chType(), CHANNEL_TYPE_LUMA, tu.blocks[tu.chType()].size()));
        int chromaResScaleInv = m_pcReshape->calculateChromaAdjVpduNei( tu, area.pos() );
        piReco.scaleSignal( chromaResScaleInv, slice.clpRng( compID ) );
      }
    }

    for( const CompArea &area : tu.blocks )
    {
      if( !area.valid() )
      {
        continue;
      }

      const ComponentID compID = area.compID();
      const ChannelType chType = toChannelType( compID );

      PelBuf piPred;
      if( chType == CHANNEL_TYPE_LUMA )
      {
        Position pos = Position( tu.Y().x - cu.lumaPos().x, tu.Y().y - cu.lumaPos().y );
        piPred = cs.getPredBuf( cu ).Y().subBuf( pos, tu.lumaSize() );
      }
      else
      {
        piPred = cs.getPredBuf( cu ).Cb().subBuf( Position( 0, 0 ), tu.chromaSize() );
      }

      const uint32_t uiChFinalMode  = PU::getFinalIntraMode( cu, chType );

      bool predRegDiffFromTB = CU::isPredRegDiffFromTB( *tu.cu, compID );
      bool firstTBInPredReg  = CU::isFirstTBInPredReg( *tu.cu, compID, area );
      CompArea areaPredReg   ( COMPONENT_Y, area );

      if( CU::isMIP( cu, chType ) )
      {
        m_pcIntraPred->initIntraPatternChType( tu, area );
        m_pcIntraPred->initIntraMip( cu, area );
        m_pcIntraPred->predIntraMip( compID, piPred, cu );
      }
      else
      {
        const bool bUseFilteredPredictions = isLuma( chType ) && cu.ispMode() == NOT_INTRA_SUBPARTITIONS && IntraPrediction::useFilteredIntraRefSamples( compID, cu, tu );
        
        if( tu.cu->ispMode() && isLuma( compID ) )
        {
          PelBuf piReco = cs.getRecoBuf( area );
          if( predRegDiffFromTB )
          {
            if( firstTBInPredReg )
            {
              CU::adjustPredArea( areaPredReg );
              m_pcIntraPred->initIntraPatternChTypeISP( *tu.cu, areaPredReg, piReco );
            }
          }
          else
          {
            m_pcIntraPred->initIntraPatternChTypeISP( *tu.cu, area, piReco );
          }
        }
        else
        {
          m_pcIntraPred->initIntraPatternChType( tu, area, bUseFilteredPredictions );
        }

        if( compID != COMPONENT_Y && PU::isLMCMode( uiChFinalMode ) )
        {
          m_pcIntraPred->xGetLumaRecPixels( cu, area );
          m_pcIntraPred->predIntraChromaLM( compID, piPred, cu, area, uiChFinalMode );
        }
        else
        {
          if( predRegDiffFromTB )
          {
            if( firstTBInPredReg )
            {
              Position pos( areaPredReg.x - cu.lumaPos().x, areaPredReg.y - cu.lumaPos().y );
              piPred      = cs.getPredBuf( cu ).Y().subBuf( pos, areaPredReg.size() );

              m_pcIntraPred->predIntraAng( compID, piPred, cu, bUseFilteredPredictions );
            }
          }
          else
            m_pcIntraPred->predIntraAng( compID, piPred, cu, bUseFilteredPredictions );
        }
      }

      PelBuf piReco = cs.getRecoBuf( area );
      PelBuf piResi = cs.getRecoBuf( area );

      piReco.reconstruct( piPred, piResi, slice.clpRng( compID ) );
    }
  }
}

void DecCu::xDeriveCUMV( CodingUnit &cu, MotionHist& hist )
{
  if( cu.mergeFlag() )
  {
    MergeCtx mrgCtx;

    if( cu.mmvdFlag() )
    {
      int fPosBaseIdx = cu.mmvdIdx / MMVD_MAX_REFINE_NUM;

      PU::getInterMergeCandidates     ( cu, mrgCtx, hist, fPosBaseIdx + 1 );
      PU::getInterMMVDMergeCandidates ( cu, mrgCtx,       cu.mmvdIdx );
      mrgCtx.setMmvdMergeCandiInfo    ( cu, cu.mmvdIdx );

      PU::spanMotionInfo              ( cu );
    }
    else if( cu.geoFlag() )
    {
      PU::getGeoMergeCandidates( cu, m_geoMrgCtx, hist );

      const uint8_t splitDir = cu.geoSplitDir;
      const uint8_t candIdx0 = cu.geoMergeIdx0();
      const uint8_t candIdx1 = cu.geoMergeIdx1();

      PU::spanGeoMotionInfo( cu, m_geoMrgCtx, splitDir, candIdx0, candIdx1 );
    }
    else if( cu.affineFlag() )
    {
      AffineMergeCtx affineMergeCtx;

      if( cu.sps->getSBTMVPEnabledFlag() )
      {
        affineMergeCtx.subPuMvpMiBuf = cu.getMotionBuf();
      }

      int mergeIdx = cu.mergeIdx();
      PU::getAffineMergeCand( cu, affineMergeCtx, mergeIdx );

      cu.setInterDir        ( affineMergeCtx.interDirNeighbours[mergeIdx] );
      cu.setAffineType      ( affineMergeCtx.affineType        [mergeIdx] );
      cu.setBcwIdx          ( affineMergeCtx.BcwIdx            [mergeIdx] );
      cu.setMergeType       ( affineMergeCtx.mergeType         [mergeIdx] );

      if( cu.mergeType() == MRG_TYPE_SUBPU_ATMVP )
      {
        cu.refIdx[0] = affineMergeCtx.mvFieldNeighbours[( cu.mergeIdx() << 1 ) + 0][0].mfRefIdx;
        cu.refIdx[1] = affineMergeCtx.mvFieldNeighbours[( cu.mergeIdx() << 1 ) + 1][0].mfRefIdx;
      }
      else
      {
        for( int i = 0; i < 2; ++i )
        {
          if( cu.slice->getNumRefIdx( RefPicList( i ) ) > 0 )
          {
            MvField* mvField = affineMergeCtx.mvFieldNeighbours[( cu.mergeIdx() << 1 ) + i];

            PU::setAllAffineMvField( cu, mvField, RefPicList( i ) );
          }
        }
      }

      PU::spanMotionInfo( cu );
    }
    else
    {
      if( CU::isIBC( cu ) )
        PU::getIBCMergeCandidates  ( cu, mrgCtx, hist, cu.mergeIdx() );
      else
        PU::getInterMergeCandidates( cu, mrgCtx, hist, cu.mergeIdx() );

      mrgCtx.setMergeInfo         ( cu, cu.mergeIdx() );

      PU::spanMotionInfo          ( cu );
    }
  }
  else
  {
    if( cu.imv() && !cu.affineFlag() )
    {
      PU::applyImv      ( cu, hist );
      PU::spanMotionInfo( cu );
    }
    else
    {
      if( cu.affineFlag() )
      {
        for ( uint32_t uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
        {
          RefPicList eRefList = RefPicList( uiRefListIdx );
          if( cu.slice->getNumRefIdx( eRefList ) > 0 && ( cu.interDir() & ( 1 << uiRefListIdx ) ) )
          {
            AffineAMVPInfo affineAMVPInfo;
            PU::fillAffineMvpCand( cu, eRefList, cu.refIdx[eRefList], affineAMVPInfo );

            const unsigned mvp_idx = cu.mvpIdx[eRefList];

            //    Mv mv[3];
            CHECK( cu.refIdx[eRefList] < 0, "Unexpected negative refIdx." );
            const int imvShift = cu.imv() == 2 ? MV_FRACTIONAL_BITS_DIFF : 0;
            Mv mv0 = cu.mv[eRefList][0];
            Mv mv1 = cu.mv[eRefList][1];

            mv0 <<= imvShift;
            mv1 <<= imvShift;

            Mv mvLT = affineAMVPInfo.mvCandLT[mvp_idx] + mv0;
            Mv mvRT = affineAMVPInfo.mvCandRT[mvp_idx] + mv1;
            mvRT   += mv0;
            if( cu.imv() != 1 )
            {
              mvLT.changePrecision( MV_PRECISION_QUARTER, MV_PRECISION_INTERNAL );
              mvRT.changePrecision( MV_PRECISION_QUARTER, MV_PRECISION_INTERNAL );
            }

            Mv mvLB;
            if( cu.affineType() == AFFINEMODEL_6PARAM )
            {
              Mv mv2 = cu.mv[eRefList][2];
              mv2 <<= imvShift;
              mvLB = affineAMVPInfo.mvCandLB[mvp_idx] + mv2;
              mvLB += mv0;
              if ( cu.imv() != 1 )
              {
                mvLB.changePrecision( MV_PRECISION_QUARTER, MV_PRECISION_INTERNAL );
              }
            }
            PU::setAllAffineMv( cu, mvLT, mvRT, mvLB, eRefList, true );
          }
        }
      }
      else if( CU::isIBC( cu ) && cu.interDir() == 1 )
      {
        AMVPInfo amvpInfo;
        PU::fillIBCMvpCand( cu, amvpInfo, hist );
        Mv mvd = cu.mv[REF_PIC_LIST_0][0];
        mvd <<= 4;
        if( cu.slice->getPicHeader()->getMaxNumIBCMergeCand() == 1 )
        {
          CHECK( cu.mvpIdx[REF_PIC_LIST_0], "mvpIdx for IBC mode should be 0" );
        }
        cu.mv[REF_PIC_LIST_0][0] = amvpInfo.mvCand[cu.mvpIdx[REF_PIC_LIST_0]] + mvd;
        cu.mv[REF_PIC_LIST_0][0] . mvCliptoStorageBitDepth();
      }
      else
      {
        for( uint32_t uiRefListIdx = 0; uiRefListIdx < 2; uiRefListIdx++ )
        {
          RefPicList eRefList = RefPicList( uiRefListIdx );
          if( ( cu.slice->getNumRefIdx( eRefList ) > 0 || ( eRefList == REF_PIC_LIST_0 && CU::isIBC( cu ) ) ) && ( cu.interDir() & ( 1 << uiRefListIdx ) ) )
          {
            AMVPInfo amvpInfo;
            PU::fillMvpCand( cu, eRefList, cu.refIdx[eRefList], amvpInfo, hist );

            Mv mvd              = cu.mv[eRefList][0];
            mvd                 . changePrecisionAmvr( cu.imv(), MV_PRECISION_INTERNAL);
            cu.mv [eRefList][0] = amvpInfo.mvCand[cu.mvpIdx[eRefList]] + mvd;
            cu.mv [eRefList][0] . mvCliptoStorageBitDepth();
          }
        }
      }
      PU::spanMotionInfo( cu );
    }
  }

  bool isIbcSmallBlk = CU::isIBC( cu ) && ( cu.lwidth() * cu.lheight() <= 16 );
  if( !cu.affineFlag() && !cu.geoFlag() && !isIbcSmallBlk )
  {
    const unsigned log2ParallelMergeLevel = (cu.sps->getLog2ParallelMergeLevelMinus2() + 2);
    const unsigned xBr = cu.Y().width  + cu.Y().x;
    const unsigned yBr = cu.Y().height + cu.Y().y;
    bool enableHmvp      = ((xBr >> log2ParallelMergeLevel) > (cu.Y().x >> log2ParallelMergeLevel)) && ((yBr >> log2ParallelMergeLevel) > (cu.Y().y >> log2ParallelMergeLevel));
    bool isIbc           = CU::isIBC( cu );
    bool enableInsertion = isIbc || enableHmvp;

    if( enableInsertion )
    {
      HPMVInfo mi( cu.getMotionInfo(), cu.interDir() == 3 ? cu.BcwIdx() : BCW_DEFAULT, cu.imv() == IMV_HPEL );
      if( isIbc ) mi.mhRefIdx[0] = MH_NOT_VALID + 1;
      MotionHist::addMiToLut( isIbc ? hist.motionLutIbc : hist.motionLut, mi );
    }
  }
}

}
