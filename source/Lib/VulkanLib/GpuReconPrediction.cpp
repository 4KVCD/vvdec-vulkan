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

/** \file     GpuReconPrediction.cpp
    \brief    GpuRecon's reconstruction passes: the residuals, inter prediction, and intra prediction (see GpuRecon.h)
*/

#include "GpuRecon.h"

#include "vvdec_gpu/GpuShaderIds.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/MotionInfo.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Slice.h"

#include <cstring>

namespace vvdec
{
namespace gpu
{

// ====================================================================================================================
// What the GPU covers
// ====================================================================================================================

bool GpuRecon::intraSupported( const CodingStructure& cs )
{
  return intraSupported( *cs.sps );
}

bool GpuRecon::intraSupported( const SPS& sps )
{
  return !sps.getUseColorTrans();
}

bool GpuRecon::interSupported( const SPS& sps )
{
  return intraSupported( sps ) && sps.getBitDepth() <= 10;
}

bool GpuRecon::interSupported( const CodingStructure& cs )
{
  return interSupported( *cs.sps );
}

bool GpuRecon::residualSupported( const CodingStructure& cs )
{
  return cs.sps->getBitDepth() <= 10;
}

// ====================================================================================================================
// The residuals
// ====================================================================================================================

// The push constants of trafo.slang
struct TrafoPush
{
  VkDeviceAddress jobs, plane0, plane1, plane2, levels0, levels1, levels2, parsedLevels, matrices, lfnst, scaling;
  int32_t         stride0, stride12, count, bitDepth;
  VkDeviceAddress countAt;
};
static_assert( sizeof( TrafoPush ) == 112, "TrafoPush: the layout of trafo.slang's Params" );

void GpuRecon::recordTrafo( const CodingStructure& cs, const std::vector<std::vector<TrafoJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf,
                            const Planes& levels, const Planes& planes )
{
  // Without the front end, the CTU tasks' jobs are uploaded.
  size_t total = 0;
  if( !m_useFrontEnd )
  {
    for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
    {
      total += jobs[ctu].size();
    }
    if( total == 0 )
    {
      return;
    }
    const size_t size = total * sizeof( TrafoJob );
    if( buf.size < size )
    {
      m_ctx->free( buf );
      buf = m_ctx->alloc( size * 2, MemKind::Upload );
    }
    TrafoJob* all = reinterpret_cast<TrafoJob*>( buf.mapped );
    for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
    {
      memcpy( all, jobs[ctu].data(), jobs[ctu].size() * sizeof( TrafoJob ) );
      all += jobs[ctu].size();
    }
  }

  TrafoPush push{};
  push.jobs     = m_useFrontEnd ? m_frontEnd.jobs[0].addr : buf.addr;
  push.plane0   = planes.addr[COMPONENT_Y];
  push.plane1   = planes.addr[COMPONENT_Cb];
  push.plane2   = planes.addr[COMPONENT_Cr];
  push.levels0  = levels.addr[COMPONENT_Y];
  push.levels1  = levels.addr[COMPONENT_Cb];
  push.levels2  = levels.addr[COMPONENT_Cr];
  push.parsedLevels = m_frontEnd.recordLevels;   // (the GPU parse's: its jobs say where in them)
  push.matrices = m_tables.transform.addr;
  push.lfnst    = m_tables.lfnst.addr;
  push.scaling  = m_scalingOn ? m_scaling.addr : 0;
  push.stride0  = planes.stride[COMPONENT_Y];
  push.stride12 = planes.stride[COMPONENT_Cb];
  push.count    = m_useFrontEnd ? -1 : int32_t( total );
  push.countAt  = frontEndCount( kPassTrafo );
  push.bitDepth = cs.sps->getBitDepth();
  CHECK( planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );
  // (the levels in the picture in host memory, when the CPU parsed it: none for a picture the GPU parsed, whose levels
  // are in parsedLevels)
  CHECK( levels.addr[COMPONENT_Y] && ( levels.stride[COMPONENT_Y] != planes.stride[COMPONENT_Y] || levels.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cb] ),
         "the levels laid out differently" );

  // one workgroup per job
  bind( kShader_trafo );
  pushConstants( push );
  if( m_useFrontEnd )
  {
    dispatchIndirect( kPassTrafo );
  }
  else
  {
    dispatchJobs( total );
  }
  mark( "trafo" );
}

// ====================================================================================================================
// Inter prediction
// ====================================================================================================================

// The push constants of inter.slang
struct InterPush
{
  VkDeviceAddress jobs, refs, dst0, dst1, dst2, dmvr, colMotion, lmcsFwd, defer0, defer1, defer2, geo, subpics;
  int32_t         stride0, stride12, count, picWidth, picHeight, ctuSize, bitDepth, ctuLog2, widthInCtus, num8x8, wide, scaleX, scaleY, numComp, wrapSps,
                  wrapPps, wrapOffset, subpicOn, curWinLeft, curWinTop, horColloc, verColloc;
  VkDeviceAddress countAt;
};
static_assert( sizeof( InterPush ) == 200, "InterPush: the layout of inter.slang's Params" );
// the colocated motion as inter.slang writes DMVR's refinements into it
static_assert( sizeof( ColocatedMotionInfo ) == 20 && offsetof( ColocatedMotionInfo, mv ) == 0 && sizeof( Mv ) == 8,
               "the colocated motion is not laid out as shaders/inter.slang writes it" );

GpuRecon::InterCounts GpuRecon::uploadInterJobs( const std::vector<std::vector<InterJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf )
{
  size_t total = 0;
  for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
  {
    total += jobs[ctu].size();
  }
  InterCounts counts;
  if( total == 0 )
  {
    return counts;
  }
  const size_t size = total * sizeof( InterJob );
  if( buf.size < size )
  {
    m_ctx->free( buf );
    buf = m_ctx->alloc( size * 2, MemKind::Upload );
  }
  // The BDOF tiles go after the others, then the affine ones (two jobs each), for dispatches of the inter_bdof and
  // inter_affine variants of their own: their code would slow the others down. (The front end lays its jobs out the
  // same way.)
  for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
  {
    for( size_t i = 0; i < jobs[ctu].size(); i++ )
    {
      const uint32_t flags = jobs[ctu][i].w[1] >> 16;
      counts.bdof   += flags & InterJob::kBdof ? 1 : 0;
      counts.affine += flags & InterJob::kAffine ? 2 : 0;
      i             += flags & InterJob::kAffine ? 1 : 0;
    }
  }
  counts.regular   = total - counts.bdof - counts.affine;
  InterJob* all    = reinterpret_cast<InterJob*>( buf.mapped );
  InterJob* bdof   = all + counts.regular;
  InterJob* affine = bdof + counts.bdof;
  for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
  {
    if( counts.bdof + counts.affine == 0 )
    {
      memcpy( all, jobs[ctu].data(), jobs[ctu].size() * sizeof( InterJob ) );
      all += jobs[ctu].size();
      continue;
    }
    for( size_t i = 0; i < jobs[ctu].size(); i++ )
    {
      const uint32_t flags = jobs[ctu][i].w[1] >> 16;
      if( flags & InterJob::kAffine )
      {
        *affine++ = jobs[ctu][i];
        *affine++ = jobs[ctu][++i];
      }
      else
      {
        *( flags & InterJob::kBdof ? bdof++ : all++ ) = jobs[ctu][i];
      }
    }
  }
  return counts;
}

void GpuRecon::recordInter( const CodingStructure& cs, const std::vector<std::vector<InterJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf,
                            const Planes& planes )
{
  // Without the front end, the CTU tasks' jobs are uploaded.
  InterCounts counts;
  if( !m_useFrontEnd )
  {
    counts = uploadInterJobs( jobs, ctuBegin, ctuEnd, buf );
    if( counts.regular + counts.bdof + counts.affine == 0 )
    {
      return;
    }
  }

  const PreCalcValues& pcv = *cs.pcv;
  InterPush            push{};
  push.jobs = m_useFrontEnd ? m_frontEnd.jobs[1].addr : buf.addr;
  push.refs = m_refs.addr;
  push.dst0 = planes.addr[COMPONENT_Y];
  push.dst1 = planes.addr[COMPONENT_Cb];
  push.dst2 = planes.addr[COMPONENT_Cr];
  // DMVR's refinements for vvdec's DMVR MV cache: with the GPU front end nothing reads them (the GPU stores the refined
  // motion for TMVP itself, below), so the shader skips those writes to host memory (none: 0)
  push.dmvr = m_useFrontEnd ? 0 : m_dmvr.addr;
  // DMVR's refinements go into the picture's colocated motion (for TMVP), in memory from the GPU's picture
  // allocator (Picture::gpuMemory)
  CHECK_FATAL( !cs.m_colMiAllocation, "Vulkan: the colocated motion out of the GPU's reach" );
  push.colMotion = static_cast<HostPlane*>( cs.m_colMiAllocation )->host.addr;
  if( m_lmcsOn )
  {
    push.lmcsFwd = m_lmcs.addr;
  }
  push.defer0      = m_deferPlanes.addr[COMPONENT_Y];
  push.defer1      = m_deferPlanes.addr[COMPONENT_Cb];
  push.defer2      = m_deferPlanes.addr[COMPONENT_Cr];
  push.geo         = m_tables.geoWeights.addr;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  push.num8x8      = int32_t( pcv.num8x8CtuBlks );
  push.stride0     = planes.stride[COMPONENT_Y];
  push.stride12    = planes.stride[COMPONENT_Cb];
  push.count       = int32_t( counts.regular );
  push.picWidth    = pcv.lumaWidth;
  push.picHeight   = pcv.lumaHeight;
  push.ctuSize     = pcv.maxCUWidth;
  push.bitDepth    = cs.sps->getBitDepth();
  push.numComp     = int32_t( getNumberValidComponents( pcv.chrFormat ) );
  push.scaleX      = push.numComp > 1 ? getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) : 0;
  push.scaleY      = push.numComp > 1 ? getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) : 0;
  push.wrapSps     = cs.sps->getUseWrapAround() ? 1 : 0;
  push.wrapPps     = cs.pps->getUseWrapAround() ? 1 : 0;
  push.wrapOffset  = cs.pps->getUseWrapAround() ? cs.pps->getWrapAroundOffset() : 0;
  // subpictures treated as pictures: each CTU's rectangle (its subpicture's or the picture's), written with the
  // picture's first jobs
  const PPS& pps = *cs.pps;
  for( int i = 0; pps.getNumSubPics() > 1 && i < pps.getNumSubPics(); i++ )
  {
    push.subpicOn |= pps.getSubPic( i ).getTreatedAsPicFlag() ? 1 : 0;
  }
  if( push.subpicOn )
  {
    push.subpics = ctuBegin == 0 ? subpicRects( cs ) : m_subpics.addr;
  }
  // RPR: the picture's scaling window and its chroma's siting; whether any reference is scaled
  const Window& window = pps.getScalingWindow();
  push.curWinLeft      = window.getWindowLeftOffset() * SPS::getWinUnitX( cs.sps->getChromaFormatIdc() );
  push.curWinTop       = window.getWindowTopOffset() * SPS::getWinUnitY( cs.sps->getChromaFormatIdc() );
  push.horColloc       = cs.sps->getHorCollocatedChromaFlag() ? 1 : 0;
  push.verColloc       = cs.sps->getVerCollocatedChromaFlag() ? 1 : 0;
  bool scaled          = false;
  for( const Slice* slice: cs.picture->slices )
  {
    for( int list = 0; list < 2; list++ )
    {
      for( int refIdx = 0; refIdx < slice->getNumRefIdx( RefPicList( list ) ); refIdx++ )
      {
        scaled |= slice->getRefPic( RefPicList( list ), refIdx )->isRefScaled( &pps );
      }
    }
  }
  // The shader's tile paths (wide) load the planes' rows in aligned pieces of 16 and 8 bytes, up to 7 luma (3 chroma)
  // samples before a window and 9 (5) after it. So the rows must start 16-byte aligned and the margins be wide enough
  // (the reference pictures' planes are laid out as this one's); their chroma windows are 4:2:0's. With RPR,
  // subpictures treated as pictures or wraparound, the shader takes them only for the jobs they suit (tileJob: no
  // scaled reference; borderTiles: inside the bounds).
  push.wide = pcv.chrFormat == CHROMA_420 ? 1 : 0;
  for( int comp = 0; comp < (int) cs.picture->m_bufs[PIC_RECONSTRUCTION].bufs.size(); comp++ )
  {
    const PelStorage&  pic   = cs.picture->m_bufs[PIC_RECONSTRUCTION];
    const VkDeviceSize left  = gpuLayout( pic, ComponentID( comp ) ).origin / sizeof( Pel );
    const VkDeviceSize right = VkDeviceSize( pic.bufs[comp].stride ) - left - pic.bufs[comp].width;
    if( planes.addr[comp] % 16 != 0 || pic.bufs[comp].stride * sizeof( Pel ) % 16 != 0 || left < 16 || right < 16 )
    {
      push.wide = 0;
    }
  }

  // Each set of tiles has its own shader variant: the regular tiles, BDOF's, and the affine pairs, each also built for
  // the pictures of push.wide without wraparound, subpictures treated as pictures or scaled references (WIDE=1: their
  // layout values are constants, see inter.slang). The sets cover different tiles, so their dispatches need no
  // barriers between them.
  static const int kInterShaders[2][3] = { { kShader_inter, kShader_inter_bdof, kShader_inter_affine },
                                           { kShader_inter_wide, kShader_inter_bdof_wide, kShader_inter_affine_wide } };
  const bool constant = push.wide && !push.wrapSps && !push.subpicOn && !scaled;
  const int* shaders  = kInterShaders[constant ? 1 : 0];
  if( m_useFrontEnd )
  {
    // each set from its first job, with its count on the GPU
    push.count = -1;
    for( int set = 0; set < 3; set++ )
    {
      const FrontPass pass = FrontPass( kPassInter + set );
      push.countAt         = frontEndCount( pass );
      bind( shaders[set] );
      pushConstants( push );
      dispatchIndirect( pass );
    }
    mark( "inter" );
    return;
  }
  // one workgroup per tile (an affine pair counts as one)
  if( counts.regular > 0 )
  {
    bind( shaders[0] );
    pushConstants( push );
    dispatchJobs( counts.regular );
  }
  if( counts.bdof > 0 )
  {
    push.jobs  = buf.addr + counts.regular * sizeof( InterJob );
    push.count = int32_t( counts.bdof );
    bind( shaders[1] );
    pushConstants( push );
    dispatchJobs( counts.bdof );
  }
  if( counts.affine > 0 )
  {
    push.jobs  = buf.addr + ( counts.regular + counts.bdof ) * sizeof( InterJob );
    push.count = int32_t( counts.affine / 2 );
    bind( shaders[2] );
    pushConstants( push );
    dispatchJobs( counts.affine / 2 );
  }
  mark( "inter" );
}

// ====================================================================================================================
// Intra prediction
// ====================================================================================================================

// The push constants of intra.slang
struct IntraPush
{
  VkDeviceAddress plane0, plane1, plane2, jobs, waits, sync, mip, lmcsScale, defer0, defer1, defer2;
  int32_t         stride0, stride12, count, bitDepth, ctuLog2, scaleX, scaleY, collocated, picWidth, picHeight;
  VkDeviceAddress countAt;
};
static_assert( sizeof( IntraPush ) == 136, "IntraPush: the layout of intra.slang's Params" );

size_t GpuRecon::uploadIntraJobs( const std::vector<std::vector<IntraJob>>& jobs )
{
  // The jobs, wave by wave, with a big block cut into bands of rows of at most kBand samples, one workgroup each: a
  // wave lasts as long as its slowest workgroup, which is the one with the most samples per thread (intra.slang).
  static constexpr int kBand = 256;
  const auto bandRows = []( const IntraJob& job ) { return std::max( 1, kBand / job.width() ); };
  const auto bands    = [&]( const IntraJob& job ) { return ( job.height() + bandRows( job ) - 1 ) / bandRows( job ); };
  int    maxWave = 0;
  size_t total   = 0;
  for( const auto& ctu: jobs )
  {
    for( const IntraJob& job: ctu )
    {
      maxWave = std::max( maxWave, job.wave() );
      total  += bands( job );
    }
  }
  if( total == 0 )
  {
    return 0;
  }
  // each wave's first band
  std::vector<uint32_t> first( size_t( maxWave ) + 2, 0 );
  for( const auto& ctu: jobs )
  {
    for( const IntraJob& job: ctu )
    {
      first[job.wave() + 1] += bands( job );
    }
  }
  for( int wave = 1; wave <= maxWave + 1; wave++ )
  {
    first[wave] += first[wave - 1];
  }
  // The bands, then each band's wait: the number of bands of the waves before its own, which must be done first
  // (intra.slang).
  const size_t size = total * ( sizeof( IntraJob ) + sizeof( uint32_t ) );
  if( m_cpuJobs.intra.size < size )
  {
    m_ctx->free( m_cpuJobs.intra );
    m_cpuJobs.intra = m_ctx->alloc( size * 2, MemKind::Upload );
  }
  IntraJob*             sorted = reinterpret_cast<IntraJob*>( m_cpuJobs.intra.mapped );
  uint32_t*             waits  = reinterpret_cast<uint32_t*>( sorted + total );
  std::vector<uint32_t> next( first.begin(), first.end() - 1 );
  for( const auto& ctu: jobs )
  {
    for( const IntraJob& job: ctu )
    {
      const int rows = bandRows( job );
      for( int row = 0; row < job.height(); row += rows )
      {
        const uint32_t at   = next[job.wave()]++;
        IntraJob       band = job;
        band.setRows( row, std::min( rows, job.height() - row ) );
        sorted[at] = band;
        waits[at]  = first[job.wave()];
      }
    }
  }
  return total;
}

void GpuRecon::recordIntra( const CodingStructure& cs, const std::vector<std::vector<IntraJob>>& jobs, const Planes& planes )
{
  // Without the front end, the CTU tasks' jobs are cut into bands, sorted by wave and uploaded.
  size_t total = 0;
  if( !m_useFrontEnd )
  {
    total = uploadIntraJobs( jobs );
    if( total == 0 )
    {
      return;
    }
  }
  if( !m_intraSync.valid() )
  {
    m_intraSync = m_ctx->alloc( 256, MemKind::Device );
  }

  const PreCalcValues& pcv = *cs.pcv;
  IntraPush            push{};
  push.plane0 = planes.addr[COMPONENT_Y];
  push.plane1 = planes.addr[COMPONENT_Cb];
  push.plane2 = planes.addr[COMPONENT_Cr];
  if( m_useFrontEnd )
  {
    push.jobs    = m_frontEnd.sortedBands.addr;
    push.waits   = m_frontEnd.bandWaits.addr;
    push.count   = -1;
    push.countAt = frontEndCount( kPassIntra );
  }
  else
  {
    push.jobs  = m_cpuJobs.intra.addr;
    push.waits = m_cpuJobs.intra.addr + total * sizeof( IntraJob );
    push.count = int32_t( total );
  }
  push.sync       = m_intraSync.addr;
  push.mip        = m_tables.mip.addr;
  push.stride0    = planes.stride[COMPONENT_Y];
  push.stride12   = planes.stride[COMPONENT_Cb];
  push.bitDepth   = cs.sps->getBitDepth();
  push.ctuLog2    = getLog2( pcv.maxCUWidth );
  push.scaleX     = getComponentScaleX( COMPONENT_Cb, pcv.chrFormat );
  push.scaleY     = getComponentScaleY( COMPONENT_Cb, pcv.chrFormat );
  push.collocated = cs.sps->getCclmCollocatedChromaFlag() ? 1 : 0;
  push.picWidth   = pcv.lumaWidth;
  push.picHeight  = pcv.lumaHeight;
  if( m_lmcsOn )
  {
    push.lmcsScale = m_lmcs.addr + m_lmcsCount * 2 * sizeof( int16_t );
  }
  push.defer0 = m_deferPlanes.addr[COMPONENT_Y];
  push.defer1 = m_deferPlanes.addr[COMPONENT_Cb];
  push.defer2 = m_deferPlanes.addr[COMPONENT_Cr];
  CHECK( planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );

  // The counts of the jobs taken and done start from zero; they and what the work before wrote (the residuals,
  // inter's predictions) become visible to the jobs.
  m_ctx->api().vkCmdFillBuffer( m_cmd, m_intraSync.buffer, 0, 2 * sizeof( uint32_t ), 0 );
  memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT );
  // all the waves in one dispatch, one workgroup per band
  bind( kShader_intra );
  pushConstants( push );
  if( m_useFrontEnd )
  {
    dispatchIndirect( kPassIntra );
  }
  else
  {
    dispatchJobs( total );
  }
  mark( "intra" );
}

}   // namespace gpu
}   // namespace vvdec
