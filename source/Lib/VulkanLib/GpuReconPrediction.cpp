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

// Room for size bytes in buf, an upload buffer of the CTU tasks' jobs. When it is smaller, it is replaced with one of
// twice the size, so that the next pictures, which may have a few more jobs, seldom need a new one.
static void ensureJobUpload( Context* ctx, Buffer& buf, size_t size )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( size * 2, MemKind::Upload );
  }
}

size_t GpuRecon::uploadTrafoJobs( const std::vector<std::vector<TrafoJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf )
{
  size_t total = 0;
  for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
  {
    total += jobs[ctu].size();
  }
  if( total == 0 )
  {
    return 0;
  }
  ensureJobUpload( m_ctx, buf, total * sizeof( TrafoJob ) );
  TrafoJob* all = reinterpret_cast<TrafoJob*>( buf.mapped );
  for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
  {
    memcpy( all, jobs[ctu].data(), jobs[ctu].size() * sizeof( TrafoJob ) );
    all += jobs[ctu].size();
  }
  return total;
}

void GpuRecon::recordTrafo( const CodingStructure& cs, const std::vector<std::vector<TrafoJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf,
                            const Planes& levels, const Planes& planes )
{
  CHECK( planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );
  // levels is the picture in host memory when the CPU parsed it. A picture that the GPU parsed has none there: its
  // levels are in parsedLevels.
  CHECK( levels.addr[COMPONENT_Y] && ( levels.stride[COMPONENT_Y] != planes.stride[COMPONENT_Y] || levels.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cb] ),
         "the levels laid out differently" );
  // Without the front end, the CTU tasks' jobs are uploaded.
  size_t total = 0;
  if( !m_useFrontEnd )
  {
    total = uploadTrafoJobs( jobs, ctuBegin, ctuEnd, buf );
    if( total == 0 )
    {
      return;
    }
  }

  TrafoPush push{};
  push.jobs         = m_useFrontEnd ? m_frontEnd.jobs[kTrafoJobs].addr : buf.addr;
  push.plane0       = planes.addr[COMPONENT_Y];
  push.plane1       = planes.addr[COMPONENT_Cb];
  push.plane2       = planes.addr[COMPONENT_Cr];
  push.levels0      = levels.addr[COMPONENT_Y];
  push.levels1      = levels.addr[COMPONENT_Cb];
  push.levels2      = levels.addr[COMPONENT_Cr];
  push.parsedLevels = m_frontEnd.recordLevels;   // the GPU parse's levels; its jobs say where in them
  push.matrices     = m_tables.transform.addr;
  push.lfnst        = m_tables.lfnst.addr;
  push.scaling      = m_scalingOn ? m_scaling.addr : 0;
  push.stride0      = planes.stride[COMPONENT_Y];
  push.stride12     = planes.stride[COMPONENT_Cb];
  push.count        = m_useFrontEnd ? -1 : int32_t( total );
  push.countAt      = frontEndCount( kPassTrafo );
  push.bitDepth     = cs.sps->getBitDepth();

  // One workgroup per job
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

// The sets of inter tiles, each run by a shader variant of its own, in the order of the job lists and of the front end's
// passes (kPassInter + set)
enum InterSet
{
  kRegularTiles,
  kBdofTiles,
  kAffineTiles,   // two jobs per tile
  kInterSets
};

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
  ensureJobUpload( m_ctx, buf, total * sizeof( InterJob ) );
  // The BDOF tiles go after the others, then the affine ones (two jobs each), so that the inter_bdof and inter_affine
  // variants get dispatches of their own: their code would slow the other tiles down. The front end lays its jobs out
  // the same way.
  for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
  {
    for( size_t i = 0; i < jobs[ctu].size(); i++ )
    {
      const uint32_t flags = jobs[ctu][i].flags();
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
      const uint32_t flags = jobs[ctu][i].flags();
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

// Whether a reference picture of the picture's slices is scaled (RPR)
static bool anyReferenceScaled( const CodingStructure& cs )
{
  const PPS& pps = *cs.pps;
  for( const Slice* slice: cs.picture->slices )
  {
    for( int list = 0; list < 2; list++ )
    {
      for( int refIdx = 0; refIdx < slice->getNumRefIdx( RefPicList( list ) ); refIdx++ )
      {
        if( slice->getRefPic( RefPicList( list ), refIdx )->isRefScaled( &pps ) )
        {
          return true;
        }
      }
    }
  }
  return false;
}

// Whether the picture pic's planes, at addr on the GPU, suit the inter shader's tile paths (push.wide). The tile paths
// load the planes' rows in aligned pieces of 16 and 8 bytes, up to 7 luma (3 chroma) samples before a window and 9 (5)
// after it. So the rows must start 16-byte aligned and the margins be at least 16 samples wide (the reference pictures'
// planes are laid out as this one's). The tile paths' own chroma code is 4:2:0's: 4:2:2 and 4:4:4 tiles use
// chromaTile's, and monochrome pictures have no chroma. With RPR, subpictures treated as pictures or wraparound, the
// shader takes the tile paths only for the jobs they suit (tileJob: no scaled reference; borderTiles: inside the
// bounds).
static bool planesSuitTiles( const PelStorage& pic, const VkDeviceAddress addr[MAX_NUM_COMPONENT] )
{
  static constexpr VkDeviceSize kRowAlignment = 16;   // bytes
  static constexpr VkDeviceSize kMinMargin    = 16;   // samples
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    const VkDeviceSize left  = gpuLayout( pic, ComponentID( comp ) ).origin / sizeof( Pel );
    const VkDeviceSize right = VkDeviceSize( pic.bufs[comp].stride ) - left - pic.bufs[comp].width;
    if( addr[comp] % kRowAlignment != 0 || pic.bufs[comp].stride * sizeof( Pel ) % kRowAlignment != 0 || left < kMinMargin || right < kMinMargin )
    {
      return false;
    }
  }
  return true;
}

// inter.slang's push constants for the picture, apart from its jobs, their count and the subpictures' rectangles (which
// recordInter sets): the references, the planes it writes, where DMVR's refinements go, the picture's geometry and
// chroma format, and its wraparound, subpicture and RPR settings
void GpuRecon::interPush( const CodingStructure& cs, const Planes& planes, InterPush& push ) const
{
  const PreCalcValues& pcv = *cs.pcv;
  const PPS&           pps = *cs.pps;
  push      = InterPush{};
  push.refs = m_refs.addr;
  push.dst0 = planes.addr[COMPONENT_Y];
  push.dst1 = planes.addr[COMPONENT_Cb];
  push.dst2 = planes.addr[COMPONENT_Cr];
  // DMVR's refinements for vvdec's DMVR MV cache. With the GPU front end nothing reads them, because the GPU stores
  // the refined motion for TMVP itself (colMotion, below). The address is 0 then, and the shader skips those writes to
  // host memory.
  push.dmvr = m_useFrontEnd ? 0 : m_dmvr.addr;
  // DMVR's refinements also go into the picture's colocated motion for TMVP, which is in memory from the GPU's picture
  // allocator (Picture::gpuMemory).
  CHECK_FATAL( !cs.m_colMiAllocation, "Vulkan: the colocated motion out of the GPU's reach" );
  push.colMotion = static_cast<HostPlane*>( cs.m_colMiAllocation )->host.addr;
  if( m_lmcsOn )
  {
    push.lmcsFwd = lmcsForward();
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
  push.picWidth    = pcv.lumaWidth;
  push.picHeight   = pcv.lumaHeight;
  push.ctuSize     = pcv.maxCUWidth;
  push.bitDepth    = cs.sps->getBitDepth();
  push.numComp     = int32_t( getNumberValidComponents( pcv.chrFormat ) );
  push.scaleX      = chromaShiftX( pcv.chrFormat );
  push.scaleY      = chromaShiftY( pcv.chrFormat );
  push.wide        = planesSuitTiles( cs.picture->m_bufs[PIC_RECONSTRUCTION], planes.addr ) ? 1 : 0;
  push.wrapSps     = cs.sps->getUseWrapAround() ? 1 : 0;
  push.wrapPps     = pps.getUseWrapAround() ? 1 : 0;
  push.wrapOffset  = pps.getUseWrapAround() ? pps.getWrapAroundOffset() : 0;
  push.subpicOn    = subpicturesTreatedAsPictures( pps ) ? 1 : 0;
  // RPR: the picture's scaling window and its chroma's siting
  const Window& window = pps.getScalingWindow();
  push.curWinLeft      = window.getWindowLeftOffset() * SPS::getWinUnitX( cs.sps->getChromaFormatIdc() );
  push.curWinTop       = window.getWindowTopOffset() * SPS::getWinUnitY( cs.sps->getChromaFormatIdc() );
  push.horColloc       = cs.sps->getHorCollocatedChromaFlag() ? 1 : 0;
  push.verColloc       = cs.sps->getVerCollocatedChromaFlag() ? 1 : 0;
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
  InterPush push;
  interPush( cs, planes, push );
  if( push.subpicOn )
  {
    // Each CTU's rectangle (its subpicture's or the picture's), written along with the picture's first jobs
    push.subpics = ctuBegin == 0 ? subpicRects( cs ) : m_subpics.addr;
  }

  // Each set of tiles has its own shader variant, each also built for the pictures of push.wide without wraparound,
  // subpictures treated as pictures or scaled references (WIDE=1: their layout values are constants, see inter.slang).
  // The sets cover different tiles, so their dispatches need no barriers between them.
  static const int kInterShaders[2][kInterSets] = { { kShader_inter, kShader_inter_bdof, kShader_inter_affine },
                                                    { kShader_inter_wide, kShader_inter_bdof_wide, kShader_inter_affine_wide } };
  const bool constant = push.wide && cs.pcv->chrFormat == CHROMA_420 && !push.wrapSps && !push.subpicOn && !anyReferenceScaled( cs );
  const int* shaders  = kInterShaders[constant ? 1 : 0];
  if( m_useFrontEnd )
  {
    // Each set from its first job, with its count on the GPU
    push.jobs  = m_frontEnd.jobs[kInterJobs].addr;
    push.count = -1;
    for( int set = 0; set < kInterSets; set++ )
    {
      const FrontPass pass = FrontPass( kPassInter + set );
      push.countAt         = frontEndCount( pass );
      bind( shaders[set] );
      pushConstants( push );
      dispatchIndirect( pass );
    }
  }
  else
  {
    // One workgroup per tile; the sets lie one after another in buf.
    const size_t setJobs[kInterSets] = { counts.regular, counts.bdof, counts.affine };
    push.jobs                        = buf.addr;
    for( int set = 0; set < kInterSets; set++ )
    {
      const size_t tiles = set == kAffineTiles ? setJobs[set] / 2 : setJobs[set];
      if( tiles > 0 )
      {
        push.count = int32_t( tiles );
        bind( shaders[set] );
        pushConstants( push );
        dispatchJobs( tiles );
      }
      push.jobs += setJobs[set] * sizeof( InterJob );
    }
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
// The words of intra.slang's sync buffer (the counts of the jobs taken and done come first)
static constexpr size_t kIntraSyncWords = 64;

size_t GpuRecon::uploadIntraJobs( const std::vector<std::vector<IntraJob>>& jobs )
{
  // The jobs go up wave by wave, a big block cut into bands of rows (IntraJob::bands), one workgroup per band.
  int    maxWave = 0;
  size_t total   = 0;
  for( const auto& ctu: jobs )
  {
    for( const IntraJob& job: ctu )
    {
      maxWave = std::max( maxWave, job.wave() );
      total  += job.bands();
    }
  }
  if( total == 0 )
  {
    return 0;
  }
  // Each wave's first band: first[w] is the number of bands of the waves before w.
  std::vector<uint32_t> first( size_t( maxWave ) + 2, 0 );
  for( const auto& ctu: jobs )
  {
    for( const IntraJob& job: ctu )
    {
      first[job.wave() + 1] += job.bands();
    }
  }
  for( int wave = 1; wave <= maxWave + 1; wave++ )
  {
    first[wave] += first[wave - 1];
  }
  // The bands, then each band's wait: the number of bands of the waves before its own, which must be done first
  // (intra.slang).
  ensureJobUpload( m_ctx, m_cpuJobs.intra, total * ( sizeof( IntraJob ) + sizeof( uint32_t ) ) );
  IntraJob*             sorted = reinterpret_cast<IntraJob*>( m_cpuJobs.intra.mapped );
  uint32_t*             waits  = reinterpret_cast<uint32_t*>( sorted + total );
  std::vector<uint32_t> next( first.begin(), first.end() - 1 );
  for( const auto& ctu: jobs )
  {
    for( const IntraJob& job: ctu )
    {
      const int rows = job.bandRows();
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
    m_intraSync = m_ctx->alloc( kIntraSyncWords * sizeof( uint32_t ), MemKind::Device );
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
    push.lmcsScale = lmcsChromaScales();
  }
  push.defer0 = m_deferPlanes.addr[COMPONENT_Y];
  push.defer1 = m_deferPlanes.addr[COMPONENT_Cb];
  push.defer2 = m_deferPlanes.addr[COMPONENT_Cr];
  CHECK( planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );

  // The counts of the jobs taken and done start from zero. They, and what the work before wrote (the residuals,
  // inter's predictions), become visible to the jobs.
  m_ctx->api().vkCmdFillBuffer( m_cmd, m_intraSync.buffer, 0, 2 * sizeof( uint32_t ), 0 );
  memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT );
  // All the waves in one dispatch, one workgroup per band
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
