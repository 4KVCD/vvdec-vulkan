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

/** \file     GpuReconParse.cpp
    \brief    GpuRecon's side of the GPU parse: whether the GPU parses a picture instead of the CPU (VVDEC_GPU_PARSE=2:
              the ParseService parses it, the front end takes its records), and the check of the GPU's parse against
              the CPU's (VVDEC_GPU_PARSE=1); see GpuParseService.h

    The check parses each picture's slice data on the GPU too, with a ParseSlot of its own, in short submissions on
    the compute queue until every substream is parsed (this driver silently cuts short a submission that runs for
    more than about 1.2 s of GPU time). The picture's first part then copies the records to raster order in host
    memory and compares the levels with the CPU's parse's; compareParse compares the records, each CTU's SAO and ALF
    parameters and the room of the front end's job lists with the CPU's.
*/

#include "GpuRecon.h"

#include "GpuParseService.h"
#include "vvdec_gpu/GpuShaderIds.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"

#include <cstring>

namespace vvdec
{
namespace gpu
{

static constexpr uint32_t kParseDispatches = 8;   // the check's dispatches per submission

// The push constants of parsecompact.slang's CHECK variant
struct ParseCheckPush
{
  VkDeviceAddress tus, gpu, cpu0, cpu1, cpu2, counts;
  int32_t         numTus, cpuStride0, cpuStride12, width, height, csx, csy;
};
static_assert( sizeof( ParseCheckPush ) == 80, "ParseCheckPush: the layout of parsecompact.slang's CHECK Params" );

// Makes buf hold at least size bytes of kind, in steps of 64 KB.
static void ensureParseBuffer( Context* ctx, Buffer& buf, size_t size, MemKind kind )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( ( size + 65535 ) & ~size_t( 65535 ), kind );
  }
}

bool GpuRecon::parsesPicture( const Picture& pic, const SPS& sps )
{
  // (the colocated motion in the GPU's memory too: the coding structure takes the picture's allocator for it)
  return pic.cs && pic.cs->colMiAllocator && parsesPictures( pic.gpuMemory, sps );
}

bool GpuRecon::parsesPictures( const UserAllocator* gpuMemory, const SPS& sps )
{
  // The GPU parses a picture instead of the CPU only when the GPU front end takes it, which needs every pass on the
  // GPU: a sequence within interSupported (which covers intra prediction and the residuals), the picture in the
  // GPU's memory, and the GPU front end requested. The parse runs on a queue of its own.
  if( !gpuMemory )
  {
    return false;
  }
  const Context* ctx     = static_cast<const Context*>( gpuMemory->opaque );
  const Options& options = ctx->options();
  return options.parseDrives && options.frontEnd && !options.frontEndCheck && ctx->parseQueue() && interSupported( sps );
}

void GpuRecon::submitParseCommands( const std::function<void()>& record )
{
  if( !m_parse.pool )
  {
    m_parse.pool = m_ctx->createCommandPool( m_ctx->queueFamily() );
    m_ctx->allocateCommandBuffers( m_parse.pool, 1, &m_parse.cmd );
    m_parse.fence = m_ctx->createFence();
  }
  const Api&            vk        = m_ctx->api();
  const VkCommandBuffer recording = m_cmd;   // (the picture's first part, recorded meanwhile)
  m_cmd                           = m_parse.cmd;
  vk.vkResetCommandPool( m_ctx->device(), m_parse.pool, 0 );
  beginOneTime( m_cmd );
  fullBarrier();   // the inputs and what the submission before wrote, visible
  record();
  hostReadBarrier();
  vk.vkEndCommandBuffer( m_cmd );
  vk.vkResetFences( m_ctx->device(), 1, &m_parse.fence );
  m_ctx->submit( m_cmd, m_parse.fence );
  const VkResult result = vk.vkWaitForFences( m_ctx->device(), 1, &m_parse.fence, VK_TRUE, UINT64_MAX );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: waiting for the GPU's parse failed (" << result << ")" );
  }
  m_cmd = recording;
}

uint32_t GpuRecon::runParse( ParseSlot& slot )
{
  uint32_t submissions = 0;
  while( true )
  {
    submitParseCommands( [&]() {
      slot.recordClear( m_cmd );
      fullBarrier();
      for( uint32_t i = 0; i < kParseDispatches; i++ )
      {
        slot.recordStep( m_cmd );
        computeBarrier();
      }
      slot.recordCompaction( m_cmd );
    } );
    submissions++;
    const ParseSlot::Progress progress = slot.progress();
    if( progress == ParseSlot::Progress::Overflow )
    {
      // more records than the CPU's parse made (the comparison will tell), or more levels than the samples (the
      // chunks' unused ends): room for what the parse counted
      Buffer levels = slot.takeLevels();
      ensureParseBuffer( m_ctx, levels, ( slot.levelsCounted() * 5 / 4 + 4096 ) * sizeof( int16_t ), MemKind::Device );
      StagedRecords staged = slot.takeStaged();
      ParseSlot::growStaged( m_ctx, staged, slot.cusCounted() * 5 / 4 + 64, slot.tusCounted() * 5 / 4 + 64 );
      slot.restart( staged, levels );
    }
    else if( progress != ParseSlot::Progress::Parsing )
    {
      slot.failed = progress == ParseSlot::Progress::Failed;
      return submissions;
    }
  }
}

void GpuRecon::recordParse( const CodingStructure& cs, const PictureJobs& jobs, const Planes& cpuLevels )
{
  m_parse.recorded = false;
  if( m_parse.contexts == 0 )
  {
    m_parse.contexts = checkParseContexts() ? 1 : -1;
  }
  if( m_parse.contexts < 0 )
  {
    return;
  }
  const PreCalcValues& pcv    = *cs.pcv;
  size_t               numCus = 0;
  size_t               numTus = 0;
  for( const auto& ctu: *jobs.cus )
  {
    numCus += ctu.size();
  }
  for( const auto& ctu: *jobs.tus )
  {
    numTus += ctu.size();
  }
  if( !m_parse.slot )
  {
    m_parse.slot = std::make_unique<ParseSlot>( m_ctx );
  }
  ParseSlot& slot = *m_parse.slot;
  // the levels into the slot's own buffer (the CPU's parse has its levels in the picture), for the check below: the
  // one of the picture before, with room for a level a sample of each plane
  const bool   chroma        = isChromaEnabled( pcv.chrFormat );
  const int    csx           = chroma ? int( getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) ) : 0;
  const int    csy           = chroma ? int( getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) ) : 0;
  const size_t chromaSamples = chroma ? size_t( pcv.lumaWidth >> csx ) * ( pcv.lumaHeight >> csy ) : 0;
  Buffer       levels        = slot.takeLevels();
  ensureParseBuffer( m_ctx, levels, ( size_t( pcv.lumaWidth ) * pcv.lumaHeight + 2 * chromaSamples ) * sizeof( int16_t ), MemKind::Device );
  // VVDEC_GPU_PROFILE with the shader clock: parse_prof's cycle counts, added up over the pictures (parse_prof reads
  // threads' values as parse does: with VK_EXT_shader_subgroup_ballot)
  if( m_ctx->profiling() && m_ctx->shaderClock() && m_ctx->subgroupBallot() )
  {
    ensureParseBuffer( m_ctx, m_parse.prof, 16 * sizeof( uint32_t ), MemKind::HostCached );
    if( !m_parse.profRead )
    {
      memset( m_parse.prof.mapped, 0, 16 * sizeof( uint32_t ) );
    }
    slot.setProfile( m_parse.prof.addr );
  }
  // room for as many records as the CPU's parse made
  StagedRecords staged = slot.takeStaged();
  ParseSlot::growStaged( m_ctx, staged, numCus, numTus );
  if( !slot.prepare( cs, staged, levels ) )
  {
    return;
  }
  m_parse.stats.submissions += runParse( slot );
  m_parse.recorded = true;
  if( slot.failed )
  {
    return;   // (compareParse counts it)
  }

  // the records in raster order, in host memory: copied in m_cmd (the picture's first part)
  ensureParseBuffer( m_ctx, m_parse.outCus, std::max<size_t>( slot.numCus(), 1 ) * sizeof( CuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.outTus, std::max<size_t>( slot.numTus(), 1 ) * sizeof( TuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.outCtus, pcv.sizeInCtus * sizeof( CtuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.check, 4 * sizeof( uint32_t ), MemKind::HostCached );
  slot.recordScatter( m_cmd, m_parse.outCus.addr, m_parse.outTus.addr, m_parse.outCtus.addr );
  computeBarrier();

  // The levels against the CPU's parse's (in the picture in host memory, which the residual pass reads), one
  // workgroup per TU
  m_ctx->api().vkCmdFillBuffer( m_cmd, m_parse.check.buffer, 0, VK_WHOLE_SIZE, 0 );
  transferToComputeBarrier();
  ParseCheckPush check{};
  check.tus         = m_parse.outTus.addr;
  check.gpu         = slot.levels();
  check.cpu0        = cpuLevels.addr[COMPONENT_Y];
  check.cpu1        = chroma ? cpuLevels.addr[COMPONENT_Cb] : 0;
  check.cpu2        = chroma ? cpuLevels.addr[COMPONENT_Cr] : 0;
  check.counts      = m_parse.check.addr;
  check.numTus      = int32_t( slot.numTus() );
  check.cpuStride0  = cpuLevels.stride[COMPONENT_Y];
  check.cpuStride12 = chroma ? cpuLevels.stride[COMPONENT_Cb] : 0;
  check.width       = int32_t( pcv.lumaWidth );
  check.height      = int32_t( pcv.lumaHeight );
  check.csx         = csx;
  check.csy         = csy;
  if( slot.numTus() > 0 )
  {
    bind( kShader_parsecheck );
    pushConstants( check );
    dispatchJobs( slot.numTus() );
    computeBarrier();
  }
  mark( "parse checks" );
}

void GpuRecon::compareParse( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, const std::vector<std::vector<TuRecord>>& tus )
{
  if( !m_parse.recorded )
  {
    return;
  }
  m_parse.recorded       = false;
  const ParseSlot& slot  = *m_parse.slot;
  ParseStats&      stats = m_parse.stats;
  stats.substreams += slot.input().numStreams;
  stats.runs += slot.runs();
  if( m_parse.prof.mapped )
  {
    // the profile's sums, the buffer cleared for the next picture
    uint32_t* p = reinterpret_cast<uint32_t*>( m_parse.prof.mapped );
    for( int i = 0; i < 16; i++ )
    {
      stats.prof[i] += p[i];
      p[i] = 0;
    }
    m_parse.profRead = true;
  }
  if( slot.failed )
  {
    // given up: every record counts as differing
    stats.pictures++;
    stats.errors++;
    for( size_t ctu = 0; ctu < cus.size(); ctu++ )
    {
      stats.cusCompared += int64_t( cus[ctu].size() );
      stats.cusDiffer += int64_t( cus[ctu].size() );
      stats.tusCompared += int64_t( tus[ctu].size() );
      stats.tusDiffer += int64_t( tus[ctu].size() );
    }
    if( stats.reported < 20 )
    {
      stats.reported++;
      msg( WARNING, "GPU parse: POC %d: %u substreams gave up\n", cs.picture->poc, slot.errors() );
    }
    return;
  }
  gpu::compareParse( cs, reinterpret_cast<const uint32_t*>( m_parse.outCus.mapped ), reinterpret_cast<const uint32_t*>( m_parse.outTus.mapped ),
                     reinterpret_cast<const uint32_t*>( m_parse.outCtus.mapped ), slot.ctuOut(), cus, tus, cs.m_gpuSaoAlf,
                     reinterpret_cast<const uint32_t*>( m_parse.check.mapped ), slot.errors(), stats );
  // The room of the front end's job lists, as the parse added it up, against gpu::frontCapacities of the CPU's records
  const FrontCapacities  cpu = frontCapacities( cs, cus, tus );
  const FrontCapacities& gpu = slot.capacities();
  stats.capsCompared++;
  if( cpu.trafo != gpu.trafo || cpu.inter != gpu.inter || cpu.intra != gpu.intra || cpu.bands != gpu.bands || cpu.edges != gpu.edges )
  {
    stats.capsDiffer++;
    if( stats.reported < 20 )
    {
      stats.reported++;
      msg( WARNING, "GPU parse: POC %d job lists' room: CPU %zu %zu %zu %zu %zu, GPU %zu %zu %zu %zu %zu\n", cs.picture->poc, cpu.trafo, cpu.inter,
           cpu.intra, cpu.bands, cpu.edges, gpu.trafo, gpu.inter, gpu.intra, gpu.bands, gpu.edges );
    }
  }
}

void GpuRecon::ParseCheck::release( Context* ctx )
{
  slot.reset();
  for( Buffer* buf: { &outCus, &outTus, &outCtus, &check, &prof } )
  {
    ctx->free( *buf );
  }
  ctx->destroyFence( fence );
  ctx->destroyCommandPool( pool );
}

}   // namespace gpu
}   // namespace vvdec
