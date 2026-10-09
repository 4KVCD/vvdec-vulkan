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
    \brief    GpuRecon's GPU parse of the slice data (VVDEC_GPU_PARSE=1, checked against the CPU's parse; see GpuParse.h)

    parse.slang parses each substream (a slice, tile or CTU row with WPP) with one 32-thread workgroup. A dispatch
    parses each substream up to a budget of bins and saves where it stopped in the state buffer, so the parse runs
    as many dispatches. They go to the GPU in short submissions fed by the host until every substream is parsed:
    this driver silently cuts short a submission that runs for more than about 1.2 s of GPU time.
*/

#include "GpuRecon.h"

#include "vvdec_gpu/GpuShaderIds.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"

#include <cstring>

namespace vvdec
{
namespace gpu
{

static constexpr size_t   kParseStateWords   = 1024;   // parse.slang's kStateWords
static constexpr size_t   kParseStateDone    = 1;      // parse.slang's kStDone: the substream is parsed (or given up)
static constexpr size_t   kParseStateNextCtu = 0;      // parse.slang's kStK: the substream's next CTU
static constexpr int32_t  kParseBinBudget    = 1 << 16;   // bins per substream and dispatch (a CU counts as 32): ~10 ms
static constexpr uint32_t kParseDispatches   = 8;         // dispatches per submission
static constexpr uint32_t kParseMaxStalls    = 4;         // submissions in a row without progress before giving up

// The push constants of parse.slang
struct ParsePush
{
  VkDeviceAddress bits, streams, ctuList, ctuInfo, slices, pic, scanTab, cus, tus, ctuOut, bottomRows, levels0, levels1, levels2, sync, wppCtx;
  int32_t         numStreams, stride0, stride12;
  VkDeviceAddress prof;
  VkDeviceAddress state;
  int32_t         binBudget;
};
static_assert( sizeof( ParsePush ) == 168, "ParsePush: the layout of parse.slang's Params" );

// The push constants of parsecompact.slang
struct ParseCompactPush
{
  VkDeviceAddress cusIn, tusIn, ctuOut, ctuInfo, cus, tus, ctus, totals;
  int32_t         numCtus, cuSlots, tuSlots, pass, cuCap, tuCap;
};
static_assert( sizeof( ParseCompactPush ) == 88, "ParseCompactPush: the layout of parsecompact.slang's Params" );

// The push constants of parsecompact.slang's CHECK variant
struct ParseCheckPush
{
  VkDeviceAddress tus, gpu0, gpu1, gpu2, cpu0, cpu1, cpu2, counts;
  int32_t         numTus, gpuStride0, gpuStride12, cpuStride0, cpuStride12, width, height, csx, csy;
};
static_assert( sizeof( ParseCheckPush ) == 104, "ParseCheckPush: the layout of parsecompact.slang's CHECK Params" );

// Makes buf hold at least size bytes of kind, in steps of 64 KB.
static void ensureParseBuffer( Context* ctx, Buffer& buf, size_t size, MemKind kind )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( ( size + 65535 ) & ~size_t( 65535 ), kind );
  }
}

void GpuRecon::uploadParseInput( const CodingStructure& cs, size_t numCus, size_t numTus )
{
  const ParseInput&    in   = m_parse.in;
  const PreCalcValues& pcv  = *cs.pcv;
  const size_t         ctus = pcv.sizeInCtus;
  const auto upload = [&]( Buffer& buf, const void* data, size_t size ) {
    ensureParseBuffer( m_ctx, buf, std::max<size_t>( size, 16 ), MemKind::Upload );
    memcpy( buf.mapped, data, size );
  };
  // (the parse reads the slice data 128 bytes at a time: 256 zero bytes follow it)
  ensureParseBuffer( m_ctx, m_parse.bits, in.bits.size() + 256, MemKind::Upload );
  memcpy( m_parse.bits.mapped, in.bits.data(), in.bits.size() );
  memset( m_parse.bits.mapped + in.bits.size(), 0, 256 );
  upload( m_parse.streams, in.streams.data(), in.streams.size() * sizeof( uint32_t ) );
  upload( m_parse.ctuList, in.ctuList.data(), in.ctuList.size() * sizeof( uint32_t ) );
  upload( m_parse.ctuInfo, in.ctuInfo.data(), in.ctuInfo.size() * sizeof( uint32_t ) );
  upload( m_parse.slices, in.slices.data(), in.slices.size() * sizeof( uint32_t ) );
  upload( m_parse.pic, in.pic.data(), in.pic.size() * sizeof( int32_t ) );
  if( !m_parse.scans.mapped )
  {
    ensureParseBuffer( m_ctx, m_parse.scans, kParseScanTab * sizeof( uint32_t ), MemKind::Upload );
    writeParseScans( reinterpret_cast<uint32_t*>( m_parse.scans.mapped ) );
  }
  // what the parse writes: the records in each CTU's slots, each CTU's bottom rows, the levels, the CTUs' readiness
  // and the counters, WPP's saved contexts, per CTU its counts and SAO and ALF parameters, each substream's state
  const bool chroma = isChromaEnabled( pcv.chrFormat );
  const int  csx    = chroma ? int( getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) ) : 0;
  const int  csy    = chroma ? int( getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) ) : 0;
  const size_t chromaSamples = std::max<size_t>( size_t( pcv.lumaWidth >> csx ) * ( pcv.lumaHeight >> csy ), 1 );
  ensureParseBuffer( m_ctx, m_parse.cus, ctus * in.cuSlots * sizeof( CuRecord ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.tus, ctus * in.tuSlots * sizeof( TuRecord ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.bottomRows, ctus * 64 * 2 * sizeof( uint32_t ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.levels[0], size_t( pcv.lumaWidth ) * pcv.lumaHeight * sizeof( int16_t ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.levels[1], chromaSamples * sizeof( int16_t ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.levels[2], chromaSamples * sizeof( int16_t ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.sync, ( 2 + ctus ) * sizeof( uint32_t ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.state, std::max<size_t>( in.numStreams, 1 ) * kParseStateWords * sizeof( uint32_t ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.wpp, std::max<size_t>( in.wppSlots, 1 ) * in.contexts * sizeof( uint32_t ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.ctuOut, ctus * kParseCtuOutWords * sizeof( uint32_t ), MemKind::HostCached );
  // the compacted records, the totals and the levels check's counts, in host memory for compareParse
  m_parse.cuCap = std::max<size_t>( numCus, 1 );
  m_parse.tuCap = std::max<size_t>( numTus, 1 );
  ensureParseBuffer( m_ctx, m_parse.outCus, m_parse.cuCap * sizeof( CuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.outTus, m_parse.tuCap * sizeof( TuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.outCtus, ctus * sizeof( CtuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.totals, 4 * sizeof( uint32_t ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.check, 4 * sizeof( uint32_t ), MemKind::HostCached );
}

uint32_t GpuRecon::runParse( const void* push, size_t pushSize, int shader )
{
  if( !m_parse.pool )
  {
    m_parse.pool = m_ctx->createCommandPool( m_ctx->queueFamily() );
    m_ctx->allocateCommandBuffers( m_parse.pool, 1, &m_parse.cmd );
    m_parse.fence = m_ctx->createFence();
  }
  // The substreams' state, the CTUs' readiness and the counters start from zero (host memory the GPU reaches), and so
  // do the CTUs' counts the parse writes.
  memset( m_parse.sync.mapped, 0, m_parse.sync.size );
  memset( m_parse.state.mapped, 0, m_parse.state.size );
  memset( m_parse.ctuOut.mapped, 0, m_parse.ctuOut.size );

  const Api&            vk        = m_ctx->api();
  const VkCommandBuffer recording = m_cmd;   // (the picture's first part, recorded meanwhile)
  const uint32_t        streams   = m_parse.in.numStreams;
  const uint32_t*       state     = reinterpret_cast<const uint32_t*>( m_parse.state.mapped );
  uint32_t              submissions = 0, stalls = 0;
  uint64_t              lastProgress = 0;
  m_cmd                              = m_parse.cmd;
  while( true )
  {
    vk.vkResetCommandPool( m_ctx->device(), m_parse.pool, 0 );
    beginOneTime( m_cmd );
    fullBarrier();   // the inputs and what the submission before wrote, visible
    bind( shader );
    pushConstants( push, pushSize );
    for( uint32_t i = 0; i < kParseDispatches; i++ )
    {
      dispatchGroups( streams );   // one workgroup per substream
      computeBarrier();
    }
    hostReadBarrier();
    vk.vkEndCommandBuffer( m_cmd );
    vk.vkResetFences( m_ctx->device(), 1, &m_parse.fence );
    m_ctx->submit( m_cmd, m_parse.fence );
    const VkResult result = vk.vkWaitForFences( m_ctx->device(), 1, &m_parse.fence, VK_TRUE, UINT64_MAX );
    if( result != VK_SUCCESS )
    {
      THROW_FATAL( "Vulkan: waiting for the GPU's parse failed (" << result << ")" );
    }
    submissions++;
    // done when every substream is; given up (as compareParse counts it) when none advances any more
    uint32_t done     = 0;
    uint64_t progress = 0;
    for( uint32_t s = 0; s < streams; s++ )
    {
      done     += state[s * kParseStateWords + kParseStateDone] != 0 ? 1 : 0;
      progress += state[s * kParseStateWords + kParseStateNextCtu];
    }
    if( done == streams )
    {
      break;
    }
    stalls       = progress == lastProgress ? stalls + 1 : 0;
    lastProgress = progress;
    if( stalls >= kParseMaxStalls )
    {
      break;
    }
  }
  m_cmd = recording;
  return submissions;
}

void GpuRecon::recordParse( const CodingStructure& cs, const PictureJobs& jobs, const Planes& cpuLevels )
{
  m_parse.recorded = false;
  if( m_parse.contexts == 0 )
  {
    m_parse.contexts = checkParseContexts() ? 1 : -1;
  }
  if( m_parse.contexts < 0 || !buildParseInput( cs, m_parse.in ) )
  {
    return;
  }
  const PreCalcValues& pcv    = *cs.pcv;
  const ParseInput&    in     = m_parse.in;
  const size_t         ctus   = pcv.sizeInCtus;
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
  uploadParseInput( cs, numCus, numTus );

  const bool chroma = isChromaEnabled( pcv.chrFormat );
  const int  csx    = chroma ? int( getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) ) : 0;
  const int  csy    = chroma ? int( getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) ) : 0;
  ParsePush  push{};
  push.bits       = m_parse.bits.addr;
  push.streams    = m_parse.streams.addr;
  push.ctuList    = m_parse.ctuList.addr;
  push.ctuInfo    = m_parse.ctuInfo.addr;
  push.slices     = m_parse.slices.addr;
  push.pic        = m_parse.pic.addr;
  push.scanTab    = m_parse.scans.addr;
  push.cus        = m_parse.cus.addr;
  push.tus        = m_parse.tus.addr;
  push.ctuOut     = m_parse.ctuOut.addr;
  push.bottomRows = m_parse.bottomRows.addr;
  push.levels0    = m_parse.levels[0].addr;
  push.levels1    = m_parse.levels[1].addr;
  push.levels2    = m_parse.levels[2].addr;
  push.sync       = m_parse.sync.addr;
  push.wppCtx     = m_parse.wpp.addr;
  push.numStreams = int32_t( in.numStreams );
  push.stride0    = int32_t( pcv.lumaWidth );
  push.stride12   = int32_t( pcv.lumaWidth >> csx );
  push.state      = m_parse.state.addr;
  push.binBudget  = kParseBinBudget;
  // VVDEC_GPU_PROFILE with the shader clock: parse_prof's cycle counts, added up over the pictures
  const bool prof = m_ctx->profiling() && m_ctx->shaderClock();
  if( prof )
  {
    ensureParseBuffer( m_ctx, m_parse.prof, 16 * sizeof( uint32_t ), MemKind::HostCached );
    if( !m_parse.profRead )
    {
      memset( m_parse.prof.mapped, 0, 16 * sizeof( uint32_t ) );
    }
    push.prof = m_parse.prof.addr;
  }
  m_parse.stats.submissions += runParse( &push, sizeof( push ), prof ? kShader_parse_prof : kShader_parse );

  // The records gathered from the CTUs' slots in raster order (pass 0 counts, pass 1 copies, one workgroup per CTU)
  ParseCompactPush compact{};
  compact.cusIn   = m_parse.cus.addr;
  compact.tusIn   = m_parse.tus.addr;
  compact.ctuOut  = m_parse.ctuOut.addr;
  compact.ctuInfo = m_parse.ctuInfo.addr;
  compact.cus     = m_parse.outCus.addr;
  compact.tus     = m_parse.outTus.addr;
  compact.ctus    = m_parse.outCtus.addr;
  compact.totals  = m_parse.totals.addr;
  compact.numCtus = int32_t( ctus );
  compact.cuSlots = int32_t( in.cuSlots );
  compact.tuSlots = int32_t( in.tuSlots );
  compact.cuCap   = int32_t( m_parse.cuCap );
  compact.tuCap   = int32_t( m_parse.tuCap );
  bind( kShader_parsecompact );
  compact.pass = 0;
  pushConstants( compact );
  dispatchGroups( 1 );
  computeBarrier();
  compact.pass = 1;
  pushConstants( compact );
  dispatchJobs( ctus );
  computeBarrier();

  // The levels against the CPU's parse's (in the picture in host memory, which the residual pass reads), one
  // workgroup per TU
  m_ctx->api().vkCmdFillBuffer( m_cmd, m_parse.check.buffer, 0, VK_WHOLE_SIZE, 0 );
  transferToComputeBarrier();
  ParseCheckPush check{};
  check.tus         = m_parse.outTus.addr;
  check.gpu0        = m_parse.levels[0].addr;
  check.gpu1        = m_parse.levels[1].addr;
  check.gpu2        = m_parse.levels[2].addr;
  check.cpu0        = cpuLevels.addr[COMPONENT_Y];
  check.cpu1        = chroma ? cpuLevels.addr[COMPONENT_Cb] : 0;
  check.cpu2        = chroma ? cpuLevels.addr[COMPONENT_Cr] : 0;
  check.counts      = m_parse.check.addr;
  check.numTus      = int32_t( numTus );
  check.gpuStride0  = int32_t( pcv.lumaWidth );
  check.gpuStride12 = int32_t( pcv.lumaWidth >> csx );
  check.cpuStride0  = cpuLevels.stride[COMPONENT_Y];
  check.cpuStride12 = chroma ? cpuLevels.stride[COMPONENT_Cb] : 0;
  check.width       = int32_t( pcv.lumaWidth );
  check.height      = int32_t( pcv.lumaHeight );
  check.csx         = csx;
  check.csy         = csy;
  if( numTus > 0 )
  {
    bind( kShader_parsecheck );
    pushConstants( check );
    dispatchJobs( numTus );
    computeBarrier();
  }
  mark( "parse checks" );
  m_parse.recorded = true;
}

void GpuRecon::compareParse( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, const std::vector<std::vector<TuRecord>>& tus )
{
  if( !m_parse.recorded )
  {
    return;
  }
  m_parse.recorded = false;
  // sync: [0] the substreams' runs (one per substream and dispatch), [1] the substreams that gave up. A substream
  // left unparsed counts as given up too.
  const uint32_t* sync       = reinterpret_cast<const uint32_t*>( m_parse.sync.mapped );
  const uint32_t* state      = reinterpret_cast<const uint32_t*>( m_parse.state.mapped );
  uint32_t        unfinished = 0;
  for( uint32_t s = 0; s < m_parse.in.numStreams; s++ )
  {
    unfinished += state[s * kParseStateWords + kParseStateDone] == 0 ? 1 : 0;
  }
  m_parse.stats.substreams += m_parse.in.numStreams;
  m_parse.stats.runs += sync[0];
  if( m_parse.prof.mapped )
  {
    // the profile's sums, the buffer cleared for the next picture
    uint32_t* p = reinterpret_cast<uint32_t*>( m_parse.prof.mapped );
    for( int i = 0; i < 16; i++ )
    {
      m_parse.stats.prof[i] += p[i];
      p[i] = 0;
    }
    m_parse.profRead = true;
  }
  gpu::compareParse( cs, reinterpret_cast<const uint32_t*>( m_parse.outCus.mapped ), reinterpret_cast<const uint32_t*>( m_parse.outTus.mapped ),
                     reinterpret_cast<const uint32_t*>( m_parse.outCtus.mapped ), reinterpret_cast<const uint32_t*>( m_parse.ctuOut.mapped ),
                     reinterpret_cast<const uint32_t*>( m_parse.totals.mapped ), cus, tus, cs.m_gpuSaoAlf,
                     reinterpret_cast<const uint32_t*>( m_parse.check.mapped ), sync[1] + unfinished, m_parse.stats );
}

void GpuRecon::ParseBuffers::release( Context* ctx )
{
  for( Buffer* buf: { &bits, &streams, &ctuList, &ctuInfo, &slices, &pic, &scans, &cus, &tus, &bottomRows, &levels[0], &levels[1], &levels[2], &sync,
                      &wpp, &ctuOut, &state, &outCus, &outTus, &outCtus, &totals, &check, &prof } )
  {
    ctx->free( *buf );
  }
  ctx->destroyFence( fence );
  ctx->destroyCommandPool( pool );
}

}   // namespace gpu
}   // namespace vvdec
