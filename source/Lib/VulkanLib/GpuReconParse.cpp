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
    \brief    GpuRecon's GPU parse of the slice data: instead of the CPU's parse (VVDEC_GPU_PARSE=2), or beside it and
              checked against it (VVDEC_GPU_PARSE=1); see GpuParse.h

    parse.slang parses each substream (a slice, tile or CTU row with WPP) with one 32-thread workgroup. A dispatch
    parses each substream up to a budget of bins and saves where it stopped in the state buffer, so the parse runs
    as many dispatches. They go to the GPU in short submissions fed by the host until every substream is parsed:
    this driver silently cuts short a submission that runs for more than about 1.2 s of GPU time.

    The parse writes each CTU's records into slots of their own; parsecompact.slang then gathers them in raster order,
    as the CPU's CTU tasks lay them out (GpuFrontEnd.h).
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

static constexpr size_t   kParseStateWords   = 1024;      // parse.slang's kStateWords
static constexpr size_t   kParseStateDone    = 1;         // parse.slang's kStDone: the substream is parsed (or given up)
static constexpr size_t   kParseStateNextCtu = 0;         // parse.slang's kStK: the substream's next CTU
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

// A CTU's SAO and ALF parameters as the parse writes them (ctuOut's words 2 to 8; gpu::captureSaoAlf's layout) into
// its CTU data, as the CPU's parse leaves them.
static void applySaoAlf( CtuData& ctu, const uint32_t words[7] )
{
  for( int comp = 0; comp < 3; comp++ )
  {
    SAOOffset&     o    = ctu.saoParam[comp];
    const uint32_t mode = words[2 * comp];
    o.reset();
    o.modeIdc = SAOMode( mode & 15 );
    if( o.modeIdc == SAO_MODE_OFF )
    {
      continue;
    }
    o.typeIdc     = int( ( mode >> 4 ) & 15 );
    o.typeAuxInfo = int( int8_t( ( mode >> 8 ) & 0xff ) );   // (-1 where the parse leaves none)
    if( o.modeIdc == SAO_MODE_NEW )
    {
      int v[4];
      for( int k = 0; k < 4; k++ )
      {
        v[k] = int( int8_t( ( words[2 * comp + 1] >> ( 8 * k ) ) & 0xff ) );
      }
      if( o.typeIdc == SAO_TYPE_START_BO )
      {
        for( int k = 0; k < 4; k++ )
        {
          o.offset[( o.typeAuxInfo + k ) % MAX_NUM_SAO_CLASSES] = v[k];
        }
      }
      else
      {
        o.offset[0] = v[0];
        o.offset[1] = v[1];
        o.offset[3] = v[2];
        o.offset[4] = v[3];
      }
    }
  }
  CtuAlfData&    a   = ctu.alfParam;
  const uint32_t alf = words[6];
  a.alfCtuEnableFlag[0]   = uint8_t( alf & 1 );
  a.alfCtuEnableFlag[1]   = uint8_t( ( alf >> 1 ) & 1 );
  a.alfCtuEnableFlag[2]   = uint8_t( ( alf >> 2 ) & 1 );
  a.alfCtbFilterIndex     = short( ( alf >> 3 ) & 31 );
  a.alfCtuAlternative[0]  = uint8_t( ( alf >> 8 ) & 15 );
  a.alfCtuAlternative[1]  = uint8_t( ( alf >> 12 ) & 15 );
  a.ccAlfFilterControl[0] = uint8_t( ( alf >> 16 ) & 15 );
  a.ccAlfFilterControl[1] = uint8_t( ( alf >> 20 ) & 15 );
}

bool GpuRecon::parsesPicture( const Picture& pic, const SPS& sps )
{
  // The GPU parses a picture instead of the CPU only when the GPU front end takes it, which needs every pass on the
  // GPU: a sequence within interSupported (which covers intra prediction and the residuals), the colocated motion
  // in the GPU's memory, and the GPU front end requested.
  if( !pic.gpuMemory || !pic.cs || !pic.cs->colMiAllocator )
  {
    return false;
  }
  const Options& options = static_cast<const Context*>( pic.gpuMemory->opaque )->options();
  return options.parseDrives && options.frontEnd && !options.frontEndCheck && interSupported( sps );
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
  // what the parse writes: the records in each CTU's slots, each CTU's bottom rows, the CTUs' readiness and the
  // counters, WPP's saved contexts, per CTU its counts and SAO and ALF parameters, each substream's state
  ensureParseBuffer( m_ctx, m_parse.cus, ctus * in.cuSlots * sizeof( CuRecord ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.tus, ctus * in.tuSlots * sizeof( TuRecord ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.bottomRows, ctus * 64 * 2 * sizeof( uint32_t ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.sync, ( 2 + ctus ) * sizeof( uint32_t ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.state, std::max<size_t>( in.numStreams, 1 ) * kParseStateWords * sizeof( uint32_t ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.wpp, std::max<size_t>( in.wppSlots, 1 ) * in.contexts * sizeof( uint32_t ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.ctuOut, ctus * kParseCtuOutWords * sizeof( uint32_t ), MemKind::HostCached );
  // the compacted records, their totals and the levels check's counts, in host memory
  m_parse.cuCap = std::max<size_t>( numCus, 1 );
  m_parse.tuCap = std::max<size_t>( numTus, 1 );
  ensureParseBuffer( m_ctx, m_parse.outCus, m_parse.cuCap * sizeof( CuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.outTus, m_parse.tuCap * sizeof( TuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.outCtus, ctus * sizeof( CtuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.totals, 4 * sizeof( uint32_t ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.check, 4 * sizeof( uint32_t ), MemKind::HostCached );
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
  const VkCommandBuffer recording = m_cmd;   // (with the check, the picture's first part, recorded meanwhile)
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

uint32_t GpuRecon::runParse( const void* push, size_t pushSize, int shader )
{
  // The substreams' state, the CTUs' readiness and the counters start from zero (host memory the GPU reaches), and so
  // do the CTUs' counts the parse writes.
  memset( m_parse.sync.mapped, 0, m_parse.sync.size );
  memset( m_parse.state.mapped, 0, m_parse.state.size );
  memset( m_parse.ctuOut.mapped, 0, m_parse.ctuOut.size );

  const uint32_t  streams      = m_parse.in.numStreams;
  const uint32_t* state        = reinterpret_cast<const uint32_t*>( m_parse.state.mapped );
  uint32_t        submissions  = 0;
  uint32_t        stalls       = 0;
  uint64_t        lastProgress = 0;
  while( true )
  {
    submitParseCommands( [&]() {
      bind( shader );
      pushConstants( push, pushSize );
      for( uint32_t i = 0; i < kParseDispatches; i++ )
      {
        dispatchGroups( streams );   // one workgroup per substream
        computeBarrier();
      }
    } );
    submissions++;
    // done when every substream is; given up (as compareParse and parsePicture count it) when none advances any more
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
  return submissions;
}

uint32_t GpuRecon::parseSubstreams( const CodingStructure& cs, const Planes* levels )
{
  const PreCalcValues& pcv    = *cs.pcv;
  const bool           chroma = isChromaEnabled( pcv.chrFormat );
  const int            csx    = chroma ? int( getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) ) : 0;
  ParsePush            push{};
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
  if( levels )
  {
    push.levels0  = levels->addr[COMPONENT_Y];
    push.levels1  = chroma ? levels->addr[COMPONENT_Cb] : 0;
    push.levels2  = chroma ? levels->addr[COMPONENT_Cr] : 0;
    push.stride0  = levels->stride[COMPONENT_Y];
    push.stride12 = chroma ? levels->stride[COMPONENT_Cb] : 0;
  }
  else
  {
    push.levels0  = m_parse.levels[0].addr;
    push.levels1  = m_parse.levels[1].addr;
    push.levels2  = m_parse.levels[2].addr;
    push.stride0  = int32_t( pcv.lumaWidth );
    push.stride12 = int32_t( pcv.lumaWidth >> csx );
  }
  push.sync       = m_parse.sync.addr;
  push.wppCtx     = m_parse.wpp.addr;
  push.numStreams = int32_t( m_parse.in.numStreams );
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
  return runParse( &push, sizeof( push ), prof ? kShader_parse_prof : kShader_parse );
}

void GpuRecon::recordParseCompaction( size_t ctus, int pass )
{
  // pass 0 works out each CTU's first CU and TU (one workgroup); pass 1 copies the records (one workgroup per CTU)
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
  compact.cuSlots = int32_t( m_parse.in.cuSlots );
  compact.tuSlots = int32_t( m_parse.in.tuSlots );
  compact.cuCap   = int32_t( m_parse.cuCap );
  compact.tuCap   = int32_t( m_parse.tuCap );
  compact.pass    = pass;
  bind( kShader_parsecompact );
  pushConstants( compact );
  if( pass == 0 )
  {
    dispatchGroups( 1 );
  }
  else
  {
    dispatchJobs( ctus );
  }
  computeBarrier();
}

uint32_t GpuRecon::parseErrors() const
{
  // sync[1]: the substreams that gave up; a substream left unparsed counts as given up too
  const uint32_t* sync   = reinterpret_cast<const uint32_t*>( m_parse.sync.mapped );
  const uint32_t* state  = reinterpret_cast<const uint32_t*>( m_parse.state.mapped );
  uint32_t        errors = sync[1];
  for( uint32_t s = 0; s < m_parse.in.numStreams; s++ )
  {
    errors += state[s * kParseStateWords + kParseStateDone] == 0 ? 1 : 0;
  }
  return errors;
}

bool GpuRecon::parsePicture( CodingStructure& cs, const PelStorage& pic, std::vector<std::vector<CuRecord>>& cus, std::vector<std::vector<TuRecord>>& tus )
{
  if( m_parse.contexts == 0 )
  {
    m_parse.contexts = checkParseContexts() ? 1 : -1;
  }
  CHECK( m_parse.contexts < 0, "Vulkan: the GPU parse's context tables differ from vvdec's" );
  CHECK( !buildParseInput( cs, m_parse.in ), "Vulkan: the slice data for the GPU parse is missing" );
  const size_t ctus = cs.pcv->sizeInCtus;
  uploadParseInput( cs, 1, 1 );   // (the records' room once their totals are known)

  // The coefficient levels go into the picture in host memory, where the CPU's parse would leave them for the
  // residual pass.
  const Planes levels = hostPlanes( pic );
  m_parse.stats.submissions += parseSubstreams( cs, &levels );
  m_parse.stats.pictures++;
  m_parse.stats.substreams += m_parse.in.numStreams;
  m_parse.stats.runs += reinterpret_cast<const uint32_t*>( m_parse.sync.mapped )[0];
  if( parseErrors() != 0 )
  {
    m_parse.stats.errors++;
    return false;
  }

  // the records in raster order: pass 0 counts them, then they are copied into room of their exact size
  submitParseCommands( [&]() { recordParseCompaction( ctus, 0 ); } );
  const uint32_t* totals = reinterpret_cast<const uint32_t*>( m_parse.totals.mapped );
  m_parse.cuCap          = std::max<size_t>( totals[0], 1 );
  m_parse.tuCap          = std::max<size_t>( totals[1], 1 );
  ensureParseBuffer( m_ctx, m_parse.outCus, m_parse.cuCap * sizeof( CuRecord ), MemKind::HostCached );
  ensureParseBuffer( m_ctx, m_parse.outTus, m_parse.tuCap * sizeof( TuRecord ), MemKind::HostCached );
  submitParseCommands( [&]() { recordParseCompaction( ctus, 1 ); } );

  // the records CTU by CTU, as the CTU tasks would have written them, and each CTU's SAO and ALF parameters into its
  // CTU data
  const CuRecord*  gpuCus  = reinterpret_cast<const CuRecord*>( m_parse.outCus.mapped );
  const TuRecord*  gpuTus  = reinterpret_cast<const TuRecord*>( m_parse.outTus.mapped );
  const CtuRecord* gpuCtus = reinterpret_cast<const CtuRecord*>( m_parse.outCtus.mapped );
  const uint32_t*  ctuOut  = reinterpret_cast<const uint32_t*>( m_parse.ctuOut.mapped );
  cus.resize( ctus );
  tus.resize( ctus );
  for( size_t ctu = 0; ctu < ctus; ctu++ )
  {
    const uint32_t firstCu = gpuCtus[ctu].w[0];
    const uint32_t numCus  = ctuOut[ctu * kParseCtuOutWords];
    const uint32_t firstTu = gpuCtus[ctu].w[3];
    const uint32_t numTus  = ctuOut[ctu * kParseCtuOutWords + 1];
    CHECK( firstCu + numCus > m_parse.cuCap || firstTu + numTus > m_parse.tuCap, "Vulkan: the GPU parse's records past their totals" );
    cus[ctu].assign( gpuCus + firstCu, gpuCus + firstCu + numCus );
    tus[ctu].assign( gpuTus + firstTu, gpuTus + firstTu + numTus );
    applySaoAlf( cs.getCtuData( int( ctu ) ), ctuOut + ctu * kParseCtuOutWords + 2 );
  }
  return true;
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
  // the levels into buffers of the parse's own (the CPU's parse has its levels in the picture), for the check below
  const bool   chroma        = isChromaEnabled( pcv.chrFormat );
  const int    csx           = chroma ? int( getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) ) : 0;
  const int    csy           = chroma ? int( getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) ) : 0;
  const size_t chromaSamples = std::max<size_t>( size_t( pcv.lumaWidth >> csx ) * ( pcv.lumaHeight >> csy ), 1 );
  ensureParseBuffer( m_ctx, m_parse.levels[0], size_t( pcv.lumaWidth ) * pcv.lumaHeight * sizeof( int16_t ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.levels[1], chromaSamples * sizeof( int16_t ), MemKind::Device );
  ensureParseBuffer( m_ctx, m_parse.levels[2], chromaSamples * sizeof( int16_t ), MemKind::Device );
  m_parse.stats.submissions += parseSubstreams( cs, nullptr );

  // the records gathered in raster order, into m_cmd (the picture's first part)
  recordParseCompaction( ctus, 0 );
  recordParseCompaction( ctus, 1 );

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
  m_parse.stats.substreams += m_parse.in.numStreams;
  m_parse.stats.runs += reinterpret_cast<const uint32_t*>( m_parse.sync.mapped )[0];   // (one per substream and dispatch)
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
                     reinterpret_cast<const uint32_t*>( m_parse.check.mapped ), parseErrors(), m_parse.stats );
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
