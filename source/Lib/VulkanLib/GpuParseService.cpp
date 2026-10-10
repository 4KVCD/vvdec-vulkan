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

/** \file     GpuParseService.cpp
    \brief    The GPU's parse of a picture's slice data (ParseSlot), and of the pictures ahead of their
              reconstruction (ParseService); see GpuParseService.h
*/

#include "GpuParseService.h"

#include "GpuPictureMemory.h"
#include "vvdec_gpu/GpuShaderIds.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace vvdec
{
namespace gpu
{

// parse.slang's state of a substream between dispatches
static constexpr size_t   kParseStateWords   = 1024;      // kStateWords
static constexpr size_t   kParseStateNextCtu = 0;         // kStK: the substream's next CTU
static constexpr size_t   kParseStateDone    = 1;         // kStDone: the substream is parsed (or given up)
// The bins per substream and dispatch (a CU counts as 32): a few ms alone. Longer submissions on the parse queue hold
// up the pictures' front ends on the async compute queues (VideoQ 1080p, 2^16 bins: a picture's GPU work took 16 ms
// from its submission instead of 7, 115 fps; 2^13: 7.8 ms, 149 fps; 2^14 as fast, 4K 46 -> 54 fps; 2^12: slower
// again, from the many submissions).
static constexpr int32_t  kParseBinBudget    = 1 << 14;
static constexpr uint32_t kParseMaxStalls    = 4;         // submissions in a row without progress before giving up
static constexpr uint32_t kParseLocalBits    = 12;        // a meta word: the CTU above the record's place in the CTU
static constexpr size_t   kParseCounters     = 8;         // the staged CUs and TUs, the room of the 5 job lists, a spare
static constexpr size_t   kParseTotals       = 16;        // the CUs and TUs, then the counters (compaction pass 0)
static constexpr size_t   kGroupsPerRow      = 32768;     // the shaders take group.y as rows of 32768 workgroups

// The service: each picture's dispatches in one submission (one, so that a picture is handed over as soon as it is
// parsed), the pictures parsed at a time
static constexpr uint32_t kStepsPerSubmission = 1;
static constexpr size_t   kMaxParsing         = 16;

// The push constants of parse.slang
struct ParsePush
{
  VkDeviceAddress bits, streams, ctuList, ctuInfo, slices, pic, scanTab, cus, tus, ctuOut, bottomRows, levels0, levels1, levels2, sync, wppCtx;
  int32_t         numStreams, stride0, stride12;
  VkDeviceAddress prof;
  VkDeviceAddress state;
  int32_t         binBudget;
  VkDeviceAddress cuMeta, tuMeta, counters;
  int32_t         cuCap, tuCap;
};
static_assert( sizeof( ParsePush ) == 200, "ParsePush: the layout of parse.slang's Params" );

// The push constants of parsecompact.slang (pass 0: the CTU records and totals; 1, 2, 3: the scatter of the CU, TU
// and CTU records)
struct ParseCompactPush
{
  VkDeviceAddress cusIn, tusIn, cuMeta, tuMeta, ctuOut, ctuInfo, sync, counters, cus, tus, ctus, ctusOut, totals;
  int32_t         numCtus, numCus, numTus, pass;
};
static_assert( sizeof( ParseCompactPush ) == 120, "ParseCompactPush: the layout of parsecompact.slang's Params" );

// Makes buf hold at least size bytes of kind, in steps of 64 KB.
static void ensureBuffer( Context* ctx, Buffer& buf, size_t size, MemKind kind )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( ( size + 65535 ) & ~size_t( 65535 ), kind );
  }
}

// Records into cmd the dispatch of shader with the push constants push over groups workgroups (in rows of
// kGroupsPerRow).
template<class Push>
static void dispatch( Context* ctx, VkCommandBuffer cmd, int shader, const Push& push, size_t groups )
{
  if( groups == 0 )
  {
    return;
  }
  const Api& vk = ctx->api();
  vk.vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline( shader ) );
  vk.vkCmdPushConstants( cmd, ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, uint32_t( sizeof( Push ) ), &push );
  vk.vkCmdDispatch( cmd, uint32_t( std::min( groups, kGroupsPerRow ) ), uint32_t( ( groups + kGroupsPerRow - 1 ) / kGroupsPerRow ), 1 );
}

// Records into cmd a barrier: the writes of the stages src visible to the accesses dstAccess of the stages dst
static void barrier( Context* ctx, VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkAccessFlags dstAccess )
{
  VkMemoryBarrier memory{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memory.dstAccessMask = dstAccess;
  ctx->api().vkCmdPipelineBarrier( cmd, src, dst, 0, 1, &memory, 0, nullptr, 0, nullptr );
}

static void computeBarrier( Context* ctx, VkCommandBuffer cmd )
{
  barrier( ctx, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT );
}

static void hostReadBarrier( Context* ctx, VkCommandBuffer cmd )
{
  barrier( ctx, cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT );
}

// what the submission before wrote and the fills before it (recordClear), visible to the compute shaders
static void startBarrier( Context* ctx, VkCommandBuffer cmd )
{
  VkMemoryBarrier memory{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  memory.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  memory.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  ctx->api().vkCmdPipelineBarrier( cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                                   &memory, 0, nullptr, 0, nullptr );
}

// ====================================================================================================================
// ParseSlot: one picture's parse
// ====================================================================================================================

ParseSlot::~ParseSlot()
{
  for( Buffer* buf: { &m_bits, &m_streams, &m_ctuList, &m_ctuInfo, &m_slices, &m_pic, &m_scans, &m_cus, &m_tus, &m_cuMeta, &m_tuMeta, &m_bottomRows,
                      &m_wpp, &m_sync, &m_state, &m_ctuOut, &m_counters, &m_ctus, &m_totals } )
  {
    m_ctx->free( *buf );
  }
}

bool ParseSlot::prepare( const CodingStructure& cs, const LevelPlanes& levels, size_t minCus, size_t minTus )
{
  if( !buildParseInput( cs, m_in ) )
  {
    return false;
  }
  CHECK( m_in.cuSlots >= ( 1u << kParseLocalBits ) || m_in.tuSlots >= ( 1u << kParseLocalBits ), "Vulkan: more records in a CTU than the parse's meta words hold" );
  m_numCtus = cs.pcv->sizeInCtus;
  m_levels  = levels;

  // the inputs: the slice data (which the parse reads 128 bytes at a time: 256 zero bytes follow it), the
  // substreams, the CTUs in decoding order and their slices and tiles, the slices' parameters and contexts, the
  // picture's parameters, the scans
  const auto upload = [&]( Buffer& buf, const void* data, size_t size ) {
    ensureBuffer( m_ctx, buf, std::max<size_t>( size, 16 ), MemKind::Upload );
    memcpy( buf.mapped, data, size );
  };
  ensureBuffer( m_ctx, m_bits, m_in.bits.size() + 256, MemKind::Upload );
  memcpy( m_bits.mapped, m_in.bits.data(), m_in.bits.size() );
  memset( m_bits.mapped + m_in.bits.size(), 0, 256 );
  upload( m_streams, m_in.streams.data(), m_in.streams.size() * sizeof( uint32_t ) );
  upload( m_ctuList, m_in.ctuList.data(), m_in.ctuList.size() * sizeof( uint32_t ) );
  upload( m_ctuInfo, m_in.ctuInfo.data(), m_in.ctuInfo.size() * sizeof( uint32_t ) );
  upload( m_slices, m_in.slices.data(), m_in.slices.size() * sizeof( uint32_t ) );
  upload( m_pic, m_in.pic.data(), m_in.pic.size() * sizeof( int32_t ) );
  if( !m_scans.mapped )
  {
    ensureBuffer( m_ctx, m_scans, kParseScanTab * sizeof( uint32_t ), MemKind::Upload );
    writeParseScans( reinterpret_cast<uint32_t*>( m_scans.mapped ) );
  }

  // what the parse writes
  ensureBuffer( m_ctx, m_bottomRows, m_numCtus * 64 * 2 * sizeof( uint32_t ), MemKind::Device );
  ensureBuffer( m_ctx, m_wpp, std::max<size_t>( m_in.wppSlots, 1 ) * m_in.contexts * sizeof( uint32_t ), MemKind::Device );
  ensureBuffer( m_ctx, m_sync, ( 2 + m_numCtus ) * sizeof( uint32_t ), MemKind::HostCached );
  ensureBuffer( m_ctx, m_state, std::max<size_t>( m_in.numStreams, 1 ) * kParseStateWords * sizeof( uint32_t ), MemKind::HostCached );
  ensureBuffer( m_ctx, m_ctuOut, m_numCtus * kParseCtuOutWords * sizeof( uint32_t ), MemKind::HostCached );
  ensureBuffer( m_ctx, m_counters, kParseCounters * sizeof( uint32_t ), MemKind::Device );   // (an atomic a CU: fast in GPU memory)
  ensureBuffer( m_ctx, m_ctus, m_numCtus * sizeof( CtuRecord ), MemKind::HostCached );
  ensureBuffer( m_ctx, m_totals, kParseTotals * sizeof( uint32_t ), MemKind::HostCached );
  makeRoom( minCus, minTus );
  clear();
  return true;
}

void ParseSlot::makeRoom( size_t cus, size_t tus )
{
  ensureBuffer( m_ctx, m_cus, std::max<size_t>( cus, 1 ) * sizeof( CuRecord ), MemKind::Device );
  ensureBuffer( m_ctx, m_cuMeta, std::max<size_t>( cus, 1 ) * sizeof( uint32_t ), MemKind::Device );
  ensureBuffer( m_ctx, m_tus, std::max<size_t>( tus, 1 ) * sizeof( TuRecord ), MemKind::Device );
  ensureBuffer( m_ctx, m_tuMeta, std::max<size_t>( tus, 1 ) * sizeof( uint32_t ), MemKind::Device );
  // (all the room the buffers have)
  m_cuCap = std::min<size_t>( m_cus.size / sizeof( CuRecord ), m_cuMeta.size / sizeof( uint32_t ) );
  m_tuCap = std::min<size_t>( m_tus.size / sizeof( TuRecord ), m_tuMeta.size / sizeof( uint32_t ) );
}

void ParseSlot::clear()
{
  // (host memory the GPU reaches: the slot is idle)
  memset( m_sync.mapped, 0, m_sync.size );
  memset( m_state.mapped, 0, m_state.size );
  memset( m_ctuOut.mapped, 0, m_ctuOut.size );
  memset( m_totals.mapped, 0, m_totals.size );
  m_fresh        = true;   // (the counters, in GPU memory: recordClear)
  m_numCus       = 0;
  m_numTus       = 0;
  m_caps         = FrontCapacities();
  m_lastProgress = 0;
  m_stalls       = 0;
  m_parseValue   = 0;
  failed         = false;
}

void ParseSlot::recordClear( VkCommandBuffer cmd )
{
  if( m_fresh )
  {
    m_ctx->api().vkCmdFillBuffer( cmd, m_counters.buffer, 0, VK_WHOLE_SIZE, 0 );
    m_fresh = false;
  }
}

void ParseSlot::recordStep( VkCommandBuffer cmd ) const
{
  ParsePush push{};
  push.bits       = m_bits.addr;
  push.streams    = m_streams.addr;
  push.ctuList    = m_ctuList.addr;
  push.ctuInfo    = m_ctuInfo.addr;
  push.slices     = m_slices.addr;
  push.pic        = m_pic.addr;
  push.scanTab    = m_scans.addr;
  push.cus        = m_cus.addr;
  push.tus        = m_tus.addr;
  push.ctuOut     = m_ctuOut.addr;
  push.bottomRows = m_bottomRows.addr;
  push.levels0    = m_levels.addr[0];
  push.levels1    = m_levels.addr[1];
  push.levels2    = m_levels.addr[2];
  push.sync       = m_sync.addr;
  push.wppCtx     = m_wpp.addr;
  push.numStreams = int32_t( m_in.numStreams );
  push.stride0    = m_levels.stride0;
  push.stride12   = m_levels.stride12;
  push.prof       = m_prof;
  push.state      = m_state.addr;
  push.binBudget  = kParseBinBudget;
  push.cuMeta     = m_cuMeta.addr;
  push.tuMeta     = m_tuMeta.addr;
  push.counters   = m_counters.addr;
  push.cuCap      = int32_t( std::min<size_t>( m_cuCap, INT32_MAX ) );
  push.tuCap      = int32_t( std::min<size_t>( m_tuCap, INT32_MAX ) );
  dispatch( m_ctx, cmd, m_prof ? kShader_parse_prof : kShader_parse, push, m_in.numStreams );   // a workgroup per substream
}

void ParseSlot::recordCompaction( VkCommandBuffer cmd ) const
{
  ParseCompactPush push{};
  push.ctuOut   = m_ctuOut.addr;
  push.ctuInfo  = m_ctuInfo.addr;
  push.sync     = m_sync.addr;
  push.counters = m_counters.addr;
  push.ctus     = m_ctus.addr;
  push.totals   = m_totals.addr;
  push.numCtus  = int32_t( m_numCtus );
  push.pass     = 0;
  dispatch( m_ctx, cmd, kShader_parsecompact, push, 1 );
}

ParseSlot::Progress ParseSlot::progress()
{
  // the substreams parsed, and how far they all are
  const uint32_t* state    = reinterpret_cast<const uint32_t*>( m_state.mapped );
  uint32_t        done     = 0;
  uint64_t        progress = 0;
  for( uint32_t s = 0; s < m_in.numStreams; s++ )
  {
    done += state[s * kParseStateWords + kParseStateDone] != 0 ? 1 : 0;
    progress += state[s * kParseStateWords + kParseStateNextCtu];
  }
  if( done < m_in.numStreams )
  {
    m_stalls       = progress == m_lastProgress ? m_stalls + 1 : 0;
    m_lastProgress = progress;
    return m_stalls >= kParseMaxStalls ? Progress::Failed : Progress::Parsing;
  }
  if( errors() != 0 )
  {
    return Progress::Failed;
  }
  // all parsed, and the compaction's first pass done: the staged records within their room, as many as the CTUs
  // counted
  const uint32_t* sums = totals();
  if( sums[2] > m_cuCap || sums[3] > m_tuCap )
  {
    return Progress::Overflow;
  }
  if( sums[0] != sums[2] || sums[1] != sums[3] )
  {
    return Progress::Failed;   // (the CTUs' counts and the records staged disagree: a bug)
  }
  m_numCus = sums[0];
  m_numTus = sums[1];
  // the job lists' room, as gpu::frontCapacities works it out (with its margins)
  m_caps.trafo = size_t( sums[4] ) + 64;
  m_caps.inter = size_t( sums[5] ) + 64;
  m_caps.intra = size_t( sums[6] ) + 64;
  m_caps.bands = size_t( sums[7] ) + 64;
  m_caps.edges = size_t( sums[8] ) + 64;
  return Progress::Done;
}

void ParseSlot::restart()
{
  // room for the records the parse counted, and a quarter more
  const uint32_t* sums = totals();
  makeRoom( size_t( sums[2] ) * 5 / 4 + 64, size_t( sums[3] ) * 5 / 4 + 64 );
  clear();
}

const uint32_t* ParseSlot::saoAlf( int ctu ) const
{
  return ctuOut() + size_t( ctu ) * kParseCtuOutWords + 2;
}

void ParseSlot::recordScatter( VkCommandBuffer cmd, VkDeviceAddress cus, VkDeviceAddress tus, VkDeviceAddress ctus ) const
{
  ParseCompactPush push{};
  push.cusIn   = m_cus.addr;
  push.tusIn   = m_tus.addr;
  push.cuMeta  = m_cuMeta.addr;
  push.tuMeta  = m_tuMeta.addr;
  push.ctus    = m_ctus.addr;
  push.cus     = cus;
  push.tus     = tus;
  push.ctusOut = ctus;
  push.numCtus = int32_t( m_numCtus );
  push.numCus  = int32_t( m_numCus );
  push.numTus  = int32_t( m_numTus );
  // a workgroup copies 32 CU records, 64 TU records or 256 CTU records
  push.pass = 1;
  dispatch( m_ctx, cmd, kShader_parsecompact, push, ( m_numCus + 31 ) / 32 );
  push.pass = 2;
  dispatch( m_ctx, cmd, kShader_parsecompact, push, ( m_numTus + 63 ) / 64 );
  push.pass = 3;
  dispatch( m_ctx, cmd, kShader_parsecompact, push, ( m_numCtus + 255 ) / 256 );
}

uint32_t ParseSlot::runs() const
{
  return m_sync.mapped ? reinterpret_cast<const uint32_t*>( m_sync.mapped )[0] : 0;   // (none before the first prepare)
}

uint32_t ParseSlot::errors() const
{
  // sync[1]: the substreams that gave up; a substream left unparsed counts as given up too
  if( !m_sync.mapped || !m_state.mapped )
  {
    return 1;   // (never prepared)
  }
  const uint32_t* state  = reinterpret_cast<const uint32_t*>( m_state.mapped );
  uint32_t        errors = reinterpret_cast<const uint32_t*>( m_sync.mapped )[1];
  for( uint32_t s = 0; s < m_in.numStreams; s++ )
  {
    errors += state[s * kParseStateWords + kParseStateDone] == 0 ? 1 : 0;
  }
  return errors;
}

// A CTU's SAO and ALF parameters as the parse writes them (gpu::captureSaoAlf's layout) into its CTU data, as the
// CPU's parse leaves them
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
        // band offset: the 4 bands from the first one on
        for( int k = 0; k < 4; k++ )
        {
          o.offset[( o.typeAuxInfo + k ) % MAX_NUM_SAO_CLASSES] = v[k];
        }
      }
      else
      {
        // edge offset: the classes 0, 1, 3 and 4 (2 has none)
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

void applyParsedSaoAlf( CodingStructure& cs, const ParseSlot& parsed )
{
  for( int ctu = 0; ctu < int( cs.pcv->sizeInCtus ); ctu++ )
  {
    applySaoAlf( cs.getCtuData( ctu ), parsed.saoAlf( ctu ) );
  }
}

// ====================================================================================================================
// ParseService: the pictures parsed ahead of their reconstruction
// ====================================================================================================================

ParseService::ParseService( Context* ctx, std::function<void()> wakeUp ) : m_ctx( ctx ), m_wakeUp( std::move( wakeUp ) )
{
  CHECK( !ctx->parseQueue(), "Vulkan: the GPU parse ahead of the reconstruction needs an async compute queue" );
  m_pool = m_ctx->createCommandPool( m_ctx->asyncFamily() );
  m_ctx->allocateCommandBuffers( m_pool, 1, &m_cmd );
  m_fence = m_ctx->createFence();
  if( m_ctx->profiling() && m_ctx->shaderClock() )
  {
    m_profBuf = m_ctx->alloc( 16 * sizeof( uint32_t ), MemKind::HostCached );
    memset( m_profBuf.mapped, 0, 16 * sizeof( uint32_t ) );
    m_prof = m_profBuf.addr;
  }
  m_thread = std::thread( &ParseService::run, this );
}

ParseService::~ParseService()
{
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    m_stop = true;
  }
  m_wake.notify_all();
  m_thread.join();
  // the front ends that read the slots given back done (for a few seconds at most: a GPU that stopped never gets there)
  uint64_t last = 0;
  for( const auto& released: m_released )
  {
    last = std::max( last, released.second );
  }
  const auto giveUp = std::chrono::steady_clock::now() + std::chrono::seconds( 5 );
  while( m_ctx->firstPartsDone() < last && std::chrono::steady_clock::now() < giveUp )
  {
    std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
  }
  m_slots.clear();
  m_ctx->free( m_profBuf );
  m_ctx->destroyFence( m_fence );
  m_ctx->destroyCommandPool( m_pool );
}

void ParseService::enqueue( Picture* pic )
{
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    if( !m_broken )
    {
      m_waiting.push_back( pic );
      m_wake.notify_one();
      return;
    }
    pic->gpuParseSlot = nullptr;   // (fails at once)
    pic->gpuParsed.unlock_nothrow();
  }
  m_wakeUp();
}

void ParseService::release( ParseSlot* slot, uint64_t firstPartValue )
{
  std::lock_guard<std::mutex> lock( m_mutex );
  slot->picture = nullptr;
  m_released.emplace_back( slot, firstPartValue );
}

ParseStats ParseService::stats() const
{
  std::lock_guard<std::mutex> lock( m_mutex );
  ParseStats stats = m_stats;
  if( m_profBuf.mapped )
  {
    const uint32_t* prof = reinterpret_cast<const uint32_t*>( m_profBuf.mapped );
    for( int i = 0; i < 16; i++ )
    {
      stats.prof[i] = prof[i];
    }
  }
  return stats;
}

void ParseService::run()
{
  try
  {
    while( true )
    {
      // the pictures enqueued since, oldest first, as many as may be parsed at a time
      std::vector<Picture*> starting;
      {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_wake.wait( lock, [&]() { return m_stop || !m_parsing.empty() || !m_waiting.empty(); } );
        if( m_stop )
        {
          return;
        }
        std::sort( m_waiting.begin(), m_waiting.end(),
                   []( const Picture* a, const Picture* b ) { return a->decodingOrderNumber < b->decodingOrderNumber; } );
        const size_t count = std::min( m_waiting.size(), kMaxParsing - std::min( kMaxParsing, m_parsing.size() ) );
        starting.assign( m_waiting.begin(), m_waiting.begin() + count );
        m_waiting.erase( m_waiting.begin(), m_waiting.begin() + count );
      }
      for( Picture* pic: starting )
      {
        startPicture( pic );
      }
      if( !m_parsing.empty() )
      {
        submitSteps();
        finishPictures();
      }
    }
  }
  catch( ... )
  {
    // the GPU failed (or the service did): the pictures waiting for their parse fail, and so do those enqueued later
    std::lock_guard<std::mutex> lock( m_mutex );
    for( ParseSlot* slot: m_parsing )
    {
      slot->failed = true;
      slot->picture->gpuParseSlot = slot;
      slot->picture->gpuParsed.unlock_nothrow();
    }
    m_parsing.clear();
    m_broken = true;
    for( Picture* pic: m_waiting )
    {
      pic->gpuParseSlot = nullptr;
      pic->gpuParsed.unlock_nothrow();
    }
    m_waiting.clear();
    m_wakeUp();
  }
}

ParseSlot* ParseService::takeSlot()
{
  // the slots given back whose pictures' first parts are done become free
  const uint64_t done = m_ctx->firstPartsDone();
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    for( size_t i = 0; i < m_released.size(); )
    {
      if( m_released[i].second <= done )
      {
        m_free.push_back( m_released[i].first );
        m_released[i] = m_released.back();
        m_released.pop_back();
      }
      else
      {
        i++;
      }
    }
  }
  if( !m_free.empty() )
  {
    ParseSlot* slot = m_free.back();
    m_free.pop_back();
    return slot;
  }
  m_slots.push_back( std::make_unique<ParseSlot>( m_ctx ) );
  return m_slots.back().get();
}

void ParseService::startPicture( Picture* pic )
{
  ParseSlot* slot = takeSlot();
  slot->picture   = pic;
  slot->setProfile( m_prof );
  // the levels go into the picture in host memory, where the residual pass reads them
  VkDeviceAddress addr[MAX_NUM_COMPONENT]   = {};
  int             stride[MAX_NUM_COMPONENT] = {};
  const bool      host                      = hostPictureAddresses( pic->m_bufs[PIC_RECONSTRUCTION], addr, stride );
  LevelPlanes     levels;
  levels.addr[0]  = addr[0];
  levels.addr[1]  = addr[1];
  levels.addr[2]  = addr[2];
  levels.stride0  = stride[0];
  levels.stride12 = stride[1];
  // room for a quarter more records than any picture so far had, and at least 32 a CTU
  const size_t ctus = pic->cs->pcv->sizeInCtus;
  if( !host || !slot->prepare( *pic->cs, levels, std::max( m_maxCus * 5 / 4, ctus * 32 ), std::max( m_maxTus * 5 / 4, ctus * 32 ) ) )
  {
    handOver( slot, true );
    return;
  }
  m_parsing.push_back( slot );
  std::lock_guard<std::mutex> lock( m_mutex );
  m_stats.pictures++;
  m_stats.substreams += slot->input().numStreams;
}

void ParseService::submitSteps()
{
  const Api& vk = m_ctx->api();
  vk.vkResetCommandPool( m_ctx->device(), m_pool, 0 );
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( m_cmd, &begin );
  // the counters of the pictures starting cleared; what the submission before wrote visible
  for( ParseSlot* slot: m_parsing )
  {
    slot->recordClear( m_cmd );
  }
  startBarrier( m_ctx, m_cmd );
  // the pictures' dispatches side by side, a barrier after each round; then their compaction's first passes
  for( uint32_t step = 0; step < kStepsPerSubmission; step++ )
  {
    for( const ParseSlot* slot: m_parsing )
    {
      slot->recordStep( m_cmd );
    }
    computeBarrier( m_ctx, m_cmd );
  }
  for( const ParseSlot* slot: m_parsing )
  {
    slot->recordCompaction( m_cmd );
  }
  hostReadBarrier( m_ctx, m_cmd );
  vk.vkEndCommandBuffer( m_cmd );
  vk.vkResetFences( m_ctx->device(), 1, &m_fence );
  const uint64_t value = m_ctx->submitParse( m_cmd, m_fence );
  const VkResult result = vk.vkWaitForFences( m_ctx->device(), 1, &m_fence, VK_TRUE, UINT64_MAX );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: waiting for the GPU's parse failed (" << result << ")" );
  }
  for( ParseSlot* slot: m_parsing )
  {
    slot->setParseValue( value );
  }
  std::lock_guard<std::mutex> lock( m_mutex );
  m_stats.submissions++;
  m_stats.parsingSum += int64_t( m_parsing.size() );
}

void ParseService::finishPictures()
{
  for( size_t i = 0; i < m_parsing.size(); )
  {
    ParseSlot*                slot     = m_parsing[i];
    const ParseSlot::Progress progress = slot->progress();
    if( progress == ParseSlot::Progress::Parsing )
    {
      i++;
      continue;
    }
    if( progress == ParseSlot::Progress::Overflow )
    {
      // more records than the room: parsed again with room for them
      slot->restart();
      std::lock_guard<std::mutex> lock( m_mutex );
      m_stats.restarts++;
      i++;
      continue;
    }
    m_parsing.erase( m_parsing.begin() + i );
    handOver( slot, progress != ParseSlot::Progress::Done );
  }
}

void ParseService::handOver( ParseSlot* slot, bool failed )
{
  slot->failed = failed;
  if( !failed )
  {
    // the most records a picture had: the room the next pictures' slots start with
    m_maxCus = std::max( m_maxCus, slot->numCus() );
    m_maxTus = std::max( m_maxTus, slot->numTus() );
  }
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    m_stats.runs += slot->runs();
    m_stats.errors += failed ? 1 : 0;
    m_stats.maxCus = int64_t( m_maxCus );
    m_stats.maxTus = int64_t( m_maxTus );
  }
  Picture* pic      = slot->picture;
  pic->gpuParseSlot = slot;
  pic->gpuParsed.unlock_nothrow();
  m_wakeUp();   // (the picture's tasks may wait in a sleeping thread pool)
}

}   // namespace gpu
}   // namespace vvdec
