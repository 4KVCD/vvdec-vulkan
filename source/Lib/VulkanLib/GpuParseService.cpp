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

#include "vvdec_gpu/GpuShaderIds.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"

#include <algorithm>
#include <cstring>

namespace vvdec
{
namespace gpu
{

// parse.slang's state of a substream between dispatches
static constexpr size_t   kParseStateWords   = 1024;      // kStateWords
static constexpr size_t   kParseStateNextCtu = 0;         // kStK: the substream's next CTU
static constexpr size_t   kParseStateDone    = 1;         // kStDone: the substream is parsed (or given up)
// The bins per substream and dispatch (a CU counts as 32). A step's workgroups hold their compute units' shared memory
// (two of them fill a unit's) for as long as the step runs, and the steps run one after another, a submission each:
// shorter steps cost more submissions, but decode faster, as if the reconstruction's passes on the other queues got
// in between more often (15 W, 2^12 against 2^14: Beekeeper slower 4K +3.8%, Top Gun slower 1080p +4.5%, Conan
// 1080p +39%, Top Gun 1080p +21%, the film grain stream fg_tg4k +0.6%; 2^13 about as fast on the slower streams but
// +24% / +15% on Conan and Top Gun; 2^10 Conan -20% from 2^12; 2^15 -6%). With few substreams in parse (kFewStreams
// at most: pictures without WPP or tiles parse as one substream each), all of them large (a substream of
// kLargeStreamBytes or more each: ~15 bins a byte, so more than two steps of 2^16 bins), the steps take
// kParseBinBudgetFew: a 4K all-intra picture has ~1.5M bins, and short steps left the GPU idle between the
// submissions (isp_ai_4k: 2^14 21.4 fps, 2^16 25.1). Beside a small picture they don't: it would wait for the long
// step's end to be handed over, and the pictures after it to start (VVenC random access without WPP: fg_videoq 1080p,
// 2^16 109.0 fps, 2^14 207.4).
static constexpr int32_t  kParseBinBudget    = ParseSlot::kStepBins;
static constexpr int32_t  kParseBinBudgetFew = 1 << 16;
static constexpr size_t   kFewStreams        = 32;
static constexpr uint32_t kLargeStreamBytes  = 8192;
static constexpr uint32_t kParseMaxStalls    = 4;         // submissions in a row without progress before giving up
static constexpr uint32_t kParseLocalBits    = 12;        // a meta word: the CTU above the record's place in the CTU
static constexpr size_t   kParseCounters     = 8;         // the staged CUs and TUs, the room of the 5 job lists, the levels
static constexpr size_t   kParseTotals       = 16;        // the CUs and TUs, then the counters (compaction pass 0)
static constexpr size_t   kParseLevelChunk   = 4096;      // parse.slang's kLevelChunk: the levels a substream takes at a time
static constexpr size_t   kGroupsPerRow      = 32768;     // the shaders take group.y as rows of 32768 workgroups

// The service: each picture's dispatches in one submission (one, so that a picture is handed over as soon as it is
// parsed), the pictures parsed at a time; the buffers of the parse results kept for reuse, in each size.
// The pictures in parse: at least kMinParsing, and more while their substreams number fewer than kStreamsInParse (up
// to kMaxParsing). Each step has a workgroup per substream, of which two fit a compute unit: fewer substreams leave
// them idle (Top Gun slower 1080p, 18 substreams a picture: 8 pictures 299.3 fps, 10 314.1, 12 314.9, 14 314.1;
// Beekeeper slower 4K, 26 a picture: 8 139.4, 12 139.6, 16 139.1). Each picture in parse holds its results' buffers
// (1080p: 10 pictures 73 MB more commit than 8). With 16 at a time the intra picture that starts VideoQ 1080p parsed
// in 800 ms (~190 ms alone) while the first frame waited for it.
static constexpr uint32_t kStepsPerSubmission = 1;
static constexpr size_t   kMinParsing         = 8;
static constexpr size_t   kMaxParsing         = 16;
static constexpr size_t   kStreamsInParse     = 180;
// The pictures in parse and the parse results not given back yet (their reconstruction still to read them) together:
// no more start while this many are. With short steps the parse runs far ahead of the reconstruction (up to ~100
// pictures as the decode starts), and its lead hides the long parses of intra pictures, but every picture parsed ahead
// holds its records. 48 keeps most of the lead's speed for a little memory (15 W, Top Gun slower 1080p: no cap 394.7
// fps and 221 MB of GPU memory at the peak, 48: 392.0 and 213 MB, 32: 382.6 and 205 MB, 24: 374.9; Beekeeper slower
// 4K: 172.9 / 543, 48: 172.0 / 503, 32: 170.6 / 498).
static constexpr size_t   kMaxAhead           = 48;

// The bytes of a picture's largest substream (its parse's longest)
static uint32_t largestStreamBytes( const ParseInput& in )
{
  uint32_t most = 0;
  for( uint32_t s = 0; s < in.numStreams; s++ )
  {
    most = std::max( most, in.streams[size_t( s ) * kParseStreamWords + 1] );
  }
  return most;
}

// The substreams of a picture's slices (each slice's entry points + 1), the workgroups of its parse steps
static size_t substreamsOf( const Picture* pic )
{
  size_t streams = 0;
  for( const Slice* slice: pic->slices )
  {
    streams += size_t( slice->getNumEntryPoints() ) + 1;
  }
  return streams;
}

// The push constants of parse.slang
struct ParsePush
{
  VkDeviceAddress bits, streams, ctuList, ctuInfo, slices, pic, scanTab, cus, tus, ctuOut, bottomRows, levels, sync, wppCtx;
  int32_t         numStreams, levelCap;
  VkDeviceAddress prof;
  VkDeviceAddress state;
  int32_t         binBudget;
  VkDeviceAddress cuMeta, tuMeta, counters;
  int32_t         cuCap, tuCap;
};
static_assert( sizeof( ParsePush ) == 176, "ParsePush: the layout of parse.slang's Params" );

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
                      &m_wpp, &m_sync, &m_state, &m_ctuOut, &m_counters, &m_ctus, &m_totals, &m_levels } )
  {
    m_ctx->free( *buf );
  }
}

bool ParseSlot::prepare( const CodingStructure& cs, StagedRecords staged, Buffer levels )
{
  m_ctx->free( m_levels );   // (none: taken back after the picture before)
  m_levels   = levels;
  m_levelCap = m_levels.size / sizeof( int16_t );
  if( !buildParseInput( cs, m_in ) )
  {
    return false;
  }
  CHECK( m_in.cuSlots >= ( 1u << kParseLocalBits ) || m_in.tuSlots >= ( 1u << kParseLocalBits ), "Vulkan: more records in a CTU than the parse's meta words hold" );
  m_numCtus = cs.pcv->sizeInCtus;

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
  setStaged( staged );
  clear();
  return true;
}

void ParseSlot::setStaged( StagedRecords staged )
{
  for( Buffer* buf: { &m_cus, &m_tus, &m_cuMeta, &m_tuMeta } )
  {
    m_ctx->free( *buf );   // (none: taken back before)
  }
  m_cus    = staged.cus;
  m_tus    = staged.tus;
  m_cuMeta = staged.cuMeta;
  m_tuMeta = staged.tuMeta;
  // (all the room the buffers have)
  m_cuCap = std::min<size_t>( m_cus.size / sizeof( CuRecord ), m_cuMeta.size / sizeof( uint32_t ) );
  m_tuCap = std::min<size_t>( m_tus.size / sizeof( TuRecord ), m_tuMeta.size / sizeof( uint32_t ) );
}

void ParseSlot::growStaged( Context* ctx, StagedRecords& staged, size_t cus, size_t tus )
{
  ensureBuffer( ctx, staged.cus, std::max<size_t>( cus, 1 ) * sizeof( CuRecord ), MemKind::Device );
  ensureBuffer( ctx, staged.cuMeta, std::max<size_t>( cus, 1 ) * sizeof( uint32_t ), MemKind::Device );
  ensureBuffer( ctx, staged.tus, std::max<size_t>( tus, 1 ) * sizeof( TuRecord ), MemKind::Device );
  ensureBuffer( ctx, staged.tuMeta, std::max<size_t>( tus, 1 ) * sizeof( uint32_t ), MemKind::Device );
}

StagedRecords ParseSlot::takeStaged()
{
  const StagedRecords staged{ m_cus, m_tus, m_cuMeta, m_tuMeta };
  m_cus = m_tus = m_cuMeta = m_tuMeta = Buffer();
  m_cuCap = m_tuCap = 0;
  return staged;
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
  m_numLevels    = 0;
  m_caps         = FrontCapacities();
  m_lastProgress = 0;
  m_stalls       = 0;
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

void ParseSlot::recordStep( VkCommandBuffer cmd, int32_t binBudget ) const
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
  push.levels     = m_levels.addr;
  push.sync       = m_sync.addr;
  push.wppCtx     = m_wpp.addr;
  push.numStreams = int32_t( m_in.numStreams );
  push.levelCap   = int32_t( std::min<size_t>( m_levelCap, INT32_MAX ) );
  push.prof       = m_prof;
  push.state      = m_state.addr;
  push.binBudget  = binBudget;
  push.cuMeta     = m_cuMeta.addr;
  push.tuMeta     = m_tuMeta.addr;
  push.counters   = m_counters.addr;
  push.cuCap      = int32_t( std::min<size_t>( m_cuCap, INT32_MAX ) );
  push.tuCap      = int32_t( std::min<size_t>( m_tuCap, INT32_MAX ) );
  // (parse_prof for the profile; parse_shuffle where the device lacks what parse reads threads' values with; the
  // _ctu64 variants for CTUs of 64 samples at most, their maps a quarter of the shared memory; the _lite ones for
  // pictures that leave off the tools they leave out)
  const bool ctu64  = picture->cs->pcv->maxCUWidth <= 64;
  const bool lite   = ctu64 && m_in.lite;
  const int  shader = m_prof                    ? ( lite ? kShader_parse_prof_ctu64_lite : ctu64 ? kShader_parse_prof_ctu64 : kShader_parse_prof )
                      : m_ctx->subgroupBallot() ? ( lite ? kShader_parse_ctu64_lite : ctu64 ? kShader_parse_ctu64 : kShader_parse )
                                                : kShader_parse_shuffle;
  dispatch( m_ctx, cmd, shader, push, m_in.numStreams );   // a workgroup per substream
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
  // all parsed, and the compaction's first pass done: the staged records and the levels within their room, as many
  // records as the CTUs counted
  const uint32_t* sums = totals();
  if( sums[2] > m_cuCap || sums[3] > m_tuCap || sums[9] > m_levelCap )
  {
    return Progress::Overflow;
  }
  if( sums[0] != sums[2] || sums[1] != sums[3] )
  {
    return Progress::Failed;   // (the CTUs' counts and the records staged disagree: a bug)
  }
  m_numCus    = sums[0];
  m_numTus    = sums[1];
  m_numLevels = sums[9];
  // the job lists' room, as gpu::frontCapacities works it out (with its margins)
  m_caps.trafo = size_t( sums[4] ) + 64;
  m_caps.inter = size_t( sums[5] ) + 64;
  m_caps.intra = size_t( sums[6] ) + 64;
  m_caps.bands = size_t( sums[7] ) + 64;
  m_caps.edges = size_t( sums[8] ) + 64;
  return Progress::Done;
}

void ParseSlot::restart( StagedRecords staged, Buffer levels )
{
  setStaged( staged );
  m_ctx->free( m_levels );   // (none: taken back first)
  m_levels   = levels;
  m_levelCap = m_levels.size / sizeof( int16_t );
  clear();
}

size_t ParseSlot::cusCounted() const
{
  return totals()[2];
}

size_t ParseSlot::tusCounted() const
{
  return totals()[3];
}

size_t ParseSlot::levelsCounted() const
{
  return totals()[9];
}

Buffer ParseSlot::takeLevels()
{
  const Buffer levels = m_levels;
  m_levels            = Buffer();
  m_levelCap          = 0;
  return levels;
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

void ParseSlot::recordLevelsCopy( VkCommandBuffer cmd, const Buffer& levels ) const
{
  if( m_numLevels > 0 )
  {
    const VkBufferCopy region{ m_levels.offset, levels.offset, m_numLevels * sizeof( int16_t ) };
    m_ctx->api().vkCmdCopyBuffer( cmd, m_levels.buffer, levels.buffer, 1, &region );
  }
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

void applyParsedSaoAlf( CodingStructure& cs, const ParsedPicture& parsed )
{
  for( int ctu = 0; ctu < int( cs.pcv->sizeInCtus ); ctu++ )
  {
    applySaoAlf( cs.getCtuData( ctu ), parsed.saoAlf( ctu ) );
  }
}

// ====================================================================================================================
// ParseService: the pictures parsed ahead of their reconstruction
// ====================================================================================================================

ParseService::ParseService( Context* ctx, std::function<void()> wakeUp )
  : m_ctx( ctx ), m_wakeUp( std::move( wakeUp ) ), m_records( ctx, kIdlePictures )
{
  CHECK( !ctx->parseQueue(), "Vulkan: the GPU parse ahead of the reconstruction needs an async compute queue" );
  m_pool = m_ctx->createCommandPool( m_ctx->asyncFamily() );
  m_ctx->allocateCommandBuffers( m_pool, 1, &m_cmd );
  m_fence     = m_ctx->createFence();
  m_copyPool  = m_ctx->createCommandPool( m_ctx->asyncFamily() );
  m_ctx->allocateCommandBuffers( m_copyPool, 1, &m_copyCmd );
  m_copyFence = m_ctx->createFence();
  if( m_ctx->profiling() && m_ctx->shaderClock() && m_ctx->subgroupBallot() )   // (parse_prof reads as parse does)
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
  // the records and levels that slots still hold back to the arena first (a slot frees what it holds as buffers of
  // its own); the results' go with the arena
  for( const auto& slot: m_slots )
  {
    giveRecords( slot.get() );
  }
  m_slots.clear();
  m_ctx->free( m_profBuf );
  m_ctx->destroyFence( m_fence );
  m_ctx->destroyCommandPool( m_pool );
  m_ctx->destroyFence( m_copyFence );
  m_ctx->destroyCommandPool( m_copyPool );
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
    pic->gpuParseResult = nullptr;   // (fails at once)
    pic->gpuParsed.unlock_nothrow();
  }
  m_wakeUp();
}

void ParseService::release( ParsedPicture* parsed )
{
  std::lock_guard<std::mutex> lock( m_mutex );
  parsed->picture = nullptr;
  m_released.push_back( parsed );
  m_wake.notify_one();   // (a picture waiting for room ahead may start: kMaxAhead)
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
        // (the results handed over or being copied, which their pictures have not given back: kMaxAhead)
        const auto held = [&]() { return m_results.size() - m_freeResults.size() - m_released.size(); };
        m_wake.wait( lock, [&]() {
          return m_stop || !m_parsing.empty() || !m_copying.empty() || ( !m_waiting.empty() && m_parsing.size() + held() < kMaxAhead );
        } );
        if( m_stop )
        {
          return;
        }
        std::sort( m_waiting.begin(), m_waiting.end(),
                   []( const Picture* a, const Picture* b ) { return a->decodingOrderNumber < b->decodingOrderNumber; } );
        size_t streams = 0;
        for( const ParseSlot* slot: m_parsing )
        {
          streams += slot->input().numStreams;
        }
        const size_t ahead = m_parsing.size() + held();
        size_t       count = 0;
        while( count < m_waiting.size() && m_parsing.size() + count < kMaxParsing
               && ( m_parsing.size() + count < kMinParsing || streams < kStreamsInParse ) && ahead + count < kMaxAhead )
        {
          streams += substreamsOf( m_waiting[count] );
          count++;
        }
        starting.assign( m_waiting.begin(), m_waiting.begin() + count );
        m_waiting.erase( m_waiting.begin(), m_waiting.begin() + count );
      }
      for( Picture* pic: starting )
      {
        startPicture( pic );
      }
      if( !m_parsing.empty() || !m_copying.empty() )
      {
        submit();
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
      slot->picture->gpuParseResult = nullptr;
      slot->picture->gpuParsed.unlock_nothrow();
    }
    m_parsing.clear();
    for( const auto& copy: m_copying )
    {
      copy.first->picture->gpuParseResult = nullptr;
      copy.first->picture->gpuParsed.unlock_nothrow();
    }
    m_copying.clear();
    m_broken = true;
    for( Picture* pic: m_waiting )
    {
      pic->gpuParseResult = nullptr;
      pic->gpuParsed.unlock_nothrow();
    }
    m_waiting.clear();
    m_wakeUp();
  }
}

ParseSlot* ParseService::takeSlot()
{
  if( !m_free.empty() )
  {
    ParseSlot* slot = m_free.back();
    m_free.pop_back();
    return slot;
  }
  m_slots.push_back( std::make_unique<ParseSlot>( m_ctx ) );
  return m_slots.back().get();
}

ParsedPicture* ParseService::takeParsed()
{
  // the results given back become free, their records back to the arena
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    for( ParsedPicture* result: m_released )
    {
      for( Buffer* buf: { &result->m_cus, &result->m_tus, &result->m_ctus, &result->m_levels } )
      {
        m_records.give( *buf );
      }
      m_freeResults.push_back( result );
    }
    m_released.clear();
  }
  if( !m_freeResults.empty() )
  {
    ParsedPicture* result = m_freeResults.back();
    m_freeResults.pop_back();
    return result;
  }
  m_results.push_back( std::make_unique<ParsedPicture>() );
  return m_results.back().get();
}

StagedRecords ParseService::takeStaged( size_t cus, size_t tus )
{
  StagedRecords staged;
  staged.cus    = m_records.take( std::max<size_t>( cus, 1 ) * sizeof( CuRecord ) );
  staged.cuMeta = m_records.take( std::max<size_t>( cus, 1 ) * sizeof( uint32_t ) );
  staged.tus    = m_records.take( std::max<size_t>( tus, 1 ) * sizeof( TuRecord ) );
  staged.tuMeta = m_records.take( std::max<size_t>( tus, 1 ) * sizeof( uint32_t ) );
  return staged;
}

void ParseService::giveRecords( ParseSlot* slot )
{
  StagedRecords staged = slot->takeStaged();
  Buffer        levels = slot->takeLevels();
  for( Buffer* buf: { &staged.cus, &staged.tus, &staged.cuMeta, &staged.tuMeta, &levels } )
  {
    m_records.give( *buf );
  }
}

void ParseService::giveSlot( ParseSlot* slot )
{
  giveRecords( slot );
  m_free.push_back( slot );
}

// Whether all the picture's slices are intra slices
static bool intraPicture( const Picture& pic )
{
  return std::all_of( pic.slices.begin(), pic.slices.end(), []( const Slice* s ) { return s->isIntra(); } );
}

void ParseService::Counts::raise( const Counts& counts )
{
  cus    = std::max( cus, counts.cus );
  tus    = std::max( tus, counts.tus );
  levels = std::max( levels, counts.levels );
  bytes  = std::max( bytes, counts.bytes );
}

// The bytes of a picture's slice data
static size_t sliceDataBytes( const Picture& pic )
{
  size_t bytes = 0;
  for( const Slice* slice: pic.slices )
  {
    bytes += slice->parseTaskParams.gpuSliceData.size();
  }
  return bytes;
}

// The room a picture's parse starts with, for its staged records (CUs, TUs) and its levels. A picture with more is
// parsed again (Overflow). The room is the most the pictures of its kind had so far (each count on its own), with
// some to spare, and scaled up when the picture has more slice data than any of them had (by the ratio of the bytes,
// at most 16):
// - an intra picture: a quarter more than the intra pictures had;
// - another picture: twice what the pictures of its temporal layer had, which are alike (Beekeeper slower 4K: layer 0
//   up to 17,000 CUs, layer 5 up to 1,800); before its layer's first, a quarter more than any such picture had;
// - before the first of its kind: 32 records a CTU for an intra picture (VideoQ's have up to 23 with the dual tree),
//   8 for the others (VideoQ's have 2 to 4); half the samples' levels for an intra picture, a 32nd for the others.
// The levels' room is at least a chunk a substream (each takes its levels in chunks), and at most twice the samples.
// (The parse's counts replayed: Beekeeper slower 4K, a picture's room 2.75 MB with a quarter more than the most of its
// kind, 1.27 MB with this, the pictures parsed again 14 and none; Top Gun slower 1080p, 2.14 and 0.88 MB, 6 and 1.)
ParseService::Counts ParseService::startRoom( const Picture& pic ) const
{
  const PreCalcValues& pcv     = *pic.cs->pcv;
  const size_t         luma    = size_t( pcv.lumaWidth ) * pcv.lumaHeight;
  const size_t         chroma  = isChromaEnabled( pcv.chrFormat )
                                   ? luma >> ( getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) + getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) )
                                   : 0;
  const size_t         samples = luma + 2 * chroma;
  const size_t         ctus    = pcv.sizeInCtus;
  const bool           intra   = intraPicture( pic );
  const Counts*        layer   = !intra && pic.tempLayer < MAX_TLAYER ? &m_mostLayer[pic.tempLayer] : nullptr;
  const Counts*        kind    = intra ? &m_mostIntra : &m_mostInter;
  const Counts*        most    = layer && layer->cus > 0 ? layer : ( kind->cus > 0 ? kind : nullptr );
  Counts               room;
  if( most )
  {
    const double more  = std::min( std::max( 1.0, double( sliceDataBytes( pic ) ) / double( std::max<size_t>( most->bytes, 1 ) ) ), 16.0 );
    const double scale = ( most == layer ? 2.0 : 1.25 ) * more;
    room.cus           = size_t( double( most->cus ) * scale ) + 64;
    room.tus           = size_t( double( most->tus ) * scale ) + 64;
    room.levels        = size_t( double( most->levels ) * scale ) + kParseLevelChunk;
  }
  else
  {
    room = { ctus * ( intra ? 32 : 8 ), ctus * ( intra ? 32 : 8 ), intra ? samples / 2 : samples / 32 };
  }
  room.levels = std::min( std::max( room.levels, substreamsOf( &pic ) * kParseLevelChunk ), 2 * samples + 65536 );
  return room;
}

void ParseService::startPicture( Picture* pic )
{
  m_records.setChunkSize( RangeArena::chunkSizeFor( size_t( pic->cs->pcv->lumaWidth ) * pic->cs->pcv->lumaHeight ) );
  m_records.tick();
  ParseSlot* slot = takeSlot();
  slot->picture   = pic;
  slot->setProfile( m_prof );
  // room for the records and the levels of a picture of its kind, from the arena
  const Counts room = startRoom( *pic );
  if( !slot->prepare( *pic->cs, takeStaged( room.cus, room.tus ), m_records.take( room.levels * sizeof( int16_t ) ) ) )
  {
    giveSlot( slot );
    {
      std::lock_guard<std::mutex> lock( m_mutex );
      m_stats.errors++;
    }
    handOver( pic, nullptr );
    return;
  }
  m_parsing.push_back( slot );
  std::lock_guard<std::mutex> lock( m_mutex );
  m_stats.pictures++;
  m_stats.substreams += slot->input().numStreams;
}

// Waits for fence, which a submission of the parse's signals (its steps' or its copies')
static void waitFor( Context* ctx, VkFence fence )
{
  const VkResult result = ctx->api().vkWaitForFences( ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: waiting for the GPU's parse failed (" << result << ")" );
  }
}

void ParseService::submit()
{
  const bool     copies    = !m_copying.empty();
  const bool     steps     = !m_parsing.empty();
  const uint64_t copyValue = copies ? submitCopies() : 0;
  if( steps )
  {
    submitSteps();
  }
  if( copies )
  {
    // the pictures copied: to their reconstruction (whose front end waits for copyValue), their slots free for the
    // next pictures (the steps running meanwhile do not use them)
    waitFor( m_ctx, m_copyFence );
    for( const auto& copy: m_copying )
    {
      copy.second->m_parseValue = copyValue;
      giveSlot( copy.first );
      handOver( copy.second->picture, copy.second );
    }
    m_copying.clear();
  }
  if( steps )
  {
    waitFor( m_ctx, m_fence );
    std::lock_guard<std::mutex> lock( m_mutex );
    m_stats.submissions++;
    m_stats.parsingSum += int64_t( m_parsing.size() );
  }
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
  size_t streams  = 0;
  bool   allLarge = true;
  for( const ParseSlot* slot: m_parsing )
  {
    streams += slot->input().numStreams;
    allLarge = allLarge && largestStreamBytes( slot->input() ) >= kLargeStreamBytes;
  }
  const int32_t budget = streams <= kFewStreams && allLarge ? kParseBinBudgetFew : kParseBinBudget;
  for( uint32_t step = 0; step < kStepsPerSubmission; step++ )
  {
    for( const ParseSlot* slot: m_parsing )
    {
      slot->recordStep( m_cmd, budget );
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
  m_stepsValue = m_ctx->submitParse( m_cmd, m_fence );
}

uint64_t ParseService::submitCopies()
{
  const Api& vk = m_ctx->api();
  vk.vkResetCommandPool( m_ctx->device(), m_copyPool, 0 );
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( m_copyCmd, &begin );
  startBarrier( m_ctx, m_copyCmd );   // (what the parse wrote, visible)
  for( const auto& copy: m_copying )
  {
    copy.first->recordScatter( m_copyCmd, copy.second->cus(), copy.second->tus(), copy.second->ctus() );
    copy.first->recordLevelsCopy( m_copyCmd, copy.second->m_levels );
  }
  vk.vkEndCommandBuffer( m_copyCmd );
  vk.vkResetFences( m_ctx->device(), 1, &m_copyFence );
  return m_ctx->submitParseCopy( m_copyCmd, m_copyFence, m_stepsValue );   // (after the steps that parsed them)
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
      // more records or levels than the room: parsed again with room for them (a quarter more than counted)
      StagedRecords staged = slot->takeStaged();
      Buffer        levels = slot->takeLevels();
      const size_t  cus    = std::max( slot->cusCounted() * 5 / 4 + 64, staged.cus.size / sizeof( CuRecord ) );
      const size_t  tus    = std::max( slot->tusCounted() * 5 / 4 + 64, staged.tus.size / sizeof( TuRecord ) );
      const size_t  room   = std::max( slot->levelsCounted() * 5 / 4 + 4096, levels.size / sizeof( int16_t ) );
      for( Buffer* buf: { &staged.cus, &staged.tus, &staged.cuMeta, &staged.tuMeta, &levels } )
      {
        m_records.give( *buf );
      }
      slot->restart( takeStaged( cus, tus ), m_records.take( room * sizeof( int16_t ) ) );
      std::lock_guard<std::mutex> lock( m_mutex );
      m_stats.restarts++;
      i++;
      continue;
    }
    m_parsing.erase( m_parsing.begin() + i );
    const bool failed = progress != ParseSlot::Progress::Done;
    if( !failed )
    {
      // the most records and levels a picture of its kind had: the room the next ones' parses start with
      const Picture& pic = *slot->picture;
      const Counts   counts{ slot->numCus(), slot->numTus(), slot->numLevels(), sliceDataBytes( pic ) };
      if( intraPicture( pic ) )
      {
        m_mostIntra.raise( counts );
      }
      else
      {
        m_mostInter.raise( counts );
        if( pic.tempLayer < MAX_TLAYER )
        {
          m_mostLayer[pic.tempLayer].raise( counts );
        }
      }
    }
    {
      std::lock_guard<std::mutex> lock( m_mutex );
      m_stats.runs += slot->runs();
      m_stats.errors += failed ? 1 : 0;
      m_stats.maxCus = int64_t( std::max( m_mostIntra.cus, m_mostInter.cus ) );
      m_stats.maxTus = int64_t( std::max( m_mostIntra.tus, m_mostInter.tus ) );
    }
    if( failed )
    {
      giveSlot( slot );
      handOver( slot->picture, nullptr );
      continue;
    }
    // parsed: its records copied into a result of their size (the next submit), then handed over
    ParsedPicture* result = takeParsed();
    result->picture       = slot->picture;
    result->m_numCus      = slot->numCus();
    result->m_numTus      = slot->numTus();
    result->m_caps        = slot->capacities();
    result->m_saoAlf.resize( slot->numCtus() * ParsedPicture::kSaoAlfWords );
    for( size_t ctu = 0; ctu < slot->numCtus(); ctu++ )
    {
      const uint32_t* words = slot->saoAlf( int( ctu ) );
      std::copy( words, words + ParsedPicture::kSaoAlfWords, result->m_saoAlf.begin() + ctu * ParsedPicture::kSaoAlfWords );
    }
    result->m_cus       = m_records.take( std::max<size_t>( result->m_numCus, 1 ) * sizeof( CuRecord ) );
    result->m_tus       = m_records.take( std::max<size_t>( result->m_numTus, 1 ) * sizeof( TuRecord ) );
    result->m_ctus      = m_records.take( slot->numCtus() * sizeof( CtuRecord ) );
    result->m_numLevels = slot->numLevels();
    result->m_levels    = m_records.take( std::max<size_t>( result->m_numLevels, 1 ) * sizeof( int16_t ) );
    m_copying.emplace_back( slot, result );
  }
}

void ParseService::handOver( Picture* pic, ParsedPicture* parsed )
{
  pic->gpuParseResult = parsed;
  pic->gpuParsed.unlock_nothrow();
  m_wakeUp();   // (the picture's tasks may wait in a sleeping thread pool)
}

}   // namespace gpu
}   // namespace vvdec
