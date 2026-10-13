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
// The bins each substream parses per dispatch (a CU counts as 32 bins). A step's workgroups hold their compute units'
// shared memory for as long as the step runs, and the steps run one after another. Short steps let the
// reconstruction's passes on the other queues run in between, which decodes faster although it costs more submissions
// (2^12 against 2^14 bins at 15 W: Conan 1080p +39%, Beekeeper slower 4K +3.8%).
// Long steps (kParseBinBudgetFew) are taken while at most kFewStreams substreams are in parse and every picture in parse
// has a substream of at least kLargeStreamBytes, as all-intra pictures without WPP do: their short steps would leave
// the GPU idle between submissions. Don't take long steps beside a small picture: it would wait for the long step to
// end before it is handed over, and so would the pictures after it.
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

// How many pictures are in parse at a time. Each step runs one workgroup per substream, and two workgroups fill a
// compute unit, so too few substreams leave units idle. At least kMinParsing pictures are in parse, and more start
// while their substreams number fewer than kStreamsInParse, up to kMaxParsing. Each picture in parse holds the room
// for its parse results, so more pictures cost memory. Each submission runs kStepsPerSubmission steps of every picture
// in parse, so that a picture can be handed over as soon as its parse is done.
static constexpr uint32_t kStepsPerSubmission = 1;
static constexpr size_t   kMinParsing         = 8;
static constexpr size_t   kMaxParsing         = 16;
static constexpr size_t   kStreamsInParse     = 180;
// The most pictures in parse or parsed and not yet given back (their reconstruction still has to read the results):
// no more start while this many are. Short steps let the parse run far ahead of the reconstruction, and that lead hides
// the long parses of intra pictures, but every picture parsed ahead holds its records. 48 keeps almost all of the
// lead's speed for a little memory (Beekeeper slower 4K at 15 W: 172.0 fps and 503 MB of GPU memory, against 172.9 fps
// and 543 MB without a cap).
static constexpr size_t   kMaxAhead           = 48;
// An intra picture with at most kSerialStreams substreams (no WPP, one tile) parses in one or two long chains, several
// times as long as a B picture's parse. Such a picture starts as soon as it is enqueued, ahead of the pictures
// enqueued before it, so that its parse is done when the reconstruction reaches it; at most kSerialIntraInParse of
// them are in parse at once. Don't start the intra pictures of WPP streams early: they parse their rows side by side,
// and starting them early only holds back the pictures before them.
static constexpr size_t   kSerialStreams      = 2;
static constexpr size_t   kSerialIntraInParse = 2;
// In an all-intra stream without WPP every picture's parse is one such long chain, and the parse holds the whole decode
// back. So while every picture in parse is a serial intra picture, more of them parse side by side, up to
// kMaxSerialIntraParsing. Each one holds its parse room (4K: ~25 MB of GPU memory). With 24 instead of 16 the all-intra
// 4K streams decode 32% to 39% faster at 15 W for 250 to 300 MB more memory; 32 is no faster. (Random-access streams
// without WPP gain nothing from more pictures in parse.)
static constexpr size_t   kMaxSerialIntraParsing = 24;

// Returns the size in bytes of the picture's largest substream, the one whose parse takes longest
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
  uint32_t        serial;
};
static_assert( sizeof( ParseCompactPush ) == 128, "ParseCompactPush: the layout of parsecompact.slang's Params" );


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
    m_ctx->ensure( buf, std::max<size_t>( size, 16 ), MemKind::Upload );
    memcpy( buf.mapped, data, size );
  };
  m_ctx->ensure( m_bits, m_in.bits.size() + 256, MemKind::Upload );
  memcpy( m_bits.mapped, m_in.bits.data(), m_in.bits.size() );
  memset( m_bits.mapped + m_in.bits.size(), 0, 256 );
  upload( m_streams, m_in.streams.data(), m_in.streams.size() * sizeof( uint32_t ) );
  upload( m_ctuList, m_in.ctuList.data(), m_in.ctuList.size() * sizeof( uint32_t ) );
  upload( m_ctuInfo, m_in.ctuInfo.data(), m_in.ctuInfo.size() * sizeof( uint32_t ) );
  upload( m_slices, m_in.slices.data(), m_in.slices.size() * sizeof( uint32_t ) );
  upload( m_pic, m_in.pic.data(), m_in.pic.size() * sizeof( int32_t ) );
  if( !m_scans.mapped )
  {
    m_ctx->ensure( m_scans, kParseScanTab * sizeof( uint32_t ), MemKind::Upload );
    writeParseScans( reinterpret_cast<uint32_t*>( m_scans.mapped ) );
  }

  // what the parse writes
  m_ctx->ensure( m_bottomRows, m_numCtus * 64 * 2 * sizeof( uint32_t ), MemKind::Device );
  m_ctx->ensure( m_wpp, std::max<size_t>( m_in.wppSlots, 1 ) * m_in.contexts * sizeof( uint32_t ), MemKind::Device );
  m_ctx->ensure( m_sync, ( 2 + m_numCtus ) * sizeof( uint32_t ), MemKind::HostCached );
  m_ctx->ensure( m_state, std::max<size_t>( m_in.numStreams, 1 ) * kParseStateWords * sizeof( uint32_t ), MemKind::HostCached );
  m_ctx->ensure( m_ctuOut, m_numCtus * kParseCtuOutWords * sizeof( uint32_t ), MemKind::HostCached );
  m_ctx->ensure( m_counters, kParseCounters * sizeof( uint32_t ), MemKind::Device );   // (an atomic a CU: fast in GPU memory)
  m_ctx->ensure( m_ctus, m_numCtus * sizeof( CtuRecord ), MemKind::HostCached );
  m_ctx->ensure( m_totals, 2 * kParseTotals * sizeof( uint32_t ), MemKind::HostCached );   // (two halves: recordCompaction)
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
  ctx->ensure( staged.cus, std::max<size_t>( cus, 1 ) * sizeof( CuRecord ), MemKind::Device );
  ctx->ensure( staged.cuMeta, std::max<size_t>( cus, 1 ) * sizeof( uint32_t ), MemKind::Device );
  ctx->ensure( staged.tus, std::max<size_t>( tus, 1 ) * sizeof( TuRecord ), MemKind::Device );
  ctx->ensure( staged.tuMeta, std::max<size_t>( tus, 1 ) * sizeof( uint32_t ), MemKind::Device );
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

void ParseSlot::recordCompaction( VkCommandBuffer cmd, uint32_t serial ) const
{
  ParseCompactPush push{};
  push.ctuOut   = m_ctuOut.addr;
  push.ctuInfo  = m_ctuInfo.addr;
  push.sync     = m_sync.addr;
  push.counters = m_counters.addr;
  push.ctus     = m_ctus.addr;
  push.totals   = m_totals.addr + VkDeviceAddress( serial & 1u ) * kParseTotals * sizeof( uint32_t );
  push.numCtus  = int32_t( m_numCtus );
  push.pass     = 0;
  push.serial   = serial;
  dispatch( m_ctx, cmd, kShader_parsecompact, push, 1 );
}

const uint32_t* ParseSlot::totals() const
{
  return reinterpret_cast<const uint32_t*>( m_totals.mapped ) + m_half * kParseTotals;
}

ParseSlot::Progress ParseSlot::progress( uint32_t serial )
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
  // (the state is read as it is now: a step after the one asked about may have parsed the last CTUs, and then the
  // totals of the step asked about are not the picture's: the next step's are)
  m_half = serial & 1u;
  if( serial != 0 && totals()[15] != serial )
  {
    return Progress::Parsing;
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
  for( StepFrame& frame: m_frames )
  {
    frame.pool = m_ctx->createCommandPool( m_ctx->asyncFamily() );
    m_ctx->allocateCommandBuffers( frame.pool, 1, &frame.cmd );
    frame.fence = m_ctx->createFence();
  }
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
  for( StepFrame& frame: m_frames )
  {
    m_ctx->destroyFence( frame.fence );
    m_ctx->destroyCommandPool( frame.pool );
  }
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

static bool serialIntraPicture( const Picture& pic );

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
          return m_stop || !m_parsing.empty() || !m_copying.empty() || m_inFlight > 0 || !m_restarting.empty() || !m_freeing.empty()
                 || ( !m_waiting.empty() && m_parsing.size() + held() < kMaxAhead );
        } );
        if( m_stop )
        {
          lock.unlock();
          waitAll();   // (before the slots and frames go)
          return;
        }
        std::sort( m_waiting.begin(), m_waiting.end(),
                   []( const Picture* a, const Picture* b ) { return a->decodingOrderNumber < b->decodingOrderNumber; } );
        size_t streams = 0;
        for( const ParseSlot* slot: m_parsing )
        {
          streams += slot->input().numStreams;
        }
        const size_t ahead          = m_parsing.size() + held();
        bool         allSerialIntra = std::all_of( m_parsing.begin(), m_parsing.end(),
                                                   []( const ParseSlot* slot ) { return serialIntraPicture( *slot->picture ); } );
        size_t       count          = 0;
        while( count < m_waiting.size() && ahead + count < kMaxAhead )
        {
          const size_t inParse = m_parsing.size() + count;
          allSerialIntra       = allSerialIntra && serialIntraPicture( *m_waiting[count] );
          const size_t cap     = allSerialIntra ? kMaxSerialIntraParsing : kMaxParsing;
          if( inParse >= cap || ( inParse >= kMinParsing && streams >= kStreamsInParse ) )
          {
            break;
          }
          streams += substreamsOf( m_waiting[count] );
          count++;
        }
        starting.assign( m_waiting.begin(), m_waiting.begin() + count );
        m_waiting.erase( m_waiting.begin(), m_waiting.begin() + count );
        // then the intra pictures that parse in long chains, ahead of their turn (kSerialStreams)
        size_t serialIntra = 0;
        for( const ParseSlot* slot: m_parsing )
        {
          serialIntra += serialIntraPicture( *slot->picture ) ? 1 : 0;
        }
        for( const Picture* pic: starting )
        {
          serialIntra += serialIntraPicture( *pic ) ? 1 : 0;
        }
        for( auto it = m_waiting.begin(); it != m_waiting.end() && serialIntra < kSerialIntraInParse; )
        {
          if( serialIntraPicture( **it ) )
          {
            starting.push_back( *it );
            it = m_waiting.erase( it );
            serialIntra++;
          }
          else
          {
            ++it;
          }
        }
      }
      for( Picture* pic: starting )
      {
        startPicture( pic );
      }
      runRound();
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
    for( ParseSlot* slot: m_restarting )
    {
      slot->picture->gpuParseResult = nullptr;
      slot->picture->gpuParsed.unlock_nothrow();
    }
    m_restarting.clear();
    m_freeing.clear();   // (their pictures were handed over)
    for( const auto* copies: { &m_copying, &m_copied } )
    {
      for( const auto& copy: *copies )
      {
        copy.first->picture->gpuParseResult = nullptr;
        copy.first->picture->gpuParsed.unlock_nothrow();
      }
    }
    m_copying.clear();
    m_copied.clear();
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

// An intra picture whose parse is one or two long chains (kSerialStreams)
static bool serialIntraPicture( const Picture& pic )
{
  return substreamsOf( &pic ) <= kSerialStreams && intraPicture( pic );
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

// How much room startRoom gives a picture's parse beyond what similar pictures needed
static constexpr double kKindRoomScale  = 1.25;   // over the most any picture of its kind (intra or not) had
static constexpr double kLayerRoomScale = 2.0;    // over the most the pictures of its temporal layer had
static constexpr double kMaxBytesScale  = 16.0;   // the most the room grows with the picture's share of slice data
static constexpr size_t kRecordsSpare   = 64;     // records added to every room
// The room before any picture of the kind was parsed: records per CTU and levels per sample
static constexpr size_t kFirstIntraRecordsPerCtu = 32;
static constexpr size_t kFirstInterRecordsPerCtu = 8;
static constexpr size_t kFirstIntraSamplesPerLevel = 2;
static constexpr size_t kFirstInterSamplesPerLevel = 32;
// The levels' room is at most twice the samples, plus slack for the substreams' last chunks of levels, which are
// partly unused
static constexpr size_t kLevelsChunkSlack = 16 * kParseLevelChunk;

// Returns the room a picture's parse starts with, for its staged records (CUs, TUs) and its levels. A picture that
// needs more is parsed again (Overflow). The room is the most that the pictures like it needed so far, each count on
// its own, with some to spare:
// - for an intra picture, kKindRoomScale times what the intra pictures needed;
// - for another picture, kLayerRoomScale times what the pictures of its temporal layer needed, since the pictures of a
//   layer are alike (Beekeeper slower 4K: layer 0 up to 17,000 CUs, layer 5 up to 1,800). Before its layer's first
//   picture, kKindRoomScale times what any such picture needed;
// - before the first picture of its kind, kFirstIntra... or kFirstInter... records a CTU and levels a sample.
// The room grows with the ratio of the picture's slice data to the most those pictures had (at most kMaxBytesScale).
// The levels' room is at least one chunk a substream, since each substream takes its levels a chunk at a time.
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
    const double bytesRatio = double( sliceDataBytes( pic ) ) / double( std::max<size_t>( most->bytes, 1 ) );
    const double more       = std::min( std::max( 1.0, bytesRatio ), kMaxBytesScale );
    const double scale      = ( most == layer ? kLayerRoomScale : kKindRoomScale ) * more;
    room.cus                = size_t( double( most->cus ) * scale ) + kRecordsSpare;
    room.tus                = size_t( double( most->tus ) * scale ) + kRecordsSpare;
    room.levels             = size_t( double( most->levels ) * scale ) + kParseLevelChunk;
  }
  else
  {
    const size_t records = ctus * ( intra ? kFirstIntraRecordsPerCtu : kFirstInterRecordsPerCtu );
    room                 = { records, records, samples / ( intra ? kFirstIntraSamplesPerLevel : kFirstInterSamplesPerLevel ) };
  }
  room.levels = std::min( std::max( room.levels, substreamsOf( &pic ) * kParseLevelChunk ), 2 * samples + kLevelsChunkSlack );
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

// Returns how many steps to keep in the parse queue: 2 while a picture whose parse is one or two long chains
// (kSerialStreams) is in parse, else 1. With two steps queued, the GPU runs the next step while this thread takes the
// results of the one before, which shortens such a picture's parse (Conan 1080p +7.2% at 15 W). Pictures with WPP parse
// their rows side by side, and a second step would only weigh on the reconstruction.
size_t ParseService::stepsAhead() const
{
  for( const ParseSlot* slot: m_parsing )
  {
    if( slot->input().numStreams <= kSerialStreams )
    {
      return 2;
    }
  }
  return 1;
}

// Runs one round of the service: queues steps up to stepsAhead(), copies and hands over the pictures found parsed, and
// takes the results of the oldest step in flight
void ParseService::runRound()
{
  // Keep stepsAhead() steps in the parse queue while pictures are in parse.
  const size_t ahead = stepsAhead();
  while( m_inFlight < ahead && !m_parsing.empty() )
  {
    submitSteps();
  }
  // Copy the records of the pictures found parsed. On the GPU the copy waits for the steps that still step their slots:
  // at most the oldest step in flight, since the steps after it leave those pictures out. When those steps are done
  // already, hand the pictures over at once.
  if( !m_copying.empty() && m_copied.empty() )
  {
    submitCopies();
  }
  if( !m_copied.empty() && m_copyAfter <= m_doneValue )
  {
    handOverCopied();
  }
  // Take the results of the oldest step.
  if( m_inFlight > 0 )
  {
    StepFrame& frame = oldestFrame();
    // A copy in flight waits for this step on the GPU, so waiting for the copy covers both with one wake-up.
    if( !m_copied.empty() )
    {
      waitFor( m_ctx, m_copyFence );
    }
    waitFor( m_ctx, frame.fence );
    m_doneValue = frame.value;
    retireOldestFrame();
    {
      std::lock_guard<std::mutex> lock( m_mutex );
      m_stats.submissions++;
      m_stats.parsingSum += int64_t( frame.slots.size() );
    }
    finishPictures( frame );
  }
  // Hand over the pictures whose copy waited for that step.
  if( !m_copied.empty() )
  {
    handOverCopied();
  }
  endSlots();
}

// Hands the copied pictures to their reconstruction, whose front ends wait for m_copyValue on the GPU, and frees their
// slots. The copy waited for the steps that stepped those slots, which are done, so it ends soon.
void ParseService::handOverCopied()
{
  waitFor( m_ctx, m_copyFence );
  for( const auto& copy: m_copied )
  {
    copy.second->m_parseValue = m_copyValue;
    giveSlot( copy.first );
    handOver( copy.second->picture, copy.second );
  }
  m_copied.clear();
}

void ParseService::submitSteps()
{
  StepFrame& frame = nextFreeFrame();
  const Api& vk    = m_ctx->api();
  vk.vkResetCommandPool( m_ctx->device(), frame.pool, 0 );
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( frame.cmd, &begin );
  // the counters of the pictures starting cleared; what the submission before wrote visible
  for( ParseSlot* slot: m_parsing )
  {
    slot->recordClear( frame.cmd );
  }
  startBarrier( m_ctx, frame.cmd );
  // the pictures' dispatches side by side, a barrier after each round; then their compaction's first passes
  size_t streams  = 0;
  bool   allLarge = true;
  for( const ParseSlot* slot: m_parsing )
  {
    streams += slot->input().numStreams;
    allLarge = allLarge && largestStreamBytes( slot->input() ) >= kLargeStreamBytes;
  }
  const int32_t budget = streams <= kFewStreams && allLarge ? kParseBinBudgetFew : kParseBinBudget;
  for( uint32_t s = 0; s < kStepsPerSubmission; s++ )
  {
    for( const ParseSlot* slot: m_parsing )
    {
      slot->recordStep( frame.cmd, budget );
    }
    computeBarrier( m_ctx, frame.cmd );
  }
  m_serial     = m_serial == UINT32_MAX ? 1 : m_serial + 1;   // (0 is no step's)
  frame.serial = m_serial;
  for( const ParseSlot* slot: m_parsing )
  {
    slot->recordCompaction( frame.cmd, frame.serial );
  }
  hostReadBarrier( m_ctx, frame.cmd );
  vk.vkEndCommandBuffer( frame.cmd );
  vk.vkResetFences( m_ctx->device(), 1, &frame.fence );
  frame.value = m_ctx->submitParse( frame.cmd, frame.fence );
  frame.slots = m_parsing;
  for( ParseSlot* slot: m_parsing )
  {
    slot->lastStep = frame.value;
  }
  m_inFlight++;
}

void ParseService::submitCopies()
{
  const Api& vk = m_ctx->api();
  vk.vkResetCommandPool( m_ctx->device(), m_copyPool, 0 );
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( m_copyCmd, &begin );
  startBarrier( m_ctx, m_copyCmd );   // (what the parse wrote, visible)
  uint64_t after = 0;
  for( const auto& copy: m_copying )
  {
    copy.first->recordScatter( m_copyCmd, copy.second->cus(), copy.second->tus(), copy.second->ctus() );
    copy.first->recordLevelsCopy( m_copyCmd, copy.second->m_levels );
    after = std::max( after, copy.first->lastStep );
  }
  vk.vkEndCommandBuffer( m_copyCmd );
  vk.vkResetFences( m_ctx->device(), 1, &m_copyFence );
  m_copyValue = m_ctx->submitParseCopy( m_copyCmd, m_copyFence, after );   // (after the steps that stepped them)
  m_copyAfter = after;
  m_copied.swap( m_copying );
}

void ParseService::finishPictures( const StepFrame& frame )
{
  for( ParseSlot* slot: frame.slots )
  {
    const auto at = std::find( m_parsing.begin(), m_parsing.end(), slot );
    if( at == m_parsing.end() )
    {
      continue;   // (its parse ended at the step before)
    }
    const ParseSlot::Progress progress = slot->progress( frame.serial );
    if( progress == ParseSlot::Progress::Parsing )
    {
      continue;
    }
    m_parsing.erase( at );
    if( progress == ParseSlot::Progress::Overflow )
    {
      // parsed again with room for its records once no step in flight steps it (endSlots)
      m_restarting.push_back( slot );
      std::lock_guard<std::mutex> lock( m_mutex );
      m_stats.restarts++;
      continue;
    }
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
      handOver( slot->picture, nullptr );
      m_freeing.push_back( slot );   // (free once no step in flight steps it: endSlots)
      continue;
    }
    // parsed: its records copied into a result of their size (the next round), then handed over
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

void ParseService::endSlots()
{
  for( size_t i = 0; i < m_restarting.size(); )
  {
    ParseSlot* slot = m_restarting[i];
    if( slot->lastStep > m_doneValue )
    {
      i++;
      continue;
    }
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
    m_parsing.push_back( slot );
    m_restarting.erase( m_restarting.begin() + i );
  }
  for( size_t i = 0; i < m_freeing.size(); )
  {
    if( m_freeing[i]->lastStep > m_doneValue )
    {
      i++;
      continue;
    }
    giveSlot( m_freeing[i] );
    m_freeing.erase( m_freeing.begin() + i );
  }
}

void ParseService::waitAll()
{
  while( m_inFlight > 0 )
  {
    waitFor( m_ctx, oldestFrame().fence );
    retireOldestFrame();
  }
  if( !m_copied.empty() )
  {
    waitFor( m_ctx, m_copyFence );
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
