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

/** \file     GpuReconFrontEnd.cpp
    \brief    GpuRecon's GPU front end: motion derivation, deblocking parameters, jobs, intra order and deblocking
              edges, from the parse's records (see GpuRecon.h and GpuFrontEnd.h)
*/

#include "GpuRecon.h"

#include "vvdec_gpu/GpuShaderIds.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Slice.h"

#include <cstring>

namespace vvdec
{
namespace gpu
{

// The push constants of mider.slang
struct MiderPush
{
  VkDeviceAddress cus, ctus, slices, cuMap, cuMotion, motion, progress, subpics;
  int32_t         numCus, widthInCtus, heightInCtus, log2Ctu, width, height, widthIn4, plevel, wpp, maxNumMergeCand, subpicOn, sps;
  VkDeviceAddress cuMapC;
  int32_t         csx, csy, tileCols, colOn;
  VkDeviceAddress colMotion;
};
static_assert( sizeof( MiderPush ) == 144, "MiderPush: the layout of mider.slang's Params" );

// The push constants of dbparams.slang
struct DbParamsPush
{
  VkDeviceAddress cus, tus, ctus, slices, cuMapL, cuMapC, cuMotion, motion, lfp, ctuInfo;
  int32_t         numCus, widthInCtus, heightInCtus, log2Ctu, width, height, widthIn4, csx, csy, chroma, qpBdOffset, pass, flags;
  int32_t         vbPosHor[4], vbPosVer[4];   // the count, then the positions (in the shader, scalars: numVbHor, vbHor0..2...)
  uint32_t        lfpTag;
};
static_assert( sizeof( DbParamsPush ) == 168, "DbParamsPush: the layout of dbparams.slang's Params" );   // (the tag in the padding)

// The push constants of jobs.slang
struct JobsPush
{
  VkDeviceAddress cus, tus, ctus, slices, jobSlices, cuMapL, cuMapC, cuMotion, motion, tables, scaling, counts, blocks, trafo, inter, intra;
  int32_t         numCus, numBlocks, widthInCtus, heightInCtus, log2Ctu, width, height, widthIn4, chromaFormat, csx, csy, bitDepth, qpBdOffset,
                  tsMinQp;
  uint32_t        sps, ph;
  int32_t         pass, capTrafo, capInter, capIntra, num8x8;
  uint32_t        waveTag;
  VkDeviceAddress waveMap, progress, args;
};
static_assert( sizeof( JobsPush ) == 240, "JobsPush: the layout of jobs.slang's Params" );

// The push constants of intrasort.slang
struct IntraSortPush
{
  VkDeviceAddress jobs, count, keysA, keysB, hist, bandAt, bandSums, waveFirst, sorted, waits, flags;
  int32_t         numBlocks, pass, shift, capacity;
  VkDeviceAddress args;
};
static_assert( sizeof( IntraSortPush ) == 112, "IntraSortPush: the layout of intrasort.slang's Params" );

// The push constants of edges.slang
struct EdgesPush
{
  VkDeviceAddress lfp, slices, counts, lists, edges;
  int32_t         widthInCtus, heightInCtus, log2Ctu, width, height, chroma, csx, csy, pass, capacity;
  uint32_t        lfpTag;
  VkDeviceAddress args;
};
static_assert( sizeof( EdgesPush ) == 96, "EdgesPush: the layout of edges.slang's Params" );

// The size of a job in words, per job buffer (JobKind)
static constexpr size_t kJobWords[kJobKinds] = { sizeof( TrafoJob ) / sizeof( uint32_t ), sizeof( InterJob ) / sizeof( uint32_t ),
                                                 sizeof( IntraJob ) / sizeof( uint32_t ) };
// The entries (jobs, intra bands, edges) added to the room of the front end's lists, beyond what the picture can need
static constexpr size_t kJobBufferSpare = 64;
// The words of the passes' dispatch arguments and job counts (FrontPass)
static constexpr size_t kArgsWords = 64;
// The words of the front end's records per 4x4 block or CU:
static constexpr size_t kMotionWords   = 5;    // mider.slang's motion of a 4x4 block
static constexpr size_t kLfpWords      = 2;    // dbparams.slang's parameters of a 4x4 block in one direction
static constexpr size_t kCuMotionWords = 16;   // mider.slang's motion of a CU
static constexpr size_t kWaveMapWords  = 3;    // jobs.slang's wave map: per 4x4 luma area, a word in each component's plane
// The tags in the deblocking parameters and in the wave map run from 1 to kMaxTag, then start again at 1.
static constexpr uint32_t kMaxTag = 65535;
// intrasort.slang's passes (its pc.pass and kPass*): two digits of a radix sort by wave (HIST, SCAN, SCATTER), then the
// jobs' bands (BANDS, BSCAN, PLACE, WAITS); SMALL does all of it in one workgroup for up to 1024 jobs.
enum IntraSortPass
{
  kSortHist     = 0,
  kSortScan     = 1,
  kSortScatter  = 2,
  kSortBands    = 3,
  kSortBandScan = 4,
  kSortPlace    = 5,
  kSortWaits    = 6,
  kSortSmall    = 7,
};
// Where intrasort.slang's SMALL pass writes the dispatch arguments of its other passes in those words (its kArgs*):
// those of a workgroup per block of jobs, and those of a single workgroup
static constexpr size_t kSortArgsBlocks = 50;
static constexpr size_t kSortArgsSingle = 53;
// intrasort.slang's sizes: the jobs of a block (its kBlock: a workgroup sorts a block), the bins of a digit's
// histogram (the radix sort's digits have 8 bits), and the waves whose first band waveFirst holds (a wave past them
// sets a flag)
static constexpr size_t kSortBlockJobs = 1024;
static constexpr size_t kSortDigitBins = 256;
static constexpr size_t kSortMaxWaves  = 65536;
// The words of the edge lists' firsts and counts (four lists of 2 words, and one more)
static constexpr size_t kEdgeListsWords = 9;


// Returns the address of the picture's slice table (writeSliceTable).
static VkDeviceAddress sliceTable( const CodingStructure& cs )
{
  return hostAllocationAddress( cs.m_gpuSliceAllocation );
}

// Returns the bytes of the motion of the picture's 4x4 blocks.
static size_t motionBytes( const PreCalcValues& pcv )
{
  return size_t( pcv.sizeInCtus ) * pcv.num4x4CtuBlks * kMotionWords * sizeof( int32_t );
}

// Returns the bytes of the deblocking parameters of the picture's 4x4 blocks, in both directions.
static size_t lfpBytes( const PreCalcValues& pcv )
{
  return size_t( pcv.sizeInCtus ) * 2 * pcv.num4x4CtuBlks * kLfpWords * sizeof( uint32_t );
}

// Returns the timeline value of the latest first part of the picture's colocated pictures (0: none). The front end
// waits for it, because a colocated picture's inter prediction writes DMVR's refined motion into its colocated motion
// in its first part, and TMVP reads that motion.
static uint64_t colocatedFirstParts( const CodingStructure& cs )
{
  uint64_t value = 0;
  for( const Slice* slice: cs.picture->slices )
  {
    if( const Picture* col = colocatedPicture( *slice ) )
    {
      value = std::max( value, col->gpuFirstValue );
    }
  }
  return value;
}

// ====================================================================================================================
// The front end's submissions on the async compute queue
// ====================================================================================================================

void GpuRecon::recordFrontEndAsync( const CodingStructure& cs, const PictureJobs& jobs )
{
  m_ctx->api().vkResetCommandPool( m_ctx->device(), m_poolFrontEnd, 0 );
  m_cmd = m_cmdFrontEnd1;
  m_ctx->beginOneTime( m_cmd );
  // What the front ends before wrote, such as the colocated pictures' motion, becomes visible.
  fullBarrier();
  // Part 1 waits for the first parts of the picture's colocated pictures and, with the GPU's parse, for the picture's
  // parsed records.
  m_frontEndWait      = colocatedFirstParts( cs );
  m_frontEndParseWait = jobs.parsed ? jobs.parsed->parseValue() : 0;
  // Part 1 (motion and jobs) ends after the jobs (recordJobs calls splitFrontEnd); the picture's first part waits
  // for it. Part 2 holds the rest, which the picture's rest waits for.
  m_splitFrontEnd = true;
  recordFrontEnd( cs, jobs );
  m_splitFrontEnd = false;
  if( m_cmd == m_cmdFrontEnd1 )
  {
    splitFrontEnd();   // no jobs were recorded, so part 2 stays empty
  }
  m_ctx->api().vkEndCommandBuffer( m_cmd );
  m_ctx->submitFront( m_asyncQueue, m_cmdFrontEnd2, 0, m_frontEnd2Done );
  m_waitFrontEnd1 = m_frontEnd1Done;
  m_waitFrontEnd2 = m_frontEnd2Done;
}

void GpuRecon::splitFrontEnd()
{
  m_ctx->api().vkEndCommandBuffer( m_cmdFrontEnd1 );
  m_ctx->submitFront( m_asyncQueue, m_cmdFrontEnd1, m_frontEndWait, m_frontEnd1Done, m_frontEndParseWait );
  m_cmd = m_cmdFrontEnd2;
  m_ctx->beginOneTime( m_cmd );
  fullBarrier();   // what part 1 wrote becomes visible
}

// ====================================================================================================================
// The front end
// ====================================================================================================================

bool GpuRecon::recordFrontEnd( const CodingStructure& cs, const PictureJobs& jobs, const std::function<void()>& beside )
{
  CHECK( !cs.m_gpuSliceAllocation, "the GPU's motion derivation without the picture's slice table" );
  const size_t numCus = jobs.parsed ? useParsedRecords( cs, *jobs.parsed ) : uploadRecords( cs, jobs );
  if( numCus > 0 )
  {
    recordMotion( cs, numCus );
    recordJobs( cs, jobs, numCus );   // with the async queue, part 2 begins after the jobs
    recordDeblockParams( cs, numCus );
    // These run side by side, with one barrier after them all (recordEdges' first): the caller's work, the intra
    // jobs' waves (a wavefront of a few threads) and the edges' first pass.
    if( beside )
    {
      indirectBarrier();   // the caller's work reads the jobs' dispatch arguments
      beside();
    }
    recordJobWaves( cs, jobs, numCus );
    recordEdges( cs, jobs );
    if( jobs.intra || m_useFrontEnd )
    {
      recordIntraOrder( jobs );
    }
    if( !m_useFrontEnd )
    {
      copyJobsForCheck( jobs );
    }
    // The check's copies become visible to the shaders that follow.
    transferToComputeBarrier();
  }
  mark( "front end" );
  return numCus > 0 && beside;
}

// Copies the records of all the CTUs, CTU after CTU, to dst. The CPU writes them in order.
template<typename Record>
static void copyRecords( const std::vector<std::vector<Record>>& perCtu, uint8_t* dst )
{
  Record* to = reinterpret_cast<Record*>( dst );
  for( const auto& ctu: perCtu )
  {
    if( !ctu.empty() )
    {
      memcpy( to, ctu.data(), ctu.size() * sizeof( Record ) );
      to += ctu.size();
    }
  }
}

size_t GpuRecon::uploadRecords( const CodingStructure& cs, const PictureJobs& jobs )
{
  const std::vector<std::vector<CuRecord>>& cus    = *jobs.cus;
  const std::vector<std::vector<TuRecord>>& tus    = *jobs.tus;
  const size_t                              numCus = countPerCtu( cus );
  ensureFrontEndBuffers( cs, numCus, countPerCtu( tus ) );
  m_frontEnd.recordCus    = m_frontEnd.cus.addr;
  m_frontEnd.recordTus    = m_frontEnd.tus.addr;
  m_frontEnd.recordCtus   = m_frontEnd.ctus.addr;
  m_frontEnd.recordLevels = 0;   // the CPU's parse left the levels in the picture
  if( m_useFrontEnd )
  {
    m_frontEnd.capacities = frontCapacities( cs, cus, tus );
  }
  else
  {
    m_ctx->ensure( m_check.motion, motionBytes( *cs.pcv ), MemKind::HostCached );
  }
  copyRecords( cus, m_frontEnd.cus.mapped );
  copyRecords( tus, m_frontEnd.tus.mapped );
  writeCtuRecords( cs, cus, tus, reinterpret_cast<CtuRecord*>( m_frontEnd.ctus.mapped ) );
  return numCus;
}

size_t GpuRecon::useParsedRecords( const CodingStructure& cs, const ParsedPicture& parsed )
{
  // The job buffers' room as the parse added it up, and the records in raster order where the parse left them. The
  // front end's part 1 waits for the parse queue's timeline value that put them there.
  ensureFrontEndBuffers( cs, parsed.numCus(), parsed.numTus(), false );
  m_frontEnd.capacities = parsed.capacities();
  m_frontEnd.recordCus    = parsed.cus();
  m_frontEnd.recordTus    = parsed.tus();
  m_frontEnd.recordCtus   = parsed.ctus();
  m_frontEnd.recordLevels = parsed.levels();
  return parsed.numCus();
}

void GpuRecon::ensureFrontEndBuffers( const CodingStructure& cs, size_t numCus, size_t numTus, bool records )
{
  // The records are written by the CPU in order, or by the GPU's parse (its scatter). What only the GPU reads and
  // writes (the CU maps, the CUs' motion, the rows' progress) is in its own memory. Don't put those in host memory:
  // mider.slang's waits for each read become far longer (VideoQ 1080p took 0.6 s per picture).
  const PreCalcValues& pcv       = *cs.pcv;
  const int            widthIn4  = blocks4x4( pcv.lumaWidth );
  const int            heightIn4 = blocks4x4( pcv.lumaHeight );
  if( records )
  {
    m_ctx->ensure( m_frontEnd.cus, std::max<size_t>( numCus, 1 ) * sizeof( CuRecord ), MemKind::Upload );
    m_ctx->ensure( m_frontEnd.tus, std::max<size_t>( numTus, 1 ) * sizeof( TuRecord ), MemKind::Upload );
    m_ctx->ensure( m_frontEnd.ctus, pcv.sizeInCtus * sizeof( CtuRecord ), MemKind::Upload );
  }
  {
    std::lock_guard<std::mutex> lock( m_shared->frontEndMutex );
    m_shared->frontEnd.setChunkSize( RangeArena::chunkSizeFor( size_t( pcv.lumaWidth ) * pcv.lumaHeight ) );
  }
  m_ctx->ensure( m_frontEnd.cuMapLuma, size_t( widthIn4 ) * heightIn4 * sizeof( uint32_t ), MemKind::Device );
  m_ctx->ensure( m_frontEnd.cuMapChroma, size_t( widthIn4 ) * heightIn4 * sizeof( uint32_t ), MemKind::Device );
  takeFrontEnd( m_frontEnd.cuMotion, std::max<size_t>( numCus, 1 ) * kCuMotionWords * sizeof( int32_t ) );
  m_ctx->ensure( m_frontEnd.motion, motionBytes( pcv ), MemKind::Device );
  // mider.slang's progress: a word per CTU row and tile column, then the count of the segments taken
  m_ctx->ensure( m_frontEnd.rowProgress, ( size_t( pcv.heightInCtus ) * cs.pps->getNumTileColumns() + 1 ) * sizeof( uint32_t ), MemKind::Device );
}

void GpuRecon::takeFrontEnd( Buffer& buf, size_t size )
{
  if( buf.valid() )
  {
    CHECK_FATAL( buf.size < size, "Vulkan: a front end buffer taken twice for a picture, larger the second time" );
    return;
  }
  std::lock_guard<std::mutex> lock( m_shared->frontEndMutex );
  buf = m_shared->frontEnd.take( size );
  m_frontEndRanges.push_back( &buf );
}

void GpuRecon::giveFrontEnd()
{
  if( m_frontEndRanges.empty() )
  {
    return;
  }
  std::lock_guard<std::mutex> lock( m_shared->frontEndMutex );
  for( Buffer* buf: m_frontEndRanges )
  {
    m_shared->frontEnd.give( *buf );
  }
  m_frontEndRanges.clear();
  m_shared->frontEnd.tick();   // a tick per picture
}

// mider.slang's push constants for the picture: the records it reads, the maps and the motion it writes, the
// picture's geometry, the SPS's motion tools, and where the colocated motion goes
void GpuRecon::miderPush( const CodingStructure& cs, size_t numCus, MiderPush& push )
{
  const PreCalcValues& pcv = *cs.pcv;
  push                 = MiderPush{};
  push.cus             = m_frontEnd.recordCus;
  push.ctus            = m_frontEnd.recordCtus;
  push.slices          = sliceTable( cs );
  push.cuMap           = m_frontEnd.cuMapLuma.addr;
  push.cuMotion        = m_frontEnd.cuMotion.addr;
  push.motion          = m_frontEnd.motion.addr;
  push.progress        = m_frontEnd.rowProgress.addr;
  push.numCus          = int32_t( numCus );
  push.widthInCtus     = int32_t( pcv.widthInCtus );
  push.heightInCtus    = int32_t( pcv.heightInCtus );
  push.log2Ctu         = int32_t( pcv.maxCUWidthLog2 );
  push.width           = int32_t( pcv.lumaWidth );
  push.height          = int32_t( pcv.lumaHeight );
  push.widthIn4        = blocks4x4( pcv.lumaWidth );
  push.plevel          = int32_t( cs.sps->getLog2ParallelMergeLevelMinus2() + 2 );
  push.wpp             = cs.sps->getEntropyCodingSyncEnabledFlag() ? 1 : 0;
  push.maxNumMergeCand = int32_t( cs.sps->getMaxNumMergeCand() );
  push.subpics         = subpicRects( cs );
  push.subpicOn        = push.subpics != 0 ? 1 : 0;
  push.sps             = ( cs.sps->getSBTMVPEnabledFlag() ? 1 : 0 ) | ( cs.sps->getUseAffine() ? 2 : 0 ) | ( cs.sps->getUseAffineType() ? 4 : 0 );
  push.cuMapC          = m_frontEnd.cuMapChroma.addr;
  push.csx             = chromaShiftX( pcv.chrFormat );
  push.csy             = chromaShiftY( pcv.chrFormat );
  push.tileCols        = int32_t( cs.pps->getNumTileColumns() );
  // mider also writes the picture's colocated motion, except in the check mode, which copies the motion to the host.
  push.colOn           = m_useFrontEnd && cs.m_colMiAllocation ? 1 : 0;
  push.colMotion       = push.colOn ? static_cast<HostPlane*>( cs.m_colMiAllocation )->host.addr : 0;
}

void GpuRecon::recordMotion( const CodingStructure& cs, size_t numCus )
{
  const PreCalcValues& pcv = *cs.pcv;
  MiderPush            push;
  miderPush( cs, numCus, push );

  // The progress of the segments starts from zero.
  m_ctx->api().vkCmdFillBuffer( m_cmd, m_frontEnd.rowProgress.buffer, 0, VK_WHOLE_SIZE, 0 );
  transferToComputeBarrier();
  // First each 4x4 block's CU (mider_cumap: one workgroup per CU), then the motion (mider: one workgroup per CTU row of
  // each tile column).
  bind( kShader_mider_cumap );
  pushConstants( push );
  dispatchJobs( numCus );
  computeBarrier();
  // The lite variant leaves out GPM, affine and SbTMVP, for an SPS that turns them all off.
  const SPS& sps  = *cs.sps;
  const bool lite = !sps.getUseGeo() && !sps.getUseAffine() && !sps.getSBTMVPEnabledFlag();
  dispatch( lite ? kShader_mider_lite : kShader_mider, push, uint32_t( pcv.heightInCtus ) * uint32_t( push.tileCols ) );
  mark( "mider" );
  // In the check mode the 4x4 blocks' motion goes to host memory; otherwise mider wrote the colocated motion.
  memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT );
  if( !m_useFrontEnd )
  {
    VkBufferCopy region{ 0, 0, motionBytes( pcv ) };
    m_ctx->api().vkCmdCopyBuffer( m_cmd, m_frontEnd.motion.buffer, m_check.motion.buffer, 1, &region );
  }
}

// dbparams.slang's threads per CU in its per-CU passes (kThreadsPerCu), threads per workgroup, and workgroups per row
// of its dispatches
static constexpr size_t   kDbParamsThreadsPerCu = 8;
static constexpr size_t   kDbParamsGroupThreads = 64;
static constexpr uint32_t kDbParamsGroupsPerRow = 1024;

// Per CTU, its subpicture's index and whether loop filtering crosses the subpicture's boundaries (bit 16), uploaded
// into m_frontEnd.ctuInfo
void GpuRecon::uploadSubpicInfo( const CodingStructure& cs )
{
  const PreCalcValues& pcv = *cs.pcv;
  m_ctx->ensure( m_frontEnd.ctuInfo, pcv.sizeInCtus * sizeof( uint32_t ), MemKind::Upload );
  uint32_t* info = reinterpret_cast<uint32_t*>( m_frontEnd.ctuInfo.mapped );
  for( unsigned ctu = 0; ctu < pcv.sizeInCtus; ctu++ )
  {
    const SubPic& sub = cs.pps->getSubPicFromPos( ctuPosition( pcv, ctu ) );
    info[ctu]         = sub.getSubPicIdx() | ( sub.getloopFilterAcrossSubPicEnabledFlag() ? 1u << 16 : 0u );
  }
}

// Returns the picture's tag in the deblocking parameters. The parameters must read as zeroed at the start, as the CTU
// tasks' LF_INIT leaves them. Each entry carries the tag of the picture that wrote it, and an entry with another tag
// reads as zero. So the map needs zeroing (recorded here) only when it is new and when the tags wrap around. In the
// check mode it is zeroed for every picture, with tag 0, because the CPU compares the whole map.
uint32_t GpuRecon::nextLfpTag()
{
  FrontEndBuffers& fe = m_frontEnd;
  fe.lfpTag           = m_useFrontEnd ? fe.lfpTag % kMaxTag + 1 : 0;
  if( fe.lfpTag <= 1 || fe.lfp.buffer != fe.lfpZeroed )
  {
    m_ctx->api().vkCmdFillBuffer( m_cmd, fe.lfp.buffer, 0, VK_WHOLE_SIZE, 0 );
    transferToComputeBarrier();
    fe.lfpZeroed = fe.lfp.buffer;
  }
  return fe.lfpTag;
}

// dbparams.slang's push constants for the picture: the records, maps and motion it reads, the parameters it writes,
// the picture's geometry, and its loop filter flags and virtual boundaries
void GpuRecon::dbParamsPush( const CodingStructure& cs, size_t numCus, uint32_t lfpTag, DbParamsPush& push ) const
{
  const PreCalcValues& pcv = *cs.pcv;
  push              = DbParamsPush{};
  push.cus          = m_frontEnd.recordCus;
  push.tus          = m_frontEnd.recordTus;
  push.ctus         = m_frontEnd.recordCtus;
  push.slices       = sliceTable( cs );
  push.cuMapL       = m_frontEnd.cuMapLuma.addr;
  push.cuMapC       = m_frontEnd.cuMapChroma.addr;
  push.cuMotion     = m_frontEnd.cuMotion.addr;
  push.motion       = m_frontEnd.motion.addr;
  push.lfp          = m_frontEnd.lfp.addr;
  push.ctuInfo      = m_frontEnd.ctuInfo.addr;
  push.numCus       = int32_t( numCus );
  push.widthInCtus  = int32_t( pcv.widthInCtus );
  push.heightInCtus = int32_t( pcv.heightInCtus );
  push.log2Ctu      = int32_t( pcv.maxCUWidthLog2 );
  push.width        = int32_t( pcv.lumaWidth );
  push.height       = int32_t( pcv.lumaHeight );
  push.widthIn4     = blocks4x4( pcv.lumaWidth );
  push.csx          = chromaShiftX( pcv.chrFormat );
  push.csy          = chromaShiftY( pcv.chrFormat );
  push.chroma       = isChromaEnabled( pcv.chrFormat ) ? 1 : 0;
  push.qpBdOffset   = cs.sps->getQpBDOffset();
  push.lfpTag       = lfpTag;
  push.flags        = ( cs.pps->getLoopFilterAcrossSlicesEnabledFlag() ? 1 : 0 ) | ( cs.pps->getLoopFilterAcrossTilesEnabledFlag() ? 2 : 0 )
                      | ( cs.sps->getSubPicInfoPresentFlag() ? 4 : 0 );
  const PicHeader& ph = *cs.picHeader;
  if( ph.getVirtualBoundariesPresentFlag() )
  {
    push.vbPosHor[0] = int32_t( ph.getNumHorVirtualBoundaries() );
    push.vbPosVer[0] = int32_t( ph.getNumVerVirtualBoundaries() );
    for( unsigned i = 0; i < ph.getNumHorVirtualBoundaries(); i++ )
    {
      push.vbPosHor[1 + i] = int32_t( ph.getVirtualBoundariesPosY( i ) );
    }
    for( unsigned i = 0; i < ph.getNumVerVirtualBoundaries(); i++ )
    {
      push.vbPosVer[1 + i] = int32_t( ph.getVirtualBoundariesPosX( i ) );
    }
  }
}

void GpuRecon::recordDeblockParams( const CodingStructure& cs, size_t numCus )
{
  const PreCalcValues& pcv = *cs.pcv;
  m_ctx->ensure( m_frontEnd.lfp, lfpBytes( pcv ), MemKind::Device );
  if( !m_useFrontEnd )
  {
    m_ctx->ensure( m_check.lfp, lfpBytes( pcv ), MemKind::HostCached );
  }
  uploadSubpicInfo( cs );
  DbParamsPush push;
  dbParamsPush( cs, numCus, nextLfpTag(), push );
  // Pass 0 takes the luma tree's CUs, kDbParamsThreadsPerCu threads per CU. Then, when the SPS allows sub-block motion
  // or ISP, dbparams_sub takes the edges inside those CUs, one thread per 4x4 block. Pass 1 takes the chroma tree's
  // CUs, as pass 0 does. Each dispatch lays its workgroups out in rows of kDbParamsGroupsPerRow.
  const auto dispatchThreads = [&]( size_t threads ) {
    const uint32_t groups = uint32_t( ( threads + kDbParamsGroupThreads - 1 ) / kDbParamsGroupThreads );
    dispatchGroups( std::min<uint32_t>( groups, kDbParamsGroupsPerRow ), ( groups + kDbParamsGroupsPerRow - 1 ) / kDbParamsGroupsPerRow );
    computeBarrier();
  };
  const SPS& sps = *cs.sps;
  bind( kShader_dbparams );
  push.pass = 0;
  pushConstants( push );
  dispatchThreads( numCus * kDbParamsThreadsPerCu );
  if( sps.getUseAffine() || sps.getSBTMVPEnabledFlag() || sps.getUseISP() )
  {
    bind( kShader_dbparams_sub );
    pushConstants( push );
    dispatchThreads( size_t( push.widthIn4 ) * size_t( blocks4x4( pcv.lumaHeight ) ) );
    bind( kShader_dbparams );
  }
  push.pass = 1;
  pushConstants( push );
  dispatchThreads( numCus * kDbParamsThreadsPerCu );
  mark( "dbparams" );
  if( !m_useFrontEnd )
  {
    memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT );
    VkBufferCopy region{ 0, 0, lfpBytes( pcv ) };
    m_ctx->api().vkCmdCopyBuffer( m_cmd, m_frontEnd.lfp.buffer, m_check.lfp.buffer, 1, &region );
    m_check.lfpRecorded = true;
  }
}

// ====================================================================================================================
// Jobs and intra order
// ====================================================================================================================

void GpuRecon::allocateJobBuffers( const CodingStructure& cs, const PictureJobs& jobs, size_t numCus )
{
  const PreCalcValues& pcv = *cs.pcv;
  // The job buffers' room: with the front end, what the records can need (frontCapacities); in the check mode, the
  // CTU tasks' counts. Either way with kJobBufferSpare jobs to spare.
  const auto total = []( const auto* lists ) { return lists ? countPerCtu( *lists ) : size_t( 0 ); };
  const FrontCapacities& caps              = m_frontEnd.capacities;
  const size_t           counts[kJobKinds] = { m_useFrontEnd ? caps.trafo : total( jobs.trafo ), m_useFrontEnd ? caps.inter : total( jobs.inter ),
                                               m_useFrontEnd ? caps.intra : total( jobs.intra ) };
  if( !m_frontEnd.args.valid() )
  {
    m_frontEnd.args = m_ctx->alloc( kArgsWords * sizeof( uint32_t ), MemKind::Device );
  }
  for( int k = 0; k < kJobKinds; k++ )
  {
    m_frontEnd.jobCapacity[k] = counts[k] + kJobBufferSpare;
    takeFrontEnd( m_frontEnd.jobs[k], m_frontEnd.jobCapacity[k] * kJobWords[k] * sizeof( uint32_t ) );
    if( !m_useFrontEnd )
    {
      m_ctx->ensure( m_check.jobs[k], m_frontEnd.jobCapacity[k] * kJobWords[k] * sizeof( uint32_t ), MemKind::HostCached );
    }
  }
  if( !m_frontEnd.jobTables.valid() )
  {
    m_frontEnd.jobTables = m_ctx->alloc( kJobTableWords * sizeof( uint32_t ), MemKind::Upload );
    writeJobTables( reinterpret_cast<uint32_t*>( m_frontEnd.jobTables.mapped ) );
  }
  m_ctx->ensure( m_frontEnd.jobSlices, cs.picture->slices.size() * kJobSliceWords * sizeof( uint32_t ), MemKind::Upload );
  writeJobSlices( cs, *jobs.refs, reinterpret_cast<uint32_t*>( m_frontEnd.jobSlices.mapped ) );
  // The scaling lists' matrix offsets (setDequantOffsets), 0 where none were set
  m_ctx->ensure( m_frontEnd.jobScaling, kDequantOffsets * sizeof( uint32_t ), MemKind::Upload );
  uint32_t* scaling = reinterpret_cast<uint32_t*>( m_frontEnd.jobScaling.mapped );
  for( size_t i = 0; i < kDequantOffsets; i++ )
  {
    scaling[i] = i < m_dequantOffsets.size() ? m_dequantOffsets[i] : 0;
  }
  // Per CU its job counts, a word per job list; per block of kCusPerJobBlock CUs their sums, then the totals
  m_frontEnd.jobBlockCount = int( ( numCus + kCusPerJobBlock - 1 ) / kCusPerJobBlock );
  takeFrontEnd( m_frontEnd.jobCounts, std::max<size_t>( numCus, 1 ) * kJobLists * sizeof( uint32_t ) );
  takeFrontEnd( m_frontEnd.jobBlocks, ( size_t( m_frontEnd.jobBlockCount ) + 1 ) * kJobLists * sizeof( uint32_t ) );
  if( !m_useFrontEnd )
  {
    m_ctx->ensure( m_check.jobBlocks, ( size_t( m_frontEnd.jobBlockCount ) + 1 ) * kJobLists * sizeof( uint32_t ), MemKind::HostCached );
  }
  const int widthIn4  = blocks4x4( pcv.lumaWidth );
  const int heightIn4 = blocks4x4( pcv.lumaHeight );
  m_ctx->ensure( m_frontEnd.waveMap, size_t( widthIn4 ) * heightIn4 * kWaveMapWords * sizeof( uint32_t ), MemKind::Device );
  m_ctx->ensure( m_frontEnd.jobProgress, ( pcv.heightInCtus + 1 ) * sizeof( uint32_t ), MemKind::Device );
}

// The SPS's tools as jobs.slang's sps flags hold them
static uint32_t jobsSpsFlags( const SPS& sps )
{
  return ( sps.getUseMTS() ? 1u : 0u ) | ( sps.getUseImplicitMTS() ? 2u : 0u ) | ( sps.getUseIntraMTS() ? 4u : 0u ) | ( sps.getUseInterMTS() ? 8u : 0u )
         | ( sps.getUseLFNST() ? 16u : 0u ) | ( sps.getDisableScalingMatrixForLfnstBlks() ? 32u : 0u )
         | ( sps.getScalingMatrixForAlternativeColourSpaceDisabledFlag() ? 64u : 0u ) | ( sps.getScalingMatrixDesignatedColourSpaceFlag() ? 128u : 0u )
         | ( sps.getUseBIO() ? 256u : 0u ) | ( sps.getUseDMVR() ? 512u : 0u ) | ( sps.getUsePROF() ? 1024u : 0u ) | ( sps.getUseBcw() ? 2048u : 0u )
         | ( sps.getEntropyCodingSyncEnabledFlag() ? 4096u : 0u );
}

// The picture header's and the PPS's tools as jobs.slang's ph flags hold them, with whether the GPU makes the
// residuals (trafo) and the inter predictions (inter)
static uint32_t jobsPictureFlags( const PicHeader& ph, const PPS& pps, bool trafo, bool inter )
{
  return ( ph.getDisDmvrFlag() ? 1u : 0u ) | ( ph.getDisBdofFlag() ? 2u : 0u ) | ( ph.getDisProfFlag() ? 4u : 0u ) | ( ph.getJointCbCrSignFlag() ? 8u : 0u )
         | ( ph.getLmcsEnabledFlag() ? 16u : 0u ) | ( ph.getLmcsChromaResidualScaleFlag() ? 32u : 0u ) | ( pps.getUseWP() ? 64u : 0u )
         | ( pps.getWPBiPred() ? 128u : 0u ) | ( trafo ? 256u : 0u ) | ( inter ? 512u : 0u );
}

void GpuRecon::jobsPush( const CodingStructure& cs, const PictureJobs& jobs, size_t numCus, JobsPush& push ) const
{
  const PreCalcValues& pcv = *cs.pcv;
  const SPS&           sps = *cs.sps;
  push.cus          = m_frontEnd.recordCus;
  push.tus          = m_frontEnd.recordTus;
  push.ctus         = m_frontEnd.recordCtus;
  push.slices       = sliceTable( cs );
  push.jobSlices    = m_frontEnd.jobSlices.addr;
  push.cuMapL       = m_frontEnd.cuMapLuma.addr;
  push.cuMapC       = m_frontEnd.cuMapChroma.addr;
  push.cuMotion     = m_frontEnd.cuMotion.addr;
  push.motion       = m_frontEnd.motion.addr;
  push.tables       = m_frontEnd.jobTables.addr;
  push.scaling      = m_frontEnd.jobScaling.addr;
  push.counts       = m_frontEnd.jobCounts.addr;
  push.blocks       = m_frontEnd.jobBlocks.addr;
  push.trafo        = m_frontEnd.jobs[kTrafoJobs].addr;
  push.inter        = m_frontEnd.jobs[kInterJobs].addr;
  push.intra        = m_frontEnd.jobs[kIntraJobs].addr;
  push.numCus       = int32_t( numCus );
  push.numBlocks    = m_frontEnd.jobBlockCount;
  push.widthInCtus  = int32_t( pcv.widthInCtus );
  push.heightInCtus = int32_t( pcv.heightInCtus );
  push.log2Ctu      = int32_t( pcv.maxCUWidthLog2 );
  push.width        = int32_t( pcv.lumaWidth );
  push.height       = int32_t( pcv.lumaHeight );
  push.widthIn4     = blocks4x4( pcv.lumaWidth );
  push.chromaFormat = int32_t( pcv.chrFormat );
  push.csx          = chromaShiftX( pcv.chrFormat );
  push.csy          = chromaShiftY( pcv.chrFormat );
  push.bitDepth     = sps.getBitDepth();
  push.qpBdOffset   = sps.getQpBDOffset();
  push.tsMinQp      = 4 + 6 * sps.getInternalMinusInputBitDepth();
  push.sps          = jobsSpsFlags( sps );
  push.ph           = jobsPictureFlags( *cs.picHeader, *cs.pps, jobs.trafo || ( m_useFrontEnd && m_frontEndTrafo ),
                                        jobs.inter || ( m_useFrontEnd && m_frontEndInter ) );
  push.capTrafo     = int32_t( m_frontEnd.jobCapacity[kTrafoJobs] );
  push.capInter     = int32_t( m_frontEnd.jobCapacity[kInterJobs] );
  push.capIntra     = int32_t( m_frontEnd.jobCapacity[kIntraJobs] );
  push.num8x8   = int32_t( pcv.num8x8CtuBlks );
  push.waveMap  = m_frontEnd.waveMap.addr;
  push.waveTag  = m_frontEnd.waveTag;
  push.progress = m_frontEnd.jobProgress.addr;
  push.args     = m_frontEnd.args.addr;
}

void GpuRecon::recordJobs( const CodingStructure& cs, const PictureJobs& jobs, size_t numCus )
{
  allocateJobBuffers( cs, jobs, numCus );
  // Each entry of the wave map carries the tag of the picture that wrote it, and an entry with another tag reads as
  // wave 0. So the map needs zeroing only when it is new and when the tags wrap around.
  FrontEndBuffers& fe = m_frontEnd;
  fe.waveTag          = fe.waveTag % kMaxTag + 1;
  JobsPush push{};
  jobsPush( cs, jobs, numCus, push );

  // The wave map (when it needs it), the rows' progress and the passes' arguments start from zero.
  const Api& vk = m_ctx->api();
  if( fe.waveTag == 1 || fe.waveMap.buffer != fe.waveZeroed )
  {
    vk.vkCmdFillBuffer( m_cmd, fe.waveMap.buffer, 0, VK_WHOLE_SIZE, 0 );
    fe.waveZeroed = fe.waveMap.buffer;
  }
  vk.vkCmdFillBuffer( m_cmd, m_frontEnd.jobProgress.buffer, 0, VK_WHOLE_SIZE, 0 );
  vk.vkCmdFillBuffer( m_cmd, m_frontEnd.args.buffer, 0, VK_WHOLE_SIZE, 0 );
  transferToComputeBarrier();
  // Pass 0 counts each CU's jobs (jobs_count, the variant that only counts), pass 1 adds the counts up, and pass 2
  // writes the jobs. Passes 0 and 2 take one workgroup per block of kCusPerJobBlock CUs, pass 1 a single workgroup.
  for( int pass = 0; pass < 3; pass++ )
  {
    push.pass = pass;
    if( pass < 2 )
    {
      bind( pass == 0 ? kShader_jobs_count : kShader_jobs );
    }
    pushConstants( push );
    dispatchGroups( pass == 1 ? 1u : uint32_t( std::max( m_frontEnd.jobBlockCount, 1 ) ) );
    computeBarrier();
  }
  mark( "jobs" );
  if( m_splitFrontEnd )
  {
    splitFrontEnd();
  }
}

void GpuRecon::recordJobWaves( const CodingStructure& cs, const PictureJobs& jobs, size_t numCus )
{
  JobsPush push{};
  jobsPush( cs, jobs, numCus, push );
  // Pass 3 works out the intra jobs' waves (jobs_waves: one workgroup per CTU row). The barrier after it is the next
  // pass's.
  push.pass = 3;
  dispatch( kShader_jobs_waves, push, uint32_t( cs.pcv->heightInCtus ) );
}

// The room for the picture's intra bands: with the front end, what the records can need; in the check mode, the CTU
// tasks' jobs cut into bands as recordIntra cuts them.
size_t GpuRecon::intraBandCapacity( const PictureJobs& jobs ) const
{
  if( m_useFrontEnd )
  {
    return m_frontEnd.capacities.bands;
  }
  size_t bands = 0;
  for( const auto& ctu: *jobs.intra )
  {
    for( const IntraJob& job: ctu )
    {
      bands += job.bands();
    }
  }
  return bands;
}

// Takes the intra sort's buffers, with room for the intra job buffer's jobs (m_frontEnd.jobCapacity[kIntraJobs]) and
// for bands bands (kJobBufferSpare more for safety), and in the check mode the host buffers that compareFrontEnd
// reads. Then zeroes the sort's flags.
void GpuRecon::takeIntraSortBuffers( size_t bands )
{
  FrontEndBuffers& fe    = m_frontEnd;
  fe.sortCapacity        = bands + kJobBufferSpare;
  fe.sortBlocks          = int( ( fe.jobCapacity[kIntraJobs] + kSortBlockJobs - 1 ) / kSortBlockJobs );
  const size_t blockJobs = size_t( fe.sortBlocks ) * kSortBlockJobs;
  takeFrontEnd( fe.sortKeys, blockJobs * 2 * sizeof( uint32_t ) );   // two halves: the keys before and after a digit
  takeFrontEnd( fe.sortHistogram, size_t( fe.sortBlocks ) * kSortDigitBins * sizeof( uint32_t ) );
  takeFrontEnd( fe.bandAt, blockJobs * sizeof( uint32_t ) );
  takeFrontEnd( fe.bandSums, ( size_t( fe.sortBlocks ) + 1 ) * sizeof( uint32_t ) );
  m_ctx->ensure( fe.waveFirst, kSortMaxWaves * sizeof( uint32_t ), MemKind::Device );
  takeFrontEnd( fe.sortedBands, fe.sortCapacity * sizeof( IntraJob ) );
  takeFrontEnd( fe.bandWaits, fe.sortCapacity * sizeof( uint32_t ) );
  m_ctx->ensure( fe.sortFlags, 4 * sizeof( uint32_t ), MemKind::Device );
  if( !m_useFrontEnd )
  {
    m_ctx->ensure( m_check.sortedBands, fe.sortCapacity * sizeof( IntraJob ), MemKind::HostCached );
    m_ctx->ensure( m_check.bandWaits, fe.sortCapacity * sizeof( uint32_t ), MemKind::HostCached );
    m_ctx->ensure( m_check.sortInfo, 4 * sizeof( uint32_t ), MemKind::HostCached );
  }
  m_ctx->api().vkCmdFillBuffer( m_cmd, fe.sortFlags.buffer, 0, VK_WHOLE_SIZE, 0 );
  transferToComputeBarrier();
}

// intrasort.slang's push constants for the picture, except the pass and the digit's shift, which change per dispatch
void GpuRecon::intraSortPush( IntraSortPush& push ) const
{
  const FrontEndBuffers& fe = m_frontEnd;
  push                      = IntraSortPush{};
  push.jobs                 = fe.jobs[kIntraJobs].addr;
  push.count                = fe.jobBlocks.addr + ( size_t( fe.jobBlockCount ) * kJobLists + kListIntra ) * sizeof( uint32_t );   // jobs.slang's intra total
  push.args                 = fe.args.addr;
  push.keysA                = fe.sortKeys.addr;
  push.keysB                = fe.sortKeys.addr + size_t( fe.sortBlocks ) * kSortBlockJobs * sizeof( uint32_t );
  push.hist                 = fe.sortHistogram.addr;
  push.bandAt               = fe.bandAt.addr;
  push.bandSums             = fe.bandSums.addr;
  push.waveFirst            = fe.waveFirst.addr;
  push.sorted               = fe.sortedBands.addr;
  push.waits                = fe.bandWaits.addr;
  push.flags                = fe.sortFlags.addr;
  push.numBlocks            = fe.sortBlocks;
  push.capacity             = int32_t( fe.sortCapacity );
}

void GpuRecon::recordIntraOrder( const PictureJobs& jobs )
{
  takeIntraSortBuffers( intraBandCapacity( jobs ) );
  IntraSortPush push;
  intraSortPush( push );
  bind( kShader_intrasort );
  // SMALL sorts up to kSortBlockJobs jobs on its own. It also writes the dispatch arguments of the passes after it: no
  // workgroups when it sorted the jobs itself, else a workgroup per block (or a single one).
  push.pass  = kSortSmall;
  push.shift = 0;
  pushConstants( push );
  dispatchGroups( 1 );
  // With room for one block of jobs (most B pictures), SMALL sorts them all, so the other passes are not recorded.
  if( m_frontEnd.sortBlocks <= 1 )
  {
    computeBarrier();
    mark( "intrasort" );
    return;
  }
  indirectBarrier();
  const auto pass = [&]( IntraSortPass p, int shift, bool perBlock ) {
    push.pass  = p;
    push.shift = shift;
    pushConstants( push );
    m_ctx->api().vkCmdDispatchIndirect( m_cmd, m_frontEnd.args.buffer, ( perBlock ? kSortArgsBlocks : kSortArgsSingle ) * sizeof( uint32_t ) );
    computeBarrier();
  };
  // The radix sort by wave, two digits of 8 bits (the histograms, their sums, the scatter), then the bands
  for( int shift = 0; shift <= 8; shift += 8 )
  {
    pass( kSortHist, shift, true );
    pass( kSortScan, shift, false );
    pass( kSortScatter, shift, true );
  }
  pass( kSortBands, 0, true );
  pass( kSortBandScan, 0, false );
  pass( kSortPlace, 0, true );
  pass( kSortWaits, 0, true );
  mark( "intrasort" );
}

void GpuRecon::copyJobsForCheck( const PictureJobs& jobs )
{
  // The job buffers, their counts and the intra order are copied to host memory for compareFrontEnd.
  const Api& vk = m_ctx->api();
  memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT );
  for( int k = 0; k < kJobKinds; k++ )
  {
    VkBufferCopy region{ m_frontEnd.jobs[k].offset, 0, m_frontEnd.jobCapacity[k] * kJobWords[k] * sizeof( uint32_t ) };
    vk.vkCmdCopyBuffer( m_cmd, m_frontEnd.jobs[k].buffer, m_check.jobs[k].buffer, 1, &region );
  }
  VkBufferCopy blocks{ m_frontEnd.jobBlocks.offset, 0, ( size_t( m_frontEnd.jobBlockCount ) + 1 ) * kJobLists * sizeof( uint32_t ) };
  vk.vkCmdCopyBuffer( m_cmd, m_frontEnd.jobBlocks.buffer, m_check.jobBlocks.buffer, 1, &blocks );
  if( jobs.intra )
  {
    VkBufferCopy sorted{ m_frontEnd.sortedBands.offset, 0, m_frontEnd.sortCapacity * sizeof( IntraJob ) };
    vk.vkCmdCopyBuffer( m_cmd, m_frontEnd.sortedBands.buffer, m_check.sortedBands.buffer, 1, &sorted );
    VkBufferCopy waits{ m_frontEnd.bandWaits.offset, 0, m_frontEnd.sortCapacity * sizeof( uint32_t ) };
    vk.vkCmdCopyBuffer( m_cmd, m_frontEnd.bandWaits.buffer, m_check.bandWaits.buffer, 1, &waits );
    // The band count (the last of the band sums) and the flags go to their words of sortInfo.
    VkBufferCopy count{ m_frontEnd.bandSums.offset + size_t( m_frontEnd.sortBlocks ) * sizeof( uint32_t ),
                        FrontEndCheck::kSortInfoBandCount * sizeof( uint32_t ), sizeof( uint32_t ) };
    vk.vkCmdCopyBuffer( m_cmd, m_frontEnd.bandSums.buffer, m_check.sortInfo.buffer, 1, &count );
    VkBufferCopy flags{ 0, FrontEndCheck::kSortInfoFlags * sizeof( uint32_t ), sizeof( uint32_t ) };
    vk.vkCmdCopyBuffer( m_cmd, m_frontEnd.sortFlags.buffer, m_check.sortInfo.buffer, 1, &flags );
  }
  m_check.jobsRecorded = true;
}

// ====================================================================================================================
// Deblocking edges
// ====================================================================================================================

void GpuRecon::recordEdges( const CodingStructure& cs, const PictureJobs& jobs )
{
  if( !jobs.edges && !m_useFrontEnd )
  {
    return;
  }
  // The room for edges: with the front end, what the records can need; in the check mode, the CTU tasks' count with
  // kJobBufferSpare to spare.
  const PreCalcValues& pcv = *cs.pcv;
  size_t               cpu = 0;
  static const std::vector<EdgeLists> none;
  for( const EdgeLists& ctu: m_useFrontEnd ? none : *jobs.edges )
  {
    for( const auto& list: ctu.lists )
    {
      cpu += list.size();
    }
  }
  m_frontEnd.edgeCapacity = m_useFrontEnd ? m_frontEnd.capacities.edges : cpu + kJobBufferSpare;
  takeFrontEnd( m_frontEnd.edges, m_frontEnd.edgeCapacity * sizeof( Edge ) );
  // Per CTU, a count per edge list
  m_ctx->ensure( m_frontEnd.edgeCounts, size_t( pcv.sizeInCtus ) * 4 * sizeof( uint32_t ), MemKind::Device );
  m_ctx->ensure( m_frontEnd.edgeLists, kEdgeListsWords * sizeof( uint32_t ), MemKind::Device );
  if( !m_useFrontEnd )
  {
    m_ctx->ensure( m_check.edges, m_frontEnd.edgeCapacity * sizeof( Edge ), MemKind::HostCached );
    m_ctx->ensure( m_check.edgeLists, kEdgeListsWords * sizeof( uint32_t ), MemKind::HostCached );
  }
  EdgesPush push{};
  push.lfp          = m_frontEnd.lfp.addr;
  push.lfpTag       = m_frontEnd.lfpTag;
  push.slices       = sliceTable( cs );
  push.counts       = m_frontEnd.edgeCounts.addr;
  push.lists        = m_frontEnd.edgeLists.addr;
  push.edges        = m_frontEnd.edges.addr;
  push.widthInCtus  = int32_t( pcv.widthInCtus );
  push.heightInCtus = int32_t( pcv.heightInCtus );
  push.log2Ctu      = int32_t( pcv.maxCUWidthLog2 );
  push.width        = int32_t( pcv.lumaWidth );
  push.height       = int32_t( pcv.lumaHeight );
  push.chroma       = isChromaEnabled( pcv.chrFormat ) ? 1 : 0;
  push.csx          = chromaShiftX( pcv.chrFormat );
  push.csy          = chromaShiftY( pcv.chrFormat );
  push.capacity     = int32_t( m_frontEnd.edgeCapacity );
  push.args         = m_frontEnd.args.addr;
  // Pass 0 counts each CTU's edges, pass 1 adds the counts up, and pass 2 writes the edges. Passes 0 and 2 take one
  // workgroup per CTU, pass 1 a single workgroup.
  bind( kShader_edges );
  for( int pass = 0; pass < 3; pass++ )
  {
    push.pass = pass;
    pushConstants( push );
    dispatchGroups( pass == 1 ? 1u : uint32_t( pcv.sizeInCtus ) );
    computeBarrier();
  }
  mark( "edges" );
  if( m_useFrontEnd )
  {
    return;
  }
  // The edges, and each list's first edge and count, are copied to host memory for compareFrontEnd.
  memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT );
  VkBufferCopy edges{ m_frontEnd.edges.offset, 0, m_frontEnd.edgeCapacity * sizeof( Edge ) };
  m_ctx->api().vkCmdCopyBuffer( m_cmd, m_frontEnd.edges.buffer, m_check.edges.buffer, 1, &edges );
  VkBufferCopy lists{ 0, 0, kEdgeListsWords * sizeof( uint32_t ) };
  m_ctx->api().vkCmdCopyBuffer( m_cmd, m_frontEnd.edgeLists.buffer, m_check.edgeLists.buffer, 1, &lists );
  m_check.edgesRecorded = true;
}

// ====================================================================================================================
// The buffers
// ====================================================================================================================

void GpuRecon::FrontEndBuffers::release( Context* ctx )
{
  for( Buffer* buf: { &cus, &tus, &ctus, &cuMapLuma, &cuMapChroma, &cuMotion, &motion, &rowProgress, &ctuInfo, &lfp, &jobTables, &jobSlices,
                      &jobScaling, &jobCounts, &jobBlocks, &jobs[kTrafoJobs], &jobs[kInterJobs], &jobs[kIntraJobs], &waveMap, &jobProgress, &args,
                      &sortKeys, &sortHistogram, &bandAt, &bandSums, &waveFirst, &sortedBands, &bandWaits, &sortFlags, &edgeCounts, &edgeLists, &edges } )
  {
    ctx->free( *buf );
  }
}

void GpuRecon::FrontEndCheck::release( Context* ctx )
{
  for( Buffer* buf: { &motion, &lfp, &jobBlocks, &jobs[kTrafoJobs], &jobs[kInterJobs], &jobs[kIntraJobs], &sortedBands, &bandWaits, &sortInfo, &edges,
                      &edgeLists } )
  {
    ctx->free( *buf );
  }
}

}   // namespace gpu
}   // namespace vvdec
