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

/** \file     GpuRecon.cpp
    \brief    GpuRecon's construction, commands and submissions, the picture's planes and tables, and the order of a
              picture's work (see GpuRecon.h)
*/

#include "GpuRecon.h"


#include "CommonLib/AdaptiveLoopFilter.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/MipData.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Rom.h"
#include "CommonLib/SampleAdaptiveOffset.h"
#include "CommonLib/Slice.h"

#include <chrono>
#include <cstring>
#include <utility>

namespace vvdec
{
namespace gpu
{

// Dispatches of one workgroup per job lay their workgroups out in rows of kGroupsPerRow, so a job shader finds its job
// at x + kGroupsPerRow * y of its workgroup (dispatchJobs; common.slang's workgroupIndex does the same).
static constexpr size_t kGroupsPerRow = 32768;
// The first word of the passes' job counts in the front end's argument buffer (see FrontPass; common.slang's
// kFrontCounts must match it).
static constexpr size_t kFrontCounts = 32;

// ====================================================================================================================
// The constant tables
// ====================================================================================================================

// The transform matrices, in the order trafo.slang expects: DCT-II of 2 to 64 points, then DST-VII and DCT-VIII of
// 4 to 32 points.
static Buffer uploadTransformMatrices( Context* ctx )
{
  const std::pair<const TMatrixCoeff*, int> matrices[] = {
    { g_trCoreDCT2P2[0], 2 },  { g_trCoreDCT2P4[0], 4 },  { g_trCoreDCT2P8[0], 8 },  { g_trCoreDCT2P16[0], 16 }, { g_trCoreDCT2P32[0], 32 },
    { g_trCoreDCT2P64[0], 64 }, { g_trCoreDST7P4[0], 4 },  { g_trCoreDST7P8[0], 8 },  { g_trCoreDST7P16[0], 16 }, { g_trCoreDST7P32[0], 32 },
    { g_trCoreDCT8P4[0], 4 },  { g_trCoreDCT8P8[0], 8 },  { g_trCoreDCT8P16[0], 16 }, { g_trCoreDCT8P32[0], 32 },
  };
  size_t count = 0;
  for( const auto& matrix: matrices )
  {
    count += size_t( matrix.second ) * matrix.second;
  }
  CHECK_FATAL( count != 8180, "Vulkan: the transform matrices are not those of shaders/trafo.slang" );
  Buffer   buf = ctx->alloc( count * sizeof( int16_t ), MemKind::Upload );
  int16_t* to  = reinterpret_cast<int16_t*>( buf.mapped );
  for( const auto& matrix: matrices )
  {
    for( int i = 0; i < matrix.second * matrix.second; i++ )
    {
      *to++ = int16_t( matrix.first[i] );
    }
  }
  return buf;
}

// LFNST's matrices, in the order trafo.slang expects: 8x8, then 4x4.
static Buffer uploadLfnstMatrices( Context* ctx )
{
  static_assert( sizeof( g_lfnst8x8 ) == 6144 && sizeof( g_lfnst4x4 ) == 2048, "LFNST's matrices are not those of shaders/trafo.slang" );
  Buffer buf = ctx->alloc( sizeof( g_lfnst8x8 ) + sizeof( g_lfnst4x4 ), MemKind::Upload );
  memcpy( buf.mapped, g_lfnst8x8, sizeof( g_lfnst8x8 ) );
  memcpy( buf.mapped + sizeof( g_lfnst8x8 ), g_lfnst4x4, sizeof( g_lfnst4x4 ) );
  return buf;
}

// MIP's matrices, in the order intra.slang expects: 4x4, 8x8, 16x16.
static Buffer uploadMipMatrices( Context* ctx )
{
  static_assert( sizeof( mipMatrix4x4 ) == 1024 && sizeof( mipMatrix8x8 ) == 1024 && sizeof( mipMatrix16x16 ) == 2688,
                 "MIP's matrices are not those of shaders/intra.slang" );
  Buffer buf = ctx->alloc( sizeof( mipMatrix4x4 ) + sizeof( mipMatrix8x8 ) + sizeof( mipMatrix16x16 ), MemKind::Upload );
  memcpy( buf.mapped, mipMatrix4x4, sizeof( mipMatrix4x4 ) );
  memcpy( buf.mapped + sizeof( mipMatrix4x4 ), mipMatrix8x8, sizeof( mipMatrix8x8 ) );
  memcpy( buf.mapped + sizeof( mipMatrix4x4 ) + sizeof( mipMatrix8x8 ), mipMatrix16x16, sizeof( mipMatrix16x16 ) );
  return buf;
}

// GPM's prestored weight masks (g_globalGeoWeights), for inter.slang.
static Buffer uploadGeoWeights( Context* ctx )
{
  static_assert( sizeof( g_globalGeoWeights ) == GEO_NUM_PRESTORED_MASK * GEO_WEIGHT_MASK_SIZE * GEO_WEIGHT_MASK_SIZE * sizeof( int16_t ) && GEO_WEIGHT_MASK_SIZE == 112,
                 "GPM's masks are not those of shaders/inter.slang" );
  Buffer buf = ctx->alloc( sizeof( g_globalGeoWeights ), MemKind::Upload );
  memcpy( buf.mapped, g_globalGeoWeights, sizeof( g_globalGeoWeights ) );
  return buf;
}

// ====================================================================================================================
// Construction
// ====================================================================================================================

GpuRecon::GpuRecon( Context* context, SharedReconBuffers* shared ) : m_ctx( context ), m_shared( shared )
{
  CHECK_FATAL( !shared, "Vulkan: a GpuRecon without the buffers the decoder's instances share" );
  // the compute queue's commands: a picture's first part, its rest and its early rows
  m_pool = m_ctx->createCommandPool( m_ctx->queueFamily() );
  VkCommandBuffer cmds[3];
  m_ctx->allocateCommandBuffers( m_pool, 3, cmds );
  m_cmdFirst = m_cmd = cmds[0];
  m_cmdRest          = cmds[1];
  m_cmdEarlyRows     = cmds[2];
  m_fence            = m_ctx->createFence();
  m_fenceFirst       = m_ctx->createFence();

  // the front end's commands on an async compute queue
  if( m_ctx->asyncQueue() )
  {
    m_poolFrontEnd = m_ctx->createCommandPool( m_ctx->asyncFamily() );
    VkCommandBuffer frontEndCmds[2];
    m_ctx->allocateCommandBuffers( m_poolFrontEnd, 2, frontEndCmds );
    m_cmdFrontEnd1  = frontEndCmds[0];
    m_cmdFrontEnd2  = frontEndCmds[1];
    m_frontEnd1Done = m_ctx->createSemaphore();
    m_frontEnd2Done = m_ctx->createSemaphore();
    m_asyncQueue    = m_ctx->takeAsyncQueue();
  }

  // the downloads' commands on the copy queue
  if( m_ctx->transferQueue() )
  {
    for( Download& download: m_downloads )
    {
      download.pool = m_ctx->createCommandPool( m_ctx->transferFamily() );
      m_ctx->allocateCommandBuffers( download.pool, 1, &download.cmd );
      download.semaphore = m_ctx->createSemaphore();
      download.fence     = m_ctx->createFence();
    }
  }

  m_tables.transform  = uploadTransformMatrices( m_ctx );
  m_tables.lfnst      = uploadLfnstMatrices( m_ctx );
  m_tables.mip        = uploadMipMatrices( m_ctx );
  m_tables.geoWeights = uploadGeoWeights( m_ctx );

  if( m_ctx->profiling() )
  {
    m_queries          = m_ctx->createTimestampQueries( kMaxMarks );
    m_earlyRowsQueries = m_ctx->createTimestampQueries( kEarlyRowsMarks );
    if( m_ctx->transferQueue() && m_ctx->transferTimestamps() )
    {
      for( Download& download: m_downloads )
      {
        download.queries = m_ctx->createTimestampQueries( 2 );
      }
    }
  }
}

GpuRecon::~GpuRecon()
{
  m_ctx->destroyQueries( m_queries );
  m_ctx->destroyQueries( m_earlyRowsQueries );
  for( Download& download: m_downloads )
  {
    m_ctx->destroyQueries( download.queries );
    m_ctx->destroyFence( download.fence );
    m_ctx->destroySemaphore( download.semaphore );
    m_ctx->destroyCommandPool( download.pool );
  }
  m_ctx->destroyFence( m_fence );
  m_ctx->destroyFence( m_fenceFirst );
  m_ctx->destroySemaphore( m_frontEnd1Done );
  m_ctx->destroySemaphore( m_frontEnd2Done );
  m_ctx->destroyCommandPool( m_poolFrontEnd );
  m_ctx->destroyCommandPool( m_pool );
  for( Buffer* buf: { &m_params, &m_refs, &m_dmvr, &m_intraSync, &m_subpics, &m_lmcs, &m_defer, &m_saoRing, &m_scaling, &m_tables.transform, &m_tables.lfnst,
                      &m_tables.mip, &m_tables.geoWeights, &m_cpuJobs.trafo, &m_cpuJobs.inter, &m_cpuJobs.intra, &m_cpuJobs.earlyTrafo,
                      &m_cpuJobs.earlyInter, &m_cpuJobs.edges } )
  {
    m_ctx->free( *buf );
  }
  for( Buffer& buf: m_scratch )
  {
    m_ctx->free( buf );
  }
  giveFrontEnd();   // (its ranges first: release frees buffers of their own)
  m_frontEnd.release( m_ctx );
  m_check.release( m_ctx );
  m_parse.release( m_ctx );
}

// ====================================================================================================================
// Commands and submissions
// ====================================================================================================================

void GpuRecon::beginOneTime( VkCommandBuffer cmd )
{
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  m_ctx->api().vkBeginCommandBuffer( cmd, &begin );
}

void GpuRecon::beginCommands( bool resetPool )
{
  const Api& vk = m_ctx->api();
  if( resetPool )
  {
    vk.vkResetCommandPool( m_ctx->device(), m_pool, 0 );
  }
  beginOneTime( m_cmd );

  m_marks   = 0;
  m_gapMark = 0;
  if( m_queries )
  {
    vk.vkCmdResetQueryPool( m_cmd, m_queries, 0, kMaxMarks );
    vk.vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queries, m_marks++ );
  }

  // What earlier submissions wrote (the reference pictures) must be visible to this one. Not after the picture's
  // early rows (resetPool false): their submission began with this barrier, and the rest of the first part neither
  // reads nor writes what the early rows or the work submitted since write (the early rows' tiles; other pictures'
  // planes and scratch planes), so the GPU may start it while the early rows are still running (VideoQ 1080p +0.5%).
  if( resetPool )
  {
    fullBarrier();
  }
}

void GpuRecon::splitCommands()
{
  const Api& vk = m_ctx->api();

  // The first part's writes, DMVR's refinements in host memory among them, become visible to the CPU.
  hostReadBarrier();
  vk.vkEndCommandBuffer( m_cmd );
  // Submitted at once: the rest is recorded while it runs.
  vk.vkResetFences( m_ctx->device(), 1, &m_fenceFirst );
  m_submitted     = std::chrono::steady_clock::now();
  m_firstValue     = m_ctx->submitFirst( m_cmdFirst, m_fenceFirst, m_waitFrontEnd1, m_firstParseWait );
  m_waitFrontEnd1  = VK_NULL_HANDLE;
  m_firstParseWait = 0;

  // The rest, after what the first part wrote
  m_cmd = m_cmdRest;
  beginOneTime( m_cmd );
  fullBarrier();
  // In the GPU profile, the time until the rest begins (other pictures' work may run between the parts) counts on
  // its own, not as the picture's or its next pass's.
  if( m_queries )
  {
    mark( "between the parts" );
    m_gapMark = m_marks - 1;
  }
}

int GpuRecon::submitPicture( const PelStorage& pic, bool waitFirst )
{
  const Api& vk = m_ctx->api();

  // A picture with stand-ins of its planes (the GPU parsed it) is downloaded only when it is about to be output
  // (PlanePool::startDownload), from its twins. Otherwise, without a copy queue, the download ends the rest's
  // commands. The rest's writes become visible to the CPU (and to the copies).
  const bool downloads = hostPictureAddresses( pic, nullptr, nullptr );
  if( downloads && !m_ctx->transferQueue() )
  {
    copyPlanes( pic, kTwin, kHost );
  }
  hostReadBarrier();
  vk.vkEndCommandBuffer( m_cmd );

  int turn = -1;
  if( downloads && m_ctx->transferQueue() )
  {
    turn = m_download;
    m_download ^= 1;
    recordDownload( pic, turn );
  }
  const Download* download = turn >= 0 ? &m_downloads[turn] : nullptr;

  vk.vkResetFences( m_ctx->device(), 1, &m_fence );
  m_ctx->submitRest( m_cmdRest, m_fence, download ? download->semaphore : VK_NULL_HANDLE, download ? download->cmd : VK_NULL_HANDLE,
                     download ? download->fence : VK_NULL_HANDLE, m_waitFrontEnd2 );
  m_waitFrontEnd2 = VK_NULL_HANDLE;
  if( waitFirst )
  {
    const VkResult result = vk.vkWaitForFences( m_ctx->device(), 1, &m_fenceFirst, VK_TRUE, UINT64_MAX );
    if( result != VK_SUCCESS )
    {
      THROW_FATAL( "Vulkan: waiting for the GPU failed (" << result << ")" );
    }
  }
  return turn;
}

void GpuRecon::recordDownload( const PelStorage& pic, int turn )
{
  // The copies of the picture's planes on the GPU to host memory, in the commands of the download whose turn it is
  // (the other download may still be running).
  const Api& vk       = m_ctx->api();
  Download&  download = m_downloads[turn];
  CHECK_FATAL( download.pending.load( std::memory_order_acquire ), "Vulkan: a download not waited for (endDownload) before its commands are reused" );
  vk.vkResetCommandPool( m_ctx->device(), download.pool, 0 );
  beginOneTime( download.cmd );
  if( download.queries )
  {
    vk.vkCmdResetQueryPool( download.cmd, download.queries, 0, 2 );
    vk.vkCmdWriteTimestamp( download.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, download.queries, 0 );
  }
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    const HostPlane* plane  = hostPlane( pic, ComponentID( comp ) );
    const GpuLayout  layout = gpuLayout( pic, ComponentID( comp ) );
    VkBufferCopy     region{ 0, layout.hostOffset, layout.size };
    vk.vkCmdCopyBuffer( download.cmd, planeTwin( pic, ComponentID( comp ) ).buffer, plane->host.buffer, 1, &region );
  }
  memoryBarrier( download.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT );
  if( download.queries )
  {
    vk.vkCmdWriteTimestamp( download.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, download.queries, 1 );
  }
  vk.vkEndCommandBuffer( download.cmd );
  vk.vkResetFences( m_ctx->device(), 1, &download.fence );
  download.pending.store( true, std::memory_order_release );
}

void GpuRecon::endCompute()
{
  const Api&     vk     = m_ctx->api();
  const VkResult result = vk.vkWaitForFences( m_ctx->device(), 1, &m_fence, VK_TRUE, UINT64_MAX );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: waiting for the GPU failed (" << result << ")" );
  }
  readTimestamps( std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - m_submitted ).count() );
  giveFrontEnd();   // (the rest done: so are the front end's parts, which the picture's parts wait for)
}

void GpuRecon::endDownload( int turn )
{
  if( turn < 0 )
  {
    return;
  }
  const Api&     vk       = m_ctx->api();
  Download&      download = m_downloads[turn];
  const VkResult result   = vk.vkWaitForFences( m_ctx->device(), 1, &download.fence, VK_TRUE, UINT64_MAX );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: waiting for the GPU failed (" << result << ")" );
  }
  if( download.queries )
  {
    uint64_t stamps[2];
    vk.vkGetQueryPoolResults( m_ctx->device(), download.queries, 0, 2, sizeof( stamps ), stamps, sizeof( uint64_t ),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT );
    m_ctx->addTime( "download (copies)", double( stamps[1] - stamps[0] ) * m_ctx->timestampPeriod() * 1e-6 );
  }
  download.pending.store( false, std::memory_order_release );
}

void GpuRecon::mark( const char* label )
{
  if( m_cmd == m_cmdFrontEnd1 || m_cmd == m_cmdFrontEnd2 )
  {
    return;   // the async queue's time overlaps the compute queue's: not counted
  }
  if( m_cmd == m_cmdEarlyRows )
  {
    if( m_earlyRowsQueries && m_earlyRowsMarks < kEarlyRowsMarks )
    {
      m_earlyRowsLabels[m_earlyRowsMarks] = label;
      m_ctx->api().vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_earlyRowsQueries, m_earlyRowsMarks++ );
    }
    return;
  }
  if( m_queries && m_marks < kMaxMarks )
  {
    m_labels[m_marks] = label;
    m_ctx->api().vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queries, m_marks++ );
  }
}

void GpuRecon::readTimestamps( double waited )
{
  const Api& vk = m_ctx->api();
  if( !m_queries || m_marks <= 1 )
  {
    return;
  }
  m_ctx->addTime( "wait (CPU)", waited );
  uint64_t stamps[kMaxMarks];
  vk.vkGetQueryPoolResults( m_ctx->device(), m_queries, 0, m_marks, sizeof( stamps ), stamps, sizeof( uint64_t ),
                            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT );
  const double toMs = m_ctx->timestampPeriod() * 1e-6;
  // The early rows' passes count with the rest's passes of the same name: a picture's trafo or inter time is its
  // early rows' plus its rest's.
  double   earlyRows[kEarlyRowsMarks] = {};
  uint64_t earlyRowsSpan              = 0;
  if( m_earlyRowsQueries && m_earlyRowsMarks > 1 )
  {
    uint64_t earlyStamps[kEarlyRowsMarks];
    vk.vkGetQueryPoolResults( m_ctx->device(), m_earlyRowsQueries, 0, m_earlyRowsMarks, sizeof( earlyStamps ), earlyStamps, sizeof( uint64_t ),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT );
    for( uint32_t j = 1; j < m_earlyRowsMarks; j++ )
    {
      earlyRows[j] = double( earlyStamps[j] - earlyStamps[j - 1] ) * toMs;
    }
    earlyRowsSpan = earlyStamps[m_earlyRowsMarks - 1] - earlyStamps[0];
  }
  for( uint32_t i = 1; i < m_marks; i++ )
  {
    double time = double( stamps[i] - stamps[i - 1] ) * toMs;
    for( uint32_t j = 1; j < m_earlyRowsMarks; j++ )
    {
      if( earlyRows[j] > 0 && strcmp( m_earlyRowsLabels[j], m_labels[i] ) == 0 )
      {
        time += earlyRows[j];
        earlyRows[j] = 0;
      }
    }
    m_ctx->addTime( m_labels[i], time );
  }
  for( uint32_t j = 1; j < m_earlyRowsMarks; j++ )
  {
    if( earlyRows[j] > 0 )
    {
      m_ctx->addTime( m_earlyRowsLabels[j], earlyRows[j] );
    }
  }
  const uint64_t between = m_gapMark > 0 ? stamps[m_gapMark] - stamps[m_gapMark - 1] : 0;
  m_ctx->addTime( "submission (GPU)", double( stamps[m_marks - 1] - stamps[0] - between + earlyRowsSpan ) * toMs );
}

// --- Barriers ---

void GpuRecon::memoryBarrier( VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
                              VkAccessFlags dstAccess )
{
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = srcAccess;
  barrier.dstAccessMask = dstAccess;
  m_ctx->api().vkCmdPipelineBarrier( cmd, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr );
}

void GpuRecon::memoryBarrier( VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess )
{
  memoryBarrier( m_cmd, srcStage, srcAccess, dstStage, dstAccess );
}

void GpuRecon::fullBarrier()
{
  memoryBarrier( VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT );
}

void GpuRecon::computeBarrier()
{
  memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT );
}

void GpuRecon::transferToComputeBarrier()
{
  memoryBarrier( VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT );
}

void GpuRecon::computeToTransferBarrier()
{
  memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT );
}

void GpuRecon::hostReadBarrier()
{
  memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT );
}

void GpuRecon::indirectBarrier()
{
  memoryBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT );
}

// --- Dispatches ---

void GpuRecon::bind( int shader )
{
  m_ctx->api().vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( shader ) );
}

void GpuRecon::pushConstants( const void* data, size_t size )
{
  m_ctx->api().vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, uint32_t( size ), data );
}

void GpuRecon::dispatchGroups( uint32_t groupsX, uint32_t groupsY )
{
  m_ctx->api().vkCmdDispatch( m_cmd, groupsX, groupsY, 1 );
}

void GpuRecon::dispatchJobs( size_t count )
{
  dispatchGroups( uint32_t( std::min<size_t>( count, kGroupsPerRow ) ), uint32_t( ( count + kGroupsPerRow - 1 ) / kGroupsPerRow ) );
}

void GpuRecon::dispatchIndirect( FrontPass pass )
{
  m_ctx->api().vkCmdDispatchIndirect( m_cmd, m_frontEnd.args.buffer, VkDeviceSize( 3 * pass ) * sizeof( uint32_t ) );
}

VkDeviceAddress GpuRecon::frontEndCount( FrontPass pass ) const
{
  return m_frontEnd.args.addr + VkDeviceSize( kFrontCounts + 2 * pass ) * sizeof( uint32_t );
}

// ====================================================================================================================
// The picture's planes, tables and parameters
// ====================================================================================================================

GpuRecon::Planes GpuRecon::hostPlanes( const PelStorage& pic )
{
  // (none for a picture with stand-ins of its planes: the GPU parsed it, so no pass reads its planes in host memory)
  Planes planes{};
  if( !hostPictureAddresses( pic, planes.addr, planes.stride ) )
  {
    return Planes{};
  }
  return planes;
}

GpuRecon::Planes GpuRecon::twinPlanes( const PelStorage& pic )
{
  Planes planes{};
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    Buffer& twin = planeTwin( pic, ComponentID( comp ) );
    if( !twin.valid() )
    {
      twin = m_ctx->allocPooled( size_t( gpuLayout( pic, ComponentID( comp ) ).size ) );
    }
    planes.addr[comp]   = twin.addr + gpuLayout( pic, ComponentID( comp ) ).origin;
    planes.stride[comp] = int32_t( pic.bufs[comp].stride );
  }
  return planes;
}

GpuRecon::Planes GpuRecon::scratchPlanes( const PelStorage& pic )
{
  Planes planes{};
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    const GpuLayout layout = gpuLayout( pic, ComponentID( comp ) );
    if( !m_scratch[comp].valid() )
    {
      m_scratch[comp] = m_ctx->allocPooled( size_t( layout.size ) );
    }
    planes.addr[comp]   = m_scratch[comp].addr + layout.origin;
    planes.stride[comp] = int32_t( pic.bufs[comp].stride );
  }
  return planes;
}

void GpuRecon::copyPlanes( const PelStorage& pic, int from, int to )
{
  const Api& vk = m_ctx->api();
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    const Buffer*    host    = from == kHost || to == kHost ? &hostPlane( pic, ComponentID( comp ) )->host : nullptr;
    const Buffer*    bufs[3] = { host, &planeTwin( pic, ComponentID( comp ) ), &m_scratch[comp] };
    const GpuLayout  layout  = gpuLayout( pic, ComponentID( comp ) );
    VkBufferCopy     region{ from == kHost ? layout.hostOffset : 0, to == kHost ? layout.hostOffset : 0, layout.size };
    vk.vkCmdCopyBuffer( m_cmd, bufs[from]->buffer, bufs[to]->buffer, 1, &region );
  }
  mark( from == kHost ? "upload" : ( to == kHost ? "download" : "copy" ) );
}

// A reference picture as inter.slang reads it (RefPlanes)
struct RefEntry
{
  VkDeviceAddress y, cb, cr;
  uint32_t        size, strides;
  int32_t         ratioX, ratioY, winLeft, winTop, scaled, unused[3];
};
static_assert( sizeof( RefEntry ) == 64, "the reference table laid out differently from shaders/inter.slang's" );

void GpuRecon::writeRefTable( const CodingStructure& cs, const PictureJobs& jobs, const PelStorage& pic, std::vector<const Picture*>& uploads )
{
  // A reference picture the GPU parsed gets its planes for its output, and gives them back, on other threads
  // (PlanePool), which moves its twins from one set of plane records to the other. Its records are read under the
  // pool's lock, or a read could find them half swapped: a wrong address for the GPU, or a layout check failing.
  std::unique_lock<std::mutex> lock;
  if( jobs.planes )
  {
    lock = jobs.planes->lockPlanes();
  }
  const RefTable& refs = *jobs.refs;
  const size_t    size = std::max<size_t>( refs.pictures().size(), 1 ) * sizeof( RefEntry );
  if( m_refs.size < size )
  {
    m_ctx->free( m_refs );
    m_refs = m_ctx->alloc( std::max<size_t>( size, 1024 ), MemKind::Upload );
  }
  RefEntry* table = reinterpret_cast<RefEntry*>( m_refs.mapped );
  for( const void* ref: refs.pictures() )
  {
    const Picture* refPic = static_cast<const Picture*>( ref );
    const Planes   planes = twinPlanes( refPic->m_bufs[PIC_RECONSTRUCTION] );
    RefEntry       entry{};
    entry.y       = planes.addr[COMPONENT_Y];
    entry.cb      = planes.addr[COMPONENT_Cb];
    entry.cr      = planes.addr[COMPONENT_Cr];
    entry.size    = uint32_t( refPic->lwidth() ) | uint32_t( refPic->lheight() ) << 16;
    entry.strides = uint32_t( planes.stride[COMPONENT_Y] ) | uint32_t( planes.stride[COMPONENT_Cb] ) << 16;
    // RPR: the scaling ratio of the slices' reference (the same for every slice of the picture), and the
    // reference's scaling window (xPredInterBlkRPR)
    entry.scaled = refPic->isRefScaled( cs.pps.get() ) ? 1 : 0;
    entry.ratioX = entry.ratioY = 1 << SCALE_RATIO_BITS;
    for( const Slice* slice: cs.picture->slices )
    {
      for( int list = 0; list < 2; list++ )
      {
        for( int refIdx = 0; refIdx < slice->getNumRefIdx( RefPicList( list ) ); refIdx++ )
        {
          if( slice->getRefPic( RefPicList( list ), refIdx ) == refPic )
          {
            entry.ratioX = slice->getScalingRatio( RefPicList( list ), refIdx ).first;
            entry.ratioY = slice->getScalingRatio( RefPicList( list ), refIdx ).second;
          }
        }
      }
    }
    const Window& window = refPic->slices[0]->getPPS()->getScalingWindow();
    entry.winLeft        = window.getWindowLeftOffset() * SPS::getWinUnitX( cs.sps->getChromaFormatIdc() );
    entry.winTop         = window.getWindowTopOffset() * SPS::getWinUnitY( cs.sps->getChromaFormatIdc() );
    CHECK( !entry.scaled && ( planes.stride[COMPONENT_Y] != pic.bufs[COMPONENT_Y].stride || planes.stride[COMPONENT_Cb] != pic.bufs[COMPONENT_Cb].stride ),
           "Vulkan: a reference picture laid out differently" );
    *table++ = entry;
    if( !refPic->gpuCurrent )
    {
      uploads.push_back( refPic );
    }
  }
}

VkDeviceAddress GpuRecon::subpicRects( const CodingStructure& cs )
{
  const PPS&           pps = *cs.pps;
  const PreCalcValues& pcv = *cs.pcv;
  bool                 on  = false;
  for( int i = 0; pps.getNumSubPics() > 1 && i < pps.getNumSubPics(); i++ )
  {
    on |= pps.getSubPic( i ).getTreatedAsPicFlag();
  }
  if( !on )
  {
    return 0;
  }
  const size_t size = alignUp( size_t( pcv.sizeInCtus ) * 4 * sizeof( uint16_t ), 256 );
  if( m_subpics.size < size )
  {
    m_ctx->free( m_subpics );
    m_subpics = m_ctx->alloc( size, MemKind::Upload );
  }
  uint16_t* rects = reinterpret_cast<uint16_t*>( m_subpics.mapped );
  for( unsigned ctu = 0; ctu < pcv.sizeInCtus; ctu++ )
  {
    const Position pos( ( ctu % pcv.widthInCtus ) * pcv.maxCUWidth, ( ctu / pcv.widthInCtus ) * pcv.maxCUHeight );
    const SubPic&  sub   = pps.getSubPicFromPos( pos );
    const bool     asPic = sub.getTreatedAsPicFlag();
    rects[ctu * 4]       = uint16_t( asPic ? sub.getSubPicLeft() : 0 );
    rects[ctu * 4 + 1]   = uint16_t( asPic ? sub.getSubPicTop() : 0 );
    rects[ctu * 4 + 2]   = uint16_t( asPic ? sub.getSubPicRight() : pcv.lumaWidth - 1 );
    rects[ctu * 4 + 3]   = uint16_t( asPic ? sub.getSubPicBottom() : pcv.lumaHeight - 1 );
  }
  return m_subpics.addr;
}

uint8_t* GpuRecon::params( size_t size )
{
  if( m_params.size < size )
  {
    m_ctx->free( m_params );
    m_params = m_ctx->alloc( size, MemKind::Upload );
  }
  return m_params.mapped;
}

Mv* GpuRecon::dmvrMvCache( size_t count )
{
  const size_t size = count * sizeof( Mv );
  if( m_dmvr.size < size )
  {
    m_ctx->free( m_dmvr );
    m_dmvr = m_ctx->alloc( size, MemKind::HostCached );
  }
  return reinterpret_cast<Mv*>( m_dmvr.mapped );
}

// ====================================================================================================================
// A picture's work
// ====================================================================================================================

void GpuRecon::setPicture( const PelStorage& pic, const LmcsTables* tables, bool defer, const int* scalingMatrices, size_t scalingCount )
{
  // (the instance's previous picture is done: its work no longer reads these, nor its front end's ranges, should
  // endCompute not have given them back)
  giveFrontEnd();
  m_scalingOn = scalingMatrices != nullptr;
  if( scalingMatrices )
  {
    // the matrices' factors, a byte each (the scaling lists' coefficients are 1 .. 255; 0 where a matrix has none)
    if( m_scaling.size < scalingCount )
    {
      m_ctx->free( m_scaling );
      m_scaling = m_ctx->alloc( alignUp( scalingCount, 256 ), MemKind::Upload );
    }
    for( size_t i = 0; i < scalingCount; i++ )
    {
      CHECK( scalingMatrices[i] < 0 || scalingMatrices[i] > 255, "a scaling list's coefficient out of range" );
      m_scaling.mapped[i] = uint8_t( scalingMatrices[i] );
    }
  }
  m_lmcsOn = tables != nullptr;
  if( tables )
  {
    m_lmcsCount       = tables->fwd.size();
    const size_t size = m_lmcsCount * ( 2 * sizeof( int16_t ) + sizeof( int32_t ) );
    if( m_lmcs.size < size )
    {
      m_ctx->free( m_lmcs );
      m_lmcs = m_ctx->alloc( size, MemKind::Upload );
    }
    memcpy( m_lmcs.mapped, tables->fwd.data(), m_lmcsCount * sizeof( int16_t ) );
    memcpy( m_lmcs.mapped + m_lmcsCount * sizeof( int16_t ), tables->inv.data(), m_lmcsCount * sizeof( int16_t ) );
    memcpy( m_lmcs.mapped + m_lmcsCount * 2 * sizeof( int16_t ), tables->chromaScale.data(), m_lmcsCount * sizeof( int32_t ) );
  }
  m_deferOn = defer;   // (bindDeferPlanes gives them room when the picture's work is recorded)
}

void GpuRecon::bindDeferPlanes( const PelStorage& pic )
{
  if( !m_deferOn )
  {
    return;
  }
  // the planes inter prediction keeps predictions aside in, laid out as the picture's planes (from their first
  // sample)
  size_t size = 0;
  size_t at[MAX_NUM_COMPONENT];
  for( size_t comp = 0; comp < pic.bufs.size(); comp++ )
  {
    at[comp] = size;
    size    += alignUp( size_t( pic.bufs[comp].stride ) * pic.bufs[comp].height * sizeof( Pel ), 256 );
  }
  if( m_defer.size < size )
  {
    // (the planes may be the instances' shared ones, which another instance's picture queued before may still use)
    m_ctx->freeAfterQueued( m_defer );
    m_defer = m_ctx->alloc( size, MemKind::Device );
  }
  for( size_t comp = 0; comp < pic.bufs.size(); comp++ )
  {
    m_deferPlanes.addr[comp]   = m_defer.addr + at[comp];
    m_deferPlanes.stride[comp] = pic.bufs[comp].stride;
  }
}

GpuRecon::SharedDeferTurn::SharedDeferTurn( GpuRecon& recon, SharedReconBuffers* shared ) : m_recon( recon ), m_shared( shared )
{
  if( m_shared )
  {
    m_lock = std::unique_lock<std::mutex>( m_shared->deferMutex );
    swapBuffers();
  }
}

GpuRecon::SharedDeferTurn::~SharedDeferTurn()
{
  if( m_shared )
  {
    swapBuffers();   // (the instance's own back, before the lock goes)
  }
}

void GpuRecon::SharedDeferTurn::swapBuffers()
{
  std::swap( m_recon.m_defer, m_shared->defer );
}

SharedReconBuffers::~SharedReconBuffers()
{
  ctx->free( defer );
}

void GpuRecon::prepareFilters( CodingStructure& cs, bool deblock, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf )
{
  // the parameters, one after the other: deblocking's, ALF's, SAO's, then LMCS's (per CTU, whether its slice uses
  // LMCS: rspCtuBcw maps its luma back)
  m_prepared.dbSize   = deblockParamSize( cs );
  m_prepared.alfSize  = alf ? alfParamSize( cs ) : 0;
  m_prepared.saoSize  = sao ? saoParamSize( cs ) : 0;
  m_prepared.lmcsSize = m_lmcsOn ? alignUp( size_t( cs.pcv->sizeInCtus ), 256 ) : 0;
  uint8_t* base       = params( m_prepared.dbSize + m_prepared.alfSize + m_prepared.saoSize + m_prepared.lmcsSize );
  if( m_lmcsOn )
  {
    uint8_t* ctus = base + m_prepared.dbSize + m_prepared.alfSize + m_prepared.saoSize;
    for( unsigned ctu = 0; ctu < cs.pcv->sizeInCtus; ctu++ )
    {
      ctus[ctu] = cs.getCtuData( ctu ).slice->getLmcsEnabledFlag() ? 1 : 0;
    }
  }
  m_prepared.deblock   = deblock && prepareDeblock( cs, base );
  m_prepared.sao       = sao && prepareSao( cs, *sao, base + m_prepared.dbSize + m_prepared.alfSize );
  m_prepared.alfBounds = alf && prepareAlf( cs, *alf, base + m_prepared.dbSize );
  m_prepared.picture   = cs.picture;
}

void GpuRecon::startEarlyRows( CodingStructure& cs, const PictureJobs& jobs, size_t ctus, PelStorage& pic )
{
  const Api&                  vk = m_ctx->api();
  std::vector<const Picture*> uploads;
  if( jobs.inter )
  {
    writeRefTable( cs, jobs, pic, uploads );
  }
  // the early rows come only with the CPU front end, which shares nothing (their work comes before startPicture's)
  CHECK_FATAL( jobs.shared, "Vulkan: the early rows of a picture with shared buffers" );
  m_useFrontEnd   = false;
  m_frontEndTrafo = false;
  m_frontEndInter = false;
  bindDeferPlanes( pic );

  // The early rows' commands, after what earlier submissions wrote. The pool is reset: the instance's previous
  // picture is done.
  m_cmd = m_cmdEarlyRows;
  vk.vkResetCommandPool( m_ctx->device(), m_pool, 0 );
  beginOneTime( m_cmd );
  m_earlyRowsMarks = 0;
  if( m_earlyRowsQueries )
  {
    vk.vkCmdResetQueryPool( m_cmd, m_earlyRowsQueries, 0, kEarlyRowsMarks );
    vk.vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_earlyRowsQueries, m_earlyRowsMarks++ );
  }
  fullBarrier();
  for( const Picture* refPic: uploads )
  {
    copyPlanes( refPic->m_bufs[PIC_RECONSTRUCTION], kHost, kTwin );
  }
  transferToComputeBarrier();
  const Planes planes = twinPlanes( pic );
  recordTrafo( cs, *jobs.trafo, 0, ctus, m_cpuJobs.earlyTrafo, hostPlanes( pic ), planes );
  if( jobs.inter )
  {
    computeBarrier();
    recordInter( cs, *jobs.inter, 0, ctus, m_cpuJobs.earlyInter, planes );
  }
  vk.vkEndCommandBuffer( m_cmd );
  // Submitted now. The first part's commands, submitted after these, begin with a barrier over them.
  m_ctx->submitFirst( m_cmd, VK_NULL_HANDLE );
  for( const Picture* refPic: uploads )
  {
    refPic->gpuCurrent = true;   // for the work submitted after this
  }
}

void GpuRecon::takePictureSettings( CodingStructure& cs, const PictureJobs& jobs, bool deblock, SampleAdaptiveOffset* sao,
                                    const AdaptiveLoopFilter* alf, PelStorage& pic )
{
  if( m_prepared.picture != cs.picture )
  {
    prepareFilters( cs, deblock, sao, alf );
  }
  m_prepared.picture = nullptr;   // the next picture's parameters are worked out anew
  m_useFrontEnd      = jobs.useFrontEnd;
  m_frontEndTrafo    = jobs.frontEndTrafo;
  m_frontEndInter    = jobs.frontEndInter;
  bindDeferPlanes( pic );
}

// The GPU front end runs on the async compute queue, beside the work submitted before: its passes are latency-bound (a
// few waves each, the motion derivation one per CTU row), and the reconstruction of the pictures before keeps the GPU
// busy meanwhile. Part 1 of a picture the GPU parsed waits there for the picture's parsed records
// (recordFrontEndAsync). Otherwise the first part holds the front end (and for a picture the GPU parsed, waits for its
// records itself).
bool GpuRecon::startFrontEnd( const CodingStructure& cs, const PictureJobs& jobs )
{
  const bool async = jobs.useFrontEnd && jobs.cus && m_cmdFrontEnd1 != VK_NULL_HANDLE;
  if( async )
  {
    recordFrontEndAsync( cs, jobs );
  }
  else if( jobs.useFrontEnd && jobs.parsed && m_ctx->asyncQueue() )
  {
    m_firstParseWait = jobs.parsed->parseValue();
  }
  return async;
}

void GpuRecon::recordRest( const CodingStructure& cs, const PictureJobs& jobs, PelStorage& pic, const Planes& planes, bool alf )
{
  if( jobs.intra || jobs.useFrontEnd )
  {
    static const std::vector<std::vector<IntraJob>> noIntra;
    recordIntra( cs, jobs.useFrontEnd ? noIntra : *jobs.intra, planes );   // in place
    computeBarrier();
    if( m_lmcsOn )
    {
      recordLmcs( cs, planes, lmcsParams() );   // in place
      computeBarrier();
    }
  }
  const bool inScratch = recordInLoopFilters( cs, jobs, pic, planes, alf );
  computeToTransferBarrier();
  if( inScratch )
  {
    // The luma result is in the scratch plane: it becomes the picture's luma twin, and its former twin the scratch
    // plane (they have the same size). (The chroma result is in the twin: recordAlf.)
    std::swap( planeTwin( pic, COMPONENT_Y ), m_scratch[COMPONENT_Y] );
  }
}

int GpuRecon::startPicture( CodingStructure& cs, const PictureJobs& jobs, bool deblock, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf,
                            PelStorage& pic, bool waitFirst )
{
  // With the GPU's parse, the shared deferred-prediction planes, held until the picture's rest is submitted (the return)
  const SharedDeferTurn deferTurn( *this, jobs.shared );
  takePictureSettings( cs, jobs, deblock, sao, alf, pic );

  // After the early rows (startEarlyRows), the command pool is already reset, the reference table written and the
  // reference pictures uploaded.
  const bool                  afterEarlyRows = jobs.earlyCtus > 0;
  std::vector<const Picture*> uploads;
  if( !afterEarlyRows )
  {
    if( jobs.inter || ( jobs.useFrontEnd && jobs.frontEndInter ) )
    {
      writeRefTable( cs, jobs, pic, uploads );
    }
    m_earlyRowsMarks = 0;
  }
  const bool frontEndAsync = startFrontEnd( cs, jobs );

  // The picture is reconstructed in the GPU's own memory, which its shaders reach faster than host memory, and
  // stays there as a reference picture.
  m_cmd = m_cmdFirst;
  beginCommands( !afterEarlyRows );
  for( const Picture* refPic: uploads )
  {
    copyPlanes( refPic->m_bufs[PIC_RECONSTRUCTION], kHost, kTwin );
  }
  const Planes planes = twinPlanes( pic );
  recordFirstPart( cs, jobs, pic, planes, frontEndAsync );
  splitCommands();
  cs.picture->gpuFirstValue = m_firstValue;   // for the front ends of the pictures whose colocated picture this is
  recordRest( cs, jobs, pic, planes, alf != nullptr );
  const int download = submitPicture( pic, waitFirst );
  // The scratch planes (after a swap, the picture's former twin) go back to the pool at once. The GPU runs the
  // pictures' work one after the other (each part begins with a barrier over all the work before it), so the work of
  // whichever picture takes them next comes after this one's. The download reads none of them.
  for( Buffer& buf: m_scratch )
  {
    m_ctx->recycle( buf );
  }
  for( const Picture* refPic: uploads )
  {
    refPic->gpuCurrent = true;   // for the work submitted after this
  }
  return download;
}

void GpuRecon::recordFirstPart( const CodingStructure& cs, const PictureJobs& jobs, const PelStorage& pic, const Planes& planes, bool frontEndAsync )
{
  const bool trafo = jobs.trafo || ( jobs.useFrontEnd && jobs.frontEndTrafo );
  const bool inter = jobs.inter || ( jobs.useFrontEnd && jobs.frontEndInter );
  // When the GPU makes the residuals, nothing of the picture is uploaded: the residual pass reads the coefficient
  // levels from the picture in host memory. Otherwise what the CPU made of the picture is uploaded first.
  if( !trafo )
  {
    copyPlanes( pic, kHost, kTwin );
  }
  transferToComputeBarrier();
  if( jobs.parse && jobs.cus )
  {
    recordParse( cs, jobs, hostPlanes( pic ) );
  }
  static const std::vector<std::vector<TrafoJob>> noTrafo;
  static const std::vector<std::vector<InterJob>> noInter;
  const auto recordResiduals = [&]() {
    recordTrafo( cs, jobs.useFrontEnd ? noTrafo : *jobs.trafo, jobs.earlyCtus, jobs.useFrontEnd ? 0 : jobs.trafo->size(), m_cpuJobs.trafo,
                 hostPlanes( pic ), planes );
  };
  // With the front end here, the residuals go to the GPU beside its last passes (recordFrontEnd's beside)
  bool trafoDone = false;
  if( ( jobs.cus || jobs.parsed ) && !frontEndAsync )
  {
    const bool beside = trafo && jobs.useFrontEnd;
    trafoDone         = recordFrontEnd( cs, jobs, beside ? std::function<void()>( recordResiduals ) : std::function<void()>() );
  }
  if( jobs.useFrontEnd )
  {
    indirectBarrier();
  }
  if( trafo && !trafoDone )
  {
    recordResiduals();
    computeBarrier();
  }
  if( inter )
  {
    recordInter( cs, jobs.useFrontEnd ? noInter : *jobs.inter, jobs.earlyCtus, jobs.useFrontEnd ? 0 : jobs.inter->size(), m_cpuJobs.inter,
                 planes );   // in place
    computeBarrier();
  }
}

bool GpuRecon::recordInLoopFilters( const CodingStructure& cs, const PictureJobs& jobs, const PelStorage& pic, const Planes& planes, bool alf )
{
  const bool deblock = m_prepared.deblock;
  const bool sao     = m_prepared.sao;
  if( deblock )
  {
    static const std::vector<EdgeLists> noEdges;
    recordDeblock( cs, planes, deblockParams(), jobs.useFrontEnd ? noEdges : *jobs.edges );   // in place
  }
  if( sao )
  {
    if( deblock )
    {
      computeBarrier();
    }
    recordSao( cs, planes, saoParams() );   // in place
  }
  if( alf )
  {
    if( deblock || sao )
    {
      computeBarrier();
    }
    recordAlf( cs, planes, scratchPlanes( pic ), alfParams() );
  }
  return alf;
}

}   // namespace gpu
}   // namespace vvdec
