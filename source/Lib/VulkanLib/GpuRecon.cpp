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
    \brief    the stages of a picture's reconstruction that run on the GPU
*/

#include "GpuRecon.h"

#ifdef _MSC_VER
#  include <intrin.h>
#endif

#include "vvdec_gpu/GpuShaders.h"

#include "CommonLib/AdaptiveLoopFilter.h"
#include "CommonLib/MipData.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/Rom.h"
#include "CommonLib/Picture.h"
#include "CommonLib/SampleAdaptiveOffset.h"
#include "CommonLib/Slice.h"

#include <chrono>
#include <cstring>
#include <utility>

namespace vvdec
{
namespace gpu
{

// ====================================================================================================================
// picture buffers
// ====================================================================================================================

// a plane of a picture: in host memory the GPU reaches, where vvdec's CPU
// code reads and writes it, and its twin in the GPU's own memory, where the
// GPU reconstructs the picture and keeps it as a reference (made when first used)
struct HostPlane
{
  Buffer host;
  Buffer twin;
};

static void* createHostPlane( void* opaque, vvdecComponentType, uint32_t size, uint32_t, void** allocator )
{
  Context*   context = static_cast<Context*>( opaque );
  HostPlane* plane   = new HostPlane;
  plane->host        = context->alloc( size, MemKind::HostCached );
  if( !plane->host.mapped )
  {
    context->free( plane->host );
    delete plane;
    return nullptr;
  }
  *allocator = plane;
  return plane->host.mapped;
}

static void unrefHostPlane( void* opaque, void* allocator )
{
  HostPlane* plane = static_cast<HostPlane*>( allocator );
  static_cast<Context*>( opaque )->free( plane->host );
  static_cast<Context*>( opaque )->free( plane->twin );
  delete plane;
}

static HostPlane* hostPlane( const PelStorage& storage, ComponentID comp )
{
  HostPlane* plane = static_cast<HostPlane*>( storage.getBufAllocator( comp ) );
  CHECK_FATAL( !plane || !storage.isExternAllocator(), "Vulkan: a picture buffer the GPU cannot reach" );
  return plane;
}

UserAllocator hostPictureAllocator( Context* context )
{
  return UserAllocator( createHostPlane, unrefHostPlane, context );
}

void releaseTwins( Picture& pic )
{
  if( !pic.gpuMemory )
  {
    return;
  }
  pic.gpuCurrent              = false;   // were it referenced after all: uploaded again
  Context*          context   = static_cast<Context*>( pic.gpuMemory->opaque );
  const PelStorage& planes    = pic.m_bufs[PIC_RECONSTRUCTION];
  for( int comp = 0; comp < (int) planes.bufs.size(); comp++ )
  {
    HostPlane* plane = static_cast<HostPlane*>( planes.getBufAllocator( ComponentID( comp ) ) );
    if( plane )
    {
      context->recycle( plane->twin );
    }
  }
}

// where a plane's sample (0, 0) is in its buffer
static VkDeviceSize planeOffset( const PelStorage& storage, ComponentID comp )
{
  const Buffer&  host   = hostPlane( storage, comp )->host;
  const uint8_t* origin = reinterpret_cast<const uint8_t*>( storage.bufs[comp].buf );
  CHECK_FATAL( origin < host.mapped || origin >= host.mapped + host.size, "Vulkan: a plane outside its buffer" );
  return VkDeviceSize( origin - host.mapped );
}

// a plane on the GPU (the twin of a host plane, or scratch memory like it):
// the host plane's rows of the picture -- its extension to whole CTUs
// included, not its margins above and below, which no shader reads -- with
// the same stride, so that a sample's place in its row is the same
struct GpuLayout
{
  VkDeviceSize hostOffset;   // where its first row is in the host buffer
  VkDeviceSize size;         // its bytes
  VkDeviceSize origin;       // where sample (0, 0) is in it
};

static GpuLayout gpuLayout( const PelStorage& storage, ComponentID comp )
{
  const Buffer&      host   = hostPlane( storage, comp )->host;
  const VkDeviceSize offset = planeOffset( storage, comp );
  const VkDeviceSize stride = VkDeviceSize( storage.bufs[comp].stride ) * sizeof( Pel );
  const VkDeviceSize top    = offset / stride;                // the rows of margin above (as many below)
  const VkDeviceSize rows   = host.size / stride - 2 * top;   // host.size: the rows, an extra sample, rounding
  CHECK_FATAL( rows < VkDeviceSize( storage.bufs[comp].height ), "Vulkan: a plane laid out unexpectedly" );
  return { top * stride, rows * stride, offset % stride };
}

// ====================================================================================================================
// GpuRecon
// ====================================================================================================================

GpuRecon::GpuRecon( Context* context ) : m_ctx( context )
{
  const Api& vk = m_ctx->api();

  VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
  poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  poolInfo.queueFamilyIndex = m_ctx->queueFamily();
  if( vk.vkCreateCommandPool( m_ctx->device(), &poolInfo, nullptr, &m_pool ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateCommandPool failed" );
  }

  VkCommandBufferAllocateInfo cmdInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  cmdInfo.commandPool        = m_pool;
  cmdInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdInfo.commandBufferCount = 3;
  VkCommandBuffer cmds[3];
  if( vk.vkAllocateCommandBuffers( m_ctx->device(), &cmdInfo, cmds ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkAllocateCommandBuffers failed" );
  }
  m_cmdFirst = m_cmd = cmds[0];
  m_cmdRest  = cmds[1];
  m_cmdBand  = cmds[2];

  VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  if( vk.vkCreateFence( m_ctx->device(), &fenceInfo, nullptr, &m_fence ) != VK_SUCCESS
      || vk.vkCreateFence( m_ctx->device(), &fenceInfo, nullptr, &m_fenceFirst ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateFence failed" );
  }

  if( m_ctx->transferQueue() )
  {
    VkCommandPoolCreateInfo transferPoolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    transferPoolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    transferPoolInfo.queueFamilyIndex = m_ctx->transferFamily();
    VkCommandBufferAllocateInfo transferCmdInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    transferCmdInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    transferCmdInfo.commandBufferCount = 1;
    VkSemaphoreCreateInfo semaphoreInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    for( Download& download: m_downloads )
    {
      if( vk.vkCreateCommandPool( m_ctx->device(), &transferPoolInfo, nullptr, &download.pool ) != VK_SUCCESS
          || ( transferCmdInfo.commandPool = download.pool, vk.vkAllocateCommandBuffers( m_ctx->device(), &transferCmdInfo, &download.cmd ) ) != VK_SUCCESS
          || vk.vkCreateSemaphore( m_ctx->device(), &semaphoreInfo, nullptr, &download.semaphore ) != VK_SUCCESS
          || vk.vkCreateFence( m_ctx->device(), &fenceInfo, nullptr, &download.fence ) != VK_SUCCESS )
      {
        THROW_FATAL( "Vulkan: making the copy engine's commands failed" );
      }
    }
  }

  // the transform matrices, in the order of shaders/trafo.slang: DCT-II from 2
  // to 64 points, DST-VII and DCT-VIII from 4 to 32
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
    m_matrices = m_ctx->alloc( count * sizeof( int16_t ), MemKind::Upload );
    int16_t* to = reinterpret_cast<int16_t*>( m_matrices.mapped );
    for( const auto& matrix: matrices )
    {
      for( int i = 0; i < matrix.second * matrix.second; i++ )
      {
        *to++ = int16_t( matrix.first[i] );
      }
    }
  }

  // GPM's prestored masks of weights (g_globalGeoWeights), for shaders/inter.slang
  {
    static_assert( sizeof( g_globalGeoWeights ) == GEO_NUM_PRESTORED_MASK * GEO_WEIGHT_MASK_SIZE * GEO_WEIGHT_MASK_SIZE * sizeof( int16_t ) && GEO_WEIGHT_MASK_SIZE == 112,
                   "GPM's masks are not those of shaders/inter.slang" );
    m_geo = m_ctx->alloc( sizeof( g_globalGeoWeights ), MemKind::Upload );
    memcpy( m_geo.mapped, g_globalGeoWeights, sizeof( g_globalGeoWeights ) );
  }

  // LFNST's matrices, in the order of shaders/trafo.slang: 8x8, then 4x4
  {
    static_assert( sizeof( g_lfnst8x8 ) == 6144 && sizeof( g_lfnst4x4 ) == 2048, "LFNST's matrices are not those of shaders/trafo.slang" );
    m_lfnst = m_ctx->alloc( sizeof( g_lfnst8x8 ) + sizeof( g_lfnst4x4 ), MemKind::Upload );
    memcpy( m_lfnst.mapped, g_lfnst8x8, sizeof( g_lfnst8x8 ) );
    memcpy( m_lfnst.mapped + sizeof( g_lfnst8x8 ), g_lfnst4x4, sizeof( g_lfnst4x4 ) );
  }

  // MIP's matrices, in the order of shaders/intra.slang: 4x4, 8x8, 16x16
  {
    static_assert( sizeof( mipMatrix4x4 ) == 1024 && sizeof( mipMatrix8x8 ) == 1024 && sizeof( mipMatrix16x16 ) == 2688,
                   "MIP's matrices are not those of shaders/intra.slang" );
    m_mip = m_ctx->alloc( sizeof( mipMatrix4x4 ) + sizeof( mipMatrix8x8 ) + sizeof( mipMatrix16x16 ), MemKind::Upload );
    memcpy( m_mip.mapped, mipMatrix4x4, sizeof( mipMatrix4x4 ) );
    memcpy( m_mip.mapped + sizeof( mipMatrix4x4 ), mipMatrix8x8, sizeof( mipMatrix8x8 ) );
    memcpy( m_mip.mapped + sizeof( mipMatrix4x4 ) + sizeof( mipMatrix8x8 ), mipMatrix16x16, sizeof( mipMatrix16x16 ) );
  }

  if( m_ctx->profiling() )
  {
    VkQueryPoolCreateInfo queryInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    queryInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = kMaxMarks;
    queryInfo.queryCount = kBandMarks;
    if( vk.vkCreateQueryPool( m_ctx->device(), &queryInfo, nullptr, &m_bandQueries ) != VK_SUCCESS )
    {
      THROW_FATAL( "Vulkan: vkCreateQueryPool failed" );
    }
    queryInfo.queryCount = kMaxMarks;
    if( vk.vkCreateQueryPool( m_ctx->device(), &queryInfo, nullptr, &m_queries ) != VK_SUCCESS )
    {
      THROW_FATAL( "Vulkan: vkCreateQueryPool failed" );
    }
    if( m_ctx->transferQueue() && m_ctx->transferTimestamps() )
    {
      queryInfo.queryCount = 2;
      for( Download& download: m_downloads )
      {
        if( vk.vkCreateQueryPool( m_ctx->device(), &queryInfo, nullptr, &download.queries ) != VK_SUCCESS )
        {
          THROW_FATAL( "Vulkan: vkCreateQueryPool failed" );
        }
      }
    }
  }
}

GpuRecon::~GpuRecon()
{
  const Api& vk = m_ctx->api();
  if( m_queries )
  {
    vk.vkDestroyQueryPool( m_ctx->device(), m_queries, nullptr );
  }
  if( m_bandQueries )
  {
    vk.vkDestroyQueryPool( m_ctx->device(), m_bandQueries, nullptr );
  }
  for( Download& download: m_downloads )
  {
    if( download.queries )
    {
      vk.vkDestroyQueryPool( m_ctx->device(), download.queries, nullptr );
    }
    if( download.fence )
    {
      vk.vkDestroyFence( m_ctx->device(), download.fence, nullptr );
    }
    if( download.semaphore )
    {
      vk.vkDestroySemaphore( m_ctx->device(), download.semaphore, nullptr );
    }
    if( download.pool )
    {
      vk.vkDestroyCommandPool( m_ctx->device(), download.pool, nullptr );
    }
  }
  if( m_fence )
  {
    vk.vkDestroyFence( m_ctx->device(), m_fence, nullptr );
  }
  if( m_fenceFirst )
  {
    vk.vkDestroyFence( m_ctx->device(), m_fenceFirst, nullptr );
  }
  if( m_pool )
  {
    vk.vkDestroyCommandPool( m_ctx->device(), m_pool, nullptr );
  }
  m_ctx->free( m_params );
  m_ctx->free( m_dmvr );
  m_ctx->free( m_refs );
  m_ctx->free( m_bandInterJobs );
  m_ctx->free( m_bandTrafoJobs );
  m_ctx->free( m_intraJobs );
  m_ctx->free( m_intraSync );
  m_ctx->free( m_interJobs );
  m_ctx->free( m_trafoJobs );
  m_ctx->free( m_edges );
  m_ctx->free( m_matrices );
  m_ctx->free( m_mip );
  m_ctx->free( m_lfnst );
  m_ctx->free( m_geo );
  m_ctx->free( m_lmcs );
  m_ctx->free( m_defer );
  m_ctx->free( m_scaling );
  m_ctx->free( m_subpics );
  for( Buffer& buf: m_scratch )
  {
    m_ctx->free( buf );
  }
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

void GpuRecon::beginCommands( bool resetPool )
{
  const Api& vk = m_ctx->api();
  if( resetPool )
  {
    vk.vkResetCommandPool( m_ctx->device(), m_pool, 0 );
  }
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( m_cmd, &begin );

  m_marks   = 0;
  m_gapMark = 0;
  if( m_queries )
  {
    vk.vkCmdResetQueryPool( m_cmd, m_queries, 0, kMaxMarks );
    vk.vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queries, m_marks++ );
  }

  // what earlier submissions wrote (the reference pictures), visible to this
  // one. Not after the picture's band (resetPool false): the band began with
  // that barrier, and the rest of the first part neither reads nor writes what
  // the band or the work queued since does (the band's tiles; other pictures'
  // planes and scratch planes) -- the GPU may start it while the band ends
  // (VideoQ 1080p +0.5%)
  if( resetPool )
  {
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
  }
}

void GpuRecon::mark( const char* label )
{
  if( m_cmd == m_cmdBand )
  {
    if( m_bandQueries && m_bandMarks < kBandMarks )
    {
      m_bandLabels[m_bandMarks] = label;
      m_ctx->api().vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_bandQueries, m_bandMarks++ );
    }
    return;
  }
  if( m_queries && m_marks < kMaxMarks )
  {
    m_labels[m_marks] = label;
    m_ctx->api().vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queries, m_marks++ );
  }
}

void GpuRecon::computeBarrier()
{
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  m_ctx->api().vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                                     nullptr );
}

void GpuRecon::splitCommands()
{
  const Api& vk = m_ctx->api();

  // the first part's writes (DMVR's refinements in host memory among them)
  // visible to the CPU
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                           nullptr, 0, nullptr );
  vk.vkEndCommandBuffer( m_cmd );
  // to the GPU at once: the rest is recorded while it runs
  vk.vkResetFences( m_ctx->device(), 1, &m_fenceFirst );
  m_submitted = std::chrono::steady_clock::now();
  m_ctx->submitFirst( m_cmdFirst, m_fenceFirst );

  // the rest, after what the first part wrote
  m_cmd = m_cmdRest;
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( m_cmd, &begin );
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
  // the GPU profile: the time until the rest begins (other pictures' work may
  // run between the parts) counts apart, not as the picture's or its next pass's
  if( m_queries )
  {
    mark( "between the parts" );
    m_gapMark = m_marks - 1;
  }
}

int GpuRecon::submitPicture( const PelStorage& pic, bool waitFirst )
{
  const Api& vk = m_ctx->api();

  // the rest's commands: without the copy engine, the download at their end;
  // their writes visible to the CPU (and to the copies)
  if( !m_ctx->transferQueue() )
  {
    copyPlanes( pic, kTwin, kHost );
  }
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                           nullptr, 0, nullptr );
  vk.vkEndCommandBuffer( m_cmd );

  int       turn     = -1;
  Download* download = nullptr;
  if( m_ctx->transferQueue() )
  {
    // the copies: the picture's planes on the GPU to host memory, with the
    // download's commands whose turn it is (the other's may still run)
    turn     = m_download;
    download = &m_downloads[turn];
    m_download ^= 1;
    CHECK_FATAL( download->pending.load( std::memory_order_acquire ), "Vulkan: a download not waited for (endDownload) before its commands are reused" );
    vk.vkResetCommandPool( m_ctx->device(), download->pool, 0 );
    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk.vkBeginCommandBuffer( download->cmd, &begin );
    if( download->queries )
    {
      vk.vkCmdResetQueryPool( download->cmd, download->queries, 0, 2 );
      vk.vkCmdWriteTimestamp( download->cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, download->queries, 0 );
    }
    for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
    {
      const HostPlane* plane  = hostPlane( pic, ComponentID( comp ) );
      const GpuLayout  layout = gpuLayout( pic, ComponentID( comp ) );
      VkBufferCopy     region{ 0, layout.hostOffset, layout.size };
      vk.vkCmdCopyBuffer( download->cmd, plane->twin.buffer, plane->host.buffer, 1, &region );
    }
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vk.vkCmdPipelineBarrier( download->cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
    if( download->queries )
    {
      vk.vkCmdWriteTimestamp( download->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, download->queries, 1 );
    }
    vk.vkEndCommandBuffer( download->cmd );
    vk.vkResetFences( m_ctx->device(), 1, &download->fence );
    download->pending.store( true, std::memory_order_release );
  }

  vk.vkResetFences( m_ctx->device(), 1, &m_fence );
  m_ctx->submitRest( m_cmdRest, m_fence, download ? download->semaphore : VK_NULL_HANDLE, download ? download->cmd : VK_NULL_HANDLE,
                     download ? download->fence : VK_NULL_HANDLE );
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

void GpuRecon::endCompute()
{
  const Api&     vk     = m_ctx->api();
  const VkResult result = vk.vkWaitForFences( m_ctx->device(), 1, &m_fence, VK_TRUE, UINT64_MAX );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: waiting for the GPU failed (" << result << ")" );
  }
  readTimestamps( std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - m_submitted ).count() );
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

void GpuRecon::readTimestamps( double waited )
{
  const Api& vk = m_ctx->api();
  if( m_queries && m_marks > 1 )
  {
    m_ctx->addTime( "wait (CPU)", waited );
    uint64_t stamps[kMaxMarks];
    vk.vkGetQueryPoolResults( m_ctx->device(), m_queries, 0, m_marks, sizeof( stamps ), stamps, sizeof( uint64_t ),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT );
    const double toMs = m_ctx->timestampPeriod() * 1e-6;
    // the band's passes, counted with the rest's of the same name (a picture's
    // trafo or inter time is its band's and its rest's)
    double   band[kBandMarks] = {};
    uint64_t bandSpan         = 0;
    if( m_bandQueries && m_bandMarks > 1 )
    {
      uint64_t bandStamps[kBandMarks];
      vk.vkGetQueryPoolResults( m_ctx->device(), m_bandQueries, 0, m_bandMarks, sizeof( bandStamps ), bandStamps, sizeof( uint64_t ),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT );
      for( uint32_t j = 1; j < m_bandMarks; j++ )
      {
        band[j] = double( bandStamps[j] - bandStamps[j - 1] ) * toMs;
      }
      bandSpan = bandStamps[m_bandMarks - 1] - bandStamps[0];
    }
    for( uint32_t i = 1; i < m_marks; i++ )
    {
      double time = double( stamps[i] - stamps[i - 1] ) * toMs;
      for( uint32_t j = 1; j < m_bandMarks; j++ )
      {
        if( band[j] > 0 && strcmp( m_bandLabels[j], m_labels[i] ) == 0 )
        {
          time += band[j];
          band[j] = 0;
        }
      }
      m_ctx->addTime( m_labels[i], time );
    }
    for( uint32_t j = 1; j < m_bandMarks; j++ )
    {
      if( band[j] > 0 )
      {
        m_ctx->addTime( m_bandLabels[j], band[j] );
      }
    }
    const uint64_t between = m_gapMark > 0 ? stamps[m_gapMark] - stamps[m_gapMark - 1] : 0;
    m_ctx->addTime( "submission (GPU)", double( stamps[m_marks - 1] - stamps[0] - between + bandSpan ) * toMs );
  }
}

// ====================================================================================================================
// SAO and ALF
// ====================================================================================================================

// the push constants of shaders/alf.slang
struct AlfPush
{
  VkDeviceAddress src0, dst0, src1, dst1, srcLuma, ctus, coeff, clip, ccCoeff;
  int32_t         srcStride, dstStride, lumaStride, width, height, lumaWidth, lumaHeight, ctuLog2, widthInCtus, bitDepth, scaleX, scaleY;
  int32_t         numVbVer, numVbHor, vbVer[3], vbHor[3];
};

// the push constants of shaders/sao.slang
struct SaoPush
{
  VkDeviceAddress src, dst, ctus;
  int32_t         srcStride, dstStride, width, height, ctuWidth, ctuHeight, widthInCtus, comp, bitDepth, scaleX, scaleY;
  int32_t         numVbVer, numVbHor, vbVer[3], vbHor[3];
};

// the picture's virtual boundaries (luma samples) into the push constants of
// ALF or SAO
template<class Push>
static void setVirtualBoundaries( const PicHeader& picHeader, Push& push )
{
  push.numVbVer = push.numVbHor = 0;
  if( picHeader.getVirtualBoundariesPresentFlag() )
  {
    push.numVbVer = int32_t( picHeader.getNumVerVirtualBoundaries() );
    push.numVbHor = int32_t( picHeader.getNumHorVirtualBoundaries() );
    CHECK( push.numVbVer > 3 || push.numVbHor > 3, "too many virtual boundaries" );
    for( int i = 0; i < push.numVbVer; i++ )
    {
      push.vbVer[i] = int32_t( picHeader.getVirtualBoundariesPosX( i ) );
    }
    for( int i = 0; i < push.numVbHor; i++ )
    {
      push.vbHor[i] = int32_t( picHeader.getVirtualBoundariesPosY( i ) );
    }
  }
}

// the ALF parameters: the filter tables, luma sets [16 fixed + 8 APS][transpose][class][13],
// chroma [APS][alternative][7], CC-ALF [APS][component][filter][8], then 8 bytes per CTU
constexpr size_t kAlfSetSize     = MAX_NUM_ALF_TRANSPOSE_ID * MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF;
constexpr size_t kAlfChromaSize  = ALF_CTB_MAX_NUM_APS * MAX_NUM_ALF_ALTERNATIVES_CHROMA * MAX_NUM_ALF_CHROMA_COEFF;
constexpr size_t kAlfCcSize      = ALF_CTB_MAX_NUM_APS * 2 * MAX_NUM_CC_ALF_FILTERS * MAX_NUM_CC_ALF_CHROMA_COEFF;
constexpr size_t kAlfLumaCoeff   = 0;
constexpr size_t kAlfLumaClip    = kAlfLumaCoeff + NUM_TOTAL_FILTER_SETS * kAlfSetSize * sizeof( int16_t );
constexpr size_t kAlfChromaCoeff = kAlfLumaClip + NUM_TOTAL_FILTER_SETS * kAlfSetSize * sizeof( int16_t );
constexpr size_t kAlfChromaClip  = kAlfChromaCoeff + kAlfChromaSize * sizeof( int16_t );
constexpr size_t kAlfCcCoeff     = kAlfChromaClip + kAlfChromaSize * sizeof( int16_t );
constexpr size_t kAlfCtus        = ( kAlfCcCoeff + kAlfCcSize * sizeof( int16_t ) + 255 ) & ~size_t( 255 );
static_assert( kAlfSetSize == 4 * 25 * 13, "the layout of shaders/alf.slang" );

size_t GpuRecon::alfParamSize( const CodingStructure& cs )
{
  return ( kAlfCtus + size_t( cs.pcv->sizeInCtus ) * 8 + 255 ) & ~size_t( 255 );
}

size_t GpuRecon::saoParamSize( const CodingStructure& cs )
{
  return ( size_t( cs.pcv->sizeInCtus ) * 3 * 8 * sizeof( int16_t ) + 255 ) & ~size_t( 255 );
}

bool GpuRecon::prepareAlf( const CodingStructure& cs, const AdaptiveLoopFilter& alfFilter, uint8_t* base )
{
  bool bounds = false;
  const PreCalcValues& pcv = *cs.pcv;

  int16_t* lumaCoeff   = reinterpret_cast<int16_t*>( base + kAlfLumaCoeff );
  int16_t* lumaClip    = reinterpret_cast<int16_t*>( base + kAlfLumaClip );
  int16_t* chromaCoeff = reinterpret_cast<int16_t*>( base + kAlfChromaCoeff );
  int16_t* chromaClip  = reinterpret_cast<int16_t*>( base + kAlfChromaClip );
  int16_t* ccCoeff     = reinterpret_cast<int16_t*>( base + kAlfCcCoeff );
  uint8_t* ctus        = base + kAlfCtus;

  for( int set = 0; set < NUM_FIXED_FILTER_SETS; set++ )
  {
    memcpy( lumaCoeff + set * kAlfSetSize, alfFilter.getFixedFilterSetCoeff( set ), kAlfSetSize * sizeof( int16_t ) );
    memcpy( lumaClip  + set * kAlfSetSize, alfFilter.getClipDefault(),               kAlfSetSize * sizeof( int16_t ) );
  }
  for( const Slice* slice: cs.picture->slices )
  {
    if( slice->getAlfEnabledFlag( COMPONENT_Y ) )
    {
      for( int i = 0; i < slice->getNumAlfAps(); i++ )
      {
        const int  apsId = slice->getAlfApsIdsLuma()[i];
        const APS* aps   = slice->getAlfAPS( apsId );
        CHECK( aps == nullptr, "invalid APS" );
        const AlfSliceParam& alfParam = aps->getAlfAPSParam();
        memcpy( lumaCoeff + ( NUM_FIXED_FILTER_SETS + apsId ) * kAlfSetSize, alfParam.lumaCoeffFinal, kAlfSetSize * sizeof( int16_t ) );
        memcpy( lumaClip  + ( NUM_FIXED_FILTER_SETS + apsId ) * kAlfSetSize, alfParam.lumaClippFinal, kAlfSetSize * sizeof( int16_t ) );
      }
    }
    if( slice->getAlfEnabledFlag( COMPONENT_Cb ) || slice->getAlfEnabledFlag( COMPONENT_Cr ) )
    {
      const int  apsId = slice->getAlfApsIdChroma();
      const APS* aps   = slice->getAlfAPS( apsId );
      CHECK( aps == nullptr, "invalid APS" );
      const AlfSliceParam& alfParam = aps->getAlfAPSParam();
      constexpr size_t     perAps   = MAX_NUM_ALF_ALTERNATIVES_CHROMA * MAX_NUM_ALF_CHROMA_COEFF;
      memcpy( chromaCoeff + apsId * perAps, alfParam.chromaCoeff,    perAps * sizeof( int16_t ) );
      memcpy( chromaClip  + apsId * perAps, alfParam.chrmClippFinal, perAps * sizeof( int16_t ) );
    }
    for( int comp = 1; comp < 3; comp++ )
    {
      if( slice->getCcAlfEnabledFlag( comp ) )
      {
        const int  apsId = comp == 1 ? slice->getCcAlfCbApsId() : slice->getCcAlfCrApsId();
        const APS* aps   = slice->getAlfAPS( apsId );
        CHECK( aps == nullptr, "invalid APS" );
        constexpr size_t perComp = MAX_NUM_CC_ALF_FILTERS * MAX_NUM_CC_ALF_CHROMA_COEFF;
        memcpy( ccCoeff + ( apsId * 2 + comp - 1 ) * perComp, aps->getCcAlfAPSParam().ccAlfCoeff[comp - 1], perComp * sizeof( int16_t ) );
      }
    }
  }

  for( int ctuIdx = 0; ctuIdx < (int) pcv.sizeInCtus; ctuIdx++ )
  {
    const CtuData&    ctuData = cs.getCtuData( ctuIdx );
    const Slice*      slice   = ctuData.cuPtr[0][0]->slice;
    const CtuAlfData& alf     = ctuData.alfParam;
    uint8_t*          ctu     = ctus + ctuIdx * 8;

    ctu[0] = 255;
    if( alf.alfCtuEnableFlag[COMPONENT_Y] )
    {
      const int index = alf.alfCtbFilterIndex;
      ctu[0] = uint8_t( index < NUM_FIXED_FILTER_SETS ? index : NUM_FIXED_FILTER_SETS + slice->getAlfApsIdsLuma()[index - NUM_FIXED_FILTER_SETS] );
    }
    ctu[1] = uint8_t( slice->getAlfApsIdChroma() & ( ALF_CTB_MAX_NUM_APS - 1 ) );
    ctu[2] = alf.alfCtuEnableFlag[COMPONENT_Cb] ? alf.alfCtuAlternative[0] : 255;
    ctu[3] = alf.alfCtuEnableFlag[COMPONENT_Cr] ? alf.alfCtuAlternative[1] : 255;
    ctu[4] = slice->getCcAlfEnabledFlag( 1 ) && alf.ccAlfFilterControl[0] ? uint8_t( slice->getCcAlfCbApsId() * 4 + alf.ccAlfFilterControl[0] - 1 ) : 255;
    ctu[5] = slice->getCcAlfEnabledFlag( 2 ) && alf.ccAlfFilterControl[1] ? uint8_t( slice->getCcAlfCrApsId() * 4 + alf.ccAlfFilterControl[1] - 1 ) : 255;
    // the CTU's clipped sides (virtual boundaries on them, slice, tile and
    // subpicture boundaries without loop filtering across), its raster-slice
    // corners, and whether virtual boundaries split it
    const Position pos( ( ctuIdx % pcv.widthInCtus ) * pcv.maxCUWidth, ( ctuIdx / pcv.widthInCtus ) * pcv.maxCUHeight );
    const Area     area( pos, Size( std::min( pcv.maxCUWidth, pcv.lumaWidth - pos.x ), std::min( pcv.maxCUHeight, pcv.lumaHeight - pos.y ) ) );
    bool           clipTop, clipBottom, clipLeft, clipRight;
    int            numHor, numVer, horPos[3], verPos[3], rasterPad;
    bounds |= AdaptiveLoopFilter::isClipOrCrossedByVirtualBoundaries( cs, area, clipTop, clipBottom, clipLeft, clipRight, numHor, numVer, horPos, verPos,
                                                                      rasterPad );
    ctu[6] = uint8_t( int( clipTop ) | int( clipBottom ) << 1 | int( clipLeft ) << 2 | int( clipRight ) << 3 | rasterPad << 4 );
    ctu[7] = 0;
  }
  return bounds;
}

void GpuRecon::recordAlf( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params )
{
  const PreCalcValues& pcv = *cs.pcv;
  const Api&           vk  = m_ctx->api();

  AlfPush push{};
  push.ctus        = params + kAlfCtus;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  push.bitDepth    = cs.sps->getBitDepth();
  push.lumaWidth   = pcv.lumaWidth;
  push.lumaHeight  = pcv.lumaHeight;
  push.lumaStride  = src.stride[COMPONENT_Y];
  // with boundaries (prepareAlf), the BOUNDS variants
  const bool bounds = m_prepared.alfBounds;
  setVirtualBoundaries( *cs.picHeader, push );

  // luma (its shader loads and stores the rows 8 bytes at a time, from
  // multiples of 4 samples)
  CHECK_FATAL( src.addr[COMPONENT_Y] % 8 != 0 || src.stride[COMPONENT_Y] % 4 != 0 || dst.addr[COMPONENT_Y] % 8 != 0 || dst.stride[COMPONENT_Y] % 4 != 0,
               "Vulkan: the luma rows are not 8-byte aligned" );
  push.src0      = src.addr[COMPONENT_Y];
  push.dst0      = dst.addr[COMPONENT_Y];
  push.coeff     = params + kAlfLumaCoeff;
  push.clip      = params + kAlfLumaClip;
  push.srcStride = src.stride[COMPONENT_Y];
  push.dstStride = dst.stride[COMPONENT_Y];
  push.width     = pcv.lumaWidth;
  push.height    = pcv.lumaHeight;
  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( bounds ? kShader_alf_luma_bounds : kShader_alf_luma ) );
  vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
  vk.vkCmdDispatch( m_cmd, ( ( push.width + 3 ) / 4 + 7 ) / 8, ( ( push.height + 3 ) / 4 + 7 ) / 8, 1 );
  mark( "alf luma" );

  if( isChromaEnabled( pcv.chrFormat ) )
  {
    CHECK( src.stride[COMPONENT_Cb] != src.stride[COMPONENT_Cr] || dst.stride[COMPONENT_Cb] != dst.stride[COMPONENT_Cr],
           "the chroma planes differ in stride" );
    push.src0      = src.addr[COMPONENT_Cb];
    push.dst0      = dst.addr[COMPONENT_Cb];
    push.src1      = src.addr[COMPONENT_Cr];
    push.dst1      = dst.addr[COMPONENT_Cr];
    push.srcLuma   = src.addr[COMPONENT_Y];
    push.coeff     = params + kAlfChromaCoeff;
    push.clip      = params + kAlfChromaClip;
    push.ccCoeff   = params + kAlfCcCoeff;
    push.srcStride = src.stride[COMPONENT_Cb];
    push.dstStride = dst.stride[COMPONENT_Cb];
    push.scaleX    = getComponentScaleX( COMPONENT_Cb, pcv.chrFormat );
    push.scaleY    = getComponentScaleY( COMPONENT_Cb, pcv.chrFormat );
    push.width     = pcv.lumaWidth >> push.scaleX;
    push.height    = pcv.lumaHeight >> push.scaleY;
    vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( bounds ? kShader_alf_chroma_bounds : kShader_alf_chroma ) );
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    vk.vkCmdDispatch( m_cmd, ( ( push.width + 3 ) / 4 + 7 ) / 8, ( ( push.height + 3 ) / 4 + 7 ) / 8, 1 );
    mark( "alf chroma" );
  }
}

bool GpuRecon::prepareSao( CodingStructure& cs, SampleAdaptiveOffset& sao, uint8_t* base )
{
  const int numCtus = (int) cs.pcv->sizeInCtus;
  const int numComp = (int) getNumberValidComponents( cs.pcv->chrFormat );
  int16_t*  ctus    = reinterpret_cast<int16_t*>( base );
  bool      any     = false;

  // in raster order: a merged CTU copies the parameters of its left or above neighbour
  for( int ctuIdx = 0; ctuIdx < numCtus; ctuIdx++ )
  {
    sao.reconstructCtuParam( cs, ctuIdx );
  }

  for( int ctuIdx = 0; ctuIdx < numCtus; ctuIdx++ )
  {
    const SAOBlkParam& param = cs.getCtuData( ctuIdx ).saoParam;
    int16_t            avail = -1;   // derived when a component of the CTU uses SAO
    for( int comp = 0; comp < 3; comp++ )
    {
      int16_t*         ctu    = ctus + ( ctuIdx * 3 + comp ) * 8;
      const SAOOffset& offset = param[comp];
      memset( ctu, 0, 8 * sizeof( int16_t ) );
      ctu[0] = -1;
      if( comp >= numComp || offset.modeIdc == SAO_MODE_OFF )
      {
        continue;
      }
      CHECK( offset.modeIdc != SAO_MODE_NEW || offset.typeIdc < SAO_TYPE_START_EO || offset.typeIdc > SAO_TYPE_BO, "unresolved SAO parameters" );
      if( avail < 0 )
      {
        avail = sao.neighbourAvailability( cs, ctuIdx );
      }
      any    = true;
      ctu[0] = int16_t( offset.typeIdc );
      ctu[2] = avail;
      if( offset.typeIdc == SAO_TYPE_BO )
      {
        ctu[1] = int16_t( offset.typeAuxInfo );
        for( int i = 0; i < 4; i++ )
        {
          ctu[4 + i] = int16_t( offset.offset[( offset.typeAuxInfo + i ) % NUM_SAO_BO_CLASSES] );
        }
      }
      else
      {
        ctu[4] = int16_t( offset.offset[0] );
        ctu[5] = int16_t( offset.offset[1] );
        ctu[6] = int16_t( offset.offset[3] );
        ctu[7] = int16_t( offset.offset[4] );
      }
    }
  }
  return any;
}

void GpuRecon::recordSao( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params )
{
  const PreCalcValues& pcv = *cs.pcv;
  const Api&           vk  = m_ctx->api();

  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_sao ) );
  for( int comp = 0; comp < (int) getNumberValidComponents( pcv.chrFormat ); comp++ )
  {
    const ComponentID compID = ComponentID( comp );
    SaoPush           push{};
    push.src         = src.addr[compID];
    push.dst         = dst.addr[compID];
    push.ctus        = params;
    push.srcStride   = src.stride[compID];
    push.dstStride   = dst.stride[compID];
    push.width       = pcv.lumaWidth >> getComponentScaleX( compID, pcv.chrFormat );
    push.height      = pcv.lumaHeight >> getComponentScaleY( compID, pcv.chrFormat );
    push.ctuWidth    = pcv.maxCUWidth >> getComponentScaleX( compID, pcv.chrFormat );
    push.ctuHeight   = pcv.maxCUHeight >> getComponentScaleY( compID, pcv.chrFormat );
    push.widthInCtus = pcv.widthInCtus;
    push.comp        = comp;
    push.bitDepth    = cs.sps->getBitDepth();
    push.scaleX      = getComponentScaleX( compID, pcv.chrFormat );
    push.scaleY      = getComponentScaleY( compID, pcv.chrFormat );
    setVirtualBoundaries( *cs.picHeader, push );
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    vk.vkCmdDispatch( m_cmd, ( push.width + 15 ) / 16, ( push.height + 15 ) / 16, 1 );
  }
  mark( "sao" );
}

GpuRecon::Planes GpuRecon::hostPlanes( const PelStorage& pic )
{
  Planes planes{};
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    planes.addr[comp]   = hostPlane( pic, ComponentID( comp ) )->host.addr + planeOffset( pic, ComponentID( comp ) );
    planes.stride[comp] = int32_t( pic.bufs[comp].stride );
  }
  return planes;
}

GpuRecon::Planes GpuRecon::twinPlanes( const PelStorage& pic )
{
  Planes planes{};
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    HostPlane* plane = hostPlane( pic, ComponentID( comp ) );
    if( !plane->twin.valid() )
    {
      plane->twin = m_ctx->allocPooled( size_t( gpuLayout( pic, ComponentID( comp ) ).size ) );
    }
    planes.addr[comp]   = plane->twin.addr + gpuLayout( pic, ComponentID( comp ) ).origin;
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

void GpuRecon::transferBarrier( bool toTransfer )
{
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = toTransfer ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = toTransfer ? VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  m_ctx->api().vkCmdPipelineBarrier( m_cmd, toTransfer ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     toTransfer ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                                     nullptr );
}

void GpuRecon::copyPlanes( const PelStorage& pic, int from, int to )
{
  const Api& vk = m_ctx->api();
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    const HostPlane* plane    = hostPlane( pic, ComponentID( comp ) );
    const Buffer*    bufs[3]  = { &plane->host, &plane->twin, &m_scratch[comp] };
    const GpuLayout  layout   = gpuLayout( pic, ComponentID( comp ) );
    VkBufferCopy     region{ from == kHost ? layout.hostOffset : 0, to == kHost ? layout.hostOffset : 0, layout.size };
    vk.vkCmdCopyBuffer( m_cmd, bufs[from]->buffer, bufs[to]->buffer, 1, &region );
  }
  mark( from == kHost ? "upload" : ( to == kHost ? "download" : "copy" ) );
}

// the push constants of shaders/deblock.slang
struct DeblockPush
{
  VkDeviceAddress pic0, pic1, ctus, edges;
  int32_t         stride, width, height, ctuLog2, widthInCtus, bitDepth, dir, scaleX, scaleY, count;
  int32_t         ladfNum, ladfOffset[5], ladfBound[4];
};

// the index of the lowest bit set in v (not 0)
static inline int lowestBit( uint32_t v )
{
#ifdef _MSC_VER
  unsigned long index;
  _BitScanForward( &index, v );
  return int( index );
#else
  return __builtin_ctz( v );
#endif
}

void GpuRecon::listEdges( const CodingStructure& cs, int ctuRsAddr, EdgeLists& edges )
{
  edges.clear();
  const PreCalcValues& pcv = *cs.pcv;
  const CtuData&       ctu = cs.getCtuData( ctuRsAddr );
  if( ctu.slice->getDeblockingFilterDisable() )
  {
    return;
  }
  const int  n      = int( pcv.maxCUWidth ) >> 2;   // 4x4 blocks across a CTU (32 at most: a row's fit the bits of an int)
  const int  x0     = ( ctuRsAddr % int( pcv.widthInCtus ) ) * int( pcv.maxCUWidth );
  const int  y0     = ( ctuRsAddr / int( pcv.widthInCtus ) ) * int( pcv.maxCUHeight );
  const bool chroma = isChromaEnabled( pcv.chrFormat );
  const int  sx     = chroma ? getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) : 0;
  const int  sy     = chroma ? getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) : 0;
  const int  cols   = std::min( n, ( int( pcv.lumaWidth ) - x0 + 3 ) >> 2 );
  const int  rows   = std::min( n, ( int( pcv.lumaHeight ) - y0 + 3 ) >> 2 );
  for( int dir = 0; dir < 2; dir++ )
  {
    const LoopFilterParam* lfp = ctu.lfParam[dir];
    // the luma grid of the chroma segments: across the edge every 8 chroma
    // samples (at the 4:2:0 rate), along it as many lines as a 4x4 luma block has
    const int across = dir == 0 ? ( ( 8 << sx ) / 4 * ( 4 >> sx ) ) << sx : ( ( 8 << sy ) / 4 * ( 4 >> sy ) ) << sy;
    const int gridX  = dir == 0 ? across : ( 4 >> sx ) << sx;
    const int gridY  = dir == 0 ? ( 4 >> sy ) << sy : across;
    CHECKD( ( gridX & ( gridX - 1 ) ) || ( gridY & ( gridY - 1 ) ), "the chroma grid is a power of 2" );
    // the 4x4 blocks inside the picture, row by row (no divisions: this runs
    // for every CTU, between its motion and its jobs)
    for( int r = 0; r < rows; r++ )
    {
      const LoopFilterParam* row      = lfp + r * n;
      const int              ly       = y0 + r * 4;
      const bool             chromaOn = chroma && ( ly & ( gridY - 1 ) ) == 0;
      // the row's blocks with an edge, as bits (no branch a block: most have none)
      uint32_t               any      = 0;
      for( int c = 0; c < cols; c++ )
      {
        any |= uint32_t( row[c].bs != 0 ) << c;
      }
      while( any )
      {
        const int              c  = lowestBit( any );
        any                      &= any - 1;
        const LoopFilterParam& p  = row[c];
        const int              lx = x0 + c * 4;
        if( p.bs & 3 )
        {
          edges.lists[dir].push_back(
            { uint32_t( lx ) | uint32_t( ly ) << 16, uint32_t( uint8_t( p.qp[0] ) ) | uint32_t( p.bs ) << 8 | uint32_t( p.sideMaxFiltLength ) << 16 } );
        }
        if( chromaOn && ( p.bs & 0x3c ) && ( lx & ( gridX - 1 ) ) == 0 )
        {
          edges.lists[2 + dir].push_back( { uint32_t( lx >> sx ) | uint32_t( ly >> sy ) << 16,
                                            uint32_t( uint8_t( p.qp[1] ) ) | uint32_t( uint8_t( p.qp[2] ) ) << 8 | uint32_t( p.bs ) << 16
                                              | uint32_t( p.flags ) << 24 } );
        }
      }
    }
  }
}

bool GpuRecon::filtersSupported( const CodingStructure& cs )
{
  return true;
}

size_t GpuRecon::deblockParamSize( const CodingStructure& cs )
{
  return ( size_t( cs.pcv->sizeInCtus ) * 8 + 255 ) & ~size_t( 255 );
}

bool GpuRecon::prepareDeblock( const CodingStructure& cs, uint8_t* base )
{
  bool any = false;
  for( int ctuIdx = 0; ctuIdx < (int) cs.pcv->sizeInCtus; ctuIdx++ )
  {
    const Slice& slice = *cs.getCtuData( ctuIdx ).slice;
    int8_t*      ctu   = reinterpret_cast<int8_t*>( base + ctuIdx * 8 );
    ctu[0] = slice.getDeblockingFilterDisable() ? 1 : 0;
    ctu[1] = int8_t( slice.getDeblockingFilterBetaOffsetDiv2() );
    ctu[2] = int8_t( slice.getDeblockingFilterTcOffsetDiv2() );
    ctu[3] = int8_t( slice.getDeblockingFilterCbBetaOffsetDiv2() );
    ctu[4] = int8_t( slice.getDeblockingFilterCbTcOffsetDiv2() );
    ctu[5] = int8_t( slice.getDeblockingFilterCrBetaOffsetDiv2() );
    ctu[6] = int8_t( slice.getDeblockingFilterCrTcOffsetDiv2() );
    ctu[7] = 0;
    any |= !slice.getDeblockingFilterDisable();
  }
  return any;
}

void GpuRecon::recordDeblock( const CodingStructure& cs, const Planes& planes, VkDeviceAddress params, const std::vector<EdgeLists>& edges )
{
  const PreCalcValues& pcv = *cs.pcv;
  const Api&           vk  = m_ctx->api();
  const bool           chroma = isChromaEnabled( pcv.chrFormat );

  // the edges, list after list
  size_t first[5] = { 0, 0, 0, 0, 0 };
  for( int l = 0; l < 4; l++ )
  {
    first[l + 1] = first[l];
    for( const EdgeLists& ctu: edges )
    {
      first[l + 1] += ctu.lists[l].size();
    }
  }
  if( m_edges.size < first[4] * sizeof( Edge ) )
  {
    m_ctx->free( m_edges );
    m_edges = m_ctx->alloc( std::max<size_t>( first[4], 1024 ) * 2 * sizeof( Edge ), MemKind::Upload );
  }
  Edge* all = reinterpret_cast<Edge*>( m_edges.mapped );
  for( int l = 0; l < 4; l++ )
  {
    for( const EdgeLists& ctu: edges )
    {
      memcpy( all, ctu.lists[l].data(), ctu.lists[l].size() * sizeof( Edge ) );
      all += ctu.lists[l].size();
    }
  }

  DeblockPush push{};
  push.ctus        = params;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  push.bitDepth    = cs.sps->getBitDepth();
  push.scaleX      = chroma ? getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) : 0;
  push.scaleY      = chroma ? getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) : 0;
  // LADF (the luma's): its intervals' QP offsets and lower bounds
  const SPS& sps = *cs.sps;
  if( sps.getLadfEnabled() )
  {
    push.ladfNum = sps.getLadfNumIntervals();
    for( int k = 0; k < push.ladfNum; k++ )
    {
      push.ladfOffset[k] = sps.getLadfQpOffset( k );
      if( k > 0 )
      {
        push.ladfBound[k - 1] = sps.getLadfIntervalLowerBound( k );
      }
    }
  }

  for( int dir = 0; dir < 2; dir++ )
  {
    if( dir == 1 )
    {
      computeBarrier();   // the horizontal edges take what the vertical ones left
    }
    push.dir = dir;

    // a thread per edge of the list (the luma filters load and store 4
    // samples at a time, from multiples of 4: the rows 8-byte aligned)
    CHECK_FATAL( planes.addr[COMPONENT_Y] % 8 != 0 || planes.stride[COMPONENT_Y] % 4 != 0, "Vulkan: the luma rows are not 8-byte aligned" );
    push.pic0   = planes.addr[COMPONENT_Y];
    push.pic1   = 0;
    push.stride = planes.stride[COMPONENT_Y];
    push.width  = pcv.lumaWidth;
    push.height = pcv.lumaHeight;
    push.edges  = m_edges.addr + first[dir] * sizeof( Edge );
    push.count  = int32_t( first[dir + 1] - first[dir] );
    if( push.count > 0 )
    {
      vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( dir == 0 ? kShader_deblock_luma_ver : kShader_deblock_luma_hor ) );
      vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
      vk.vkCmdDispatch( m_cmd, uint32_t( ( push.count + 63 ) / 64 ), 1, 1 );
    }

    if( chroma )
    {
      CHECK( planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );
      push.pic0   = planes.addr[COMPONENT_Cb];
      push.pic1   = planes.addr[COMPONENT_Cr];
      push.stride = planes.stride[COMPONENT_Cb];
      push.width  = pcv.lumaWidth >> push.scaleX;
      push.height = pcv.lumaHeight >> push.scaleY;
      push.edges  = m_edges.addr + first[2 + dir] * sizeof( Edge );
      push.count  = int32_t( first[3 + dir] - first[2 + dir] );
      if( push.count > 0 )
      {
        vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_deblock_chroma ) );
        vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
        vk.vkCmdDispatch( m_cmd, uint32_t( ( push.count + 63 ) / 64 ), 1, 1 );
      }
    }
    mark( dir == 0 ? "deblock ver" : "deblock hor" );
  }
}

// the push constants of shaders/intra.slang
struct IntraPush
{
  VkDeviceAddress plane0, plane1, plane2, jobs, waits, sync, mip, lmcsScale, defer0, defer1, defer2;
  int32_t         stride0, stride12, count, bitDepth, ctuLog2, scaleX, scaleY, collocated, picWidth, picHeight;
};

bool GpuRecon::intraSupported( const CodingStructure& cs )
{
  return intraSupported( *cs.sps );
}

bool GpuRecon::intraSupported( const SPS& sps )
{
  return !sps.getUseColorTrans();
}

void GpuRecon::recordIntra( const CodingStructure& cs, const std::vector<std::vector<IntraJob>>& jobs, const Planes& planes )
{
  // the jobs, wave by wave, a big block in bands of rows of kBand samples at
  // most, a workgroup each: a wave lasts as long as its slowest workgroup,
  // and that is the one with the most samples a thread (shaders/intra.slang)
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
    return;
  }
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
  // the jobs, then each one's count of the jobs of the waves before its, which
  // it waits for (shaders/intra.slang)
  const size_t size = total * ( sizeof( IntraJob ) + sizeof( uint32_t ) );
  if( m_intraJobs.size < size )
  {
    m_ctx->free( m_intraJobs );
    m_intraJobs = m_ctx->alloc( size * 2, MemKind::Upload );
  }
  IntraJob*             sorted = reinterpret_cast<IntraJob*>( m_intraJobs.mapped );
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
        waits[at] = first[job.wave()];
      }
    }
  }
  if( !m_intraSync.valid() )
  {
    m_intraSync = m_ctx->alloc( 256, MemKind::Device );
  }

  const PreCalcValues& pcv = *cs.pcv;
  const Api&           vk  = m_ctx->api();
  IntraPush            push{};
  push.plane0     = planes.addr[COMPONENT_Y];
  push.plane1     = planes.addr[COMPONENT_Cb];
  push.plane2     = planes.addr[COMPONENT_Cr];
  push.jobs       = m_intraJobs.addr;
  push.waits      = m_intraJobs.addr + total * sizeof( IntraJob );
  push.sync       = m_intraSync.addr;
  push.mip        = m_mip.addr;
  push.count      = int32_t( total );
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
  // the counts of the jobs taken and done from zero; they and what the work
  // before wrote (the residuals, inter's predictions) visible to the jobs
  vk.vkCmdFillBuffer( m_cmd, m_intraSync.buffer, 0, 2 * sizeof( uint32_t ), 0 );
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                           &barrier, 0, nullptr, 0, nullptr );
  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_intra ) );
  vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
  // all the waves: a workgroup a job, in rows of 32768 (shaders/intra.slang)
  vk.vkCmdDispatch( m_cmd, uint32_t( std::min<size_t>( total, 32768 ) ), uint32_t( ( total + 32767 ) / 32768 ), 1 );
  mark( "intra" );
}

// the push constants of shaders/inter.slang
struct InterPush
{
  VkDeviceAddress jobs, refs, dst0, dst1, dst2, dmvr, colMotion, lmcsFwd, defer0, defer1, defer2, geo, subpics;
  int32_t         stride0, stride12, count, picWidth, picHeight, ctuSize, bitDepth, ctuLog2, widthInCtus, num8x8, wide, scaleX, scaleY, numComp, wrapSps,
                  wrapPps, wrapOffset, subpicOn, curWinLeft, curWinTop, horColloc, verColloc;
};
// the colocated motion as shaders/inter.slang writes DMVR's refinements into it
static_assert( sizeof( ColocatedMotionInfo ) == 20 && offsetof( ColocatedMotionInfo, mv ) == 0 && sizeof( Mv ) == 8,
               "the colocated motion is not laid out as shaders/inter.slang writes it" );

bool GpuRecon::interSupported( const SPS& sps )
{
  return intraSupported( sps ) && sps.getBitDepth() <= 10;
}

bool GpuRecon::interSupported( const CodingStructure& cs )
{
  return interSupported( *cs.sps );
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

void GpuRecon::recordInter( const CodingStructure& cs, const std::vector<std::vector<InterJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf,
                            const Planes& planes )
{
  size_t total = 0;
  for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
  {
    total += jobs[ctu].size();
  }
  if( total == 0 )
  {
    return;
  }
  const size_t size = total * sizeof( InterJob );
  if( buf.size < size )
  {
    m_ctx->free( buf );
    buf = m_ctx->alloc( size * 2, MemKind::Upload );
  }
  // the BDOF tiles after the others, then the affine ones (two jobs each),
  // for dispatches of the inter_bdof and inter_affine variants of their own
  // (their code would slow the others down)
  size_t bdofs = 0, affines = 0;
  for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
  {
    for( size_t i = 0; i < jobs[ctu].size(); i++ )
    {
      const uint32_t flags = jobs[ctu][i].w[1] >> 16;
      bdofs   += flags & InterJob::kBdof ? 1 : 0;
      affines += flags & InterJob::kAffine ? 2 : 0;
      i       += flags & InterJob::kAffine ? 1 : 0;
    }
  }
  const size_t regular = total - bdofs - affines;
  InterJob*    all     = reinterpret_cast<InterJob*>( buf.mapped );
  InterJob*    bdof    = all + regular;
  InterJob*    affine  = bdof + bdofs;
  for( size_t ctu = ctuBegin; ctu < ctuEnd; ctu++ )
  {
    if( bdofs + affines == 0 )
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

  const PreCalcValues& pcv = *cs.pcv;
  InterPush            push{};
  push.jobs      = buf.addr;
  push.refs      = m_refs.addr;
  push.dst0      = planes.addr[COMPONENT_Y];
  push.dst1      = planes.addr[COMPONENT_Cb];
  push.dst2      = planes.addr[COMPONENT_Cr];
  push.dmvr      = m_dmvr.addr;
  // DMVR's refinements into the picture's colocated motion (TMVP), in the
  // memory of the GPU's picture allocator (Picture::gpuMemory)
  CHECK_FATAL( !cs.m_colMiAllocation, "Vulkan: the colocated motion out of the GPU's reach" );
  push.colMotion   = static_cast<HostPlane*>( cs.m_colMiAllocation )->host.addr;
  if( m_lmcsOn )
  {
    push.lmcsFwd = m_lmcs.addr;
  }
  push.defer0 = m_deferPlanes.addr[COMPONENT_Y];
  push.defer1 = m_deferPlanes.addr[COMPONENT_Cb];
  push.defer2 = m_deferPlanes.addr[COMPONENT_Cr];
  push.geo    = m_geo.addr;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  push.num8x8      = int32_t( pcv.num8x8CtuBlks );
  push.stride0   = planes.stride[COMPONENT_Y];
  push.stride12  = planes.stride[COMPONENT_Cb];
  push.count     = int32_t( regular );
  push.picWidth  = pcv.lumaWidth;
  push.picHeight = pcv.lumaHeight;
  push.ctuSize   = pcv.maxCUWidth;
  push.bitDepth  = cs.sps->getBitDepth();
  push.numComp   = int32_t( getNumberValidComponents( pcv.chrFormat ) );
  push.scaleX    = push.numComp > 1 ? getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) : 0;
  push.scaleY    = push.numComp > 1 ? getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) : 0;
  // the shader's tile16 loads the planes' rows in aligned pieces of 16 and 8
  // bytes, up to 7 luma (3 chroma) samples before a window and 9 (5) after it:
  // the rows must begin 16-byte aligned, the margins be wide enough (the
  // reference pictures' planes are laid out as this one's); its chroma windows
  // are 4:2:0's, it reads no wraparound copy
  push.wrapSps    = cs.sps->getUseWrapAround() ? 1 : 0;
  push.wrapPps    = cs.pps->getUseWrapAround() ? 1 : 0;
  push.wrapOffset = cs.pps->getUseWrapAround() ? cs.pps->getWrapAroundOffset() : 0;
  // subpictures treated as pictures: each CTU's rectangle (its subpicture's or
  // the picture's), written with the picture's first jobs
  const PPS& pps = *cs.pps;
  for( int i = 0; pps.getNumSubPics() > 1 && i < pps.getNumSubPics(); i++ )
  {
    push.subpicOn |= pps.getSubPic( i ).getTreatedAsPicFlag() ? 1 : 0;
  }
  if( push.subpicOn )
  {
    const size_t size = ( size_t( pcv.sizeInCtus ) * 4 * sizeof( uint16_t ) + 255 ) & ~size_t( 255 );
    if( ctuBegin == 0 )
    {
      if( m_subpics.size < size )
      {
        m_ctx->free( m_subpics );
        m_subpics = m_ctx->alloc( size, MemKind::Upload );
      }
      uint16_t* rects = reinterpret_cast<uint16_t*>( m_subpics.mapped );
      for( unsigned ctu = 0; ctu < pcv.sizeInCtus; ctu++ )
      {
        const Position pos( ( ctu % pcv.widthInCtus ) * pcv.maxCUWidth, ( ctu / pcv.widthInCtus ) * pcv.maxCUHeight );
        const SubPic&  sub     = pps.getSubPicFromPos( pos );
        const bool     asPic   = sub.getTreatedAsPicFlag();
        rects[ctu * 4]         = uint16_t( asPic ? sub.getSubPicLeft() : 0 );
        rects[ctu * 4 + 1]     = uint16_t( asPic ? sub.getSubPicTop() : 0 );
        rects[ctu * 4 + 2]     = uint16_t( asPic ? sub.getSubPicRight() : pcv.lumaWidth - 1 );
        rects[ctu * 4 + 3]     = uint16_t( asPic ? sub.getSubPicBottom() : pcv.lumaHeight - 1 );
      }
    }
    push.subpics = m_subpics.addr;
  }
  // RPR: the picture's scaling window, its chroma's siting; a scaled
  // reference anywhere
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
  // (tile16's reads keep to the picture, from references of its size)
  push.wide = pcv.chrFormat == CHROMA_420 && !push.wrapSps && !push.subpicOn && !scaled ? 1 : 0;
  for( int comp = 0; comp < (int) cs.picture->m_bufs[PIC_RECONSTRUCTION].bufs.size(); comp++ )
  {
    const PelStorage& pic    = cs.picture->m_bufs[PIC_RECONSTRUCTION];
    const VkDeviceSize left  = gpuLayout( pic, ComponentID( comp ) ).origin / sizeof( Pel );
    const VkDeviceSize right = VkDeviceSize( pic.bufs[comp].stride ) - left - pic.bufs[comp].width;
    if( planes.addr[comp] % 16 != 0 || pic.bufs[comp].stride * sizeof( Pel ) % 16 != 0 || left < 16 || right < 16 )
    {
      push.wide = 0;
    }
  }

  const Api& vk = m_ctx->api();
  if( regular > 0 )
  {
    vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_inter ) );
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    // a workgroup per job, in rows of 32768 (shaders/inter.slang)
    vk.vkCmdDispatch( m_cmd, uint32_t( std::min<size_t>( regular, 32768 ) ), uint32_t( ( regular + 32767 ) / 32768 ), 1 );
  }
  // (the tiles of the dispatches apart: no barriers between them)
  if( bdofs > 0 )
  {
    push.jobs  = buf.addr + regular * sizeof( InterJob );
    push.count = int32_t( bdofs );
    vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_inter_bdof ) );
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    vk.vkCmdDispatch( m_cmd, uint32_t( std::min<size_t>( bdofs, 32768 ) ), uint32_t( ( bdofs + 32767 ) / 32768 ), 1 );
  }
  if( affines > 0 )
  {
    push.jobs  = buf.addr + ( regular + bdofs ) * sizeof( InterJob );
    push.count = int32_t( affines / 2 );
    vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_inter_affine ) );
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    vk.vkCmdDispatch( m_cmd, uint32_t( std::min<size_t>( affines / 2, 32768 ) ), uint32_t( ( affines / 2 + 32767 ) / 32768 ), 1 );
  }
  mark( "inter" );
}

// the push constants of shaders/trafo.slang
struct TrafoPush
{
  VkDeviceAddress jobs, plane0, plane1, plane2, levels0, levels1, levels2, matrices, lfnst, scaling;
  int32_t         stride0, stride12, count, bitDepth;
};

bool GpuRecon::residualSupported( const CodingStructure& cs )
{
  return cs.sps->getBitDepth() <= 10;
}

void GpuRecon::recordTrafo( const CodingStructure& cs, const std::vector<std::vector<TrafoJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf,
                            const Planes& levels, const Planes& planes )
{
  size_t total = 0;
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

  TrafoPush push{};
  push.jobs     = buf.addr;
  push.plane0   = planes.addr[COMPONENT_Y];
  push.plane1   = planes.addr[COMPONENT_Cb];
  push.plane2   = planes.addr[COMPONENT_Cr];
  push.levels0  = levels.addr[COMPONENT_Y];
  push.levels1  = levels.addr[COMPONENT_Cb];
  push.levels2  = levels.addr[COMPONENT_Cr];
  push.matrices = m_matrices.addr;
  push.lfnst    = m_lfnst.addr;
  push.scaling  = m_scalingOn ? m_scaling.addr : 0;
  push.stride0  = planes.stride[COMPONENT_Y];
  push.stride12 = planes.stride[COMPONENT_Cb];
  push.count    = int32_t( total );
  push.bitDepth = cs.sps->getBitDepth();
  CHECK( planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );
  CHECK( levels.stride[COMPONENT_Y] != planes.stride[COMPONENT_Y] || levels.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cb],
         "the levels laid out differently" );

  const Api& vk = m_ctx->api();
  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_trafo ) );
  vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
  // a workgroup per job, in rows of 32768 (shaders/trafo.slang)
  vk.vkCmdDispatch( m_cmd, uint32_t( std::min<size_t>( total, 32768 ) ), uint32_t( ( total + 32767 ) / 32768 ), 1 );
  mark( "trafo" );
}

void GpuRecon::prepareFilters( CodingStructure& cs, bool deblock, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf )
{
  m_prepared.dbSize  = deblockParamSize( cs );
  m_prepared.alfSize = alf ? alfParamSize( cs ) : 0;
  m_prepared.saoSize = sao ? saoParamSize( cs ) : 0;
  // LMCS: per CTU, whether its slice uses it (rspCtuBcw maps its luma back)
  m_prepared.lmcsSize = m_lmcsOn ? ( size_t( cs.pcv->sizeInCtus ) + 255 ) & ~size_t( 255 ) : 0;
  uint8_t* base       = params( m_prepared.dbSize + m_prepared.alfSize + m_prepared.saoSize + m_prepared.lmcsSize );
  if( m_lmcsOn )
  {
    uint8_t* ctus = base + m_prepared.dbSize + m_prepared.alfSize + m_prepared.saoSize;
    for( unsigned ctu = 0; ctu < cs.pcv->sizeInCtus; ctu++ )
    {
      ctus[ctu] = cs.getCtuData( ctu ).slice->getLmcsEnabledFlag() ? 1 : 0;
    }
  }
  m_prepared.deblock = deblock && prepareDeblock( cs, base );
  m_prepared.sao     = sao && prepareSao( cs, *sao, base + m_prepared.dbSize + m_prepared.alfSize );
  m_prepared.alfBounds = alf && prepareAlf( cs, *alf, base + m_prepared.dbSize );
  m_prepared.picture = cs.picture;
}

void GpuRecon::setPicture( const PelStorage& pic, const LmcsTables* tables, bool defer, const int* scalingMatrices, size_t scalingCount )
{
  // the instance's picture before is done: its work no more reads these
  m_scalingOn = scalingMatrices != nullptr;
  if( scalingMatrices )
  {
    // the matrices' factors, a byte each (the scaling lists' coefficients:
    // 1 .. 255, 0 where a matrix has none)
    if( m_scaling.size < scalingCount )
    {
      m_ctx->free( m_scaling );
      m_scaling = m_ctx->alloc( ( scalingCount + 255 ) & ~size_t( 255 ), MemKind::Upload );
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
  if( defer )
  {
    // the planes inter's predictions are kept aside in: laid out as the
    // picture's planes (from their first sample)
    size_t size = 0;
    size_t at[MAX_NUM_COMPONENT];
    for( size_t comp = 0; comp < pic.bufs.size(); comp++ )
    {
      at[comp] = size;
      size    += ( size_t( pic.bufs[comp].stride ) * pic.bufs[comp].height * sizeof( Pel ) + 255 ) & ~size_t( 255 );
    }
    if( m_defer.size < size )
    {
      m_ctx->free( m_defer );
      m_defer = m_ctx->alloc( size, MemKind::Device );
    }
    for( size_t comp = 0; comp < pic.bufs.size(); comp++ )
    {
      m_deferPlanes.addr[comp]   = m_defer.addr + at[comp];
      m_deferPlanes.stride[comp] = pic.bufs[comp].stride;
    }
  }
}

// the push constants of shaders/lmcs.slang
struct LmcsPush
{
  VkDeviceAddress plane, lut, ctus;
  int32_t         stride, width, height, ctuLog2, widthInCtus;
};

void GpuRecon::recordLmcs( const CodingStructure& cs, const Planes& planes, VkDeviceAddress ctus )
{
  const PreCalcValues& pcv = *cs.pcv;
  LmcsPush             push{};
  push.plane       = planes.addr[COMPONENT_Y];
  push.lut         = m_lmcs.addr + m_lmcsCount * sizeof( int16_t );
  push.ctus        = ctus;
  push.stride      = planes.stride[COMPONENT_Y];
  push.width       = pcv.lumaWidth;
  push.height      = pcv.lumaHeight;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  CHECK( pcv.lumaWidth % 4 != 0, "Vulkan: LMCS takes 4 luma samples at a time" );

  const Api& vk = m_ctx->api();
  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_lmcs ) );
  vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
  // a workgroup 256 samples of a row (shaders/lmcs.slang)
  vk.vkCmdDispatch( m_cmd, uint32_t( ( pcv.lumaWidth + 255 ) / 256 ), uint32_t( pcv.lumaHeight ), 1 );
  mark( "lmcs" );
}

// a reference picture as shaders/inter.slang takes it (RefPlanes)
struct RefEntry
{
  VkDeviceAddress y, cb, cr;
  uint32_t        size, strides;
  int32_t         ratioX, ratioY, winLeft, winTop, scaled, unused[3];
};
static_assert( sizeof( RefEntry ) == 64, "the reference table laid out differently from shaders/inter.slang's" );

void GpuRecon::writeRefTable( const CodingStructure& cs, const RefTable& refs, const PelStorage& pic, std::vector<const Picture*>& uploads )
{
  const size_t size = std::max<size_t>( refs.pictures().size(), 1 ) * sizeof( RefEntry );
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
    // RPR: the scaling ratio of the slices' reference (the picture's own for
    // every slice), the reference's scaling window (xPredInterBlkRPR)
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

void GpuRecon::startBand( CodingStructure& cs, const PictureJobs& jobs, size_t ctus, PelStorage& pic )
{
  const Api&                  vk = m_ctx->api();
  std::vector<const Picture*> uploads;
  if( jobs.inter )
  {
    writeRefTable( cs, *jobs.refs, pic, uploads );
  }

  // the band's commands (the pool reset: the instance's previous picture is
  // done), after what earlier submissions wrote
  m_cmd = m_cmdBand;
  vk.vkResetCommandPool( m_ctx->device(), m_pool, 0 );
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( m_cmd, &begin );
  m_bandMarks = 0;
  if( m_bandQueries )
  {
    vk.vkCmdResetQueryPool( m_cmd, m_bandQueries, 0, kBandMarks );
    vk.vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_bandQueries, m_bandMarks++ );
  }
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
  for( const Picture* refPic: uploads )
  {
    copyPlanes( refPic->m_bufs[PIC_RECONSTRUCTION], kHost, kTwin );
  }
  transferBarrier( false );
  const Planes planes = twinPlanes( pic );
  recordTrafo( cs, *jobs.trafo, 0, ctus, m_bandTrafoJobs, hostPlanes( pic ), planes );
  if( jobs.inter )
  {
    computeBarrier();
    recordInter( cs, *jobs.inter, 0, ctus, m_bandInterJobs, planes );
  }
  vk.vkEndCommandBuffer( m_cmd );
  // to the GPU now: the first part's commands, queued after these, begin with
  // a barrier over them
  m_ctx->submitFirst( m_cmd, VK_NULL_HANDLE );
  for( const Picture* refPic: uploads )
  {
    refPic->gpuCurrent = true;   // for the work queued after this
  }
}

int GpuRecon::startPicture( CodingStructure& cs, const PictureJobs& jobs, bool deblock, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf,
                            PelStorage& pic, bool waitFirst )
{
  if( m_prepared.picture != cs.picture )
  {
    prepareFilters( cs, deblock, sao, alf );
  }
  m_prepared.picture   = nullptr;   // the next picture's anew
  const size_t dbSize  = m_prepared.dbSize;
  const size_t alfSize = m_prepared.alfSize;
  const size_t saoSize = m_prepared.saoSize;

  // after a band (startBand): the command pool reset, the reference table
  // written, the references uploaded
  const bool                  band = jobs.bandCtus > 0;
  std::vector<const Picture*> uploads;
  if( jobs.inter && !band )
  {
    writeRefTable( cs, *jobs.refs, pic, uploads );
  }
  if( !band )
  {
    m_bandMarks = 0;
  }

  const bool doDeblock = m_prepared.deblock;
  const bool doSao     = m_prepared.sao;
  // in the GPU's own memory, which its shaders reach faster than the host's:
  // the picture goes in (what the CPU made of it), is finished there, stays
  // there as a reference, and comes back for the output. When the GPU makes
  // all of it, from the residuals on, nothing goes in: the residual pass reads
  // the levels from the host's picture.
  m_cmd = m_cmdFirst;
  beginCommands( !band );
  for( const Picture* refPic: uploads )
  {
    copyPlanes( refPic->m_bufs[PIC_RECONSTRUCTION], kHost, kTwin );
  }
  Planes planes[2] = { twinPlanes( pic ), Planes{} };   // the scratch planes taken just before SAO and ALF
  int    cur       = 0;
  if( !jobs.trafo )
  {
    copyPlanes( pic, kHost, kTwin );
  }
  transferBarrier( false );
  if( jobs.trafo )
  {
    recordTrafo( cs, *jobs.trafo, jobs.bandCtus, jobs.trafo->size(), m_trafoJobs, hostPlanes( pic ), planes[cur] );
    computeBarrier();
  }
  if( jobs.inter )
  {
    recordInter( cs, *jobs.inter, jobs.bandCtus, jobs.inter->size(), m_interJobs, planes[cur] );   // in place
    computeBarrier();
  }
  splitCommands();
  if( jobs.intra )
  {
    recordIntra( cs, *jobs.intra, planes[cur] );   // in place
    computeBarrier();
    if( m_lmcsOn )
    {
      recordLmcs( cs, planes[cur], m_params.addr + dbSize + alfSize + saoSize );   // in place
      computeBarrier();
    }
  }
  if( doDeblock )
  {
    recordDeblock( cs, planes[cur], m_params.addr, *jobs.edges );   // in place
  }
  if( doSao || alf )
  {
    planes[1] = scratchPlanes( pic );
  }
  if( doSao )
  {
    if( doDeblock )
    {
      computeBarrier();
    }
    recordSao( cs, planes[cur], planes[1 - cur], m_params.addr + dbSize + alfSize );
    cur = 1 - cur;
  }
  if( alf )
  {
    if( doDeblock || doSao )
    {
      computeBarrier();
    }
    recordAlf( cs, planes[cur], planes[1 - cur], m_params.addr + dbSize );
    cur = 1 - cur;
  }
  transferBarrier( true );
  if( cur == 1 )
  {
    // the result is in the scratch planes: they become the picture's own
    // planes on the GPU, and its former ones the scratch planes (same sizes)
    for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
    {
      std::swap( hostPlane( pic, ComponentID( comp ) )->twin, m_scratch[comp] );
    }
  }
  const int download = submitPicture( pic, waitFirst );
  // the scratch planes (after a swap, the picture's former ones) back to the
  // pool at once: the GPU runs the pictures' work one after the other (each
  // part begins with a barrier over all the work before it), so the work of
  // whichever picture takes them next comes after this one's (the download
  // reads none of them)
  for( Buffer& buf: m_scratch )
  {
    m_ctx->recycle( buf );
  }
  for( const Picture* refPic: uploads )
  {
    refPic->gpuCurrent = true;   // for the work queued after this
  }
  return download;
}

}   // namespace gpu
}   // namespace vvdec
