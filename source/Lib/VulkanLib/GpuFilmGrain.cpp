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

/** \file     GpuFilmGrain.cpp
    \brief    film grain synthesis of the output frames on the GPU
*/

#include "GpuFilmGrain.h"

#include "vvdec_gpu/GpuShaders.h"

#include "CommonLib/CommonDef.h"
#include "FilmGrain/FilmGrain.h"

#include <cstring>

namespace vvdec
{
namespace gpu
{

// FilmGrainImpl's patterns [2][VFGS_MAX_PATTERNS + 1][64][64], then sLUT[3][256] and pLUT[3][256]
static constexpr size_t kPatternBytes = 2 * ( VFGS_MAX_PATTERNS + 1 ) * 64 * 64;
static constexpr size_t kTableBytes   = kPatternBytes + 2 * 3 * 256;

// the push constants of shaders/filmgrain.slang
struct FilmGrainPush
{
  VkDeviceAddress src, dst, pattern, luts, seeds;
  int32_t         srcStride, dstStride, width, readWidth, lumaWidth, comp, subx, suby, bs, scaleShift, dstBytes, present;
};
static_assert( sizeof( FilmGrainPush ) == 88, "FilmGrainPush: the layout of filmgrain.slang's Params" );

GpuFilmGrain::GpuFilmGrain( Context* ctx ) : m_ctx( ctx )
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
  cmdInfo.commandBufferCount = 1;
  if( vk.vkAllocateCommandBuffers( m_ctx->device(), &cmdInfo, &m_cmd ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkAllocateCommandBuffers failed" );
  }
  VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  if( vk.vkCreateFence( m_ctx->device(), &fenceInfo, nullptr, &m_fence ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateFence failed" );
  }
  m_tables = m_ctx->alloc( kTableBytes, MemKind::Upload );
}

GpuFilmGrain::~GpuFilmGrain()
{
  const Api& vk = m_ctx->api();
  vk.vkDestroyFence( m_ctx->device(), m_fence, nullptr );
  vk.vkDestroyCommandPool( m_ctx->device(), m_pool, nullptr );
  m_ctx->free( m_tables );
  m_ctx->free( m_seeds );
}

uint32_t* GpuFilmGrain::seeds( size_t count )
{
  const size_t size = count * sizeof( uint32_t );
  if( m_seeds.size < size )
  {
    m_ctx->free( m_seeds );
    m_seeds = m_ctx->alloc( ( size + 4095 ) & ~size_t( 4095 ), MemKind::Upload );
  }
  return reinterpret_cast<uint32_t*>( m_seeds.mapped );
}

void GpuFilmGrain::apply( const FilmGrain& grain, int numPlanes, const Plane src[3], const Plane dst[3], int dstBytes, int lumaWidth, int lumaHeight )
{
  const Api&           vk   = m_ctx->api();
  const FilmGrainImpl& impl = grain.impl();
  // FilmGrainImpl::add_grain_block's checks, where a component has grain
  bool anyPresent = false;
  for( int c = 0; c < numPlanes; c++ )
  {
    anyPresent |= grain.componentPresent( c );
  }
  if( anyPresent )
  {
    CHECK( lumaWidth <= 128, "wrong width" );
    CHECK( impl.bitShift() != 0 && impl.bitShift() != 2, "wrong bs" );
    CHECK( impl.scaleShift() + impl.bitShift() < 8 || impl.scaleShift() + impl.bitShift() > 13, "wrong scale_shift" );
  }

  // the tables as the synthesis has them now (the GPU's work before is done)
  memcpy( m_tables.mapped, impl.patterns(), kPatternBytes );
  for( int c = 0; c < 3; c++ )
  {
    memcpy( m_tables.mapped + kPatternBytes + c * 256, impl.scaleLut( c ), 256 );
    memcpy( m_tables.mapped + kPatternBytes + 768 + c * 256, impl.patternLut( c ), 256 );
  }

  vk.vkResetCommandPool( m_ctx->device(), m_pool, 0 );
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( m_cmd, &begin );
  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_filmgrain ) );
  for( int c = 0; c < numPlanes; c++ )
  {
    FilmGrainPush push{};
    push.src        = src[c].addr;
    push.dst        = dst[c].addr;
    push.pattern    = m_tables.addr;
    push.luts       = m_tables.addr + kPatternBytes;
    push.seeds      = m_seeds.addr;
    push.srcStride  = src[c].stride;
    push.dstStride  = dst[c].stride;
    push.subx       = c ? impl.subX() : 1;
    push.suby       = c ? impl.subY() : 1;
    push.width      = lumaWidth / push.subx;
    push.readWidth  = src[c].readWidth;
    push.lumaWidth  = lumaWidth;
    push.comp       = c;
    push.bs         = impl.bitShift();
    push.scaleShift = impl.scaleShift();
    push.dstBytes   = dstBytes;
    push.present    = grain.componentPresent( c ) ? 1 : 0;
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    // a workgroup 1024 samples of a row
    vk.vkCmdDispatch( m_cmd, uint32_t( ( push.width + 1023 ) / 1024 ), uint32_t( lumaHeight / push.suby ), 1 );
  }
  // the host reads the frame after the fence
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
  vk.vkEndCommandBuffer( m_cmd );

  vk.vkResetFences( m_ctx->device(), 1, &m_fence );
  m_ctx->submit( m_cmd, m_fence );
  if( vk.vkWaitForFences( m_ctx->device(), 1, &m_fence, VK_TRUE, UINT64_MAX ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: the film grain's work failed" );
  }
}

}   // namespace gpu
}   // namespace vvdec
