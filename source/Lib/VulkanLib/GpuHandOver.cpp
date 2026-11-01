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
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

------------------------------------------------------------------------------------------- */


/** \file     GpuHandOver.cpp
    \brief    Decoded pictures copied into another Vulkan device's buffers on the same GPU
*/

#include "GpuHandOver.h"

#include "GpuPictureMemory.h"
#include "vvdec_gpu/GpuShaderIds.h"

#include "CommonLib/Picture.h"

namespace vvdec
{
namespace gpu
{

// handover.slang's push constants
struct HandOverParams
{
  VkDeviceAddress src;
  VkDeviceAddress dst;
  int32_t         srcStride;   // samples
  int32_t         dstStride;   // bytes
  int32_t         width;
  int32_t         height;
  int32_t         dstBytes;
  int32_t         shift;
};
static constexpr int kHandOverSegment = 1024;   // the samples of a row a workgroup takes (handover.slang)

HandOver::HandOver( Context* context ) : m_ctx( context ), m_copies( kCopies )
{
  for( Copy& copy: m_copies )
  {
    copy.pool = m_ctx->createCommandPool( m_ctx->queueFamily() );
    m_ctx->allocateCommandBuffers( copy.pool, 1, &copy.cmd );
    copy.fence = m_ctx->createFence();
  }
}

HandOver::~HandOver()
{
  wait();
  for( Copy& copy: m_copies )
  {
    m_ctx->destroyFence( copy.fence );
    m_ctx->destroyCommandPool( copy.pool );   // and its buffer
  }
}

void HandOver::wait()
{
  std::lock_guard<std::mutex> lock( m_mutex );
  for( Copy& copy: m_copies )
  {
    if( copy.pending )
    {
      m_ctx->api().vkWaitForFences( m_ctx->device(), 1, &copy.fence, VK_TRUE, UINT64_MAX );
      copy.pending = false;
    }
  }
}

uint64_t HandOver::pack( Picture& pic, PlanePool* pool, ComponentID comp, const Buffer& dst, VkDeviceSize dstOffset, uint32_t dstStride,
                         int dstBytes, int shift, int x, int y, int width, int height )
{
  CHECK_FATAL( pic.progress < Picture::reconstructed, "Vulkan: a picture handed over before its work on the GPU is done" );
  HandOverParams params{};
  {
    // The twin's records stay put meanwhile (a picture the pool handles on another thread)
    std::unique_lock<std::mutex> planesLock;
    if( pool )
    {
      planesLock = pool->lockPlanes();
    }
    const PelStorage& planes = pic.m_bufs[PIC_RECONSTRUCTION];
    const Buffer&     twin   = planeTwin( planes, comp );
    CHECK_FATAL( !twin.valid(), "Vulkan: a picture handed over without its twins" );
    const int stride = int( planes.bufs[comp].stride );
    params.src       = twin.addr + gpuLayout( planes, comp ).origin + VkDeviceSize( y * stride + x ) * sizeof( Pel );
    params.srcStride = stride;
  }
  params.dst       = dst.addr + dstOffset;
  params.dstStride = int32_t( dstStride );
  params.width     = width;
  params.height    = height;
  params.dstBytes  = dstBytes;
  params.shift     = shift;

  std::lock_guard<std::mutex> lock( m_mutex );
  Copy&                       copy = m_copies[m_next];
  m_next                           = ( m_next + 1 ) % m_copies.size();
  const Api&                  vk   = m_ctx->api();
  if( copy.pending )
  {
    vk.vkWaitForFences( m_ctx->device(), 1, &copy.fence, VK_TRUE, UINT64_MAX );
  }
  vk.vkResetCommandPool( m_ctx->device(), copy.pool, 0 );
  m_ctx->beginOneTime( copy.cmd );
  // After the picture's reconstruction (earlier work on this queue), and after the other device's last read of dst,
  // which its own submissions finished before it asked for this copy
  m_ctx->memoryBarrier( copy.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT );
  vk.vkCmdBindPipeline( copy.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_handover ) );
  vk.vkCmdPushConstants( copy.cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( params ), &params );
  vk.vkCmdDispatch( copy.cmd, uint32_t( ( width + kHandOverSegment - 1 ) / kHandOverSegment ), uint32_t( height ), 1 );
  vk.vkEndCommandBuffer( copy.cmd );
  vk.vkResetFences( m_ctx->device(), 1, &copy.fence );
  copy.pending = true;
  // The timeline's signal makes the copy's writes available to the other device
  return m_ctx->submitHandOver( copy.cmd, copy.fence );
}

}   // namespace gpu
}   // namespace vvdec
