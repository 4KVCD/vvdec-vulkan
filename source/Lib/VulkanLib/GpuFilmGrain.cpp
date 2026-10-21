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

#include "vvdec_gpu/GpuShaderIds.h"

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
  for( Slot& slot: m_slots )
  {
    slot.pool = m_ctx->createCommandPool( m_ctx->grainFamily() );
    m_ctx->allocateCommandBuffers( slot.pool, 1, &slot.cmd );
    slot.fence  = m_ctx->createFence();
    slot.tables = m_ctx->alloc( kTableBytes, MemKind::Upload );
  }
}

GpuFilmGrain::~GpuFilmGrain()
{
  for( Slot& slot: m_slots )
  {
    finish( slot );   // (the frames' grain done before the commands, tables and seeds go)
    m_ctx->destroyFence( slot.fence );
    m_ctx->destroyCommandPool( slot.pool );
    m_ctx->free( slot.tables );
    m_ctx->free( slot.seeds );
    m_ctx->free( slot.before );
    m_ctx->free( slot.after );
  }
}

void GpuFilmGrain::finish( Slot& slot )
{
  if( slot.ticket < 0 )
  {
    return;
  }
  if( m_ctx->api().vkWaitForFences( m_ctx->device(), 1, &slot.fence, VK_TRUE, UINT64_MAX ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: the film grain's work failed" );
  }
  slot.ticket = -1;
}

bool GpuFilmGrain::done( int64_t ticket )
{
  Slot& slot = m_slots[ticket % kSlots];
  if( slot.ticket != ticket )
  {
    return true;   // (its slot was freed: done)
  }
  const VkResult status = m_ctx->api().vkGetFenceStatus( m_ctx->device(), slot.fence );
  if( status == VK_NOT_READY )
  {
    return false;
  }
  if( status != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: the film grain's work failed" );
  }
  slot.ticket = -1;
  return true;
}

void GpuFilmGrain::wait( int64_t ticket )
{
  Slot& slot = m_slots[ticket % kSlots];
  if( slot.ticket == ticket )
  {
    finish( slot );
  }
}

uint32_t* GpuFilmGrain::seeds( size_t count )
{
  Slot& slot = m_slots[m_nextTicket % kSlots];
  finish( slot );   // (the frame that had the slot before)
  const size_t size = count * sizeof( uint32_t );
  if( slot.seeds.size < size )
  {
    m_ctx->free( slot.seeds );
    slot.seeds = m_ctx->alloc( ( size + 4095 ) & ~size_t( 4095 ), MemKind::Upload );
  }
  return reinterpret_cast<uint32_t*>( slot.seeds.mapped );
}

int64_t GpuFilmGrain::apply( const FilmGrain& grain, int numPlanes, const Plane src[3], const Plane dst[3], int dstBytes, int lumaWidth, int lumaHeight )
{
  const Api&           vk   = m_ctx->api();
  Slot&                slot = m_slots[m_nextTicket % kSlots];   // (seeds() freed it)
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

  // the tables as the synthesis has them now
  memcpy( slot.tables.mapped, impl.patterns(), kPatternBytes );
  for( int c = 0; c < 3; c++ )
  {
    memcpy( slot.tables.mapped + kPatternBytes + c * 256, impl.scaleLut( c ), 256 );
    memcpy( slot.tables.mapped + kPatternBytes + 768 + c * 256, impl.patternLut( c ), 256 );
  }

  vk.vkResetCommandPool( m_ctx->device(), slot.pool, 0 );
  m_ctx->beginOneTime( slot.cmd );

  // On a discrete GPU, the planes' rows that the shader reads are copied into slot.before (whole, from the first row's
  // sample 0 to the last row's last sample read), and it writes into slot.after, with the same layouts.
  const bool      copies = m_ctx->discrete();
  VkDeviceAddress srcAt[3], dstAt[3];
  if( copies )
  {
    VkDeviceSize srcSpan[3], dstSpan[3], srcTotal = 0, dstTotal = 0;
    for( int c = 0; c < numPlanes; c++ )
    {
      const int rows = lumaHeight / ( c ? impl.subY() : 1 );
      const int cols = lumaWidth / ( c ? impl.subX() : 1 );
      srcSpan[c]     = ( VkDeviceSize( rows - 1 ) * src[c].stride + src[c].readWidth ) * sizeof( int16_t );
      dstSpan[c]     = VkDeviceSize( rows - 1 ) * dst[c].stride + VkDeviceSize( cols ) * dstBytes;
      srcAt[c]       = srcTotal;
      dstAt[c]       = dstTotal;
      srcTotal      += ( srcSpan[c] + 255 ) & ~VkDeviceSize( 255 );
      dstTotal      += ( dstSpan[c] + 255 ) & ~VkDeviceSize( 255 );
    }
    m_ctx->ensure( slot.before, size_t( srcTotal ), MemKind::Device );
    m_ctx->ensure( slot.after, size_t( dstTotal ), MemKind::Device );
    for( int c = 0; c < numPlanes; c++ )
    {
      const VkBufferCopy region{ src[c].offset, srcAt[c], srcSpan[c] };
      vk.vkCmdCopyBuffer( slot.cmd, src[c].buffer, slot.before.buffer, 1, &region );
      srcAt[c] += slot.before.addr;
      dstAt[c] += slot.after.addr;
    }
    m_ctx->memoryBarrier( slot.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT );
  }

  vk.vkCmdBindPipeline( slot.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_filmgrain ) );
  for( int c = 0; c < numPlanes; c++ )
  {
    FilmGrainPush push{};
    push.src        = copies ? srcAt[c] : src[c].addr;
    push.dst        = copies ? dstAt[c] : dst[c].addr;
    push.pattern    = slot.tables.addr;
    push.luts       = slot.tables.addr + kPatternBytes;
    push.seeds      = slot.seeds.addr;
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
    vk.vkCmdPushConstants( slot.cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    // a workgroup 1024 samples of a row
    vk.vkCmdDispatch( slot.cmd, uint32_t( ( push.width + 1023 ) / 1024 ), uint32_t( lumaHeight / push.suby ), 1 );
  }
  if( copies )
  {
    // slot.after's rows into the output, row by row: the bytes between the rows (dst may be the picture) stay as they are
    m_ctx->memoryBarrier( slot.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_ACCESS_TRANSFER_READ_BIT );
    for( int c = 0; c < numPlanes; c++ )
    {
      const int          rows     = lumaHeight / ( c ? impl.subY() : 1 );
      const VkDeviceSize rowBytes = VkDeviceSize( lumaWidth / ( c ? impl.subX() : 1 ) ) * dstBytes;
      slot.rows.resize( size_t( rows ) );
      for( int y = 0; y < rows; y++ )
      {
        const VkDeviceSize at = VkDeviceSize( y ) * dst[c].stride;
        slot.rows[y]          = { dstAt[c] - slot.after.addr + at, dst[c].offset + at, rowBytes };
      }
      vk.vkCmdCopyBuffer( slot.cmd, slot.after.buffer, dst[c].buffer, uint32_t( rows ), slot.rows.data() );
    }
  }
  // The CPU reads the frame once the fence is signalled: the last writes (the copies', or the shader's) become visible to
  // it.
  m_ctx->memoryBarrier( slot.cmd, copies ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        copies ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT );
  vk.vkEndCommandBuffer( slot.cmd );

  vk.vkResetFences( m_ctx->device(), 1, &slot.fence );
  m_ctx->submitGrain( slot.cmd, slot.fence );
  slot.ticket = m_nextTicket++;
  return slot.ticket;
}

}   // namespace gpu
}   // namespace vvdec
