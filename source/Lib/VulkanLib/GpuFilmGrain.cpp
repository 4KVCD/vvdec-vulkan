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

// The samples of a row that one workgroup of filmgrain.slang takes, and its threads
static constexpr int    kGroupSamples = 1024;
static constexpr int    kGroupThreads = 256;
// The block seeds' buffer grows in steps of this many bytes.
static constexpr size_t kSeedsStep    = 4096;

// the push constants of shaders/filmgrain.slang
struct FilmGrainPush
{
  VkDeviceAddress src, dst, pattern, luts, seeds, halos;
  int32_t         srcStride, dstStride, width, readWidth, lumaWidth, comp, subx, suby, bs, scaleShift, dstBytes, present, rows;
};
static_assert( sizeof( FilmGrainPush ) == 104, "FilmGrainPush: the layout of filmgrain.slang's Params" );

// Whether the byte ranges [a, a + aBytes) and [b, b + bBytes) overlap
static bool overlaps( VkDeviceAddress a, VkDeviceSize aBytes, VkDeviceAddress b, VkDeviceSize bBytes )
{
  return a < b + bBytes && b < a + aBytes;
}

// On an integrated GPU the shader works in the frames themselves. For a plane whose grain goes in place (dst is src,
// the plane has grain, and its rows more than one segment), a workgroup must not read the samples next to its segment
// from the plane: the neighbouring workgroup may already have written its grain there. So the samples on both sides of
// each boundary between two segments are first copied into halos (filmgrain.slang, HALOS=1), and the grain reads them
// from there (push[c].halos). srcSpan/dstSpan: the bytes each plane's grain reads and writes. A 4K 4:2:0 frame's
// halos: 35 KB. A copy of the frame's planes (25 MB) instead cost 8.7 % (Intel Graphics of the 285K, Beekeeper 4K
// with grain: 58.6 -> 53.5 fps); the halos cost nothing measurable there.
static void recordHalos( Context* ctx, VkCommandBuffer cmd, Buffer& halos, int numPlanes, FilmGrainPush push[3], const VkDeviceSize srcSpan[3],
                         const VkDeviceSize dstSpan[3] )
{
  VkDeviceSize boundaries[3] = {};   // per plane: its rows' boundaries between two segments
  VkDeviceSize at[3] = {}, total = 0;
  bool         inPlace[3] = {};
  for( int c = 0; c < numPlanes; c++ )
  {
    // the planes' dispatches have no barriers between them: no plane's grain may write another's source
    for( int d = 0; d < numPlanes; d++ )
    {
      CHECK( d != c && overlaps( push[c].dst, dstSpan[c], push[d].src, srcSpan[d] ), "film grain: a plane's output overlaps another plane's source" );
    }
    const int segments = ( push[c].width + kGroupSamples - 1 ) / kGroupSamples;
    inPlace[c]         = push[c].present && segments > 1 && overlaps( push[c].dst, dstSpan[c], push[c].src, srcSpan[c] );
    if( inPlace[c] )
    {
      // the halos take the place of exactly the samples the grain writes over
      CHECK( push[c].dst != push[c].src || push[c].dstStride != 2 * push[c].srcStride || push[c].dstBytes != 2,
             "film grain: an output that overlaps its source other than in place" );
      boundaries[c] = VkDeviceSize( push[c].rows ) * ( segments - 1 );
      at[c]         = total;
      total        += ( boundaries[c] * 2 * sizeof( int16_t ) + 255 ) & ~VkDeviceSize( 255 );
    }
  }
  if( total == 0 )
  {
    return;
  }
  ctx->ensure( halos, size_t( total ), MemKind::Device );
  const Api& vk = ctx->api();
  vk.vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline( kShader_filmgrain_halos ) );
  for( int c = 0; c < numPlanes; c++ )
  {
    if( inPlace[c] )
    {
      push[c].halos = halos.addr + at[c];
      vk.vkCmdPushConstants( cmd, ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push[c] ), &push[c] );
      vk.vkCmdDispatch( cmd, uint32_t( ( boundaries[c] + kGroupThreads - 1 ) / kGroupThreads ), 1, 1 );
    }
  }
  // the halos' writes before the grain's reads of them; and their reads of the planes before the grain's writes there
  ctx->memoryBarrier( cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_ACCESS_SHADER_READ_BIT );
}

// The push constants of a pass's plane c (no halos)
static FilmGrainPush pushOf( const GpuFilmGrain::Pass& pass, int c )
{
  FilmGrainPush push{};
  push.src        = pass.planes[c].src;
  push.dst        = pass.planes[c].dst;
  push.pattern    = pass.tables;
  push.luts       = pass.tables + FilmGrainTables::kScaleLuts;
  push.seeds      = pass.seeds;
  push.srcStride  = pass.planes[c].srcStride;
  push.dstStride  = pass.planes[c].dstStride;
  push.subx       = c ? pass.subX : 1;
  push.suby       = c ? pass.subY : 1;
  push.width      = pass.lumaWidth / push.subx;
  push.readWidth  = pass.planes[c].readWidth;
  push.lumaWidth  = pass.lumaWidth;
  push.comp       = c;
  push.bs         = pass.bitShift;
  push.scaleShift = pass.scaleShift;
  push.dstBytes   = pass.dstBytes;
  push.present    = pass.present[c] ? 1 : 0;
  push.rows       = pass.lumaHeight / push.suby;
  return push;
}

// The grain's dispatches, a plane after the other (no barriers between them)
static void dispatchGrain( Context* ctx, VkCommandBuffer cmd, int numPlanes, const FilmGrainPush push[3] )
{
  const Api& vk = ctx->api();
  vk.vkCmdBindPipeline( cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline( kShader_filmgrain ) );
  for( int c = 0; c < numPlanes; c++ )
  {
    // (a plane without an address would have the shader write at address 0, which loses the device)
    CHECK( push[c].src == 0 || push[c].dst == 0, "film grain: a plane without its source or output address" );
    vk.vkCmdPushConstants( cmd, ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push[c] ), &push[c] );
    vk.vkCmdDispatch( cmd, uint32_t( ( push[c].width + kGroupSamples - 1 ) / kGroupSamples ), uint32_t( push[c].rows ), 1 );
  }
}

void GpuFilmGrain::record( Context* ctx, VkCommandBuffer cmd, const Pass& pass )
{
  FilmGrainPush push[3];
  for( int c = 0; c < pass.numPlanes; c++ )
  {
    push[c] = pushOf( pass, c );
  }
  dispatchGrain( ctx, cmd, pass.numPlanes, push );
}

GpuFilmGrain::GpuFilmGrain( Context* ctx ) : m_ctx( ctx )
{
  for( Slot& slot: m_slots )
  {
    slot.pool = m_ctx->createCommandPool( m_ctx->grainFamily() );
    m_ctx->allocateCommandBuffers( slot.pool, 1, &slot.cmd );
    slot.fence  = m_ctx->createFence();
    slot.tables = m_ctx->alloc( FilmGrainTables::kBytes, MemKind::Upload );
  }
}

GpuFilmGrain::~GpuFilmGrain()
{
  for( Slot& slot: m_slots )
  {
    // The frame's grain must be done before the commands, tables and seeds go. finish() would throw on a failed wait,
    // and an exception leaving a destructor ends the program, so the destructor waits itself: a failed wait means a
    // lost device, whose work has stopped anyway.
    if( slot.ticket >= 0 )
    {
      m_ctx->api().vkWaitForFences( m_ctx->device(), 1, &slot.fence, VK_TRUE, UINT64_MAX );
    }
    m_ctx->destroyFence( slot.fence );
    m_ctx->destroyCommandPool( slot.pool );
    m_ctx->free( slot.tables );
    m_ctx->free( slot.seeds );
    m_ctx->free( slot.before );
    m_ctx->free( slot.after );
    m_ctx->free( slot.halos );
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
    return true;   // its slot was freed, so its grain is done
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
  finish( slot );   // the frame that had the slot before must be done
  const size_t size = count * sizeof( uint32_t );
  if( slot.seeds.size < size )
  {
    m_ctx->free( slot.seeds );
    slot.seeds = m_ctx->alloc( alignUp( size, kSeedsStep ), MemKind::Upload );
  }
  return reinterpret_cast<uint32_t*>( slot.seeds.mapped );
}

int64_t GpuFilmGrain::apply( const FilmGrain& grain, int numPlanes, const Plane src[3], const Plane dst[3], int dstBytes, int lumaWidth, int lumaHeight )
{
  const Api&           vk   = m_ctx->api();
  Slot&                slot = m_slots[m_nextTicket % kSlots];   // seeds() freed it
  const FilmGrainImpl& impl = grain.impl();
  if( grain.anyPresent( numPlanes ) )
  {
    grain.checkFrame( lumaWidth );
  }
  grain.writeTables( slot.tables.mapped );   // the tables as the synthesis has them now

  vk.vkResetCommandPool( m_ctx->device(), slot.pool, 0 );
  m_ctx->beginOneTime( slot.cmd );

  // The bytes of each plane that the shader reads and writes: whole rows, from the first row's sample 0 to the last
  // row's last sample read or written
  VkDeviceSize srcSpan[3], dstSpan[3];
  for( int c = 0; c < numPlanes; c++ )
  {
    const int rows = lumaHeight / ( c ? impl.subY() : 1 );
    const int cols = lumaWidth / ( c ? impl.subX() : 1 );
    srcSpan[c]     = ( VkDeviceSize( rows - 1 ) * src[c].stride + src[c].readWidth ) * sizeof( int16_t );
    dstSpan[c]     = VkDeviceSize( rows - 1 ) * dst[c].stride + VkDeviceSize( cols ) * dstBytes;
  }

  // On a discrete GPU, the planes' rows that the shader reads are copied into slot.before, and it writes into
  // slot.after, with the same layouts.
  const bool      copies = m_ctx->discrete();
  VkDeviceAddress srcAt[3], dstAt[3];
  if( copies )
  {
    VkDeviceSize srcTotal = 0, dstTotal = 0;
    for( int c = 0; c < numPlanes; c++ )
    {
      srcAt[c]  = srcTotal;
      dstAt[c]  = dstTotal;
      srcTotal += ( srcSpan[c] + 255 ) & ~VkDeviceSize( 255 );
      dstTotal += ( dstSpan[c] + 255 ) & ~VkDeviceSize( 255 );
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

  Pass pass;
  pass.numPlanes  = numPlanes;
  pass.lumaWidth  = lumaWidth;
  pass.lumaHeight = lumaHeight;
  pass.dstBytes   = dstBytes;
  pass.tables     = slot.tables.addr;
  pass.seeds      = slot.seeds.addr;
  pass.bitShift   = impl.bitShift();
  pass.scaleShift = impl.scaleShift();
  pass.subX       = impl.subX();
  pass.subY       = impl.subY();
  FilmGrainPush push[3];
  for( int c = 0; c < numPlanes; c++ )
  {
    // The plane's fields one by one: MSVC 19.44 (/O2) compiled `pass.planes[c] = { ... }` followed by the inlined
    // pushOf with pushOf's loads of src and dst ahead of the braced list's store, so the grain got addresses 0.
    Pass::PlaneIo& plane = pass.planes[c];
    plane.src            = copies ? srcAt[c] : src[c].addr;
    plane.dst            = copies ? dstAt[c] : dst[c].addr;
    plane.srcStride      = src[c].stride;
    plane.dstStride      = dst[c].stride;
    plane.readWidth      = src[c].readWidth;
    pass.present[c]      = grain.componentPresent( c );
    push[c]              = pushOf( pass, c );
  }
  if( !copies )
  {
    recordHalos( m_ctx, slot.cmd, slot.halos, numPlanes, push, srcSpan, dstSpan );
  }
  dispatchGrain( m_ctx, slot.cmd, numPlanes, push );
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
