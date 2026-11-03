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

#include "GpuFilmGrain.h"
#include "GpuPictureMemory.h"
#include "vvdec_gpu/GpuShaderIds.h"

#include "FilmGrain/FilmGrain.h"

#include "CommonLib/Picture.h"

#include <cstdlib>

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

// A copy's timestamps (Copy::queries): its start, after its first barrier (earlier work done), its end
enum HandOverStamp : uint32_t
{
  kStampStart,
  kStampBarrier,
  kStampEnd,
  kStamps
};

HandOver::HandOver( Context* context ) : m_ctx( context ), m_copies( kCopies )
{
  const char* timed = getenv( "VVDEC_HANDOVER_TIMES" );
  m_timed           = timed && atoi( timed ) != 0;
  for( Copy& copy: m_copies )
  {
    copy.pool = m_ctx->createCommandPool( m_ctx->queueFamily() );
    m_ctx->allocateCommandBuffers( copy.pool, 1, &copy.cmd );
    copy.fence = m_ctx->createFence();
    if( m_timed )
    {
      copy.queries = m_ctx->createTimestampQueries( kStamps );   // the compute queue's family has timestamps
    }
  }
}

HandOver::~HandOver()
{
  wait();
  if( m_timed && m_times.copies )
  {
    printTimes( "in all" );
  }
  for( Copy& copy: m_copies )
  {
    m_ctx->destroyQueries( copy.queries );
    m_ctx->destroyFence( copy.fence );
    m_ctx->destroyCommandPool( copy.pool );   // and its buffer
    m_ctx->free( copy.grainTables );
  }
  m_ctx->free( m_grainBefore );
  m_ctx->free( m_grainAfter );
}

void HandOver::wait()
{
  std::lock_guard<std::mutex> lock( m_mutex );
  waitPendingLocked();
}

void HandOver::waitPendingLocked()
{
  for( Copy& copy: m_copies )
  {
    if( copy.pending )
    {
      m_ctx->api().vkWaitForFences( m_ctx->device(), 1, &copy.fence, VK_TRUE, UINT64_MAX );
      copy.pending = false;
      if( m_timed )
      {
        addTimes( copy );
      }
    }
  }
}

VkResult HandOver::waitFor( uint64_t value, uint64_t timeoutNs )
{
  // Not under m_mutex: pack may wait on a ring slot's fence meanwhile
  const VkResult result = m_ctx->waitHandOver( value, timeoutNs );
  if( result == VK_SUCCESS && m_timed )
  {
    const Clock::time_point     now = Clock::now();
    std::lock_guard<std::mutex> lock( m_mutex );
    for( Copy& copy: m_copies )
    {
      if( copy.pending && copy.value == value && copy.waitedMs < 0 )
      {
        copy.waitedMs = std::chrono::duration<double, std::milli>( now - copy.submitted ).count();
      }
    }
  }
  return result;
}

void HandOver::addTimes( Copy& copy )
{
  uint64_t stamps[kStamps];
  if( m_ctx->api().vkGetQueryPoolResults( m_ctx->device(), copy.queries, 0, kStamps, sizeof( stamps ), stamps, sizeof( stamps[0] ),
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT )
      != VK_SUCCESS )
  {
    return;
  }
  const double msPerTick = m_ctx->timestampPeriod() * 1e-6;   // timestampPeriod: nanoseconds a tick
  const double queuedMs  = double( stamps[kStampBarrier] - stamps[kStampStart] ) * msPerTick;
  const double packMs    = double( stamps[kStampEnd] - stamps[kStampBarrier] ) * msPerTick;
  m_times.copies++;
  m_times.queuedMs += queuedMs;
  m_times.packMs += packMs;
  if( copy.waitedMs >= 0 )
  {
    m_times.waited++;
    m_times.delayMs += copy.waitedMs - ( queuedMs + packMs );
  }
  if( m_times.copies % kTimesEvery == 0 )
  {
    printTimes( "so far" );
  }
}

void HandOver::printTimes( const char* when ) const
{
  msg( INFO, "Vulkan hand-over, %s: %ld copies, means in ms: queued on the GPU %.3f, pack %.3f; before the GPU started it %.3f (%ld waited for)\n",
       when, m_times.copies, m_times.queuedMs / double( m_times.copies ), m_times.packMs / double( m_times.copies ),
       m_times.waited ? m_times.delayMs / double( m_times.waited ) : 0.0, m_times.waited );
}

// The address of the plane's top-left sample (0, 0) for the shaders, and its stride in samples: in the twin of a
// picture on the GPU, else in its host plane. Called with the pool's lock held.
static VkDeviceAddress planeOrigin( const Picture& pic, ComponentID comp, int& stride )
{
  const PelStorage& planes = pic.m_bufs[PIC_RECONSTRUCTION];
  stride                   = int( planes.bufs[comp].stride );
  if( pic.gpuNeedsDownload )
  {
    const Buffer& twin = planeTwin( planes, comp );
    CHECK_FATAL( !twin.valid(), "Vulkan: a picture handed over without its twins" );
    return twin.addr + gpuLayout( planes, comp ).origin;
  }
  VkDeviceAddress host[MAX_NUM_COMPONENT];
  int             strides[MAX_NUM_COMPONENT];
  CHECK_FATAL( !hostPictureAddresses( planes, host, strides ), "Vulkan: a picture handed over from memory the GPU cannot reach" );
  return host[comp];
}

HandOver::GrainSource HandOver::grainSource( const Picture& pic, ComponentID comp, VkDeviceAddress origin, int stride, const FilmGrainFrame& grain )
{
  const PelStorage& planes = pic.m_bufs[PIC_RECONSTRUCTION];
  const int         sx     = getComponentScaleX( comp, planes.chromaFormat );
  const int         sy     = getComponentScaleY( comp, planes.chromaFormat );
  const HostPlane*  host   = pic.gpuNeedsDownload ? nullptr : hostPlane( planes, comp );
  CHECK_FATAL( !pic.gpuNeedsDownload && !host, "Vulkan: a picture handed over from memory the GPU cannot reach" );
  const Buffer&     buffer = pic.gpuNeedsDownload ? planeTwin( planes, comp ) : host->host;
  GrainSource       src;
  src.addr      = origin + VkDeviceSize( ( grain.confTop >> sy ) * stride + ( grain.confLeft >> sx ) ) * sizeof( Pel );
  src.stride    = stride;
  src.readWidth = int( planes.bufs[comp].width ) - ( grain.confLeft >> sx );
  src.buffer    = buffer.buffer;
  src.offset    = src.addr - buffer.addr + buffer.offset;
  return src;
}

void HandOver::recordGrain( Copy& copy, const FilmGrainFrame& grain, const GrainSource* src, int count, bool fromHost, VkDeviceAddress* afterAt )
{
  // Each plane's samples the grain reads and writes, in the same layout in m_grainAfter (and m_grainBefore): whole
  // rows, from the first row's sample 0 to the last row's last sample read
  VkDeviceSize span[MAX_NUM_COMPONENT], at[MAX_NUM_COMPONENT], total = 0;
  for( int c = 0; c < count; c++ )
  {
    const int rows = grain.height / ( c ? grain.subY : 1 );
    span[c]        = ( VkDeviceSize( rows - 1 ) * src[c].stride + src[c].readWidth ) * sizeof( Pel );
    at[c]          = total;
    total         += ( span[c] + 255 ) & ~VkDeviceSize( 255 );
  }
  // On a discrete GPU, planes in host memory are first copied into its own memory: the grain shader's reads of host
  // memory would cross the bus one by one (GpuFilmGrain)
  const bool viaBefore = fromHost && m_ctx->discrete();
  if( m_grainAfter.size < total || ( viaBefore && m_grainBefore.size < total ) )
  {
    waitPendingLocked();   // the copies under way may still read or write them
    m_ctx->ensure( m_grainAfter, size_t( total ), MemKind::Device );
    if( viaBefore )
    {
      m_ctx->ensure( m_grainBefore, size_t( total ), MemKind::Device );
    }
  }
  // The tables and the seeds: the copy's own (its last use is done)
  const size_t seedBytes = grain.seeds.size() * sizeof( uint32_t );
  m_ctx->ensure( copy.grainTables, FilmGrainTables::kBytes + seedBytes, MemKind::Upload );
  memcpy( copy.grainTables.mapped, grain.tables, FilmGrainTables::kBytes );
  memcpy( copy.grainTables.mapped + FilmGrainTables::kBytes, grain.seeds.data(), seedBytes );

  const Api& vk = m_ctx->api();
  if( viaBefore )
  {
    for( int c = 0; c < count; c++ )
    {
      const VkBufferCopy region{ src[c].offset, at[c], span[c] };
      vk.vkCmdCopyBuffer( copy.cmd, src[c].buffer, m_grainBefore.buffer, 1, &region );
    }
    m_ctx->memoryBarrier( copy.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT );
  }
  GpuFilmGrain::Pass pass;
  pass.numPlanes  = count;
  pass.lumaWidth  = grain.width;
  pass.lumaHeight = grain.height;
  pass.dstBytes   = int( sizeof( Pel ) );
  pass.tables     = copy.grainTables.addr;
  pass.seeds      = copy.grainTables.addr + FilmGrainTables::kBytes;
  pass.bitShift   = grain.bitShift;
  pass.scaleShift = grain.scaleShift;
  pass.subX       = grain.subX;
  pass.subY       = grain.subY;
  for( int c = 0; c < count; c++ )
  {
    // (field by field, as GpuFilmGrain::apply: MSVC miscompiled a braced list assigned to a plane there)
    GpuFilmGrain::Pass::PlaneIo& plane = pass.planes[c];
    afterAt[c]                         = m_grainAfter.addr + at[c];
    plane.src                          = viaBefore ? m_grainBefore.addr + at[c] : src[c].addr;
    plane.dst                          = afterAt[c];
    plane.srcStride                    = src[c].stride;
    plane.dstStride                    = src[c].stride * int( sizeof( Pel ) );
    plane.readWidth                    = src[c].readWidth;
    pass.present[c]                    = grain.present[c];
  }
  GpuFilmGrain::record( m_ctx, copy.cmd, pass );
  // the grain's writes before the copy's reads
  m_ctx->memoryBarrier( copy.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT );
}

uint64_t HandOver::pack( Picture& pic, PlanePool* pool, const Plane* planes, int count, const Buffer& dst, int dstBytes, int shift,
                         const FilmGrainFrame* grain )
{
  CHECK_FATAL( pic.progress < Picture::reconstructed, "Vulkan: a picture handed over before its work on the GPU is done" );
  CHECK_FATAL( count < 1 || count > MAX_NUM_COMPONENT, "Vulkan: a hand-over of no planes or too many" );
  HandOverParams params[MAX_NUM_COMPONENT]{};
  GrainSource    grainSrc[MAX_NUM_COMPONENT];
  bool           fromHost = false;   // the copy reads the host planes (planeOrigin), not the twins
  {
    // The planes' records stay put meanwhile (a picture the pool handles on another thread)
    std::unique_lock<std::mutex> planesLock;
    if( pool )
    {
      planesLock = pool->lockPlanes();
    }
    fromHost = !pic.gpuNeedsDownload;
    for( int i = 0; i < count; i++ )
    {
      const Plane& plane  = planes[i];
      int          stride = 0;
      const VkDeviceAddress origin = planeOrigin( pic, plane.comp, stride );
      params[i].src       = origin + VkDeviceSize( plane.y * stride + plane.x ) * sizeof( Pel );
      params[i].srcStride = stride;
      params[i].dst       = dst.addr + plane.dstOffset;
      params[i].dstStride = int32_t( plane.dstStride );
      params[i].width     = plane.width;
      params[i].height    = plane.height;
      params[i].dstBytes  = dstBytes;
      params[i].shift     = shift;
      if( grain )
      {
        CHECK_FATAL( plane.comp != ComponentID( i ), "Vulkan: a hand-over with film grain of planes out of order" );
        grainSrc[i] = grainSource( pic, plane.comp, origin, stride, *grain );
      }
    }
  }

  std::lock_guard<std::mutex> lock( m_mutex );
  Copy&                       copy = m_copies[m_next];
  m_next                           = ( m_next + 1 ) % m_copies.size();
  const Api&                  vk   = m_ctx->api();
  if( copy.pending )
  {
    vk.vkWaitForFences( m_ctx->device(), 1, &copy.fence, VK_TRUE, UINT64_MAX );
    copy.pending = false;
    if( m_timed )
    {
      addTimes( copy );
    }
  }
  vk.vkResetCommandPool( m_ctx->device(), copy.pool, 0 );
  m_ctx->beginOneTime( copy.cmd );
  if( m_timed )
  {
    vk.vkCmdResetQueryPool( copy.cmd, copy.queries, 0, kStamps );
    vk.vkCmdWriteTimestamp( copy.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, copy.queries, kStampStart );
  }
  // After the picture's reconstruction (earlier work on this queue), and after the other device's last read of dst,
  // which its own submissions finished before it asked for this copy; with film grain also after the copies before,
  // which read and wrote the same scratch (recordGrain)
  const VkPipelineStageFlags afterStages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | ( grain ? VK_PIPELINE_STAGE_TRANSFER_BIT : 0 );
  const VkAccessFlags        afterAccess =
    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | ( grain ? VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT : 0 );
  m_ctx->memoryBarrier( copy.cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_WRITE_BIT, afterStages, afterAccess );
  if( m_timed )
  {
    vk.vkCmdWriteTimestamp( copy.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, copy.queries, kStampBarrier );
  }
  if( grain )
  {
    // The copy reads the planes with their grain, in m_grainAfter: the same rectangles, counted there from the frame's
    // top-left sample (Plane's x and y count from the picture's)
    const ChromaFormat chromaFormat = pic.m_bufs[PIC_RECONSTRUCTION].chromaFormat;
    VkDeviceAddress    afterAt[MAX_NUM_COMPONENT];
    recordGrain( copy, *grain, grainSrc, count, fromHost, afterAt );
    for( int i = 0; i < count; i++ )
    {
      const int x = planes[i].x - ( grain->confLeft >> getComponentScaleX( planes[i].comp, chromaFormat ) );
      const int y = planes[i].y - ( grain->confTop >> getComponentScaleY( planes[i].comp, chromaFormat ) );
      CHECK_FATAL( x < 0 || y < 0, "Vulkan: a hand-over rectangle outside the frame with film grain" );
      params[i].src = afterAt[i] + VkDeviceSize( y * params[i].srcStride + x ) * sizeof( Pel );
    }
  }
  vk.vkCmdBindPipeline( copy.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_handover ) );
  // The planes' copies write disjoint ranges: no barrier between them
  for( int i = 0; i < count; i++ )
  {
    vk.vkCmdPushConstants( copy.cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( params[i] ), &params[i] );
    vk.vkCmdDispatch( copy.cmd, uint32_t( ( params[i].width + kHandOverSegment - 1 ) / kHandOverSegment ), uint32_t( params[i].height ),
                      1 );
  }
  // Into host memory, the writes must be made visible to the host: a fence or semaphore alone does not (for another
  // device's memory the timeline's signal below is what counts, and this costs nothing)
  m_ctx->memoryBarrier( copy.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_ACCESS_HOST_READ_BIT );
  if( m_timed )
  {
    vk.vkCmdWriteTimestamp( copy.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, copy.queries, kStampEnd );
  }
  vk.vkEndCommandBuffer( copy.cmd );
  vk.vkResetFences( m_ctx->device(), 1, &copy.fence );
  copy.waitedMs  = -1;
  copy.submitted = Clock::now();
  // The timeline's signal makes the copy's writes available to the other device
  copy.value   = m_ctx->submitHandOver( copy.cmd, copy.fence );
  copy.pending = true;   // only once submitted: a fence of a failed submission is never signalled
  if( fromHost )
  {
    // The decoder writes the host planes again only once the copy has read them (the CPU's parse writes a reused
    // picture's planes; the twins' writers are ordered after the copy on the queue). The application holds the
    // picture until this returns, so no reuse can come before the store.
    pic.handOverValue.store( copy.value, std::memory_order_release );
  }
  return copy.value;
}

}   // namespace gpu
}   // namespace vvdec
