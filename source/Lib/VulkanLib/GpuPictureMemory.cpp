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

/** \file     GpuPictureMemory.cpp
    \brief    Where the pictures' planes live: host memory that the GPU can reach, and each plane's twin in GPU memory
*/

#include "GpuPictureMemory.h"

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <sys/mman.h>
#endif

#include "CommonLib/Picture.h"

#include <algorithm>

namespace vvdec
{
namespace gpu
{

// ====================================================================================================================
// The planes' records: host planes (HostPlane) and their stand-ins (StandIn)
// ====================================================================================================================

static void* createHostPlane( void* opaque, vvdecComponentType, uint32_t size, uint32_t, void** allocator )
{
  Context*   context = static_cast<Context*>( opaque );
  HostPlane* plane   = new HostPlane;
  plane->host        = context->alloc( size, MemKind::HostCached );
  plane->bytes       = size;
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

// A stand-in of a plane: address space reserved without memory, of the plane's size; and the plane's twin while
// the picture has stand-ins (PlanePool: a picture reconstructed on the GPU before it is downloaded)
struct StandIn
{
  void*  base = nullptr;
  size_t size = 0;
  Buffer twin;
};

static void* reserveStandIn( void*, vvdecComponentType, uint32_t size, uint32_t, void** allocator )
{
#ifdef _WIN32
  void* base = VirtualAlloc( nullptr, size, MEM_RESERVE, PAGE_NOACCESS );
#else
  void* base = mmap( nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0 );
  base       = base == MAP_FAILED ? nullptr : base;
#endif
  if( !base )
  {
    return nullptr;
  }
  StandIn* standIn = new StandIn;
  standIn->base    = base;
  standIn->size    = size;
  *allocator       = standIn;
  return base;
}

static void releaseStandIn( void* opaque, void* allocator )
{
  StandIn* standIn = static_cast<StandIn*>( allocator );
  static_cast<Context*>( opaque )->free( standIn->twin );
#ifdef _WIN32
  VirtualFree( standIn->base, 0, MEM_RELEASE );
#else
  munmap( standIn->base, standIn->size );
#endif
  delete standIn;
}

// The records of a picture's planes: none of the GPU path's (planes in the CPU's own memory), host planes, or their
// stand-ins
enum class PlaneRecords
{
  None,
  Host,
  StandIns,
};

static PlaneRecords planeRecords( const PelStorage& storage )
{
  const UserAllocator* allocator = storage.getUserAllocator();
  if( storage.bufs.empty() || !storage.isExternAllocator() || !allocator )
  {
    return PlaneRecords::None;
  }
  for( int comp = 0; comp < (int) storage.bufs.size(); comp++ )
  {
    if( !storage.getBufAllocator( ComponentID( comp ) ) )
    {
      return PlaneRecords::None;
    }
  }
  if( allocator->create == createHostPlane )
  {
    return PlaneRecords::Host;
  }
  return allocator->create == reserveStandIn ? PlaneRecords::StandIns : PlaneRecords::None;
}

// Where a plane's memory (or its stand-in's address space) begins, and the size the plane asked for: the same for a
// plane and its stand-in, so that both give the same layout on the GPU (gpuLayout)
struct PlaneSpan
{
  const uint8_t* base;
  size_t         bytes;
};

static PlaneSpan planeSpan( const PelStorage& storage, ComponentID comp )
{
  const PlaneRecords records = planeRecords( storage );
  CHECK_FATAL( records == PlaneRecords::None, "Vulkan: a picture buffer the GPU cannot reach" );
  const void* record = storage.getBufAllocator( comp );
  if( records == PlaneRecords::Host )
  {
    const HostPlane* plane = static_cast<const HostPlane*>( record );
    return { plane->host.mapped, plane->bytes };
  }
  const StandIn* standIn = static_cast<const StandIn*>( record );
  return { static_cast<const uint8_t*>( standIn->base ), standIn->size };
}

UserAllocator hostPictureAllocator( Context* context )
{
  return UserAllocator( createHostPlane, unrefHostPlane, context );
}

bool hostPictureAddresses( const PelStorage& pic, VkDeviceAddress addr[MAX_NUM_COMPONENT], int stride[MAX_NUM_COMPONENT] )
{
  if( planeRecords( pic ) != PlaneRecords::Host )
  {
    return false;
  }
  if( addr )
  {
    for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
    {
      addr[comp]   = hostPlane( pic, ComponentID( comp ) )->host.addr + planeOffset( pic, ComponentID( comp ) );
      stride[comp] = int( pic.bufs[comp].stride );
    }
  }
  return true;
}

VkDeviceAddress hostAllocationAddress( void* allocation )
{
  return static_cast<HostPlane*>( allocation )->host.addr;
}

void releaseTwins( Picture& pic )
{
  const PelStorage& planes = pic.m_bufs[PIC_RECONSTRUCTION];
  if( !pic.gpuMemory || planeRecords( planes ) == PlaneRecords::None )
  {
    return;   // (planes in the CPU's own memory: no twins)
  }
  pic.gpuCurrent   = false;   // should it be referenced after all, it is uploaded again
  Context* context = static_cast<Context*>( pic.gpuMemory->opaque );
  for( int comp = 0; comp < (int) planes.bufs.size(); comp++ )
  {
    context->recycle( planeTwin( planes, ComponentID( comp ) ) );
  }
}

HostPlane* hostPlane( const PelStorage& storage, ComponentID comp )
{
  HostPlane* plane = static_cast<HostPlane*>( storage.getBufAllocator( comp ) );
  CHECK_FATAL( !plane || !storage.isExternAllocator(), "Vulkan: a picture buffer the GPU cannot reach" );
  // (a picture's stand-ins have no HostPlane: the GPU must never get their addresses)
  CHECK_FATAL( storage.getUserAllocator()->create != createHostPlane, "Vulkan: a picture's stand-ins used as its planes" );
  return plane;
}

Buffer& planeTwin( const PelStorage& storage, ComponentID comp )
{
  const PlaneRecords records = planeRecords( storage );
  CHECK_FATAL( records == PlaneRecords::None, "Vulkan: a picture buffer the GPU cannot reach" );
  void* record = storage.getBufAllocator( comp );
  return records == PlaneRecords::Host ? static_cast<HostPlane*>( record )->twin : static_cast<StandIn*>( record )->twin;
}

VkDeviceSize planeOffset( const PelStorage& storage, ComponentID comp )
{
  const PlaneSpan span   = planeSpan( storage, comp );
  const uint8_t*  origin = reinterpret_cast<const uint8_t*>( storage.bufs[comp].buf );
  CHECK_FATAL( origin < span.base || origin >= span.base + span.bytes, "Vulkan: a plane outside its buffer" );
  return VkDeviceSize( origin - span.base );
}

GpuLayout gpuLayout( const PelStorage& storage, ComponentID comp )
{
  const PlaneSpan    span   = planeSpan( storage, comp );
  const VkDeviceSize offset = planeOffset( storage, comp );
  const VkDeviceSize stride = VkDeviceSize( storage.bufs[comp].stride ) * sizeof( Pel );
  const VkDeviceSize top    = offset / stride;                                 // the rows of margin above (as many are below)
  const VkDeviceSize rows   = VkDeviceSize( span.bytes ) / stride - 2 * top;   // (the plane's memory also has an extra sample)
  CHECK_FATAL( rows < VkDeviceSize( storage.bufs[comp].height ), "Vulkan: a plane laid out unexpectedly" );
  return { top * stride, rows * stride, offset % stride };
}

// ====================================================================================================================
// PlanePool
// ====================================================================================================================

PlanePool::PlanePool( const UserAllocator* planes )
  : m_planes( planes ), m_standIns( reserveStandIn, releaseStandIn, planes->opaque ), m_ctx( static_cast<Context*>( planes->opaque ) )
{
}

PlanePool::~PlanePool()
{
  // the downloads done, and their commands gone
  for( const auto& download: m_downloads )
  {
    if( download->pic )
    {
      m_ctx->api().vkWaitForFences( m_ctx->device(), 1, &download->fence, VK_TRUE, UINT64_MAX );
    }
    m_ctx->destroyFence( download->fence );
    m_ctx->destroyCommandPool( download->pool );
  }
  // (the planes kept go back to their allocator with m_free; the colocated motion too)
  for( const ColMotion& c: m_freeColMotion )
  {
    m_planes->unref( m_planes->opaque, c.allocation );
  }
}

bool PlanePool::hasPlanes( const Picture& pic ) const
{
  return pic.m_bufs[PIC_RECONSTRUCTION].getUserAllocator() != &m_standIns;
}

std::unique_ptr<PelStorage> PlanePool::makeStorage( const Picture& pic, const UserAllocator* allocator ) const
{
  auto storage = std::make_unique<PelStorage>();
  storage->create( pic.chromaFormat, pic.lumaSize(), pic.maxCUSize, pic.margin, MEMORY_ALIGN_DEF_SIZE, true, allocator );
  return storage;
}

// Whether two storages have the same layout (PelStorage::swap's condition)
static bool sameLayout( const PelStorage& a, const PelStorage& b )
{
  if( a.chromaFormat != b.chromaFormat || a.bufs.size() != b.bufs.size() )
  {
    return false;
  }
  for( size_t c = 0; c < a.bufs.size(); c++ )
  {
    if( a.bufs[c].width != b.bufs[c].width || a.bufs[c].height != b.bufs[c].height || a.bufs[c].stride != b.bufs[c].stride
        || a.getBufSize( int( c ) ) != b.getBufSize( int( c ) ) )
    {
      return false;
    }
  }
  return true;
}

void PlanePool::acquireColMotion( Picture& pic )
{
  std::lock_guard<std::mutex> lock( m_mutex );
  acquireColMotionLocked( pic );
}

void PlanePool::acquire( Picture& pic )
{
  std::unique_lock<std::mutex> lock( m_mutex );
  waitDownloadLocked( lock, pic );   // (one running for it: its planes are the CPU's to write now)
  acquireColMotionLocked( pic );
  acquirePlanesLocked( pic );
  pic.gpuNeedsDownload = false;
}

void PlanePool::acquireColMotionLocked( Picture& pic )
{
  // the colocated motion, of its size from the pool, else new
  if( !pic.cs || pic.cs->m_colMiMap || !pic.cs->colMiAllocator )
  {
    return;
  }
  const uint32_t bytes = uint32_t( sizeof( ColocatedMotionInfo ) * pic.cs->colMotionEntries() );
  ColMotion      c{};
  for( auto it = m_freeColMotion.begin(); it != m_freeColMotion.end(); ++it )
  {
    if( it->bytes == bytes )
    {
      c = *it;
      m_freeColMotion.erase( it );
      break;
    }
  }
  if( !c.allocation )
  {
    c.bytes = bytes;
    c.map   = m_planes->create( m_planes->opaque, VVDEC_CT_Y, bytes, MEMORY_ALIGN_DEF_SIZE, &c.allocation );
    CHECK( !c.map, "no memory for the colocated motion" );
  }
  pic.cs->attachColMotion( static_cast<ColocatedMotionInfo*>( c.map ), c.allocation );
}

void PlanePool::acquirePlanesLocked( Picture& pic )
{
  if( hasPlanes( pic ) )
  {
    return;
  }
  // planes of its layout from the pool, else new ones
  PelStorage&                 current = pic.m_bufs[PIC_RECONSTRUCTION];
  std::unique_ptr<PelStorage> planes;
  for( auto it = m_free.begin(); it != m_free.end(); ++it )
  {
    if( sameLayout( **it, current ) )
    {
      planes = std::move( *it );
      m_free.erase( it );
      break;
    }
  }
  if( !planes )
  {
    planes = makeStorage( pic, m_planes );
  }
  // its twins (a picture reconstructed with its stand-ins) go with it to the planes
  for( int comp = 0; comp < (int) current.bufs.size(); comp++ )
  {
    std::swap( planeTwin( current, ComponentID( comp ) ), planeTwin( *planes, ComponentID( comp ) ) );
  }
  current.swap( *planes );   // (planes now holds the stand-ins)
  pic.gpuSwappedPlanes = std::move( planes );
  if( pic.cs )
  {
    pic.cs->rebindPicBufs();
  }
}

void PlanePool::release( Picture& pic )
{
  std::unique_lock<std::mutex> lock( m_mutex );
  waitDownloadLocked( lock, pic );   // (one started ahead that its output did not need after all)
  pic.gpuNeedsDownload = false;
  // the colocated motion back to the pool (a picture the CPU parsed has it from its set-up: from the same allocator)
  if( pic.cs && pic.cs->m_colMiAllocation && pic.cs->colMiAllocator == m_planes )
  {
    const uint32_t bytes = uint32_t( sizeof( ColocatedMotionInfo ) * pic.cs->m_colMiMapSize );
    void*          map   = pic.cs->m_colMiMap;
    m_freeColMotion.push_back( ColMotion{ pic.cs->detachColMotion(), map, bytes } );
  }
  // its twins back to the GPU's pool (from its planes or their stand-ins), then its planes to this one
  releaseTwins( pic );
  swapStandInsLocked( pic );
}

void PlanePool::releasePlanes( Picture& pic )
{
  std::unique_lock<std::mutex> lock( m_mutex );
  waitDownloadLocked( lock, pic );
  if( pic.gpuCurrent )
  {
    swapStandInsLocked( pic );
  }
}

void PlanePool::swapStandInsLocked( Picture& pic )
{
  if( !hasPlanes( pic ) )
  {
    return;
  }
  // the stand-ins swapped out when the planes came in, or new ones (a picture made with its planes); its twins (when
  // it has them still) go with it
  std::unique_ptr<PelStorage> standIns = std::move( pic.gpuSwappedPlanes );
  if( !standIns )
  {
    standIns = makeStorage( pic, &m_standIns );
  }
  PelStorage& current = pic.m_bufs[PIC_RECONSTRUCTION];
  for( int comp = 0; comp < (int) current.bufs.size(); comp++ )
  {
    std::swap( planeTwin( current, ComponentID( comp ) ), planeTwin( *standIns, ComponentID( comp ) ) );
  }
  current.swap( *standIns );   // (standIns now holds the planes)
  m_free.push_back( std::move( standIns ) );
  if( pic.cs )
  {
    pic.cs->rebindPicBufs();
  }
}

// --------------------------------------------------------------------------------------------------------------------
// The downloads
// --------------------------------------------------------------------------------------------------------------------

void PlanePool::startDownload( Picture& pic )
{
  std::lock_guard<std::mutex> lock( m_mutex );
  startDownloadLocked( pic );
}

void PlanePool::finishDownload( Picture& pic )
{
  std::unique_lock<std::mutex> lock( m_mutex );
  if( !pic.gpuNeedsDownload )
  {
    return;
  }
  startDownloadLocked( pic );
  waitDownloadLocked( lock, pic );
  pic.gpuNeedsDownload = false;
}

void PlanePool::waitDownloads()
{
  std::lock_guard<std::mutex> lock( m_mutex );
  for( const auto& download: m_downloads )
  {
    if( download->pic )
    {
      m_ctx->api().vkWaitForFences( m_ctx->device(), 1, &download->fence, VK_TRUE, UINT64_MAX );
      download->pic = nullptr;
    }
  }
}

PlanePool::Download* PlanePool::downloadOf( const Picture& pic ) const
{
  for( const auto& download: m_downloads )
  {
    if( download->pic == &pic )
    {
      return download.get();
    }
  }
  return nullptr;
}

PlanePool::Download& PlanePool::freeDownload()
{
  for( const auto& download: m_downloads )
  {
    if( !download->pic )
    {
      return *download;
    }
  }
  auto download  = std::make_unique<Download>();
  download->pool = m_ctx->createCommandPool( m_ctx->copyFamily() );
  m_ctx->allocateCommandBuffers( download->pool, 1, &download->cmd );
  download->fence = m_ctx->createFence();
  m_downloads.push_back( std::move( download ) );
  return *m_downloads.back();
}

void PlanePool::startDownloadLocked( Picture& pic )
{
  if( !pic.gpuNeedsDownload || downloadOf( pic ) )
  {
    return;
  }
  CHECK_FATAL( pic.progress < Picture::reconstructed, "Vulkan: a picture downloaded before its work on the GPU is done" );
  acquirePlanesLocked( pic );
  // the copies of its twins into its planes' rows that hold the picture (gpuLayout), on the copy queue: its work on
  // the GPU is done (it is reconstructed), and its last commands made their writes visible to the host
  Download&         download = freeDownload();
  const Api&        vk       = m_ctx->api();
  const PelStorage& planes   = pic.m_bufs[PIC_RECONSTRUCTION];
  vk.vkResetCommandPool( m_ctx->device(), download.pool, 0 );
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( download.cmd, &begin );
  for( int comp = 0; comp < (int) planes.bufs.size(); comp++ )
  {
    const HostPlane*   plane  = hostPlane( planes, ComponentID( comp ) );
    const GpuLayout    layout = gpuLayout( planes, ComponentID( comp ) );
    const VkBufferCopy region{ 0, layout.hostOffset, layout.size };
    CHECK_FATAL( !plane->twin.valid(), "Vulkan: a picture to download without its twins" );
    vk.vkCmdCopyBuffer( download.cmd, plane->twin.buffer, plane->host.buffer, 1, &region );
  }
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };   // (the copies' writes visible to the CPU)
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vk.vkCmdPipelineBarrier( download.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
  vk.vkEndCommandBuffer( download.cmd );
  vk.vkResetFences( m_ctx->device(), 1, &download.fence );
  m_ctx->submitCopy( download.cmd, download.fence );
  download.pic = &pic;
}

void PlanePool::waitDownloadLocked( std::unique_lock<std::mutex>& lock, Picture& pic )
{
  Download* download = downloadOf( pic );
  if( !download )
  {
    return;
  }
  // (unlocked while it waits: the other pictures' downloads may start meanwhile; the record stays where it is)
  const VkFence fence = download->fence;
  lock.unlock();
  const VkResult result = m_ctx->api().vkWaitForFences( m_ctx->device(), 1, &fence, VK_TRUE, UINT64_MAX );
  lock.lock();
  download->pic = nullptr;
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: waiting for a picture's download failed (" << result << ")" );
  }
}

}   // namespace gpu
}   // namespace vvdec
