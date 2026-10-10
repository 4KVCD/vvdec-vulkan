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

namespace vvdec
{
namespace gpu
{

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

UserAllocator hostPictureAllocator( Context* context )
{
  return UserAllocator( createHostPlane, unrefHostPlane, context );
}

bool hostPictureAddresses( const PelStorage& pic, VkDeviceAddress addr[MAX_NUM_COMPONENT], int stride[MAX_NUM_COMPONENT] )
{
  const UserAllocator* allocator = pic.getUserAllocator();
  if( pic.bufs.empty() || !pic.isExternAllocator() || !allocator || allocator->create != createHostPlane )
  {
    return false;
  }
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    if( !pic.getBufAllocator( comp ) )
    {
      return false;
    }
    if( addr )
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
  if( !pic.gpuMemory || !hostPictureAddresses( pic.m_bufs[PIC_RECONSTRUCTION], nullptr, nullptr ) )
  {
    return;   // (stand-ins: no twins)
  }
  pic.gpuCurrent              = false;   // should it be referenced after all, it is uploaded again
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

// ====================================================================================================================
// PlanePool
// ====================================================================================================================

// A stand-in plane: address space reserved without memory
struct StandIn
{
  void*  base = nullptr;
  size_t size = 0;
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
  *allocator = new StandIn{ base, size };
  return base;
}

static void releaseStandIn( void*, void* allocator )
{
  StandIn* standIn = static_cast<StandIn*>( allocator );
#ifdef _WIN32
  VirtualFree( standIn->base, 0, MEM_RELEASE );
#else
  munmap( standIn->base, standIn->size );
#endif
  delete standIn;
}

PlanePool::PlanePool( const UserAllocator* planes ) : m_planes( planes ), m_standIns( reserveStandIn, releaseStandIn, nullptr ) {}

PlanePool::~PlanePool() = default;   // (the planes kept go back to their allocator)

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

void PlanePool::acquire( Picture& pic )
{
  std::lock_guard<std::mutex> lock( m_mutex );
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
  current.swap( *planes );   // (planes now holds the stand-ins)
  pic.gpuSwappedPlanes = std::move( planes );
  if( pic.cs )
  {
    pic.cs->rebindPicBufs();
  }
}

void PlanePool::release( Picture& pic )
{
  std::lock_guard<std::mutex> lock( m_mutex );
  if( !hasPlanes( pic ) )
  {
    return;
  }
  releaseTwins( pic );
  // the stand-ins swapped out when the planes came in, or new ones (a picture made with its planes)
  std::unique_ptr<PelStorage> standIns = std::move( pic.gpuSwappedPlanes );
  if( !standIns )
  {
    standIns = makeStorage( pic, &m_standIns );
  }
  pic.m_bufs[PIC_RECONSTRUCTION].swap( *standIns );   // (standIns now holds the planes)
  m_free.push_back( std::move( standIns ) );
  if( pic.cs )
  {
    pic.cs->rebindPicBufs();
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

VkDeviceSize planeOffset( const PelStorage& storage, ComponentID comp )
{
  const Buffer&  host   = hostPlane( storage, comp )->host;
  const uint8_t* origin = reinterpret_cast<const uint8_t*>( storage.bufs[comp].buf );
  CHECK_FATAL( origin < host.mapped || origin >= host.mapped + host.size, "Vulkan: a plane outside its buffer" );
  return VkDeviceSize( origin - host.mapped );
}

GpuLayout gpuLayout( const PelStorage& storage, ComponentID comp )
{
  const Buffer&      host   = hostPlane( storage, comp )->host;
  const VkDeviceSize offset = planeOffset( storage, comp );
  const VkDeviceSize stride = VkDeviceSize( storage.bufs[comp].stride ) * sizeof( Pel );
  const VkDeviceSize top    = offset / stride;                // the rows of margin above (as many are below)
  const VkDeviceSize rows   = host.size / stride - 2 * top;   // (host.size also covers an extra sample and rounding)
  CHECK_FATAL( rows < VkDeviceSize( storage.bufs[comp].height ), "Vulkan: a plane laid out unexpectedly" );
  return { top * stride, rows * stride, offset % stride };
}

}   // namespace gpu
}   // namespace vvdec
