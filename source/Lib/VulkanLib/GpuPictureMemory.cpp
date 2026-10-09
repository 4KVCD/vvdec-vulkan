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
  if( !pic.gpuMemory )
  {
    return;
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

HostPlane* hostPlane( const PelStorage& storage, ComponentID comp )
{
  HostPlane* plane = static_cast<HostPlane*>( storage.getBufAllocator( comp ) );
  CHECK_FATAL( !plane || !storage.isExternAllocator(), "Vulkan: a picture buffer the GPU cannot reach" );
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
