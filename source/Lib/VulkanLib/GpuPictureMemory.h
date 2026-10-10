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

/** \file     GpuPictureMemory.h
    \brief    Where the pictures' planes live: host memory that the GPU can reach, and each plane's twin in GPU memory
*/

#pragma once

#include "GpuContext.h"

#include "CommonLib/Buffer.h"
#include "CommonLib/Common.h"

#include <memory>
#include <mutex>
#include <vector>

namespace vvdec
{
struct Picture;

namespace gpu
{

// A plane of a picture. host is in host memory that the GPU can reach, where vvdec's CPU code reads and writes the
// plane. twin is its copy in the GPU's own memory, where the GPU reconstructs the picture and keeps it as a
// reference picture: allocated when first used, and returned to the pool when no picture references the picture
// any more (releaseTwins).
struct HostPlane
{
  Buffer host;
  Buffer twin;
};

// The UserAllocator callbacks that put the pictures' planes into host memory that the GPU can reach. Each plane's
// allocator handle is its HostPlane.
UserAllocator hostPictureAllocator( Context* context );
// Whether a picture's planes come from hostPictureAllocator. If so, and addr and stride are not null, they receive
// each plane's address of sample (0, 0) and its stride in samples.
bool hostPictureAddresses( const PelStorage& pic, VkDeviceAddress addr[MAX_NUM_COMPONENT], int stride[MAX_NUM_COMPONENT] );
// The GPU address of an allocation from hostPictureAllocator, given its allocator handle.
VkDeviceAddress hostAllocationAddress( void* allocation );
// Returns the twins of a reconstructed picture that no picture will reference any more to the pool. The picture is
// then no longer gpuCurrent.
void releaseTwins( Picture& pic );

// The planes of the pictures the GPU parses (VVDEC_GPU_PARSE=2): a picture has them only from when its work goes to
// the GPU (DecLibRecon::queueGpuTask) until it is finished and may be reused (PicListManager::markUnusedPicturesReusable).
// Before and after, the PelStorage of its planes holds stand-ins with the same layout -- address space reserved without
// memory -- so that the sizes and strides are there for whatever reads them, and a stray access to the samples faults
// at once. The pool keeps the planes given back for the next pictures.
class PlanePool
{
public:
  // planes: the allocator of the planes (hostPictureAllocator's)
  explicit PlanePool( const UserAllocator* planes );
  ~PlanePool();
  PlanePool( const PlanePool& )            = delete;
  PlanePool& operator=( const PlanePool& ) = delete;

  // The stand-ins' allocator, for the picture list's new pictures (PicListManager::setGpuPlanes)
  UserAllocator* standIns() { return &m_standIns; }
  // Swaps planes (from the pool, or new ones) in for the picture's stand-ins; nothing when it has its planes
  void acquire( Picture& pic );
  // Swaps stand-ins in for the picture's planes, which go back to the pool, their twins to the GPU's pool; nothing when
  // it has stand-ins. For a finished picture whose buffer may be reused, and a buffer used for a picture the GPU parses.
  void release( Picture& pic );

private:
  bool                        hasPlanes( const Picture& pic ) const;
  std::unique_ptr<PelStorage> makeStorage( const Picture& pic, const UserAllocator* allocator ) const;   // of its layout

  const UserAllocator*                     m_planes;
  UserAllocator                            m_standIns;
  std::mutex                               m_mutex;   // (acquire and release run on the decoder's threads)
  std::vector<std::unique_ptr<PelStorage>> m_free;    // planes no picture has
};

// The HostPlane of a component of a picture whose planes come from hostPictureAllocator.
HostPlane* hostPlane( const PelStorage& storage, ComponentID comp );
// Where a plane's sample (0, 0) is in its host buffer, in bytes.
VkDeviceSize planeOffset( const PelStorage& storage, ComponentID comp );

// How a plane is laid out on the GPU (in its twin, or in scratch memory like it): the host plane's rows that hold
// the picture, including its extension to whole CTUs but not its margins above and below, which no shader reads.
// The stride is the host plane's, so that a sample has the same place within its row.
struct GpuLayout
{
  VkDeviceSize hostOffset;   // where the first of these rows starts in the host buffer
  VkDeviceSize size;         // the size of these rows in bytes
  VkDeviceSize origin;       // where sample (0, 0) is, counted from the start of the first row
};
GpuLayout gpuLayout( const PelStorage& storage, ComponentID comp );

}   // namespace gpu
}   // namespace vvdec
