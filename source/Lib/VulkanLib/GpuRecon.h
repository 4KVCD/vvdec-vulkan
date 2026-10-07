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

/** \file     GpuRecon.h
    \brief    the stages of a picture's reconstruction that run on the GPU

    The pictures live in host memory the GPU reaches (hostPictureAllocator), so
    a stage on the GPU reads and writes vvdec's own picture buffers.
*/

#pragma once

#include "GpuContext.h"

#include "CommonLib/Common.h"
#include "CommonLib/Buffer.h"

namespace vvdec
{

class CodingStructure;
class AdaptiveLoopFilter;

namespace gpu
{

// UserAllocator callbacks putting the planes of the pictures in host memory
// the GPU reaches; each plane's allocator handle is its gpu::Buffer
UserAllocator hostPictureAllocator( Context* context );

class GpuRecon
{
public:
  explicit GpuRecon( Context* context );
  ~GpuRecon();
  GpuRecon( const GpuRecon& )            = delete;
  GpuRecon& operator=( const GpuRecon& ) = delete;

  // whether the GPU's ALF covers the picture: no CTU of it is clipped by a
  // slice, tile or subpicture boundary the filters may not cross, nor by a
  // virtual boundary
  static bool alfSupported( const CodingStructure& cs );

  // ALF and CC-ALF of the picture, from src (the reconstruction after SAO) into dst
  void alf( const CodingStructure& cs, const AdaptiveLoopFilter& alf, const PelStorage& src, PelStorage& dst );

private:
  void     beginCommands();
  void     submitAndWait();
  uint8_t* params( size_t size );   // the host-visible parameter buffer, at least size bytes

  Context*        m_ctx;
  VkCommandPool   m_pool  = VK_NULL_HANDLE;
  VkCommandBuffer m_cmd   = VK_NULL_HANDLE;
  VkFence         m_fence = VK_NULL_HANDLE;
  Buffer          m_params;
};

}   // namespace gpu
}   // namespace vvdec
