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
class SampleAdaptiveOffset;

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
  // whether the GPU's SAO covers the picture: no virtual boundaries
  static bool saoSupported( const CodingStructure& cs );
  // whether the GPU covers all the in-loop filters of the picture: deblocking
  // (not with luma-adaptive QP offsets), SAO and ALF
  static bool filtersSupported( const CodingStructure& cs );

  // the deblocking parameters of the current picture (vvdec's LF_INIT writes
  // them): count of them, in host memory the GPU reads
  LoopFilterParam* loopFilterParams( size_t count );

  // the in-loop filters of the picture in pic, which the CPU reconstructed:
  // deblocking, SAO (when sao is given), ALF with CC-ALF (when alf is given);
  // the result back in pic
  void filterPicture( CodingStructure& cs, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf, PelStorage& pic );

private:
  void     beginCommands();
  void     computeBarrier();
  void     mark( const char* label );   // with VVDEC_GPU_PROFILE: the GPU time since the last mark counts as label
  void     submitAndWait();
  uint8_t* params( size_t size );   // the host-visible parameter buffer, at least size bytes

  // the planes of a picture as the shaders see them
  struct Planes
  {
    VkDeviceAddress addr[MAX_NUM_COMPONENT];
    int32_t         stride[MAX_NUM_COMPONENT];
  };
  Planes        mirrorPlanes( int mirror, const PelStorage& pic );   // GPU memory laid out as pic's buffers
  void          copyPlanes( const PelStorage& pic, int mirror, bool toMirror );

  static size_t deblockParamSize( const CodingStructure& cs );
  static size_t alfParamSize( const CodingStructure& cs );
  static size_t saoParamSize( const CodingStructure& cs );
  bool          prepareDeblock( const CodingStructure& cs, uint8_t* params );   // false: no slice deblocks
  void          prepareAlf( const CodingStructure& cs, const AdaptiveLoopFilter& alf, uint8_t* params );
  bool          prepareSao( CodingStructure& cs, SampleAdaptiveOffset& sao, uint8_t* params );   // false: no CTU uses SAO
  void          recordDeblock( const CodingStructure& cs, const Planes& planes, VkDeviceAddress params );
  void          recordAlf( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params );
  void          recordSao( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params );

  static constexpr uint32_t kMaxMarks = 64;

  Context*        m_ctx;
  VkCommandPool   m_pool    = VK_NULL_HANDLE;
  VkCommandBuffer m_cmd     = VK_NULL_HANDLE;
  VkFence         m_fence   = VK_NULL_HANDLE;
  VkQueryPool     m_queries = VK_NULL_HANDLE;   // the profiling timestamps
  const char*     m_labels[kMaxMarks];
  uint32_t        m_marks   = 0;
  Buffer          m_params;
  Buffer          m_mirror[2][MAX_NUM_COMPONENT];   // pictures in the GPU's own memory
  Buffer          m_lfp;                            // the deblocking parameters
};

}   // namespace gpu
}   // namespace vvdec
