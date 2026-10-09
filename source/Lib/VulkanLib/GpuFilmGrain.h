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

/** \file     GpuFilmGrain.h
    \brief    film grain synthesis of the output frames on the GPU
*/

#pragma once

#include "GpuContext.h"

namespace vvdec
{
class FilmGrain;

namespace gpu
{

// film grain synthesis of output frames on the GPU (shaders/filmgrain.slang),
// as vvdec's FilmGrain adds it on the CPU (VVDecImpl::xAddGrain): the frames
// in host memory the GPU reaches (the pictures', or the output's copies)
class GpuFilmGrain
{
public:
  explicit GpuFilmGrain( Context* ctx );
  ~GpuFilmGrain();
  GpuFilmGrain( const GpuFilmGrain& )            = delete;
  GpuFilmGrain& operator=( const GpuFilmGrain& ) = delete;

  // a plane of the frame: its sample (0, 0) and its stride
  struct Plane
  {
    VkDeviceAddress addr      = 0;
    int             stride    = 0;   // src: samples (16 bits each); dst: bytes
    int             readWidth = 0;   // src: the samples a row has from (0, 0) on (past the frame's width: the picture's)
  };

  // where the frame's block seeds go (FilmGrain::prepareBlockSeeds): count of them
  uint32_t* seeds( size_t count );
  // the grain of a frame of luma lumaWidth x lumaHeight -- its planes' subsampling, bit depth and models as grain has
  // them, its block seeds already written -- from src (16 bits a sample) into dst (dstBytes a sample; may be src).
  // Returns when the GPU is done.
  void apply( const FilmGrain& grain, int numPlanes, const Plane src[3], const Plane dst[3], int dstBytes, int lumaWidth, int lumaHeight );

private:
  Context*        m_ctx;
  Buffer          m_tables;   // the patterns, then sLUT and pLUT
  Buffer          m_seeds;
  VkCommandPool   m_pool  = VK_NULL_HANDLE;
  VkCommandBuffer m_cmd   = VK_NULL_HANDLE;
  VkFence         m_fence = VK_NULL_HANDLE;
};

}   // namespace gpu
}   // namespace vvdec
