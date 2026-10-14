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
    \brief    Film grain synthesis on the output frames, on the GPU
*/

#pragma once

#include "GpuContext.h"

namespace vvdec
{
class FilmGrain;

namespace gpu
{

// Adds film grain to output frames on the GPU (filmgrain.slang), bit-exact with vvdec's FilmGrain on the CPU
// (VVDecImpl::xAddGrain). The frames are in host memory the GPU can reach: the pictures' planes, or the output's
// copies of them. On a discrete GPU the shader works on copies of them in GPU memory (the copies in and out are in
// the frame's commands): there its own reads and writes of host memory cross the bus one by one (RTX 5090, a 4K
// frame's samples read and written: 1.6 s in host memory, 0.1 ms in its own). The grain of a frame is under way while
// the decoder goes on (up to kSlots frames at a time); each frame has a ticket to wait for (VVDecImpl hands a frame
// out once its grain is done).
class GpuFilmGrain
{
public:
  static constexpr int kSlots = 4;   // the frames whose grain may be under way at a time

  explicit GpuFilmGrain( Context* ctx );
  ~GpuFilmGrain();
  GpuFilmGrain( const GpuFilmGrain& )            = delete;
  GpuFilmGrain& operator=( const GpuFilmGrain& ) = delete;

  // A plane of a frame: the address of its sample (0, 0) and its stride; and the buffer it is in, with that sample's
  // offset in it (for the copies on a discrete GPU).
  struct Plane
  {
    VkDeviceAddress addr      = 0;
    int             stride    = 0;   // source: in samples (16 bits each); destination: in bytes
    int             readWidth = 0;   // source: the samples a row has from x = 0 on (can be wider than the frame: the picture's)
    VkBuffer        buffer    = VK_NULL_HANDLE;
    VkDeviceSize    offset    = 0;
  };

  // Where the next frame's block seeds go (FilmGrain::prepareBlockSeeds): room for count seeds, in the next frame's
  // slot (once the frame that had it before is done).
  uint32_t* seeds( size_t count );
  // Adds the grain to a frame of lumaWidth x lumaHeight luma samples, reading src (16 bits per sample) and writing
  // dst (dstBytes per sample; dst may be src). The planes' subsampling, the bit depth and the grain models come from
  // grain; the block seeds must already be written (seeds()). Returns at once, with the frame's ticket: src and dst
  // must stay as they are until the ticket is done.
  int64_t apply( const FilmGrain& grain, int numPlanes, const Plane src[3], const Plane dst[3], int dstBytes, int lumaWidth, int lumaHeight );
  // Whether the grain of the frame with the ticket is done; wait() returns once it is.
  bool done( int64_t ticket );
  void wait( int64_t ticket );

private:
  // The work of a frame: its tables (the grain patterns, then per component the scale table (sLUT), then the
  // pattern tables (pLUT)), its block seeds, on a discrete GPU the frame's copies in GPU memory (before and after the
  // grain), its commands; and the ticket of its frame while the GPU may be at it.
  struct Slot
  {
    Buffer          tables;
    Buffer          seeds;
    Buffer          before, after;
    std::vector<VkBufferCopy> rows;   // (the copies of the output's rows: kept for the next frame's)
    VkCommandPool   pool   = VK_NULL_HANDLE;
    VkCommandBuffer cmd    = VK_NULL_HANDLE;
    VkFence         fence  = VK_NULL_HANDLE;
    int64_t         ticket = -1;   // the frame under way (-1: none)
  };

  void finish( Slot& slot );   // waits until the slot's frame is done, and frees the slot

  Context* m_ctx;
  Slot     m_slots[kSlots];
  int64_t  m_nextTicket = 0;   // the next frame's (its slot: the ticket modulo kSlots)
};

}   // namespace gpu
}   // namespace vvdec
