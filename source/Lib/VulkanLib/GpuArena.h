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

/** \file     GpuArena.h
    \brief    GPU memory handed out in ranges of a few large buffers (RangeArena)
*/

#pragma once

#include "GpuContext.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace vvdec
{
namespace gpu
{

// GPU memory handed out in ranges of a few large buffers (chunks), for buffers of many sizes that come and go all
// the time (the parse's records: a few for each picture). A buffer of its own for each would cost the thread a
// fraction of a millisecond to allocate and again to free; buffers kept for reuse in power-of-two sizes would hold
// about twice the memory in use at once (Beekeeper slower 4K: 176 to 195 MB for up to 96 MB in use) and need
// hundreds of allocations, where the chunks need a dozen. A range is a Buffer that owns no memory
// (Buffer::isRange): buffer is its chunk's, offset where it starts there; give() takes it back, Context::free refuses
// it. A range goes into the smallest gap that holds it; a chunk with no range taken for idleTicks ticks is freed.
class RangeArena
{
public:
  RangeArena( Context* ctx, uint64_t idleTicks ) : m_ctx( ctx ), m_idleTicks( idleTicks ) {}
  ~RangeArena();   // frees the chunks (the ranges still taken go with them)
  RangeArena( const RangeArena& )            = delete;
  RangeArena& operator=( const RangeArena& ) = delete;

  // The size of the chunks allocated from now on (a range larger than that gets a chunk of its own)
  void   setChunkSize( VkDeviceSize size ) { m_chunkSize = size; }
  // The chunks' size for pictures of lumaSamples samples: twice that in bytes, in steps of 1 MB, at most 8 MB.
  // (Replays of the parse's takes and gives: Beekeeper slower 4K, up to 96 MB of records and levels at once, held
  // 105 MB in 8 MB chunks, 112 MB in 16 MB ones, 128 MB in 32 MB ones; Top Gun slower 1080p, up to 76 MB at once, held
  // 80 MB in 4 MB chunks.)
  static VkDeviceSize chunkSizeFor( size_t lumaSamples );
  // A range of GPU memory (MemKind::Device) of at least size bytes, in steps of kAlign
  Buffer take( VkDeviceSize size );
  // Takes the range back, for the next takes: the GPU must be done with it (or have its work on it queued before
  // whatever takes it next). range is left empty.
  void   give( Buffer& range );
  // Counts a tick, and frees the chunks with no range taken for idleTicks ticks
  void   tick();

private:
  struct Chunk
  {
    Buffer                               buf;
    std::map<VkDeviceSize, VkDeviceSize> gaps;           // the free ranges: start -> size
    VkDeviceSize                         taken    = 0;   // the bytes of the ranges taken
    uint64_t                             lastTake = 0;   // the tick of the last range taken
  };
  static constexpr VkDeviceSize kAlign = 256;            // (a range never shares a cache line with another)
  static constexpr VkDeviceSize kLarge = 1 << 20;        // a chunk of a range's own: its size in steps of this

  Chunk& addChunk( VkDeviceSize size );
  Chunk& chunkOf( const Buffer& range );
  // The gap of the chunk that holds size bytes and is the smallest (the first of those); false: none
  bool   bestGap( VkDeviceSize size, Chunk*& chunk, VkDeviceSize& start ) const;
  // Makes the bytes from start a gap again, merged with the gaps beside them
  static void addGap( Chunk& chunk, VkDeviceSize start, VkDeviceSize size );

  Context*                            m_ctx;
  uint64_t                            m_idleTicks;
  uint64_t                            m_ticks     = 0;
  VkDeviceSize                        m_chunkSize = VkDeviceSize( 8 ) << 20;
  std::vector<std::unique_ptr<Chunk>> m_chunks;   // (in the order they were allocated)
};

}   // namespace gpu
}   // namespace vvdec
