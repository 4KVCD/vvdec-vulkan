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

/** \file     GpuArena.cpp
    \brief    GPU memory handed out in ranges of a few large buffers (RangeArena); see GpuArena.h
*/

#include "GpuArena.h"

#include "CommonLib/CommonDef.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace vvdec
{
namespace gpu
{

RangeArena::~RangeArena()
{
  for( auto& chunk: m_chunks )
  {
    m_ctx->free( chunk->buf );
  }
}

Buffer RangeArena::take( VkDeviceSize size )
{
  size = ( std::max<VkDeviceSize>( size, 1 ) + kAlign - 1 ) & ~( kAlign - 1 );
  Chunk*       chunk = nullptr;
  VkDeviceSize start = 0;
  if( !bestGap( size, chunk, start ) )
  {
    // none holds it: a new chunk (one of its own for a range larger than the chunks)
    chunk = &addChunk( std::max( m_chunkSize, ( size + kLarge - 1 ) & ~( kLarge - 1 ) ) );
    start = 0;
  }
  const auto         gap  = chunk->gaps.find( start );
  const VkDeviceSize room = gap->second;
  chunk->gaps.erase( gap );
  if( room > size )
  {
    chunk->gaps.emplace( start + size, room - size );   // (the rest of the gap)
  }
  chunk->taken += size;
  chunk->lastTake = m_ticks;

  Buffer range;
  range.buffer = chunk->buf.buffer;
  range.size   = size;
  range.addr   = chunk->buf.addr + start;
  range.kind   = MemKind::Device;
  range.offset = start;
  return range;
}

void RangeArena::give( Buffer& range )
{
  if( !range.valid() )
  {
    return;
  }
  Chunk& chunk = chunkOf( range );
  chunk.taken -= range.size;
  addGap( chunk, range.offset, range.size );
  range = Buffer();
}

void RangeArena::tick()
{
  m_ticks++;
  for( size_t i = 0; i < m_chunks.size(); )
  {
    Chunk& chunk = *m_chunks[i];
    if( chunk.taken == 0 && m_ticks - chunk.lastTake > m_idleTicks )
    {
      m_ctx->free( chunk.buf );   // (no range taken: the GPU is done with all of it)
      m_chunks.erase( m_chunks.begin() + i );
      continue;
    }
    i++;
  }
}

RangeArena::Chunk& RangeArena::addChunk( VkDeviceSize size )
{
  auto chunk = std::make_unique<Chunk>();
  chunk->buf = m_ctx->alloc( size_t( size ), MemKind::Device );
  chunk->gaps.emplace( 0, chunk->buf.size );
  m_chunks.push_back( std::move( chunk ) );
  return *m_chunks.back();
}

RangeArena::Chunk& RangeArena::chunkOf( const Buffer& range )
{
  const auto it = std::find_if( m_chunks.begin(), m_chunks.end(),
                                [&]( const std::unique_ptr<Chunk>& chunk ) { return chunk->buf.buffer == range.buffer; } );
  CHECK_FATAL( it == m_chunks.end() || !range.isRange(), "Vulkan: a buffer given back to an arena that did not hand it out" );
  return **it;
}

bool RangeArena::bestGap( VkDeviceSize size, Chunk*& chunk, VkDeviceSize& start ) const
{
  VkDeviceSize best = 0;
  for( const auto& candidate: m_chunks )
  {
    for( const auto& gap: candidate->gaps )
    {
      if( gap.second >= size && ( best == 0 || gap.second < best ) )
      {
        best  = gap.second;
        chunk = candidate.get();
        start = gap.first;
      }
    }
  }
  return best > 0;
}

void RangeArena::addGap( Chunk& chunk, VkDeviceSize start, VkDeviceSize size )
{
  auto next = chunk.gaps.lower_bound( start );
  if( next != chunk.gaps.end() && next->first == start + size )
  {
    size += next->second;   // (the gap after it, merged)
    next = chunk.gaps.erase( next );
  }
  if( next != chunk.gaps.begin() )
  {
    const auto prev = std::prev( next );
    if( prev->first + prev->second == start )
    {
      prev->second += size;   // (merged into the gap before it)
      return;
    }
  }
  chunk.gaps.emplace_hint( next, start, size );
}

}   // namespace gpu
}   // namespace vvdec
