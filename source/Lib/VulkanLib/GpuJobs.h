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

/** \file     GpuJobs.h
    \brief    what the CPU hands the GPU's reconstruction of a picture, block by block
*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vvdec
{
namespace gpu
{

// one transform block of one component that the GPU predicts and
// reconstructs; the layout of shaders/intra.slang
struct IntraJob
{
  enum Kind
  {
    kAngular = 0,   // planar, DC and the angular modes
    kBdpcm   = 1,
    kCclm    = 2,
    kIbc     = 3,
  };
  enum Flags
  {
    kResidual = 1,   // the picture holds the block's residual
    kFiltered = 2,   // the [1 2 1]-filtered references
  };

  uint32_t w[8];

  void set( int x, int y, int width, int height, int comp, int kind, int mode, int flags, int bdpcmDir )
  {
    w[0] = uint32_t( x ) | uint32_t( y ) << 16;
    w[1] = uint32_t( width ) | uint32_t( height ) << 8 | uint32_t( comp ) << 16 | uint32_t( kind ) << 24;
    w[2] = uint32_t( mode ) | uint32_t( flags ) << 8 | uint32_t( bdpcmDir ) << 16;
    w[3] = w[4] = w[5] = w[6] = w[7] = 0;
  }
  void setNeighbors( int topLeft, int above, int left ) { w[3] = uint32_t( topLeft ) | uint32_t( above ) << 8 | uint32_t( left ) << 16; }
  void setCclm( bool lumaAbove, bool lumaLeft, bool above, bool left, int actualTop, int actualLeft )
  {
    w[4] = uint32_t( lumaAbove ) << 16 | uint32_t( lumaLeft ) << 17 | uint32_t( above ) << 18 | uint32_t( left ) << 19;
    w[5] = uint32_t( actualTop ) | uint32_t( actualLeft ) << 16;
  }
  void setBlockVector( int x, int y ) { w[6] = uint32_t( uint16_t( int16_t( x ) ) ) | uint32_t( uint16_t( int16_t( y ) ) ) << 16; }
  void setWave( int wave ) { w[7] = uint32_t( wave ); }
  int  wave() const { return int( w[7] ); }
};

// the wave of every block of a picture's planes, at the granularity of the
// smallest block: a block's wave is one above the highest wave of the samples
// it reads; inter blocks are wave 0, there before the waves begin
class WaveMap
{
public:
  void reset( int lumaWidth, int lumaHeight, int scaleX, int scaleY, bool chroma )
  {
    for( int c = 0; c < 3; c++ )
    {
      const int sx = c == 0 ? 0 : scaleX, sy = c == 0 ? 0 : scaleY;
      m_shiftX[c]  = 2 - sx;   // 4 luma samples
      m_shiftY[c]  = 2 - sy;
      m_width[c]   = c > 0 && !chroma ? 0 : ( ( lumaWidth >> sx ) + ( 1 << m_shiftX[c] ) - 1 ) >> m_shiftX[c];
      m_height[c]  = c > 0 && !chroma ? 0 : ( ( lumaHeight >> sy ) + ( 1 << m_shiftY[c] ) - 1 ) >> m_shiftY[c];
      m_map[c].assign( size_t( m_width[c] ) * m_height[c], 0 );
    }
  }

  // the highest wave in the rectangle (in samples of component c), clipped to the picture
  int maxIn( int c, int x, int y, int width, int height ) const
  {
    const int x0 = std::max( x, 0 ) >> m_shiftX[c];
    const int y0 = std::max( y, 0 ) >> m_shiftY[c];
    const int x1 = std::min( ( x + width - 1 ) >> m_shiftX[c], m_width[c] - 1 );
    const int y1 = std::min( ( y + height - 1 ) >> m_shiftY[c], m_height[c] - 1 );
    int       wave = 0;
    if( x + width <= 0 || y + height <= 0 )
    {
      return 0;
    }
    for( int j = y0; j <= y1; j++ )
    {
      const uint16_t* row = &m_map[c][size_t( j ) * m_width[c]];
      for( int i = x0; i <= x1; i++ )
      {
        wave = std::max<int>( wave, row[i] );
      }
    }
    return wave;
  }

  void set( int c, int x, int y, int width, int height, int wave )
  {
    const int x0 = x >> m_shiftX[c];
    const int y0 = y >> m_shiftY[c];
    const int x1 = std::min( ( x + width - 1 ) >> m_shiftX[c], m_width[c] - 1 );
    const int y1 = std::min( ( y + height - 1 ) >> m_shiftY[c], m_height[c] - 1 );
    for( int j = y0; j <= y1; j++ )
    {
      std::fill( &m_map[c][size_t( j ) * m_width[c] + x0], &m_map[c][size_t( j ) * m_width[c] + x1] + 1, uint16_t( wave ) );
    }
  }

private:
  std::vector<uint16_t> m_map[3];
  int                   m_width[3]  = { 0, 0, 0 };
  int                   m_height[3] = { 0, 0, 0 };
  int                   m_shiftX[3] = { 2, 1, 1 };
  int                   m_shiftY[3] = { 2, 1, 1 };
};

}   // namespace gpu
}   // namespace vvdec
