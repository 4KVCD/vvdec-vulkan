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
struct CodingUnit;

namespace gpu
{

// one transform block of one component whose levels the GPU dequantizes and
// inverse transforms into its residual; the layout of shaders/trafo.slang
struct TrafoJob
{
  enum Flags
  {
    kTransformSkip = 1,
    kBdpcmHor      = 2,
    kBdpcmVer      = 4,
    kZero          = 8,   // no residual: zeros (a block of an inter CU whose component has residuals elsewhere)
  };

  uint32_t w[8];

  // ict: the joint Cb-Cr mode (TU::getICTMode, signed) that derives the other
  // chroma component's residual from this one's, 0 for none
  void set( int x, int y, int log2Width, int log2Height, int comp, int typeHor, int typeVer, int flags, int ict, int lastX, int lastY, int scale,
            int rightShift, int inputMaximum )
  {
    w[0] = uint32_t( x ) | uint32_t( y ) << 16;
    w[1] = uint32_t( log2Width ) | uint32_t( log2Height ) << 4 | uint32_t( comp ) << 8 | uint32_t( typeHor ) << 12 | uint32_t( typeVer ) << 14
           | uint32_t( flags ) << 16 | uint32_t( ict + 4 ) << 24;
    w[2] = uint32_t( lastX ) | uint32_t( lastY ) << 8;
    w[3] = uint32_t( scale );
    w[4] = uint32_t( rightShift );
    w[5] = uint32_t( inputMaximum );
    w[6] = w[7] = 0;
  }
};

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
  int  width() const { return int( w[1] & 255 ); }
  int  height() const { return int( ( w[1] >> 8 ) & 255 ); }
  // on the GPU, in place of the wave: the rows of the block one workgroup
  // makes (GpuRecon::recordIntra cuts the big blocks into bands)
  void setRows( int first, int count ) { w[7] = uint32_t( first ) | uint32_t( count ) << 16; }
};

// a tile of up to 16x16 luma samples of an inter CU (DMVR's subblock when it
// refines) with its chroma; the layout of shaders/inter.slang
struct InterJob
{
  enum Flags
  {
    kL0        = 1,
    kL1        = 2,
    kDmvr      = 4,
    kHalfPel   = 8,    // AMVR's half-sample luma filter
    kResidualY = 16,   // and kResidualY << 1, << 2 for Cb, Cr: the picture holds the residual
  };

  uint32_t w[12];

  void set( int x, int y, int width, int height, int flags, int mv0x, int mv0y, int mv1x, int mv1y, int cuX, int cuY, int ref0, int ref1,
            int dmvrIndex )
  {
    w[0]  = uint32_t( x ) | uint32_t( y ) << 16;
    w[1]  = uint32_t( width ) | uint32_t( height ) << 8 | uint32_t( flags ) << 16;
    w[2]  = uint32_t( mv0x );
    w[3]  = uint32_t( mv0y );
    w[4]  = uint32_t( mv1x );
    w[5]  = uint32_t( mv1y );
    w[6]  = uint32_t( cuX ) | uint32_t( cuY ) << 16;
    w[7]  = uint32_t( ref0 ) | uint32_t( ref1 ) << 8;
    w[8]  = uint32_t( dmvrIndex );
    w[9]  = w[10] = w[11] = 0;
  }
};

// the reference pictures of a picture's inter jobs: a slot each
class RefTable
{
public:
  void clear() { m_pictures.clear(); }
  // during the picture's set-up, before the CTU tasks look slots up
  void add( const void* picture )
  {
    if( std::find( m_pictures.begin(), m_pictures.end(), picture ) == m_pictures.end() )
    {
      m_pictures.push_back( picture );
    }
  }
  int slot( const void* picture ) const { return int( std::find( m_pictures.begin(), m_pictures.end(), picture ) - m_pictures.begin() ); }
  const std::vector<const void*>& pictures() const { return m_pictures; }

private:
  std::vector<const void*> m_pictures;
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

// the edges of a CTU the deblocking filters (GpuRecon::listEdges), each
// x | y << 16: luma's vertical and horizontal ones (luma samples, a 4x4
// block's edge), chroma's (chroma samples, a segment as shaders/deblock.slang
// visits them)
struct EdgeLists
{
  std::vector<uint32_t> lists[4];

  void clear()
  {
    for( auto& list: lists )
    {
      list.clear();
    }
  }
};

// where the CTU tasks put a CTU's jobs for the stages of the picture that run
// on the GPU (null: the CPU runs that stage)
struct CtuJobs
{
  std::vector<TrafoJob>* trafo = nullptr;
  std::vector<InterJob>* inter = nullptr;
  std::vector<IntraJob>* intra = nullptr;
  WaveMap*               waves = nullptr;   // with intra
  const RefTable*        refs  = nullptr;   // with inter
  std::vector<const CodingUnit*>* dmvr = nullptr;   // with inter: the CUs DMVR refines
};

// a picture's jobs, per CTU, for the stages that run on the GPU (null: the CPU ran that stage)
struct PictureJobs
{
  const std::vector<std::vector<TrafoJob>>* trafo = nullptr;
  const std::vector<std::vector<InterJob>>* inter = nullptr;
  const std::vector<std::vector<IntraJob>>* intra = nullptr;
  const RefTable*                           refs  = nullptr;   // with inter
  const std::vector<EdgeLists>*             edges = nullptr;   // per CTU, with deblocking
};

}   // namespace gpu
}   // namespace vvdec
