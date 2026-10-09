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
  // LFNST (TrQuant::xInvLfnst): the CU's index (1, 2), the set of the block's
  // intra mode (g_lfnstLut), whether its output is transposed, whether it
  // makes the top-left 8x8 (the blocks at least 8x8) and whether it takes 8
  // inputs (the blocks of 4x4 and 8x8) rather than 16
  void setLfnst( int index, int set, bool transposed, bool block8, bool inputs8 )
  {
    w[6] = uint32_t( index ) | uint32_t( set ) << 2 | uint32_t( transposed ) << 4 | uint32_t( block8 ) << 5 | uint32_t( inputs8 ) << 6;
  }
  // an explicit scaling list (Quant's DeQuantScaling): where the block's
  // matrix (Quant::getDequantCoeff) begins in the picture's matrices, its
  // factor of the coefficient (x, y) at y * width + x
  void setScalingList( int offset ) { w[7] = uint32_t( offset + 1 ); }
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
    kMip     = 4,   // matrix-based intra prediction (the mode: the MIP mode)
    kLmcs    = 5,   // an inter CU's chroma under LMCS: its prediction (kept aside) plus its residual, scaled
    kCiip    = 6,   // a CIIP CU's component: inter's prediction (kept aside) blended with planar's, plus the residual
  };
  enum Flags
  {
    kResidual      = 1,   // the picture holds the block's residual
    kFiltered      = 2,   // the [1 2 1]-filtered references
    kMipTransposed = 4,
    kIsp           = 8,   // a luma sub-partition of an ISP CU (setIsp)
  };

  uint32_t w[12];

  // refLine: the reference line the block predicts from (MRL: luma's
  // cu.multiRefIdx(), 0 .. 2)
  void set( int x, int y, int width, int height, int comp, int kind, int mode, int flags, int bdpcmDir, int refLine = 0 )
  {
    w[0] = uint32_t( x ) | uint32_t( y ) << 16;
    w[1] = uint32_t( width ) | uint32_t( height ) << 8 | uint32_t( comp ) << 16 | uint32_t( kind ) << 24;
    w[2] = uint32_t( mode ) | uint32_t( flags ) << 8 | uint32_t( bdpcmDir ) << 16 | uint32_t( refLine ) << 24;
    w[3] = w[4] = w[5] = w[6] = w[7] = w[8] = w[9] = w[10] = w[11] = 0;
  }
  void setNeighbors( int topLeft, int above, int left ) { w[3] = uint32_t( topLeft ) | uint32_t( above ) << 8 | uint32_t( left ) << 16; }
  void setCclm( bool lumaAbove, bool lumaLeft, bool above, bool left, int actualTop, int actualLeft )
  {
    w[4] = uint32_t( lumaAbove ) << 16 | uint32_t( lumaLeft ) << 17 | uint32_t( above ) << 18 | uint32_t( left ) << 19;
    w[5] = uint32_t( actualTop ) | uint32_t( actualLeft ) << 16;
  }
  void setBlockVector( int x, int y ) { w[6] = uint32_t( uint16_t( int16_t( x ) ) ) | uint32_t( uint16_t( int16_t( y ) ) ) << 16; }
  // CIIP: planar's weight in the blend (predBlendIntraCiip's wIntra, 1 .. 3;
  // 0: inter's prediction as it is)
  void setCiip( int intraWeight ) { w[6] = uint32_t( intraWeight ); }
  // ISP: the CU (luma samples), its split (vvdec's ISPType: 1 horizontal, 2
  // vertical) and whether the neighbours left of and above the sub-partition's
  // prediction block are there (initIntraPatternChTypeISP's isLeftAvail and
  // isAboveAvail); the neighbours' units (setNeighbors) are the CU's
  void setIsp( int cuX, int cuY, int cuWidth, int cuHeight, int split, bool leftAvail, bool aboveAvail )
  {
    w[4] = uint32_t( cuX ) | uint32_t( cuY ) << 16;
    w[5] = uint32_t( cuWidth ) | uint32_t( cuHeight ) << 8 | uint32_t( split ) << 16 | uint32_t( leftAvail ) << 18 | uint32_t( aboveAvail ) << 19;
  }
  // LMCS's chroma residual scaling (Reshape::calculateChromaAdjVpduNei): the
  // luma of the CU at (x, y) its average comes from, left of it and above
  // it (where there), 1 << log2Count samples a side
  void setLmcs( int x, int y, bool left, bool above, int log2Count )
  {
    w[8] = uint32_t( x ) | uint32_t( y ) << 16;
    w[9] = 1u | uint32_t( left ) << 1 | uint32_t( above ) << 2 | uint32_t( log2Count ) << 4;
  }
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
    kLumaMap   = 128,  // LMCS: the luma prediction forward-mapped
    kDeferCb   = 256,  // and kDeferCb << 1 for Cr, under LMCS's chroma scaling: the prediction kept aside, the
                       // residual left in the picture, for an intra job of kind IntraJob::kLmcs
    kDeferY    = 1024, // CIIP: so for the luma too (with kDeferCb and Cr), for kind IntraJob::kCiip
    kBcw       = 2048, // BCW: the bi-prediction weighted (setBcw)
    kBdof      = 4096, // BDOF (with DMVR: where its cost leaves it on), in the dispatch of the inter_bdof variant
    kAffine    = 8192, // an affine CU's tile: this job and the next make an affine job (setAffine), for the inter_affine
                       // variant
    kGeo       = 16384, // GPM: the "L0" and "L1" predictions are the partitions', blended (setGeo)
    kWp        = 32768, // weighted prediction (setWp)
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
  // the width of the block whose position (cuX, cuY) and size clip the motion
  // (clipMv, wrapClipMv): the CU's, or SbTMVP's joined sub-PUs'
  void setClipWidth( int width ) { w[7] |= uint32_t( width ) << 20; }
  // BCW: L1's weight, of 8 (addWeightedAvg's w1; L0's: 8 - w1)
  void setBcw( int weight1 )
  {
    w[1] |= uint32_t( kBcw ) << 16;
    w[7] |= uint32_t( weight1 + 2 ) << 16;
  }

  // weighted prediction: per component its weights, offsets (at 8 bits) and
  // denominator, of the first list's prediction (uni-prediction: the list's
  // whose prediction it is) and, with bi, the second's
  void setWp( const int weight[2][3], const int offset[2][3], const int log2Denom[3], bool bi )
  {
    w[1] |= uint32_t( kWp ) << 16;
    for( int c = 0; c < 3; c++ )
    {
      w[8 + c] = ( uint32_t( weight[0][c] ) & 1023 ) | ( uint32_t( weight[1][c] ) & 1023 ) << 10 | uint32_t( log2Denom[c] ) << 20
                 | ( uint32_t( offset[0][c] ) & 255 ) << 24;
    }
    w[11] = ( uint32_t( offset[1][0] ) & 255 ) | ( uint32_t( offset[1][1] ) & 255 ) << 8 | ( uint32_t( offset[1][2] ) & 255 ) << 16 | uint32_t( bi ) << 24;
  }

  // GPM: the weights' mask (g_globalGeoWeights), its mirroring and the CU's
  // offsets into it (xWeightedGeoBlk)
  void setGeo( int mask, int mirror, int offsetX, int offsetY )
  {
    w[1] |= uint32_t( kGeo ) << 16;
    w[9] = uint32_t( mask ) | uint32_t( mirror ) << 8 | uint32_t( offsetX ) << 16 | uint32_t( offsetY ) << 24;
  }

  // an affine tile, as two jobs (shaders/inter.slang's AFFINE variant reads
  // them as one of 24 words): this one the tile (set, kAffine, the CU at
  // w[6]) and, at w[2..5], each list's base motion; the next one the CU's
  // size, whether PROF refines each list and whether its motion spreads too
  // far (one motion for all then), and each list's steps across and down:
  // the subblocks' motion as setAllAffineMv works it out (all << 7)
  static void setAffine( InterJob& job, InterJob& next, int cuWidth, int cuHeight, const bool prof[2], const bool spread[2], const int params[2][6] )
  {
    job.w[2]  = uint32_t( params[0][0] );
    job.w[3]  = uint32_t( params[0][1] );
    job.w[4]  = uint32_t( params[1][0] );
    job.w[5]  = uint32_t( params[1][1] );
    next.w[0] = uint32_t( cuWidth ) | uint32_t( cuHeight ) << 8 | uint32_t( prof[0] ) << 16 | uint32_t( prof[1] ) << 17
                | uint32_t( spread[0] ) << 18 | uint32_t( spread[1] ) << 19;
    for( int k = 0; k < 4; k++ )
    {
      next.w[1 + k] = uint32_t( params[0][2 + k] );
      next.w[5 + k] = uint32_t( params[1][2 + k] );
    }
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

// an edge the deblocking filters, as shaders/deblock.slang takes it: where
// (x | y << 16), and the bytes of its LoopFilterParam the shader needs
// (luma: qp | bs << 8 | sideMaxFiltLength << 16; chroma: Cb's qp | Cr's qp
// << 8 | bs << 16 | flags << 24) -- else the shader read them from the
// parameters of all the 4x4 blocks in host memory (at 4K, 3 MB a direction
// for the vertical edges: 0.47 ms a picture, 0.12 for the horizontal ones,
// which read half the cache lines)
struct Edge
{
  uint32_t xy;
  uint32_t params;
};

// the edges of a CTU the deblocking filters (GpuRecon::listEdges): luma's
// vertical and horizontal ones (luma samples, a 4x4 block's edge), chroma's
// (chroma samples, a segment as shaders/deblock.slang visits them)
struct EdgeLists
{
  std::vector<Edge> lists[4];

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
};

// a picture's LMCS for the GPU (Reshape::gpuTables): the tables of the 1 <<
// bit depth luma values
struct LmcsTables
{
  std::vector<int16_t> fwd;           // the forward luma mapping
  std::vector<int16_t> inv;           // the inverse one
  std::vector<int32_t> chromaScale;   // the chroma residual scale of an average luma
  bool                 chroma = false;   // the picture scales its chroma residuals
};

struct CuRecord;
struct TuRecord;

// a picture's jobs, per CTU, for the stages that run on the GPU (null: the CPU ran that stage)
struct PictureJobs
{
  const std::vector<std::vector<TrafoJob>>* trafo = nullptr;
  const std::vector<std::vector<InterJob>>* inter = nullptr;
  const std::vector<std::vector<IntraJob>>* intra = nullptr;
  const RefTable*                           refs  = nullptr;   // with inter
  const std::vector<EdgeLists>*             edges = nullptr;   // per CTU, with deblocking
  // the first CTUs, bandCtus of them, whose residuals and inter blocks went to
  // the GPU ahead of the rest (GpuRecon::startBand)
  size_t                                    bandCtus = 0;
  // the CUs of the parse, per CTU: the GPU derives their motion too, for the
  // comparison with the CPU's (VVDEC_GPU_MIDER; GpuRecon::compareMider)
  const std::vector<std::vector<CuRecord>>* cus = nullptr;
  const std::vector<std::vector<TuRecord>>* tus = nullptr;   // with cus: their TUs, per CTU
  // the GPU's front end drives the picture (VVDEC_GPU_FRONT): from cus and tus it derives the motion, the deblocking's
  // parameters and edges, the jobs (the residuals' with frontTrafo, inter's with frontInter) -- trafo, inter, intra and
  // edges unused
  bool                                      front      = false;
  bool                                      frontTrafo = false;
  bool                                      frontInter = false;
};

}   // namespace gpu
}   // namespace vvdec
