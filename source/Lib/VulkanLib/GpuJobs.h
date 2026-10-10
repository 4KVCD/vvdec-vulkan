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
    \brief    The jobs: the work items of the GPU's reconstruction passes, one per block, in the layouts the shaders read

    The CTU tasks (CPU front end) or jobs.slang (GPU front end) fill these lists. Each job type's set functions are
    the reference for its layout; the shaders read the same words.
*/

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace vvdec
{
namespace gpu
{

// A transform block of one component. trafo.slang dequantizes its coefficients and inverse-transforms them into
// the residual. 8 words.
struct TrafoJob
{
  enum Flags
  {
    kTransformSkip = 1,
    kBdpcmHor      = 2,
    kBdpcmVer      = 4,
    kZero          = 8,   // no coefficients: the job writes zeros, for an inter CU with a residual elsewhere in the component
  };

  uint32_t w[8];

  // (x, y), log2Width, log2Height: the block, in samples of component comp. typeHor, typeVer: the transform types.
  // ict: the joint Cb-Cr mode (TU::getICTMode, signed) that derives the other chroma component's residual from this
  // one's; 0 for none. lastX, lastY: the largest x and y of a nonzero coefficient. scale, rightShift, inputMaximum:
  // the dequantization's scale and shift, and the limit the coefficient levels are clipped to.
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
  // LFNST (TrQuant::xInvLfnst): the CU's LFNST index (1 or 2); the transform set chosen by the block's intra mode
  // (g_lfnstLut); whether the output is transposed; whether the output covers the top-left 8x8 (blocks of at least
  // 8x8); and whether it takes 8 inputs (4x4 and 8x8 blocks) rather than 16.
  void setLfnst( int index, int set, bool transposed, bool block8, bool inputs8 )
  {
    w[6] = uint32_t( index ) | uint32_t( set ) << 2 | uint32_t( transposed ) << 4 | uint32_t( block8 ) << 5 | uint32_t( inputs8 ) << 6;
  }
  // An explicit scaling list (Quant's DeQuantScaling): offset is where the block's matrix (Quant::getDequantCoeff)
  // starts in the picture's matrices; the factor of coefficient (x, y) is at offset + y * width + x. Stored plus one,
  // so that 0 means no scaling list.
  void setScalingList( int offset ) { w[7] = uint32_t( offset + 1 ); }
};

// A transform block of one component that intra.slang predicts and reconstructs. 12 words.
struct IntraJob
{
  enum Kind
  {
    kAngular = 0,   // planar, DC and the angular modes
    kBdpcm   = 1,
    kCclm    = 2,
    kIbc     = 3,
    kMip     = 4,   // matrix-based intra prediction (the job's mode is the MIP mode)
    kLmcs    = 5,   // an inter CU's chroma under LMCS: its prediction (kept aside by inter.slang) plus its scaled residual
    kCiip    = 6,   // a CIIP CU's component: the inter prediction (kept aside) blended with planar's, plus the residual
  };
  enum Flags
  {
    kResidual      = 1,   // the picture holds the block's residual
    kFiltered      = 2,   // predict from the [1 2 1]-filtered reference samples
    kMipTransposed = 4,
    kIsp           = 8,   // a luma sub-partition of an ISP CU (setIsp)
  };

  uint32_t w[12];

  // refLine: the reference line the block predicts from (MRL: luma's cu.multiRefIdx(), 0 .. 2).
  void set( int x, int y, int width, int height, int comp, int kind, int mode, int flags, int bdpcmDir, int refLine = 0 )
  {
    w[0] = uint32_t( x ) | uint32_t( y ) << 16;
    w[1] = uint32_t( width ) | uint32_t( height ) << 8 | uint32_t( comp ) << 16 | uint32_t( kind ) << 24;
    w[2] = uint32_t( mode ) | uint32_t( flags ) << 8 | uint32_t( bdpcmDir ) << 16 | uint32_t( refLine ) << 24;
    w[3] = w[4] = w[5] = w[6] = w[7] = w[8] = w[9] = w[10] = w[11] = 0;
  }
  // How many units of reference samples are available top-left, above (with above-right) and left (with
  // below-left), as IntraPrediction's xDeriveNeighborSize counts them.
  void setNeighbors( int topLeft, int above, int left ) { w[3] = uint32_t( topLeft ) | uint32_t( above ) << 8 | uint32_t( left ) << 16; }
  // CCLM: whether the luma template and the chroma block have neighbours above and left, and how many neighbouring
  // samples are available above and left (IntraPrediction::getLmAvailability).
  void setCclm( bool lumaAbove, bool lumaLeft, bool above, bool left, int actualTop, int actualLeft )
  {
    w[4] = uint32_t( lumaAbove ) << 16 | uint32_t( lumaLeft ) << 17 | uint32_t( above ) << 18 | uint32_t( left ) << 19;
    w[5] = uint32_t( actualTop ) | uint32_t( actualLeft ) << 16;
  }
  // IBC: the block vector.
  void setBlockVector( int x, int y ) { w[6] = uint32_t( uint16_t( int16_t( x ) ) ) | uint32_t( uint16_t( int16_t( y ) ) ) << 16; }
  // CIIP: planar's weight in the blend (predBlendIntraCiip's wIntra, 1 .. 3); 0 leaves the inter prediction as it is.
  void setCiip( int intraWeight ) { w[6] = uint32_t( intraWeight ); }
  // ISP: the CU (in luma samples), its split (vvdec's ISPType: 1 horizontal, 2 vertical), and whether the
  // neighbours left of and above the sub-partition's prediction block are available (initIntraPatternChTypeISP's
  // isLeftAvail and isAboveAvail). The neighbour counts (setNeighbors) are the CU's.
  void setIsp( int cuX, int cuY, int cuWidth, int cuHeight, int split, bool leftAvail, bool aboveAvail )
  {
    w[4] = uint32_t( cuX ) | uint32_t( cuY ) << 16;
    w[5] = uint32_t( cuWidth ) | uint32_t( cuHeight ) << 8 | uint32_t( split ) << 16 | uint32_t( leftAvail ) << 18 | uint32_t( aboveAvail ) << 19;
  }
  // LMCS's chroma residual scaling (Reshape::calculateChromaAdjVpduNei): the scale comes from the average of the luma
  // samples left of and above the CU at (x, y), where available, 1 << log2Count samples on each side.
  void setLmcs( int x, int y, bool left, bool above, int log2Count )
  {
    w[8] = uint32_t( x ) | uint32_t( y ) << 16;
    w[9] = 1u | uint32_t( left ) << 1 | uint32_t( above ) << 2 | uint32_t( log2Count ) << 4;
  }
  // The job's wave (see WaveMap).
  void setWave( int wave ) { w[7] = uint32_t( wave ); }
  int  wave() const { return int( w[7] ); }
  int  width() const { return int( w[1] & 255 ); }
  int  height() const { return int( ( w[1] >> 8 ) & 255 ); }
  // On the GPU the wave's word holds the rows of the block that one workgroup makes instead: GpuRecon::recordIntra
  // cuts big blocks into bands.
  void setRows( int first, int count ) { w[7] = uint32_t( first ) | uint32_t( count ) << 16; }
};

// A tile of up to 16x16 luma samples of an inter CU, with its chroma, that inter.slang predicts (with DMVR, the tile
// is DMVR's subblock). 12 words.
struct InterJob
{
  enum Flags
  {
    kL0        = 1,
    kL1        = 2,
    kDmvr      = 4,
    kHalfPel   = 8,    // AMVR's half-sample luma filter
    kResidualY = 16,   // (and kResidualY << 1, << 2 for Cb, Cr) the picture holds a residual to add
    kLumaMap   = 128,  // LMCS: forward-map the luma prediction
    kDeferCb   = 256,  // (and kDeferCb << 1 for Cr) LMCS's chroma scaling: keep the prediction aside and leave the
                       // residual in the picture, for an intra job of kind IntraJob::kLmcs
    kDeferY    = 1024, // CIIP: the same for luma (with kDeferCb and its Cr flag), for an intra job of kind IntraJob::kCiip
    kBcw       = 2048, // BCW: weighted bi-prediction (setBcw)
    kBdof      = 4096, // BDOF (with DMVR: where DMVR's cost leaves it on), run by the inter_bdof variant
    kAffine    = 8192, // an affine CU's tile: this job and the next make one affine job (setAffine), run by the
                       // inter_affine variant
    kGeo       = 16384, // GPM: the "L0" and "L1" predictions are the two partitions', blended (setGeo)
    kWp        = 32768, // weighted prediction (setWp)
  };

  uint32_t w[12];

  // mv0, mv1: the motion of list 0 and list 1. (cuX, cuY): the CU's position, which clips the motion with the width
  // setClipWidth gives. ref0, ref1: the reference pictures' slots in the picture's RefTable. dmvrIndex: with DMVR,
  // where the tile's refinement goes in the picture's refinements (CodingUnit::mvdL0SubPuOff plus the subblock's
  // index).
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
  // The width of the block that clips the motion (clipMv, wrapClipMv): the CU's, or with SbTMVP the joined sub-PUs'.
  void setClipWidth( int width ) { w[7] |= uint32_t( width ) << 20; }
  // BCW: list 1's weight out of 8 (addWeightedAvg's w1; list 0's is 8 - w1). Stored plus 2, as the weights go down
  // to -2.
  void setBcw( int weight1 )
  {
    w[1] |= uint32_t( kBcw ) << 16;
    w[7] |= uint32_t( weight1 + 2 ) << 16;
  }

  // Weighted prediction: per component, the weight, offset (at 8 bits) and log2 denominator of the first prediction
  // (for uni-prediction, of the list it predicts from) and, with bi, the second prediction's weight and offset.
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

  // GPM: the weight mask (g_globalGeoWeights), whether it is mirrored, and the CU's offsets into it (xWeightedGeoBlk).
  void setGeo( int mask, int mirror, int offsetX, int offsetY )
  {
    w[1] |= uint32_t( kGeo ) << 16;
    w[9] = uint32_t( mask ) | uint32_t( mirror ) << 8 | uint32_t( offsetX ) << 16 | uint32_t( offsetY ) << 24;
  }

  // An affine tile takes two jobs, which inter.slang's affine variant reads as one job of 24 words. The first is the
  // tile (set with kAffine and the CU's position in w[6]) with each list's base motion in w[2..5]. The second holds
  // the CU's size; whether PROF refines each list; whether each list's motion spreads too far (then every subblock
  // takes the same motion); and each list's motion steps across and down. Together they give the subblocks' motion
  // as setAllAffineMv works it out (all << 7).
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

// The reference pictures of a picture's inter jobs, one slot each.
class RefTable
{
public:
  void clear() { m_pictures.clear(); }
  // During the picture's set-up, before the CTU tasks look slots up.
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

// The wave of every 4x4 luma area of a picture's planes (2x2 chroma samples in 4:2:0). A block's wave is one more
// than the highest wave of the samples it reads. Inter blocks are wave 0: they are done before the waves start.
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

  // The highest wave in the rectangle (in samples of component c), clipped to the picture.
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

// An edge the deblocking filters, as deblock.slang reads it: its position (x | y << 16) and the bytes of its
// LoopFilterParam the shader needs (luma: qp | bs << 8 | sideMaxFiltLength << 16; chroma: Cb's qp | Cr's qp << 8 |
// bs << 16 | flags << 24). Carrying them here spares the shader reading the parameters of every 4x4 block from host
// memory (at 4K, 3 MB per direction: 0.47 ms per picture for the vertical edges, 0.12 ms for the horizontal ones,
// which read half the cache lines).
struct Edge
{
  uint32_t xy;
  uint32_t params;
};

// The edges of a CTU that the deblocking filters (GpuRecon::listEdges). lists[0] and lists[1]: luma's vertical and
// horizontal edges, in luma samples, one per edge of a 4x4 block. lists[2] and lists[3]: chroma's, in chroma
// samples, one per segment as deblock.slang visits them.
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

// Where the CTU tasks put a CTU's jobs for the passes that run on the GPU (null: the CPU runs that pass).
struct CtuJobs
{
  std::vector<TrafoJob>* trafo = nullptr;
  std::vector<InterJob>* inter = nullptr;
  std::vector<IntraJob>* intra = nullptr;
  WaveMap*               waves = nullptr;   // with intra
  const RefTable*        refs  = nullptr;   // with inter
};

// A picture's LMCS tables for the GPU (Reshape::gpuTables), one entry per luma value (1 << bit depth of them).
struct LmcsTables
{
  std::vector<int16_t> fwd;           // the forward luma mapping
  std::vector<int16_t> inv;           // the inverse luma mapping
  std::vector<int32_t> chromaScale;   // the chroma residual scale for an average luma value
  bool                 chroma = false;   // the picture scales its chroma residuals
};

struct CuRecord;
struct TuRecord;

// A picture's jobs, per CTU, for the passes that run on the GPU (null: the CPU ran that pass).
class ParseSlot;

struct PictureJobs
{
  const std::vector<std::vector<TrafoJob>>* trafo = nullptr;
  const std::vector<std::vector<InterJob>>* inter = nullptr;
  const std::vector<std::vector<IntraJob>>* intra = nullptr;
  const RefTable*                           refs  = nullptr;   // with inter
  const std::vector<EdgeLists>*             edges = nullptr;   // per CTU, with deblocking
  // The early rows (CPU front end): the first earlyCtus CTUs, whose residuals and inter blocks went to the GPU ahead
  // of the rest (GpuRecon::startEarlyRows).
  size_t                                    earlyCtus = 0;
  // The parse's CU and TU records, per CTU (GpuFrontEnd.h). The GPU front end works from them: it drives the picture
  // with front, and without it runs beside the CPU front end for the comparison (VVDEC_GPU_MIDER=1;
  // GpuRecon::compareFrontEnd).
  const std::vector<std::vector<CuRecord>>* cus = nullptr;
  const std::vector<std::vector<TuRecord>>* tus = nullptr;
  // The GPU front end drives the picture: from cus and tus it derives the motion, the deblocking parameters and
  // edges, and the jobs (the residual jobs with frontEndTrafo, the inter jobs with frontEndInter). trafo, inter,
  // intra and edges are then unused.
  bool                                      useFrontEnd   = false;
  bool                                      frontEndTrafo = false;
  bool                                      frontEndInter = false;
  // VVDEC_GPU_PARSE=1: the GPU parses the picture's slice data too, compared with the CPU's parse (cus, tus)
  bool                                      parse         = false;
  // VVDEC_GPU_PARSE=2: the GPU parsed the picture instead of the CPU (ParseService): the front end takes its records
  // from this slot, and cus and tus are empty
  const ParseSlot*                          parsed        = nullptr;
};

}   // namespace gpu
}   // namespace vvdec
