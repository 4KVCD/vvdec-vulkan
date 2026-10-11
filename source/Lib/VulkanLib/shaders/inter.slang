// Inter prediction and the reconstruction of inter blocks, as vvdec's
// InterPrediction and InterpolationFilter do them: uni- and bi-prediction with
// the 8-tap luma and 4-tap chroma filters (and AMVR's half-sample filter),
// and decoder-side motion vector refinement (DMVR). A workgroup takes one
// tile of up to 16x16 luma samples of a CU (the DMVR subblock when DMVR
// refines the CU) together with its chroma, predicts it, adds the residual
// that the picture holds and writes the result; its 64 threads share out the
// tile's samples.
//
// A job is 12 uints (VulkanLib/GpuJobs.h, InterJob):
//   0: x | y << 16 of the tile (luma)      1: width | height << 8 | flags << 16
//   2, 3: L0 motion vector (1/16 luma)     4, 5: L1 motion vector
//   6: x | y << 16 of the CU               7: L0 reference | L1 reference << 8 (slots in the reference table)
//   8: where DMVR's refinement goes (vvdec's DMVR MV cache)
//   7, bits 16..19: BCW: L1's weight + 2 (out of 8; L0's is 8 minus L1's), with flag 2048; bits 20..31: the width of
//      the block whose position and size clip the motion (the CU, or SbTMVP's joined sub-PUs)
// flags: 1 L0, 2 L1, 4 DMVR, 8 AMVR half-sample filter, 16/32/64 residual in Y/Cb/Cr, 128 LMCS forward mapping of
// the luma prediction, 256/512 (LMCS chroma scaling, CIIP) keep the Cb/Cr prediction aside (defer1/2) and leave the
// residual in the picture for intra.slang's kind 5 or 6, 1024 (CIIP) the same for luma (defer0), 2048 BCW weights,
// 4096 BDOF (the variant built with BDOF=1 takes those tiles), 8192 an affine tile (the variant built with AFFINE=1
// takes those, two jobs each: see affineMain), 16384 GPM (the partitions' predictions as L0's and L1's, blended with
// the weights of word 9: mask | mirror << 8 | offset x << 16 | offset y << 24), 32768 weighted prediction (see
// wpSample)

#include "common.slang"

// A reference picture (GpuRecon::writeRefTable): its planes and, for RPR, its
// size, strides, scaling ratio (SCALE_RATIO_BITS: 1 << 14 is 1x), scaling
// window offset (luma samples) and whether it is scaled (Picture::isRefScaled).
struct RefPlanes
{
  int16_t* y;
  int16_t* cb;
  int16_t* cr;
  uint     size;      // luma: width | height << 16
  uint     strides;   // luma | chroma << 16
  int      ratioX;
  int      ratioY;
  int      winLeft;
  int      winTop;
  int      scaled;
  int      unused0;
  int      unused1;
  int      unused2;
};

struct Params
{
  uint*      jobs;
  RefPlanes* refs;
  int16_t*   dst0;
  int16_t*   dst1;
  int16_t*   dst2;
  int*       dmvr;        // DMVR's refinements (vvdec's DMVR MV cache, as vvdec's Mv: hor, ver)
  int*       colMotion;   // the picture's colocated motion: vvdec's ColocatedMotionInfo, 5 ints
  int16_t*   lmcsFwd;     // LMCS: the forward luma mapping, one entry per luma value
  int16_t*   defer0;      // LMCS, CIIP: the predictions kept aside, laid out like the planes
  int16_t*   defer1;
  int16_t*   defer2;
  int16_t*   geo;         // GPM's prestored masks of weights (g_globalGeoWeights), 112 x 112 each
  // with subpicOn: per CTU, the rectangle (luma: left, top, right, bottom, inclusive) that the CTU's blocks read from
  // and clip their motion to: its subpicture's if that is treated as a picture, else the picture's
  uint16_t*  subpics;
  int        stride0;
  int        stride12;
  int        count;       // number of jobs; negative: countAt holds it
  int        picWidth;    // luma samples
  int        picHeight;
  int        ctuSize;
  int        bitDepth;
  int        ctuLog2;     // the CTU size of the colocated motion's layout
  int        widthInCtus;
  int        num8x8;      // colocated-motion entries per CTU (one per 8x8 block)
  int        wide;        // nonzero when the planes' rows start 16-byte aligned with margins of 16 samples or more, so
                          // tile16 may load them in 8- and 16-byte pieces (and the chroma is 4:2:0, as tile16 expects)
  int        scaleX;      // chroma subsampling shifts: 1, 1 (4:2:0), 1, 0 (4:2:2), 0, 0 (4:4:4)
  int        scaleY;
  int        numComp;     // number of components: 1 (4:0:0) or 3
  // Wraparound (vvdec's): with the SPS's flag, clipMv acts as wrapClipMv and motion compensation reads the reference's
  // wraparound copy; with the PPS's flag, the copy's margins wrap by wrapOffset (luma samples; 0 without the PPS's
  // flag), and the flag also decides where DMVR and affine read.
  int        wrapSps;
  int        wrapPps;
  int        wrapOffset;
  int        subpicOn;    // nonzero when subpictures are treated as pictures (see subpics)
  int        curWinLeft;  // RPR: the current picture's scaling window offset (luma samples)
  int        curWinTop;
  int        horColloc;   // RPR: the chroma siting (the SPS's collocated flags)
  int        verColloc;
  uint*      countAt;     // when count < 0: the number of jobs and the index of the first, from the GPU front end
};

[[vk::push_constant]] Params pc;

// The pass's jobs start at the first job that the front end wrote, or at the buffer's start.
uint* jobBase() { return pc.jobs + ( pc.count >= 0 ? 0u : pc.countAt[1] ) * 12; }

#define FLAG_L0 1
#define FLAG_L1 2
#define FLAG_DMVR 4
#define FLAG_HPEL 8
#define FLAG_RES_Y 16
#define FLAG_LUMA_MAP 128
#define FLAG_DEFER_CB 256
#define FLAG_DEFER_Y 1024
#define FLAG_BCW 2048
#define FLAG_BDOF 4096
#define FLAG_GEO 16384
#define FLAG_WP 32768

// Weighted prediction (addWeightUni, addWeightBi) of a component's 14-bit predictions. Job word 8 + comp holds
// w0 | w1 << 10 | log2 denominator << 20 | o0 << 24; word 11 holds o1 of Y, Cb and Cr (8 bits each) | bi << 24.
// The offsets are at 8-bit precision; a uni-prediction keeps its list's weight and offset in w0 and o0.
int wpSample( uint params, uint offsets1, int comp, int p0, int p1, int maxVal )
{
  const int w0    = int( params << 22 ) >> 22;
  const int w1    = int( ( params >> 10 ) << 22 ) >> 22;
  const int denom = int( ( params >> 20 ) & 15 );
  const int scale = 1 << ( pc.bitDepth - 8 );
  const int o0    = ( int( params ) >> 24 ) * scale;
  const int head  = max( 2, 14 - pc.bitDepth );
  if( ( offsets1 >> 24 ) != 0 )
  {
    const int o1    = ( int( ( offsets1 >> ( 8 * comp ) ) << 24 ) >> 24 ) * scale;
    const int shift = denom + 1 + head;
    const int add   = ( 1 << ( shift - 1 ) ) + ( o0 + o1 ) * ( 1 << ( shift - 1 ) ) + ( w0 + w1 ) * 8192;   // IF_INTERNAL_OFFS
    return clip3( 0, maxVal, ( w0 * p0 + w1 * p1 + add ) >> shift );
  }
  const int shift = denom + head;
  const int round = shift > 0 ? 1 << ( shift - 1 ) : 0;
  return clip3( 0, maxVal, ( ( w0 * ( p0 + 8192 ) + round ) >> shift ) + o0 );
}

// GPM (xWeightedGeoBlk): the weight (out of 8) of the first partition at the
// CU's luma sample (lx, ly), read from the mask, mirrored as needed, at the
// CU's offsets.
int geoWeight( uint params, int lx, int ly )
{
  const int mask   = int( params & 255 );
  const int mirror = int( ( params >> 8 ) & 255 );
  const int ox     = int( ( params >> 16 ) & 255 );
  const int oy     = int( params >> 24 );
  const int at     = mirror == 2 ? ( 111 - oy - ly ) * 112 + ox + lx : ( oy + ly ) * 112 + ( mirror == 1 ? 111 - ox - lx : ox + lx );
  return int( pc.geo[mask * 112 * 112 + at] );
}

// Blends the partitions' 14-bit predictions.
int geoBlend( int p0, int p1, int weight, int maxVal )
{
  const int shift = max( 2, 14 - pc.bitDepth ) + 3;
  return clip3( 0, maxVal, ( weight * p0 + ( 8 - weight ) * p1 + ( 1 << ( shift - 1 ) ) + ( 8192 << 3 ) ) >> shift );   // IF_INTERNAL_OFFS << 3
}
#ifndef BDOF
#define BDOF 0
#endif
#ifndef AFFINE
#define AFFINE 0
#endif

// The average of a bi-prediction's two 14-bit predictions (addAvg), or their
// weighted average with BCW's weights (addWeightedAvg; w1 is out of 8).
int biAverage( int p0, int p1, int flags, int w1, int shiftNum, int offset, int maxVal )
{
  if( ( flags & FLAG_BCW ) != 0 )
  {
    const int shift = shiftNum + 2;
    return clip3( 0, maxVal, ( ( 8 - w1 ) * p0 + w1 * p1 + ( 1 << ( shift - 1 ) ) + ( 8192 << 3 ) ) >> shift );   // IF_INTERNAL_OFFS << 3
  }
  return clip3( 0, maxVal, ( p0 + p1 + offset ) >> shiftNum );
}

#define IF_INTERNAL_OFFS 8192
#define THREADS 64
#define PER_THREAD 4   // a 16x16 tile's luma samples per thread

static const int kLuma[128] = {
   0, 0,   0, 64,  0,   0,  0,  0,
   0, 1,  -3, 63,  4,  -2,  1,  0,
  -1, 2,  -5, 62,  8,  -3,  1,  0,
  -1, 3,  -8, 60, 13,  -4,  1,  0,
  -1, 4, -10, 58, 17,  -5,  1,  0,
  -1, 4, -11, 52, 26,  -8,  3, -1,
  -1, 3,  -9, 47, 31, -10,  4, -1,
  -1, 4, -11, 45, 34, -10,  4, -1,
  -1, 4, -11, 40, 40, -11,  4, -1,
  -1, 4, -10, 34, 45, -11,  4, -1,
  -1, 4, -10, 31, 47,  -9,  3, -1,
  -1, 3,  -8, 26, 52, -11,  4, -1,
   0, 1,  -5, 17, 58, -10,  4, -1,
   0, 1,  -4, 13, 60,  -8,  3, -1,
   0, 1,  -3,  8, 62,  -5,  2, -1,
   0, 1,  -2,  4, 63,  -3,  1,  0
};
static const int kLumaHpel[8] = { 0, 3, 9, 20, 20, 9, 3, 0 };
static const int kChroma[128] = {
   0, 64,  0,  0,  -1, 63,  2,  0,  -2, 62,  4,  0,  -2, 60,  7, -1,  -2, 58, 10, -2,  -3, 57, 12, -2,  -4, 56, 14, -2,  -4, 55, 15, -2,
  -4, 54, 16, -2,  -5, 53, 18, -2,  -6, 52, 20, -2,  -6, 49, 24, -3,  -6, 46, 28, -4,  -5, 44, 29, -4,  -4, 42, 30, -4,  -4, 39, 33, -4,
  -4, 36, 36, -4,  -4, 33, 39, -4,  -4, 30, 42, -4,  -4, 29, 44, -5,  -4, 28, 46, -6,  -3, 24, 49, -6,  -2, 20, 52, -6,  -2, 18, 53, -5,
  -2, 16, 54, -4,  -2, 15, 55, -4,  -2, 14, 56, -4,  -2, 12, 57, -3,  -2, 10, 58, -2,  -1,  7, 60, -2,   0,  4, 62, -2,   0,  2, 63, -1
};
static const int16_t kBilinear[32] = { 16, 0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
// The 6-tap luma filter vvdec uses for 4x4 blocks (affine), then RPR's filters
// (InterpolationFilter's m_lumaFilterRPR1 and 2, used past 1.25x and 1.75x;
// then the affine variants; then the chroma ones).
static const int kLuma4x4[128] = {
   0, 0,   0, 64,  0,   0,  0,  0,
   0, 1,  -3, 63,  4,  -2,  1,  0,
   0, 1,  -5, 62,  8,  -3,  1,  0,
   0, 2,  -8, 60, 13,  -4,  1,  0,
   0, 3, -10, 58, 17,  -5,  1,  0,
   0, 3, -11, 52, 26,  -8,  2,  0,
   0, 2,  -9, 47, 31, -10,  3,  0,
   0, 3, -11, 45, 34, -10,  3,  0,
   0, 3, -11, 40, 40, -11,  3,  0,
   0, 3, -10, 34, 45, -11,  3,  0,
   0, 3, -10, 31, 47,  -9,  2,  0,
   0, 2,  -8, 26, 52, -11,  3,  0,
   0, 1,  -5, 17, 58, -10,  3,  0,
   0, 1,  -4, 13, 60,  -8,  2,  0,
   0, 1,  -3,  8, 62,  -5,  1,  0,
   0, 1,  -2,  4, 63,  -3,  1,  0
};
static const int kLumaRPR1[128] = {
   -1,  -5,  17,  42,  17,  -5,  -1,   0,
    0,  -5,  15,  41,  19,  -5,  -1,   0,
    0,  -5,  13,  40,  21,  -4,  -1,   0,
    0,  -5,  11,  39,  24,  -4,  -2,   1,
    0,  -5,   9,  38,  26,  -3,  -2,   1,
    0,  -5,   7,  38,  28,  -2,  -3,   1,
    1,  -5,   5,  36,  30,  -1,  -3,   1,
    1,  -4,   3,  35,  32,   0,  -4,   1,
    1,  -4,   2,  33,  33,   2,  -4,   1,
    1,  -4,   0,  32,  35,   3,  -4,   1,
    1,  -3,  -1,  30,  36,   5,  -5,   1,
    1,  -3,  -2,  28,  38,   7,  -5,   0,
    1,  -2,  -3,  26,  38,   9,  -5,   0,
    1,  -2,  -4,  24,  39,  11,  -5,   0,
    0,  -1,  -4,  21,  40,  13,  -5,   0,
    0,  -1,  -5,  19,  41,  15,  -5,   0
};
static const int kLumaRPR2[128] = {
   -4,   2,  20,  28,  20,   2,  -4,   0,
   -4,   0,  19,  29,  21,   5,  -4,  -2,
   -4,  -1,  18,  29,  22,   6,  -4,  -2,
   -4,  -1,  16,  29,  23,   7,  -4,  -2,
   -4,  -1,  16,  28,  24,   7,  -4,  -2,
   -4,  -1,  14,  28,  25,   8,  -4,  -2,
   -3,  -3,  14,  27,  26,   9,  -3,  -3,
   -3,  -1,  12,  28,  25,  10,  -4,  -3,
   -3,  -3,  11,  27,  27,  11,  -3,  -3,
   -3,  -4,  10,  25,  28,  12,  -1,  -3,
   -3,  -3,   9,  26,  27,  14,  -3,  -3,
   -2,  -4,   8,  25,  28,  14,  -1,  -4,
   -2,  -4,   7,  24,  28,  16,  -1,  -4,
   -2,  -4,   7,  23,  29,  16,  -1,  -4,
   -2,  -4,   6,  22,  29,  18,  -1,  -4,
   -2,  -4,   5,  21,  29,  19,   0,  -4
};
static const int kAffineLumaRPR1[128] = {
    0,  -6,  17,  42,  17,  -5,  -1,   0,
    0,  -5,  15,  41,  19,  -5,  -1,   0,
    0,  -5,  13,  40,  21,  -4,  -1,   0,
    0,  -5,  11,  39,  24,  -4,  -1,   0,
    0,  -5,   9,  38,  26,  -3,  -1,   0,
    0,  -5,   7,  38,  28,  -2,  -2,   0,
    0,  -4,   5,  36,  30,  -1,  -2,   0,
    0,  -3,   3,  35,  32,   0,  -3,   0,
    0,  -3,   2,  33,  33,   2,  -3,   0,
    0,  -3,   0,  32,  35,   3,  -3,   0,
    0,  -2,  -1,  30,  36,   5,  -4,   0,
    0,  -2,  -2,  28,  38,   7,  -5,   0,
    0,  -1,  -3,  26,  38,   9,  -5,   0,
    0,  -1,  -4,  24,  39,  11,  -5,   0,
    0,  -1,  -4,  21,  40,  13,  -5,   0,
    0,  -1,  -5,  19,  41,  15,  -5,   0
};
static const int kAffineLumaRPR2[128] = {
    0,  -2,  20,  28,  20,   2,  -4,   0,
    0,  -4,  19,  29,  21,   5,  -6,   0,
    0,  -5,  18,  29,  22,   6,  -6,   0,
    0,  -5,  16,  29,  23,   7,  -6,   0,
    0,  -5,  16,  28,  24,   7,  -6,   0,
    0,  -5,  14,  28,  25,   8,  -6,   0,
    0,  -6,  14,  27,  26,   9,  -6,   0,
    0,  -4,  12,  28,  25,  10,  -7,   0,
    0,  -6,  11,  27,  27,  11,  -6,   0,
    0,  -7,  10,  25,  28,  12,  -4,   0,
    0,  -6,   9,  26,  27,  14,  -6,   0,
    0,  -6,   8,  25,  28,  14,  -5,   0,
    0,  -6,   7,  24,  28,  16,  -5,   0,
    0,  -6,   7,  23,  29,  16,  -5,   0,
    0,  -6,   6,  22,  29,  18,  -5,   0,
    0,  -6,   5,  21,  29,  19,  -4,   0
};
static const int kChromaRPR1[128] = {
   12,  40,  12,   0,
   11,  40,  13,   0,
   10,  40,  15,  -1,
    9,  40,  16,  -1,
    8,  40,  17,  -1,
    8,  39,  18,  -1,
    7,  39,  19,  -1,
    6,  38,  21,  -1,
    5,  38,  22,  -1,
    4,  38,  23,  -1,
    4,  37,  24,  -1,
    3,  36,  25,   0,
    3,  35,  26,   0,
    2,  34,  28,   0,
    2,  33,  29,   0,
    1,  33,  30,   0,
    1,  31,  31,   1,
    0,  30,  33,   1,
    0,  29,  33,   2,
    0,  28,  34,   2,
    0,  26,  35,   3,
    0,  25,  36,   3,
   -1,  24,  37,   4,
   -1,  23,  38,   4,
   -1,  22,  38,   5,
   -1,  21,  38,   6,
   -1,  19,  39,   7,
   -1,  18,  39,   8,
   -1,  17,  40,   8,
   -1,  16,  40,   9,
   -1,  15,  40,  10,
    0,  13,  40,  11
};
static const int kChromaRPR2[128] = {
   17,  30,  17,   0,
   17,  30,  18,  -1,
   16,  30,  18,   0,
   16,  30,  18,   0,
   15,  30,  18,   1,
   14,  30,  18,   2,
   13,  29,  19,   3,
   13,  29,  19,   3,
   12,  29,  20,   3,
   11,  28,  21,   4,
   10,  28,  22,   4,
   10,  27,  22,   5,
    9,  27,  23,   5,
    9,  26,  24,   5,
    8,  26,  24,   6,
    7,  26,  25,   6,
    7,  25,  25,   7,
    6,  25,  26,   7,
    6,  24,  26,   8,
    5,  24,  26,   9,
    5,  23,  27,   9,
    5,  22,  27,  10,
    4,  22,  28,  10,
    4,  21,  28,  11,
    3,  20,  29,  12,
    3,  19,  29,  13,
    3,  19,  29,  13,
    2,  18,  30,  14,
    1,  18,  30,  15,
    0,  18,  30,  16,
    0,  18,  30,  16,
   -1,  18,  30,  17
};

// RPR's filter taps by filter index (InterpolationFilter::filterHor, filterVer).
// Luma: 0 the regular filter (its half-sample variant with hpel), 2 the 4x4
// one, 3 and 4 the RPR ones, 5 and 6 affine's RPR ones. Chroma: 3 and 4 the RPR
// ones, else the regular one. The phase is in 16ths for luma, 32nds for chroma.
int rprCoeff( int comp, int filter, int phase, bool hpel, int k )
{
  if( comp == 0 )
  {
    if( filter == 0 )
    {
      return ( hpel && phase == 8 ) ? kLumaHpel[k] : kLuma[phase * 8 + k];
    }
    if( filter == 2 )
    {
      return kLuma4x4[phase * 8 + k];
    }
    if( filter == 3 )
    {
      return kLumaRPR1[phase * 8 + k];
    }
    if( filter == 4 )
    {
      return kLumaRPR2[phase * 8 + k];
    }
    return filter == 5 ? kAffineLumaRPR1[phase * 8 + k] : kAffineLumaRPR2[phase * 8 + k];
  }
  if( filter == 3 )
  {
    return kChromaRPR1[phase * 4 + k];
  }
  if( filter == 4 )
  {
    return kChromaRPR2[phase * 4 + k];
  }
  return kChroma[phase * 4 + k];
}

// xPredInterBlkRPR: predicts sample (col, row) of the block at (bx, by) (in
// component comp's samples) from the scaled reference r (plane p) with motion
// mv (1/16 luma sample, not clipped). The sample's position in the reference
// follows from the scaling ratio and the scaling windows. The 2-D filter is
// that of the filter index (2 for affine), or a stronger one past 1.25x and
// 1.75x; each tap's sample is clamped into the reference (its border
// extension), and the half-sample filter applies only at 1x. With bi, returns
// the 14-bit intermediate, else the final value.
int rprSample( RefPlanes r, int16_t* p, int comp, int bx, int by, int col, int row, int2 mv, bool bi, bool hpel, int filterIndex )
{
  const int sx     = comp == 0 ? 0 : pc.scaleX;
  const int sy     = comp == 0 ? 0 : pc.scaleY;
  const int refW   = int( r.size & 0xffffu ) >> sx;
  const int refH   = int( r.size >> 16 ) >> sy;
  const int stride = comp == 0 ? int( r.strides & 0xffffu ) : int( r.strides >> 16 );
  const int one    = 1 << 14;   // SCALE_1X
  const int thr1   = one * 5 / 4;
  const int thr2   = one * 7 / 4;
  int       xFilter = r.ratioX > thr2 ? 4 : ( r.ratioX > thr1 ? 3 : filterIndex );
  int       yFilter = r.ratioY > thr2 ? 4 : ( r.ratioY > thr1 ? 3 : filterIndex );
  if( comp == 0 && filterIndex == 2 )
  {
    xFilter += r.ratioX > thr1 ? 2 : 0;
    yFilter += r.ratioY > thr1 ? 2 : 0;
  }

  // the block's position in the reference, in 1 / 1024 samples
  const int posX = ( ( bx << sx ) - pc.curWinLeft ) >> sx;
  const int posY = ( ( by << sy ) - pc.curWinTop ) >> sy;
  const int addX = comp == 0 ? 0 : ( 1 - pc.horColloc ) * 8 * ( r.ratioX - one );
  const int addY = comp == 0 ? 0 : ( 1 - pc.verColloc ) * 8 * ( r.ratioY - one );
  int64_t   x0   = ( ( int64_t( posX ) << ( 4 + sx ) ) + int64_t( mv.x ) ) * int64_t( r.ratioX ) + int64_t( addX );
  int64_t   y0   = ( ( int64_t( posY ) << ( 4 + sy ) ) + int64_t( mv.y ) ) * int64_t( r.ratioY ) + int64_t( addY );
  const int64_t rx = ( ( x0 < 0 ? -x0 : x0 ) + ( int64_t( 1 ) << ( 7 + sx ) ) ) >> ( 8 + sx );
  const int64_t ry = ( ( y0 < 0 ? -y0 : y0 ) + ( int64_t( 1 ) << ( 7 + sy ) ) ) >> ( 8 + sy );
  x0 = ( x0 < 0 ? -rx : rx ) + ( int64_t( r.winLeft ) << ( 10 - sx ) );
  y0 = ( y0 < 0 ? -ry : ry ) + ( int64_t( r.winTop ) << ( 10 - sy ) );

  // The sample's integer position and phase
  const int px    = int( x0 ) + col * ( ( r.ratioX + 8 ) >> 4 ) + ( 1 << ( 5 - sx ) );
  const int py    = int( y0 ) + row * ( ( r.ratioY + 8 ) >> 4 ) + ( 1 << ( 5 - sy ) );
  const int xInt  = px >> 10;
  const int yInt  = py >> 10;
  const int xFrac = ( px >> ( 6 - sx ) ) & ( ( 16 << sx ) - 1 );
  const int yFrac = ( py >> ( 6 - sy ) ) & ( ( 16 << sy ) - 1 );
  const int xPhase = comp == 0 ? xFrac : xFrac << ( 1 - sx );
  const int yPhase = comp == 0 ? yFrac : yFrac << ( 1 - sy );
  const bool hpelX = hpel && r.ratioX == one;
  const bool hpelY = hpel && r.ratioY == one;

  const int taps     = comp == 0 ? 8 : 4;
  const int half     = taps / 2 - 1;
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int shift1   = 6 - headRoom;
  const int offset1  = -IF_INTERNAL_OFFS * ( 1 << shift1 );
  int       sum      = 0;
  for( int t = 0; t < taps; t++ )
  {
    const int yy = clamp( yInt - half + t, 0, refH - 1 );
    int       h  = 0;
    for( int k = 0; k < taps; k++ )
    {
      h += rprCoeff( comp, xFilter, xPhase, hpelX, k ) * int( p[yy * stride + clamp( xInt - half + k, 0, refW - 1 )] );
    }
    sum += rprCoeff( comp, yFilter, yPhase, hpelY, t ) * ( ( h + offset1 ) >> shift1 );
  }
  const int maxVal = ( 1 << pc.bitDepth ) - 1;
  return bi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
}

// Predicts the thread's samples t = tid + THREADS k of a w x h tile from a
// scaled reference (rprSample). The tile lies (ox, oy) into the block at
// (bx, by) that xPredInterBlkRPR predicts as a whole (the CU, or SbTMVP's
// joined sub-PUs), since the sample positions step from that block's origin.
void rprBlock( int tid, int slot, int comp, int bx, int by, int ox, int oy, int w, int h, int2 mv, bool bi, bool hpel, inout int out[PER_THREAD] )
{
  const RefPlanes r     = pc.refs[slot];
  int16_t*        p     = comp == 0 ? r.y : ( comp == 1 ? r.cb : r.cr );
  const int       log2W = firstbithigh( uint( w ) );
  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int t = tid + THREADS * k;
    if( t < w * h )
    {
      out[k] = rprSample( r, p, comp, bx, by, ox + ( t & ( w - 1 ) ), oy + ( t >> log2W ), mv, bi, hpel, 0 );
    }
  }
}

// Shared by the workgroup: the reference samples that a block's filter reads,
// the horizontal pass of a 2-D filter, and DMVR's bilinear predictions, SADs
// and result. The filters' arrays (the reference samples s_ref and the
// horizontal pass s_tmp) and DMVR's bilinear predictions (s_bil0, s_bil1) are
// never in use at the same time, so they share one pool of 16-bit values
// (each fits in 16 bits); this lets more workgroups fit on a compute unit.
#define REF_AT 0
#define TMP_AT ( ( 16 + 7 ) * ( 16 + 7 ) )
#define BIL0_AT 0
#define BIL1_AT ( 20 * 20 )
// tile16's use of the pool: the luma windows of the two lists (rows of YS
// samples), then a work area for DMVR's bilinear predictions or the filters'
// horizontal passes. Once luma is done, the chroma windows take the luma
// windows' place.
#define YW 23                             // a luma window's side: 16 + 7
#define YS 23                             // stored row length: a window row from its first sample
#define CWD 11                            // a chroma window's side: 8 + 3
#define CS 12                             // stored row length: a window row from its first sample (11 used)
#define WY_AT( l ) ( ( l ) * YW * YS )
#define WORK_AT ( 2 * YW * YS )
#define WC_AT( b ) ( ( b ) * CWD * CS )   // b: 2 * list + chroma component - 1
#define CH_AT WC_AT( 4 )                  // the chroma filters' horizontal passes, after the chroma windows (4 x 88)
// BDOF's predictions (s_bdof) lie past what the pool holds while they are in use: the generic path's filters (s_ref,
// s_tmp: 897 values) and tile16's chroma (windows and horizontal passes: 880 values)
#define BDOF_AT ( ( 16 + 7 ) * ( 16 + 7 ) + ( 16 + 7 ) * 16 )
// (the affine variant holds the reference samples of up to 16 subblocks, 9 x 9 each)
#if AFFINE
groupshared int16_t s_pool[16 * 81];
#else
groupshared int16_t s_pool[WORK_AT + 2 * 20 * 20];
#endif
#define s_ref( i ) int( s_pool[REF_AT + ( i )] )
#define s_tmp( i ) int( s_pool[TMP_AT + ( i )] )
#define s_bil0( i ) int( s_pool[BIL0_AT + ( i )] )
#define s_bil1( i ) int( s_pool[BIL1_AT + ( i )] )
// DMVR's SADs, and BDOF's sums (s_bsum) once DMVR is done with them
#if BDOF
groupshared int  s_ints[16 * 5];
#else
groupshared int  s_ints[25];
#endif
#define s_sad s_ints
groupshared int2 s_delta;

int log2Int( int v )
{
  return firstbithigh( uint( v ) );
}

int16_t* refPlane( int slot, int comp )
{
  RefPlanes r = pc.refs[slot];
  return comp == 0 ? r.y : ( comp == 1 ? r.cb : r.cr );
}

// The rectangle (luma, inclusive) that the job's block reads from and clips its
// motion to: the picture, or its subpicture when that is treated as a picture
// (vvdec's subpicture reference buffers, clipMvInSubpic).
static int4 g_bounds;

void setBounds( int cuX, int cuY )
{
  g_bounds = int4( 0, 0, pc.picWidth - 1, pc.picHeight - 1 );
  if( pc.subpicOn != 0 )
  {
    uint16_t* r = pc.subpics + ( ( cuY >> pc.ctuLog2 ) * pc.widthInCtus + ( cuX >> pc.ctuLog2 ) ) * 4;
    g_bounds    = int4( int( r[0] ), int( r[1] ), int( r[2] ), int( r[3] ) );
  }
}

// g_bounds in component comp's samples
int4 boundsOf( int comp )
{
  const int sx = comp == 0 ? 0 : pc.scaleX;
  const int sy = comp == 0 ? 0 : pc.scaleY;
  return int4( g_bounds.x >> sx, g_bounds.y >> sy, ( ( g_bounds.z + 1 ) >> sx ) - 1, ( ( g_bounds.w + 1 ) >> sy ) - 1 );
}

// A reference sample, with its coordinates clamped into g_bounds: vvdec's border extension (of the picture, or of
// the subpicture's copy). With wrap, the wraparound copy's sample (extendPicBorderWrap): within the wraparound offset
// beyond the left and right edges, the margins hold the samples that far across the picture; further out, the edge's.
int refAt( int16_t* p, int comp, int x, int y, bool wrap )
{
  const int4 b = boundsOf( comp );
  if( wrap )
  {
    const int w      = comp == 0 ? pc.picWidth : pc.picWidth >> pc.scaleX;
    const int offset = comp == 0 ? pc.wrapOffset : pc.wrapOffset >> pc.scaleX;
    x = x < 0 ? ( -x - 1 < offset ? x + offset : 0 ) : ( x >= w ? ( x - w < offset ? x - offset : w - 1 ) : x );
  }
  return int( p[clamp( y, b.y, b.w ) * ( comp == 0 ? pc.stride0 : pc.stride12 ) + clamp( x, b.x, b.z )] );
}

// refAt, but with a window (DMVR's padded prefetch) the coordinates are first clamped into the window
int refSample( int16_t* p, int comp, int x, int y, bool windowed, int4 window, bool wrap )
{
  if( windowed )
  {
    x = clamp( x, window.x, window.x + window.z - 1 );
    y = clamp( y, window.y, window.y + window.w - 1 );
  }
  return refAt( p, comp, x, y, wrap );
}

// wrapClipMv: motion that points sideways past the picture's extension moves
// by the wraparound offset (shifted tells whether it did) and is then clipped,
// for the block at (x, y) of width w.
int2 wrapClipMv( int2 mv, int x, int y, int w, out bool shifted )
{
  const int horMax = ( pc.picWidth + pc.ctuSize - w + 8 - x - 1 ) * 16;
  const int horMin = ( -pc.ctuSize - 8 - x + 1 ) * 16;
  const int verMax = ( pc.picHeight + 8 - y - 1 ) * 16;
  const int verMin = ( -pc.ctuSize - 8 - y + 1 ) * 16;
  int       mvX    = mv.x;
  shifted          = false;
  if( mvX > horMax )
  {
    mvX     = min( horMax, max( horMin, mvX - pc.wrapOffset * 16 ) );
    shifted = true;
  }
  if( mvX < horMin )
  {
    mvX     = min( horMax, max( horMin, mvX + pc.wrapOffset * 16 ) );
    shifted = true;
  }
  return int2( mvX, min( verMax, max( verMin, mv.y ) ) );
}

// clipMvInPic, clipMvInSubpic: clips the motion so that the block at (x, y), of
// width w, stays near g_bounds (the picture, or its subpicture); with the SPS's
// wraparound, wrapClipMv instead.
int2 clipMv( int2 mv, int x, int y, int w )
{
  if( pc.wrapSps != 0 )
  {
    bool shifted;
    return wrapClipMv( mv, x, y, w, shifted );
  }
  const int horMax = ( g_bounds.z + 1 + 8 - x - 1 ) * 16;
  const int horMin = ( -pc.ctuSize - 8 - ( x - g_bounds.x ) + 1 ) * 16;
  const int verMax = ( g_bounds.w + 1 + 8 - y - 1 ) * 16;
  const int verMin = ( -pc.ctuSize - 8 - ( y - g_bounds.y ) + 1 ) * 16;
  return int2( min( horMax, max( horMin, mv.x ) ), min( verMax, max( verMin, mv.y ) ) );
}

int coeff( int comp, int frac, bool hpel, int k )
{
  if( comp == 0 )
  {
    return ( hpel && frac == 8 ) ? kLumaHpel[k] : kLuma[frac * 8 + k];
  }
  return kChroma[frac * 4 + k];
}

// xPredInterBlk for a w x h block of a component with an N-tap filter; the
// thread's samples s = tid + THREADS * k go to out[k]. Sample (i, j) filters
// the reference around (baseX + i, baseY + j) with mv's fractional parts. With
// bi, the result is the 14-bit intermediate for averaging, else the final,
// clipped value. The whole workgroup calls it together: the reference samples
// that the filter reads go to shared memory first, each clamped (and
// windowed) once.
void predictBlock<let N : int>( int tid, int slot, int comp, int bx, int by, int w, int h, int2 mv, bool bi, bool hpel, bool windowed,
                                int4 window, int2 base, bool wrap, inout int out[PER_THREAD] )
{
  int16_t*  ref      = refPlane( slot, comp );
  // The motion in the component's samples (1/16 luma sample is 1 / ( 16 <<
  // scale ) of a chroma sample), and the chroma filter's phase in 32nds
  // (InterpolationFilter's frac << ( 1 - scale )).
  const int sx       = comp == 0 ? 0 : pc.scaleX;
  const int sy       = comp == 0 ? 0 : pc.scaleY;
  const int xFrac    = mv.x & ( ( 16 << sx ) - 1 );
  const int yFrac    = mv.y & ( ( 16 << sy ) - 1 );
  const int xPhase   = comp == 0 ? xFrac : xFrac << ( 1 - sx );
  const int yPhase   = comp == 0 ? yFrac : yFrac << ( 1 - sy );
  const int baseX    = windowed ? base.x : bx + ( mv.x >> ( 4 + sx ) );
  const int baseY    = windowed ? base.y : by + ( mv.y >> ( 4 + sy ) );
  const int half     = N / 2 - 1;
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int log2W    = log2Int( w );

  int16_t cx[N], cy[N];   // 16-bit taps, so the compiler uses full-rate 24-bit multiply-adds
  [unroll]
  for( int k = 0; k < N; k++ )
  {
    cx[k] = int16_t( coeff( comp, xPhase, hpel, k ) );
    cy[k] = int16_t( coeff( comp, yPhase, hpel, k ) );
  }

  // The filter reads rw x rh samples from (baseX - ox, baseY - oy).
  const int ox = xFrac != 0 ? half : 0;
  const int oy = yFrac != 0 ? half : 0;
  const int rw = w + ( xFrac != 0 ? N - 1 : 0 );
  const int rh = h + ( yFrac != 0 ? N - 1 : 0 );
  const int4 b = boundsOf( comp );
  if( !windowed && baseX - ox >= b.x && baseX - ox + rw <= b.z + 1 && baseY - oy >= b.y && baseY - oy + rh <= b.w + 1 )
  {
    // Inside the picture (most blocks): copy the samples without clamping.
    const int stride = comp == 0 ? pc.stride0 : pc.stride12;
    int16_t*  src    = ref + ( baseY - oy ) * stride + baseX - ox;
    // The thread's sample (c, r) is src[at]; each step advances THREADS samples.
    const int stepR  = THREADS / rw;
    const int stepC  = THREADS - stepR * rw;
    const int stepAt = stepR * stride + stepC;
    const int r      = tid / rw;
    int       c      = tid - r * rw;
    int       at     = r * stride + c;
    for( int s = tid; s < rw * rh; s += THREADS )
    {
      s_pool[REF_AT + s] = src[at];
      c  += stepC;
      at += stepAt;
      if( c >= rw )
      {
        c  -= rw;
        at += stride - rw;
      }
    }
  }
  else
  {
    for( int s = tid; s < rw * rh; s += THREADS )
    {
      const int r = s / rw;
      const int c = s - r * rw;
      s_pool[REF_AT + s] = int16_t( refSample( ref, comp, baseX - ox + c, baseY - oy + r, windowed, window, wrap ) );
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if( xFrac != 0 && yFrac != 0 )
  {
    // The horizontal pass, over the rows that the vertical pass reads.
    const int shift1  = 6 - headRoom;
    const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
    for( int s = tid; s < rh * w; s += THREADS )
    {
      const int r   = s >> log2W;
      const int i   = s & ( w - 1 );
      int       sum = 0;
      const int at  = s + r * ( N - 1 );   // r * rw + i, rw = w + N - 1
      [unroll]
      for( int k = 0; k < N; k++ )
      {
        sum += int( cx[k] ) * s_ref( at + k );
      }
      s_pool[TMP_AT + s] = int16_t( ( sum + offset1 ) >> shift1 );
    }
    GroupMemoryBarrierWithGroupSync();
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int s = tid + THREADS * k;
      if( s < w * h )
      {
        const int j   = s >> log2W;
        const int i   = s & ( w - 1 );
        int       sum = 0;
        [unroll]
        for( int t = 0; t < N; t++ )
        {
          sum += int( cy[t] ) * s_tmp( s + ( t << log2W ) );   // ( j + t ) * w + i
        }
        out[k] = bi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
      }
    }
  }
  else
  {
    const int shiftV = bi ? 6 - headRoom : 6;
    const int offset = bi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int s = tid + THREADS * k;
      if( s < w * h )
      {
        const int j = s >> log2W;
        const int i = s & ( w - 1 );
        if( xFrac == 0 && yFrac == 0 )
        {
          const int v = s_ref( s );   // rw = w
          out[k]      = bi ? v * ( 1 << headRoom ) - IF_INTERNAL_OFFS : v;
        }
        else
        {
          int sum = 0;
          if( yFrac != 0 )
          {
            [unroll]
            for( int t = 0; t < N; t++ )
            {
              sum += int( cy[t] ) * s_ref( s + ( t << log2W ) );   // ( j + t ) * rw + i, rw = w
            }
          }
          else
          {
            const int at = s + j * ( N - 1 );   // j * rw + i, rw = w + N - 1
            [unroll]
            for( int t = 0; t < N; t++ )
            {
              sum += int( cx[t] ) * s_ref( at + t );
            }
          }
          const int v = ( sum + offset ) >> shiftV;
          out[k]      = bi ? v : clip3( 0, maxVal, v );
        }
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();   // before the next block writes the shared memory
}

// --------------------------------------------------------------------------------------------------------------------
// DMVR (xProcessDMVR): the bilinear predictions around the merge motion, the SADs of the 5x5 integer offsets
// (over every other row), and the error surface's sub-sample step
// --------------------------------------------------------------------------------------------------------------------

// xinitMC's bilinear prediction of the CU from mvStart (the clipped merge motion minus 2 samples), at sample (u, v)
// relative to the CU
int bilinearAt( int slot, int cuX, int cuY, int2 mvStart, int u, int v, bool wrap )
{
  int16_t*  ref   = refPlane( slot, 0 );
  const int xFrac = mvStart.x & 15;
  const int yFrac = mvStart.y & 15;
  const int x     = cuX + ( mvStart.x >> 4 ) + u;
  const int y     = cuY + ( mvStart.y >> 4 ) + v;
  const int shift = 4 - ( 10 - pc.bitDepth );
  if( xFrac == 0 && yFrac == 0 )
  {
    return refAt( ref, 0, x, y, wrap ) * ( 1 << ( 10 - pc.bitDepth ) );
  }
  if( yFrac == 0 )
  {
    return ( kBilinear[xFrac * 2] * refAt( ref, 0, x, y, wrap ) + kBilinear[xFrac * 2 + 1] * refAt( ref, 0, x + 1, y, wrap ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  }
  if( xFrac == 0 )
  {
    return ( kBilinear[yFrac * 2] * refAt( ref, 0, x, y, wrap ) + kBilinear[yFrac * 2 + 1] * refAt( ref, 0, x, y + 1, wrap ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  }
  const int t0 = ( kBilinear[xFrac * 2] * refAt( ref, 0, x, y, wrap ) + kBilinear[xFrac * 2 + 1] * refAt( ref, 0, x + 1, y, wrap ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  const int t1 = ( kBilinear[xFrac * 2] * refAt( ref, 0, x, y + 1, wrap ) + kBilinear[xFrac * 2 + 1] * refAt( ref, 0, x + 1, y + 1, wrap ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  return ( kBilinear[yFrac * 2] * t0 + kBilinear[yFrac * 2 + 1] * t1 + 8 ) >> 4;
}

// bilinearAt for reads that stay inside the picture: p points at sample (u, v),
// and nothing is clamped.
int bilinearInside( int16_t* p, int xFrac, int yFrac )
{
  const int      shift = 4 - ( 10 - pc.bitDepth );
  if( xFrac == 0 && yFrac == 0 )
  {
    return int( p[0] ) * ( 1 << ( 10 - pc.bitDepth ) );
  }
  if( yFrac == 0 )
  {
    return ( kBilinear[xFrac * 2] * int( p[0] ) + kBilinear[xFrac * 2 + 1] * int( p[1] ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  }
  if( xFrac == 0 )
  {
    return ( kBilinear[yFrac * 2] * int( p[0] ) + kBilinear[yFrac * 2 + 1] * int( p[pc.stride0] ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  }
  const int t0 = ( kBilinear[xFrac * 2] * int( p[0] ) + kBilinear[xFrac * 2 + 1] * int( p[1] ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  const int t1 = ( kBilinear[xFrac * 2] * int( p[pc.stride0] ) + kBilinear[xFrac * 2 + 1] * int( p[pc.stride0 + 1] ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  return ( kBilinear[yFrac * 2] * t0 + kBilinear[yFrac * 2 + 1] * t1 + 8 ) >> 4;
}

// Row j of the SAD between L0 moved by (hor, ver) and L1 moved by (-hor, -ver).
int dmvrSadRow( int dx, int j, int hor, int ver )
{
  int sum = 0;
  for( int i = 0; i < dx; i++ )
  {
    sum += abs( s_bil0( ( 2 + ver + j ) * 20 + 2 + hor + i ) - s_bil1( ( 2 - ver + j ) * 20 + 2 - hor + i ) );
  }
  return sum;
}

int divForMaxq7( int n, int d )
{
  int sign = 0;
  if( n < 0 )
  {
    sign = 1;
    n    = -n;
  }
  int q = 0;
  d     = d << 3;
  if( n >= d )
  {
    n -= d;
    q++;
  }
  q = q << 1;
  d = d >> 1;
  if( n >= d )
  {
    n -= d;
    q++;
  }
  q = q << 1;
  if( n >= ( d >> 1 ) )
  {
    q++;
  }
  return sign != 0 ? -q : q;
}

// xSubPelErrorSrfc along one axis: s0 is the best cost, sm the one before it, sp the one after it.
int subPel( int s0, int sm, int sp, int current )
{
  const int denominator = sm + sp - ( s0 << 1 );
  if( denominator == 0 )
  {
    return current;
  }
  if( sm != s0 && sp != s0 )
  {
    return divForMaxq7( ( sm - sp ) * 16, denominator );
  }
  return sm == s0 ? -8 : 8;
}

// The subblock's refinement (in 1/16 luma samples) from the SADs; the centre's cost is reduced by a quarter.
int2 dmvrDelta( int dx, int dy )
{
  int minCost = s_sad[12];
  if( minCost < dx * dy )
  {
    return int2( 0, 0 );
  }
  int bestH = 0, bestV = 0;
  for( int ver = -2; ver <= 2; ver++ )
  {
    for( int hor = -2; hor <= 2; hor++ )
    {
      const int cost = s_sad[( ver + 2 ) * 5 + hor + 2];
      if( cost < minCost )
      {
        minCost = cost;
        bestH   = hor;
        bestV   = ver;
      }
    }
  }
  int2 delta = int2( bestH * 16, bestV * 16 );
  if( abs( delta.x ) != 32 && abs( delta.y ) != 32 )
  {
    const int c = ( bestV + 2 ) * 5 + bestH + 2;
    delta.x += subPel( s_sad[c], s_sad[c - 1], s_sad[c + 1], 0 );
    delta.y += subPel( s_sad[c], s_sad[c - 5], s_sad[c + 5], 0 );
  }
  return delta;
}

int2 clipStorage( int2 mv )
{
  return int2( clamp( mv.x, -( 1 << 17 ), ( 1 << 17 ) - 1 ), clamp( mv.y, -( 1 << 17 ), ( 1 << 17 ) - 1 ) );
}

#if BDOF
// --------------------------------------------------------------------------------------------------------------------
// BDOF, bi-directional optical flow (vvdec's applyBiOptFlow, BiOptFlowCore, gradFilterCore). s_bdof holds each list's
// 14-bit luma prediction of the tile plus a one-sample border, which comes from the reference's integer samples
// nearest the motion (xPredInterBlk). Each 4x4 block's motion offset (vx, vy) follows from the gradients and the
// differences over the 6 x 6 samples around it (where those fall in the border, the nearest samples and gradients
// inside the tile stand in). The two predictions are then averaged with the offset's correction.
// --------------------------------------------------------------------------------------------------------------------
#define BS 18                                     // a list's row length in s_bdof: the tile (16 at most) plus border
#define s_bdof( i ) s_pool[BDOF_AT + ( i )]       // (2 * BS * BS values)
#define s_bsum s_ints                             // per 4x4: sums of |gx|, |gy|, di*sgn(gx), di*sgn(gy), gx*sgn(gy)
groupshared int     s_minCost;                    // DMVR's best cost (BDOF applies only where it is 2 w h or more)

int bdofP( int list, int x, int y )   // (x, y): -1 .. w, -1 .. h
{
  return int( s_bdof( list * BS * BS + ( y + 1 ) * BS + x + 1 ) );
}

// The gradients at (x, y) inside the tile, from the samples around it (including the border's).
int2 bdofGrad( int list, int x, int y )
{
  return int2( ( bdofP( list, x + 1, y ) >> 6 ) - ( bdofP( list, x - 1, y ) >> 6 ), ( bdofP( list, x, y + 1 ) >> 6 ) - ( bdofP( list, x, y - 1 ) >> 6 ) );
}

// The place (bx, by) of the border's sample r around a w x h tile: the rows above and below, then the columns left
// and right (2 (w + 2) + 2 h samples).
int2 bdofBorderAt( int r, int w, int h )
{
  if( r < w + 2 )
  {
    return int2( r - 1, -1 );
  }
  if( r < 2 * ( w + 2 ) )
  {
    return int2( r - ( w + 2 ) - 1, h );
  }
  if( r < 2 * ( w + 2 ) + h )
  {
    return int2( -1, r - 2 * ( w + 2 ) );
  }
  return int2( w, r - 2 * ( w + 2 ) - h );
}

// A border sample of s_bdof from the reference's sample v: at 14 bits.
int16_t bdofBorderValue( int v )
{
  return int16_t( v * ( 1 << max( 2, 14 - pc.bitDepth ) ) - IF_INTERNAL_OFFS );
}

// Stores a list's prediction in s_bdof: the thread's samples, plus the border from the reference's samples nearest
// the motion (fractions below 8 round down), at 14 bits; (baseX, baseY) is the integer position of the prediction.
void bdofStore( int tid, int list, int w, int h, int pred[PER_THREAD], int slot, int baseX, int baseY, int2 mv, bool windowed, int4 window, bool wrap )
{
  const int log2W = log2Int( w );
  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int t = tid + THREADS * k;
    if( t < w * h )
    {
      s_bdof( list * BS * BS + ( ( t >> log2W ) + 1 ) * BS + ( t & ( w - 1 ) ) + 1 ) = int16_t( pred[k] );
    }
  }
  const int ring = 2 * ( w + 2 ) + 2 * h;
  const int xOff = ( mv.x & 15 ) < 8 ? 1 : 0;
  const int yOff = ( mv.y & 15 ) < 8 ? 1 : 0;
  int16_t*  ref  = refPlane( slot, 0 );
  for( int r = tid; r < ring; r += THREADS )
  {
    const int2 b = bdofBorderAt( r, w, h );
    const int  v = refSample( ref, 0, baseX + b.x + 1 - xOff, baseY + b.y + 1 - yOff, windowed, window, wrap );
    s_bdof( list * BS * BS + ( b.y + 1 ) * BS + b.x + 1 ) = bdofBorderValue( v );
  }
}

// Adds part q (q = 0 .. 3, nine samples) of the sums over the 6 x 6 window of the tile's 4x4 block sb into s_bsum,
// which starts zeroed.
void bdofAddSums( int sb, int q, int w, int h )
{
  const int nsx   = w >> 2;
  const int sx    = sb % nsx;
  const int sy    = sb / nsx;
  int       absGX = 0, absGY = 0, dIX = 0, dIY = 0, signGX = 0;
  for( int p = q * 9; p < q * 9 + 9; p++ )
  {
    const int  cx = clamp( 4 * sx - 1 + p % 6, 0, w - 1 );
    const int  cy = clamp( 4 * sy - 1 + p / 6, 0, h - 1 );
    const int2 g0 = bdofGrad( 0, cx, cy );
    const int2 g1 = bdofGrad( 1, cx, cy );
    const int  gx = ( g0.x + g1.x ) >> 1;
    const int  gy = ( g0.y + g1.y ) >> 1;
    const int  di = ( bdofP( 1, cx, cy ) >> 4 ) - ( bdofP( 0, cx, cy ) >> 4 );
    absGX  += abs( gx );
    absGY  += abs( gy );
    dIX    += gx < 0 ? -di : ( gx == 0 ? 0 : di );
    dIY    += gy < 0 ? -di : ( gy == 0 ? 0 : di );
    signGX += gy < 0 ? -gx : ( gy == 0 ? 0 : gx );
  }
  InterlockedAdd( s_bsum[sb * 5], absGX );
  InterlockedAdd( s_bsum[sb * 5 + 1], absGY );
  InterlockedAdd( s_bsum[sb * 5 + 2], dIX );
  InterlockedAdd( s_bsum[sb * 5 + 3], dIY );
  InterlockedAdd( s_bsum[sb * 5 + 4], signGX );
}

// The motion offset (vx, vy) of the tile's 4x4 block sb, from its sums in s_bsum.
int2 bdofOffset( int sb )
{
  const int absGX  = s_bsum[sb * 5];
  const int absGY  = s_bsum[sb * 5 + 1];
  const int dIX    = s_bsum[sb * 5 + 2];
  const int dIY    = s_bsum[sb * 5 + 3];
  const int signGX = s_bsum[sb * 5 + 4];
  // rightShiftMSB: shift right by the position of the denominator's highest bit
  const int vx   = absGX == 0 ? 0 : clamp( ( dIX * 4 ) >> firstbithigh( uint( absGX ) ), -15, 15 );
  const int data = ( ( vx * ( signGX >> 12 ) ) * 4096 + vx * ( signGX & 4095 ) ) >> 1;
  const int vy   = absGY == 0 ? 0 : clamp( ( dIY * 4 - data ) >> firstbithigh( uint( absGY ) ), -15, 15 );
  return int2( vx, vy );
}

// The tile's sample (i, j): the average of the two predictions with the correction of its block's offset v.
int bdofSample( int i, int j, int2 v )
{
  const int  shiftNum = 15 - pc.bitDepth;
  const int  offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
  const int  maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int2 g0       = bdofGrad( 0, i, j );
  const int2 g1       = bdofGrad( 1, i, j );
  const int  b        = v.x * ( g0.x - g1.x ) + v.y * ( g0.y - g1.y );
  return clip3( 0, maxVal, int( int16_t( ( bdofP( 0, i, j ) + bdofP( 1, i, j ) + b + offset ) >> shiftNum ) ) );
}

// After both lists' bdofStore and a barrier: corrects and averages the thread's samples of the tile.
void bdofApply( int tid, int w, int h, out int result[PER_THREAD] )
{
  const int nsx = w >> 2;
  const int nsb = nsx * ( h >> 2 );
  // The sums over each 4x4 block's 6 x 6 window: four threads per 4x4 block, nine samples each.
  if( tid < 4 * nsb )
  {
    bdofAddSums( tid >> 2, tid & 3, w, h );
  }
  GroupMemoryBarrierWithGroupSync();
  // (each thread works out the offsets of its samples' blocks: no barrier to share them)
  const int log2W = log2Int( w );
  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int t = tid + THREADS * k;
    result[k]   = 0;
    if( t < w * h )
    {
      const int i = t & ( w - 1 );
      const int j = t >> log2W;
      result[k]   = bdofSample( i, j, bdofOffset( ( j >> 2 ) * nsx + ( i >> 2 ) ) );
    }
  }
}
#endif

// --------------------------------------------------------------------------------------------------------------------
// tile16: a 16x16 tile whose reference samples all lie inside the picture (nearly every tile). All its global reads
// are issued at once, in aligned loads of 16 and 8 bytes, before the first wait for memory: each list's luma window
// of 23 x 23 samples around its motion (with DMVR, around the merge motion; the search and the refined prediction
// both read inside it), the chroma windows of 11 x 11, and the residuals. DMVR and the filters then work from shared
// memory. (Reading and waiting one sample at a time, a tile waited for memory some 50 times; the shader is
// latency-bound: with a third fewer waves per SIMD it ran a third slower.)
// In the inter_bdof variant, tile16 also applies BDOF: its steps run in the phases between tile16's barriers, so
// that a BDOF tile waits at no more barriers than another tile.
// --------------------------------------------------------------------------------------------------------------------

// Sample (r, c) of the window whose sample (0, 0) is at index at; clamped into the window when DMVR's refinement
// moved the integer position (xPrefetchPad's padding).
int winY( int at, int r, int c, bool clampIt )
{
  if( clampIt )
  {
    r = clamp( r, 0, YW - 1 );
    c = clamp( c, 0, YW - 1 );
  }
  return int( s_pool[at + r * YS + c] );
}

int winC( int at, int r, int c, bool clampIt )
{
  if( clampIt )
  {
    r = clamp( r, 0, CWD - 1 );
    c = clamp( c, 0, CWD - 1 );
  }
  return int( s_pool[at + r * CS + c] );
}

// One SAD row of DMVR's search, from the bilinear predictions at b0 and b1 (rows of 20 samples).
int sadRowAt( int b0, int b1, int j, int hor, int ver )
{
  int sum = 0;
  [unroll]
  for( int i = 0; i < 16; i++ )
  {
    sum += abs( int( s_pool[b0 + ( 2 + ver + j ) * 20 + 2 + hor + i] ) - int( s_pool[b1 + ( 2 - ver + j ) * 20 + 2 - hor + i] ) );
  }
  return sum;
}

// The 4 luma predictions (j, i0 .. i0 + 3) of a list from its window (whose sample (0, 0) is at index at), with
// predictBlock's arithmetic; tmp holds the horizontal pass of a 2-D filter.
int4 lumaPred( int at, int tmp, int j, int i0, int rx0, int ry0, bool clampIt, int xFrac, int yFrac, int16_t cx[8], int16_t cy[8], bool bi )
{
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  int4      out;
  if( xFrac != 0 && yFrac != 0 )
  {
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      int sum = 0;
      [unroll]
      for( int t = 0; t < 8; t++ )
      {
        sum += int( cy[t] ) * int( s_pool[tmp + ( j + t ) * 16 + i0 + o] );
      }
      out[o] = bi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
    }
    return out;
  }
  const int shiftV = bi ? 6 - headRoom : 6;
  const int offset = bi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
  if( xFrac != 0 )
  {
    int w[11];
    [unroll]
    for( int e = 0; e < 11; e++ )
    {
      w[e] = winY( at, ry0 + j, rx0 + i0 + e, clampIt );
    }
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      int sum = 0;
      [unroll]
      for( int k = 0; k < 8; k++ )
      {
        sum += int( cx[k] ) * w[o + k];
      }
      const int v = ( sum + offset ) >> shiftV;
      out[o]      = bi ? v : clip3( 0, maxVal, v );
    }
    return out;
  }
  if( yFrac != 0 )
  {
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      int sum = 0;
      [unroll]
      for( int t = 0; t < 8; t++ )
      {
        sum += int( cy[t] ) * winY( at, ry0 + j + t, rx0 + i0 + o, clampIt );
      }
      const int v = ( sum + offset ) >> shiftV;
      out[o]      = bi ? v : clip3( 0, maxVal, v );
    }
    return out;
  }
  [unroll]
  for( int o = 0; o < 4; o++ )
  {
    const int v = winY( at, ry0 + j, rx0 + i0 + o, clampIt );
    out[o]      = bi ? v * ( 1 << headRoom ) - IF_INTERNAL_OFFS : v;
  }
  return out;
}

// The 2 chroma predictions (j, i0) and (j, i0 + 1) of a list from its window, as lumaPred does with 4 taps; tmp
// holds the horizontal pass, in rows of 8.
int2 chromaPred( int at, int tmp, int j, int i0, int rx0, int ry0, bool clampIt, int xFrac, int yFrac, int16_t cx[4], int16_t cy[4], bool bi )
{
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  int2      out;
  if( xFrac != 0 && yFrac != 0 )
  {
    [unroll]
    for( int o = 0; o < 2; o++ )
    {
      int sum = 0;
      [unroll]
      for( int t = 0; t < 4; t++ )
      {
        sum += int( cy[t] ) * int( s_pool[tmp + ( j + t ) * 8 + i0 + o] );
      }
      out[o] = bi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
    }
    return out;
  }
  const int shiftV = bi ? 6 - headRoom : 6;
  const int offset = bi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
  if( xFrac != 0 )
  {
    int w[5];
    [unroll]
    for( int e = 0; e < 5; e++ )
    {
      w[e] = winC( at, ry0 + j, rx0 + i0 + e, clampIt );
    }
    [unroll]
    for( int o = 0; o < 2; o++ )
    {
      int sum = 0;
      [unroll]
      for( int k = 0; k < 4; k++ )
      {
        sum += int( cx[k] ) * w[o + k];
      }
      const int v = ( sum + offset ) >> shiftV;
      out[o]      = bi ? v : clip3( 0, maxVal, v );
    }
    return out;
  }
  if( yFrac != 0 )
  {
    [unroll]
    for( int o = 0; o < 2; o++ )
    {
      int sum = 0;
      [unroll]
      for( int t = 0; t < 4; t++ )
      {
        sum += int( cy[t] ) * winC( at, ry0 + j + t, rx0 + i0 + o, clampIt );
      }
      const int v = ( sum + offset ) >> shiftV;
      out[o]      = bi ? v : clip3( 0, maxVal, v );
    }
    return out;
  }
  [unroll]
  for( int o = 0; o < 2; o++ )
  {
    const int v = winC( at, ry0 + j, rx0 + i0 + o, clampIt );
    out[o]      = bi ? v * ( 1 << headRoom ) - IF_INTERNAL_OFFS : v;
  }
  return out;
}

int lo16( uint v )
{
  return int( v << 16 ) >> 16;
}

int hi16( uint v )
{
  return int( v ) >> 16;
}

// dmvrDelta from the SADs in s_sad, with the centre's cost (reduced by a quarter) passed in; each thread computes it.
int sadOr( int i, int centre )
{
  return i == 12 ? centre : s_sad[i];
}

int2 dmvrDeltaFrom( int centre )
{
  int minCost = centre;
  int best    = 12;
  for( int i = 0; i < 25; i++ )
  {
    const int cost = sadOr( i, centre );
    if( cost < minCost )
    {
      minCost = cost;
      best    = i;
    }
  }
  const int bestH = best % 5 - 2, bestV = best / 5 - 2;
  int2      delta = int2( bestH * 16, bestV * 16 );
  if( abs( delta.x ) != 32 && abs( delta.y ) != 32 )
  {
    delta.x += subPel( sadOr( best, centre ), sadOr( best - 1, centre ), sadOr( best + 1, centre ), 0 );
    delta.y += subPel( sadOr( best, centre ), sadOr( best - 5, centre ), sadOr( best + 5, centre ), 0 );
  }
  return delta;
}

// DMVR's bilinear prediction (xinitMC, as bilinearInside) of the window sample at s_pool[p] (rows of YS samples)
int bilinearW( int p, int xFrac, int yFrac, int shift )
{
  const int16_t c0x = int16_t( 16 - xFrac ), c1x = int16_t( xFrac ), c0y = int16_t( 16 - yFrac ), c1y = int16_t( yFrac );
  if( xFrac == 0 && yFrac == 0 )
  {
    return int( s_pool[p] ) * ( 1 << ( 10 - pc.bitDepth ) );
  }
  if( yFrac == 0 )
  {
    return ( int( c0x ) * int( s_pool[p] ) + int( c1x ) * int( s_pool[p + 1] ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  }
  if( xFrac == 0 )
  {
    return ( int( c0y ) * int( s_pool[p] ) + int( c1y ) * int( s_pool[p + YS] ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  }
  const int16_t t0 = int16_t( ( int( c0x ) * int( s_pool[p] ) + int( c1x ) * int( s_pool[p + 1] ) + ( 1 << ( shift - 1 ) ) ) >> shift );
  const int16_t t1 = int16_t( ( int( c0x ) * int( s_pool[p + YS] ) + int( c1x ) * int( s_pool[p + YS + 1] ) + ( 1 << ( shift - 1 ) ) ) >> shift );
  return ( int( c0y ) * int( t0 ) + int( c1y ) * int( t1 ) + 8 ) >> 4;
}

#if BDOF
// tile16's BDOF border: the thread's share of the border around both lists' predictions (2 x 68 samples, the k-th at
// ring place tid + THREADS k), from the lists' windows (the window's sample (3, 3) lies at the merge motion's integer
// position), clamped into the window when DMVR's refinement moved the integer position, as the prefetch's padding is.
// Read while the luma windows are there; tile16BdofStore stores it with the predictions once they are gone.
static const int kTileRing = 2 * ( 16 + 2 ) + 2 * 16;
void tile16BdofBorder( int tid, int2 mvs[2], int2 mvR[2], bool clampY[2], out int16_t border[3] )
{
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int r = min( tid + THREADS * k, 2 * kTileRing - 1 );
    // (the list is picked with selects: an array indexed per lane would go to scratch memory)
    const int  s       = r >= kTileRing ? 1 : 0;
    const int2 m       = s == 0 ? mvR[0] : mvR[1];
    const int2 merge   = s == 0 ? mvs[0] : mvs[1];
    const bool clampIt = s == 0 ? clampY[0] : clampY[1];
    const int2 b       = bdofBorderAt( r - kTileRing * s, 16, 16 );
    const int  xOff    = ( m.x & 15 ) < 8 ? 1 : 0;
    const int  yOff    = ( m.y & 15 ) < 8 ? 1 : 0;
    const int  c       = 3 + ( m.x >> 4 ) - ( merge.x >> 4 ) + b.x + 1 - xOff;
    const int  row     = 3 + ( m.y >> 4 ) - ( merge.y >> 4 ) + b.y + 1 - yOff;
    border[k]          = bdofBorderValue( winY( s == 0 ? WY_AT( 0 ) : WY_AT( 1 ), row, c, clampIt ) );
  }
}

// tile16's bdofStore: the thread's four luma predictions (j, i0 .. i0 + 3) of each list into s_bdof, and its share of
// the border (tile16BdofBorder)
void tile16BdofStore( int tid, int j, int i0, int4 predY[2], int16_t border[3] )
{
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      s_bdof( s * BS * BS + ( j + 1 ) * BS + i0 + o + 1 ) = int16_t( predY[s][o] );
    }
  }
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int r = tid + THREADS * k;
    if( r < 2 * kTileRing )
    {
      const int  s = r >= kTileRing ? 1 : 0;
      const int2 b = bdofBorderAt( r - kTileRing * s, 16, 16 );
      s_bdof( s * BS * BS + ( b.y + 1 ) * BS + b.x + 1 ) = border[k];
    }
  }
}

// DMVR's best cost (the centre's reduced by a quarter), from the SADs in s_sad: BDOF applies only from 2 w h on.
int dmvrMinCost( int centre )
{
  int minCost = centre;
  for( int i = 0; i < 25; i++ )
  {
    minCost = min( minCost, sadOr( i, centre ) );
  }
  return minCost;
}
#endif

// Whether tile16 may take the tile: each list's windows must lie inside the picture.
bool tile16Inside( int x, int y, int flags, int2 mv0, int2 mv1 )
{
  bool inside = true;
  [unroll]
  for( int list = 0; list < 2; list++ )
  {
    if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) != 0 )
    {
      const int2 m  = list == 0 ? mv0 : mv1;
      const int  wx = x + ( m.x >> 4 ) - 3, wy = y + ( m.y >> 4 ) - 3;
      const int  cx = ( x >> 1 ) + ( m.x >> 5 ) - 1, cy = ( y >> 1 ) + ( m.y >> 5 ) - 1;
      inside        = inside && wx >= 0 && wx + YW <= pc.picWidth && wy >= 0 && wy + YW <= pc.picHeight && cx >= 0 && cx + CWD <= ( pc.picWidth >> 1 )
               && cy >= 0 && cy + CWD <= ( pc.picHeight >> 1 );
    }
  }
  return inside;
}

void tile16( int tid, int index, uint* job, int x, int y, int flags, int2 mv0, int2 mv1, int ref0, int ref1, bool bi, bool dmvr, bool hpel )
{
  const int bcwW1 = int( ( job[7] >> 16 ) & 15 ) - 2;   // BCW's L1 weight
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int headRoom = max( 2, 14 - pc.bitDepth );

  // The lists' slots: the first list used goes in slot 0; with both lists, L1 goes in slot 1.
  const int  lists = bi ? 2 : 1;
  const bool l0    = ( flags & FLAG_L0 ) != 0;
  int2       mvs[2]  = { l0 ? mv0 : mv1, mv1 };
  int        refs[2] = { l0 ? ref0 : ref1, bi ? ref1 : ( l0 ? ref0 : ref1 ) };
  // The planes, read once for the workgroup (each load below picks its own).
  int16_t*   planeY[2]  = { refPlane( refs[0], 0 ), refPlane( refs[1], 0 ) };
  int16_t*   planeCb[2] = { refPlane( refs[0], 1 ), refPlane( refs[1], 1 ) };
  int16_t*   planeCr[2] = { refPlane( refs[0], 2 ), refPlane( refs[1], 2 ) };
  int        wx[2], wy[2], leadY[2], cxs[2], cys[2], leadC[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    wx[s]    = x + ( mvs[s].x >> 4 ) - 3;
    wy[s]    = y + ( mvs[s].y >> 4 ) - 3;
    leadY[s] = wx[s] & 7;   // samples before the window in its aligned 16 bytes
    cxs[s]   = ( x >> 1 ) + ( mvs[s].x >> 5 ) - 1;
    cys[s]   = ( y >> 1 ) + ( mvs[s].y >> 5 ) - 1;
    leadC[s] = cxs[s] & 3;   // the same for chroma, in aligned 8 bytes
  }

  // All of the tile's reads, none of them in a branch, so that they wait for memory together: after a branch the
  // compiler makes the next read wait for all those before it. Luma windows: 23 rows of 4 x 16 bytes per list (a
  // thread past the last chunk reads the last one again, and stores nothing).
  uint4 yChunk[3];
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int task = min( tid + THREADS * k, 92 * lists - 1 );
    const int s    = task >= 92 ? 1 : 0;
    const int rem  = task - 92 * s;
    // (the slot is picked with selects: an array indexed per lane would go to scratch memory)
    const int row0 = s == 0 ? wy[0] : wy[1];
    const int col0 = s == 0 ? wx[0] - leadY[0] : wx[1] - leadY[1];
    int16_t*  p    = ( s == 0 ? planeY[0] : planeY[1] ) + ( row0 + ( rem >> 2 ) ) * pc.stride0 + col0 + 8 * ( rem & 3 );
    yChunk[k]      = *(uint4*)p;
  }
  // Chroma windows: 11 rows of 4 x 8 bytes per component and list (the same).
  uint2 cChunk[3];
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int task = min( tid + THREADS * k, 88 * lists - 1 );
    const int s    = task >= 88 ? 1 : 0;
    const int rem  = task - 88 * s;
    const int comp = rem >= 44 ? 1 : 0;
    const int rem2 = rem - 44 * comp;
    int16_t*  base = s == 0 ? ( comp == 0 ? planeCb[0] : planeCr[0] ) : ( comp == 0 ? planeCb[1] : planeCr[1] );
    const int row0 = s == 0 ? cys[0] : cys[1];
    const int col0 = s == 0 ? cxs[0] - leadC[0] : cxs[1] - leadC[1];
    int16_t*  p    = base + ( row0 + ( rem2 >> 2 ) ) * pc.stride12 + col0 + 4 * ( rem2 & 3 );
    cChunk[k]      = *(uint2*)p;
  }
  // The residuals that the picture holds where the thread's predictions go, when the block has any: luma (j, i0 ..
  // i0 + 3) and chroma (cj, ci0 .. ci0 + 1) of component cc. Both chroma planes are read at the thread's place when
  // either has a residual (the other is another thread's: no more memory traffic; and the branch is the same for all
  // the threads), and the thread's picked after: a plane picked before reading would have its pointer read per
  // thread, one more round trip.
  const int  j     = tid >> 2;
  const int  i0    = ( tid & 3 ) << 2;
  const int  cc    = tid >> 5;
  const int  cj    = ( tid & 31 ) >> 2;
  const int  ci0   = ( tid & 3 ) << 1;
  const int  atY   = ( y + j ) * pc.stride0 + x + i0;
  const int  atC   = ( ( y >> 1 ) + cj ) * pc.stride12 + ( x >> 1 ) + ci0;
  int16_t*   srcY  = pc.dst0 + atY;
  int16_t*   srcCb = pc.dst1 + atC;
  int16_t*   srcCr = pc.dst2 + atC;
  uint2      resY  = uint2( 0, 0 );
  uint       resCb = 0, resCr = 0;
  if( ( flags & FLAG_RES_Y ) != 0 )
  {
    resY = *(uint2*)srcY;
  }
  if( ( flags & ( ( FLAG_RES_Y << 1 ) | ( FLAG_RES_Y << 2 ) ) ) != 0 )
  {
    resCb = *(uint*)srcCb;
    resCr = *(uint*)srcCr;
  }
  const uint resC = cc == 0 ? resCb : resCr;

  // Store the luma windows in shared memory.
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int task = tid + THREADS * k;
    if( task < 92 * lists )
    {
      const int s   = task >= 92 ? 1 : 0;
      const int rem = task - 92 * s;
      // (the row from the window's first sample: the chunk's samples before it, and past the row's room, are dropped)
      const int   at   = WY_AT( s ) + ( rem >> 2 ) * YS;
      const int   c0   = 8 * ( rem & 3 ) - ( s == 0 ? leadY[0] : leadY[1] );
      const uint4 v    = yChunk[k];
      const uint  w[4] = { v.x, v.y, v.z, v.w };
      [unroll]
      for( int m = 0; m < 8; m++ )
      {
        if( uint( c0 + m ) < uint( YS ) )
        {
          s_pool[at + c0 + m] = int16_t( w[m >> 1] >> ( 16 * ( m & 1 ) ) );
        }
      }
    }
  }
  if( tid < 25 )
  {
    s_sad[tid] = 0;
  }
#if BDOF
  int dmvrCost = 2 * 16 * 16;   // (without DMVR, BDOF applies wherever the tile's flag says)
#endif
  GroupMemoryBarrierWithGroupSync();

  // DMVR, as xProcessDMVR: the centre's SAD first. When it is small enough (83% of VideoQ's subblocks) there is no
  // refinement, and neither the other 24 SADs nor most bilinear predictions are needed.
  int2 mvR[2] = { mvs[0], mvs[1] };
  if( dmvr )
  {
    const int shift = 4 - ( 10 - pc.bitDepth );
    // Rows 2, 4 .. 16 and columns 2 .. 17 of the bilinear predictions (the windows' samples from (3, 3)), 2 per thread
    int part = 0;
    [unroll]
    for( int k = 0; k < 2; k++ )
    {
      const int e  = tid + THREADS * k;
      const int rc = ( 3 + 2 * ( e >> 4 ) ) * YS + 3 + ( e & 15 );
      part += abs( bilinearW( WY_AT( 0 ) + rc, mvs[0].x & 15, mvs[0].y & 15, shift )
                   - bilinearW( WY_AT( 1 ) + rc, mvs[1].x & 15, mvs[1].y & 15, shift ) );
    }
    const int waveSum = WaveActiveSum( part );
    if( WaveIsFirstLane() )
    {
      InterlockedAdd( s_sad[12], waveSum );
    }
    GroupMemoryBarrierWithGroupSync();
    const int  centre = s_sad[12] - ( s_sad[12] >> 2 );
    const bool search = centre >= 16 * 16;
    if( search )
    {
      // All the bilinear predictions of the 20 x 20 samples from the merge motion minus 2 samples (the windows'
      // samples from (1, 1)), then the other 24 SADs over every other row
      [unroll]
      for( int s = 0; s < 2; s++ )
      {
        const int at = WY_AT( s ) + YS + 1;
        for( int o = tid; o < 400; o += THREADS )
        {
          const int v = o / 20;
          s_pool[WORK_AT + 400 * s + o] = int16_t( bilinearW( at + v * YS + o - 20 * v, mvs[s].x & 15, mvs[s].y & 15, shift ) );
        }
      }
      GroupMemoryBarrierWithGroupSync();
      for( int p = tid; p < 25 * 8; p += THREADS )
      {
        const int pos = p % 25;
        if( pos != 12 )
        {
          InterlockedAdd( s_sad[pos], sadRowAt( WORK_AT, WORK_AT + 400, 2 * ( p / 25 ), pos % 5 - 2, pos / 5 - 2 ) );
        }
      }
      GroupMemoryBarrierWithGroupSync();
    }
    // Every thread works out the refinement itself (so none waits for another), from the SADs with the centre's
    // reduced by a quarter.
    const int2 delta = search ? dmvrDeltaFrom( centre ) : int2( 0, 0 );
#if BDOF
    dmvrCost = search ? dmvrMinCost( centre ) : centre;
#endif
    // Threads 0 .. 15 write the refined motion where TMVP reads it (the 8x8 grid) into the colocated motion, which
    // mider.slang (GPU front end) or DecCu's storeDmvrMotion (CPU front end) stored before, one int each; threads
    // 16 and 17 write vvdec's DMVR MV cache.
    if( tid < 16 )
    {
      const int x2 = ( ( x - 1 ) & ~7 ) + 8 + 8 * ( ( tid >> 2 ) & 1 );
      const int y2 = ( ( y - 1 ) & ~7 ) + 8 + 8 * ( tid >> 3 );
      if( x2 < x + 16 && y2 < y + 16 )
      {
        const int  mask = ( 1 << ( pc.ctuLog2 - 3 ) ) - 1;
        const int  ctu  = ( y2 >> pc.ctuLog2 ) * pc.widthInCtus + ( x2 >> pc.ctuLog2 );
        const int2 m    = ( tid & 2 ) == 0 ? mv0 + delta : mv1 - delta;
        pc.colMotion[( ctu * pc.num8x8 + ( ( x2 >> 3 ) & mask ) + ( ( ( y2 >> 3 ) & mask ) << ( pc.ctuLog2 - 3 ) ) ) * 5 + ( tid & 3 )] = ( tid & 1 ) == 0 ? m.x : m.y;
      }
    }
    else if( tid < 18 )
    {
      pc.dmvr[int( job[8] ) * 2 + tid - 16] = tid == 16 ? delta.x : delta.y;
    }
    mvR[0] = clipStorage( mv0 + delta );
    mvR[1] = clipStorage( mv1 - delta );
  }
#if BDOF
  const bool bdof = ( flags & FLAG_BDOF ) != 0 && dmvrCost >= 2 * 16 * 16;
#endif

  // Luma: the horizontal passes of the 2-D filters, 23 rows of 16 samples in groups of 4.
  int16_t cxY[2][8], cyY[2][8];
  int     rx0[2], ry0[2];
  bool    clampY[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    const int2 mv    = mvR[s];
    const int  xFrac = mv.x & 15, yFrac = mv.y & 15;
    const int2 dI    = int2( ( mv.x >> 4 ) - ( mvs[s].x >> 4 ), ( mv.y >> 4 ) - ( mvs[s].y >> 4 ) );
    [unroll]
    for( int k = 0; k < 8; k++ )
    {
      cxY[s][k] = int16_t( coeff( 0, xFrac, hpel, k ) );
      cyY[s][k] = int16_t( coeff( 0, yFrac, hpel, k ) );
    }
    rx0[s]    = 3 + dI.x - ( xFrac != 0 ? 3 : 0 );
    ry0[s]    = 3 + dI.y - ( yFrac != 0 ? 3 : 0 );
    clampY[s] = dI.x != 0 || dI.y != 0;
    if( s < lists && xFrac != 0 && yFrac != 0 )
    {
      const int shift1  = 6 - headRoom;
      const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
      const int at      = WY_AT( s );
      for( int g = tid; g < 92; g += THREADS )
      {
        const int r  = g >> 2;
        const int gi = ( g & 3 ) << 2;
        int       w[11];
        [unroll]
        for( int e = 0; e < 11; e++ )
        {
          w[e] = winY( at, ry0[s] + r, rx0[s] + gi + e, clampY[s] );
        }
        [unroll]
        for( int o = 0; o < 4; o++ )
        {
          int sum = 0;
          [unroll]
          for( int k = 0; k < 8; k++ )
          {
            sum += int( cxY[s][k] ) * w[o + k];
          }
          s_pool[WORK_AT + 368 * s + r * 16 + gi + o] = int16_t( ( sum + offset1 ) >> shift1 );
        }
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  int4 predY[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    if( s < lists )
    {
      predY[s] = lumaPred( WY_AT( s ), WORK_AT + 368 * s, j, i0, rx0[s], ry0[s], clampY[s], mvR[s].x & 15, mvR[s].y & 15, cxY[s], cyY[s], bi );
    }
  }
#if BDOF
  // BDOF: the border read now, from the luma windows; after the barrier, both predictions and the border into s_bdof
  // (the luma passes' part of the pool, free then) and the sums zeroed; the 4x4 blocks' sums after the next barrier,
  // and the corrected samples, each thread with its block's offset, after the one after.
  int16_t border[3];
  if( bdof )
  {
    tile16BdofBorder( tid, mvs, mvR, clampY, border );
  }
#endif
  GroupMemoryBarrierWithGroupSync();   // the luma windows are read; the chroma windows replace them
#if BDOF
  if( bdof )
  {
    tile16BdofStore( tid, j, i0, predY, border );
    for( int i = tid; i < 16 * 5; i += THREADS )
    {
      s_bsum[i] = 0;
    }
  }
#endif

  // Chroma: store the windows in shared memory, then the horizontal passes of the 2-D filters, 11 rows of 8 in pairs.
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int task = tid + THREADS * k;
    if( task < 88 * lists )
    {
      const int s    = task >= 88 ? 1 : 0;
      const int rem  = task - 88 * s;
      const int comp = rem >= 44 ? 1 : 0;
      const int rem2 = rem - 44 * comp;
      // (the row from the window's first sample: the chunk's samples before it, and past the row's room, are dropped)
      const int   at   = WC_AT( 2 * s + comp ) + ( rem2 >> 2 ) * CS;
      const int   c0   = 4 * ( rem2 & 3 ) - ( s == 0 ? leadC[0] : leadC[1] );
      const uint2 v    = cChunk[k];
      const uint  w[2] = { v.x, v.y };
      [unroll]
      for( int m = 0; m < 4; m++ )
      {
        if( uint( c0 + m ) < uint( CS ) )
        {
          s_pool[at + c0 + m] = int16_t( w[m >> 1] >> ( 16 * ( m & 1 ) ) );
        }
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
#if BDOF
  if( bdof )
  {
    bdofAddSums( tid >> 2, tid & 3, 16, 16 );
  }
#endif
  int16_t cxC[2][4], cyC[2][4];
  int     crx0[2], cry0[2];
  bool    clampC[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    const int2 mv    = mvR[s];
    const int  xFrac = mv.x & 31, yFrac = mv.y & 31;
    const int2 dI    = int2( ( mv.x >> 5 ) - ( mvs[s].x >> 5 ), ( mv.y >> 5 ) - ( mvs[s].y >> 5 ) );
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      cxC[s][k] = int16_t( kChroma[xFrac * 4 + k] );
      cyC[s][k] = int16_t( kChroma[yFrac * 4 + k] );
    }
    crx0[s]   = 1 + dI.x - ( xFrac != 0 ? 1 : 0 );
    cry0[s]   = 1 + dI.y - ( yFrac != 0 ? 1 : 0 );
    clampC[s] = dI.x != 0 || dI.y != 0;
    if( s < lists && xFrac != 0 && yFrac != 0 )
    {
      const int shift1  = 6 - headRoom;
      const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
      for( int g = tid; g < 88; g += THREADS )
      {
        const int comp = g >= 44 ? 1 : 0;
        const int gg   = g - 44 * comp;
        const int r    = gg >> 2;
        const int gi   = ( gg & 3 ) << 1;
        const int at   = WC_AT( 2 * s + comp );
        int       w[5];
        [unroll]
        for( int e = 0; e < 5; e++ )
        {
          w[e] = winC( at, cry0[s] + r, crx0[s] + gi + e, clampC[s] );
        }
        [unroll]
        for( int o = 0; o < 2; o++ )
        {
          int sum = 0;
          [unroll]
          for( int k = 0; k < 4; k++ )
          {
            sum += int( cxC[s][k] ) * w[o + k];
          }
          s_pool[CH_AT + 88 * ( 2 * s + comp ) + r * 8 + gi + o] = int16_t( ( sum + offset1 ) >> shift1 );
        }
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  int2 predC[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    if( s < lists )
    {
      predC[s] = chromaPred( WC_AT( 2 * s + cc ), CH_AT + 88 * ( 2 * s + cc ), cj, ci0, crx0[s], cry0[s], clampC[s], mvR[s].x & 31,
                             mvR[s].y & 31, cxC[s], cyC[s], bi );
    }
  }

  // Average the predictions, add the residuals and write the results.
  const int shiftNum = headRoom + 1;
  const int offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
  int4      vy;
  [unroll]
  for( int o = 0; o < 4; o++ )
  {
    vy[o] = bi ? biAverage( predY[0][o], predY[1][o], flags, bcwW1, shiftNum, offset, maxVal ) : predY[0][o];
  }
#if BDOF
  if( bdof )
  {
    const int2 v = bdofOffset( ( j >> 2 ) * 4 + ( i0 >> 2 ) );
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      vy[o] = bdofSample( i0 + o, j, v );
    }
  }
#endif
  if( ( flags & FLAG_LUMA_MAP ) != 0 )
  {
    vy = int4( pc.lmcsFwd[vy.x], pc.lmcsFwd[vy.y], pc.lmcsFwd[vy.z], pc.lmcsFwd[vy.w] );
  }
  if( ( flags & FLAG_RES_Y ) != 0 )
  {
    vy = clamp( vy + int4( lo16( resY.x ), hi16( resY.x ), lo16( resY.y ), hi16( resY.y ) ), 0, maxVal );
  }
  int16_t* outY = ( ( flags & FLAG_DEFER_Y ) != 0 ? pc.defer0 : pc.dst0 ) + atY;
  *(uint2*)outY  = uint2( uint( vy.x & 0xffff ) | uint( vy.y ) << 16, uint( vy.z & 0xffff ) | uint( vy.w ) << 16 );
  int2 vc;
  [unroll]
  for( int o = 0; o < 2; o++ )
  {
    vc[o] = bi ? biAverage( predC[0][o], predC[1][o], flags, bcwW1, shiftNum, offset, maxVal ) : predC[0][o];
  }
  if( ( flags & ( FLAG_RES_Y << ( 1 + cc ) ) ) != 0 )
  {
    vc = clamp( vc + int2( lo16( resC ), hi16( resC ) ), 0, maxVal );
  }
  // (each component's place first, then the thread's: see the residuals)
  int16_t* outCb = ( ( flags & FLAG_DEFER_CB ) != 0 ? pc.defer1 : pc.dst1 ) + atC;
  int16_t* outCr = ( ( flags & ( FLAG_DEFER_CB << 1 ) ) != 0 ? pc.defer2 : pc.dst2 ) + atC;
  *(uint*)( cc == 0 ? outCb : outCr ) = uint( vc.x & 0xffff ) | uint( vc.y ) << 16;
}


#if AFFINE
// --------------------------------------------------------------------------------------------------------------------
// Affine motion (xPredAffineBlk): each 4x4 subblock of the tile has its own motion, worked out from the CU's base
// motion and its steps across and down, as setAllAffineMv computes it for the motion buffer. Luma uses the 6-tap
// filter that vvdec uses for 4x4 blocks, refined by PROF; a chroma 4x4 block (8x8 luma) uses the average motion of
// its top-left and bottom-right luma subblocks. A job is 24 words (InterJob::setAffine):
//   0: x | y << 16 of the tile (luma)     1: width | height << 8 | flags << 16 (an InterJob's)
//   2, 3: L0's base motion (x, y; << 7)   4, 5: L1's       6: x | y << 16 of the CU
//   7: L0 reference | L1 reference << 8 | BCW's L1 weight + 2 << 16
//   12: the CU's width | height << 8 | PROF L0 << 16 | PROF L1 << 17 | spread L0 << 18 | spread L1 << 19
//   13..16: L0's steps (across x, y, down x, y; << 7)   17..20: L1's
// --------------------------------------------------------------------------------------------------------------------
groupshared int16_t s_ext[16 * 36];   // PROF: a list's 4x4 predictions with their borders, 6 x 6 each

int roundAffine( int v, int shift )
{
  return ( v + ( 1 << ( shift - 1 ) ) - ( v >= 0 ? 1 : 0 ) ) >> shift;
}

// Clips an affine subblock's motion as xPredAffineBlk does: with the PPS's
// wraparound, by wrapClipMv for the subblock's luma at (x, y) of width w (wrap
// tells whether it reads the wraparound copy); otherwise into the picture's
// extension around the CU.
int2 affineClip( int2 mv, int x, int y, int w, int cuX, int cuY, out bool wrap )
{
  wrap = false;
  if( pc.wrapPps != 0 )
  {
    bool shifted;
    const int2 clipped = wrapClipMv( mv, x, y, w, shifted );
    wrap               = !shifted;
    return clipped;
  }
  // (in a subpicture treated as a picture, as clipMvInSubpic clips it at the CU)
  const int horMax = ( g_bounds.z + 1 + 8 - cuX - 1 ) * 16;
  const int horMin = ( -pc.ctuSize - 8 - ( cuX - g_bounds.x ) + 1 ) * 16;
  const int verMax = ( g_bounds.w + 1 + 8 - cuY - 1 ) * 16;
  const int verMin = ( -pc.ctuSize - 8 - ( cuY - g_bounds.y ) + 1 ) * 16;
  return int2( min( horMax, max( horMin, mv.x ) ), min( verMax, max( verMin, mv.y ) ) );
}

// A list's affine model from the job: its base motion, and its steps across and down. (The words are picked with
// selects: the job's array indexed by the list would go to scratch memory.)
int2 affineBase( uint job[24], int list )
{
  return list == 0 ? int2( int( job[2] ), int( job[3] ) ) : int2( int( job[4] ), int( job[5] ) );
}

int2 affineHor( uint job[24], int list )
{
  return list == 0 ? int2( int( job[13] ), int( job[14] ) ) : int2( int( job[17] ), int( job[18] ) );
}

int2 affineVer( uint job[24], int list )
{
  return list == 0 ? int2( int( job[15] ), int( job[16] ) ) : int2( int( job[19] ), int( job[20] ) );
}

// The motion of the CU's luma subblock (sx, sy).
int2 affineMv( uint job[24], int list, int sx, int sy, int cw, int ch, bool spread )
{
  const int2 base = affineBase( job, list );
  const int2 hor  = affineHor( job, list );
  const int2 ver  = affineVer( job, list );
  const int2 pos  = spread ? int2( cw >> 1, ch >> 1 ) : int2( 2 + 4 * sx, 2 + 4 * sy );
  const int2 v    = base + hor * pos.x + ver * pos.y;
  return clipStorage( int2( roundAffine( v.x, 7 ), roundAffine( v.y, 7 ) ) );
}

// Clips an affine subblock's motion for a scaled reference (xPredAffineBlk):
// into the picture's extension around the CU, without wraparound.
int2 affineClipScaled( int2 mv, int cuX, int cuY )
{
  const int horMax = ( pc.picWidth + 8 - cuX - 1 ) * 16;
  const int horMin = ( -pc.ctuSize - 8 - cuX + 1 ) * 16;
  const int verMax = ( pc.picHeight + 8 - cuY - 1 ) * 16;
  const int verMin = ( -pc.ctuSize - 8 - cuY + 1 ) * 16;
  return int2( min( horMax, max( horMin, mv.x ) ), min( verMax, max( verMin, mv.y ) ) );
}

// The tile's subblocks and their motion (affineMotion works them out once for the whole tile): per list, each luma
// subblock's motion as affineClip leaves it, and whether it reads the wraparound copy; the same for each chroma
// subblock (affineChromaMv).
groupshared int2 s_sbMv[2][16];
groupshared int  s_sbWrap[2][16];
groupshared int2 s_cbMv[2][16];
groupshared int  s_cbWrap[2][16];

// A list's luma prediction of the thread's four samples, at 14 bits for a bi-prediction or PROF, else final: the
// tile's 4x4 subblocks are shared out four threads each (the subblock tid / 4, its row tid % 4), and each thread
// predicts its row's four samples. Each subblock's filter (the 6-tap filter vvdec uses for 4x4 blocks: kLuma4x4's
// first and last taps are zero) reads 9 x 9 reference samples from 2 before the subblock's integer position, which
// its four threads load into shared memory together. The 2-D filter is separable: each thread filters the six rows
// its samples need across, then down.
void affineLuma( int tid, uint job[24], int list, int x, int y, int w, int h, int cuX, int cuY, int cw, int ch, bool spread, bool prof, bool bi,
                 out int4 pred )
{
  pred              = int4( 0, 0, 0, 0 );
  const int  slot   = int( ( job[7] >> ( 8 * list ) ) & 255 );
  const int  nsx    = w >> 2;
  const int  nsb    = nsx * ( h >> 2 );
  const int  sb     = tid >> 2;   // the thread's subblock in the tile, and its row in it
  const int  v      = tid & 3;
  const bool mine   = sb < nsb;
  const int  ox     = ( x - cuX ) >> 2;   // the tile's first subblock in the CU
  const int  oy     = ( y - cuY ) >> 2;
  const int  sx     = ox + sb % nsx;      // the thread's subblock in the CU
  const int  sy     = oy + sb / nsx;
  if( pc.refs[slot].scaled != 0 )
  {
    // A scaled reference (RPR): each 4x4 block by xPredInterBlkRPR with
    // filter index 2, without PROF.
    const RefPlanes r = pc.refs[slot];
    if( mine )
    {
      const int2 mv = affineClipScaled( affineMv( job, list, sx, sy, cw, ch, spread ), cuX, cuY );
      [unroll]
      for( int o = 0; o < 4; o++ )
      {
        pred[o] = rprSample( r, r.y, 0, cuX + 4 * sx, cuY + 4 * sy, o, v, mv, bi, false, 2 );
      }
    }
    return;
  }
  int16_t*   ref      = refPlane( slot, 0 );
  const int  headRoom = max( 2, 14 - pc.bitDepth );
  const int  maxVal   = ( 1 << pc.bitDepth ) - 1;
  const bool hi       = bi || prof;
  const int  at       = sb * 81;   // the subblock's window: (r, c) is the reference sample (r - 2, c - 2) from its position

  // The windows: the subblock's rows v, v + 4 (and 8) from the thread
  if( mine )
  {
    const int2 mv   = s_sbMv[list][sb];
    const bool wrap = s_sbWrap[list][sb] != 0;
    const int  bx   = cuX + 4 * sx + ( mv.x >> 4 ) - 2;
    const int  by   = cuY + 4 * sy + ( mv.y >> 4 ) - 2;
    for( int r = v; r < 9; r += 4 )
    {
      [unroll]
      for( int c = 0; c < 9; c++ )
      {
        s_pool[at + r * 9 + c] = int16_t( refAt( ref, 0, bx + c, by + r, wrap ) );
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  const int2 mv    = s_sbMv[list][sb];
  const int  xFrac = mv.x & 15;
  const int  yFrac = mv.y & 15;
  if( mine )
  {
    if( xFrac == 0 && yFrac == 0 )
    {
      [unroll]
      for( int o = 0; o < 4; o++ )
      {
        const int s0 = int( s_pool[at + ( v + 2 ) * 9 + o + 2] );
        pred[o]      = hi ? s0 * ( 1 << headRoom ) - IF_INTERNAL_OFFS : s0;
      }
    }
    else if( xFrac != 0 && yFrac != 0 )
    {
      // Across: the window's rows v .. v + 5 (the filter's rows 1 .. 6), each over columns o + 0 .. o + 5
      const int shift1  = 6 - headRoom;
      const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
      int4      across[6];
      [unroll]
      for( int r = 0; r < 6; r++ )
      {
        int row[9];
        [unroll]
        for( int c = 0; c < 9; c++ )
        {
          row[c] = int( s_pool[at + ( v + r ) * 9 + c] );
        }
        [unroll]
        for( int o = 0; o < 4; o++ )
        {
          int hs = 0;
          [unroll]
          for( int c = 1; c < 7; c++ )
          {
            hs += kLuma4x4[xFrac * 8 + c] * row[o + c - 1];
          }
          across[r][o] = ( hs + offset1 ) >> shift1;
        }
      }
      // Down
      [unroll]
      for( int o = 0; o < 4; o++ )
      {
        int sum = 0;
        [unroll]
        for( int r = 1; r < 7; r++ )
        {
          sum += kLuma4x4[yFrac * 8 + r] * across[r - 1][o];
        }
        pred[o] = hi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
      }
    }
    else
    {
      const int shiftV = hi ? 6 - headRoom : 6;
      const int offset = hi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
      [unroll]
      for( int o = 0; o < 4; o++ )
      {
        int sum = 0;
        [unroll]
        for( int c = 1; c < 7; c++ )
        {
          sum += yFrac != 0 ? kLuma4x4[yFrac * 8 + c] * int( s_pool[at + ( v + c - 1 ) * 9 + o + 2] )
                            : kLuma4x4[xFrac * 8 + c] * int( s_pool[at + ( v + 2 ) * 9 + o + c - 1] );
        }
        const int r = ( sum + offset ) >> shiftV;
        pred[o]     = hi ? r : clip3( 0, maxVal, r );
      }
    }
  }
  if( prof )
  {
    // PROF: each 4x4 block gets a border from the reference samples nearest its motion (within the window); then the
    // gradients inside, and each sample moves along them by its offset from the subblock's motion (dMvScaleHor / Ver)
    if( mine )
    {
      [unroll]
      for( int o = 0; o < 4; o++ )
      {
        s_ext[sb * 36 + ( v + 1 ) * 6 + o + 1] = int16_t( pred[o] );
      }
      // the border's samples q = v, v + 4 .. (20 a subblock)
      for( int q = v; q < 20; q += 4 )
      {
        int bx, by;
        if( q < 6 )
        {
          bx = q - 1;
          by = -1;
        }
        else if( q < 12 )
        {
          bx = q - 7;
          by = 4;
        }
        else if( q < 16 )
        {
          bx = -1;
          by = q - 12;
        }
        else
        {
          bx = 4;
          by = q - 16;
        }
        const int sm = int( s_pool[at + ( by + ( yFrac >> 3 ) + 2 ) * 9 + bx + ( xFrac >> 3 ) + 2] );
        s_ext[sb * 36 + ( by + 1 ) * 6 + bx + 1] = int16_t( sm * ( 1 << headRoom ) - IF_INTERNAL_OFFS );
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if( mine )
    {
      const int2 hor    = affineHor( job, list );
      const int2 ver    = affineVer( job, list );
      const int  quadHX = hor.x * 4;
      const int  quadHY = hor.y * 4;
      const int  quadVX = ver.x * 4;
      const int  quadVY = ver.y * 4;
      const int  d0H    = ( ( hor.x + ver.x ) * 2 ) - ( ( quadHX + quadVX ) * 2 );
      const int  d0V    = ( ( hor.y + ver.y ) * 2 ) - ( ( quadHY + quadVY ) * 2 );
      const int limit   = 1 << max( pc.bitDepth + 1, 13 );
      const int offset  = ( 1 << ( headRoom - 1 ) ) + IF_INTERNAL_OFFS;
      [unroll]
      for( int o = 0; o < 4; o++ )
      {
        const int e   = sb * 36 + ( v + 1 ) * 6 + o + 1;
        const int gx  = ( int( s_ext[e + 1] ) >> 6 ) - ( int( s_ext[e - 1] ) >> 6 );
        const int gy  = ( int( s_ext[e + 6] ) >> 6 ) - ( int( s_ext[e - 6] ) >> 6 );
        const int dmH = clamp( roundAffine( d0H + o * quadHX + v * quadVX, 8 ), -31, 31 );
        const int dmV = clamp( roundAffine( d0V + o * quadHY + v * quadVY, 8 ), -31, 31 );
        const int dI  = clamp( dmH * gx + dmV * gy, -limit, limit - 1 );
        const int val = int( int16_t( pred[o] + dI ) );
        pred[o]       = bi ? val : clip3( 0, maxVal, ( val + offset ) >> headRoom );
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();   // before the next writes to the shared memory
}

// The motion of the CU's chroma subblock (csx, csy) of 4 x 4 chroma samples (xPredAffineBlk's chromaMvFld), clipped
// by affineClip (wrap is affineClip's). In 4:4:4 it is the luma subblock's motion; otherwise the rounded average of
// two luma subblocks' motion: the two the chroma subblock covers (4:2:2), or the top-left and bottom-right of the
// four (4:2:0).
int2 affineChromaMv( uint job[24], int list, int csx, int csy, int cw, int ch, bool spread, int cuX, int cuY, out bool wrap )
{
  const int lx = cuX + ( ( 4 * csx ) << pc.scaleX );   // the subblock's luma position
  const int ly = cuY + ( ( 4 * csy ) << pc.scaleY );
  if( pc.scaleX == 0 )
  {
    return affineClip( affineMv( job, list, csx, csy, cw, ch, spread ), lx, ly, 4, cuX, cuY, wrap );
  }
  const int  sy  = csy << pc.scaleY;
  const int2 sum = affineMv( job, list, 2 * csx, sy, cw, ch, spread ) + affineMv( job, list, 2 * csx + 1, sy + pc.scaleY, cw, ch, spread );
  return affineClip( int2( roundAffine( sum.x, 1 ), roundAffine( sum.y, 1 ) ), lx, ly, 4 << pc.scaleX, cuX, cuY, wrap );
}

// A list's chroma prediction of the tile for component comp: the thread's samples t = tid + THREADS k, at 14 bits
// for a bi-prediction, else final.
void affineChroma( int tid, uint job[24], int list, int comp, int x, int y, int w, int h, int cuX, int cuY, int cw, int ch, bool spread,
                   bool bi, out int pred[PER_THREAD] )
{
  const int slot = int( ( job[7] >> ( 8 * list ) ) & 255 );
  if( pc.refs[slot].scaled != 0 )
  {
    // A scaled reference (RPR): each chroma 4x4 block by xPredInterBlkRPR with
    // filter index 2, its motion clipped as the luma's is.
    const RefPlanes r     = pc.refs[slot];
    const int       tw    = w >> pc.scaleX;
    const int       log2T = firstbithigh( uint( tw ) );
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int t = tid + THREADS * k;
      pred[k]     = 0;
      if( t < tw * ( h >> pc.scaleY ) )
      {
        const int i   = t & ( tw - 1 );
        const int j   = t >> log2T;
        const int csx = ( ( ( x - cuX ) >> pc.scaleX ) >> 2 ) + ( i >> 2 );
        const int csy = ( ( ( y - cuY ) >> pc.scaleY ) >> 2 ) + ( j >> 2 );
        int2      mv;
        if( pc.scaleX == 0 )
        {
          mv = affineMv( job, list, csx, csy, cw, ch, spread );
        }
        else
        {
          const int  ly  = csy << pc.scaleY;
          const int2 sum = affineMv( job, list, 2 * csx, ly, cw, ch, spread ) + affineMv( job, list, 2 * csx + 1, ly + pc.scaleY, cw, ch, spread );
          mv             = int2( roundAffine( sum.x, 1 ), roundAffine( sum.y, 1 ) );
        }
        mv      = affineClipScaled( mv, cuX, cuY );
        pred[k] = rprSample( r, comp == 1 ? r.cb : r.cr, comp, ( cuX >> pc.scaleX ) + 4 * csx, ( cuY >> pc.scaleY ) + 4 * csy, i & 3, j & 3, mv, bi, false, 2 );
      }
    }
    return;
  }
  int16_t*  ref      = refPlane( slot, comp );
  const int sx       = pc.scaleX;
  const int sy       = pc.scaleY;
  const int tw       = w >> sx;   // the tile's chroma size
  const int th       = h >> sy;
  const int ncx      = tw >> 2;   // the tile's chroma subblocks across
  const int ncb      = ncx * ( th >> 2 );
  const int ocx      = ( ( x - cuX ) >> sx ) >> 2;   // the tile's first chroma subblock in the CU
  const int ocy      = ( ( y - cuY ) >> sy ) >> 2;
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int log2W    = log2Int( tw );

  // Each subblock's reference samples: 7 x 7, starting 1 sample before its integer position.
  for( int s = tid; s < ncb * 49; s += THREADS )
  {
    const int  cb   = s / 49;
    const int  r    = ( s % 49 ) / 7;
    const int  c    = s % 7;
    const int  csx  = ocx + cb % ncx;
    const int  csy  = ocy + cb / ncx;
    const int2 mv   = s_cbMv[list][cb];
    const bool wrap = s_cbWrap[list][cb] != 0;
    s_pool[s]       = int16_t( refAt( ref, comp, ( cuX >> sx ) + 4 * csx + ( mv.x >> ( 4 + sx ) ) - 1 + c, ( cuY >> sy ) + 4 * csy + ( mv.y >> ( 4 + sy ) ) - 1 + r, wrap ) );
  }
  GroupMemoryBarrierWithGroupSync();
  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int t = tid + THREADS * k;
    pred[k]     = 0;
    if( t >= tw * th )
    {
      continue;
    }
    const int  i     = t & ( tw - 1 );
    const int  j     = t >> log2W;
    const int2 mv    = s_cbMv[list][( j >> 2 ) * ncx + ( i >> 2 )];
    // The filter's phases in 32nds (InterpolationFilter's frac << ( 1 - scale ))
    const int  xFrac = ( mv.x & ( ( 16 << sx ) - 1 ) ) << ( 1 - sx );
    const int  yFrac = ( mv.y & ( ( 16 << sy ) - 1 ) ) << ( 1 - sy );
    const int  at    = ( ( j >> 2 ) * ncx + ( i >> 2 ) ) * 49;
    const int  u     = i & 3;
    const int  v     = j & 3;
    int        val;
    if( xFrac == 0 && yFrac == 0 )
    {
      const int s0 = int( s_pool[at + ( v + 1 ) * 7 + u + 1] );
      val          = bi ? s0 * ( 1 << headRoom ) - IF_INTERNAL_OFFS : s0;
    }
    else if( xFrac != 0 && yFrac != 0 )
    {
      const int shift1  = 6 - headRoom;
      const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
      int       sum2    = 0;
      for( int r = 0; r < 4; r++ )
      {
        int hs = 0;
        for( int c = 0; c < 4; c++ )
        {
          hs += kChroma[xFrac * 4 + c] * int( s_pool[at + ( v + r ) * 7 + u + c] );
        }
        sum2 += kChroma[yFrac * 4 + r] * ( ( hs + offset1 ) >> shift1 );
      }
      val = bi ? sum2 >> 6 : clip3( 0, maxVal, ( sum2 + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
    }
    else
    {
      const int shiftV = bi ? 6 - headRoom : 6;
      const int offset = bi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
      int       sum2   = 0;
      for( int c = 0; c < 4; c++ )
      {
        sum2 += yFrac != 0 ? kChroma[yFrac * 4 + c] * int( s_pool[at + ( v + c ) * 7 + u + 1] )
                           : kChroma[xFrac * 4 + c] * int( s_pool[at + ( v + 1 ) * 7 + u + c] );
      }
      const int r = ( sum2 + offset ) >> shiftV;
      val         = bi ? r : clip3( 0, maxVal, r );
    }
    pred[k] = val;
  }
  GroupMemoryBarrierWithGroupSync();
}

void affineMain( int tid, int index )
{
  uint job[24];
  [unroll]
  for( int k = 0; k < 6; k++ )
  {
    const uint4 q    = *(uint4*)( jobBase() + index * 24 + 4 * k );
    job[4 * k]       = q.x;
    job[4 * k + 1]   = q.y;
    job[4 * k + 2]   = q.z;
    job[4 * k + 3]   = q.w;
  }
  const int  x      = int( job[0] & 0xffff );
  const int  y      = int( job[0] >> 16 );
  const int  w      = int( job[1] & 255 );
  const int  h      = int( ( job[1] >> 8 ) & 255 );
  const int  flags  = int( job[1] >> 16 );
  const int  cuX    = int( job[6] & 0xffff );
  const int  cuY    = int( job[6] >> 16 );
  const int  bcwW1  = int( ( job[7] >> 16 ) & 15 ) - 2;
  const int  cw     = int( job[12] & 255 );
  const int  ch     = int( ( job[12] >> 8 ) & 255 );
  setBounds( cuX, cuY );
  const bool bi     = ( flags & ( FLAG_L0 | FLAG_L1 ) ) == ( FLAG_L0 | FLAG_L1 );
  const int  maxVal = ( 1 << pc.bitDepth ) - 1;

  // The subblocks' motion, per list: one thread each (list tid / 32; luma subblock tid % 32, or chroma tid % 32 - 16)
  {
    const int list = tid >> 5;
    const int e    = tid & 31;
    if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) != 0 )
    {
      const bool spread = ( ( job[12] >> ( 18 + list ) ) & 1 ) != 0;
      const int  nsx    = w >> 2;
      const int  ncx    = ( w >> pc.scaleX ) >> 2;
      bool       wrap;
      if( e < 16 && e < nsx * ( h >> 2 ) )
      {
        const int sx       = ( ( x - cuX ) >> 2 ) + e % nsx;
        const int sy       = ( ( y - cuY ) >> 2 ) + e / nsx;
        s_sbMv[list][e]    = affineClip( affineMv( job, list, sx, sy, cw, ch, spread ), cuX + 4 * sx, cuY + 4 * sy, 4, cuX, cuY, wrap );
        s_sbWrap[list][e]  = wrap ? 1 : 0;
      }
      else if( e >= 16 && pc.numComp > 1 && e - 16 < ncx * ( ( h >> pc.scaleY ) >> 2 ) )
      {
        const int c        = e - 16;
        const int csx      = ( ( ( x - cuX ) >> pc.scaleX ) >> 2 ) + c % ncx;
        const int csy      = ( ( ( y - cuY ) >> pc.scaleY ) >> 2 ) + c / ncx;
        s_cbMv[list][c]    = affineChromaMv( job, list, csx, csy, cw, ch, spread, cuX, cuY, wrap );
        s_cbWrap[list][c]  = wrap ? 1 : 0;
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // (weighted prediction starts from the 14-bit predictions, for a uni-prediction too)
  const bool hi = bi || ( flags & FLAG_WP ) != 0;
  int4       luma0 = int4( 0, 0, 0, 0 ), luma1 = int4( 0, 0, 0, 0 );
  for( int list = 0; list < 2; list++ )
  {
    if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) == 0 )
    {
      continue;
    }
    const bool prof   = ( ( job[12] >> ( 16 + list ) ) & 1 ) != 0;
    const bool spread = ( ( job[12] >> ( 18 + list ) ) & 1 ) != 0;
    int4       p;
    affineLuma( tid, job, list, x, y, w, h, cuX, cuY, cw, ch, spread, prof, hi, p );
    if( list == 0 || ( flags & FLAG_L0 ) == 0 )
    {
      luma0 = p;
    }
    else
    {
      luma1 = p;
    }
  }

  // Average the predictions (with BCW's weights), apply LMCS's mapping, add the residuals and write the results: the
  // thread's four samples of its subblock's row (affineLuma).
  const int shiftNum = max( 2, 14 - pc.bitDepth ) + 1;
  const int offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
  const int nsx      = w >> 2;
  if( ( tid >> 2 ) < nsx * ( h >> 2 ) )
  {
    const int sb  = tid >> 2;
    const int row = y + 4 * ( sb / nsx ) + ( tid & 3 );
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      const int at = row * pc.stride0 + x + 4 * ( sb % nsx ) + o;
      int       v  = ( flags & FLAG_WP ) != 0 ? wpSample( job[8], job[11], 0, luma0[o], luma1[o], maxVal )
                     : ( bi ? biAverage( luma0[o], luma1[o], flags, bcwW1, shiftNum, offset, maxVal ) : luma0[o] );
      if( ( flags & FLAG_LUMA_MAP ) != 0 )
      {
        v = int( pc.lmcsFwd[v] );
      }
      if( ( flags & FLAG_RES_Y ) != 0 )
      {
        v = clip3( 0, maxVal, v + int( pc.dst0[at] ) );
      }
      ( ( flags & FLAG_DEFER_Y ) != 0 ? pc.defer0 : pc.dst0 )[at] = int16_t( v );
    }
  }
  int pred0[PER_THREAD], pred1[PER_THREAD];
  // The chroma, one component at a time (reusing luma's prediction arrays).
  const int tw    = w >> pc.scaleX;   // the tile's chroma size
  const int th    = h >> pc.scaleY;
  const int log2C = log2Int( tw );
  for( int comp = 1; comp < pc.numComp; comp++ )
  {
    for( int list = 0; list < 2; list++ )
    {
      if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) == 0 )
      {
        continue;
      }
      const bool spread = ( ( job[12] >> ( 18 + list ) ) & 1 ) != 0;
      int        p[PER_THREAD];
      affineChroma( tid, job, list, comp, x, y, w, h, cuX, cuY, cw, ch, spread, hi, p );
      if( list == 0 || ( flags & FLAG_L0 ) == 0 )
      {
        pred0 = p;
      }
      else
      {
        pred1 = p;
      }
    }
    int16_t* dst = comp == 1 ? pc.dst1 : pc.dst2;
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int t = tid + THREADS * k;
      if( t < tw * th )
      {
        const int at = ( ( y >> pc.scaleY ) + ( t >> log2C ) ) * pc.stride12 + ( x >> pc.scaleX ) + ( t & ( tw - 1 ) );
        int       v  = ( flags & FLAG_WP ) != 0 ? wpSample( job[8 + comp], job[11], comp, pred0[k], pred1[k], maxVal )
                       : ( bi ? biAverage( pred0[k], pred1[k], flags, bcwW1, shiftNum, offset, maxVal ) : pred0[k] );
        if( ( flags & ( FLAG_RES_Y << comp ) ) != 0 )
        {
          v = clip3( 0, maxVal, v + int( dst[at] ) );
        }
        ( ( flags & ( FLAG_DEFER_CB << ( comp - 1 ) ) ) != 0 ? ( comp == 1 ? pc.defer1 : pc.defer2 ) : dst )[at] = int16_t( v );
      }
    }
  }
}
#endif

[shader( "compute" )]
[numthreads( THREADS, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int index = int( group.x ) + int( group.y ) * 32768;
  if( index >= ( pc.count >= 0 ? pc.count : int( pc.countAt[0] ) ) )
  {
    return;   // (the whole workgroup returns here)
  }
  const int  tid   = int( local.x );
#if AFFINE
  affineMain( tid, index );
  return;
#endif
  uint*      job   = jobBase() + index * 12;
  const int  x     = int( job[0] & 0xffff );
  const int  y     = int( job[0] >> 16 );
  const int  w     = int( job[1] & 255 );
  const int  h     = int( ( job[1] >> 8 ) & 255 );
  const int  flags = int( job[1] >> 16 );
  const int2 mv0   = int2( int( job[2] ), int( job[3] ) );
  const int2 mv1   = int2( int( job[4] ), int( job[5] ) );
  const int  cuX   = int( job[6] & 0xffff );
  const int  cuY   = int( job[6] >> 16 );
  const int  ref0  = int( job[7] & 255 );
  const int  ref1  = int( ( job[7] >> 8 ) & 255 );
  const int  bcwW1 = int( ( job[7] >> 16 ) & 15 ) - 2;   // BCW's L1 weight
  const int  cuW   = int( job[7] >> 20 );                // the clipping block's width
  setBounds( cuX, cuY );
  const bool hpel  = ( flags & FLAG_HPEL ) != 0;
  const bool bi    = ( flags & ( FLAG_L0 | FLAG_L1 ) ) == ( FLAG_L0 | FLAG_L1 );
  const bool dmvr  = ( flags & FLAG_DMVR ) != 0;
  const int  maxVal = ( 1 << pc.bitDepth ) - 1;

  if( ( flags & ( FLAG_GEO | FLAG_WP ) ) == 0 && pc.wide != 0 && w == 16 && h == 16 && tile16Inside( x, y, flags, mv0, mv1 ) )
  {
    tile16( tid, index, job, x, y, flags, mv0, mv1, ref0, ref1, bi, dmvr, hpel );
    return;
  }

  // DMVR: work out the subblock's refined motion, and write it for vvdec's TaskFinishMotionInfo.
  int2 refined0 = mv0, refined1 = mv1;
  if( dmvr )
  {
    // The bilinear predictions of the subblock's (w + 4) x (h + 4) samples
    // from the merge motion minus 2 samples (with the PPS's wraparound, from
    // the wraparound copy).
    const bool wrapB  = pc.wrapPps != 0;
    const int2 start0 = clipMv( mv0, cuX, cuY, cuW ) - int2( 32, 32 );
    const int2 start1 = clipMv( mv1, cuX, cuY, cuW ) - int2( 32, 32 );
    const int  bw     = w + 4;
    // The bilinear filters read (w + 5) x (h + 5) samples from (x00, y00) and (x01, y01).
    const int x00 = x + ( start0.x >> 4 ), y00 = y + ( start0.y >> 4 );
    const int x01 = x + ( start1.x >> 4 ), y01 = y + ( start1.y >> 4 );
    if( x00 >= g_bounds.x && x00 + w + 5 <= g_bounds.z + 1 && y00 >= g_bounds.y && y00 + h + 5 <= g_bounds.w + 1 && x01 >= g_bounds.x
        && x01 + w + 5 <= g_bounds.z + 1 && y01 >= g_bounds.y && y01 + h + 5 <= g_bounds.w + 1 )
    {
      // Inside the picture (most subblocks): no clamping.
      int16_t*  src0  = refPlane( ref0, 0 ) + y00 * pc.stride0 + x00;
      int16_t*  src1  = refPlane( ref1, 0 ) + y01 * pc.stride0 + x01;
      // The thread's sample (u, v) is at index at in the reference pictures and
      // at index out in shared memory (rows of 20); each step advances THREADS
      // samples.
      const int stepV   = THREADS / bw;
      const int stepU   = THREADS - stepV * bw;
      const int stepAt  = stepV * pc.stride0 + stepU;
      const int stepOut = stepV * 20 + stepU;
      const int v       = tid / bw;
      int       u       = tid - v * bw;
      int       at      = v * pc.stride0 + u;
      int       out     = v * 20 + u;
      for( int s = tid; s < bw * ( h + 4 ); s += THREADS )
      {
        s_pool[BIL0_AT + out] = int16_t( bilinearInside( src0 + at, start0.x & 15, start0.y & 15 ) );
        s_pool[BIL1_AT + out] = int16_t( bilinearInside( src1 + at, start1.x & 15, start1.y & 15 ) );
        u   += stepU;
        at  += stepAt;
        out += stepOut;
        if( u >= bw )
        {
          u   -= bw;
          at  += pc.stride0 - bw;
          out += 20 - bw;
        }
      }
    }
    else
    {
      for( int s = tid; s < bw * ( h + 4 ); s += THREADS )
      {
        const int v = s / bw;
        const int u = s - v * bw;
        s_pool[BIL0_AT + v * 20 + u] = int16_t( bilinearAt( ref0, cuX, cuY, start0, x - cuX + u, y - cuY + v, wrapB ) );
        s_pool[BIL1_AT + v * 20 + u] = int16_t( bilinearAt( ref1, cuX, cuY, start1, x - cuX + u, y - cuY + v, wrapB ) );
      }
    }
    if( tid < 25 )
    {
      s_sad[tid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    // The SADs of the 25 offsets over every other row: each step adds one row
    // of one offset, summed up in shared memory.
    for( int p = tid; p < 25 * ( h >> 1 ); p += THREADS )
    {
      const int pos = p % 25;
      InterlockedAdd( s_sad[pos], dmvrSadRow( w, 2 * ( p / 25 ), pos % 5 - 2, pos / 5 - 2 ) );
    }
    GroupMemoryBarrierWithGroupSync();
    if( tid == 0 )
    {
      s_sad[12] -= s_sad[12] >> 2;
      const int2 delta       = dmvrDelta( w, h );
#if BDOF
      // The best cost the search found (or the centre's, when the search stops at once)
      int minCost = s_sad[12];
      if( minCost >= w * h )
      {
        for( int i = 0; i < 25; i++ )
        {
          minCost = min( minCost, s_sad[i] );
        }
      }
      s_minCost = minCost;
#endif
      const int  index8      = int( job[8] );
      pc.dmvr[index8 * 2]     = delta.x;
      pc.dmvr[index8 * 2 + 1] = delta.y;
      s_delta                = delta;
      // Write the refined motion where TMVP reads it (the 8x8 grid) into the
      // colocated motion, which mider.slang (GPU front end) or DecCu's
      // storeDmvrMotion (CPU front end) stored before.
      const int2 m0   = mv0 + delta;
      const int2 m1   = mv1 - delta;
      const int  mask = ( 1 << ( pc.ctuLog2 - 3 ) ) - 1;
      for( int y2 = ( ( y - 1 ) & ~7 ) + 8; y2 < y + h; y2 += 8 )
      {
        for( int x2 = ( ( x - 1 ) & ~7 ) + 8; x2 < x + w; x2 += 8 )
        {
          const int ctu = ( y2 >> pc.ctuLog2 ) * pc.widthInCtus + ( x2 >> pc.ctuLog2 );
          int*      e   = pc.colMotion + ( ctu * pc.num8x8 + ( ( x2 >> 3 ) & mask ) + ( ( ( y2 >> 3 ) & mask ) << ( pc.ctuLog2 - 3 ) ) ) * 5;
          e[0] = m0.x;
          e[1] = m0.y;
          e[2] = m1.x;
          e[3] = m1.y;
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
    refined0 = clipStorage( mv0 + s_delta );
    refined1 = clipStorage( mv1 - s_delta );
  }

#if BDOF
  // BDOF: with DMVR, only where the best cost is 2 w h or more (as in xProcessDMVR)
  const bool bdof = ( flags & FLAG_BDOF ) != 0 && ( !dmvr || s_minCost >= 2 * w * h );
  if( bdof )
  {
    for( int i = tid; i < 16 * 5; i += THREADS )
    {
      s_bsum[i] = 0;   // (predictBlock's barriers come before the sums)
    }
  }
#else
  const bool bdof = false;
#endif
  [loop]
  for( int comp = 0; comp < pc.numComp; comp++ )
  {
    // The component's subsampling, and the motion's fraction bits in its samples
    const int sx     = comp == 0 ? 0 : pc.scaleX;
    const int sy     = comp == 0 ? 0 : pc.scaleY;
    const int bx     = x >> sx, by = y >> sy, bw = w >> sx, bh = h >> sy;
    const int shiftX = 4 + sx;
    const int shiftY = 4 + sy;
    int       pred0[PER_THREAD], pred1[PER_THREAD];

    [loop]
    for( int list = 0; list < 2; list++ )
    {
      if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) == 0 )
      {
        continue;
      }
      const int  slot     = list == 0 ? ref0 : ref1;
      const int2 merge    = list == 0 ? mv0 : mv1;
      const int2 mv       = list == 0 ? refined0 : refined1;
      bool       windowed = false;
      int4       window   = int4( 0, 0, 0, 0 );
      int2       base     = int2( 0, 0 );
      int2       clipped;
      bool       wrap;   // whether the reads come from the wraparound copy
      if( dmvr )
      {
        // xFinalPaddedMCForDMVR: the motion is clipped at the subblock (and
        // read from the wraparound copy with the PPS's wraparound). When the
        // refinement moves the integer position, the reads stay in the
        // prefetched window of the merge motion (xPrefetchPad: with the PPS's
        // wraparound, from the copy unless wrapClipMv moves the window's motion).
        clipped = clipMv( mv, x, y, w );
        wrap    = pc.wrapPps != 0;
        const int2 deltaInt = int2( ( mv.x >> shiftX ) - ( merge.x >> shiftX ), ( mv.y >> shiftY ) - ( merge.y >> shiftY ) );
        if( deltaInt.x != 0 || deltaInt.y != 0 )
        {
          const int  half   = comp == 0 ? 3 : 1;
          const int2 before = merge - int2( half << shiftX, half << shiftY );
          int2       start;
          if( pc.wrapPps != 0 )
          {
            bool shifted;
            start = wrapClipMv( before, x, y, w, shifted );
            wrap  = !shifted;
          }
          else
          {
            start = clipMv( before, x, y, w );
          }
          window   = int4( bx + ( start.x >> shiftX ), by + ( start.y >> shiftY ), bw + 2 * half + 1, bh + 2 * half + 1 );
          base     = int2( window.x + half + deltaInt.x, window.y + half + deltaInt.y );
          windowed = true;
        }
      }
      else
      {
        // xPredInterUni: the motion is clipped at the CU (or SbTMVP's joined
        // sub-PUs), and read from the wraparound copy with the SPS's wraparound.
        clipped = clipMv( mv, cuX, cuY, cuW );
        wrap    = pc.wrapSps != 0;
      }
      // One call per filter length keeps the shader's code small (it ran slower
      // when it outgrew the instruction cache).
      // (weighted prediction starts from the 14-bit predictions, for a uni-prediction too)
      const bool hi = bi || ( flags & FLAG_WP ) != 0;
      int        pred[PER_THREAD];
      if( pc.refs[slot].scaled != 0 )
      {
        // A scaled reference (RPR): the motion is not clipped (xPredInterUni),
        // and the samples are predicted one by one (no DMVR, no BDOF),
        // positioned from the CU's origin.
        rprBlock( tid, slot, comp, cuX >> sx, cuY >> sy, bx - ( cuX >> sx ), by - ( cuY >> sy ), bw, bh, mv, hi, hpel, pred );
      }
      else if( comp == 0 )
      {
        predictBlock<8>( tid, slot, comp, bx, by, bw, bh, clipped, hi, hpel, windowed, window, base, wrap, pred );
      }
      else
      {
        predictBlock<4>( tid, slot, comp, bx, by, bw, bh, clipped, hi, hpel, windowed, window, base, wrap, pred );
      }
      if( list == 0 || ( flags & FLAG_L0 ) == 0 )
      {
        pred0 = pred;
      }
      else
      {
        pred1 = pred;
      }
#if BDOF
      if( bdof && comp == 0 )
      {
        bdofStore( tid, list, bw, bh, pred, slot, windowed ? base.x : bx + ( clipped.x >> shiftX ), windowed ? base.y : by + ( clipped.y >> shiftY ),
                   clipped, windowed, window, wrap );
      }
#endif
    }
    int bdofOut[PER_THREAD] = { 0, 0, 0, 0 };
#if BDOF
    if( bdof && comp == 0 )
    {
      GroupMemoryBarrierWithGroupSync();
      bdofApply( tid, bw, bh, bdofOut );
    }
#endif

    int16_t*   dst      = comp == 0 ? pc.dst0 : ( comp == 1 ? pc.dst1 : pc.dst2 );
    const int  stride   = comp == 0 ? pc.stride0 : pc.stride12;
    const bool residual = ( flags & ( FLAG_RES_Y << comp ) ) != 0;
    const bool map      = comp == 0 && ( flags & FLAG_LUMA_MAP ) != 0;
    const bool defer    = comp == 0 ? ( flags & FLAG_DEFER_Y ) != 0 : ( flags & ( FLAG_DEFER_CB << ( comp - 1 ) ) ) != 0;
    int16_t*   out      = defer ? ( comp == 0 ? pc.defer0 : ( comp == 1 ? pc.defer1 : pc.defer2 ) ) : dst;
    const int  shiftNum = max( 2, 14 - pc.bitDepth ) + 1;
    const int  offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
    const int  log2W    = log2Int( bw );
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int t = tid + THREADS * k;
      if( t >= bw * bh )
      {
        break;
      }
      const int i  = t & ( bw - 1 );
      const int j  = t >> log2W;
      int       v;
      if( ( flags & FLAG_WP ) != 0 )
      {
        v = wpSample( job[8 + comp], job[11], comp, pred0[k], pred1[k], maxVal );
      }
      else if( ( flags & FLAG_GEO ) != 0 )
      {
        // (the position of the CU's luma sample under this sample)
        v = geoBlend( pred0[k], pred1[k], geoWeight( job[9], ( bx + i - ( cuX >> sx ) ) << sx, ( by + j - ( cuY >> sy ) ) << sy ), maxVal );
      }
      else
      {
        v = bdof && comp == 0 ? bdofOut[k] : ( bi ? biAverage( pred0[k], pred1[k], flags, bcwW1, shiftNum, offset, maxVal ) : pred0[k] );
      }
      const int at = ( by + j ) * stride + bx + i;
      if( map )
      {
        v = int( pc.lmcsFwd[v] );
      }
      if( residual )
      {
        v = clip3( 0, maxVal, v + int( dst[at] ) );
      }
      out[at] = int16_t( v );
    }
  }
}
