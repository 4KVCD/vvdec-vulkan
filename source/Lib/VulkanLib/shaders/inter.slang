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
  int*       dmvr;        // DMVR's refinements (vvdec's DMVR MV cache, as vvdec's Mv: hor, ver); null: none wanted
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
                          // tile may load them in 8- and 16-byte pieces, and the chroma is 4:2:0, as tile expects (see
                          // tileJob for the jobs of pictures with RPR, subpictures or wraparound)
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

// The picture's layout values. The variants built with WIDE=1 take the pictures of push.wide (GpuReconPrediction.cpp:
// 4:2:0 planes whose rows start 16-byte aligned with margins of 16 samples or more, no wraparound, no subpictures
// treated as pictures, no scaled references), for which these are constants: the compiler drops the paths those
// pictures never take (inter's code 56 -> 47 KB, BDOF's 65 -> 56 KB, affine's 45 -> 25 KB, with fewer scalar
// registers spilled; Beekeeper slower 4K +2.9%, Top Gun slower 1080p +1.0%).
#ifndef WIDE
#define WIDE 0
#endif
int  chromaScaleX() { return WIDE ? 1 : pc.scaleX; }
int  chromaScaleY() { return WIDE ? 1 : pc.scaleY; }
int  numComponents() { return WIDE ? 3 : pc.numComp; }
bool hasChroma() { return numComponents() > 1; }
// Whether tile's own chroma code runs (4:2:0's; 4:2:2 and 4:4:4 take chromaTile after the tile's luma)
bool chroma420() { return hasChroma() && chromaScaleX() == 1 && chromaScaleY() == 1; }
bool needsChromaTile() { return hasChroma() && !chroma420(); }
bool widePlanes() { return WIDE != 0 || pc.wide != 0; }
bool wrapSps() { return WIDE == 0 && pc.wrapSps != 0; }
bool wrapPps() { return WIDE == 0 && pc.wrapPps != 0; }
bool subpicsOn() { return WIDE == 0 && pc.subpicOn != 0; }
bool refScaled( int slot ) { return WIDE == 0 && pc.refs[slot].scaled != 0; }

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
  const int place  = mirror == 2 ? ( 111 - oy - ly ) * 112 + ox + lx : ( oy + ly ) * 112 + ( mirror == 1 ? 111 - ox - lx : ox + lx );
  return int( *at( pc.geo, uint( mask * 112 * 112 + place ) ) );
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

// Combines a sample's predictions from the job's lists as its flags say (p0: the first list's; p1: L1's when both are
// used): GPM's blend, with the first partition's weight gpmWeight (out of 8); weighted prediction, with component
// comp's parameters (the job's words wpParams and wpOffsets1, see wpSample); the average, with BCW's weights (bcwW1:
// L1's); or the one list's prediction, which is final already. (GPM's jobs have no weighted prediction.)
int combineLists( int flags, int comp, int p0, int p1, int gpmWeight, uint wpParams, uint wpOffsets1, int bcwW1 )
{
  const int  maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int  shiftNum = max( 2, 14 - pc.bitDepth ) + 1;
  const int  offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
  const bool bi       = ( flags & ( FLAG_L0 | FLAG_L1 ) ) == ( FLAG_L0 | FLAG_L1 );
  if( ( flags & FLAG_GEO ) != 0 )
  {
    return geoBlend( p0, p1, gpmWeight, maxVal );
  }
  if( ( flags & FLAG_WP ) != 0 )
  {
    return wpSample( wpParams, wpOffsets1, comp, p0, bi ? p1 : 0, maxVal );
  }
  return bi ? biAverage( p0, p1, flags, bcwW1, shiftNum, offset, maxVal ) : p0;
}
#define THREADS 64
#define PER_THREAD 4   // a 16x16 tile's luma samples per thread

// The interpolation filters' taps (vvdec's InterpolationFilter tables), in pc.geo after GPM's masks
// (GpuRecon.cpp's uploadGeoWeights, in this order): kLuma, then the 6-tap luma filter vvdec uses for 4x4 blocks
// (affine), RPR's filters (m_lumaFilterRPR1 and 2, used past 1.25x and 1.75x), their affine variants, then the chroma
// ones, the half-sample luma filter and the bilinear one (m_bilinearFilterPrec4). Read from memory: as the shader's
// own constant arrays, read with a run-time index, Intel's compiler copied each array into the thread's scratch
// memory at each use (VideoQ 4K on the 285K's Intel Graphics: inter 14.7 ms a picture).
static const uint kGeoValues = 6 * 112 * 112;   // GEO_NUM_PRESTORED_MASK masks of 112 x 112
struct Taps
{
  uint base;
  __subscript( int i )->int
  {
    get { return int( *at( pc.geo, kGeoValues + base + uint( i ) ) ); }
  }
};
static const Taps kLuma           = { 0 };
static const Taps kLuma4x4        = { 128 };
static const Taps kLumaRPR1       = { 256 };
static const Taps kLumaRPR2       = { 384 };
static const Taps kAffineLumaRPR1 = { 512 };
static const Taps kAffineLumaRPR2 = { 640 };
static const Taps kChroma         = { 768 };
static const Taps kChromaRPR1     = { 896 };
static const Taps kChromaRPR2     = { 1024 };
static const Taps kLumaHpel       = { 1152 };
static const Taps kBilinear       = { 1160 };

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

// How a block is predicted from a scaled reference (xPredInterBlkRPR), worked out once for the block: where its first
// sample lies in the reference and how far each next sample steps (both in 1/1024 reference samples), the filter
// along each axis (rprCoeff's index), and the reference's size and stride in the component's samples
struct RprGeometry
{
  int  x0, y0;
  int  stepX, stepY;
  int  xFilter, yFilter;
  bool hpelX, hpelY;   // the half-sample filter applies (only without scaling)
  int  sx, sy;         // the component's subsampling
  int  refW, refH, stride;
};

// The geometry of the block at (bx, by) (in component comp's samples) predicted from the scaled reference r with
// motion mv (1/16 luma sample, not clipped). The block's position in the reference follows from the scaling ratio and
// the scaling windows. The filter is that of the filter index (2 for affine), or a stronger one past 1.25x and 1.75x.
RprGeometry rprGeometry( RefPlanes r, int comp, int bx, int by, int2 mv, bool hpel, int filterIndex )
{
  RprGeometry g;
  g.sx           = comp == 0 ? 0 : chromaScaleX();
  g.sy           = comp == 0 ? 0 : chromaScaleY();
  g.refW         = int( r.size & 0xffffu ) >> g.sx;
  g.refH         = int( r.size >> 16 ) >> g.sy;
  g.stride       = comp == 0 ? int( r.strides & 0xffffu ) : int( r.strides >> 16 );
  const int one  = 1 << 14;   // SCALE_1X
  const int thr1 = one * 5 / 4;
  const int thr2 = one * 7 / 4;
  g.xFilter      = r.ratioX > thr2 ? 4 : ( r.ratioX > thr1 ? 3 : filterIndex );
  g.yFilter      = r.ratioY > thr2 ? 4 : ( r.ratioY > thr1 ? 3 : filterIndex );
  if( comp == 0 && filterIndex == 2 )
  {
    g.xFilter += r.ratioX > thr1 ? 2 : 0;
    g.yFilter += r.ratioY > thr1 ? 2 : 0;
  }
  g.hpelX = hpel && r.ratioX == one;
  g.hpelY = hpel && r.ratioY == one;

  // the block's position in the reference, in 1 / 1024 samples
  const int sx   = g.sx, sy = g.sy;
  const int posX = ( ( bx << sx ) - pc.curWinLeft ) >> sx;
  const int posY = ( ( by << sy ) - pc.curWinTop ) >> sy;
  const int addX = comp == 0 ? 0 : ( 1 - pc.horColloc ) * 8 * ( r.ratioX - one );
  const int addY = comp == 0 ? 0 : ( 1 - pc.verColloc ) * 8 * ( r.ratioY - one );
  int64_t   x0   = ( ( int64_t( posX ) << ( 4 + sx ) ) + int64_t( mv.x ) ) * int64_t( r.ratioX ) + int64_t( addX );
  int64_t   y0   = ( ( int64_t( posY ) << ( 4 + sy ) ) + int64_t( mv.y ) ) * int64_t( r.ratioY ) + int64_t( addY );
  const int64_t rx = ( ( x0 < 0 ? -x0 : x0 ) + ( int64_t( 1 ) << ( 7 + sx ) ) ) >> ( 8 + sx );
  const int64_t ry = ( ( y0 < 0 ? -y0 : y0 ) + ( int64_t( 1 ) << ( 7 + sy ) ) ) >> ( 8 + sy );
  g.x0    = int( ( x0 < 0 ? -rx : rx ) + ( int64_t( r.winLeft ) << ( 10 - sx ) ) );
  g.y0    = int( ( y0 < 0 ? -ry : ry ) + ( int64_t( r.winTop ) << ( 10 - sy ) ) );
  g.stepX = ( r.ratioX + 8 ) >> 4;
  g.stepY = ( r.ratioY + 8 ) >> 4;
  return g;
}

// The reference position of sample i along one axis of a block (start and step from RprGeometry, s the axis's
// subsampling): its integer sample, and its filter phase (in 16ths for luma, 32nds for chroma)
int2 rprAxisPosition( int start, int step, int i, int s, int comp )
{
  const int p    = start + i * step + ( 1 << ( 5 - s ) );
  const int frac = ( p >> ( 6 - s ) ) & ( ( 16 << s ) - 1 );
  return int2( p >> 10, comp == 0 ? frac : frac << ( 1 - s ) );
}

// The filter's first pass rounds to the 14-bit intermediate, its second to the result: with bi the intermediate, else
// the final value
int rprShift1( int headRoom ) { return 6 - headRoom; }
int rprOffset1( int headRoom ) { return -IF_INTERNAL_OFFS * ( 1 << rprShift1( headRoom ) ); }
int rprResult( int sum, bool bi, int headRoom )
{
  const int maxVal = ( 1 << pc.bitDepth ) - 1;
  return bi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
}

// xPredInterBlkRPR for sample (col, row) of the block of geometry g, read from plane p: the 2-D filter, each tap's
// sample clamped into the reference (its border extension). rprBlock's fallback.
int rprSample( RprGeometry g, int16_t* p, int comp, int col, int row, bool bi )
{
  const int2 x        = rprAxisPosition( g.x0, g.stepX, col, g.sx, comp );
  const int2 y        = rprAxisPosition( g.y0, g.stepY, row, g.sy, comp );
  const int  taps     = comp == 0 ? 8 : 4;
  const int  half     = taps / 2 - 1;
  const int  headRoom = max( 2, 14 - pc.bitDepth );
  int        sum      = 0;
  for( int t = 0; t < taps; t++ )
  {
    const int yy = clamp( y.x - half + t, 0, g.refH - 1 );
    int       h  = 0;
    for( int k = 0; k < taps; k++ )
    {
      h += rprCoeff( comp, g.xFilter, x.y, g.hpelX, k ) * int( p[yy * g.stride + clamp( x.x - half + k, 0, g.refW - 1 )] );
    }
    sum += rprCoeff( comp, g.yFilter, y.y, g.hpelY, t ) * ( ( h + rprOffset1( headRoom ) ) >> rprShift1( headRoom ) );
  }
  return rprResult( sum, bi, headRoom );
}

// The tile's output rows filtered together in rprBlock, so that the window of a reference up to twice as large fits the
// shared pool
static const int kRprBandRows = 8;

// Predicts the thread's samples t = tid + THREADS k of a w x h tile from a scaled reference. The tile lies (ox, oy) into
// the block at (bx, by) that xPredInterBlkRPR predicts as a whole (the CU, or SbTMVP's joined sub-PUs), since the
// sample positions step from that block's origin. The threads filter the tile's rows a band at a time: they read the
// band's window of the reference (each sample clamped into it, as its border extension) into the shared pool, then
// the horizontal pass over its rows for the tile's columns, then the vertical pass of their samples in the band.
void rprBlock( int tid, int slot, int comp, int bx, int by, int ox, int oy, int w, int h, int2 mv, bool bi, bool hpel, inout int out[PER_THREAD] )
{
  const RefPlanes   r        = pc.refs[slot];
  int16_t*          p        = comp == 0 ? r.y : ( comp == 1 ? r.cb : r.cr );
  const RprGeometry g        = rprGeometry( r, comp, bx, by, mv, hpel, 0 );
  const int         log2W    = firstbithigh( uint( w ) );
  const int         taps     = comp == 0 ? 8 : 4;
  const int         half     = taps / 2 - 1;
  const int         headRoom = max( 2, 14 - pc.bitDepth );
  const int         bandRows = min( h, kRprBandRows );
  // The window spans the reference columns from the first tap of the tile's first column to the last tap of its last
  // column. A band of rows needs at most rowsMax reference rows: the rows' positions step evenly, so any band spans as
  // many reference rows as the first band, give or take one.
  const int xFirst  = rprAxisPosition( g.x0, g.stepX, ox, g.sx, comp ).x - half;
  const int cols    = rprAxisPosition( g.x0, g.stepX, ox + w - 1, g.sx, comp ).x + taps - half - xFirst;
  const int rowsMax = ( ( ( bandRows - 1 ) * g.stepY ) >> 10 ) + 2 + taps - 1;
  if( cols * rowsMax + rowsMax * w > kPoolValues )
  {
    // (a scaling past what the standard allows: the per-sample filter)
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int t = tid + THREADS * k;
      if( t < w * h )
      {
        out[k] = rprSample( g, p, comp, ox + ( t & ( w - 1 ) ), oy + ( t >> log2W ), bi );
      }
    }
    return;
  }
  for( int band = 0; band < h; band += bandRows )
  {
    const int yFirst = rprAxisPosition( g.y0, g.stepY, oy + band, g.sy, comp ).x - half;
    const int rows   = rprAxisPosition( g.y0, g.stepY, oy + band + bandRows - 1, g.sy, comp ).x + taps - half - yFirst;
    const int tmpAt  = cols * rows;   // the horizontal pass, after the window
    for( int s = tid; s < cols * rows; s += THREADS )
    {
      const int row = s / cols;
      const int y   = clamp( yFirst + row, 0, g.refH - 1 );
      const int x   = clamp( xFirst + s - row * cols, 0, g.refW - 1 );
      s_pool[s]     = p[y * g.stride + x];
    }
    GroupMemoryBarrierWithGroupSync();
    for( int s = tid; s < rows * w; s += THREADS )
    {
      const int  row = s >> log2W;
      const int  col = s & ( w - 1 );
      const int2 x   = rprAxisPosition( g.x0, g.stepX, ox + col, g.sx, comp );
      const int  at  = row * cols + x.x - half - xFirst;
      int        sum = 0;
      for( int k = 0; k < taps; k++ )
      {
        sum += rprCoeff( comp, g.xFilter, x.y, g.hpelX, k ) * int( s_pool[at + k] );
      }
      s_pool[tmpAt + s] = int16_t( ( sum + rprOffset1( headRoom ) ) >> rprShift1( headRoom ) );
    }
    GroupMemoryBarrierWithGroupSync();
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int t   = tid + THREADS * k;
      const int row = t >> log2W;
      if( t < w * h && row >= band && row < band + bandRows )
      {
        const int2 y   = rprAxisPosition( g.y0, g.stepY, oy + row, g.sy, comp );
        const int  at  = tmpAt + ( ( y.x - half - yFirst ) << log2W ) + ( t & ( w - 1 ) );
        int        sum = 0;
        for( int tap = 0; tap < taps; tap++ )
        {
          sum += rprCoeff( comp, g.yFilter, y.y, g.hpelY, tap ) * int( s_pool[at + ( tap << log2W )] );
        }
        out[k] = rprResult( sum, bi, headRoom );
      }
    }
    GroupMemoryBarrierWithGroupSync();   // (the next band, or the next block, writes the pool)
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
// samples), then a work area for DMVR's bilinear predictions (rows 0 .. 18 of
// 20 x 20: its search reads no more) or the filters' horizontal passes. Once
// luma is done, the chroma windows take the luma windows' place. A stored row
// starts 16 bytes aligned (8 for chroma) and holds the window's row from lead
// samples before its first (lead 0 or 1: the window read from a 4-byte aligned
// place), so that a thread stores what it read of a row with one store.
#define YW 23                             // a luma window's side: 16 + 7
#define YS 24                             // stored row length: a window row from lead samples before its first
#define CWD 11                            // a chroma window's side: 8 + 3
#define CS 12                             // stored row length: a window row from lead samples before its first
#define WY_AT( l ) ( ( l ) * YW * YS )
#define WORK_AT ( 2 * YW * YS )
#define DMVR_ROWS 19                      // the rows of DMVR's bilinear predictions kept, of 20
#define WC_AT( b ) ( ( b ) * CWD * CS )   // b: 2 * list + chroma component - 1
#define CH_AT WC_AT( 4 )                  // the chroma filters' horizontal passes, after the chroma windows (4 x 88)
// BDOF's predictions (s_bdof) lie past what the pool holds while they are in use: the generic path's filters (s_ref,
// s_tmp: 897 values) and tile16's chroma (windows and horizontal passes: 880 values)
#define BDOF_AT ( ( 16 + 7 ) * ( 16 + 7 ) + ( 16 + 7 ) * 16 )
// (the affine variant holds the reference samples of up to 16 subblocks, 9 x 9 each)
// The pool's 16-bit values (rprBlock checks that its window and horizontal pass fit)
#if AFFINE
static const int kPoolValues = 16 * 81;
#else
static const int kPoolValues = WORK_AT + 2 * DMVR_ROWS * 20;
#endif
groupshared int16_t s_pool[kPoolValues];
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
  if( subpicsOn() )
  {
    uint16_t* r = pc.subpics + ( ( cuY >> pc.ctuLog2 ) * pc.widthInCtus + ( cuX >> pc.ctuLog2 ) ) * 4;
    g_bounds    = int4( int( r[0] ), int( r[1] ), int( r[2] ), int( r[3] ) );
  }
}

// g_bounds in component comp's samples
int4 boundsOf( int comp )
{
  const int sx = comp == 0 ? 0 : chromaScaleX();
  const int sy = comp == 0 ? 0 : chromaScaleY();
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
    const int w      = comp == 0 ? pc.picWidth : pc.picWidth >> chromaScaleX();
    const int offset = comp == 0 ? pc.wrapOffset : pc.wrapOffset >> chromaScaleX();
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
  if( wrapSps() )
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
  const int sx       = comp == 0 ? 0 : chromaScaleX();
  const int sy       = comp == 0 ? 0 : chromaScaleY();
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
    // (one offset, never negative: see common.slang's samplesBefore)
    int16_t*  src    = ref + ( ( baseY - oy ) * stride + baseX - ox );
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

// Adds a sample's terms to the five sums (s_bsum's order), from its gradients in the two lists (g0, g1) and its
// predictions (p0, p1).
void bdofAddTerms( inout int sums[5], int2 g0, int2 g1, int p0, int p1 )
{
  const int gx = ( g0.x + g1.x ) >> 1;
  const int gy = ( g0.y + g1.y ) >> 1;
  const int di = ( p1 >> 4 ) - ( p0 >> 4 );
  // (with the gradients' signs as -1, 0 or 1, each term is a product: |gx| = sgn(gx) gx, and the terms that a sign
  // negates, zeroes or keeps)
  const int sx = clamp( gx, -1, 1 );
  const int sy = clamp( gy, -1, 1 );
  sums[0] += sx * gx;
  sums[1] += sy * gy;
  sums[2] += sx * di;
  sums[3] += sy * di;
  sums[4] += sy * gx;
}

// Adds part q (q = 0 .. 3, nine samples) of the sums over the 6 x 6 window of the tile's 4x4 block sb into s_bsum,
// which starts zeroed.
void bdofAddSums( int sb, int q, int w, int h )
{
  const int nsx   = w >> 2;
  const int sx    = sb % nsx;
  const int sy    = sb / nsx;
  int       sums[5] = { 0, 0, 0, 0, 0 };
  for( int p = q * 9; p < q * 9 + 9; p++ )
  {
    const int cx = clamp( 4 * sx - 1 + p % 6, 0, w - 1 );
    const int cy = clamp( 4 * sy - 1 + p / 6, 0, h - 1 );
    bdofAddTerms( sums, bdofGrad( 0, cx, cy ), bdofGrad( 1, cx, cy ), bdofP( 0, cx, cy ), bdofP( 1, cx, cy ) );
  }
  [unroll]
  for( int k = 0; k < 5; k++ )
  {
    InterlockedAdd( s_bsum[sb * 5 + k], sums[k] );
  }
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
// tile: a 16x16 tile (tile16, with DMVR and BDOF), or a tile of 16x8, 8x16 or 8x8 without them (SbTMVP's pairs of
// sub-PUs and its sub-PUs, small CUs), with weighted prediction or GPM's blend too. Its reference samples lie inside
// the picture, or for a 16x16 border tile (edges) near it: then its row reads are clamped into the picture. All its
// global reads are issued at once, in loads of 16 and 8 bytes from 4-byte aligned places, before the first wait for
// memory: each list's luma window of (w + 7) x (h + 7) samples around its motion (with DMVR around the merge motion,
// since the search and the refined prediction both read inside it), the chroma windows of (w / 2 + 3) x (h / 2 + 3),
// and the residuals. DMVR and the filters then work from shared memory. The shader is latency-bound, so all reads go
// out before the first wait.
// In the inter_bdof variant, tile16 also applies BDOF: its steps run in the phases between tile16's barriers, so
// that a BDOF tile waits at no more barriers than another tile.
// --------------------------------------------------------------------------------------------------------------------

// The 16 (8) bytes at sample i of plane p, 4-byte aligned: the compiler reads them with one load.
uint4 load16( int16_t* p, uint i )
{
  uint* w = (uint*)at( p, i );
  return uint4( w[0], w[1], w[2], w[3] );
}

uint2 load8( int16_t* p, uint i )
{
  uint* w = (uint*)at( p, i );
  return uint2( w[0], w[1] );
}

// Whether storeEight stores its samples as two halves with a fence between them (set on NVIDIA's GPUs,
// Context::createPipelineOf): their compiler (driver 616.92) crashed on the inter shaders' tile paths with the eight
// 16-bit stores together, and does not with them apart.
[vk::constant_id( 0 )] const bool kSplitStores = false;

// Stores the 8 (4) samples packed in pairs in v at s_pool[at], 16 (8) bytes aligned: the compiler stores them with one
// store (with kSplitStores, two).
void storeEight( int at, uint4 v )
{
  if( kSplitStores )
  {
    s_pool[at]     = int16_t( v.x );
    s_pool[at + 1] = int16_t( v.x >> 16 );
    s_pool[at + 2] = int16_t( v.y );
    s_pool[at + 3] = int16_t( v.y >> 16 );
    GroupMemoryBarrier();
    s_pool[at + 4] = int16_t( v.z );
    s_pool[at + 5] = int16_t( v.z >> 16 );
    s_pool[at + 6] = int16_t( v.w );
    s_pool[at + 7] = int16_t( v.w >> 16 );
    return;
  }
  const uint w[4] = { v.x, v.y, v.z, v.w };
  [unroll]
  for( int m = 0; m < 8; m++ )
  {
    s_pool[at + m] = int16_t( w[m >> 1] >> ( 16 * ( m & 1 ) ) );
  }
}

void storeFour( int at, uint2 v )
{
  s_pool[at]     = int16_t( v.x );
  s_pool[at + 1] = int16_t( v.x >> 16 );
  s_pool[at + 2] = int16_t( v.y );
  s_pool[at + 3] = int16_t( v.y >> 16 );
}

// Sample (r, c) of the window whose sample (0, 0) is at index at.
int winY( int at, int r, int c )
{
  return int( s_pool[at + r * YS + c] );
}

int winC( int at, int r, int c )
{
  return int( s_pool[at + r * CS + c] );
}

// When DMVR's refinement moves a list's integer position (by dI, the refined motion's minus the merge motion's), the
// filters read the samples at the moved position clamped into the window, as xPrefetchPad's padding is. In those few
// tiles the windows are rewritten in place to hold just those samples, window(r, c) = window(clamp(r + dI.y),
// clamp(c + dI.x)) (a list that did not move is copied as it is), so that the filters read every window alike: with
// no clamping, a read takes no arithmetic of its own. Each thread reads its share of the samples (one sample place
// for both lists, packed in pairs), and writes them after a barrier.
void tile16ShiftLuma( int tid, int at[2], int2 dI[2] )
{
  uint pairs[( YW * YW + THREADS - 1 ) / THREADS];
  [unroll]
  for( int k = 0; k < ( YW * YW + THREADS - 1 ) / THREADS; k++ )
  {
    const int i = min( tid + THREADS * k, YW * YW - 1 );
    const int r = i / YW, c = i - r * YW;
    pairs[k]    = 0;
    [unroll]
    for( int s = 0; s < 2; s++ )
    {
      const int v = winY( at[s], clamp( r + dI[s].y, 0, YW - 1 ), clamp( c + dI[s].x, 0, YW - 1 ) );
      pairs[k] |= uint( v & 0xffff ) << ( 16 * s );
    }
  }
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for( int k = 0; k < ( YW * YW + THREADS - 1 ) / THREADS; k++ )
  {
    const int i = tid + THREADS * k;
    if( i < YW * YW )
    {
      const int r = i / YW, c = i - r * YW;
      s_pool[at[0] + r * YS + c] = int16_t( pairs[k] );
      s_pool[at[1] + r * YS + c] = int16_t( pairs[k] >> 16 );
    }
  }
  GroupMemoryBarrierWithGroupSync();
}

// The same for the chroma windows (at: [2 * list + component - 1]): each list's Cb and Cr windows move by the list's
// dI (in chroma samples).
void tile16ShiftChroma( int tid, int at[4], int2 dI[2] )
{
  uint pairs[2][2];   // [k][list]: Cb's and Cr's samples
  [unroll]
  for( int k = 0; k < 2; k++ )
  {
    const int i = min( tid + THREADS * k, CWD * CWD - 1 );
    const int r = i / CWD, c = i - r * CWD;
    [unroll]
    for( int s = 0; s < 2; s++ )
    {
      const int rr = clamp( r + dI[s].y, 0, CWD - 1 ), cc = clamp( c + dI[s].x, 0, CWD - 1 );
      pairs[k][s]  = uint( winC( at[2 * s], rr, cc ) & 0xffff ) | ( uint( winC( at[2 * s + 1], rr, cc ) ) << 16 );
    }
  }
  GroupMemoryBarrierWithGroupSync();
  [unroll]
  for( int k = 0; k < 2; k++ )
  {
    const int i = tid + THREADS * k;
    if( i < CWD * CWD )
    {
      const int r = i / CWD, c = i - r * CWD;
      [unroll]
      for( int s = 0; s < 2; s++ )
      {
        s_pool[at[2 * s] + r * CS + c]     = int16_t( pairs[k][s] );
        s_pool[at[2 * s + 1] + r * CS + c] = int16_t( pairs[k][s] >> 16 );
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
}

// The horizontal pass of a list's 2-D luma filter in tile: it filters the window at at (whose first row and column are
// where the taps start) across, row by row, into tmp (rows of 16). The window has the tile's rows plus 7, and each row
// is filtered over the tile's width in groups of 4 samples, 2^lg groups a row. In a 16x16 tile, each thread takes one
// group in rows 0 to 15, and the first 28 threads one more in rows 16 to 22. A thread works out where its reads start
// once; each read is that place plus a constant.
void tileLumaH( int tid, int at, int tmp, int16_t cx[8], int shift1, int offset1, int rows, int lg )
{
  const int gi = ( tid & ( ( 1 << lg ) - 1 ) ) << 2;
  [unroll]
  for( int k = 0; k < 2; k++ )
  {
    const int r = ( tid >> lg ) + ( THREADS >> lg ) * k;
    if( r < rows )
    {
      int w[11];
      [unroll]
      for( int e = 0; e < 11; e++ )
      {
        w[e] = winY( at, r, gi + e );
      }
      [unroll]
      for( int o = 0; o < 4; o++ )
      {
        int sum = 0;
        [unroll]
        for( int t = 0; t < 8; t++ )
        {
          sum += int( cx[t] ) * w[o + t];
        }
        s_pool[tmp + r * 16 + gi + o] = int16_t( ( sum + offset1 ) >> shift1 );
      }
    }
  }
}

// The horizontal passes of a list's 2-D chroma filters in tile: they filter the Cb window at at and the Cr window CWD
// rows after it (as one window of 2 CWD rows, each with the tile's rows plus 3) across, into tmp (Cb's rows of 8, then
// Cr's). Each row is filtered over the tile's width in pairs of samples, 2^lg pairs a row. In a 16x16 tile, each
// thread takes one pair in rows 0 to 15, and the first 24 threads one more in rows 16 to 21.
void tileChromaH( int tid, int at, int tmp, int16_t cx[4], int shift1, int offset1, int rows, int lg )
{
  const int gi = ( tid & ( ( 1 << lg ) - 1 ) ) << 1;
  [unroll]
  for( int k = 0; k < 2; k++ )
  {
    const int r = ( tid >> lg ) + ( THREADS >> lg ) * k;
    if( r < CWD + rows && ( r < rows || r >= CWD ) )
    {
      int w[5];
      [unroll]
      for( int e = 0; e < 5; e++ )
      {
        w[e] = winC( at, r, gi + e );
      }
      [unroll]
      for( int o = 0; o < 2; o++ )
      {
        int sum = 0;
        [unroll]
        for( int t = 0; t < 4; t++ )
        {
          sum += int( cx[t] ) * w[o + t];
        }
        s_pool[tmp + r * 8 + gi + o] = int16_t( ( sum + offset1 ) >> shift1 );
      }
    }
  }
}

// DMVR's search: one row (j: 0, 2 .. 14) of the SAD at position pos (0 .. 24: hor = pos % 5 - 2, ver = pos / 5 - 2),
// from the bilinear predictions at b0 and b1 (rows of 20 samples): L0's row 2 + ver + j, columns x = 2 + hor + i,
// against L1's row 2 - ver + j, columns x + 4 - 2 h (h = hor + 2). Both are read as pairs of samples (columns 2k,
// 2k + 1), L0's pair k against L1's pair k + 2 - h from k = h >> 1, and compared a pair at a time with 16-bit vector
// instructions. With h odd that takes 9 pairs, whose first column of the first and second of the last lie outside;
// with h even 8, and the 9th pair is the 8th again: what lies outside comes off at the end. The predictions are 10-bit
// at most (bilinearW, bit depths up to 10): none negative, and 18 differences add up in 16 bits.
typealias Pair = vector<uint16_t, 2>;
uint sadRow( int b0, int b1, int j, int pos )
{
  const int  h   = pos % 5;
  const int  ver = pos / 5 - 2;
  const bool odd = ( h & 1 ) != 0;
  const int  a0  = b0 + ( 2 + ver + j ) * 20 + 2 * ( h >> 1 );
  const int  a1  = b1 + ( 2 - ver + j ) * 20 + 2 * ( ( h >> 1 ) + 2 - h );
  Pair       sum = Pair( 0, 0 );
  Pair       d0, d8;
  [unroll]
  for( int m = 0; m < 9; m++ )
  {
    const int  at = 2 * ( m == 8 && !odd ? 7 : m );
    const Pair a  = Pair( uint16_t( s_pool[a0 + at] ), uint16_t( s_pool[a0 + at + 1] ) );
    const Pair b  = Pair( uint16_t( s_pool[a1 + at] ), uint16_t( s_pool[a1 + at + 1] ) );
    const Pair d  = max( a, b ) - min( a, b );
    sum += d;
    if( m == 0 )
    {
      d0 = d;
    }
    if( m == 8 )
    {
      d8 = d;
    }
  }
  return uint( sum.x ) + uint( sum.y ) - ( uint( odd ? d0.x : d8.x ) + uint( d8.y ) );
}

// The 4 luma predictions (j, i0 .. i0 + 3) of a list from its window (whose sample (0, 0) is at index at), with
// predictBlock's arithmetic; tmp holds the horizontal pass of a 2-D filter.
int4 lumaPred( int at, int tmp, int j, int i0, int rx0, int ry0, int xFrac, int yFrac, int16_t cx[8], int16_t cy[8], bool bi )
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
      w[e] = winY( at, ry0 + j, rx0 + i0 + e );
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
        sum += int( cy[t] ) * winY( at, ry0 + j + t, rx0 + i0 + o );
      }
      const int v = ( sum + offset ) >> shiftV;
      out[o]      = bi ? v : clip3( 0, maxVal, v );
    }
    return out;
  }
  [unroll]
  for( int o = 0; o < 4; o++ )
  {
    const int v = winY( at, ry0 + j, rx0 + i0 + o );
    out[o]      = bi ? v * ( 1 << headRoom ) - IF_INTERNAL_OFFS : v;
  }
  return out;
}

// The 2 chroma predictions (j, i0) and (j, i0 + 1) of a list from its window, as lumaPred does with 4 taps; tmp
// holds the horizontal pass, in rows of 8.
int2 chromaPred( int at, int tmp, int j, int i0, int rx0, int ry0, int xFrac, int yFrac, int16_t cx[4], int16_t cy[4], bool bi )
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
      w[e] = winC( at, ry0 + j, rx0 + i0 + e );
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
        sum += int( cy[t] ) * winC( at, ry0 + j + t, rx0 + i0 + o );
      }
      const int v = ( sum + offset ) >> shiftV;
      out[o]      = bi ? v : clip3( 0, maxVal, v );
    }
    return out;
  }
  [unroll]
  for( int o = 0; o < 2; o++ )
  {
    const int v = winC( at, ry0 + j, rx0 + i0 + o );
    out[o]      = bi ? v * ( 1 << headRoom ) - IF_INTERNAL_OFFS : v;
  }
  return out;
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
// ring place tid + THREADS k), from the lists' windows (the window's sample (3, 3) lies at the refined motion's integer
// position: where DMVR moved it, tile16ShiftLuma moved the window). Read while the luma windows are there;
// tile16BdofStore stores it with the predictions once they are gone.
static const int kTileRing = 2 * ( 16 + 2 ) + 2 * 16;
void tile16BdofBorder( int tid, int at[2], int2 mvR[2], out int16_t border[3] )
{
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int r = min( tid + THREADS * k, 2 * kTileRing - 1 );
    // (the list is picked with selects: an array indexed per lane would go to scratch memory)
    const int  s    = r >= kTileRing ? 1 : 0;
    const int2 m    = s == 0 ? mvR[0] : mvR[1];
    const int2 b    = bdofBorderAt( r - kTileRing * s, 16, 16 );
    const int  xOff = ( m.x & 15 ) < 8 ? 1 : 0;
    const int  yOff = ( m.y & 15 ) < 8 ? 1 : 0;
    border[k]       = bdofBorderValue( winY( s == 0 ? at[0] : at[1], 3 + b.y + 1 - yOff, 3 + b.x + 1 - xOff ) );
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

// tile16's BDOF sums (bdofAddSums' sums over each 4x4 block's 6 x 6 window), a row at a time: thread (j, bx) adds up
// the five sums over row j's samples in block column bx's window (columns 4 bx - 1 .. 4 bx + 4, clamped into the
// tile: an edge column counts twice), then adds them to the sums of the block rows whose windows hold row j (rows
// 4 by - 1 .. 4 by + 4, clamped: rows 0 and 15 count twice). So a sample's gradients are worked out once per window
// row rather than once per window that holds it. The thread's own samples (j, 4 bx .. 4 bx + 3), whose predictions
// it holds, come from its registers; their lists' gradient differences, which tile16BdofSample needs, it keeps in dg,
// packed in pairs.
void tile16BdofSums( int j, int bx, int4 predY[2], out uint dg[4] )
{
  const int x0      = 4 * bx;
  int       sums[5] = { 0, 0, 0, 0, 0 };
  // The own samples at 14 bits, as s_bdof holds them, and the row's samples left and right of them
  int own[2][4], left[2], right[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      own[s][o] = int( int16_t( predY[s][o] ) );
    }
    left[s]  = bdofP( s, x0 - 1, j );
    right[s] = bdofP( s, x0 + 4, j );
  }
  [unroll]
  for( int o = 0; o < 4; o++ )
  {
    int2 g[2];
    [unroll]
    for( int s = 0; s < 2; s++ )
    {
      const int l = o == 0 ? left[s] : own[s][o - 1];
      const int r = o == 3 ? right[s] : own[s][o + 1];
      g[s]        = int2( ( r >> 6 ) - ( l >> 6 ), ( bdofP( s, x0 + o, j + 1 ) >> 6 ) - ( bdofP( s, x0 + o, j - 1 ) >> 6 ) );
    }
    bdofAddTerms( sums, g[0], g[1], own[0][o], own[1][o] );
    dg[o] = uint( ( g[0].x - g[1].x ) & 0xffff ) | ( uint( g[0].y - g[1].y ) << 16 );
  }
  // The window's first and last columns, x0 - 1 and x0 + 4 clamped into the tile
  [unroll]
  for( int e = 0; e < 2; e++ )
  {
    const int x = e == 0 ? max( x0 - 1, 0 ) : min( x0 + 4, 15 );
    bdofAddTerms( sums, bdofGrad( 0, x, j ), bdofGrad( 1, x, j ), bdofP( 0, x, j ), bdofP( 1, x, j ) );
  }
  // Into the block row that holds row j (twice for the tile's first and last rows), and the neighbouring block row
  // whose window reaches row j
  const int by     = j >> 2;
  const int weight = j == 0 || j == 15 ? 2 : 1;
  const int other  = ( j & 3 ) == 3 && j < 15 ? by + 1 : ( ( j & 3 ) == 0 && j > 0 ? by - 1 : -1 );
  [unroll]
  for( int k = 0; k < 5; k++ )
  {
    InterlockedAdd( s_bsum[( by * 4 + bx ) * 5 + k], weight * sums[k] );
  }
  if( other >= 0 )
  {
    [unroll]
    for( int k = 0; k < 5; k++ )
    {
      InterlockedAdd( s_bsum[( other * 4 + bx ) * 5 + k], sums[k] );
    }
  }
}

// bdofSample for the thread's own sample o (j, 4 bx + o), from its predictions and the gradient differences that
// tile16BdofSums kept (dg)
int tile16BdofSample( int4 predY[2], uint dg, int o, int2 v )
{
  const int shiftNum = 15 - pc.bitDepth;
  const int offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int b        = v.x * lo16( dg ) + v.y * hi16( dg );
  return clip3( 0, maxVal, int( int16_t( ( int( int16_t( predY[0][o] ) ) + int( int16_t( predY[1][o] ) ) + b + offset ) >> shiftNum ) ) );
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

// Whether tile may take the tile of w x h luma samples: each list's windows must lie inside the job's bounds (the
// picture, or its subpicture when that is treated as a picture: setBounds). (DMVR's search reads inside the windows.)
bool tileInside( int x, int y, int w, int h, int flags, int2 mv0, int2 mv1 )
{
  const int4 b      = boundsOf( 0 );
  const int4 c      = boundsOf( 1 );
  bool       inside = true;
  [unroll]
  for( int list = 0; list < 2; list++ )
  {
    if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) != 0 )
    {
      const int2 m            = list == 0 ? mv0 : mv1;
      const int  wx           = x + ( m.x >> 4 ) - 3, wy = y + ( m.y >> 4 ) - 3;
      const int  sx           = chromaScaleX(), sy = chromaScaleY();
      const int  cx           = ( x >> sx ) + ( m.x >> ( 4 + sx ) ) - 1, cy = ( y >> sy ) + ( m.y >> ( 4 + sy ) ) - 1;
      const bool lumaInside   = wx >= b.x && wx + w + 7 <= b.z + 1 && wy >= b.y && wy + h + 7 <= b.w + 1;
      const bool chromaInside = cx >= c.x && cx + ( w >> sx ) + 3 <= c.z + 1 && cy >= c.y && cy + ( h >> sy ) + 3 <= c.w + 1;
      inside                  = inside && lumaInside && ( chromaInside || !hasChroma() );
    }
  }
  return inside;
}

// Whether the job may take a tile path: every job of a picture of the WIDE variants; otherwise, where the planes'
// layout allows it (widePlanes), a job whose references are not scaled (RPR: of the picture's size, read as they are).
bool tileJob( int flags, int ref0, int ref1 )
{
  return widePlanes() && !( ( flags & FLAG_L0 ) != 0 && refScaled( ref0 ) ) && !( ( flags & FLAG_L1 ) != 0 && refScaled( ref1 ) );
}

// Whether tile may take border tiles (tileNear): not where subpictures are treated as pictures or with wraparound,
// whose reads past the bounds differ from what tileBorderRow makes of the planes' margins
bool borderTiles()
{
  return !subpicsOn() && !wrapSps();
}

// Whether tile may take a 16x16 tile whose windows reach past the picture's edges (a border tile): by at most 16 luma
// samples past each edge (as many chroma samples as those cover), so that the reads past an edge stay in the planes'
// margins, and no motion needs clipping (with DMVR's refinement of up to 2 samples too).
bool tileNear( int x, int y, int flags, int2 mv0, int2 mv1 )
{
  bool near = true;
  [unroll]
  for( int list = 0; list < 2; list++ )
  {
    if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) != 0 )
    {
      const int2 m  = list == 0 ? mv0 : mv1;
      const int  wx = x + ( m.x >> 4 ) - 3, wy = y + ( m.y >> 4 ) - 3;
      const int  sx = chromaScaleX(), sy = chromaScaleY();
      const int  cx = ( x >> sx ) + ( m.x >> ( 4 + sx ) ) - 1, cy = ( y >> sy ) + ( m.y >> ( 4 + sy ) ) - 1;
      // (the tile's chroma size, which is also the margin that 16 luma samples past an edge cover)
      const int  cw = 16 >> sx, ch = 16 >> sy;
      const bool lumaNear   = wx >= -16 && wx + YW <= pc.picWidth + 16 && wy >= -16 && wy + YW <= pc.picHeight + 16;
      const bool chromaNear = cx >= -cw && cx + cw + 3 <= ( pc.picWidth >> sx ) + cw && cy >= -ch && cy + ch + 3 <= ( pc.picHeight >> sy ) + ch;
      near                  = near && lumaNear && ( chromaNear || !hasChroma() );
    }
  }
  return near;
}

// Fixes a border tile's window row of n samples at s_pool[at], whose first sample is at column x0 of the picture. The
// reads took the samples beside the picture from the planes' margins, which hold no border extension on the GPU. So the
// samples left of column 0 are set to column 0's value, and those right of the last column to the last column's value,
// which is what vvdec's border extension of the reference picture holds there. (tileNear leaves both columns in the
// row.)
void tileBorderRow( int at, int x0, int n, int width )
{
  const int first = -x0;              // the row's sample at column 0
  const int last  = width - 1 - x0;   // at the picture's last column
  if( first > 0 )
  {
    const int16_t v = s_pool[at + first];
    for( int c = 0; c < first; c++ )
    {
      s_pool[at + c] = v;
    }
  }
  if( last < n - 1 )
  {
    const int16_t v = s_pool[at + last];
    for( int c = last + 1; c < n; c++ )
    {
      s_pool[at + c] = v;
    }
  }
}

// tileBorderRow for each row of a border tile's luma windows (at: the lists' windows, x0: their first columns), a
// thread per row: threads 0 .. 22 the first list's, 32 .. 54 the second's
void tileBorderLuma( int tid, int at[2], int x0[2], bool two )
{
  const int s = tid >> 5;
  const int r = tid & 31;
  if( r < YW && ( s == 0 || two ) )
  {
    tileBorderRow( at[s] + r * YS, x0[s], YW, pc.picWidth );
  }
}

// The same for the chroma windows (at: [2 * list + component - 1]): threads 16 b .. 16 b + 10 the rows of window b
void tileBorderChroma( int tid, int at[4], int x0[2], bool two )
{
  const int b = tid >> 4;
  const int r = tid & 15;
  if( r < CWD && ( b < 2 || two ) )
  {
    tileBorderRow( at[b] + r * CS, x0[b >> 1], CWD, pc.picWidth >> 1 );
  }
}

// The residuals that the picture holds where the thread's predictions go (atY: luma, atC: chroma), when the block has
// any. Both chroma planes are read at the thread's place when either has a residual (the other is another thread's: no
// more memory traffic; and the branch is the same for all the threads), and the thread's picked after: a plane picked
// before reading would have its pointer read per thread, one more round trip.
void tileResiduals( int flags, uint atY, uint atC, inout uint2 resY, inout uint resCb, inout uint resCr )
{
  if( ( flags & FLAG_RES_Y ) != 0 )
  {
    resY = *(uint2*)at( pc.dst0, atY );
  }
  if( chroma420() && ( flags & ( ( FLAG_RES_Y << 1 ) | ( FLAG_RES_Y << 2 ) ) ) != 0 )
  {
    resCb = *(uint*)at( pc.dst1, atC );
    resCr = *(uint*)at( pc.dst2, atC );
  }
}

// --------------------------------------------------------------------------------------------------------------------
// The chroma of a tile in 4:2:2 and 4:4:4 (tile's own chroma code is 4:2:0's). Once the tile's luma is done, each
// chroma component's windows of the lists are read in chunks of 4 samples from a lead sample before the window (as tile
// reads 4:2:0's), the 2-D filters' horizontal passes run in groups of 4 samples, and each thread filters a group of 4
// samples of a row vertically from each list, combines the lists' predictions and writes them. Both components go in
// one round when their windows and passes fit the pool (all but 4:4:4's 16x16 tiles), else one after the other.
// --------------------------------------------------------------------------------------------------------------------

// q / d for 0 <= q < 256 and 0 < d < 256, without a division: q * ceil( 2^16 / d ) >> 16 is exact there (its error,
// below q / 2^16 < 1 / 256 < 1 / d, never reaches the next multiple of 1 / d)
int smallQuotient( int q, int d )
{
  return ( q * ( ( 65536 + d - 1 ) / d ) ) >> 16;
}

// Where a tile's chroma windows and horizontal passes lie in the pool, for both lists of the round's components
struct ChromaTileLayout
{
  int sx, sy;         // the chroma subsampling
  int cw, ch;         // the tile's chroma size
  int cols, rows;     // a window's size: (cw + 3) x (ch + 3)
  int rowChunks;      // the chunks of 4 samples of a stored window row (from a lead sample before the window)
  int rowLen;         // a stored row's samples (4 rowChunks: at least the lead sample and the window's row)
  int windowValues;   // a window's values in the pool: rows rowLen
  int passValues;     // a horizontal pass's: rows cw
  int comps;          // the components a round takes: 2, or 1 when both do not fit the pool
  int lgGroups;       // log2 of the groups of 4 samples of a row (cw / 4)
  int lgComp;         // log2 of the groups of a component (ch cw / 4)
};

ChromaTileLayout chromaTileLayout( int tw, int th )
{
  ChromaTileLayout L;
  L.sx           = chromaScaleX();
  L.sy           = chromaScaleY();
  L.cw           = tw >> L.sx;
  L.ch           = th >> L.sy;
  L.cols         = L.cw + 3;
  L.rows         = L.ch + 3;
  L.rowChunks    = ( 1 + L.cols + 3 ) / 4;
  L.rowLen       = 4 * L.rowChunks;
  L.windowValues = L.rows * L.rowLen;
  L.passValues   = L.rows * L.cw;
  L.comps        = 2 * 2 * ( L.windowValues + L.passValues ) <= kPoolValues ? 2 : 1;
  L.lgGroups     = log2Int( L.cw ) - 2;
  L.lgComp       = log2Int( L.ch ) + L.lgGroups;
  return L;
}

// The pool places of component k's (of the round's) window and horizontal pass of list slot s
int chromaWindowAt( ChromaTileLayout L, int k, int s )
{
  return ( 2 * k + s ) * L.windowValues;
}

int chromaPassAt( ChromaTileLayout L, int k, int s )
{
  return 2 * L.comps * L.windowValues + ( 2 * k + s ) * L.passValues;
}

// The 4 chroma predictions (j, i0 .. i0 + 3) of a list from its window (whose sample (0, 0) is at index win; rows of
// rowLen) and its horizontal pass (at pass; rows of cw), with chromaPred's arithmetic
int4 chromaGroupPred( ChromaTileLayout L, int win, int pass, int j, int i0, int xFrac, int yFrac, int16_t cx[4], int16_t cy[4], bool hi )
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
      for( int t = 0; t < 4; t++ )
      {
        sum += int( cy[t] ) * int( s_pool[pass + ( j + t ) * L.cw + i0 + o] );
      }
      out[o] = hi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
    }
    return out;
  }
  // (unfiltered, a sample's window row or column is the one after the first: the window starts a sample before the
  // motion's integer position)
  const int shiftV = hi ? 6 - headRoom : 6;
  const int offset = hi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
  if( xFrac != 0 )
  {
    int w[7];
    [unroll]
    for( int e = 0; e < 7; e++ )
    {
      w[e] = int( s_pool[win + ( j + 1 ) * L.rowLen + i0 + e] );
    }
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      int sum = 0;
      [unroll]
      for( int t = 0; t < 4; t++ )
      {
        sum += int( cx[t] ) * w[o + t];
      }
      const int v = ( sum + offset ) >> shiftV;
      out[o]      = hi ? v : clip3( 0, maxVal, v );
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
      for( int t = 0; t < 4; t++ )
      {
        sum += int( cy[t] ) * int( s_pool[win + ( j + t ) * L.rowLen + i0 + 1 + o] );
      }
      const int v = ( sum + offset ) >> shiftV;
      out[o]      = hi ? v : clip3( 0, maxVal, v );
    }
    return out;
  }
  [unroll]
  for( int o = 0; o < 4; o++ )
  {
    const int v = int( s_pool[win + ( j + 1 ) * L.rowLen + i0 + 1 + o] );
    out[o]      = hi ? v * ( 1 << headRoom ) - IF_INTERNAL_OFFS : v;
  }
  return out;
}

// Once a window's rows are stored, makes one of them hold what the filters read: the row's cols samples at s_pool[at],
// read from the picture's column x0 on, dIx columns right of the merge motion's window. In a border tile, the samples
// beside the picture take its edge's (tileBorderRow); where DMVR moved the integer position sideways, the samples past
// the merge motion's window take its edge column's (xPrefetchPad's padding).
void chromaFixRow( int at, int x0, int cols, int dIx, bool edges )
{
  if( edges )
  {
    tileBorderRow( at, x0, cols, pc.picWidth >> chromaScaleX() );
  }
  if( dIx > 0 )
  {
    const int16_t v = s_pool[at + cols - 1 - dIx];
    for( int c = cols - dIx; c < cols; c++ )
    {
      s_pool[at + c] = v;
    }
  }
  if( dIx < 0 )
  {
    const int16_t v = s_pool[at - dIx];
    for( int c = 0; c < -dIx; c++ )
    {
      s_pool[at + c] = v;
    }
  }
}

// What chromaTile needs of a tile: tile hands it to main, which calls chromaTile once for both of tile's instances
// (each call would add its own copy of chromaTile's code).
struct TileChroma
{
  int  tw, th;     // the tile's size (luma samples)
  int2 mvs[2];     // the lists' merge motion (their windows'), in tile's list slots
  int2 mvR[2];     // and their refined motion (their filters')
  int  refs[2];    // the lists' reference slots
  int  lists;      // 1 or 2
  bool hi;         // the predictions are combined (bi-prediction, weighted prediction): kept at 14 bits
  bool edges;      // a border tile (its windows reach past the picture's edges)
};

// Where each list's chroma window starts (per list slot s): one sample before the integer position of the motion. With
// DMVR that is the integer position of the refined motion, dI away from that of the merge motion. The window then keeps
// the rows of the merge motion's window (clamped into it), and chromaFixRow sets its columns past that window
// (xPrefetchPad's padding). A row is read from x0, after a lead sample when x0 is odd, so that the reads start 4-byte
// aligned. xFrac and yFrac are the refined motion's fractions.
struct ChromaWindows
{
  int  x0[2], y0[2], lead[2];
  int2 dI[2];
  int  xFrac[2], yFrac[2];
};

ChromaWindows chromaWindowsOf( int x, int y, ChromaTileLayout L, TileChroma tile )
{
  ChromaWindows cw;
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    const int2 mvs = tile.mvs[s];
    const int2 mvR = tile.mvR[s];
    cw.dI[s]       = int2( ( mvR.x >> ( 4 + L.sx ) ) - ( mvs.x >> ( 4 + L.sx ) ), ( mvR.y >> ( 4 + L.sy ) ) - ( mvs.y >> ( 4 + L.sy ) ) );
    cw.x0[s]       = ( x >> L.sx ) + ( mvs.x >> ( 4 + L.sx ) ) - 1 + cw.dI[s].x;
    cw.y0[s]       = ( y >> L.sy ) + ( mvs.y >> ( 4 + L.sy ) ) - 1;
    cw.lead[s]     = cw.x0[s] & 1;
    cw.xFrac[s]    = mvR.x & ( ( 16 << L.sx ) - 1 );
    cw.yFrac[s]    = mvR.y & ( ( 16 << L.sy ) - 1 );
  }
  return cw;
}

// Reads the round's windows (component k = w / lists, list slot s = w % lists of the round, from component first) in
// chunks of 4 samples, two a thread at most; all the reads are issued before the first store. A thread past the last
// chunk (or window) reads one that exists again and stores nothing. A border tile's rows are clamped into the picture,
// and a chunk wholly beside it is read from its first column instead (chromaFixRow sets those samples). A border tile's
// reads index from the margin sample 16 to the left of the plane's first sample, as tile's border reads do.
void readChromaWindows( int tid, ChromaTileLayout L, TileChroma tile, ChromaWindows cw, int first, out uint2 chunk[4][2] )
{
  const bool bi      = tile.lists == 2;
  const int  windows = L.comps * tile.lists;
  const int  chunks  = L.rows * L.rowChunks;
  const int  cHeight = pc.picHeight >> L.sy;
  if( tile.edges )
  {
    [unroll]
    for( int w = 0; w < 4; w++ )
    {
      const int wc = min( w, windows - 1 );
      const int s  = bi ? wc & 1 : 0;
      int16_t*  p  = uniPtr( refPlane( tile.refs[s], 1 + first + ( bi ? wc >> 1 : wc ) ) );
      [unroll]
      for( int i = 0; i < 2; i++ )
      {
        const int task = min( tid + THREADS * i, chunks - 1 );
        const int row  = smallQuotient( task, L.rowChunks );
        const int col  = cw.x0[s] - cw.lead[s] + 4 * ( task - row * L.rowChunks );
        const int c0   = col + 4 <= 0 || col >= ( pc.picWidth >> L.sx ) ? 0 : col;
        const int src  = clamp( cw.y0[s] + clamp( row + cw.dI[s].y, 0, L.rows - 1 ), 0, cHeight - 1 );
        chunk[w][i]    = load8( samplesBefore( p, 16 ), uint( src * pc.stride12 + c0 + 16 ) );
      }
    }
  }
  else
  {
    [unroll]
    for( int w = 0; w < 4; w++ )
    {
      const int wc = min( w, windows - 1 );
      const int s  = bi ? wc & 1 : 0;
      int16_t*  p  = uniPtr( refPlane( tile.refs[s], 1 + first + ( bi ? wc >> 1 : wc ) ) );
      [unroll]
      for( int i = 0; i < 2; i++ )
      {
        const int task = min( tid + THREADS * i, chunks - 1 );
        const int row  = smallQuotient( task, L.rowChunks );
        const int col  = cw.x0[s] - cw.lead[s] + 4 * ( task - row * L.rowChunks );
        const int src  = cw.y0[s] + clamp( row + cw.dI[s].y, 0, L.rows - 1 );
        chunk[w][i]    = load8( p, uint( src * pc.stride12 + col ) );
      }
    }
  }
}

// Stores the chunks readChromaWindows read into the round's windows in the pool
void storeChromaWindows( int tid, ChromaTileLayout L, TileChroma tile, uint2 chunk[4][2] )
{
  const bool bi      = tile.lists == 2;
  const int  windows = L.comps * tile.lists;
  const int  chunks  = L.rows * L.rowChunks;
  [unroll]
  for( int w = 0; w < 4; w++ )
  {
    const int kc = bi ? w >> 1 : w;
    const int s  = bi ? w & 1 : 0;
    [unroll]
    for( int i = 0; i < 2; i++ )
    {
      const int task = tid + THREADS * i;
      if( w < windows && task < chunks )
      {
        const int row = smallQuotient( task, L.rowChunks );
        storeFour( chromaWindowAt( L, kc, s ) + row * L.rowLen + 4 * ( task - row * L.rowChunks ), chunk[w][i] );
      }
    }
  }
}

// chromaFixRow for each row of the round's windows, a thread per window row
void fixChromaRows( int tid, ChromaTileLayout L, TileChroma tile, ChromaWindows cw )
{
  const bool bi      = tile.lists == 2;
  const int  windows = L.comps * tile.lists;
  for( int r = tid; r < windows * L.rows; r += THREADS )
  {
    const int w  = smallQuotient( r, L.rows );
    const int kc = bi ? w >> 1 : w;
    const int s  = bi ? w & 1 : 0;
    chromaFixRow( chromaWindowAt( L, kc, s ) + ( r - w * L.rows ) * L.rowLen + cw.lead[s], cw.x0[s], L.cols, cw.dI[s].x, tile.edges );
  }
}

// The 2-D filters' horizontal passes over the round's windows (those with both fractions), in groups of 4 samples of
// a window row, at 14 bits; cx holds each list's horizontal filter
void chromaPasses( int tid, ChromaTileLayout L, TileChroma tile, ChromaWindows cw, int16_t cx[2][4] )
{
  const bool bi       = tile.lists == 2;
  const int  windows  = L.comps * tile.lists;
  const int  headRoom = max( 2, 14 - pc.bitDepth );
  const int  shift1   = 6 - headRoom;
  const int  offset1  = -IF_INTERNAL_OFFS * ( 1 << shift1 );
  [unroll]
  for( int w = 0; w < 4; w++ )
  {
    const int kc = bi ? w >> 1 : w;
    const int s  = bi ? w & 1 : 0;
    if( w < windows && cw.xFrac[s] != 0 && cw.yFrac[s] != 0 )
    {
      const int win  = chromaWindowAt( L, kc, s ) + cw.lead[s];
      const int pass = chromaPassAt( L, kc, s );
      for( int g = tid; g < L.rows << L.lgGroups; g += THREADS )
      {
        const int r  = g >> L.lgGroups;
        const int gx = ( g & ( ( 1 << L.lgGroups ) - 1 ) ) << 2;
        int       v[7];
        [unroll]
        for( int e = 0; e < 7; e++ )
        {
          v[e] = int( s_pool[win + r * L.rowLen + gx + e] );
        }
        [unroll]
        for( int o = 0; o < 4; o++ )
        {
          int sum = 0;
          [unroll]
          for( int t = 0; t < 4; t++ )
          {
            sum += int( cx[s][t] ) * v[o + t];
          }
          s_pool[pass + r * L.cw + gx + o] = int16_t( ( sum + offset1 ) >> shift1 );
        }
      }
    }
  }
}

// The thread's group of 4 samples of component kc (k of the round), row j, from sample i0: each list's prediction,
// combined (BCW's or weighted prediction's weights, or GPM's blend), plus the residual res, written at atC
void writeChromaGroup( uint* job, int x, int y, int cuX, int cuY, int flags, ChromaTileLayout L, TileChroma tile, ChromaWindows cw,
                       int16_t cx[2][4], int16_t cy[2][4], int k, int kc, int j, int i0, uint atC, uint2 res )
{
  const int  maxVal = ( 1 << pc.bitDepth ) - 1;
  const int  bcwW1  = int( ( job[7] >> 16 ) & 15 ) - 2;   // BCW's L1 weight
  const bool geo    = ( flags & FLAG_GEO ) != 0;
  int4       pred[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    pred[s] = int4( 0, 0, 0, 0 );
    if( s < tile.lists )
    {
      pred[s] = chromaGroupPred( L, chromaWindowAt( L, k, s ) + cw.lead[s], chromaPassAt( L, k, s ), j, i0, cw.xFrac[s], cw.yFrac[s], cx[s], cy[s], tile.hi );
    }
  }
  const int4 resV = unpackSamples( res );
  int4       v;
  [unroll]
  for( int o = 0; o < 4; o++ )
  {
    // (GPM: the weight of the CU's luma sample under the chroma sample)
    const int gpmWeight = geo ? geoWeight( job[9], ( ( x >> L.sx ) + i0 + o - ( cuX >> L.sx ) ) << L.sx, ( ( y >> L.sy ) + j - ( cuY >> L.sy ) ) << L.sy ) : 0;
    v[o]                = combineLists( flags, 1 + kc, pred[0][o], pred[1][o], gpmWeight, job[9 + kc], job[11], bcwW1 );
  }
  if( ( flags & ( FLAG_RES_Y << ( 1 + kc ) ) ) != 0 )
  {
    v = clamp( v + resV, 0, maxVal );
  }
  int16_t* out = ( flags & ( FLAG_DEFER_CB << kc ) ) != 0 ? ( kc == 0 ? pc.defer1 : pc.defer2 ) : ( kc == 0 ? pc.dst1 : pc.dst2 );
  *(uint2*)at( out, atC ) = packSamples( v );
}

// The chroma of a tile at (x, y) in 4:2:2 or 4:4:4, with the job's flags; the whole workgroup calls it, after tile.
// Per round of components (both, or one when both do not fit the pool): the windows read, stored and fixed, the
// horizontal passes, then the thread's group of samples.
void chromaTile( int tid, uint* job, int x, int y, int cuX, int cuY, int flags, TileChroma tile )
{
  const ChromaTileLayout L     = chromaTileLayout( tile.tw, tile.th );
  const ChromaWindows    cw    = chromaWindowsOf( x, y, L, tile );
  const bool             moved = ( cw.dI[0].x | cw.dI[1].x ) != 0;
  // The thread's group of 4 samples: component k of the round, row j, samples i0 .. i0 + 3
  const int  k    = tid >> L.lgComp;
  const int  j    = ( tid >> L.lgGroups ) & ( L.ch - 1 );
  const int  i0   = ( tid & ( ( 1 << L.lgGroups ) - 1 ) ) << 2;
  const bool mine = k < L.comps;
  const uint atC  = uint( ( ( y >> L.sy ) + j ) * pc.stride12 + ( x >> L.sx ) + i0 );

  GroupMemoryBarrierWithGroupSync();   // (the luma is done with the pool)
  [loop]
  for( int first = 0; first < 2; first += L.comps )
  {
    uint2 chunk[4][2];
    readChromaWindows( tid, L, tile, cw, first, chunk );
    // The residuals where the thread's samples go: both planes read when either has one (the branch is the same for
    // all the threads), the thread's component's picked after
    uint2 resCb = uint2( 0, 0 ), resCr = uint2( 0, 0 );
    if( ( flags & ( ( FLAG_RES_Y << 1 ) | ( FLAG_RES_Y << 2 ) ) ) != 0 )
    {
      resCb = *(uint2*)at( pc.dst1, atC );
      resCr = *(uint2*)at( pc.dst2, atC );
    }
    storeChromaWindows( tid, L, tile, chunk );
    GroupMemoryBarrierWithGroupSync();
    if( tile.edges || moved )
    {
      fixChromaRows( tid, L, tile, cw );
      GroupMemoryBarrierWithGroupSync();
    }
    int16_t cx[2][4], cy[2][4];
    [unroll]
    for( int s = 0; s < 2; s++ )
    {
      [unroll]
      for( int t = 0; t < 4; t++ )
      {
        cx[s][t] = int16_t( kChroma[( cw.xFrac[s] << ( 1 - L.sx ) ) * 4 + t] );
        cy[s][t] = int16_t( kChroma[( cw.yFrac[s] << ( 1 - L.sy ) ) * 4 + t] );
      }
    }
    chromaPasses( tid, L, tile, cw, cx );
    GroupMemoryBarrierWithGroupSync();
    if( mine )
    {
      const int kc = first + k;   // (0: Cb, 1: Cr)
      writeChromaGroup( job, x, y, cuX, cuY, flags, L, tile, cw, cx, cy, k, kc, j, i0, atC, kc == 0 ? resCb : resCr );
    }
    GroupMemoryBarrierWithGroupSync();   // (the next round's windows replace these)
  }
}

// What tile's steps share: the tile, its lists, its windows' places in the planes and in shared memory, and the
// thread's samples. tileSetup works it out first.
struct TileSetup
{
  int  x, y;           // the tile's first sample (luma)
  int  tw, th;         // its size: 16x16 (FULL), else 16x8, 8x16 or 8x8
  int  flags;          // the job's FLAG_*
  bool edges;          // a border tile: its windows reach past the picture's edges (tileNear)
  bool wp;             // weighted prediction (wpSample), which starts from the 14-bit predictions, for a uni-prediction too
  bool geo;            // GPM: the two lists' predictions are the partitions', blended with the weights of the CU's mask
  bool hi;             // the predictions are kept at 14 bits for combining (bi-prediction, weighted prediction)
  int  lists;          // 1 or 2: slot 0 holds the first list used, slot 1 L1 when both are
  bool two;            // lists == 2
  int2 mvs[2];         // each slot's motion (with DMVR, the merge motion that the windows are read around)
  int  refs[2];        // each slot's reference picture
  // The windows' rows: of nch chunks (of 8 luma or 4 chroma samples) each, from a lead sample before the window when
  // it starts at an odd place (the reads start 4-byte aligned); rowsY luma rows (the tile's + 7) and rowsC chroma rows
  // (half the tile's + 3), yTasks and cTasks chunks of a window
  int  nch, rowsY, rowsC, yTasks, cTasks;
  int  cRem, cRow;                 // the thread's chroma chunk of a window (its row cRow; a thread past the last one
                                   // takes the last one, cRem)
  int  wx[2], wy[2], leadY[2];     // each slot's luma window: its first sample, and its lead
  int  cxs[2], cys[2], leadC[2];   // and its chroma windows'
  int  wyAt[2];                    // the luma windows' sample (0, 0) in shared memory
  int  wcAt[4];                    // the chroma windows': [2 slot + component - 1]
  // The thread's samples: luma (j, i0 .. i0 + 3) and chroma (cj, ci0 .. ci0 + 1) of component cc (0: Cb, 1: Cr), 2^lg
  // threads a row. In a tile of fewer than 16 rows the threads past its rows (or its chroma's) work on its last row
  // too, and store nothing (mineY, mineC).
  int  lg, j, i0, cc, cj, ci0;
  bool mineY, mineC;
  uint atY, atC;                   // the planes' places of the thread's luma and chroma samples
};

TileSetup tileSetup<let FULL : int>( int tid, int x, int y, int w, int h, int flags, int2 mv0, int2 mv1, int ref0, int ref1, bool bi,
                                    bool border )
{
  TileSetup t;
  t.x      = x;
  t.y      = y;
  t.tw     = FULL != 0 ? 16 : w;
  t.th     = FULL != 0 ? 16 : h;
  t.flags  = flags;
  t.edges  = FULL != 0 && border;
  t.wp     = ( flags & FLAG_WP ) != 0;
  t.geo    = ( flags & FLAG_GEO ) != 0;   // (a GPM tile has both lists)
  t.hi     = bi || t.wp;
  t.lists  = bi ? 2 : 1;
  t.two    = bi;
  // A 16x16 tile's luma chunks take two reads a thread per list (69 > 64).
  t.nch    = t.tw == 16 ? 3 : 2;
  t.rowsY  = t.th + 7;
  t.rowsC  = ( t.th >> 1 ) + 3;
  t.yTasks = t.nch * t.rowsY;
  t.cTasks = t.nch * t.rowsC;
  t.cRem   = min( tid, t.cTasks - 1 );
  t.cRow   = t.nch == 3 ? t.cRem / 3 : t.cRem >> 1;

  const bool l0 = ( flags & FLAG_L0 ) != 0;
  t.mvs[0]      = l0 ? mv0 : mv1;
  t.mvs[1]      = mv1;
  t.refs[0]     = l0 ? ref0 : ref1;
  t.refs[1]     = bi ? ref1 : ( l0 ? ref0 : ref1 );
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    t.wx[s]    = x + ( t.mvs[s].x >> 4 ) - 3;
    t.wy[s]    = y + ( t.mvs[s].y >> 4 ) - 3;
    t.leadY[s] = t.wx[s] & 1;
    t.cxs[s]   = ( x >> 1 ) + ( t.mvs[s].x >> 5 ) - 1;
    t.cys[s]   = ( y >> 1 ) + ( t.mvs[s].y >> 5 ) - 1;
    t.leadC[s] = t.cxs[s] & 1;
    t.wyAt[s]  = WY_AT( s ) + t.leadY[s];
  }
  t.wcAt[0] = WC_AT( 0 ) + t.leadC[0];
  t.wcAt[1] = WC_AT( 1 ) + t.leadC[0];
  t.wcAt[2] = WC_AT( 2 ) + t.leadC[1];
  t.wcAt[3] = WC_AT( 3 ) + t.leadC[1];

  t.lg            = t.tw == 16 ? 2 : 1;
  const int jAll  = tid >> t.lg;
  const int cjAll = ( tid & 31 ) >> t.lg;
  t.mineY         = FULL != 0 || jAll < t.th;
  t.mineC         = FULL != 0 || cjAll < ( t.th >> 1 );
  t.j             = FULL != 0 ? jAll : min( jAll, t.th - 1 );
  t.i0            = ( tid & ( ( 1 << t.lg ) - 1 ) ) << 2;
  t.cc            = tid >> 5;
  t.cj            = FULL != 0 ? cjAll : min( cjAll, ( t.th >> 1 ) - 1 );
  t.ci0           = ( tid & ( ( 1 << t.lg ) - 1 ) ) << 1;
  t.atY           = uint( ( y + t.j ) * pc.stride0 + x + t.i0 );
  t.atC           = uint( ( ( y >> 1 ) + t.cj ) * pc.stride12 + ( x >> 1 ) + t.ci0 );
  return t;
}

// All of the tile's reads, issued before the first wait for memory, in no branch but the residuals' after the windows'
// and the border tiles' (one of two whole sets of reads): after a branch the compiler makes the next read wait for all
// those before it. Each read takes one plane, the same in all the threads, so that its address is a scalar base and an
// offset (at()). Luma windows: rows of nch x 16 bytes per list (a 16x16 tile's: 23 rows of 24 samples from lead before
// the window, in two reads a thread per list; a thread past the last chunk reads the last one again, and stores
// nothing; with one list, the second list's reads are the first's again). Chroma windows (4:2:0's): rows of nch x 8
// bytes per component and list (a 16x16 tile's: 11 rows of 12 samples from lead before the window), a read a thread
// for each (Cb and Cr of the first list, then of the second). Then the residuals that the picture holds where the
// thread's samples go (resC: its chroma component's).
void tileIssueReads<let FULL : int>( int tid, TileSetup t, out uint4 yChunk[4], out uint2 cChunk[4], out uint2 resY, out uint resC )
{
  // The planes, read once for the workgroup (each load picks its own), as scalar bases for at().
  int16_t*  planeY[2]  = { uniPtr( refPlane( t.refs[0], 0 ) ), uniPtr( refPlane( t.refs[1], 0 ) ) };
  int16_t*  planeCb[2] = { uniPtr( refPlane( t.refs[0], 1 ) ), uniPtr( refPlane( t.refs[1], 1 ) ) };
  int16_t*  planeCr[2] = { uniPtr( refPlane( t.refs[0], 2 ) ), uniPtr( refPlane( t.refs[1], 2 ) ) };
  const int  cRem    = t.cRem;
  const int  cRow    = t.cRow;
  int16_t*   y1      = t.two ? planeY[1] : planeY[0];
  const uint yFirst0 = uint( t.wy[0] * pc.stride0 + t.wx[0] - t.leadY[0] );   // (the windows' first chunks)
  const uint yFirst1 = t.two ? uint( t.wy[1] * pc.stride0 + t.wx[1] - t.leadY[1] ) : yFirst0;
  int16_t*   cb1     = t.two ? planeCb[1] : planeCb[0];
  int16_t*   cr1     = t.two ? planeCr[1] : planeCr[0];
  uint       resCb   = 0, resCr = 0;
  resY               = uint2( 0, 0 );
  if( t.edges )
  {
    // A border tile: each row read is clamped into the picture (the rows past its top and bottom are its first and last
    // rows, as vvdec's border extension holds them), and a chunk wholly left or right of the picture is read from the
    // row's start instead (tileBorderLuma sets its samples); a chunk across an edge reaches 7 samples past it at most,
    // into the margin. (Chroma alike, with chunks of 4 samples.) The reads index from the margin's sample 16 left of the
    // plane's first sample: at() takes no negative index (as an offset of 2 GB, a chunk left of the first row faulted).
    // That sample's address is worked out as an integer (samplesBefore).
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      const int s   = k < 2 || !t.two ? 0 : 1;
      const int rem = min( tid + THREADS * ( k & 1 ), t.yTasks - 1 );
      const int row = rem / 3;
      const int col = t.wx[s] - t.leadY[s] + 8 * ( rem - 3 * row );
      const int c0  = col + 8 <= 0 || col >= pc.picWidth ? 0 : col;
      yChunk[k]     = load16( samplesBefore( planeY[s], 16 ),
                              uint( clamp( t.wy[s] + row, 0, pc.picHeight - 1 ) * pc.stride0 + c0 + 16 ) );
    }
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      if( chroma420() )
      {
        const int s   = k < 2 || !t.two ? 0 : 1;
        const int col = t.cxs[s] - t.leadC[s] + 4 * ( cRem - 3 * cRow );
        const int c0  = col + 4 <= 0 || col >= ( pc.picWidth >> 1 ) ? 0 : col;
        cChunk[k]     = load8( samplesBefore( ( k & 1 ) == 0 ? planeCb[s] : planeCr[s], 16 ),
                               uint( clamp( t.cys[s] + cRow, 0, ( pc.picHeight >> 1 ) - 1 ) * pc.stride12 + c0 + 16 ) );
      }
    }
    tileResiduals( t.flags, t.atY, t.atC, resY, resCb, resCr );
  }
  else
  {
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      if( FULL != 0 || ( k & 1 ) == 0 )
      {
        const int  rem = min( tid + THREADS * ( k & 1 ), t.yTasks - 1 );
        const int  row = t.nch == 3 ? rem / 3 : rem >> 1;
        const uint i   = ( k < 2 ? yFirst0 : yFirst1 ) + uint( row * pc.stride0 + 8 * ( rem - t.nch * row ) );
        yChunk[k]      = load16( k < 2 ? planeY[0] : y1, i );
      }
    }
    const uint cFirst0 = uint( t.cys[0] * pc.stride12 + t.cxs[0] - t.leadC[0] );
    const uint cFirst1 = t.two ? uint( t.cys[1] * pc.stride12 + t.cxs[1] - t.leadC[1] ) : cFirst0;
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      if( chroma420() )
      {
        const uint i = ( k < 2 ? cFirst0 : cFirst1 ) + uint( cRow * pc.stride12 + 4 * ( cRem - t.nch * cRow ) );
        cChunk[k]    = load8( k == 0 ? planeCb[0] : ( k == 1 ? planeCr[0] : ( k == 2 ? cb1 : cr1 ) ), i );
      }
    }
    tileResiduals( t.flags, t.atY, t.atC, resY, resCb, resCr );
  }
  resC = t.cc == 0 ? resCb : resCr;
}

// Stores the luma windows in shared memory, a chunk a store.
void tileStoreLumaWindows<let FULL : int>( int tid, TileSetup t, uint4 yChunk[4] )
{
  [unroll]
  for( int k = 0; k < 4; k++ )
  {
    const int rem = tid + THREADS * ( k & 1 );
    if( ( FULL != 0 || ( k & 1 ) == 0 ) && ( k < 2 || t.two ) && rem < t.yTasks )
    {
      const int row = t.nch == 3 ? rem / 3 : rem >> 1;
      storeEight( WY_AT( k >> 1 ) + row * YS + 8 * ( rem - t.nch * row ), yChunk[k] );
    }
  }
}

// Stores the chroma windows (4:2:0's) in shared memory, a chunk a store.
void tileStoreChromaWindows( int tid, TileSetup t, uint2 cChunk[4] )
{
  [unroll]
  for( int k = 0; k < 4; k++ )
  {
    if( chroma420() && ( k < 2 || t.two ) && tid < t.cTasks )
    {
      storeFour( WC_AT( k ) + t.cRow * CS + 4 * ( t.cRem - t.nch * t.cRow ), cChunk[k] );
    }
  }
}

// DMVR (xProcessDMVR) of a 16x16 tile, from its luma windows: the centre's SAD first. When it is small enough (83% of
// VideoQ's subblocks) there is no refinement, and neither the other 24 SADs nor most bilinear predictions are needed.
// Gives the refined motion (mvR, in the lists' slots) and, for BDOF, the best cost (cost). s_sad starts zeroed.
void tileDmvr( int tid, uint slot, TileSetup t, int2 mv0, int2 mv1, out int2 mvR[2], out int cost )
{
  const int shift = 4 - ( 10 - pc.bitDepth );
  // Rows 2, 4 .. 16 and columns 2 .. 17 of the bilinear predictions (the windows' samples from (3, 3)), 2 per thread
  int part = 0;
  [unroll]
  for( int k = 0; k < 2; k++ )
  {
    const int e  = tid + THREADS * k;
    const int rc = ( 3 + 2 * ( e >> 4 ) ) * YS + 3 + ( e & 15 );
    part += abs( bilinearW( t.wyAt[0] + rc, t.mvs[0].x & 15, t.mvs[0].y & 15, shift )
                 - bilinearW( t.wyAt[1] + rc, t.mvs[1].x & 15, t.mvs[1].y & 15, shift ) );
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
    // The bilinear predictions of the 20 x 20 samples from the merge motion minus 2 samples (the windows' samples
    // from (1, 1)), but their last row, which no SAD reads; then the other 24 SADs over every other row
    [unroll]
    for( int s = 0; s < 2; s++ )
    {
      const int at = t.wyAt[s] + YS + 1;
      for( int o = tid; o < DMVR_ROWS * 20; o += THREADS )
      {
        const int v = o / 20;
        s_pool[WORK_AT + DMVR_ROWS * 20 * s + o] = int16_t( bilinearW( at + v * YS + o - 20 * v, t.mvs[s].x & 15, t.mvs[s].y & 15, shift ) );
      }
    }
    GroupMemoryBarrierWithGroupSync();
    // (the 24 positions but the centre times 8 rows, 3 a thread)
    [unroll]
    for( int k = 0; k < 3; k++ )
    {
      const int p   = tid + THREADS * k;
      const int row = p / 24;
      const int pos = p - 24 * row;
      InterlockedAdd( s_sad[pos + ( pos >= 12 ? 1 : 0 )], int( sadRow( WORK_AT, WORK_AT + DMVR_ROWS * 20, 2 * row, pos + ( pos >= 12 ? 1 : 0 ) ) ) );
    }
    GroupMemoryBarrierWithGroupSync();
  }
  // Every thread works out the refinement itself (so none waits for another), from the SADs with the centre's
  // reduced by a quarter.
  const int2 delta = search ? dmvrDeltaFrom( centre ) : int2( 0, 0 );
  cost             = centre;
#if BDOF
  cost = search ? dmvrMinCost( centre ) : centre;
#endif
  // Threads 0 .. 15 write the refined motion where TMVP reads it (the 8x8 grid) into the colocated motion, which
  // mider.slang (GPU front end) or DecCu's storeDmvrMotion (CPU front end) stored before, one int each; threads
  // 16 and 17 write vvdec's DMVR MV cache (when it is wanted).
  if( tid < 16 )
  {
    const int x2 = ( ( t.x - 1 ) & ~7 ) + 8 + 8 * ( ( tid >> 2 ) & 1 );
    const int y2 = ( ( t.y - 1 ) & ~7 ) + 8 + 8 * ( tid >> 3 );
    if( x2 < t.x + 16 && y2 < t.y + 16 )
    {
      const int  mask = ( 1 << ( pc.ctuLog2 - 3 ) ) - 1;
      const int  ctu  = ( y2 >> pc.ctuLog2 ) * pc.widthInCtus + ( x2 >> pc.ctuLog2 );
      const int2 m    = ( tid & 2 ) == 0 ? mv0 + delta : mv1 - delta;
      pc.colMotion[( ctu * pc.num8x8 + ( ( x2 >> 3 ) & mask ) + ( ( ( y2 >> 3 ) & mask ) << ( pc.ctuLog2 - 3 ) ) ) * 5 + ( tid & 3 )] = ( tid & 1 ) == 0 ? m.x : m.y;
    }
  }
  else if( tid < 18 && pc.dmvr != nullptr )
  {
    pc.dmvr[int( slot ) * 2 + tid - 16] = tid == 16 ? delta.x : delta.y;
  }
  mvR[0] = clipStorage( mv0 + delta );
  mvR[1] = clipStorage( mv1 - delta );
}

// Whether DMVR moved either slot's integer position (in luma samples with fracBits 4, chroma's with 5), and by how much
// (dI: the refined motion's integer position less the merge motion's), the same in all the threads
bool integerMoved( int2 mvs[2], int2 mvR[2], int fracBits, out int2 dI[2] )
{
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    dI[s] = int2( ( mvR[s].x >> fracBits ) - ( mvs[s].x >> fracBits ), ( mvR[s].y >> fracBits ) - ( mvs[s].y >> fracBits ) );
  }
  return WaveReadLaneFirst( ( dI[0].x | dI[0].y | dI[1].x | dI[1].y ) != 0 ? 1 : 0 ) != 0;
}

// FULL: a 16x16 tile, else one of w x h (16x8, 8x16 or 8x8) without DMVR or BDOF. (Two functions: in each, the size's
// values are constants or nearly so.)
// border (FULL only): the windows reach past the picture's edges (tileNear). chroma: what chromaTile needs of the tile.
void tile<let FULL : int>( int tid, uint* job, int x, int y, int w, int h, int cuX, int cuY, int flags, int2 mv0, int2 mv1, int ref0, int ref1,
                           bool bi, bool dmvr, bool hpel, bool border, out TileChroma chroma )
{
  const TileSetup t        = tileSetup<FULL>( tid, x, y, w, h, flags, mv0, mv1, ref0, ref1, bi, border );
  const int       bcwW1    = int( ( job[7] >> 16 ) & 15 ) - 2;   // BCW's L1 weight
  // The job's words 8 .. 11 (DMVR's slot, GPM's and weighted prediction's parameters), read now, with the windows: read
  // where they are used, after the filters, each pair of them cost the tile a round trip of its own.
  const uint4     late     = uint4( job[8], job[9], job[10], job[11] );
  const int       maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int       headRoom = max( 2, 14 - pc.bitDepth );

  // All the reads, then the luma windows into shared memory (with a border tile's samples beside the picture set)
  uint4 yChunk[4];
  uint2 cChunk[4];
  uint2 resY;
  uint  resC;
  tileIssueReads<FULL>( tid, t, yChunk, cChunk, resY, resC );
  tileStoreLumaWindows<FULL>( tid, t, yChunk );
  if( FULL != 0 && tid < 25 )
  {
    s_sad[tid] = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  if( t.edges )
  {
    tileBorderLuma( tid, t.wyAt, t.wx, t.two );
    GroupMemoryBarrierWithGroupSync();
  }

  // DMVR's refined motion (without DMVR, BDOF applies wherever the tile's flag says)
  int2 mvR[2]    = { t.mvs[0], t.mvs[1] };
  int  dmvrCost  = 2 * 16 * 16;
  if( FULL != 0 && dmvr )
  {
    tileDmvr( tid, late.x, t, mv0, mv1, mvR, dmvrCost );
  }
#if BDOF
  const bool bdof = ( flags & FLAG_BDOF ) != 0 && dmvrCost >= 2 * 16 * 16;
#endif

  // Luma: where DMVR moved a list's integer position, the windows are rewritten to the moved position first
  // (tile16ShiftLuma); then the horizontal passes of the 2-D filters (a 16x16 tile's: 23 rows of 16 samples in groups
  // of 4).
  int2 dIY[2];
  if( integerMoved( t.mvs, mvR, 4, dIY ) )
  {
    tile16ShiftLuma( tid, t.wyAt, dIY );
  }
  int16_t cxY[2][8], cyY[2][8];
  int     rx0[2], ry0[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    const int xFrac = mvR[s].x & 15, yFrac = mvR[s].y & 15;
    [unroll]
    for( int k = 0; k < 8; k++ )
    {
      cxY[s][k] = int16_t( coeff( 0, xFrac, hpel, k ) );
      cyY[s][k] = int16_t( coeff( 0, yFrac, hpel, k ) );
    }
    // (a filter's taps start at the window's first column or row; unfiltered, the samples start at the motion's integer
    // position, the window's sample 3)
    rx0[s] = xFrac != 0 ? 0 : 3;
    ry0[s] = yFrac != 0 ? 0 : 3;
    if( s < t.lists && xFrac != 0 && yFrac != 0 )
    {
      const int shift1 = 6 - headRoom;
      tileLumaH( tid, t.wyAt[s], WORK_AT + 368 * s, cxY[s], shift1, -IF_INTERNAL_OFFS * ( 1 << shift1 ), t.rowsY, t.lg );
    }
  }
  GroupMemoryBarrierWithGroupSync();
  int4 predY[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    if( s < t.lists )
    {
      predY[s] = lumaPred( t.wyAt[s], WORK_AT + 368 * s, t.j, t.i0, rx0[s], ry0[s], mvR[s].x & 15, mvR[s].y & 15, cxY[s], cyY[s], t.hi );
    }
  }
#if BDOF
  // BDOF: the border read now, from the luma windows; after the barrier, both predictions and the border into s_bdof
  // (the luma passes' part of the pool, free then) and the sums zeroed; the 4x4 blocks' sums after the next barrier,
  // and the corrected samples, each thread with its block's offset, after the one after.
  int16_t border[3];
  if( bdof )
  {
    tile16BdofBorder( tid, t.wyAt, mvR, border );
  }
#endif
  GroupMemoryBarrierWithGroupSync();   // the luma windows are read; the chroma windows replace them
#if BDOF
  if( bdof )
  {
    tile16BdofStore( tid, t.j, t.i0, predY, border );
    for( int i = tid; i < 16 * 5; i += THREADS )
    {
      s_bsum[i] = 0;
    }
  }
#endif

  // Chroma (4:2:0's): the windows into shared memory (with a border tile's samples beside the picture set, and moved
  // where DMVR moved a list's integer position in chroma samples), then the horizontal passes of the 2-D filters (a
  // 16x16 tile's: 11 rows of 8 in pairs).
  tileStoreChromaWindows( tid, t, cChunk );
  GroupMemoryBarrierWithGroupSync();
  if( t.edges )
  {
    if( chroma420() )
    {
      tileBorderChroma( tid, t.wcAt, t.cxs, t.two );
    }
    GroupMemoryBarrierWithGroupSync();
  }
  int2 dIC[2];
  if( chroma420() && integerMoved( t.mvs, mvR, 5, dIC ) )
  {
    tile16ShiftChroma( tid, t.wcAt, dIC );
  }
#if BDOF
  uint dg[4];   // (the own samples' gradient differences, for tile16BdofSample)
  if( bdof )
  {
    tile16BdofSums( t.j, tid & 3, predY, dg );
  }
#endif
  int16_t cxC[2][4], cyC[2][4];
  int     crx0[2], cry0[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    const int xFrac = mvR[s].x & 31, yFrac = mvR[s].y & 31;
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      cxC[s][k] = int16_t( kChroma[xFrac * 4 + k] );
      cyC[s][k] = int16_t( kChroma[yFrac * 4 + k] );
    }
    crx0[s] = xFrac != 0 ? 0 : 1;
    cry0[s] = yFrac != 0 ? 0 : 1;
    if( chroma420() && s < t.lists && xFrac != 0 && yFrac != 0 )
    {
      const int shift1 = 6 - headRoom;
      tileChromaH( tid, t.wcAt[2 * s], CH_AT + 88 * 2 * s, cxC[s], shift1, -IF_INTERNAL_OFFS * ( 1 << shift1 ), t.rowsC, t.lg );
    }
  }
  GroupMemoryBarrierWithGroupSync();
  // GPM: the weights of the thread's luma samples and of its chroma samples (at the luma sample under each), read now
  // so that the chroma filters run while they come
  int4 geoY = int4( 0, 0, 0, 0 );
  int2 geoC = int2( 0, 0 );
  if( t.geo )
  {
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      geoY[o] = geoWeight( late.y, x - cuX + t.i0 + o, y - cuY + t.j );
    }
    [unroll]
    for( int o = 0; o < 2; o++ )
    {
      if( chroma420() )
      {
        geoC[o] = geoWeight( late.y, ( ( x >> 1 ) + t.ci0 + o - ( cuX >> 1 ) ) << 1, ( ( y >> 1 ) + t.cj - ( cuY >> 1 ) ) << 1 );
      }
    }
  }
  int2 predC[2] = { int2( 0, 0 ), int2( 0, 0 ) };
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    if( chroma420() && s < t.lists )
    {
      predC[s] = chromaPred( WC_AT( 2 * s + t.cc ) + t.leadC[s], CH_AT + 88 * ( 2 * s + t.cc ), t.cj, t.ci0, crx0[s], cry0[s], mvR[s].x & 31,
                             mvR[s].y & 31, cxC[s], cyC[s], t.hi );
    }
  }

  // Combine the predictions, add the residuals and write the results.
  int4 vy;
  [unroll]
  for( int o = 0; o < 4; o++ )
  {
    vy[o] = combineLists( flags, 0, predY[0][o], predY[1][o], geoY[o], late.x, late.w, bcwW1 );
  }
#if BDOF
  if( bdof )
  {
    const int2 v = bdofOffset( ( t.j >> 2 ) * 4 + ( t.i0 >> 2 ) );
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      vy[o] = tile16BdofSample( predY, dg[o], o, v );
    }
  }
#endif
  if( ( flags & FLAG_LUMA_MAP ) != 0 )
  {
    vy = int4( pc.lmcsFwd[vy.x], pc.lmcsFwd[vy.y], pc.lmcsFwd[vy.z], pc.lmcsFwd[vy.w] );
  }
  if( ( flags & FLAG_RES_Y ) != 0 )
  {
    vy = clamp( vy + unpackSamples( resY ), 0, maxVal );
  }
  int16_t* outY = ( flags & FLAG_DEFER_Y ) != 0 ? pc.defer0 : pc.dst0;
  if( t.mineY )
  {
    *(uint2*)at( outY, t.atY ) = packSamples( vy );
  }
  int2 vc;
  [unroll]
  for( int o = 0; o < 2; o++ )
  {
    vc[o] = combineLists( flags, 1 + t.cc, predC[0][o], predC[1][o], geoC[o], t.cc == 0 ? late.y : late.z, late.w, bcwW1 );
  }
  if( ( flags & ( FLAG_RES_Y << ( 1 + t.cc ) ) ) != 0 )
  {
    vc = clamp( vc + int2( lo16( resC ), hi16( resC ) ), 0, maxVal );
  }
  // (each component's plane first, as a scalar, then the thread's: see the residuals)
  int16_t* outCb = uniPtr( ( flags & FLAG_DEFER_CB ) != 0 ? pc.defer1 : pc.dst1 );
  int16_t* outCr = uniPtr( ( flags & ( FLAG_DEFER_CB << 1 ) ) != 0 ? pc.defer2 : pc.dst2 );
  if( chroma420() && t.mineC )
  {
    *(uint*)at( t.cc == 0 ? outCb : outCr, t.atC ) = packPair( vc );
  }
  // (4:2:2 and 4:4:4: main runs chromaTile with this)
  chroma.tw    = t.tw;
  chroma.th    = t.th;
  chroma.mvs   = t.mvs;
  chroma.mvR   = mvR;
  chroma.refs  = t.refs;
  chroma.lists = t.lists;
  chroma.hi    = t.hi;
  chroma.edges = t.edges;
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
  if( wrapPps() )
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

// The tile and the CU of an affine pair, read from the pair's job words. The tile lies at (x, y) and is w x h; its CU
// lies at (cuX, cuY) and is cw x ch.
struct AffineArea
{
  int x, y, w, h;
  int cuX, cuY, cw, ch;
};

AffineArea affineArea( uint job[24] )
{
  AffineArea a;
  a.x   = int( job[0] & 0xffff );
  a.y   = int( job[0] >> 16 );
  a.w   = int( job[1] & 255 );
  a.h   = int( ( job[1] >> 8 ) & 255 );
  a.cuX = int( job[6] & 0xffff );
  a.cuY = int( job[6] >> 16 );
  a.cw  = int( job[12] & 255 );
  a.ch  = int( ( job[12] >> 8 ) & 255 );
  return a;
}

// Two flags of each list in the pair's word 12. affineProfOf tells whether PROF refines the list's prediction.
// affineSpreadOf tells whether the list's motion spreads too far for subblocks (isSubblockVectorSpreadOverLimit), in
// which case every subblock takes the motion of the CU's centre.
bool affineProfOf( uint job[24], int list )
{
  return ( ( job[12] >> ( 16 + list ) ) & 1 ) != 0;
}

bool affineSpreadOf( uint job[24], int list )
{
  return ( ( job[12] >> ( 18 + list ) ) & 1 ) != 0;
}

// Returns the motion of luma subblock (sx, sy) of the pair's CU (a holds the pair's tile and CU).
int2 affineMv( uint job[24], int list, AffineArea a, int sx, int sy )
{
  const int2 base = affineBase( job, list );
  const int2 hor  = affineHor( job, list );
  const int2 ver  = affineVer( job, list );
  const int2 pos  = affineSpreadOf( job, list ) ? int2( a.cw >> 1, a.ch >> 1 ) : int2( 2 + 4 * sx, 2 + 4 * sy );
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

// A list's luma prediction of the thread's four samples (row v of its 4x4 subblock sb), at 14 bits for a bi-prediction
// or PROF (hi), else final, from the subblock's 9 x 9 window at s_pool[at], rows stride apart (its (r, c) is the
// reference sample (r - 2, c - 2) from the subblock's integer position): the 6-tap filter vvdec uses for 4x4 blocks
// (kLuma4x4's first and last taps are zero). The 2-D filter is separable: the subblock's four threads filter the
// window's nine rows across together (thread v the rows v, v + 4 and, the first, 8) into s_ext (nine rows of four a
// subblock), and then each filters its row's four samples down from the six rows they need. (Each thread filtering its
// six rows itself filtered most rows four times.) All the threads call it, as it waits at barriers; a thread past the
// tile's subblocks takes the last one's (sb, the same values).
int4 affineLumaFilter( int at, int stride, int sb, int v, int xFrac, int yFrac, bool hi )
{
  const int  headRoom = max( 2, 14 - pc.bitDepth );
  const int  maxVal   = ( 1 << pc.bitDepth ) - 1;
  const bool twoD     = xFrac != 0 && yFrac != 0;
  if( twoD )
  {
    const int shift1  = 6 - headRoom;
    const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
    [unroll]
    for( int k = 0; k < 3; k++ )
    {
      const int r = v + 4 * k;
      if( r < 9 )
      {
        int row[9];
        [unroll]
        for( int c = 0; c < 9; c++ )
        {
          row[c] = int( s_pool[at + r * stride + c] );
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
          s_ext[sb * 36 + r * 4 + o] = int16_t( ( hs + offset1 ) >> shift1 );
        }
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
  int4 pred;
  if( twoD )
  {
    // Down: the window's rows v .. v + 5 (the filter's rows 1 .. 6)
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      int sum = 0;
      [unroll]
      for( int r = 1; r < 7; r++ )
      {
        sum += kLuma4x4[yFrac * 8 + r] * int( s_ext[sb * 36 + ( v + r - 1 ) * 4 + o] );
      }
      pred[o] = hi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
    }
  }
  else if( xFrac == 0 && yFrac == 0 )
  {
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      const int s0 = int( s_pool[at + ( v + 2 ) * stride + o + 2] );
      pred[o]      = hi ? s0 * ( 1 << headRoom ) - IF_INTERNAL_OFFS : s0;
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
        sum += yFrac != 0 ? kLuma4x4[yFrac * 8 + c] * int( s_pool[at + ( v + c - 1 ) * stride + o + 2] )
                          : kLuma4x4[xFrac * 8 + c] * int( s_pool[at + ( v + 2 ) * stride + o + c - 1] );
      }
      const int r = ( sum + offset ) >> shiftV;
      pred[o]     = hi ? r : clip3( 0, maxVal, r );
    }
  }
  GroupMemoryBarrierWithGroupSync();   // (s_ext is read: PROF writes it next)
  return pred;
}

// PROF, which refines a list's luma prediction (pred, from affineLumaFilter with hi): each 4x4 block gets a border from
// the reference samples nearest its motion (in its window at s_pool[at], rows stride apart); then the gradients inside,
// and each sample moves along them by its offset from the subblock's motion (dMvScaleHor / Ver), at 14 bits for a
// bi-prediction, else final. All the threads call it (it waits at a barrier); mine: whether the thread has a subblock
// (sb, its row v).
void affineProf( uint job[24], int list, int sb, int v, bool mine, int at, int stride, int xFrac, int yFrac, bool bi, inout int4 pred )
{
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
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
      const int sm = int( s_pool[at + ( by + ( yFrac >> 3 ) + 2 ) * stride + bx + ( xFrac >> 3 ) + 2] );
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

// A list's luma prediction of the thread's four samples, at 14 bits for a bi-prediction or PROF, else final: the
// tile's 4x4 subblocks are shared out four threads each (the subblock tid / 4, its row tid % 4), and each thread
// predicts its row's four samples (affineLumaFilter, affineProf). Each subblock's filter reads 9 x 9 reference samples
// from 2 before the subblock's integer position, which its four threads load into shared memory together, a sample at
// a time. (The tiles whose windows fit in boxes take affineBoxTile instead.)
void affineLuma( int tid, uint job[24], int list, AffineArea a, bool bi, out int4 pred )
{
  pred              = int4( 0, 0, 0, 0 );
  const int  slot   = int( ( job[7] >> ( 8 * list ) ) & 255 );
  const int  nsx    = a.w >> 2;
  const int  nsb    = nsx * ( a.h >> 2 );
  const int  sb     = tid >> 2;   // the thread's subblock in the tile, and its row in it
  const int  v      = tid & 3;
  const bool mine   = sb < nsb;
  const int  ox     = ( a.x - a.cuX ) >> 2;   // the tile's first subblock in the CU
  const int  oy     = ( a.y - a.cuY ) >> 2;
  const int  sx     = ox + sb % nsx;      // the thread's subblock in the CU
  const int  sy     = oy + sb / nsx;
  if( refScaled( slot ) )
  {
    // A scaled reference (RPR): each 4x4 block by xPredInterBlkRPR with
    // filter index 2, without PROF.
    const RefPlanes r = pc.refs[slot];
    if( mine )
    {
      const int2 mv = affineClipScaled( affineMv( job, list, a, sx, sy ), a.cuX, a.cuY );
      [unroll]
      for( int o = 0; o < 4; o++ )
      {
        pred[o] = rprSample( rprGeometry( r, 0, a.cuX + 4 * sx, a.cuY + 4 * sy, mv, false, 2 ), r.y, 0, o, v, bi );
      }
    }
    return;
  }
  int16_t*   ref      = refPlane( slot, 0 );
  const int  at       = sb * 81;   // the subblock's window: (r, c) is the reference sample (r - 2, c - 2) from its position

  // The windows: the subblock's rows v, v + 4 (and 8) from the thread
  if( mine )
  {
    const int2 mv   = s_sbMv[list][sb];
    const bool wrap = wrapPps() && s_sbWrap[list][sb] != 0;   // (wrapPps(): constant in the WIDE variants)
    const int  bx   = a.cuX + 4 * sx + ( mv.x >> 4 ) - 2;
    const int  by   = a.cuY + 4 * sy + ( mv.y >> 4 ) - 2;
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
  const int  sbf   = min( sb, nsb - 1 );   // (a thread past the subblocks filters the last one's window again)
  const int2 mv    = s_sbMv[list][sbf];
  const int  xFrac = mv.x & 15;
  const int  yFrac = mv.y & 15;
  const bool prof  = affineProfOf( job, list );
  pred             = affineLumaFilter( sbf * 81, 9, sbf, v, xFrac, yFrac, bi || prof );
  if( !mine )
  {
    pred = int4( 0, 0, 0, 0 );
  }
  if( prof )
  {
    affineProf( job, list, sb, v, mine, at, 9, xFrac, yFrac, bi, pred );
  }
  GroupMemoryBarrierWithGroupSync();   // before the next writes to the shared memory
}

// The motion of the CU's chroma subblock (csx, csy) of 4 x 4 chroma samples (xPredAffineBlk's chromaMvFld), clipped
// by affineClip (wrap is affineClip's). In 4:4:4 it is the luma subblock's motion; otherwise the rounded average of
// two luma subblocks' motion: the two the chroma subblock covers (4:2:2), or the top-left and bottom-right of the
// four (4:2:0).
int2 affineChromaMv( uint job[24], int list, AffineArea a, int csx, int csy, out bool wrap )
{
  const int lx = a.cuX + ( ( 4 * csx ) << chromaScaleX() );   // the subblock's luma position
  const int ly = a.cuY + ( ( 4 * csy ) << chromaScaleY() );
  if( chromaScaleX() == 0 )
  {
    return affineClip( affineMv( job, list, a, csx, csy ), lx, ly, 4, a.cuX, a.cuY, wrap );
  }
  const int  sy  = csy << chromaScaleY();
  const int2 sum = affineMv( job, list, a, 2 * csx, sy ) + affineMv( job, list, a, 2 * csx + 1, sy + chromaScaleY() );
  return affineClip( int2( roundAffine( sum.x, 1 ), roundAffine( sum.y, 1 ) ), lx, ly, 4 << chromaScaleX(), a.cuX, a.cuY, wrap );
}

// A list's chroma prediction of the sample (u, v) of a 4x4 chroma subblock, at 14 bits for a bi-prediction, else final,
// from the subblock's 7 x 7 window at s_pool[at], rows stride apart (its (r, c) is the reference sample (r - 1, c - 1)
// from the subblock's integer position), with the filter's phases xFrac, yFrac (in 32nds).
int affineChromaSample( int at, int stride, int u, int v, int xFrac, int yFrac, bool bi )
{
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  if( xFrac == 0 && yFrac == 0 )
  {
    const int s0 = int( s_pool[at + ( v + 1 ) * stride + u + 1] );
    return bi ? s0 * ( 1 << headRoom ) - IF_INTERNAL_OFFS : s0;
  }
  if( xFrac != 0 && yFrac != 0 )
  {
    const int shift1  = 6 - headRoom;
    const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
    int       sum2    = 0;
    [unroll]
    for( int r = 0; r < 4; r++ )
    {
      int hs = 0;
      [unroll]
      for( int c = 0; c < 4; c++ )
      {
        hs += kChroma[xFrac * 4 + c] * int( s_pool[at + ( v + r ) * stride + u + c] );
      }
      sum2 += kChroma[yFrac * 4 + r] * ( ( hs + offset1 ) >> shift1 );
    }
    return bi ? sum2 >> 6 : clip3( 0, maxVal, ( sum2 + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
  }
  const int shiftV = bi ? 6 - headRoom : 6;
  const int offset = bi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
  int       sum2   = 0;
  [unroll]
  for( int c = 0; c < 4; c++ )
  {
    sum2 += yFrac != 0 ? kChroma[yFrac * 4 + c] * int( s_pool[at + ( v + c ) * stride + u + 1] )
                       : kChroma[xFrac * 4 + c] * int( s_pool[at + ( v + 1 ) * stride + u + c] );
  }
  const int r = ( sum2 + offset ) >> shiftV;
  return bi ? r : clip3( 0, maxVal, r );
}

// A list's chroma prediction of the tile for component comp: the thread's samples t = tid + THREADS k, at 14 bits
// for a bi-prediction, else final (affineChromaSample). Each subblock's 7 x 7 window is loaded a sample at a time.
void affineChroma( int tid, uint job[24], int list, int comp, AffineArea a, bool bi, out int pred[PER_THREAD] )
{
  const int slot = int( ( job[7] >> ( 8 * list ) ) & 255 );
  if( refScaled( slot ) )
  {
    // A scaled reference (RPR): each chroma 4x4 block by xPredInterBlkRPR with
    // filter index 2, its motion clipped as the luma's is.
    const RefPlanes r     = pc.refs[slot];
    const int       tw    = a.w >> chromaScaleX();
    const int       log2T = firstbithigh( uint( tw ) );
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int t = tid + THREADS * k;
      pred[k]     = 0;
      if( t < tw * ( a.h >> chromaScaleY() ) )
      {
        const int i   = t & ( tw - 1 );
        const int j   = t >> log2T;
        const int csx = ( ( ( a.x - a.cuX ) >> chromaScaleX() ) >> 2 ) + ( i >> 2 );
        const int csy = ( ( ( a.y - a.cuY ) >> chromaScaleY() ) >> 2 ) + ( j >> 2 );
        int2      mv;
        if( chromaScaleX() == 0 )
        {
          mv = affineMv( job, list, a, csx, csy );
        }
        else
        {
          const int  ly  = csy << chromaScaleY();
          const int2 sum = affineMv( job, list, a, 2 * csx, ly ) + affineMv( job, list, a, 2 * csx + 1, ly + chromaScaleY() );
          mv             = int2( roundAffine( sum.x, 1 ), roundAffine( sum.y, 1 ) );
        }
        mv      = affineClipScaled( mv, a.cuX, a.cuY );
        const RprGeometry g = rprGeometry( r, comp, ( a.cuX >> chromaScaleX() ) + 4 * csx, ( a.cuY >> chromaScaleY() ) + 4 * csy, mv, false, 2 );
        pred[k]             = rprSample( g, comp == 1 ? r.cb : r.cr, comp, i & 3, j & 3, bi );
      }
    }
    return;
  }
  int16_t*  ref      = refPlane( slot, comp );
  const int sx       = chromaScaleX();
  const int sy       = chromaScaleY();
  const int tw       = a.w >> sx;   // the tile's chroma size
  const int th       = a.h >> sy;
  const int ncx      = tw >> 2;   // the tile's chroma subblocks across
  const int ncb      = ncx * ( th >> 2 );
  const int ocx      = ( ( a.x - a.cuX ) >> sx ) >> 2;   // the tile's first chroma subblock in the CU
  const int ocy      = ( ( a.y - a.cuY ) >> sy ) >> 2;
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
    const bool wrap = wrapPps() && s_cbWrap[list][cb] != 0;   // (wrapPps(): constant in the WIDE variants)
    s_pool[s]       = int16_t( refAt( ref, comp, ( a.cuX >> sx ) + 4 * csx + ( mv.x >> ( 4 + sx ) ) - 1 + c, ( a.cuY >> sy ) + 4 * csy + ( mv.y >> ( 4 + sy ) ) - 1 + r, wrap ) );
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
    pred[k] = affineChromaSample( at, 7, i & 3, j & 3, xFrac, yFrac, bi );
  }
  GroupMemoryBarrierWithGroupSync();
}

// --------------------------------------------------------------------------------------------------------------------
// affineBoxTile: an affine tile whose windows lie in boxes that fit in shared memory, inside the picture (97-99% of
// the slower preset's affine tiles). The affine motion changes little across a tile, so its subblocks' windows (luma
// 9 x 9, chroma 7 x 7) overlap: a list's luma windows lie in a box of about 21 x 21 samples, and its chroma windows of
// a component in one of about 11 x 11. All the tile's reads (the boxes in chunks of 16 and 8 bytes, as tile reads its
// windows, and the residuals) are issued at once; the filters then read each subblock's window from its box in
// shared memory (affineLumaFilter, affineProf, affineChromaSample, as the subblock-by-subblock path reads its windows).
// --------------------------------------------------------------------------------------------------------------------
#define AB_ROWS 28                              // a list's luma box: at most 28 rows of AB_S samples (4 chunks of 8),
#define AB_S 32                                 // from a lead sample before the box (the lists' in turn)
#define AC_ROWS 14                              // a list's chroma box of a component: at most 14 rows of AC_S samples
#define AC_S 16                                 // (4 chunks of 4), from a lead sample before the box
#define AC_AT( b ) ( ( b ) * AC_ROWS * AC_S )   // b: 2 * slot + component - 1 (the chroma boxes replace the luma's)

// A box of windows: its first sample (x0, y0) from the tile's (in the component's samples), and its size
struct AffineBox
{
  int x0, y0, w, h;
};

// The box of the windows of a list's subblocks, from the motion of the corner subblocks m0 .. m3 (top-left, top-right,
// bottom-left, bottom-right; the last subblock lies right, bottom samples from the first). The windows' integer
// positions (motion >> fracBits, plus the subblock's place) are a rounded linear function of the subblock across and
// down, so they are largest and smallest at the corners. (Where affineClip clipped the motion they need not be, but
// then the box does not lie inside the picture.) A window starts before samples before its subblock's integer position,
// and is side samples a side.
AffineBox affineBox( int2 m0, int2 m1, int2 m2, int2 m3, int right, int bottom, int fracBits, int before, int side )
{
  const int x0 = m0.x >> fracBits, x1 = right + ( m1.x >> fracBits ), x2 = m2.x >> fracBits, x3 = right + ( m3.x >> fracBits );
  const int y0 = m0.y >> fracBits, y1 = m1.y >> fracBits, y2 = bottom + ( m2.y >> fracBits ), y3 = bottom + ( m3.y >> fracBits );
  AffineBox box;
  box.x0 = min( min( x0, x1 ), min( x2, x3 ) ) - before;
  box.y0 = min( min( y0, y1 ), min( y2, y3 ) ) - before;
  box.w  = max( max( x0, x1 ), max( x2, x3 ) ) - before + side - box.x0;
  box.h  = max( max( y0, y1 ), max( y2, y3 ) ) - before + side - box.y0;
  return box;
}

// Whether a box of the tile at (x, y) lies inside its plane (width x height samples), and its rows fit in rowSamples
// from a lead sample, maxRows of them.
bool affineBoxFits( AffineBox box, int x, int y, int width, int height, int rowSamples, int maxRows )
{
  const int bx = x + box.x0, by = y + box.y0;
  return bx >= 0 && by >= 0 && bx + box.w <= width && by + box.h <= height && box.w + ( bx & 1 ) <= rowSamples && box.h <= maxRows;
}

// The row of chunk t of a box whose rows are n chunks (2, 3 or 4)
int chunkRow( int t, int n )
{
  return n == 4 ? t >> 2 : ( n == 2 ? t >> 1 : t / 3 );
}

// The boxes of an affine tile's lists, in slots: slot 0 takes the first list the tile uses, slot 1 list 1 when the tile
// uses both (with one list, slot 1 repeats slot 0)
struct AffineTileBoxes
{
  int       lists;       // the lists the tile uses: 1 or 2
  int       listOf[2];   // the list of each slot
  AffineBox yBox[2];     // a slot's luma box
  AffineBox cBox[2];     // and its chroma box (4:2:0)
};

// Works out the tile's boxes from its corner subblocks' motion (s_sbMv, s_cbMv); false where a box does not fit
bool affineTileBoxes( int x, int y, int w, int h, int flags, bool bi, out AffineTileBoxes b )
{
  const int nsx = w >> 2;   // the luma subblocks across and down
  const int nsy = h >> 2;
  const int ncx = w >> 3;   // the chroma subblocks (4:2:0)
  const int ncy = h >> 3;
  const bool l0 = ( flags & FLAG_L0 ) != 0;
  b.lists       = bi ? 2 : 1;
  b.listOf[0]   = l0 ? 0 : 1;
  b.listOf[1]   = 1;
  bool fits     = true;
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    const int list = s < b.lists ? b.listOf[s] : b.listOf[0];   // (with one list, slot 1 repeats slot 0)
    b.yBox[s]      = affineBox( s_sbMv[list][0], s_sbMv[list][nsx - 1], s_sbMv[list][nsx * ( nsy - 1 )], s_sbMv[list][nsx * nsy - 1], 4 * ( nsx - 1 ),
                                4 * ( nsy - 1 ), 4, 2, 9 );
    b.cBox[s]      = affineBox( s_cbMv[list][0], s_cbMv[list][ncx - 1], s_cbMv[list][ncx * ( ncy - 1 )], s_cbMv[list][ncx * ncy - 1], 4 * ( ncx - 1 ),
                                4 * ( ncy - 1 ), 5, 1, 7 );
    fits           = fits && affineBoxFits( b.yBox[s], x, y, pc.picWidth, pc.picHeight, AB_S, AB_ROWS )
           && affineBoxFits( b.cBox[s], x >> 1, y >> 1, pc.picWidth >> 1, pc.picHeight >> 1, AC_S, AC_ROWS );
  }
  return fits;
}

// An affine tile's reads of its boxes and where they lie: per slot, the luma box's lead sample, chunks a row and chunks
// in all, and the chroma box's; then the chunks this thread read (yChunk[2 * slot + k], cChunk[2 * slot + component - 1])
struct AffineTileReads
{
  int   yLead[2], yNch[2], yTasks[2], cLead[2], cNch[2], cTasks[2];
  uint4 yChunk[4];
  uint2 cChunk[4];
};

// All of the tile's box reads, none in a branch (as tile's). Each read takes one plane, the same in all the threads, as
// a scalar base for at(). Luma boxes: rows of n chunks of 16 bytes from the lead sample, two reads a thread per list (a
// thread past the last chunk reads the last one again, and stores nothing). Chroma boxes: rows of chunks of 8 bytes, a
// read a thread per component and list.
AffineTileReads affineTileReads( int tid, uint job[24], int x, int y, AffineTileBoxes b )
{
  AffineTileReads r;
  int16_t*        planeY[2];
  int16_t*        planeCb[2];
  int16_t*        planeCr[2];
  uint            yFirst[2], cFirst[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    const int list = s < b.lists ? b.listOf[s] : b.listOf[0];
    const int slot = int( ( job[7] >> ( 8 * list ) ) & 255 );
    planeY[s]      = uniPtr( refPlane( slot, 0 ) );
    planeCb[s]     = uniPtr( refPlane( slot, 1 ) );
    planeCr[s]     = uniPtr( refPlane( slot, 2 ) );
    const int bx   = x + b.yBox[s].x0, by = y + b.yBox[s].y0;
    r.yLead[s]     = bx & 1;
    r.yNch[s]      = ( b.yBox[s].w + r.yLead[s] + 7 ) >> 3;
    r.yTasks[s]    = r.yNch[s] * b.yBox[s].h;
    yFirst[s]      = uint( by * pc.stride0 + bx - r.yLead[s] );
    const int cx   = ( x >> 1 ) + b.cBox[s].x0, cy = ( y >> 1 ) + b.cBox[s].y0;
    r.cLead[s]     = cx & 1;
    r.cNch[s]      = ( b.cBox[s].w + r.cLead[s] + 3 ) >> 2;
    r.cTasks[s]    = r.cNch[s] * b.cBox[s].h;
    cFirst[s]      = uint( cy * pc.stride12 + cx - r.cLead[s] );
  }
  [unroll]
  for( int k = 0; k < 4; k++ )
  {
    const int s   = k >> 1;
    const int c   = min( tid + THREADS * ( k & 1 ), r.yTasks[s] - 1 );
    const int row = chunkRow( c, r.yNch[s] );
    r.yChunk[k]   = load16( planeY[s], yFirst[s] + uint( row * pc.stride0 + 8 * ( c - r.yNch[s] * row ) ) );
  }
  [unroll]
  for( int k = 0; k < 4; k++ )
  {
    const int s   = k >> 1;
    const int c   = min( tid, r.cTasks[s] - 1 );
    const int row = chunkRow( c, r.cNch[s] );
    r.cChunk[k]   = load8( ( k & 1 ) != 0 ? planeCr[s] : planeCb[s], cFirst[s] + uint( row * pc.stride12 + 4 * ( c - r.cNch[s] * row ) ) );
  }
  return r;
}

// Where an affine tile's thread puts its predictions, and the residuals there: luma, row v of the subblock sb (a thread
// past the subblocks takes the last one's place, and stores nothing); chroma, (cj, ci0 .. ci0 + 1) of component cc,
// 2^lg threads a row (a thread past the rows takes the last one)
struct AffineTileThread
{
  int   sb, v, sbc;   // the luma subblock (sbc: the last one for a thread past them), the row in it
  bool  mine;         // the thread has a luma subblock
  uint  atY;          // where its luma predictions go
  int   cc, cj, ci0;  // its chroma component (0 Cb, 1 Cr), row and first sample
  bool  mineC;        // the thread has a chroma row
  uint  atC;          // where its chroma predictions go
  uint2 resY;         // the residuals there (zero without)
  uint  resC;
};

AffineTileThread affineTileThread( int tid, int x, int y, int w, int h, int flags )
{
  const int        nsx = w >> 2;
  const int        nsy = h >> 2;
  AffineTileThread p;
  p.sb             = tid >> 2;
  p.v              = tid & 3;
  p.mine           = p.sb < nsx * nsy;
  p.sbc            = min( p.sb, nsx * nsy - 1 );
  p.atY            = uint( ( y + 4 * ( p.sbc / nsx ) + p.v ) * pc.stride0 + x + 4 * ( p.sbc % nsx ) );
  const int lg     = w == 16 ? 2 : 1;
  p.cc             = tid >> 5;
  const int cjAll  = ( tid & 31 ) >> lg;
  p.mineC          = cjAll < ( h >> 1 );
  p.cj             = min( cjAll, ( h >> 1 ) - 1 );
  p.ci0            = ( tid & ( ( 1 << lg ) - 1 ) ) << 1;
  p.atC            = uint( ( ( y >> 1 ) + p.cj ) * pc.stride12 + ( x >> 1 ) + p.ci0 );
  p.resY           = uint2( 0, 0 );
  uint resCb = 0, resCr = 0;
  if( ( flags & FLAG_RES_Y ) != 0 )
  {
    p.resY = *(uint2*)at( pc.dst0, p.atY );
  }
  if( ( flags & ( ( FLAG_RES_Y << 1 ) | ( FLAG_RES_Y << 2 ) ) ) != 0 )
  {
    resCb = *(uint*)at( pc.dst1, p.atC );
    resCr = *(uint*)at( pc.dst2, p.atC );
  }
  p.resC = p.cc == 0 ? resCb : resCr;
  return p;
}

// Luma, a list at a time: its box into the pool (a chunk a store), then the thread's predictions from its subblock's
// window there (with PROF where the list has it)
void affineTileLuma( int tid, uint job[24], int w, bool hi, AffineTileBoxes b, AffineTileReads r, AffineTileThread p,
                     out int4 predY[2] )
{
  const int nsx = w >> 2;
  predY[0]      = int4( 0, 0, 0, 0 );
  predY[1]      = int4( 0, 0, 0, 0 );
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    if( s < b.lists )
    {
      [unroll]
      for( int k = 0; k < 2; k++ )
      {
        const int c = tid + THREADS * k;
        if( c < r.yTasks[s] )
        {
          const int row = chunkRow( c, r.yNch[s] );
          storeEight( row * AB_S + 8 * ( c - r.yNch[s] * row ), r.yChunk[2 * s + k] );
        }
      }
      GroupMemoryBarrierWithGroupSync();
      const int  list = b.listOf[s];
      const bool prof = affineProfOf( job, list );
      const int2 mv   = s_sbMv[list][p.sbc];
      // (the subblock's window: from 2 samples before its integer position)
      const int  at   = ( 4 * ( p.sbc / nsx ) + ( mv.y >> 4 ) - 2 - b.yBox[s].y0 ) * AB_S + 4 * ( p.sbc % nsx ) + ( mv.x >> 4 ) - 2 - b.yBox[s].x0
                       + r.yLead[s];
      int4       q    = affineLumaFilter( at, AB_S, p.sbc, p.v, mv.x & 15, mv.y & 15, hi || prof );
      if( prof )
      {
        affineProf( job, list, p.sbc, p.v, p.mine, at, AB_S, mv.x & 15, mv.y & 15, hi, q );
      }
      predY[s] = q;
      GroupMemoryBarrierWithGroupSync();   // the box is read: the next one replaces it
    }
  }
}

// The luma predictions averaged (with BCW's weights), mapped by LMCS, plus the residuals, written: only by the threads of
// the tile's subblocks. (A thread past them has no final predictions: affineProf leaves its 14-bit ones, which as LMCS's
// table index read far outside the table, and the GPU faulted.)
void affineTileLumaOut( uint job[24], int flags, int bcwW1, int maxVal, int4 predY[2], AffineTileThread p )
{
  if( p.mine )
  {
    int4 vy;
    [unroll]
    for( int o = 0; o < 4; o++ )
    {
      vy[o] = combineLists( flags, 0, predY[0][o], predY[1][o], 0, job[8], job[11], bcwW1 );
    }
    if( ( flags & FLAG_LUMA_MAP ) != 0 )
    {
      vy = int4( pc.lmcsFwd[vy.x], pc.lmcsFwd[vy.y], pc.lmcsFwd[vy.z], pc.lmcsFwd[vy.w] );
    }
    if( ( flags & FLAG_RES_Y ) != 0 )
    {
      vy = clamp( vy + unpackSamples( p.resY ), 0, maxVal );
    }
    *(uint2*)at( ( flags & FLAG_DEFER_Y ) != 0 ? pc.defer0 : pc.dst0, p.atY ) = packSamples( vy );
  }
}

// Chroma: the boxes into the pool (a chunk a store), then the thread's two samples of each list from its chroma
// subblock's window, averaged, plus the residuals, written
void affineTileChroma( int tid, uint job[24], int w, int flags, bool hi, int bcwW1, int maxVal, AffineTileBoxes b, AffineTileReads r,
                       AffineTileThread p )
{
  const int ncx = w >> 3;
  [unroll]
  for( int k = 0; k < 4; k++ )
  {
    const int s = k >> 1;
    if( s < b.lists && tid < r.cTasks[s] )
    {
      const int row = chunkRow( tid, r.cNch[s] );
      storeFour( AC_AT( k ) + row * AC_S + 4 * ( tid - r.cNch[s] * row ), r.cChunk[k] );
    }
  }
  GroupMemoryBarrierWithGroupSync();
  int2      predC[2] = { int2( 0, 0 ), int2( 0, 0 ) };
  const int cb       = ( p.cj >> 2 ) * ncx + ( p.ci0 >> 2 );   // the thread's chroma subblock
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    if( s < b.lists )
    {
      const int2 mv = s_cbMv[b.listOf[s]][cb];
      const int  at = AC_AT( 2 * s + p.cc ) + ( 4 * ( cb / ncx ) + ( mv.y >> 5 ) - 1 - b.cBox[s].y0 ) * AC_S + 4 * ( cb % ncx ) + ( mv.x >> 5 ) - 1
                     - b.cBox[s].x0 + r.cLead[s];
      [unroll]
      for( int o = 0; o < 2; o++ )
      {
        predC[s][o] = affineChromaSample( at, AC_S, ( p.ci0 & 3 ) + o, p.cj & 3, mv.x & 31, mv.y & 31, hi );
      }
    }
  }
  int2 vc;
  [unroll]
  for( int o = 0; o < 2; o++ )
  {
    vc[o] = combineLists( flags, 1 + p.cc, predC[0][o], predC[1][o], 0, job[9 + p.cc], job[11], bcwW1 );
  }
  if( ( flags & ( FLAG_RES_Y << ( 1 + p.cc ) ) ) != 0 )
  {
    vc = clamp( vc + int2( lo16( p.resC ), hi16( p.resC ) ), 0, maxVal );
  }
  if( p.mineC )
  {
    // (each component's plane first, as a scalar, then the thread's: see tileResiduals)
    int16_t* outCb = uniPtr( ( flags & FLAG_DEFER_CB ) != 0 ? pc.defer1 : pc.dst1 );
    int16_t* outCr = uniPtr( ( flags & ( FLAG_DEFER_CB << 1 ) ) != 0 ? pc.defer2 : pc.dst2 );
    *(uint*)at( p.cc == 0 ? outCb : outCr, p.atC ) = packPair( vc );
  }
}

// The tile from boxes, after affineMain worked out its subblocks' motion (s_sbMv, s_cbMv); returns false, having done
// nothing, where the boxes do not fit. hi: the predictions are kept at 14 bits (bi-prediction or weighted prediction).
bool affineBoxTile( int tid, uint job[24], int x, int y, int w, int h, int flags, bool bi, bool hi )
{
  AffineTileBoxes boxes;
  if( !affineTileBoxes( x, y, w, h, flags, bi, boxes ) )
  {
    return false;
  }
  const int              maxVal = ( 1 << pc.bitDepth ) - 1;
  const int              bcwW1  = int( ( job[7] >> 16 ) & 15 ) - 2;   // BCW's L1 weight
  const AffineTileReads  reads  = affineTileReads( tid, job, x, y, boxes );
  const AffineTileThread place  = affineTileThread( tid, x, y, w, h, flags );
  int4                   predY[2];
  affineTileLuma( tid, job, w, hi, boxes, reads, place, predY );
  affineTileLumaOut( job, flags, bcwW1, maxVal, predY, place );
  affineTileChroma( tid, job, w, flags, hi, bcwW1, maxVal, boxes, reads, place );
  return true;
}

// An affine pair's 24 words: its two jobs (jobs.slang's affineJobs)
void loadAffinePair( int index, out uint job[24] )
{
  [unroll]
  for( int k = 0; k < 6; k++ )
  {
    const uint4 q  = *(uint4*)( jobBase() + index * 24 + 4 * k );
    job[4 * k]     = q.x;
    job[4 * k + 1] = q.y;
    job[4 * k + 2] = q.z;
    job[4 * k + 3] = q.w;
  }
}

// The motion of the tile's subblocks per list, into s_sbMv / s_sbWrap (luma) and s_cbMv / s_cbWrap (chroma), a thread
// each: list tid / 32; luma subblock tid % 32, or chroma subblock tid % 32 - 16
void affineSubblockMotion( int tid, uint job[24], AffineArea a, int flags )
{
  const int list = tid >> 5;
  const int e    = tid & 31;
  if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) == 0 )
  {
    return;
  }
  const int nsx = a.w >> 2;
  const int ncx = ( a.w >> chromaScaleX() ) >> 2;
  bool      wrap;
  if( e < 16 && e < nsx * ( a.h >> 2 ) )
  {
    const int sx      = ( ( a.x - a.cuX ) >> 2 ) + e % nsx;
    const int sy      = ( ( a.y - a.cuY ) >> 2 ) + e / nsx;
    s_sbMv[list][e]   = affineClip( affineMv( job, list, a, sx, sy ), a.cuX + 4 * sx, a.cuY + 4 * sy, 4, a.cuX, a.cuY, wrap );
    s_sbWrap[list][e] = wrap ? 1 : 0;
  }
  else if( e >= 16 && numComponents() > 1 && e - 16 < ncx * ( ( a.h >> chromaScaleY() ) >> 2 ) )
  {
    const int c       = e - 16;
    const int csx     = ( ( ( a.x - a.cuX ) >> chromaScaleX() ) >> 2 ) + c % ncx;
    const int csy     = ( ( ( a.y - a.cuY ) >> chromaScaleY() ) >> 2 ) + c / ncx;
    s_cbMv[list][c]   = affineChromaMv( job, list, a, csx, csy, wrap );
    s_cbWrap[list][c] = wrap ? 1 : 0;
  }
}

// Writes the thread's four luma samples (of its subblock's row, as affineLuma predicted them): the lists' predictions
// averaged (with BCW's weights), mapped by LMCS, plus the residual
void writeAffineLuma( int tid, uint job[24], AffineArea a, int flags, int4 luma0, int4 luma1 )
{
  const int nsx = a.w >> 2;
  if( ( tid >> 2 ) >= nsx * ( a.h >> 2 ) )
  {
    return;
  }
  const int maxVal = ( 1 << pc.bitDepth ) - 1;
  const int bcwW1  = int( ( job[7] >> 16 ) & 15 ) - 2;   // BCW's L1 weight
  const int sb     = tid >> 2;
  const int row    = a.y + 4 * ( sb / nsx ) + ( tid & 3 );
  [unroll]
  for( int o = 0; o < 4; o++ )
  {
    const int at = row * pc.stride0 + a.x + 4 * ( sb % nsx ) + o;
    int       v  = combineLists( flags, 0, luma0[o], luma1[o], 0, job[8], job[11], bcwW1 );
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

// Chroma component comp of the tile: each list's prediction (affineChroma), combined (with BCW's weights), plus the
// residual, written; the thread's samples t = tid + THREADS k
void affineChromaComponent( int tid, uint job[24], AffineArea a, int flags, bool hi, int comp )
{
  const int maxVal = ( 1 << pc.bitDepth ) - 1;
  const int bcwW1  = int( ( job[7] >> 16 ) & 15 ) - 2;
  const int tw     = a.w >> chromaScaleX();   // the tile's chroma size
  const int th     = a.h >> chromaScaleY();
  const int log2C  = log2Int( tw );
  int       pred0[PER_THREAD], pred1[PER_THREAD];
  for( int list = 0; list < 2; list++ )
  {
    if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) == 0 )
    {
      continue;
    }
    int p[PER_THREAD];
    affineChroma( tid, job, list, comp, a, hi, p );
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
      const int at = ( ( a.y >> chromaScaleY() ) + ( t >> log2C ) ) * pc.stride12 + ( a.x >> chromaScaleX() ) + ( t & ( tw - 1 ) );
      int       v  = combineLists( flags, comp, pred0[k], pred1[k], 0, job[8 + comp], job[11], bcwW1 );
      if( ( flags & ( FLAG_RES_Y << comp ) ) != 0 )
      {
        v = clip3( 0, maxVal, v + int( dst[at] ) );
      }
      ( ( flags & ( FLAG_DEFER_CB << ( comp - 1 ) ) ) != 0 ? ( comp == 1 ? pc.defer1 : pc.defer2 ) : dst )[at] = int16_t( v );
    }
  }
}

// An affine pair's tile: the subblocks' motion, then the tile from boxes where they fit (nearly always; only in the
// WIDE variants, whose boxes read neither scaled references nor past a subpicture's or the wraparound's bounds), or
// else subblock by subblock: the lists' luma, combined and written, then each chroma component.
void affineMain( int tid, int index )
{
  uint job[24];
  loadAffinePair( index, job );
  const AffineArea area  = affineArea( job );
  const int        flags = int( job[1] >> 16 );
  setBounds( area.cuX, area.cuY );
  const bool bi = ( flags & ( FLAG_L0 | FLAG_L1 ) ) == ( FLAG_L0 | FLAG_L1 );

  affineSubblockMotion( tid, job, area, flags );
  GroupMemoryBarrierWithGroupSync();

  // (weighted prediction starts from the 14-bit predictions, for a uni-prediction too)
  const bool hi = bi || ( flags & FLAG_WP ) != 0;
  if( WIDE != 0 && affineBoxTile( tid, job, area.x, area.y, area.w, area.h, flags, bi, hi ) )
  {
    return;
  }
  int4 luma0 = int4( 0, 0, 0, 0 ), luma1 = int4( 0, 0, 0, 0 );
  for( int list = 0; list < 2; list++ )
  {
    if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) == 0 )
    {
      continue;
    }
    int4 p;
    affineLuma( tid, job, list, area, hi, p );
    if( list == 0 || ( flags & FLAG_L0 ) == 0 )
    {
      luma0 = p;
    }
    else
    {
      luma1 = p;
    }
  }
  writeAffineLuma( tid, job, area, flags, luma0, luma1 );
  for( int comp = 1; comp < numComponents(); comp++ )
  {
    affineChromaComponent( tid, job, area, flags, hi, comp );
  }
}
#endif

// An inter job (GpuJobs.h's InterJob, 12 words): the tile, its flags and motion, its CU, its references and BCW's
// weight
struct InterJob
{
  int  x, y, w, h;   // the tile, in luma samples
  int  flags;
  int2 mv0, mv1;
  int  cuX, cuY;     // the CU's corner (or that of SbTMVP's joined sub-PUs), where the motion is clipped
  int  cuW;          // and the clipping block's width
  int  ref0, ref1;   // the reference pictures' slots
  int  bcwW1;        // BCW's L1 weight
};

InterJob interJobOf( uint* job )
{
  InterJob j;
  j.x     = int( job[0] & 0xffff );
  j.y     = int( job[0] >> 16 );
  j.w     = int( job[1] & 255 );
  j.h     = int( ( job[1] >> 8 ) & 255 );
  j.flags = int( job[1] >> 16 );
  j.mv0   = int2( int( job[2] ), int( job[3] ) );
  j.mv1   = int2( int( job[4] ), int( job[5] ) );
  j.cuX   = int( job[6] & 0xffff );
  j.cuY   = int( job[6] >> 16 );
  j.ref0  = int( job[7] & 255 );
  j.ref1  = int( ( job[7] >> 8 ) & 255 );
  j.bcwW1 = int( ( job[7] >> 16 ) & 15 ) - 2;
  j.cuW   = int( job[7] >> 20 );
  return j;
}

// The tile paths: a 16 x 16 tile (with DMVR and BDOF, inside the picture or near its border), and without BDOF the
// tiles of 16 x 8, 8 x 16 and 8 x 8 (SbTMVP's pairs of sub-PUs and its sub-PUs, small CUs: half of the regular jobs
// of the slower preset's streams), which have no DMVR there; in 4:2:2 and 4:4:4 then the tile's chroma. Returns
// whether the job is done; otherwise the generic path makes it.
bool predictWithTiles( int tid, uint* job, InterJob j )
{
  const bool hpel  = ( j.flags & FLAG_HPEL ) != 0;
  const bool bi    = ( j.flags & ( FLAG_L0 | FLAG_L1 ) ) == ( FLAG_L0 | FLAG_L1 );
  const bool dmvr  = ( j.flags & FLAG_DMVR ) != 0;
  const bool tiles = tileJob( j.flags, j.ref0, j.ref1 );
  bool       tiled = false;
  TileChroma chroma;
  if( tiles && j.w == 16 && j.h == 16 )
  {
    const bool inside = tileInside( j.x, j.y, 16, 16, j.flags, j.mv0, j.mv1 );
    if( inside || ( borderTiles() && tileNear( j.x, j.y, j.flags, j.mv0, j.mv1 ) ) )
    {
      tile<1>( tid, job, j.x, j.y, 16, 16, j.cuX, j.cuY, j.flags, j.mv0, j.mv1, j.ref0, j.ref1, bi, dmvr, hpel, !inside, chroma );
      if( !needsChromaTile() )
      {
        return true;
      }
      tiled = true;
    }
  }
#if !BDOF
  if( !tiled && tiles && j.w >= 8 && j.h >= 8 && !dmvr && tileInside( j.x, j.y, j.w, j.h, j.flags, j.mv0, j.mv1 ) )
  {
    tile<0>( tid, job, j.x, j.y, j.w, j.h, j.cuX, j.cuY, j.flags, j.mv0, j.mv1, j.ref0, j.ref1, bi, false, hpel, false, chroma );
    if( !needsChromaTile() )
    {
      return true;
    }
    tiled = true;
  }
#endif
  // 4:2:2 and 4:4:4: the tile's chroma (one call for both of tile's instances)
  if( tiled )
  {
    chromaTile( tid, job, j.x, j.y, j.cuX, j.cuY, j.flags, chroma );
  }
  return tiled;
}

// DMVR: the bilinear predictions of the subblock's (w + 4) x (h + 4) samples from the merge motion minus 2 samples
// (with the PPS's wraparound, from the wraparound copy), into s_pool at BIL0_AT and BIL1_AT (rows of 20). The
// bilinear filters read (w + 5) x (h + 5) samples from each list's start.
void dmvrBilinear( int tid, InterJob j, int2 start0, int2 start1 )
{
  const int bw  = j.w + 4;
  const int x00 = j.x + ( start0.x >> 4 ), y00 = j.y + ( start0.y >> 4 );
  const int x01 = j.x + ( start1.x >> 4 ), y01 = j.y + ( start1.y >> 4 );
  if( x00 >= g_bounds.x && x00 + j.w + 5 <= g_bounds.z + 1 && y00 >= g_bounds.y && y00 + j.h + 5 <= g_bounds.w + 1 && x01 >= g_bounds.x
      && x01 + j.w + 5 <= g_bounds.z + 1 && y01 >= g_bounds.y && y01 + j.h + 5 <= g_bounds.w + 1 )
  {
    // Inside the picture (most subblocks): no clamping.
    int16_t* src0 = refPlane( j.ref0, 0 ) + y00 * pc.stride0 + x00;
    int16_t* src1 = refPlane( j.ref1, 0 ) + y01 * pc.stride0 + x01;
    // The thread's sample (u, v) is at index at in the reference pictures and at index out in shared memory (rows of
    // 20); each step advances THREADS samples.
    const int stepV   = THREADS / bw;
    const int stepU   = THREADS - stepV * bw;
    const int stepAt  = stepV * pc.stride0 + stepU;
    const int stepOut = stepV * 20 + stepU;
    const int v       = tid / bw;
    int       u       = tid - v * bw;
    int       at      = v * pc.stride0 + u;
    int       out     = v * 20 + u;
    for( int s = tid; s < bw * ( j.h + 4 ); s += THREADS )
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
    const bool wrapB = wrapPps();
    for( int s = tid; s < bw * ( j.h + 4 ); s += THREADS )
    {
      const int v = s / bw;
      const int u = s - v * bw;
      s_pool[BIL0_AT + v * 20 + u] = int16_t( bilinearAt( j.ref0, j.cuX, j.cuY, start0, j.x - j.cuX + u, j.y - j.cuY + v, wrapB ) );
      s_pool[BIL1_AT + v * 20 + u] = int16_t( bilinearAt( j.ref1, j.cuX, j.cuY, start1, j.x - j.cuX + u, j.y - j.cuY + v, wrapB ) );
    }
  }
}

// DMVR: the SADs of the 25 offsets over every other row of the bilinear predictions (each step adds one row of one
// offset, summed up in shared memory), the refinement they choose (s_delta, and with BDOF the best cost, s_minCost),
// written for vvdec's TaskFinishMotionInfo (pc.dmvr) and where TMVP reads it (the colocated motion's 8x8 grid, which
// mider.slang or DecCu's storeDmvrMotion stored before).
void dmvrRefine( int tid, uint* job, InterJob j )
{
  if( tid < 25 )
  {
    s_sad[tid] = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  for( int p = tid; p < 25 * ( j.h >> 1 ); p += THREADS )
  {
    const int pos = p % 25;
    InterlockedAdd( s_sad[pos], dmvrSadRow( j.w, 2 * ( p / 25 ), pos % 5 - 2, pos / 5 - 2 ) );
  }
  GroupMemoryBarrierWithGroupSync();
  if( tid == 0 )
  {
    s_sad[12] -= s_sad[12] >> 2;
    const int2 delta = dmvrDelta( j.w, j.h );
#if BDOF
    // The best cost the search found (or the centre's, when the search stops at once)
    int minCost = s_sad[12];
    if( minCost >= j.w * j.h )
    {
      for( int i = 0; i < 25; i++ )
      {
        minCost = min( minCost, s_sad[i] );
      }
    }
    s_minCost = minCost;
#endif
    const int index8 = int( job[8] );
    if( pc.dmvr != nullptr )
    {
      pc.dmvr[index8 * 2]     = delta.x;
      pc.dmvr[index8 * 2 + 1] = delta.y;
    }
    s_delta = delta;
    const int2 m0   = j.mv0 + delta;
    const int2 m1   = j.mv1 - delta;
    const int  mask = ( 1 << ( pc.ctuLog2 - 3 ) ) - 1;
    for( int y2 = ( ( j.y - 1 ) & ~7 ) + 8; y2 < j.y + j.h; y2 += 8 )
    {
      for( int x2 = ( ( j.x - 1 ) & ~7 ) + 8; x2 < j.x + j.w; x2 += 8 )
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
}

// A list's prediction of component comp's block (bx, by, bw x bh), from the list's (refined) motion mv and its merge
// motion: at 14 bits (hi) or final. With DMVR, the motion is clipped at the subblock (xFinalPaddedMCForDMVR), and when
// the refinement moves the integer position the reads stay in the prefetched window of the merge motion
// (xPrefetchPad); otherwise the motion is clipped at the CU (xPredInterUni). A scaled reference (RPR) is predicted
// sample by sample, positioned from the CU's origin. The window and the clipped motion are returned for BDOF.
void predictListBlock( int tid, InterJob j, int comp, int slot, int2 merge, int2 mv, bool hi, out bool windowed, out int4 window,
                       out int2 base, out int2 clipped, out bool wrap, out int pred[PER_THREAD] )
{
  const bool hpel   = ( j.flags & FLAG_HPEL ) != 0;
  const bool dmvr   = ( j.flags & FLAG_DMVR ) != 0;
  const int  sx     = comp == 0 ? 0 : chromaScaleX();
  const int  sy     = comp == 0 ? 0 : chromaScaleY();
  const int  bx     = j.x >> sx, by = j.y >> sy, bw = j.w >> sx, bh = j.h >> sy;
  const int  shiftX = 4 + sx;   // the motion's fraction bits in the component's samples
  const int  shiftY = 4 + sy;
  windowed          = false;
  window            = int4( 0, 0, 0, 0 );
  base              = int2( 0, 0 );
  if( dmvr )
  {
    // (with the PPS's wraparound, the reads come from the wraparound copy, unless wrapClipMv moves the window's motion)
    clipped = clipMv( mv, j.x, j.y, j.w );
    wrap    = wrapPps();
    const int2 deltaInt = int2( ( mv.x >> shiftX ) - ( merge.x >> shiftX ), ( mv.y >> shiftY ) - ( merge.y >> shiftY ) );
    if( deltaInt.x != 0 || deltaInt.y != 0 )
    {
      const int  half   = comp == 0 ? 3 : 1;
      const int2 before = merge - int2( half << shiftX, half << shiftY );
      int2       start;
      if( wrapPps() )
      {
        bool shifted;
        start = wrapClipMv( before, j.x, j.y, j.w, shifted );
        wrap  = !shifted;
      }
      else
      {
        start = clipMv( before, j.x, j.y, j.w );
      }
      window   = int4( bx + ( start.x >> shiftX ), by + ( start.y >> shiftY ), bw + 2 * half + 1, bh + 2 * half + 1 );
      base     = int2( window.x + half + deltaInt.x, window.y + half + deltaInt.y );
      windowed = true;
    }
  }
  else
  {
    // (the reads come from the wraparound copy with the SPS's wraparound)
    clipped = clipMv( mv, j.cuX, j.cuY, j.cuW );
    wrap    = wrapSps();
  }
  // One call per filter length keeps the shader's code small (it runs slower when it outgrows the instruction cache).
  if( refScaled( slot ) )
  {
    // (the motion is not clipped, as in xPredInterUni; no DMVR, no BDOF)
    rprBlock( tid, slot, comp, j.cuX >> sx, j.cuY >> sy, bx - ( j.cuX >> sx ), by - ( j.cuY >> sy ), bw, bh, mv, hi, hpel, pred );
  }
  else if( comp == 0 )
  {
    predictBlock<8>( tid, slot, comp, bx, by, bw, bh, clipped, hi, hpel, windowed, window, base, wrap, pred );
  }
  else
  {
    predictBlock<4>( tid, slot, comp, bx, by, bw, bh, clipped, hi, hpel, windowed, window, base, wrap, pred );
  }
}

// Writes component comp's block: the lists' predictions combined (with BCW's or weighted prediction's weights, or
// GPM's blend) or BDOF's output, mapped by LMCS (luma), plus the residual, into the picture or the deferred planes.
void storeComponent( int tid, uint* job, InterJob j, int comp, int pred0[PER_THREAD], int pred1[PER_THREAD], bool useBdof,
                     int bdofOut[PER_THREAD] )
{
  const int  sx       = comp == 0 ? 0 : chromaScaleX();
  const int  sy       = comp == 0 ? 0 : chromaScaleY();
  const int  bx       = j.x >> sx, by = j.y >> sy, bw = j.w >> sx, bh = j.h >> sy;
  const int  maxVal   = ( 1 << pc.bitDepth ) - 1;
  int16_t*   dst      = comp == 0 ? pc.dst0 : ( comp == 1 ? pc.dst1 : pc.dst2 );
  const int  stride   = comp == 0 ? pc.stride0 : pc.stride12;
  const bool residual = ( j.flags & ( FLAG_RES_Y << comp ) ) != 0;
  const bool map      = comp == 0 && ( j.flags & FLAG_LUMA_MAP ) != 0;
  const bool defer    = comp == 0 ? ( j.flags & FLAG_DEFER_Y ) != 0 : ( j.flags & ( FLAG_DEFER_CB << ( comp - 1 ) ) ) != 0;
  int16_t*   out      = defer ? ( comp == 0 ? pc.defer0 : ( comp == 1 ? pc.defer1 : pc.defer2 ) ) : dst;
  const int  log2W    = log2Int( bw );
  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int t = tid + THREADS * k;
    if( t >= bw * bh )
    {
      break;
    }
    const int i = t & ( bw - 1 );
    const int r = t >> log2W;
    // (GPM: the weight of the CU's luma sample under this sample)
    const int gpmWeight = ( j.flags & FLAG_GEO ) != 0 ? geoWeight( job[9], ( bx + i - ( j.cuX >> sx ) ) << sx, ( by + r - ( j.cuY >> sy ) ) << sy ) : 0;
    int       v         = useBdof ? bdofOut[k] : combineLists( j.flags, comp, pred0[k], pred1[k], gpmWeight, job[8 + comp], job[11], j.bcwW1 );
    const int at        = ( by + r ) * stride + bx + i;
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

// The generic path, for the jobs the tile paths do not take: DMVR's refinement, then per component each list's
// prediction (and BDOF's, for luma), combined and written.
void predictGeneric( int tid, uint* job, InterJob j )
{
  const bool bi   = ( j.flags & ( FLAG_L0 | FLAG_L1 ) ) == ( FLAG_L0 | FLAG_L1 );
  const bool dmvr = ( j.flags & FLAG_DMVR ) != 0;
  int2       refined0 = j.mv0, refined1 = j.mv1;
  if( dmvr )
  {
    dmvrBilinear( tid, j, clipMv( j.mv0, j.cuX, j.cuY, j.cuW ) - int2( 32, 32 ), clipMv( j.mv1, j.cuX, j.cuY, j.cuW ) - int2( 32, 32 ) );
    dmvrRefine( tid, job, j );
    refined0 = clipStorage( j.mv0 + s_delta );
    refined1 = clipStorage( j.mv1 - s_delta );
  }
#if BDOF
  // BDOF: with DMVR, only where the best cost is 2 w h or more (as in xProcessDMVR)
  const bool bdof = ( j.flags & FLAG_BDOF ) != 0 && ( !dmvr || s_minCost >= 2 * j.w * j.h );
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
  // (weighted prediction starts from the 14-bit predictions, for a uni-prediction too)
  const bool hi = bi || ( j.flags & FLAG_WP ) != 0;
  [loop]
  for( int comp = 0; comp < numComponents(); comp++ )
  {
    int pred0[PER_THREAD], pred1[PER_THREAD];
    [loop]
    for( int list = 0; list < 2; list++ )
    {
      if( ( j.flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) == 0 )
      {
        continue;
      }
      const int slot = list == 0 ? j.ref0 : j.ref1;
      bool      windowed;
      int4      window;
      int2      base, clipped;
      bool      wrap;
      int       pred[PER_THREAD];
      predictListBlock( tid, j, comp, slot, list == 0 ? j.mv0 : j.mv1, list == 0 ? refined0 : refined1, hi, windowed, window, base, clipped, wrap, pred );
      if( list == 0 || ( j.flags & FLAG_L0 ) == 0 )
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
        const int bx = j.x, by = j.y;
        bdofStore( tid, list, j.w, j.h, pred, slot, windowed ? base.x : bx + ( clipped.x >> 4 ), windowed ? base.y : by + ( clipped.y >> 4 ), clipped,
                   windowed, window, wrap );
      }
#endif
    }
    int bdofOut[PER_THREAD] = { 0, 0, 0, 0 };
#if BDOF
    if( bdof && comp == 0 )
    {
      GroupMemoryBarrierWithGroupSync();
      bdofApply( tid, j.w, j.h, bdofOut );
    }
#endif
    storeComponent( tid, job, j, comp, pred0, pred1, bdof && comp == 0, bdofOut );
  }
}

[shader( "compute" )]
[numthreads( THREADS, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int index = int( workgroupIndex( group ) );
  if( index >= ( pc.count >= 0 ? pc.count : int( pc.countAt[0] ) ) )
  {
    return;   // (the whole workgroup returns here)
  }
  const int  tid   = int( local.x );
#if AFFINE
  affineMain( tid, index );
  return;
#endif
  uint*          job = jobBase() + index * 12;
  const InterJob j   = interJobOf( job );
  setBounds( j.cuX, j.cuY );
  if( !predictWithTiles( tid, job, j ) )
  {
    predictGeneric( tid, job, j );
  }
}
