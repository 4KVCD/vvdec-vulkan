// Intra prediction, intra block copy (IBC) and the reconstruction of their
// transform blocks, as vvdec's IntraPrediction and DecCu::predAndReco do them.
// A workgroup predicts one transform block of one component, adds its residual
// (already in the picture, where vvdec's TrQuant left it) and writes the
// result.
//
// The blocks are grouped in waves: a block's wave comes after the waves of all
// blocks whose samples it reads. One dispatch reconstructs all the blocks of a
// picture. Each workgroup takes the next block in wave order (from a counter
// of blocks taken) and waits until all blocks of the earlier waves are done (a
// counter of blocks done). Because the blocks are taken in order, every block
// a workgroup waits for belongs to a workgroup that is already running and
// waits only for earlier blocks itself, so no workgroup ever waits for one
// that cannot run. (One dispatch per wave, each after a barrier, took 6.5 us
// per wave, and an intra-coded picture has about a thousand waves.)
//
// A wave lasts as long as its slowest workgroup, which mostly waits for
// memory. So a workgroup makes at most 256 samples (GpuRecon::recordIntra cuts
// big blocks into bands of rows), four per thread, and first issues all its
// reads of the picture together (the references, the residual, CCLM's luma,
// IBC's source); after that it computes from shared memory.
//
// A job is 12 uints (VulkanLib/GpuJobs.h, IntraJob):
//   0: x | y << 16                (in the component's samples)
//   1: width | height << 8 | component << 16 | kind << 24   (kind: 0 planar/DC/angular, 1 BDPCM, 2 CCLM, 3 IBC,
//      4 MIP, 5 the chroma of an inter CU under LMCS: inter's prediction, kept aside, plus the residual, 6 a
//      component of a CIIP CU: inter's prediction, kept aside, blended with planar's, plus the residual)
//   2: mode | flags << 8 | bdpcm direction << 16 | reference line << 24   (flags: 1 residual, 2 filtered references,
//      4 MIP transposed, 8 ISP; the reference line is MRL's, 0 .. 2)
//   3: top-left units | above units << 8 | left units << 16  (the available reference samples; for ISP, the CU's)
//   4: CCLM: luma above units | luma left units << 8 | bAbove << 16 | bLeft << 17 | aboveAvail << 18
//      | leftAvail << 19 | lm mode << 24
//      ISP: the CU's x | y << 16
//   5: CCLM: actual top template samples | actual left << 16
//      ISP: the CU's width | height << 8 | split << 16 (1 horizontal, 2 vertical) | left available << 18
//      | above available << 19
//   6: IBC: block vector x | y << 16 (int16, in the component's samples); CIIP: planar's weight (0 .. 3)
//   7: the rows of the block that this workgroup makes: first | count << 16, at
//      most 256 samples (on the CPU, this word holds the block's wave)
//   8: LMCS chroma residual scaling: x | y << 16 of the luma CU whose
//      neighbours give the scale
//   9: LMCS: 1 scaling | 2 left available | 4 above available | log2 of the samples per side << 4

#include "common.slang"
#include "intrajob.slang"

struct Params
{
  int16_t*      plane0;
  int16_t*      plane1;
  int16_t*      plane2;
  uint*         jobs;        // the picture's jobs, in wave order
  uint*         waits;       // per job: the number of jobs in earlier waves, which it waits for
  Atomic<uint>* sync;        // counters: jobs taken, jobs done (zeroed before the dispatch)
  uint8_t*      mip;         // MIP's matrices: vvdec's mipMatrix4x4, mipMatrix8x8 and mipMatrix16x16, one after another
  int*          lmcsScale;   // LMCS: the chroma residual scale for each average luma value
  int16_t*      defer0;      // LMCS, CIIP: inter's predictions, kept aside, laid out like the planes
  int16_t*      defer1;
  int16_t*      defer2;
  int           stride0;
  int           stride12;
  int           count;       // number of jobs; negative: countAt holds it
  int           bitDepth;
  int           ctuLog2;     // log2 of the CTU size in luma samples
  int           scaleX;      // chroma subsampling shifts
  int           scaleY;
  int           collocated;  // the SPS's CCLM collocated-chroma flag
  int           picWidth;    // luma samples
  int           picHeight;
  uint*         countAt;     // when count < 0: the number of jobs, from the GPU front end
};

[[vk::push_constant]] Params pc;

#define PLANAR_IDX 0
#define DC_IDX 1
#define HOR_IDX 18
#define DIA_IDX 34
#define VER_IDX 50
#define VDIA_IDX 66

#define MAX_REF 136   // 2 x 64 + 1, rounded up

static const int kAngTable[32] = { 0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 32, 35, 39, 45, 51, 57, 64, 73, 86, 102, 128, 171, 256, 341, 512, 1024 };
static const int kInvAngTable[32] = { 0, 16384, 8192, 5461, 4096, 2731, 2048, 1638, 1365, 1170, 1024, 910, 819, 712, 630, 565,
                                      512, 468, 420, 364, 321, 287, 256, 224, 191, 161, 128, 96, 64, 48, 32, 16 };
static const int kIntraFilterLuma[8] = { 24, 24, 24, 14, 2, 0, 0, 0 };
// InterpolationFilter's 4-tap chroma filter (the "cubic" one), 32 phases
static const int kCubic[128] = {
   0, 64,  0,  0,  -1, 63,  2,  0,  -2, 62,  4,  0,  -2, 60,  7, -1,  -2, 58, 10, -2,  -3, 57, 12, -2,  -4, 56, 14, -2,  -4, 55, 15, -2,
  -4, 54, 16, -2,  -5, 53, 18, -2,  -6, 52, 20, -2,  -6, 49, 24, -3,  -6, 46, 28, -4,  -5, 44, 29, -4,  -4, 42, 30, -4,  -4, 39, 33, -4,
  -4, 36, 36, -4,  -4, 33, 39, -4,  -4, 30, 42, -4,  -4, 29, 44, -5,  -4, 28, 46, -6,  -3, 24, 49, -6,  -2, 20, 52, -6,  -2, 18, 53, -5,
  -2, 16, 54, -4,  -2, 15, 55, -4,  -2, 14, 56, -4,  -2, 12, 57, -3,  -2, 10, 58, -2,  -1,  7, 60, -2,   0,  4, 62, -2,   0,  2, 63, -1
};
static const int kGauss[128] = {
  16, 32, 16,  0,  16, 32, 16,  0,  15, 31, 17,  1,  15, 31, 17,  1,  14, 30, 18,  2,  14, 30, 18,  2,  13, 29, 19,  3,  13, 29, 19,  3,
  12, 28, 20,  4,  12, 28, 20,  4,  11, 27, 21,  5,  11, 27, 21,  5,  10, 26, 22,  6,  10, 26, 22,  6,   9, 25, 23,  7,   9, 25, 23,  7,
   8, 24, 24,  8,   8, 24, 24,  8,   7, 23, 25,  9,   7, 23, 25,  9,   6, 22, 26, 10,   6, 22, 26, 10,   5, 21, 27, 11,   5, 21, 27, 11,
   4, 20, 28, 12,   4, 20, 28, 12,   3, 19, 29, 13,   3, 19, 29, 13,   2, 18, 30, 14,   2, 18, 30, 14,   1, 17, 31, 15,   1, 17, 31, 15
};
static const int kDivSigTable[16] = { 0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0 };

int log2Int( int v )
{
  return firstbithigh( uint( v ) );
}

int16_t* planeOf( int comp )
{
  return comp == 0 ? pc.plane0 : ( comp == 1 ? pc.plane1 : pc.plane2 );
}

int strideOf( int comp )
{
  return comp == 0 ? pc.stride0 : pc.stride12;
}

int sampleAt( int comp, int x, int y )
{
  return coherentSample( planeOf( comp ), y * strideOf( comp ) + x );
}

// The reference samples, shared by the block's threads and laid out as vvdec's
// xFillReferenceSamples (and xFilterReferenceSamples) lay them out: t[0] is the
// top-left sample, t[1..] the row above (including the above-right part), l[0]
// the top-left sample again, and l[1..] the column to the left (including the
// below-left part). tf and lf hold the filtered references.
groupshared int t[MAX_REF];
groupshared int l[MAX_REF];
groupshared int tf[MAX_REF];
groupshared int lf[MAX_REF];
groupshared int  s_dc;      // DC's value
groupshared int3 s_model;   // CCLM's a, b, shift
groupshared int16_t s_res[256];    // the residual of the workgroup's samples, read from the picture
groupshared int16_t s_luma[256];   // CCLM: the downsampled luma of the workgroup's samples
groupshared int     s_tmpl[4];         // CCLM: the luma of the template samples that the model picks
groupshared int     s_cubic[128];      // kCubic, for luma's angular modes

// The reads of xFillReferenceSamples. Each thread loads its own reference
// samples, t[tid + 64 k] and l[tid + 64 k] (k < 3), from the reference line r
// rows above and r columns left of the block's neighbours (MRL: r =
// cu.multiRefIdx(), which is 0 except for luma's extra lines). t[0 .. r] and
// l[0 .. r] are the corner, t[1 + r + i] is the row above from x0 + i, and
// l[1 + r + i] the column to the left from y0 + i. Where neighbours are
// missing, the samples are padded as vvdec pads them: the code works out each
// element's source sample (or the middle value) first, so that all the reads
// can be issued together.
struct References
{
  int t[3];
  int l[3];
};

// The inputs of a block's xFillReferenceSamples: the block, the lengths of its
// references (the row above and the column to the left), the reference line,
// and how many units of neighbours are available.
struct RefFill
{
  int  comp;
  int  x0, y0;
  int  predSize, predHSize;
  int  r;
  int  nTL, nAbove, nLeft;
  int  tmpA, tmpL;   // the available samples above and to the left
  bool none, all;
  int2 pad;          // the sample that the corner and the missing parts copy when only some neighbours are available
};

RefFill refFill( int comp, int x0, int y0, int predSize, int predHSize, int nTL, int nAbove, int nLeft, int r )
{
  RefFill f;
  f.comp       = comp;
  f.x0         = x0;
  f.y0         = y0;
  f.predSize   = predSize;
  f.predHSize  = predHSize;
  f.r          = r;
  f.nTL        = nTL;
  f.nAbove     = nAbove;
  f.nLeft      = nLeft;
  const int unitW = comp == 0 ? 4 : ( 4 >> pc.scaleX );
  const int unitH = comp == 0 ? 4 : ( 4 >> pc.scaleY );
  f.tmpA       = min( nAbove * unitW, predSize );
  f.tmpL       = min( nLeft * unitH, predHSize );
  f.none       = nTL + nAbove + nLeft == 0;
  f.all        = nTL + nAbove + nLeft == predSize / unitW + predHSize / unitH + 1;
  // With a left neighbour, pad from the corner if the top-left is available,
  // else from the left column's first sample; without one, from the first
  // sample of the row above.
  f.pad = nLeft > 0 ? ( nTL != 0 ? int2( x0 - 1, y0 - 1 - r ) : int2( x0 - 1 - r, y0 ) ) : int2( x0, y0 - 1 - r );
  return f;
}

// The source sample of element j of t (in row y0 - 1 - r).
int2 topSource( RefFill f, int j )
{
  const int rowY = f.y0 - 1 - f.r;
  const int colX = f.x0 - 1 - f.r;
  if( f.all )
  {
    return int2( colX + j, rowY );
  }
  if( j <= f.r )
  {
    return f.nLeft > 0 && f.nTL != 0 ? int2( colX + j, rowY ) : f.pad;
  }
  return f.nAbove != 0 ? int2( f.x0 + min( j - 1 - f.r, f.tmpA - 1 ), rowY ) : f.pad;
}

// The source sample of element i of l (in column x0 - 1 - r).
int2 leftSource( RefFill f, int i )
{
  const int rowY = f.y0 - 1 - f.r;
  const int colX = f.x0 - 1 - f.r;
  if( f.all )
  {
    return int2( colX, rowY + i );
  }
  if( f.nLeft == 0 )
  {
    return f.pad;
  }
  if( i <= f.r )
  {
    return f.nTL != 0 ? int2( colX, rowY + i ) : f.pad;
  }
  return int2( colX, f.y0 + min( i - 1 - f.r, f.tmpL - 1 ) );
}

// A reference sample's value: its source sample's, or the middle value when there are no neighbours.
int fillValue( RefFill f, int2 at )
{
  return f.none ? 1 << ( pc.bitDepth - 1 ) : sampleAt( f.comp, at.x, at.y );
}

References loadReferences( int tid, RefFill f )
{
  References refs;
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int j = tid + 64 * k;
    const int i = tid + 64 * k;
    refs.t[k]   = j <= f.predSize + f.r ? fillValue( f, topSource( f, j ) ) : 0;
    refs.l[k]   = i <= f.predHSize + f.r ? fillValue( f, leftSource( f, i ) ) : 0;
  }
  return refs;
}

// An ISP CU, from an ISP sub-partition's job (words 4 and 5, GpuJobs.h's IntraJob::setIsp): its block, whether its
// split is vertical, and whether it has neighbours to the left and above
struct IspCu
{
  int  x, y, w, h;
  bool ver;
  bool leftAvail, aboveAvail;
};

IspCu ispCuOf( uint job[12] )
{
  IspCu cu;
  cu.x          = int( job[4] & 0xffff );
  cu.y          = int( job[4] >> 16 );
  cu.w          = int( job[5] & 255 );
  cu.h          = int( ( job[5] >> 8 ) & 255 );
  cu.ver        = ( ( job[5] >> 16 ) & 3 ) == 2;
  cu.leftAvail  = ( ( job[5] >> 18 ) & 1 ) != 0;
  cu.aboveAvail = ( ( job[5] >> 19 ) & 1 ) != 0;
  return cu;
}

// initIntraPatternChTypeISP: the references of the prediction block (px, py,
// pw x ph) of one sub-partition of ISP CU cu (f is the CU's own fill, as for
// a block of the CU's size). The first sub-partition uses the CU's references.
// For the later ones, a horizontal split uses the row above (the previous
// sub-partition, its last sample repeated) and the CU's left column from that
// row down; a vertical split uses the CU's row above from that column on and
// the column to the left (the previous sub-partition, its last sample
// repeated). When the CU has no neighbours on that side, the previous
// sub-partition's sample next to the corner stands in for all of them.
References loadReferencesIsp( int tid, RefFill f, int px, int py, int pw, int ph, IspCu cu )
{
  const int  cx = cu.x, cy = cu.y, cw = cu.w, ch = cu.h;
  const bool first = px == cx && py == cy;
  References refs;
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int j = tid + 64 * k;
    int2      at;
    bool      fromFill;   // true: from the CU's fill; false: a sample of the picture
    if( first )
    {
      fromFill = true;
      at       = topSource( f, j );
    }
    else if( cu.ver )
    {
      fromFill = cu.aboveAvail;
      at       = cu.aboveAvail ? topSource( f, px - cx + j ) : int2( px - 1, cy );
    }
    else if( j == 0 )
    {
      fromFill = cu.leftAvail;
      at       = cu.leftAvail ? leftSource( f, py - cy ) : int2( cx, py - 1 );
    }
    else
    {
      fromFill = false;
      at       = int2( cx + min( j - 1, pw - 1 ), py - 1 );
    }
    refs.t[k] = j <= cw + pw ? ( fromFill ? fillValue( f, at ) : sampleAt( 0, at.x, at.y ) ) : 0;
  }
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int i = tid + 64 * k;
    int2      at;
    bool      fromFill;
    if( first )
    {
      fromFill = true;
      at       = leftSource( f, i );
    }
    else if( cu.ver )
    {
      fromFill = false;
      at       = int2( px - 1, cy + min( max( i - 1, 0 ), ph - 1 ) );   // (l[0] is t[0])
    }
    else
    {
      fromFill = cu.leftAvail;
      at       = cu.leftAvail ? leftSource( f, py - cy + i ) : int2( cx, py - 1 );
    }
    refs.l[k] = i <= ch + ph ? ( fromFill ? fillValue( f, at ) : sampleAt( 0, at.x, at.y ) ) : 0;
  }
  return refs;
}

// The writes of xFillReferenceSamples: t[0 .. topLast] and l[0 .. leftLast]
// (l[0], the corner, equals t[0]).
void storeReferences( int tid, int topLast, int leftLast, References refs )
{
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int j = tid + 64 * k;
    if( j <= topLast )
    {
      t[j] = refs.t[k];
    }
    if( j <= leftLast )
    {
      l[j] = j == 0 ? refs.t[0] : refs.l[k];
    }
  }
}

// xFilterReferenceSamples' [1 2 1] filter, run after the fill; each thread computes its own output samples.
void filterReferences( int tid, int w, int h )
{
  const int predSize  = w << 1;
  const int predHSize = h << 1;
  for( int i = tid; i <= predHSize; i += 64 )
  {
    lf[i] = i == predHSize ? l[i] : ( i == 0 ? ( l[1] + 2 * t[0] + t[1] + 2 ) >> 2 : ( l[i + 1] + 2 * l[i] + l[i - 1] + 2 ) >> 2 );
  }
  for( int i = tid; i <= predSize; i += 64 )
  {
    tf[i] = i == predSize ? t[i] : ( i == 0 ? ( l[1] + 2 * t[0] + t[1] + 2 ) >> 2 : ( t[i + 1] + 2 * t[i] + t[i - 1] + 2 ) >> 2 );
  }
}

int refT( bool filtered, int i )
{
  return filtered ? tf[i] : t[i];
}

int refL( bool filtered, int i )
{
  return filtered ? lf[i] : l[i];
}

int wideAngle( int w, int h, int predMode )
{
  if( predMode > DC_IDX && predMode <= VDIA_IDX )
  {
    static const int modeShift[6] = { 0, 6, 10, 12, 14, 15 };
    const int deltaSize = abs( log2Int( w ) - log2Int( h ) );
    if( w > h && predMode < 2 + modeShift[deltaSize] )
    {
      predMode += VDIA_IDX - 1;
    }
    else if( h > w && predMode > VDIA_IDX - modeShift[deltaSize] )
    {
      predMode -= VDIA_IDX - 1;
    }
  }
  return predMode;
}

// --------------------------------------------------------------------------------------------------------------------
// planar, DC, angular (xPredIntraPlanar, xGetPredValDc, xPredIntraAng) with PDPC, BDPCM
// --------------------------------------------------------------------------------------------------------------------

struct Angular
{
  bool isVer;
  int  predMode;
  int  angle;
  int  absAng;
  int  invAngle;
  int  width;          // as for a vertical mode (swapped for horizontal modes)
  int  height;
  int  refLength;
  int  r;              // the reference line (MRL)
  bool cubic;          // luma: the cubic filter if true, else the Gaussian one
  int  pdpcScale;      // angle 0: PDPC's scale, with lev its reach; angle > 0: angularScale (negative: no PDPC)
  int  lev;
};

// refMain[k] and refSide[k] of xPredIntraAng, before xPredIntraAng offsets them
// by the reference line (k indexes t and l)
int refMain( Angular a, bool filtered, int k )
{
  return a.isVer ? refT( filtered, k ) : refL( filtered, k );
}

int refSide( Angular a, bool filtered, int k )
{
  return a.isVer ? refL( filtered, k ) : refT( filtered, k );
}

// sizeW x sizeH is the size that decides the wide-angle modes, and topLength
// and leftLength are the references' lengths. Both come from the block, except
// for ISP: then the size is the CU's, and the lengths are the CU's width /
// height plus the block's. ISP always uses the cubic filter.
Angular setupAngular( int comp, int w, int h, int mode, int r, int sizeW, int sizeH, int topLength, int leftLength, bool isp )
{
  Angular a;
  a.predMode = wideAngle( sizeW, sizeH, mode );
  a.isVer    = a.predMode >= DIA_IDX;
  const int angMode    = a.isVer ? a.predMode - VER_IDX : -( a.predMode - HOR_IDX );
  const int absAngMode = abs( angMode );
  a.invAngle  = kInvAngTable[absAngMode];
  a.absAng    = kAngTable[absAngMode];
  a.angle     = angMode < 0 ? -a.absAng : a.absAng;
  a.width     = a.isVer ? w : h;
  a.height    = a.isVer ? h : w;
  a.refLength = a.isVer ? topLength : leftLength;
  a.r         = r;
  const int diff     = min( abs( a.predMode - HOR_IDX ), abs( a.predMode - VER_IDX ) );
  const int log2Size = ( log2Int( a.width ) + log2Int( a.height ) ) >> 1;
  // MRL's extra lines and ISP also use the cubic filter, and MRL has no PDPC.
  a.cubic = isp || !( diff > kIntraFilterLuma[log2Size] ) || r > 0;
  const bool doPdpc = w >= 4 && h >= 4 && r == 0;
  a.pdpcScale = -1;
  a.lev       = 0;
  if( a.angle == 0 )
  {
    if( doPdpc )
    {
      a.pdpcScale = ( log2Int( a.width ) - 2 + log2Int( a.height ) - 2 + 2 ) >> 2;
      a.lev       = a.pdpcScale == 0 ? min( 3, a.width ) : ( a.pdpcScale == 1 ? min( 6, a.width ) : ( a.pdpcScale == 2 ? min( 12, a.width ) : min( 24, a.width ) ) );
    }
  }
  else if( a.angle > 0 && doPdpc )
  {
    const int sideSize = a.isVer ? h : w;
    a.pdpcScale        = min( 2, log2Int( sideSize ) - ( log2Int( 3 * a.invAngle - 2 ) - 8 ) );
  }
  return a;
}

// The prediction at (x, y), in the coordinates of a vertical mode.
int predictAngular( Angular a, int comp, bool filtered, int x, int y, int maxVal )
{
  int v;
  if( a.angle == 0 )
  {
    v = refMain( a, filtered, x + 1 + a.r );
    if( a.pdpcScale >= 0 && x < a.lev )
    {
      const int wL = 32 >> min( 31, ( x << 1 ) >> a.pdpcScale );
      v = clip3( 0, maxVal, ( wL * ( refSide( a, filtered, y + 1 ) - refT( filtered, 0 ) ) + v * 64 + 32 ) >> 6 );
    }
    return v;
  }

  // (with MRL, the reference line lies r steps further from the block)
  const int deltaPos   = a.angle * ( y + 1 + a.r );
  const int deltaInt   = deltaPos >> 5;
  const int deltaFract = deltaPos & 31;
  int       p[4];
  for( int k = 0; k < 4; k++ )
  {
    // refMain is extended to the left from the side reference (negative
    // angles, before the offset of the reference line) and to the right by
    // replicating its last sample.
    const int idx = deltaInt + x + k + a.r;
    if( idx < 0 )
    {
      p[k] = refSide( a, filtered, min( ( -idx * a.invAngle + 256 ) >> 9, a.height ) );
    }
    else
    {
      p[k] = refMain( a, filtered, a.angle < 0 ? idx : min( idx, a.refLength + a.r ) );
    }
  }
  if( ( a.absAng & 31 ) == 0 )
  {
    v = p[1];
  }
  else if( comp == 0 )
  {
    const int f = deltaFract << 2;
    if( a.cubic )
    {
      v = clip3( 0, maxVal, ( s_cubic[f] * p[0] + s_cubic[f + 1] * p[1] + s_cubic[f + 2] * p[2] + s_cubic[f + 3] * p[3] + 32 ) >> 6 );
    }
    else
    {
      // kGauss: the taps of phase deltaFract are 16 - d, 32 - d, 16 + d, d with d = deltaFract / 2
      const int d = deltaFract >> 1;
      v = ( ( 16 - d ) * p[0] + ( 32 - d ) * p[1] + ( 16 + d ) * p[2] + d * p[3] + 32 ) >> 6;
    }
  }
  else
  {
    v = ( ( 32 - deltaFract ) * p[1] + deltaFract * p[2] + 16 ) >> 5;
  }

  if( a.angle > 0 && a.pdpcScale >= 0 && x < min( 3 << a.pdpcScale, a.width ) )
  {
    const int invAngleSum = 256 + ( x + 1 ) * a.invAngle;
    const int wL          = 32 >> ( ( 2 * x ) >> a.pdpcScale );
    const int left        = refSide( a, filtered, y + ( invAngleSum >> 9 ) + 1 );
    v = v + ( ( wL * ( left - v ) + 32 ) >> 6 );
  }
  return v;
}

int predictPlanar( bool filtered, int w, int h, int x, int y )
{
  const int log2W      = log2Int( w );
  const int log2H      = log2Int( h );
  const int left       = refL( filtered, y + 1 );
  const int top        = refT( filtered, x + 1 );
  const int horPred    = ( left << log2W ) + ( x + 1 ) * ( refT( filtered, w + 1 ) - left );
  const int verPred    = ( top << log2H ) + ( y + 1 ) * ( refL( filtered, h + 1 ) - top );
  return ( ( horPred << log2H ) + ( verPred << log2W ) + ( 1 << ( log2W + log2H ) ) ) >> ( 1 + log2W + log2H );
}

// (r is the reference line, MRL)
int dcValue( bool filtered, int w, int h, int r )
{
  const int denom = w == h ? ( w << 1 ) : max( w, h );
  int       sum   = 0;
  if( w >= h )
  {
    for( int i = 0; i < w; i++ )
    {
      sum += refT( filtered, 1 + r + i );
    }
  }
  if( w <= h )
  {
    for( int i = 0; i < h; i++ )
    {
      sum += refL( filtered, 1 + r + i );
    }
  }
  return ( sum + ( denom >> 1 ) ) >> log2Int( denom );
}

// PDPC of planar and DC (IntraPredSampleFilterCore)
int pdpcPlanarDc( bool filtered, int w, int h, int x, int y, int val )
{
  const int scale = ( log2Int( w ) - 2 + log2Int( h ) - 2 + 2 ) >> 2;
  const int wT    = 32 >> min( 31, ( y << 1 ) >> scale );
  const int wL    = 32 >> min( 31, ( x << 1 ) >> scale );
  return val + ( ( wL * ( refL( filtered, y + 1 ) - val ) + wT * ( refT( filtered, x + 1 ) - val ) + 32 ) >> 6 );
}

// --------------------------------------------------------------------------------------------------------------------
// MIP (Mip::PredictorMIP): the block's top and left references are averaged down to 2 or 4 samples each, a matrix
// multiplies them into the reduced prediction (4x4 or 8x8), and that is upsampled linearly to the block's size
// (horizontally, then vertically, starting from the references at the block's edges). The reduced boundary and the
// reduced prediction are kept in s_luma (CCLM's buffer), from MIP_BDRY and from 0.
// --------------------------------------------------------------------------------------------------------------------

#define MIP_BDRY 64

int mipSizeId( int w, int h )
{
  return ( w == 4 && h == 4 ) ? 0 : ( ( w == 4 || h == 4 || ( w == 8 && h == 8 ) ) ? 1 : 2 );
}

// boundaryDownsampling1D: element b of the reduced boundary (the top part for b < bdry, then the left), from t and l
int mipBoundary( int b, int bdry, int w, int h )
{
  const bool top    = b < bdry;
  const int  len    = top ? w : h;
  const int  idx    = top ? b : b - bdry;
  const int  factor = len / bdry;
  if( factor == 1 )
  {
    return top ? t[1 + idx] : l[1 + idx];
  }
  int sum = 0;
  for( int k = 0; k < factor; k++ )
  {
    sum += top ? t[1 + idx * factor + k] : l[1 + idx * factor + k];
  }
  const int log2F = log2Int( factor );
  return ( sum + ( 1 << ( log2F - 1 ) ) ) >> log2F;
}

// computeReducedPred: computes output o (in the order of the matrix's rows) of mode `mode` from the reduced
// boundary in s_luma[MIP_BDRY ..], and writes it where the (possibly transposed) reduced prediction keeps it.
void mipReducedPred( int o, int sizeId, int mode, bool transposed )
{
  const int bdry    = sizeId == 0 ? 2 : 4;
  const int pred    = sizeId < 2 ? 4 : 8;
  const int inSize  = 2 * bdry;
  const int skip    = sizeId == 2 ? 1 : 0;   // 1: the matrix's first column is unused (large blocks)
  const int perOut  = inSize - skip;
  const int base    = sizeId == 0 ? 0 : ( sizeId == 1 ? 1024 : 2048 );
  const int modeLen = ( sizeId == 0 ? 16 : ( sizeId == 1 ? 16 : 64 ) ) * perOut;
  // The input: the boundary (its left part first when transposed), relative to its first sample.
  int in[8];
  [unroll]
  for( int i = 0; i < 8; i++ )
  {
    const int src = transposed ? ( i < bdry ? bdry + i : i - bdry ) : i;
    in[i]         = i < inSize ? int( s_luma[MIP_BDRY + src] ) : 0;
  }
  const int inputOffset = in[0];
  in[0] = sizeId < 2 ? ( 1 << ( pc.bitDepth - 1 ) ) - inputOffset : 0;
  int sum = in[0];
  [unroll]
  for( int i = 1; i < 8; i++ )
  {
    in[i] = i < inSize ? in[i] - inputOffset : 0;
    sum += in[i];
  }
  const int offset = 32 - 32 * sum;   // ( 1 << ( MIP_SHIFT_MATRIX - 1 ) ) - MIP_OFFSET_MATRIX * sum
  const int at     = base + mode * modeLen + o * perOut;
  int       acc    = 0;
  for( int i = skip; i < inSize; i++ )
  {
    acc += in[i] * int( pc.mip[at + i - skip] );
  }
  const int v = clip3( 0, ( 1 << pc.bitDepth ) - 1, ( ( acc + offset ) >> 6 ) + inputOffset );
  s_luma[transposed ? ( o % pred ) * pred + o / pred : o] = int16_t( v );
}

// predictionUpsampling: row k of the reduced prediction, upsampled horizontally, at column x. Left of the first
// column it starts from the left reference at the block row where the reduced prediction's row k lands.
int mipRow( int k, int x, int w, int h, int pred )
{
  const int upH = w / pred;
  if( upH == 1 )
  {
    return int( s_luma[k * pred + x] );
  }
  const int upV    = h / pred;
  const int log2H  = log2Int( upH );
  const int j      = x >> log2H;
  const int i      = x & ( upH - 1 );
  const int before = j == 0 ? l[1 + ( upV - 1 ) + k * upV] : int( s_luma[k * pred + j - 1] );
  const int behind = int( s_luma[k * pred + j] );
  return ( ( before << log2H ) + ( upH >> 1 ) + ( i + 1 ) * ( behind - before ) ) >> log2H;
}

// The prediction at (x, y): mipRow's rows interpolated vertically, with the top references above the first row.
int mipSample( int x, int y, int w, int h, int pred )
{
  const int upV = h / pred;
  if( upV == 1 )
  {
    return mipRow( y, x, w, h, pred );
  }
  const int log2V  = log2Int( upV );
  const int k      = y >> log2V;
  const int i      = y & ( upV - 1 );
  const int before = k == 0 ? t[1 + x] : mipRow( k - 1, x, w, h, pred );
  const int behind = mipRow( k, x, w, h, pred );
  return ( ( before << log2V ) + ( upV >> 1 ) + ( i + 1 ) * ( behind - before ) ) >> log2V;
}

// --------------------------------------------------------------------------------------------------------------------
// CCLM (xGetLumaRecPixels, xGetLMParameters, linearTransform). The luma value under a chroma sample is, in 4:4:4,
// the luma sample itself; in 4:2:2, a [1 2 1] filter along the row (all of xGetLumaRecPixels' filters reduce to
// that with its 4:2:2 coefficients); in 4:2:0, a filter across two rows.
// --------------------------------------------------------------------------------------------------------------------

int lumaAt( int x, int y )
{
  return coherentSample( pc.plane0, y * pc.stride0 + x );
}

// 4:2:2: the [1 2 1] filter around the luma sample (x, y); with noLeft, the
// centre sample stands in for the missing left one.
int lumaRow422( int x, int y, bool noLeft )
{
  const int c = lumaAt( x, y );
  return ( 2 * c + ( noLeft ? c : lumaAt( x - 1, y ) ) + lumaAt( x + 1, y ) + 2 ) >> 2;
}

// The downsampled luma of chroma sample (i, j) of the block whose luma origin is (lx, ly).
int lumaDown( int lx, int ly, int i, int j, bool bAbove, bool bLeft )
{
  if( pc.scaleX == 0 )
  {
    return lumaAt( lx + i, ly + j );   // 4:4:4
  }
  if( pc.scaleY == 0 )
  {
    return lumaRow422( lx + 2 * i, ly + j, i == 0 && !bLeft );
  }
  const int x = lx + 2 * i;
  const int y = ly + 2 * j;
  if( pc.collocated != 0 )
  {
    const int up   = ( j == 0 && !bAbove ) ? lumaAt( x, y ) : lumaAt( x, y - 1 );
    const int left = ( i == 0 && !bLeft ) ? lumaAt( x, y ) : lumaAt( x - 1, y );
    return ( up + 4 * lumaAt( x, y ) + left + lumaAt( x + 1, y ) + lumaAt( x, y + 1 ) + 4 ) >> 3;
  }
  const int left0 = ( i == 0 && !bLeft ) ? lumaAt( x, y ) : lumaAt( x - 1, y );
  const int left1 = ( i == 0 && !bLeft ) ? lumaAt( x, y + 1 ) : lumaAt( x - 1, y + 1 );
  return ( 2 * lumaAt( x, y ) + lumaAt( x + 1, y ) + left0 + 2 * lumaAt( x, y + 1 ) + lumaAt( x + 1, y + 1 ) + left1 + 4 ) >> 3;
}

// The downsampled luma template above the block (sample i) and to its left (sample j).
int lumaTemplateAbove( int lx, int ly, int i, bool bLeft )
{
  // 4:4:4 and 4:2:2: the row just above the block, whether or not the block starts a CTU row.
  if( pc.scaleX == 0 )
  {
    return lumaAt( lx + i, ly - 1 );
  }
  if( pc.scaleY == 0 )
  {
    return lumaRow422( lx + 2 * i, ly - 1, i == 0 && !bLeft );
  }
  const int  x             = lx + 2 * i;
  const bool firstRowOfCtu = ( ly & ( ( 1 << pc.ctuLog2 ) - 1 ) ) == 0;
  if( firstRowOfCtu )
  {
    const int y    = ly - 1;
    const int left = ( i == 0 && !bLeft ) ? lumaAt( x, y ) : lumaAt( x - 1, y );
    return ( lumaAt( x, y ) * 2 + left + lumaAt( x + 1, y ) + 2 ) >> 2;
  }
  const int y = ly - 2;
  if( pc.collocated != 0 )
  {
    const int left = ( i == 0 && !bLeft ) ? lumaAt( x, y ) : lumaAt( x - 1, y );
    return ( lumaAt( x, y - 1 ) + 4 * lumaAt( x, y ) + left + lumaAt( x + 1, y ) + lumaAt( x, y + 1 ) + 4 ) >> 3;
  }
  const int left0 = ( i == 0 && !bLeft ) ? lumaAt( x, y ) : lumaAt( x - 1, y );
  const int left1 = ( i == 0 && !bLeft ) ? lumaAt( x, y + 1 ) : lumaAt( x - 1, y + 1 );
  return ( lumaAt( x, y ) * 2 + left0 + lumaAt( x + 1, y ) + lumaAt( x, y + 1 ) * 2 + left1 + lumaAt( x + 1, y + 1 ) + 4 ) >> 3;
}

int lumaTemplateLeft( int lx, int ly, int j, bool bAbove )
{
  // 4:4:4 and 4:2:2: the column centred 1 (4:4:4) or 2 (4:2:2) samples left of the block.
  if( pc.scaleX == 0 )
  {
    return lumaAt( lx - 1, ly + j );
  }
  if( pc.scaleY == 0 )
  {
    return lumaRow422( lx - 2, ly + j, false );
  }
  const int x = lx - 2;   // the centre, piSrc[1] of xGetLumaRecPixels
  const int y = ly + 2 * j;
  if( pc.collocated != 0 )
  {
    const int up = ( j == 0 && !bAbove ) ? lumaAt( x, y ) : lumaAt( x, y - 1 );
    return ( up + 4 * lumaAt( x, y ) + lumaAt( x - 1, y ) + lumaAt( x + 1, y ) + lumaAt( x, y + 1 ) + 4 ) >> 3;
  }
  return ( lumaAt( x, y ) * 2 + lumaAt( x - 1, y ) + lumaAt( x + 1, y ) + lumaAt( x, y + 1 ) * 2 + lumaAt( x - 1, y + 1 ) + lumaAt( x + 1, y + 1 ) + 4 ) >> 3;
}

// The template samples that the linear model picks: cntT above (t[1 + startPos0
// + k * pickStep0]), then cntL to the left (l[1 + startPos1 + k * pickStep1]);
// four at most.
struct CclmPicks
{
  int cntT;
  int cntL;
  int startPos0;
  int pickStep0;
  int startPos1;
  int pickStep1;
};

CclmPicks cclmPicks( uint lm, uint lm2 )
{
  const bool aboveAvailable = ( ( lm >> 18 ) & 1 ) != 0;
  const bool leftAvailable  = ( ( lm >> 19 ) & 1 ) != 0;
  const int  actualTop      = int( lm2 & 0xffff );
  const int  actualLeft     = int( lm2 >> 16 );

  const int aboveIs4 = leftAvailable ? 0 : 1;
  const int leftIs4  = aboveAvailable ? 0 : 1;
  CclmPicks p;
  p.startPos0 = actualTop >> ( 2 + aboveIs4 );
  p.pickStep0 = max( 1, actualTop >> ( 1 + aboveIs4 ) );
  p.startPos1 = actualLeft >> ( 2 + leftIs4 );
  p.pickStep1 = max( 1, actualLeft >> ( 1 + leftIs4 ) );
  p.cntT      = aboveAvailable ? min( actualTop, ( 1 + aboveIs4 ) << 1 ) : 0;
  p.cntL      = leftAvailable ? min( actualLeft, ( 1 + leftIs4 ) << 1 ) : 0;
  return p;
}

// The downsampled luma of template sample k (k < cntT + cntL).
int cclmTemplateLuma( int x0, int y0, uint lm, CclmPicks p, int k )
{
  const bool bAbove = ( ( lm >> 16 ) & 1 ) != 0;
  const bool bLeft  = ( ( lm >> 17 ) & 1 ) != 0;
  return k < p.cntT ? lumaTemplateAbove( x0 << pc.scaleX, y0 << pc.scaleY, p.startPos0 + k * p.pickStep0, bLeft )
                    : lumaTemplateLeft( x0 << pc.scaleX, y0 << pc.scaleY, p.startPos1 + ( k - p.cntT ) * p.pickStep1, bAbove );
}

// The linear model's a, b and shift, from the template's luma in s_tmpl and
// its chroma in t and l.
int3 cclmParameters( uint lm, uint lm2 )
{
  const bool aboveAvailable = ( ( lm >> 18 ) & 1 ) != 0;
  const bool leftAvailable  = ( ( lm >> 19 ) & 1 ) != 0;
  const CclmPicks p         = cclmPicks( lm, lm2 );

  int selLuma[4]   = { 0, 0, 0, 0 };
  int selChroma[4] = { 0, 0, 0, 0 };
  const int cntT = p.cntT, cntL = p.cntL;
  for( int k = 0; k < cntT; k++ )
  {
    selLuma[k]   = s_tmpl[k];
    selChroma[k] = t[1 + p.startPos0 + k * p.pickStep0];
  }
  for( int k = 0; k < cntL; k++ )
  {
    selLuma[cntT + k]   = s_tmpl[cntT + k];
    selChroma[cntT + k] = l[1 + p.startPos1 + k * p.pickStep1];
  }
  if( cntT + cntL == 2 )
  {
    selLuma[3] = selLuma[0]; selChroma[3] = selChroma[0];
    selLuma[2] = selLuma[1]; selChroma[2] = selChroma[1];
    selLuma[0] = selLuma[1]; selChroma[0] = selChroma[1];
    selLuma[1] = selLuma[3]; selChroma[1] = selChroma[3];
  }

  int minGrp[2] = { 0, 2 };
  int maxGrp[2] = { 1, 3 };
  if( selLuma[minGrp[0]] > selLuma[minGrp[1]] )
  {
    const int s = minGrp[0]; minGrp[0] = minGrp[1]; minGrp[1] = s;
  }
  if( selLuma[maxGrp[0]] > selLuma[maxGrp[1]] )
  {
    const int s = maxGrp[0]; maxGrp[0] = maxGrp[1]; maxGrp[1] = s;
  }
  if( selLuma[minGrp[0]] > selLuma[maxGrp[1]] )
  {
    // vvdec swaps the groups
    const int s0 = minGrp[0], s1 = minGrp[1];
    minGrp[0] = maxGrp[0]; minGrp[1] = maxGrp[1];
    maxGrp[0] = s0; maxGrp[1] = s1;
  }
  if( selLuma[minGrp[1]] > selLuma[maxGrp[0]] )
  {
    const int s = minGrp[1]; minGrp[1] = maxGrp[0]; maxGrp[0] = s;
  }
  const int minY = ( selLuma[minGrp[0]] + selLuma[minGrp[1]] + 1 ) >> 1;
  const int minC = ( selChroma[minGrp[0]] + selChroma[minGrp[1]] + 1 ) >> 1;
  const int maxY = ( selLuma[maxGrp[0]] + selLuma[maxGrp[1]] + 1 ) >> 1;
  const int maxC = ( selChroma[maxGrp[0]] + selChroma[maxGrp[1]] + 1 ) >> 1;

  if( !leftAvailable && !aboveAvailable )
  {
    return int3( 0, 1 << ( pc.bitDepth - 1 ), 0 );
  }
  const int diff = maxY - minY;
  if( diff <= 0 )
  {
    return int3( 0, minC, 0 );
  }
  const int diffC    = maxC - minC;
  int       x        = log2Int( diff );
  const int normDiff = ( ( diff << 4 ) >> x ) & 15;
  const int v        = kDivSigTable[normDiff] | 8;
  x += normDiff != 0 ? 1 : 0;
  const int y     = diffC == 0 ? 0 : log2Int( abs( diffC ) ) + 1;
  const int add   = ( 1 << y ) >> 1;
  int       a     = ( diffC * v + add ) >> y;
  int       shift = 3 + x - y;
  if( shift < 1 )
  {
    shift = 1;
    a     = a == 0 ? 0 : ( a < 0 ? -15 : 15 );
  }
  return int3( a, minC - ( ( a * minY ) >> shift ), shift );
}

// --------------------------------------------------------------------------------------------------------------------
// LMCS chroma residual scaling (Reshape::calculateChromaAdjVpduNei, AreaBuf::scaleSignal): the average of the luma
// samples (still mapped) left of and above the CU given in job word 8, 1 << log2 count samples per side, picks the
// scale. Rows and columns past the picture use the picture's last one.
// --------------------------------------------------------------------------------------------------------------------

groupshared int s_lsum[4];   // the luma sums of the workgroup's subgroups (hardware waves)

// This thread's share of the luma sum: left sample tid and above sample tid.
int lmcsLuma( int tid, uint anchor, uint info )
{
  const int x     = int( anchor & 0xffff );
  const int y     = int( anchor >> 16 );
  const int count = 1 << int( ( info >> 4 ) & 15 );
  int       sum   = 0;
  if( tid < count && ( info & 2 ) != 0 )
  {
    sum += sampleAt( 0, x - 1, min( y + tid, pc.picHeight - 1 ) );
  }
  if( tid < count && ( info & 4 ) != 0 )
  {
    sum += sampleAt( 0, min( x + tid, pc.picWidth - 1 ), y - 1 );
  }
  return sum;
}

// Sums the shares of each hardware wave into s_lsum; all threads call it, before a barrier.
void lmcsShare( int tid, int luma )
{
  const int sum = WaveActiveSum( luma );
  if( WaveIsFirstLane() )
  {
    s_lsum[tid / int( WaveGetLaneCount() )] = sum;
  }
}

// After that barrier: returns the scale.
int lmcsScale( uint info )
{
  int sum = 0;
  for( int k = 0; k < 64 / int( WaveGetLaneCount() ); k++ )
  {
    sum += s_lsum[k];
  }
  const int log2Count = int( ( info >> 4 ) & 15 );
  const int sides     = int( ( info >> 1 ) & 1 ) + int( ( info >> 2 ) & 1 );
  const int luma      = sides == 1 ? ( sum + ( 1 << ( log2Count - 1 ) ) ) >> log2Count
                        : ( sides == 2 ? ( sum + ( 1 << log2Count ) ) >> ( log2Count + 1 ) : 1 << ( pc.bitDepth - 1 ) );
  return pc.lmcsScale[luma];
}

// Scales a residual, after clipping it to the bit depth's range.
int lmcsResidual( int r, int scale )
{
  const int range = 1 << pc.bitDepth;
  const int c     = clip3( -range, range - 1, r );
  const int v     = ( abs( c ) * scale + ( 1 << 10 ) ) >> 11;
  return clip3( -32768, 32767, c < 0 ? -v : v );
}

// An intra job's block, decoded from the job's words for make() and its stages
struct IntraBlock
{
  int  x0, y0, w, h, log2W;   // the block, in samples of its component
  int  comp, kind, mode;
  uint flags;
  int  refLine;               // MRL's reference line
  bool filtered, residual;
  int  s0, m;                 // the samples the workgroup makes: the block's s0 to s0 + m - 1, a band of its rows
  uint lmcsInfo;              // LMCS: whether (bit 0) and how to scale the chroma residual
  bool lmcs;
  int  maxVal;
};

IntraBlock intraBlockOf( uint job[12] )
{
  IntraBlock b;
  b.x0       = int( job[0] & 0xffff );
  b.y0       = int( job[0] >> 16 );
  b.w        = int( job[1] & 255 );
  b.h        = int( ( job[1] >> 8 ) & 255 );
  b.log2W    = log2Int( b.w );
  b.comp     = int( ( job[1] >> 16 ) & 255 );
  b.kind     = int( job[1] >> 24 );
  b.mode     = int( job[2] & 255 );
  b.flags    = ( job[2] >> 8 ) & 255;
  b.refLine  = int( ( job[2] >> 24 ) & 3 );
  b.filtered = ( b.flags & kIntraFiltered ) != 0;
  b.residual = ( b.flags & kIntraResidual ) != 0;
  // The workgroup makes the block's rows first to first + count - 1: m samples (at most 256, four per thread); its
  // sample i is the block's sample s0 + i.
  b.s0       = int( job[7] & 0xffff ) * b.w;
  b.m        = int( job[7] >> 16 ) * b.w;
  b.lmcsInfo = job[9];
  b.lmcs     = ( b.lmcsInfo & 1 ) != 0;
  b.maxVal   = ( 1 << pc.bitDepth ) - 1;
  return b;
}

// Makes an IBC job, which copies the block from elsewhere in the picture and adds the residual, or an LMCS job (the
// chroma of an inter CU), which adds the scaled residual to the prediction kept aside. All the reads are issued
// together, then the writes.
void copyJob( int tid, uint job[12], IntraBlock b )
{
  int16_t*  dst    = planeOf( b.comp );
  const int stride = strideOf( b.comp );
  const int bvx    = int( int16_t( job[6] & 0xffff ) );
  const int bvy    = int( int16_t( job[6] >> 16 ) );
  int16_t*  src    = b.kind == kIntraLmcs ? ( b.comp == 1 ? pc.defer1 : pc.defer2 ) : dst;
  int       pred[4];
  int       res[4];
  [unroll]
  for( int k = 0; k < 4; k++ )
  {
    const int i = tid + 64 * k;
    const int x = ( b.s0 + i ) & ( b.w - 1 );
    const int y = ( b.s0 + i ) >> b.log2W;
    pred[k]     = i < b.m ? ( b.kind == kIntraLmcs ? int( src[( b.y0 + y ) * stride + b.x0 + x] ) : sampleAt( b.comp, b.x0 + x + bvx, b.y0 + y + bvy ) ) : 0;
    res[k]      = i < b.m && b.residual ? int( dst[( b.y0 + y ) * stride + b.x0 + x] ) : 0;
  }
  int scale = 0;
  if( b.lmcs )
  {
    lmcsShare( tid, lmcsLuma( tid, job[8], b.lmcsInfo ) );
    GroupMemoryBarrierWithGroupSync();
    scale = lmcsScale( b.lmcsInfo );
  }
  [unroll]
  for( int k = 0; k < 4; k++ )
  {
    const int i = tid + 64 * k;
    if( i < b.m )
    {
      const int at = ( b.y0 + ( ( b.s0 + i ) >> b.log2W ) ) * stride + b.x0 + ( ( b.s0 + i ) & ( b.w - 1 ) );
      dst[at]      = int16_t( b.residual ? clip3( 0, b.maxVal, pred[k] + ( b.lmcs ? lmcsResidual( res[k], scale ) : res[k] ) ) : pred[k] );
    }
  }
}

// The block that the prediction covers (pw x ph; the job's block starts xoff samples into it), the size that decides
// the wide-angle modes, and the references' lengths. All are the job's block's own, except for ISP sub-partitions
// (kIntraIsp): the size is the CU's, the top (left) length is the CU's width (height) plus the block's, and the
// sub-partitions of a vertical split that are 1 or 2 samples wide share one prediction block 4 samples wide
// (CU::isPredRegDiffFromTB).
struct PredBlock
{
  bool  isp;
  IspCu cu;   // (meaningful with isp only)
  int   pw, ph, xoff;
  int   sizeW, sizeH;
  int   topLength, leftLength;
};

PredBlock predBlockOf( uint job[12], IntraBlock b )
{
  PredBlock p;
  p.isp             = ( b.flags & kIntraIsp ) != 0;
  p.cu              = ispCuOf( job );
  const bool region = p.isp && p.cu.ver && ( p.cu.w == 4 || ( p.cu.w == 8 && p.cu.h > 4 ) );
  p.pw              = region ? 4 : b.w;
  p.ph              = b.h;
  p.xoff            = region ? ( b.x0 - p.cu.x ) & 3 : 0;
  p.sizeW           = p.isp ? p.cu.w : p.pw;
  p.sizeH           = p.isp ? p.cu.h : p.ph;
  p.topLength       = p.isp ? p.cu.w + p.pw : p.pw << 1;
  p.leftLength      = p.isp ? p.cu.h + p.ph : p.ph << 1;
  return p;
}

// Reads what the block needs into shared memory: first the references, then, while those are in flight, the
// residual, CCLM's luma (or CIIP's inter prediction), kCubic for luma's angular modes and LMCS's luma; then waits at a
// barrier for all of them.
void loadInputs( int tid, uint job[12], IntraBlock b, PredBlock p )
{
  int16_t*   dst    = planeOf( b.comp );
  const int  stride = strideOf( b.comp );
  const uint nb     = job[3];
  References refs;
  if( p.isp )
  {
    // the CU's fill, as for a block of the CU's size
    const RefFill f = refFill( b.comp, p.cu.x, p.cu.y, p.cu.w << 1, p.cu.h << 1, int( nb & 255 ), int( ( nb >> 8 ) & 255 ), int( ( nb >> 16 ) & 255 ), 0 );
    refs = loadReferencesIsp( tid, f, b.x0 - p.xoff, b.y0, p.pw, p.ph, p.cu );
  }
  else
  {
    refs = loadReferences( tid, refFill( b.comp, b.x0, b.y0, b.w << 1, b.h << 1, int( nb & 255 ), int( ( nb >> 8 ) & 255 ), int( ( nb >> 16 ) & 255 ), b.refLine ) );
  }
  const bool lmAbove = ( ( job[4] >> 16 ) & 1 ) != 0;
  const bool lmLeft  = ( ( job[4] >> 17 ) & 1 ) != 0;
  if( b.kind == kIntraCclm )
  {
    // CCLM: the downsampled luma of the template samples, one thread each (the block's samples follow below)
    const CclmPicks picks = cclmPicks( job[4], job[5] );
    if( tid < picks.cntT + picks.cntL )
    {
      s_tmpl[tid] = cclmTemplateLuma( b.x0, b.y0, job[4], picks, tid );
    }
  }
  if( b.kind == kIntraCclm || b.kind == kIntraCiip || b.residual )
  {
    // (CIIP keeps inter's prediction where CCLM keeps its luma, in s_luma)
    int16_t* inter = b.comp == 0 ? pc.defer0 : ( b.comp == 1 ? pc.defer1 : pc.defer2 );
    int      luma[4];
    int      res[4];
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      const int i = tid + 64 * k;
      const int x = ( b.s0 + i ) & ( b.w - 1 );
      const int y = ( b.s0 + i ) >> b.log2W;
      luma[k]     = i < b.m && b.kind == kIntraCclm ? lumaDown( b.x0 << pc.scaleX, b.y0 << pc.scaleY, x, y, lmAbove, lmLeft )
                    : ( i < b.m && b.kind == kIntraCiip ? int( inter[( b.y0 + y ) * stride + b.x0 + x] ) : 0 );
      res[k]      = i < b.m && b.residual ? int( dst[( b.y0 + y ) * stride + b.x0 + x] ) : 0;
    }
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      const int i = tid + 64 * k;
      if( i < b.m )
      {
        s_luma[i] = int16_t( luma[k] );
        s_res[i]  = int16_t( res[k] );
      }
    }
  }
  if( b.kind == kIntraAngular && b.comp == 0 )
  {
    s_cubic[tid]      = kCubic[tid];
    s_cubic[tid + 64] = kCubic[tid + 64];
  }
  if( b.lmcs )
  {
    lmcsShare( tid, lmcsLuma( tid, job[8], b.lmcsInfo ) );
  }
  storeReferences( tid, p.topLength + b.refLine, p.leftLength + b.refLine, refs );
  GroupMemoryBarrierWithGroupSync();
}

// Makes one job (a block of a component, or a band of its rows): the workgroup's threads first read together what the
// block needs from the picture, then share out the block's samples.
void make( int tid, uint job[12] )
{
  const IntraBlock b = intraBlockOf( job );
  if( b.kind == kIntraIbc || b.kind == kIntraLmcs )
  {
    copyJob( tid, job, b );
    return;   // (the whole workgroup returns here)
  }
  const PredBlock p = predBlockOf( job, b );
  loadInputs( tid, job, b, p );
  const int lmcsScaling = b.lmcs ? lmcsScale( b.lmcsInfo ) : 0;

  // Per block, derive what the predictor needs from the references.
  if( b.filtered )
  {
    filterReferences( tid, b.w, b.h );
    GroupMemoryBarrierWithGroupSync();
  }
  Angular ang;
  if( b.kind == kIntraCclm )
  {
    if( tid == 0 )
    {
      s_model = cclmParameters( job[4], job[5] );
    }
    GroupMemoryBarrierWithGroupSync();
  }
  else if( b.kind == kIntraAngular && b.mode == DC_IDX )
  {
    if( tid == 0 )
    {
      s_dc = dcValue( b.filtered, p.pw, p.ph, b.refLine );
    }
    GroupMemoryBarrierWithGroupSync();
  }
  else if( b.kind == kIntraAngular && b.mode != PLANAR_IDX )
  {
    ang = setupAngular( b.comp, p.pw, p.ph, b.mode, b.refLine, p.sizeW, p.sizeH, p.topLength, p.leftLength, p.isp );
  }
  const int mipPred = mipSizeId( b.w, b.h ) < 2 ? 4 : 8;
  if( b.kind == kIntraMip )
  {
    // MIP: the reduced boundary, then the reduced prediction (one thread per element)
    const int sizeId = mipSizeId( b.w, b.h );
    const int bdry   = sizeId == 0 ? 2 : 4;
    if( tid < 2 * bdry )
    {
      s_luma[MIP_BDRY + tid] = int16_t( mipBoundary( tid, bdry, b.w, b.h ) );
    }
    GroupMemoryBarrierWithGroupSync();
    if( tid < mipPred * mipPred )
    {
      mipReducedPred( tid, sizeId, b.mode, ( b.flags & kIntraMipTransposed ) != 0 );
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const int  dc    = s_dc;
  const int3 model = s_model;

  const int  bdpcmDir       = int( ( job[2] >> 16 ) & 255 );
  const bool doPdpcPlanarDc = ( b.kind == kIntraAngular || b.kind == kIntraCiip ) && ( b.mode == PLANAR_IDX || b.mode == DC_IDX ) && p.pw >= 4
                              && p.ph >= 4 && b.refLine == 0;
  const int  ciipIntra      = int( job[6] );   // CIIP: planar's weight

  // Compute the samples, from shared memory only.
  int16_t*  dst    = planeOf( b.comp );
  const int stride = strideOf( b.comp );
  for( int i = tid; i < b.m; i += 64 )
  {
    const int x = ( b.s0 + i ) & ( b.w - 1 );
    const int y = ( b.s0 + i ) >> b.log2W;
    int       v;
    if( b.kind == kIntraCclm )
    {
      v = clip3( 0, b.maxVal, ( ( model.x * int( s_luma[i] ) ) >> model.z ) + model.y );
    }
    else if( b.kind == kIntraBdpcm )
    {
      v = bdpcmDir == 1 ? refL( false, y + 1 ) : refT( false, x + 1 );
    }
    else if( b.kind == kIntraMip )
    {
      v = mipSample( x, y, b.w, b.h, mipPred );
    }
    else if( b.mode == PLANAR_IDX || b.mode == DC_IDX )
    {
      v = b.mode == PLANAR_IDX ? predictPlanar( b.filtered, p.pw, p.ph, x + p.xoff, y ) : dc;
      if( doPdpcPlanarDc )
      {
        v = pdpcPlanarDc( b.filtered, p.pw, p.ph, x + p.xoff, y, v );
      }
    }
    else
    {
      v = ang.isVer ? predictAngular( ang, b.comp, b.filtered, x + p.xoff, y, b.maxVal ) : predictAngular( ang, b.comp, b.filtered, y, x + p.xoff, b.maxVal );
    }
    if( b.kind == kIntraCiip )
    {
      // predBlendIntraCiip
      v = ( ( 4 - ciipIntra ) * int( s_luma[i] ) + ciipIntra * v + 2 ) >> 2;
    }
    const int at = ( b.y0 + y ) * stride + b.x0 + x;
    dst[at]      = int16_t( b.residual ? clip3( 0, b.maxVal, v + ( b.lmcs ? lmcsResidual( int( s_res[i] ), lmcsScaling ) : int( s_res[i] ) ) ) : v );
  }
}

groupshared uint s_job;   // the job the workgroup took

// One workgroup per job: each workgroup takes the next job in wave order.
[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 local : SV_GroupThreadID )
{
  const int tid = int( local.x );
  if( tid == 0 )
  {
    s_job = pc.sync[0].add( 1u );
  }
  GroupMemoryBarrierWithGroupSync();
  const uint index = WaveReadLaneFirst( s_job );
  if( index >= ( pc.count >= 0 ? uint( pc.count ) : pc.countAt[0] ) )
  {
    return;   // (the whole workgroup: the dispatch, in rows of kGroupsPerRow workgroups, has more workgroups than jobs)
  }
  // Load the job while the workgroup waits for the earlier jobs.
  const uint4 a = *(uint4*)( pc.jobs + index * 12 );
  const uint4 b = *(uint4*)( pc.jobs + index * 12 + 4 );
  const uint4 c = *(uint4*)( pc.jobs + index * 12 + 8 );
  if( tid == 0 )
  {
    const uint before = pc.waits[index];
    while( pc.sync[1].load( MemoryOrder.Relaxed ) < before )
    {
    }
  }
  // The barrier acquires: what those jobs wrote becomes visible to all the threads.
  AllMemoryBarrierWithGroupSync();
  uint job[12] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w };
  make( tid, job );
  // The barrier releases: the threads' writes complete before the job counts as done.
  AllMemoryBarrierWithGroupSync();
  if( tid == 0 )
  {
    pc.sync[1].add( 1u, MemoryOrder.Release );
  }
}
