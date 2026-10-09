// Intra prediction, intra block copy (IBC) and the reconstruction of their
// transform blocks, as vvdec's IntraPrediction and DecCu::predAndReco do it: a
// workgroup predicts one transform block of one component, adds its residual
// (already in the picture, where vvdec's TrQuant left it) and writes the
// result. The blocks come in waves: a block's wave is above the waves of
// every block whose samples it reads. One dispatch makes all the blocks of a
// picture: a workgroup takes the next block in the order of the waves (a
// count of the blocks taken) and waits until the blocks of the waves before
// its block's are done (a count of the blocks done). Taken in order, those
// are all with workgroups already running, which wait only for blocks before
// them: nothing waits for a workgroup that cannot run. (A dispatch a wave,
// each after a barrier, took 6.5 us a wave, and a picture coded intra has a
// thousand waves.)
//
// A wave lasts as long as its slowest workgroup, which mostly waits for
// memory: so a workgroup makes 256 samples at most (GpuRecon::recordIntra
// cuts the big blocks into bands of rows), four a thread, and first sends
// out all its reads of the picture together (the references, the residual,
// CCLM's luma, IBC's source), to compute from shared memory after that.
//
// A job, 12 uints (VulkanLib/GpuJobs.h, IntraJob):
//   0: x | y << 16                (in samples of its component)
//   1: width | height << 8 | component << 16 | kind << 24   (kind: 0 planar/DC/angular, 1 BDPCM, 2 CCLM, 3 IBC,
//      4 MIP, 5 an inter CU's chroma under LMCS: the prediction inter kept aside plus the residual)
//   2: mode | flags << 8 | bdpcm direction << 16 | reference line << 24   (flags: 1 residual, 2 filtered references,
//      4 MIP transposed, 8 ISP; the reference line: MRL's, 0 .. 2)
//   3: top-left units | above units << 8 | left units << 16  (the reference samples available; ISP: the CU's)
//   4: CCLM: luma above units | luma left units << 8 | bAbove << 16 | bLeft << 17 | aboveAvail << 18 | leftAvail << 19 | lm mode << 24
//      ISP: the CU's x | y << 16
//   5: CCLM: actual top template samples | actual left << 16
//      ISP: the CU's width | height << 8 | split << 16 (1 horizontal, 2 vertical) | left there << 18 | above there << 19
//   6: IBC: block vector x | y << 16 (int16, samples of the component)
//   7: the rows of the block this workgroup makes: first | count << 16, at
//      most 256 samples (on the CPU: the block's wave)
//   8: LMCS's chroma residual scaling: x | y << 16 of the luma CU whose
//      neighbours give the scale
//   9: LMCS: 1 scaling | 2 left there | 4 above there | log2 of the samples a side << 4

#include "common.slang"

struct Params
{
  int16_t*      plane0;
  int16_t*      plane1;
  int16_t*      plane2;
  uint*         jobs;        // the picture's, wave by wave
  uint*         waits;       // a job's: the jobs of the waves before its, which it waits for
  Atomic<uint>* sync;        // the jobs taken, the jobs done (zeroed before the dispatch)
  uint8_t*      mip;         // MIP's matrices: vvdec's mipMatrix4x4, mipMatrix8x8, mipMatrix16x16 in a row
  int*          lmcsScale;   // LMCS: the chroma residual scale of each average luma
  int16_t*      defer1;      // LMCS: the inter CUs' chroma predictions kept aside, laid out as the chroma planes
  int16_t*      defer2;
  int           stride0;
  int           stride12;
  int           count;       // the jobs
  int           bitDepth;
  int           ctuLog2;     // luma
  int           scaleX;      // chroma subsampling
  int           scaleY;
  int           collocated;  // sps CCLM collocated chroma
  int           picWidth;    // luma
  int           picHeight;
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
  return int( planeOf( comp )[y * strideOf( comp ) + x] );
}

// the reference samples, as vvdec's xFillReferenceSamples (and
// xFilterReferenceSamples) lay them out: t[0] is the top-left sample, t[1..]
// the row above (above-right included), l[0] the top-left again, l[1..] the
// column to the left (below-left included); shared by the block's threads
groupshared int t[MAX_REF];
groupshared int l[MAX_REF];
groupshared int tf[MAX_REF];
groupshared int lf[MAX_REF];
groupshared int  s_dc;      // DC's value
groupshared int3 s_model;   // CCLM's a, b, shift
groupshared int16_t s_res[256];    // the residual of the workgroup's samples, out of the picture
groupshared int16_t s_luma[256];   // CCLM: the downsampled luma of the workgroup's samples
groupshared int     s_tmpl[4];         // CCLM: the luma of the template samples the model picks
groupshared int     s_cubic[128];      // kCubic, for luma's angular modes

// xFillReferenceSamples' reads: the samples of the thread's references,
// t[tid + 64 k] and l[tid + 64 k] (k < 3), of the reference line r rows above
// and columns left of the block's neighbours (MRL: r = cu.multiRefIdx(), 0
// but for luma's extra lines): t[0 .. r] and l[0 .. r] the corner, t[1 + r +
// i] the row above from x0 + i, l[1 + r + i] the column to the left from y0
// + i. Where neighbours are missing, the samples are padded as vvdec pads them
// -- each element's source sample (or the middle value), worked out here, so
// that all the reads go out together
struct References
{
  int t[3];
  int l[3];
};

// a block's xFillReferenceSamples: the block, its references' lengths (the
// row above, the column to the left), the reference line and the units of
// neighbours there
struct RefFill
{
  int  comp;
  int  x0, y0;
  int  predSize, predHSize;
  int  r;
  int  nTL, nAbove, nLeft;
  int  tmpA, tmpL;   // the samples there above and to the left
  bool none, all;
  int2 pad;          // what the corner and the missing parts take when only some neighbours are there
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
  // the left: the corner if the top-left is there, else the left column's
  // first; no left: the row above's first
  f.pad = nLeft > 0 ? ( nTL != 0 ? int2( x0 - 1, y0 - 1 - r ) : int2( x0 - 1 - r, y0 ) ) : int2( x0, y0 - 1 - r );
  return f;
}

// the sample element j of t takes (of the row y0 - 1 - r)
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

// the sample element i of l takes (of the column x0 - 1 - r)
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

// a reference sample's value: its source's, or the middle value with no neighbours
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

// initIntraPatternChTypeISP: the references of the prediction block (px, py,
// pw x ph) of a sub-partition of an ISP CU (cx, cy, cw x ch; f: the CU's
// own fill, as a block of its size): the CU's for its first block; after
// that, for a horizontal split the row above (the sub-partition before, its
// last sample repeated) and the CU's left column from that row on, for a
// vertical split the CU's row above from that column on and the column to
// the left (the sub-partition before, its last sample repeated). Without the
// CU's neighbours on that side, the sub-partition before's sample next to
// the corner stands in for them all.
References loadReferencesIsp( int tid, RefFill f, int px, int py, int pw, int ph, int cx, int cy, int cw, int ch, bool ver, bool leftAvail,
                              bool aboveAvail )
{
  const bool first = px == cx && py == cy;
  References refs;
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int j = tid + 64 * k;
    int2      at;
    bool      fromFill;   // of the CU's fill, else a sample of the picture
    if( first )
    {
      fromFill = true;
      at       = topSource( f, j );
    }
    else if( ver )
    {
      fromFill = aboveAvail;
      at       = aboveAvail ? topSource( f, px - cx + j ) : int2( px - 1, cy );
    }
    else if( j == 0 )
    {
      fromFill = leftAvail;
      at       = leftAvail ? leftSource( f, py - cy ) : int2( cx, py - 1 );
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
    else if( ver )
    {
      fromFill = false;
      at       = int2( px - 1, cy + min( max( i - 1, 0 ), ph - 1 ) );   // (l[0] is t[0])
    }
    else
    {
      fromFill = leftAvail;
      at       = leftAvail ? leftSource( f, py - cy + i ) : int2( cx, py - 1 );
    }
    refs.l[k] = i <= ch + ph ? ( fromFill ? fillValue( f, at ) : sampleAt( 0, at.x, at.y ) ) : 0;
  }
  return refs;
}

// xFillReferenceSamples' writes, t[0 .. topLast] and l[0 .. leftLast] (l[0],
// the corner, is t[0])
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

// xFilterReferenceSamples' [1 2 1] filter, each sample by itself (after the fill)
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
  int  width;          // in the frame of a vertical mode
  int  height;
  int  refLength;
  int  r;              // the reference line (MRL)
  bool cubic;          // luma: the cubic filter, else the Gaussian one
  int  pdpcScale;      // angle 0: PDPC's scale, lev its reach; angle > 0: angularScale (< 0: no PDPC)
  int  lev;
};

// refMain[k] and refSide[k] of xPredIntraAng, before its offset by the
// reference line (k: an index into t and l)
int refMain( Angular a, bool filtered, int k )
{
  return a.isVer ? refT( filtered, k ) : refL( filtered, k );
}

int refSide( Angular a, bool filtered, int k )
{
  return a.isVer ? refL( filtered, k ) : refT( filtered, k );
}

// (sizeW x sizeH: the size the wide-angle modes follow, topLength and
// leftLength the references' lengths: the block's, but for ISP's: the CU's
// size, the CU's width / height on top of the block's; ISP takes the cubic
// filter always)
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
  // the cubic filter for MRL's extra lines and ISP too, and no PDPC for MRL
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

// the prediction at (x, y) of the frame of a vertical mode
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

  // (MRL: the line r steps further from the block)
  const int deltaPos   = a.angle * ( y + 1 + a.r );
  const int deltaInt   = deltaPos >> 5;
  const int deltaFract = deltaPos & 31;
  int       p[4];
  for( int k = 0; k < 4; k++ )
  {
    // refMain extended to the left from the side reference (negative angles,
    // before the offset of the reference line), to the right by replicating
    // its last sample
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
      // kGauss, whose taps of phase deltaFract are 16 - d, 32 - d, 16 + d, d for d = deltaFract / 2
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

// (r: the reference line, MRL)
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
// MIP (Mip::PredictorMIP): the block's top and left references averaged down to 2 or 4 samples each, a matrix
// times them (the reduced prediction, 4x4 or 8x8), upsampled linearly to the block (horizontally, then vertically,
// from the references at the block's edges). The reduced boundary and prediction live in s_luma (CCLM's), from
// MIP_BDRY and 0
// --------------------------------------------------------------------------------------------------------------------

#define MIP_BDRY 64

int mipSizeId( int w, int h )
{
  return ( w == 4 && h == 4 ) ? 0 : ( ( w == 4 || h == 4 || ( w == 8 && h == 8 ) ) ? 1 : 2 );
}

// boundaryDownsampling1D: element b of the reduced boundary, top's (b < bdry) then left's, from t and l
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

// computeReducedPred: output o (in the order of the matrix's rows) of mode `mode`, from the reduced boundary in
// s_luma[MIP_BDRY ..]; written where the (transposed) reduced prediction has it
void mipReducedPred( int o, int sizeId, int mode, bool transposed )
{
  const int bdry    = sizeId == 0 ? 2 : 4;
  const int pred    = sizeId < 2 ? 4 : 8;
  const int inSize  = 2 * bdry;
  const int skip    = sizeId == 2 ? 1 : 0;   // the matrix's first column, unused for the large blocks
  const int perOut  = inSize - skip;
  const int base    = sizeId == 0 ? 0 : ( sizeId == 1 ? 1024 : 2048 );
  const int modeLen = ( sizeId == 0 ? 16 : ( sizeId == 1 ? 16 : 64 ) ) * perOut;
  // the input: the boundary, left's first when transposed, rebased on its first sample
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

// predictionUpsampling: the horizontally upsampled row k of the reduced prediction at column x (the left references
// at the rows the reduced prediction's lands on)
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

// the prediction at (x, y): the rows upsampled vertically (the top references above the first)
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
// CCLM (xGetLumaRecPixels, xGetLMParameters, linearTransform), 4:2:0
// --------------------------------------------------------------------------------------------------------------------

int lumaAt( int x, int y )
{
  return int( pc.plane0[y * pc.stride0 + x] );
}

// the downsampled luma of chroma sample (i, j) of the block whose luma origin is (lx, ly)
int lumaDown( int lx, int ly, int i, int j, bool bAbove, bool bLeft )
{
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

// the downsampled luma template above the block (i) and to its left (j)
int lumaTemplateAbove( int lx, int ly, int i, bool bLeft )
{
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
  const int x = lx - 2;   // the centre, piSrc[1] of xGetLumaRecPixels
  const int y = ly + 2 * j;
  if( pc.collocated != 0 )
  {
    const int up = ( j == 0 && !bAbove ) ? lumaAt( x, y ) : lumaAt( x, y - 1 );
    return ( up + 4 * lumaAt( x, y ) + lumaAt( x - 1, y ) + lumaAt( x + 1, y ) + lumaAt( x, y + 1 ) + 4 ) >> 3;
  }
  return ( lumaAt( x, y ) * 2 + lumaAt( x - 1, y ) + lumaAt( x + 1, y ) + lumaAt( x, y + 1 ) * 2 + lumaAt( x - 1, y + 1 ) + lumaAt( x + 1, y + 1 ) + 4 ) >> 3;
}

// the template samples the linear model picks: cntT above (t[1 + startPos0 +
// k * pickStep0]), then cntL to the left (l[1 + startPos1 + k * pickStep1]);
// four at most
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

// the downsampled luma of template sample k (< cntT + cntL)
int cclmTemplateLuma( int x0, int y0, uint lm, CclmPicks p, int k )
{
  const bool bAbove = ( ( lm >> 16 ) & 1 ) != 0;
  const bool bLeft  = ( ( lm >> 17 ) & 1 ) != 0;
  return k < p.cntT ? lumaTemplateAbove( x0 << 1, y0 << 1, p.startPos0 + k * p.pickStep0, bLeft )
                    : lumaTemplateLeft( x0 << 1, y0 << 1, p.startPos1 + ( k - p.cntT ) * p.pickStep1, bAbove );
}

// the linear model: a, b, shift (from the template's luma in s_tmpl and its
// chroma in t and l)
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
// LMCS's chroma residual scaling (Reshape::calculateChromaAdjVpduNei, AreaBuf::scaleSignal): the average of the luma
// (still mapped) left of and above the CU at word 8, 1 << log2 count samples a side (rows and columns past the
// picture: its last), picks the scale
// --------------------------------------------------------------------------------------------------------------------

groupshared int s_lsum[4];   // the luma sums of the workgroup's waves

// this thread's share of the luma: the left sample tid and the above one
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

// the shares summed a wave at a time into s_lsum (all the threads, before a barrier)
void lmcsShare( int tid, int luma )
{
  const int sum = WaveActiveSum( luma );
  if( WaveIsFirstLane() )
  {
    s_lsum[tid / int( WaveGetLaneCount() )] = sum;
  }
}

// after that barrier: the scale
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

// a residual scaled (clipped to the bit depth's range first)
int lmcsResidual( int r, int scale )
{
  const int range = 1 << pc.bitDepth;
  const int c     = clip3( -range, range - 1, r );
  const int v     = ( abs( c ) * scale + ( 1 << 10 ) ) >> 11;
  return clip3( -32768, 32767, c < 0 ? -v : v );
}

// a job (a block of a component, or a band of its rows): the workgroup's
// threads read what the block needs of the picture together, then share out
// the block's samples
void make( int tid, uint job[12] )
{
  const int   x0     = int( job[0] & 0xffff );
  const int   y0     = int( job[0] >> 16 );
  const int   w      = int( job[1] & 255 );
  const int   h      = int( ( job[1] >> 8 ) & 255 );
  const int   comp   = int( ( job[1] >> 16 ) & 255 );
  const int   kind   = int( job[1] >> 24 );
  const int   mode   = int( job[2] & 255 );
  const int   flags  = int( ( job[2] >> 8 ) & 255 );
  const int   refLine = int( ( job[2] >> 24 ) & 3 );   // MRL
  const int   maxVal = ( 1 << pc.bitDepth ) - 1;
  const bool  filtered = ( flags & 2 ) != 0;
  const bool  residual = ( flags & 1 ) != 0;
  const int   log2W  = log2Int( w );
  // the workgroup's samples: the block's rows first .. first + count, m of
  // them (256 at most: four a thread), the i-th the block's sample s0 + i
  const int   s0     = int( job[7] & 0xffff ) * w;
  const int   m      = int( job[7] >> 16 ) * w;

  int16_t*  dst    = planeOf( comp );
  const int stride = strideOf( comp );
  // LMCS: the chroma residual scaled
  const uint lmcsInfo = job[9];
  const bool lmcs     = ( lmcsInfo & 1 ) != 0;

  if( kind == 3 || kind == 5 )
  {
    // IBC: the block from elsewhere in the picture, plus the residual; an
    // inter CU's chroma under LMCS: its prediction, kept aside, plus the
    // residual (the reads together, then the writes)
    const int bvx = int( int16_t( job[6] & 0xffff ) );
    const int bvy = int( int16_t( job[6] >> 16 ) );
    int16_t*  src = kind == 5 ? ( comp == 1 ? pc.defer1 : pc.defer2 ) : dst;
    int       pred[4];
    int       res[4];
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      const int i = tid + 64 * k;
      const int x = ( s0 + i ) & ( w - 1 );
      const int y = ( s0 + i ) >> log2W;
      pred[k]     = i < m ? ( kind == 5 ? int( src[( y0 + y ) * stride + x0 + x] ) : sampleAt( comp, x0 + x + bvx, y0 + y + bvy ) ) : 0;
      res[k]      = i < m && residual ? int( dst[( y0 + y ) * stride + x0 + x] ) : 0;
    }
    int scale = 0;
    if( lmcs )
    {
      lmcsShare( tid, lmcsLuma( tid, job[8], lmcsInfo ) );
      GroupMemoryBarrierWithGroupSync();
      scale = lmcsScale( lmcsInfo );
    }
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      const int i = tid + 64 * k;
      if( i < m )
      {
        const int at = ( y0 + ( ( s0 + i ) >> log2W ) ) * stride + x0 + ( ( s0 + i ) & ( w - 1 ) );
        dst[at]      = int16_t( residual ? clip3( 0, maxVal, pred[k] + ( lmcs ? lmcsResidual( res[k], scale ) : res[k] ) ) : pred[k] );
      }
    }
    return;   // the whole workgroup
  }

  // the block the prediction covers (pw x ph, the job's block xoff samples
  // into it), the size the wide-angle modes follow and the references'
  // lengths: the job's block's, but for ISP's sub-partitions (flag 8): the
  // CU's size, the CU's width (height) on top of the block's along the top
  // (left), and for a vertical split's sub-partitions 1 or 2 samples wide a
  // block 4 wide, predicted for them all (CU::isPredRegDiffFromTB)
  const bool isp   = ( flags & 8 ) != 0;
  const int  cx    = int( job[4] & 0xffff );
  const int  cy    = int( job[4] >> 16 );
  const int  cw    = int( job[5] & 255 );
  const int  ch    = int( ( job[5] >> 8 ) & 255 );
  const bool ver   = ( ( job[5] >> 16 ) & 3 ) == 2;
  const bool region = isp && ver && ( cw == 4 || ( cw == 8 && ch > 4 ) );
  const int  pw    = region ? 4 : w;
  const int  ph    = h;
  const int  xoff  = region ? ( x0 - cx ) & 3 : 0;
  const int  sizeW = isp ? cw : pw;
  const int  sizeH = isp ? ch : ph;
  const int  topLength  = isp ? cw + pw : pw << 1;
  const int  leftLength = isp ? ch + ph : ph << 1;

  // the reads: the references, then (while those are on their way) the
  // residual and CCLM's luma, kCubic for luma's angular modes
  const uint       nb   = job[3];
  References       refs;
  if( isp )
  {
    // the CU's fill, as a block of its size
    const RefFill f = refFill( comp, cx, cy, cw << 1, ch << 1, int( nb & 255 ), int( ( nb >> 8 ) & 255 ), int( ( nb >> 16 ) & 255 ), 0 );
    refs = loadReferencesIsp( tid, f, x0 - xoff, y0, pw, ph, cx, cy, cw, ch, ver, ( ( job[5] >> 18 ) & 1 ) != 0, ( ( job[5] >> 19 ) & 1 ) != 0 );
  }
  else
  {
    refs = loadReferences( tid, refFill( comp, x0, y0, w << 1, h << 1, int( nb & 255 ), int( ( nb >> 8 ) & 255 ), int( ( nb >> 16 ) & 255 ), refLine ) );
  }
  const bool       lmAbove = ( ( job[4] >> 16 ) & 1 ) != 0;
  const bool       lmLeft  = ( ( job[4] >> 17 ) & 1 ) != 0;
  if( kind == 2 )
  {
    // the downsampled luma of the samples and of the template's (a thread each)
    const CclmPicks picks = cclmPicks( job[4], job[5] );
    if( tid < picks.cntT + picks.cntL )
    {
      s_tmpl[tid] = cclmTemplateLuma( x0, y0, job[4], picks, tid );
    }
  }
  if( kind == 2 || residual )
  {
    int luma[4];
    int res[4];
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      const int i = tid + 64 * k;
      const int x = ( s0 + i ) & ( w - 1 );
      const int y = ( s0 + i ) >> log2W;
      luma[k]     = i < m && kind == 2 ? lumaDown( x0 << 1, y0 << 1, x, y, lmAbove, lmLeft ) : 0;
      res[k]      = i < m && residual ? int( dst[( y0 + y ) * stride + x0 + x] ) : 0;
    }
    [unroll]
    for( int k = 0; k < 4; k++ )
    {
      const int i = tid + 64 * k;
      if( i < m )
      {
        s_luma[i] = int16_t( luma[k] );
        s_res[i]  = int16_t( res[k] );
      }
    }
  }
  if( kind == 0 && comp == 0 )
  {
    s_cubic[tid]      = kCubic[tid];
    s_cubic[tid + 64] = kCubic[tid + 64];
  }
  if( lmcs )
  {
    lmcsShare( tid, lmcsLuma( tid, job[8], lmcsInfo ) );
  }
  storeReferences( tid, topLength + refLine, leftLength + refLine, refs );
  GroupMemoryBarrierWithGroupSync();
  const int lmcsScaling = lmcs ? lmcsScale( lmcsInfo ) : 0;

  // per block: what the predictor needs of the references
  if( filtered )
  {
    filterReferences( tid, w, h );
    GroupMemoryBarrierWithGroupSync();
  }
  Angular ang;
  if( kind == 2 )
  {
    if( tid == 0 )
    {
      s_model = cclmParameters( job[4], job[5] );
    }
    GroupMemoryBarrierWithGroupSync();
  }
  else if( kind == 0 && mode == DC_IDX )
  {
    if( tid == 0 )
    {
      s_dc = dcValue( filtered, pw, ph, refLine );
    }
    GroupMemoryBarrierWithGroupSync();
  }
  else if( kind == 0 && mode != PLANAR_IDX )
  {
    ang = setupAngular( comp, pw, ph, mode, refLine, sizeW, sizeH, topLength, leftLength, isp );
  }
  const int mipPred = mipSizeId( w, h ) < 2 ? 4 : 8;
  if( kind == 4 )
  {
    // MIP: the reduced boundary, then the reduced prediction (a thread an element)
    const int sizeId = mipSizeId( w, h );
    const int bdry   = sizeId == 0 ? 2 : 4;
    if( tid < 2 * bdry )
    {
      s_luma[MIP_BDRY + tid] = int16_t( mipBoundary( tid, bdry, w, h ) );
    }
    GroupMemoryBarrierWithGroupSync();
    if( tid < mipPred * mipPred )
    {
      mipReducedPred( tid, sizeId, mode, ( flags & 4 ) != 0 );
    }
    GroupMemoryBarrierWithGroupSync();
  }
  const int  dc    = s_dc;
  const int3 model = s_model;

  const int  bdpcmDir = int( ( job[2] >> 16 ) & 255 );
  const bool doPdpcPlanarDc = kind == 0 && ( mode == PLANAR_IDX || mode == DC_IDX ) && pw >= 4 && ph >= 4 && refLine == 0;

  // the samples, from shared memory alone
  for( int i = tid; i < m; i += 64 )
  {
    const int x = ( s0 + i ) & ( w - 1 );
    const int y = ( s0 + i ) >> log2W;
    int       v;
    if( kind == 2 )
    {
      v = clip3( 0, maxVal, ( ( model.x * int( s_luma[i] ) ) >> model.z ) + model.y );
    }
    else if( kind == 1 )
    {
      v = bdpcmDir == 1 ? refL( false, y + 1 ) : refT( false, x + 1 );
    }
    else if( kind == 4 )
    {
      v = mipSample( x, y, w, h, mipPred );
    }
    else if( mode == PLANAR_IDX || mode == DC_IDX )
    {
      v = mode == PLANAR_IDX ? predictPlanar( filtered, pw, ph, x + xoff, y ) : dc;
      if( doPdpcPlanarDc )
      {
        v = pdpcPlanarDc( filtered, pw, ph, x + xoff, y, v );
      }
    }
    else
    {
      v = ang.isVer ? predictAngular( ang, comp, filtered, x + xoff, y, maxVal ) : predictAngular( ang, comp, filtered, y, x + xoff, maxVal );
    }
    const int at = ( y0 + y ) * stride + x0 + x;
    dst[at]      = int16_t( residual ? clip3( 0, maxVal, v + ( lmcs ? lmcsResidual( int( s_res[i] ), lmcsScaling ) : int( s_res[i] ) ) ) : v );
  }
}

groupshared uint s_job;   // the job the workgroup took

// a workgroup a job: the next one, in the order of the waves
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
  if( index >= uint( pc.count ) )
  {
    return;   // the whole workgroup (the dispatch, in rows of 32768 workgroups, has more than the jobs)
  }
  // the job on its way while the workgroup waits for the jobs before
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
  // the barrier's acquire: what those jobs wrote, visible to all the threads
  AllMemoryBarrierWithGroupSync();
  uint job[12] = { a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, c.x, c.y, c.z, c.w };
  make( tid, job );
  // the barrier's release: the threads' writes done before the job counts as done
  AllMemoryBarrierWithGroupSync();
  if( tid == 0 )
  {
    pc.sync[1].add( 1u, MemoryOrder.Release );
  }
}
