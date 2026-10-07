// Intra prediction, intra block copy (IBC) and the reconstruction of their
// transform blocks, as vvdec's IntraPrediction and DecCu::predAndReco do it: a
// thread predicts one transform block of one component, adds its residual
// (already in the picture, where vvdec's TrQuant left it) and writes the
// result. The blocks come in waves: a block's wave is above the waves of
// every block whose samples it reads, so a dispatch per wave keeps the order.
//
// A job, 8 uints (VulkanLib/GpuRecon.h, GpuIntraJob):
//   0: x | y << 16                (in samples of its component)
//   1: width | height << 8 | component << 16 | kind << 24   (kind: 0 planar/DC/angular, 1 BDPCM, 2 CCLM, 3 IBC)
//   2: mode | flags << 8 | bdpcm direction << 16            (flags: 1 residual, 2 filtered references)
//   3: top-left units | above units << 8 | left units << 16  (the reference samples available)
//   4: CCLM: luma above units | luma left units << 8 | bAbove << 16 | bLeft << 17 | aboveAvail << 18 | leftAvail << 19 | lm mode << 24
//   5: CCLM: actual top template samples | actual left << 16
//   6: IBC: block vector x | y << 16 (int16, samples of the component)
//   7: wave

#include "common.slang"

struct Params
{
  int16_t* plane0;
  int16_t* plane1;
  int16_t* plane2;
  uint*    jobs;
  int      stride0;
  int      stride12;
  int      first;       // the jobs of this wave: jobs[first .. first + count)
  int      count;
  int      bitDepth;
  int      ctuLog2;     // luma
  int      scaleX;      // chroma subsampling
  int      scaleY;
  int      collocated;  // sps CCLM collocated chroma
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
// column to the left (below-left included)
static int t[MAX_REF];
static int l[MAX_REF];
static int tf[MAX_REF];
static int lf[MAX_REF];

void fillReferences( int comp, int x0, int y0, int w, int h, int nTL, int nAbove, int nLeft )
{
  const int predSize  = w << 1;
  const int predHSize = h << 1;
  const int unitW     = comp == 0 ? 4 : ( 4 >> pc.scaleX );
  const int unitH     = comp == 0 ? 4 : ( 4 >> pc.scaleY );
  const int totalUnits = predSize / unitW + predHSize / unitH + 1;
  const int numNeighbor = nTL + nAbove + nLeft;

  if( numNeighbor == 0 )
  {
    const int dc = 1 << ( pc.bitDepth - 1 );
    for( int j = 0; j <= predSize; j++ )
    {
      t[j] = dc;
    }
    for( int i = 0; i <= predHSize; i++ )
    {
      l[i] = dc;
    }
    return;
  }
  if( numNeighbor == totalUnits )
  {
    for( int j = 0; j <= predSize; j++ )
    {
      t[j] = sampleAt( comp, x0 - 1 + j, y0 - 1 );
    }
    l[0] = t[0];
    for( int i = 1; i <= predHSize; i++ )
    {
      l[i] = sampleAt( comp, x0 - 1, y0 + i - 1 );
    }
    return;
  }
  if( nLeft > 0 )
  {
    const int tmpSize = min( nLeft * unitH, predHSize );
    for( int i = 0; i < tmpSize; i++ )
    {
      l[1 + i] = sampleAt( comp, x0 - 1, y0 + i );
    }
    for( int i = tmpSize; i < predHSize; i++ )
    {
      l[1 + i] = l[tmpSize];
    }
    t[0] = nTL != 0 ? sampleAt( comp, x0 - 1, y0 - 1 ) : sampleAt( comp, x0 - 1, y0 );
    if( nAbove != 0 )
    {
      const int tmpA = min( nAbove * unitW, predSize );
      for( int i = 0; i < tmpA; i++ )
      {
        t[1 + i] = sampleAt( comp, x0 + i, y0 - 1 );
      }
      for( int i = tmpA; i < predSize; i++ )
      {
        t[1 + i] = t[tmpA];
      }
    }
    else
    {
      for( int i = 0; i < predSize; i++ )
      {
        t[1 + i] = t[0];
      }
    }
  }
  else
  {
    const int tmpA = min( nAbove * unitW, predSize );
    for( int i = 0; i < tmpA; i++ )
    {
      t[1 + i] = sampleAt( comp, x0 + i, y0 - 1 );
    }
    for( int i = tmpA; i < predSize; i++ )
    {
      t[1 + i] = t[tmpA];
    }
    const int first = sampleAt( comp, x0, y0 - 1 );
    t[0] = first;
    for( int i = 0; i < predHSize; i++ )
    {
      l[1 + i] = first;
    }
  }
  l[0] = t[0];
}

void filterReferences( int w, int h )
{
  const int predSize  = w << 1;
  const int predHSize = h << 1;
  lf[predHSize] = l[predHSize];
  for( int i = predHSize - 1; i >= 1; i-- )
  {
    lf[i] = ( l[i + 1] + 2 * l[i] + l[i - 1] + 2 ) >> 2;
  }
  tf[0] = ( l[1] + 2 * t[0] + t[1] + 2 ) >> 2;
  lf[0] = tf[0];
  for( int i = 1; i < predSize; i++ )
  {
    tf[i] = ( t[i + 1] + 2 * t[i] + t[i - 1] + 2 ) >> 2;
  }
  tf[predSize] = t[predSize];
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
  bool cubic;          // luma: the cubic filter, else the Gaussian one
  int  pdpcScale;      // angle 0: PDPC's scale, lev its reach; angle > 0: angularScale (< 0: no PDPC)
  int  lev;
};

// refMain[k] and refSide[k] of xPredIntraAng
int refMain( Angular a, bool filtered, int k )
{
  return a.isVer ? refT( filtered, k ) : refL( filtered, k );
}

int refSide( Angular a, bool filtered, int k )
{
  return a.isVer ? refL( filtered, k ) : refT( filtered, k );
}

Angular setupAngular( int comp, int w, int h, int mode )
{
  Angular a;
  a.predMode = wideAngle( w, h, mode );
  a.isVer    = a.predMode >= DIA_IDX;
  const int angMode    = a.isVer ? a.predMode - VER_IDX : -( a.predMode - HOR_IDX );
  const int absAngMode = abs( angMode );
  a.invAngle  = kInvAngTable[absAngMode];
  a.absAng    = kAngTable[absAngMode];
  a.angle     = angMode < 0 ? -a.absAng : a.absAng;
  a.width     = a.isVer ? w : h;
  a.height    = a.isVer ? h : w;
  a.refLength = a.isVer ? ( w << 1 ) : ( h << 1 );
  const int diff     = min( abs( a.predMode - HOR_IDX ), abs( a.predMode - VER_IDX ) );
  const int log2Size = ( log2Int( a.width ) + log2Int( a.height ) ) >> 1;
  a.cubic = !( diff > kIntraFilterLuma[log2Size] );
  const bool doPdpc = w >= 4 && h >= 4;
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
    v = refMain( a, filtered, x + 1 );
    if( a.pdpcScale >= 0 && x < a.lev )
    {
      const int wL = 32 >> min( 31, ( x << 1 ) >> a.pdpcScale );
      v = clip3( 0, maxVal, ( wL * ( refSide( a, filtered, y + 1 ) - refT( filtered, 0 ) ) + v * 64 + 32 ) >> 6 );
    }
    return v;
  }

  const int deltaPos   = a.angle * ( y + 1 );
  const int deltaInt   = deltaPos >> 5;
  const int deltaFract = deltaPos & 31;
  int       p[4];
  for( int k = 0; k < 4; k++ )
  {
    // refMain extended to the left from the side reference (negative angles),
    // to the right by replicating its last sample
    const int idx = deltaInt + x + k;
    if( idx < 0 )
    {
      p[k] = refSide( a, filtered, min( ( -idx * a.invAngle + 256 ) >> 9, a.height ) );
    }
    else
    {
      p[k] = refMain( a, filtered, a.angle < 0 ? idx : min( idx, a.refLength ) );
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
      v = clip3( 0, maxVal, ( kCubic[f] * p[0] + kCubic[f + 1] * p[1] + kCubic[f + 2] * p[2] + kCubic[f + 3] * p[3] + 32 ) >> 6 );
    }
    else
    {
      v = ( kGauss[f] * p[0] + kGauss[f + 1] * p[1] + kGauss[f + 2] * p[2] + kGauss[f + 3] * p[3] + 32 ) >> 6;
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

int dcValue( bool filtered, int w, int h )
{
  const int denom = w == h ? ( w << 1 ) : max( w, h );
  int       sum   = 0;
  if( w >= h )
  {
    for( int i = 0; i < w; i++ )
    {
      sum += refT( filtered, 1 + i );
    }
  }
  if( w <= h )
  {
    for( int i = 0; i < h; i++ )
    {
      sum += refL( filtered, 1 + i );
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

// the linear model: a, b, shift
int3 cclmParameters( int x0, int y0, uint lm, uint lm2 )
{
  const int  lx             = x0 << 1;
  const int  ly             = y0 << 1;
  const bool bAbove         = ( ( lm >> 16 ) & 1 ) != 0;
  const bool bLeft          = ( ( lm >> 17 ) & 1 ) != 0;
  const bool aboveAvailable = ( ( lm >> 18 ) & 1 ) != 0;
  const bool leftAvailable  = ( ( lm >> 19 ) & 1 ) != 0;
  const int  actualTop      = int( lm2 & 0xffff );
  const int  actualLeft     = int( lm2 >> 16 );

  const int aboveIs4  = leftAvailable ? 0 : 1;
  const int leftIs4   = aboveAvailable ? 0 : 1;
  const int startPos0 = actualTop >> ( 2 + aboveIs4 );
  const int pickStep0 = max( 1, actualTop >> ( 1 + aboveIs4 ) );
  const int startPos1 = actualLeft >> ( 2 + leftIs4 );
  const int pickStep1 = max( 1, actualLeft >> ( 1 + leftIs4 ) );

  int selLuma[4]   = { 0, 0, 0, 0 };
  int selChroma[4] = { 0, 0, 0, 0 };
  int cntT = 0, cntL = 0;
  if( aboveAvailable )
  {
    cntT = min( actualTop, ( 1 + aboveIs4 ) << 1 );
    for( int k = 0; k < cntT; k++ )
    {
      const int pos = startPos0 + k * pickStep0;
      selLuma[k]    = lumaTemplateAbove( lx, ly, pos, bLeft );
      selChroma[k]  = t[1 + pos];
    }
  }
  if( leftAvailable )
  {
    cntL = min( actualLeft, ( 1 + leftIs4 ) << 1 );
    for( int k = 0; k < cntL; k++ )
    {
      const int pos       = startPos1 + k * pickStep1;
      selLuma[cntT + k]   = lumaTemplateLeft( lx, ly, pos, bAbove );
      selChroma[cntT + k] = l[1 + pos];
    }
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

[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  if( int( id.x ) >= pc.count )
  {
    return;
  }
  uint*       job    = pc.jobs + ( pc.first + int( id.x ) ) * 8;
  const int   x0     = int( job[0] & 0xffff );
  const int   y0     = int( job[0] >> 16 );
  const int   w      = int( job[1] & 255 );
  const int   h      = int( ( job[1] >> 8 ) & 255 );
  const int   comp   = int( ( job[1] >> 16 ) & 255 );
  const int   kind   = int( job[1] >> 24 );
  const int   mode   = int( job[2] & 255 );
  const int   flags  = int( ( job[2] >> 8 ) & 255 );
  const int   maxVal = ( 1 << pc.bitDepth ) - 1;
  const bool  filtered = ( flags & 2 ) != 0;
  const bool  residual = ( flags & 1 ) != 0;

  int16_t*  dst    = planeOf( comp );
  const int stride = strideOf( comp );

  // per block: the references and what the predictor needs of them
  Angular ang;
  int     dc    = 0;
  int3    model = int3( 0, 0, 0 );
  bool    lmAbove = false, lmLeft = false;
  if( kind != 3 )
  {
    const uint nb = job[3];
    fillReferences( comp, x0, y0, w, h, int( nb & 255 ), int( ( nb >> 8 ) & 255 ), int( ( nb >> 16 ) & 255 ) );
    if( filtered )
    {
      filterReferences( w, h );
    }
    if( kind == 2 )
    {
      model   = cclmParameters( x0, y0, job[4], job[5] );
      lmAbove = ( ( job[4] >> 16 ) & 1 ) != 0;
      lmLeft  = ( ( job[4] >> 17 ) & 1 ) != 0;
    }
    else if( kind == 0 && mode == DC_IDX )
    {
      dc = dcValue( filtered, w, h );
    }
    else if( kind == 0 && mode != PLANAR_IDX )
    {
      ang = setupAngular( comp, w, h, mode );
    }
  }

  const int  bdpcmDir = int( ( job[2] >> 16 ) & 255 );
  const bool doPdpcPlanarDc = kind == 0 && ( mode == PLANAR_IDX || mode == DC_IDX ) && w >= 4 && h >= 4;
  const int  bvx = int( int16_t( job[6] & 0xffff ) );
  const int  bvy = int( int16_t( job[6] >> 16 ) );

  for( int y = 0; y < h; y++ )
  {
    for( int x = 0; x < w; x++ )
    {
      int v;
      if( kind == 3 )
      {
        v = sampleAt( comp, x0 + x + bvx, y0 + y + bvy );
      }
      else if( kind == 2 )
      {
        v = clip3( 0, maxVal, ( ( model.x * lumaDown( x0 << 1, y0 << 1, x, y, lmAbove, lmLeft ) ) >> model.z ) + model.y );
      }
      else if( kind == 1 )
      {
        v = bdpcmDir == 1 ? refL( false, y + 1 ) : refT( false, x + 1 );
      }
      else if( mode == PLANAR_IDX || mode == DC_IDX )
      {
        v = mode == PLANAR_IDX ? predictPlanar( filtered, w, h, x, y ) : dc;
        if( doPdpcPlanarDc )
        {
          v = pdpcPlanarDc( filtered, w, h, x, y, v );
        }
      }
      else
      {
        v = ang.isVer ? predictAngular( ang, comp, filtered, x, y, maxVal ) : predictAngular( ang, comp, filtered, y, x, maxVal );
      }
      const int i = ( y0 + y ) * stride + x0 + x;
      dst[i]      = int16_t( residual ? clip3( 0, maxVal, v + int( dst[i] ) ) : v );
    }
  }
}
