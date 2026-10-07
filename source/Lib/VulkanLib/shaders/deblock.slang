// The deblocking filter of a picture, in place, as vvdec's LoopFilter does it
// (xEdgeFilterLuma, xEdgeFilterChroma): all vertical edges (dir 0), then, in
// another dispatch, all horizontal ones (dir 1). VVC keeps the edges of one
// direction apart enough that each can be filtered on its own.
//
// CHROMA=0: a thread filters 4 lines across one luma edge of a 4x4 block.
// CHROMA=1: a thread filters the lines across one chroma edge segment, for Cb
//           and Cr, at the positions vvdec's xDeblockCtuArea visits.
//
// The edge parameters are vvdec's LoopFilterParam (6 bytes: qp[3], bs,
// sideMaxFiltLength, flags), per 4x4 luma block and direction, CTU by CTU.
// Per CTU, 8 bytes: deblocking disabled, then the beta and tc offsets (div 2)
// of Y, Cb, Cr.

#include "common.slang"

struct Params
{
  int16_t* pic0;        // luma, or Cb
  int16_t* pic1;        // Cr
  uint8_t* lfp;
  int8_t*  ctus;
  int      stride;
  int      width;       // of the plane
  int      height;
  int      ctuLog2;     // luma
  int      widthInCtus;
  int      bitDepth;
  int      num4x4;      // LoopFilterParams per CTU and direction
  int      dir;         // 0: vertical edges, 1: horizontal
  int      scaleX;      // chroma subsampling
  int      scaleY;
};

[[vk::push_constant]] Params pc;

static const int kTcTable[66] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4, 4, 5, 5, 5, 5, 7, 7, 8, 9, 10, 10, 11, 13, 14, 15, 17, 19,
  21, 24, 25, 29, 33, 36, 41, 45, 51, 57, 64, 71, 80, 89, 100, 112, 125, 141, 157, 177, 198, 222, 250, 280, 314, 352, 395
};
static const int kBetaTable[64] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32,
  34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88
};

#define MAX_QP 63

// the LoopFilterParam of the 4x4 luma block at (x, y) (luma samples)
int lfpOffset( int x, int y )
{
  const int ctu    = ( y >> pc.ctuLog2 ) * pc.widthInCtus + ( x >> pc.ctuLog2 );
  const int n      = 1 << ( pc.ctuLog2 - 2 );
  const int within = ( ( y >> 2 ) & ( n - 1 ) ) * n + ( ( x >> 2 ) & ( n - 1 ) );
  return ( pc.num4x4 * ( 2 * ctu + pc.dir ) + within ) * 6;
}

int ctuParam( int x, int y, int i )
{
  return int( pc.ctus[( ( y >> pc.ctuLog2 ) * pc.widthInCtus + ( x >> pc.ctuLog2 ) ) * 8 + i] );
}

int tcFromTable( int index )
{
  const int bd = pc.bitDepth;
  return bd < 10 ? ( ( kTcTable[index] + ( 1 << ( 9 - bd ) ) ) >> ( 10 - bd ) ) : ( kTcTable[index] << ( bd - 10 ) );
}

// a line across the edge: s(k) for k < 0 are the P samples (p0 = s(-1)), k >= 0 the Q ones (q0 = s(0))
int rd( int16_t* p, int base, int off, int k )
{
  return int( p[base + k * off] );
}

void wr( int16_t* p, int base, int off, int k, int v )
{
  p[base + k * off] = int16_t( v );
}

bool useStrongFiltering( int16_t* p, int base, int off, int d, int beta, int tc, bool largeP, bool largeQ, int maxP, int maxQ, bool chromaCtbBoundary )
{
  const int p0 = rd( p, base, off, -1 );
  const int q0 = rd( p, base, off, 0 );
  if( !( d < ( beta >> 2 ) && abs( p0 - q0 ) < ( ( tc * 5 + 1 ) >> 1 ) ) )
  {
    return false;
  }
  const int p3 = rd( p, base, off, -4 );
  const int q3 = rd( p, base, off, 3 );
  const int p1 = rd( p, base, off, -2 );
  int sp3 = chromaCtbBoundary ? abs( p1 - p0 ) : abs( p3 - p0 );
  int sq3 = abs( q3 - q0 );
  const int dStrong = sp3 + sq3;
  if( largeP || largeQ )
  {
    if( largeP )
    {
      const int pMax = rd( p, base, off, -maxP - 1 );
      if( maxP == 7 )
      {
        sp3 = sp3 + abs( rd( p, base, off, -5 ) - rd( p, base, off, -6 ) - rd( p, base, off, -7 ) + pMax );
      }
      sp3 = ( sp3 + abs( p3 - pMax ) + 1 ) >> 1;
    }
    if( largeQ )
    {
      const int qMax = rd( p, base, off, maxQ );
      if( maxQ == 7 )
      {
        sq3 = sq3 + abs( rd( p, base, off, 4 ) - rd( p, base, off, 5 ) - rd( p, base, off, 6 ) + qMax );
      }
      sq3 = ( sq3 + abs( qMax - q3 ) + 1 ) >> 1;
    }
    return ( sp3 + sq3 ) < ( ( beta * 3 ) >> 5 ) && d < ( beta >> 4 ) && abs( p0 - q0 ) < ( ( tc * 5 + 1 ) >> 1 );
  }
  return dStrong < ( beta >> 3 );
}

// |p2 - 2 p1 + p0|, or |p0 - p1| at a horizontal chroma CTU boundary
int calcDP( int16_t* p, int base, int off, bool chromaCtbBoundary )
{
  if( chromaCtbBoundary )
  {
    return abs( rd( p, base, off, -2 ) - 2 * rd( p, base, off, -2 ) + rd( p, base, off, -1 ) );
  }
  return abs( rd( p, base, off, -3 ) - 2 * rd( p, base, off, -2 ) + rd( p, base, off, -1 ) );
}

int calcDQ( int16_t* p, int base, int off )
{
  return abs( rd( p, base, off, 0 ) - 2 * rd( p, base, off, 1 ) + rd( p, base, off, 2 ) );
}

#if CHROMA == 0

static const int kDbCoeffs7[7] = { 59, 50, 41, 32, 23, 14, 5 };
static const int kDbCoeffs5[5] = { 58, 45, 32, 19, 6 };
static const int kDbCoeffs3[3] = { 53, 32, 11 };
static const int kTc7[7]       = { 6, 5, 4, 3, 2, 1, 1 };
static const int kTc3[3]       = { 6, 4, 2 };

int dbCoeff( int n, int pos )
{
  return n == 7 ? kDbCoeffs7[pos] : ( n == 5 ? kDbCoeffs5[pos] : kDbCoeffs3[pos] );
}

// the long filters (xFilteringPandQCore), one line
void filteringPandQ( int base, int off, int nP, int nQ, int tc )
{
  int ps[8], qs[8];
  for( int k = 0; k < 8; k++ )
  {
    ps[k] = rd( pc.pic0, base, off, -1 - k );
    qs[k] = rd( pc.pic0, base, off, k );
  }
  const int refP = ( ps[nP - 1] + ps[nP] + 1 ) >> 1;
  const int refQ = ( qs[nQ - 1] + qs[nQ] + 1 ) >> 1;
  int refMiddle;
  if( nP == nQ )
  {
    if( nP == 5 )
    {
      refMiddle = ( 2 * ( ps[0] + qs[0] + ps[1] + qs[1] + ps[2] + qs[2] ) + ps[3] + qs[3] + ps[4] + qs[4] + 8 ) >> 4;
    }
    else
    {
      refMiddle = ( 2 * ( ps[0] + qs[0] ) + ps[1] + qs[1] + ps[2] + qs[2] + ps[3] + qs[3] + ps[4] + qs[4] + ps[5] + qs[5] + ps[6] + qs[6] + 8 ) >> 4;
    }
  }
  else
  {
    const int longN  = max( nP, nQ );
    const int shortN = min( nP, nQ );
    if( longN == 7 && shortN == 5 )
    {
      refMiddle = ( 2 * ( ps[0] + qs[0] + ps[1] + qs[1] ) + ps[2] + qs[2] + ps[3] + qs[3] + ps[4] + qs[4] + ps[5] + qs[5] + 8 ) >> 4;
    }
    else if( longN == 7 && shortN == 3 )
    {
      int ls[7], ss[3];
      for( int k = 0; k < 7; k++ )
      {
        ls[k] = nP == 7 ? ps[k] : qs[k];
      }
      for( int k = 0; k < 3; k++ )
      {
        ss[k] = nP == 7 ? qs[k] : ps[k];
      }
      refMiddle = ( 2 * ( ls[0] + ss[0] ) + ss[0] + 2 * ( ss[1] + ss[2] ) + ls[1] + ss[1] + ls[2] + ls[3] + ls[4] + ls[5] + ls[6] + 8 ) >> 4;
    }
    else
    {
      refMiddle = ( ps[0] + qs[0] + ps[1] + qs[1] + ps[2] + qs[2] + ps[3] + qs[3] + 4 ) >> 3;
    }
  }
  for( int pos = 0; pos < nP; pos++ )
  {
    const int src    = ps[pos];
    const int cvalue = ( tc * ( nP == 3 ? kTc3[pos] : kTc7[pos] ) ) >> 1;
    const int c      = dbCoeff( nP, pos );
    wr( pc.pic0, base, off, -1 - pos, clip3( src - cvalue, src + cvalue, ( refMiddle * c + refP * ( 64 - c ) + 32 ) >> 6 ) );
  }
  for( int pos = 0; pos < nQ; pos++ )
  {
    const int src    = qs[pos];
    const int cvalue = ( tc * ( nQ == 3 ? kTc3[pos] : kTc7[pos] ) ) >> 1;
    const int c      = dbCoeff( nQ, pos );
    wr( pc.pic0, base, off, pos, clip3( src - cvalue, src + cvalue, ( refMiddle * c + refQ * ( 64 - c ) + 32 ) >> 6 ) );
  }
}

// the normal filters (xPelFilterLumaCorePel), one line
void pelFilterLuma( int base, int off, int tc, bool sw, int thrCut, bool filterSecondP, bool filterSecondQ, int maxVal )
{
  const int m1 = rd( pc.pic0, base, off, -3 );
  const int m2 = rd( pc.pic0, base, off, -2 );
  const int m3 = rd( pc.pic0, base, off, -1 );
  const int m4 = rd( pc.pic0, base, off, 0 );
  const int m5 = rd( pc.pic0, base, off, 1 );
  const int m6 = rd( pc.pic0, base, off, 2 );
  if( sw )
  {
    const int m0 = rd( pc.pic0, base, off, -4 );
    const int m7 = rd( pc.pic0, base, off, 3 );
    wr( pc.pic0, base, off, -3, clip3( m1 - tc, m1 + tc, ( 2 * m0 + 3 * m1 + m2 + m3 + m4 + 4 ) >> 3 ) );
    wr( pc.pic0, base, off, -2, clip3( m2 - 2 * tc, m2 + 2 * tc, ( m1 + m2 + m3 + m4 + 2 ) >> 2 ) );
    wr( pc.pic0, base, off, -1, clip3( m3 - 3 * tc, m3 + 3 * tc, ( m1 + 2 * m2 + 2 * m3 + 2 * m4 + m5 + 4 ) >> 3 ) );
    wr( pc.pic0, base, off, 0, clip3( m4 - 3 * tc, m4 + 3 * tc, ( m2 + 2 * m3 + 2 * m4 + 2 * m5 + m6 + 4 ) >> 3 ) );
    wr( pc.pic0, base, off, 1, clip3( m5 - 2 * tc, m5 + 2 * tc, ( m3 + m4 + m5 + m6 + 2 ) >> 2 ) );
    wr( pc.pic0, base, off, 2, clip3( m6 - tc, m6 + tc, ( m3 + m4 + m5 + 3 * m6 + 2 * m7 + 4 ) >> 3 ) );
  }
  else
  {
    int delta = ( 9 * ( m4 - m3 ) - 3 * ( m5 - m2 ) + 8 ) >> 4;
    if( abs( delta ) < thrCut )
    {
      delta         = clip3( -tc, tc, delta );
      const int tc2 = tc >> 1;
      wr( pc.pic0, base, off, -1, clip3( 0, maxVal, m3 + delta ) );
      if( filterSecondP )
      {
        const int delta1 = clip3( -tc2, tc2, ( ( ( m1 + m3 + 1 ) >> 1 ) - m2 + delta ) >> 1 );
        wr( pc.pic0, base, off, -2, clip3( 0, maxVal, m2 + delta1 ) );
      }
      wr( pc.pic0, base, off, 0, clip3( 0, maxVal, m4 - delta ) );
      if( filterSecondQ )
      {
        const int delta2 = clip3( -tc2, tc2, ( ( ( m6 + m4 + 1 ) >> 1 ) - m5 - delta ) >> 1 );
        wr( pc.pic0, base, off, 1, clip3( 0, maxVal, m5 + delta2 ) );
      }
    }
  }
}

[shader( "compute" )]
[numthreads( 8, 8, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  const int x = int( id.x ) * 4;
  const int y = int( id.y ) * 4;
  if( x >= pc.width || y >= pc.height )
  {
    return;
  }
  const int e  = lfpOffset( x, y );
  const int bs = int( pc.lfp[e + 3] ) & 3;
  if( bs == 0 || ctuParam( x, y, 0 ) != 0 )
  {
    return;
  }

  const int qp               = int( int8_t( pc.lfp[e] ) );
  const int sideMax          = int( pc.lfp[e + 4] );
  const int maxFilterLengthP = ( sideMax >> 4 ) & 7;
  const int maxFilterLengthQ = sideMax & 7;
  bool      largeP           = maxFilterLengthP > 3;
  const bool largeQ          = maxFilterLengthQ > 3;
  if( pc.dir == 1 && ( y & ( ( 1 << pc.ctuLog2 ) - 1 ) ) == 0 )
  {
    largeP = false;
  }

  const int betaOffsetDiv2 = ctuParam( x, y, 1 );
  const int tcOffsetDiv2   = ctuParam( x, y, 2 );
  const int indexTc        = clip3( 0, MAX_QP + 2, qp + 2 * ( bs - 1 ) + tcOffsetDiv2 * 2 );
  const int indexB         = clip3( 0, MAX_QP, qp + betaOffsetDiv2 * 2 );
  const int tc             = tcFromTable( indexTc );
  const int beta           = kBetaTable[indexB] << ( pc.bitDepth - 8 );
  const int sideThreshold  = ( beta + ( beta >> 1 ) ) >> 3;
  const int thrCut         = tc * 10;
  const int maxVal         = ( 1 << pc.bitDepth ) - 1;

  // across the edge: off; along it: step
  const int off   = pc.dir == 0 ? 1 : pc.stride;
  const int step  = pc.dir == 0 ? pc.stride : 1;
  const int base0 = y * pc.stride + x;
  const int base3 = base0 + 3 * step;

  const int dp0 = calcDP( pc.pic0, base0, off, false );
  const int dq0 = calcDQ( pc.pic0, base0, off );
  const int dp3 = calcDP( pc.pic0, base3, off, false );
  const int dq3 = calcDQ( pc.pic0, base3, off );
  const int d0  = dp0 + dq0;
  const int d3  = dp3 + dq3;

  if( largeP || largeQ )
  {
    const int off3 = 3 * off;
    const int dp0L = largeP ? ( ( dp0 + calcDP( pc.pic0, base0 - off3, off, false ) + 1 ) >> 1 ) : dp0;
    const int dq0L = largeQ ? ( ( dq0 + calcDQ( pc.pic0, base0 + off3, off ) + 1 ) >> 1 ) : dq0;
    const int dp3L = largeP ? ( ( dp3 + calcDP( pc.pic0, base3 - off3, off, false ) + 1 ) >> 1 ) : dp3;
    const int dq3L = largeQ ? ( ( dq3 + calcDQ( pc.pic0, base3 + off3, off ) + 1 ) >> 1 ) : dq3;
    const int d0L  = dp0L + dq0L;
    const int d3L  = dp3L + dq3L;
    if( d0L + d3L < beta )
    {
      const bool swL = useStrongFiltering( pc.pic0, base0, off, 2 * d0L, beta, tc, largeP, largeQ, maxFilterLengthP, maxFilterLengthQ, false )
                       && useStrongFiltering( pc.pic0, base3, off, 2 * d3L, beta, tc, largeP, largeQ, maxFilterLengthP, maxFilterLengthQ, false );
      if( swL )
      {
        for( int i = 0; i < 4; i++ )
        {
          filteringPandQ( base0 + i * step, off, largeP ? maxFilterLengthP : 3, largeQ ? maxFilterLengthQ : 3, tc );
        }
        return;
      }
    }
  }

  const int dp = dp0 + dp3;
  const int dq = dq0 + dq3;
  const int d  = d0 + d3;
  if( d < beta )
  {
    bool filterP = false;
    bool filterQ = false;
    if( maxFilterLengthP > 1 && maxFilterLengthQ > 1 )
    {
      filterP = dp < sideThreshold;
      filterQ = dq < sideThreshold;
    }
    bool sw = false;
    if( maxFilterLengthP > 2 && maxFilterLengthQ > 2 )
    {
      sw = useStrongFiltering( pc.pic0, base0, off, 2 * d0, beta, tc, false, false, 7, 7, false )
           && useStrongFiltering( pc.pic0, base3, off, 2 * d3, beta, tc, false, false, 7, 7, false );
    }
    for( int i = 0; i < 4; i++ )
    {
      pelFilterLuma( base0 + i * step, off, tc, sw, thrCut, filterP, filterQ, maxVal );
    }
  }
}

#else   // CHROMA

// xPelFilterChroma, one line
void pelFilterChroma( int16_t* p, int base, int off, int tc, bool sw, int maxVal, bool chromaCtbBoundary )
{
  const int m2 = rd( p, base, off, -2 );
  const int m3 = rd( p, base, off, -1 );
  const int m4 = rd( p, base, off, 0 );
  const int m5 = rd( p, base, off, 1 );
  if( sw )
  {
    const int m6 = rd( p, base, off, 2 );
    const int m7 = rd( p, base, off, 3 );
    if( chromaCtbBoundary )
    {
      wr( p, base, off, -1, clip3( m3 - tc, m3 + tc, ( 3 * m2 + 2 * m3 + m4 + m5 + m6 + 4 ) >> 3 ) );
      wr( p, base, off, 0, clip3( m4 - tc, m4 + tc, ( 2 * m2 + m3 + 2 * m4 + m5 + m6 + m7 + 4 ) >> 3 ) );
      wr( p, base, off, 1, clip3( m5 - tc, m5 + tc, ( m2 + m3 + m4 + 2 * m5 + m6 + 2 * m7 + 4 ) >> 3 ) );
      wr( p, base, off, 2, clip3( m6 - tc, m6 + tc, ( m3 + m4 + m5 + 2 * m6 + 3 * m7 + 4 ) >> 3 ) );
    }
    else
    {
      const int m0 = rd( p, base, off, -4 );
      const int m1 = rd( p, base, off, -3 );
      wr( p, base, off, -3, clip3( m1 - tc, m1 + tc, ( 3 * m0 + 2 * m1 + m2 + m3 + m4 + 4 ) >> 3 ) );
      wr( p, base, off, -2, clip3( m2 - tc, m2 + tc, ( 2 * m0 + m1 + 2 * m2 + m3 + m4 + m5 + 4 ) >> 3 ) );
      wr( p, base, off, -1, clip3( m3 - tc, m3 + tc, ( m0 + m1 + m2 + 2 * m3 + m4 + m5 + m6 + 4 ) >> 3 ) );
      wr( p, base, off, 0, clip3( m4 - tc, m4 + tc, ( m1 + m2 + m3 + 2 * m4 + m5 + m6 + m7 + 4 ) >> 3 ) );
      wr( p, base, off, 1, clip3( m5 - tc, m5 + tc, ( m2 + m3 + m4 + 2 * m5 + m6 + 2 * m7 + 4 ) >> 3 ) );
      wr( p, base, off, 2, clip3( m6 - tc, m6 + tc, ( m3 + m4 + m5 + 2 * m6 + 3 * m7 + 4 ) >> 3 ) );
    }
  }
  else
  {
    const int delta = clip3( -tc, tc, ( ( ( m4 - m3 ) * 4 ) + m2 - m5 + 4 ) >> 3 );
    wr( p, base, off, -1, clip3( 0, maxVal, m3 + delta ) );
    wr( p, base, off, 0, clip3( 0, maxVal, m4 - delta ) );
  }
}

[shader( "compute" )]
[numthreads( 8, 8, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  // the segments xDeblockCtuArea visits: along the edge as many lines as a
  // 4x4 luma block has, across it every 8 chroma samples (at the 4:2:0 rate)
  const int alongLen = pc.dir == 0 ? ( 4 >> pc.scaleY ) : ( 4 >> pc.scaleX );
  const int acrossStep = pc.dir == 0 ? ( ( 8 << pc.scaleX ) / 4 * ( 4 >> pc.scaleX ) ) : ( ( 8 << pc.scaleY ) / 4 * ( 4 >> pc.scaleY ) );
  const int x = pc.dir == 0 ? int( id.x ) * acrossStep : int( id.x ) * alongLen;
  const int y = pc.dir == 0 ? int( id.y ) * alongLen : int( id.y ) * acrossStep;
  if( x >= pc.width || y >= pc.height )
  {
    return;
  }

  const int lx = x << pc.scaleX;
  const int ly = y << pc.scaleY;
  const int e  = lfpOffset( lx, ly );
  const int bs = int( pc.lfp[e + 3] );
  const int bsC[2] = { ( bs >> 2 ) & 3, ( bs >> 4 ) & 3 };
  if( ( bsC[0] == 0 && bsC[1] == 0 ) || ctuParam( lx, ly, 0 ) != 0 )
  {
    return;
  }

  const bool largeBoundary     = ( ( int( pc.lfp[e + 5] ) >> 5 ) & 1 ) != 0;
  const bool chromaCtbBoundary = pc.dir == 1 && ( y & ( ( ( 1 << pc.ctuLog2 ) - 1 ) >> pc.scaleY ) ) == 0;
  const int  maxVal            = ( 1 << pc.bitDepth ) - 1;
  const int  off               = pc.dir == 0 ? 1 : pc.stride;
  const int  step              = pc.dir == 0 ? pc.stride : 1;
  const int  base0             = y * pc.stride + x;
  // the second line of the decisions: the next one when subsampled across, else the fourth
  const int  subSamplingShift  = pc.dir == 0 ? pc.scaleY : pc.scaleX;
  const int  base3             = base0 + ( subSamplingShift == 1 ? step : 3 * step );

  for( int c = 0; c < 2; c++ )
  {
    if( !( bsC[c] == 2 || ( largeBoundary && bsC[c] == 1 ) ) )
    {
      continue;
    }
    int16_t*  p       = c == 0 ? pc.pic0 : pc.pic1;
    const int qp      = int( int8_t( pc.lfp[e + 1 + c] ) );
    const int indexTc = clip3( 0, MAX_QP + 2, qp + 2 * ( bsC[c] - 1 ) + ctuParam( lx, ly, 4 + 2 * c ) * 2 );
    const int tc      = tcFromTable( indexTc );
    bool      sw      = false;
    if( largeBoundary )
    {
      const int indexB = clip3( 0, MAX_QP, qp + ctuParam( lx, ly, 3 + 2 * c ) * 2 );
      const int beta   = kBetaTable[indexB] * ( 1 << ( pc.bitDepth - 8 ) );
      const int dp0    = calcDP( p, base0, off, chromaCtbBoundary );
      const int dq0    = calcDQ( p, base0, off );
      const int dp3    = calcDP( p, base3, off, chromaCtbBoundary );
      const int dq3    = calcDQ( p, base3, off );
      const int d0     = dp0 + dq0;
      const int d3     = dp3 + dq3;
      if( d0 + d3 < beta )
      {
        sw = useStrongFiltering( p, base0, off, 2 * d0, beta, tc, false, false, 7, 7, chromaCtbBoundary )
             && useStrongFiltering( p, base3, off, 2 * d3, beta, tc, false, false, 7, 7, chromaCtbBoundary );
      }
    }
    for( int i = 0; i < alongLen; i++ )
    {
      pelFilterChroma( p, base0 + i * step, off, tc, sw, maxVal, chromaCtbBoundary );
    }
  }
}

#endif
