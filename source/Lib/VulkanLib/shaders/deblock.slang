// The deblocking filter of a picture, in place, as vvdec's LoopFilter does it
// (xEdgeFilterLuma, xEdgeFilterChroma): all vertical edges (dir 0), then, in
// another dispatch, all horizontal ones (dir 1). VVC keeps the edges of one
// direction apart enough that each can be filtered on its own.
//
// CHROMA=0: a thread filters 4 lines across one luma edge of a 4x4 block,
//           the vertical edges with DIR=0, the horizontal ones with DIR=1.
// CHROMA=1: a thread filters the lines across one chroma edge segment, for Cb
//           and Cr, at the positions vvdec's xDeblockCtuArea visits.
// The edges come as a list, those the CPU found to filter, each with the
// bytes of vvdec's LoopFilterParam (qp[3], bs, sideMaxFiltLength, flags) its
// filter needs (GpuRecon::listEdges, GpuJobs.h Edge).
// Per CTU, 8 bytes: deblocking disabled, then the beta and tc offsets (div 2)
// of Y, Cb, Cr.

#include "common.slang"

#ifndef DIR
#define DIR 0
#endif

struct Params
{
  int16_t* pic0;        // luma, or Cb
  int16_t* pic1;        // Cr
  int8_t*  ctus;
  uint2*   edges;       // x | y << 16 of each edge (samples of the plane), and its parameters
  int      stride;
  int      width;       // of the plane
  int      height;
  int      ctuLog2;     // luma
  int      widthInCtus;
  int      bitDepth;
  int      dir;         // 0: vertical edges, 1: horizontal
  int      scaleX;      // chroma subsampling
  int      scaleY;
  int      count;       // of edges
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
static const int kDbCoeffs5[7] = { 58, 45, 32, 19, 6, 0, 0 };
static const int kDbCoeffs3[7] = { 53, 32, 11, 0, 0, 0, 0 };
static const int kTc7[7]       = { 6, 5, 4, 3, 2, 1, 1 };
static const int kTc3[7]       = { 6, 4, 2, 0, 0, 0, 0 };

// A line across the edge, in registers: s[7 - j] = p_j, s[8 + j] = q_j (p7 .. q7; the samples a filter does not
// reach are left out of the loads, and of these arrays' uses).

// sample c of 4 in 8 bytes
int sample4( uint2 u, int c )
{
  return c == 0 ? int( u.x & 0xffff ) : ( c == 1 ? int( u.x >> 16 ) : ( c == 2 ? int( u.y & 0xffff ) : int( u.y >> 16 ) ) );
}

uint2 pack4( int a, int b, int c, int d )
{
  return uint2( uint( a ) | uint( b ) << 16, uint( c ) | uint( d ) << 16 );
}

// xUseStrongFiltering on a line
bool strongLine( int s[16], int d, int beta, int tc, bool largeP, bool largeQ, int maxP, int maxQ )
{
  const int p0 = s[7];
  const int q0 = s[8];
  if( !( d < ( beta >> 2 ) && abs( p0 - q0 ) < ( ( tc * 5 + 1 ) >> 1 ) ) )
  {
    return false;
  }
  const int p3      = s[4];
  const int q3      = s[11];
  int       sp3     = abs( p3 - p0 );
  int       sq3     = abs( q3 - q0 );
  const int dStrong = sp3 + sq3;
  if( largeP || largeQ )
  {
    if( largeP )
    {
      const int pMax = maxP == 7 ? s[0] : s[2];   // p7 or p5
      if( maxP == 7 )
      {
        sp3 = sp3 + abs( s[3] - s[2] - s[1] + pMax );   // p4 - p5 - p6 + p7
      }
      sp3 = ( sp3 + abs( p3 - pMax ) + 1 ) >> 1;
    }
    if( largeQ )
    {
      const int qMax = maxQ == 7 ? s[15] : s[13];   // q7 or q5
      if( maxQ == 7 )
      {
        sq3 = sq3 + abs( s[12] - s[13] - s[14] + qMax );   // q4 - q5 - q6 + q7
      }
      sq3 = ( sq3 + abs( qMax - q3 ) + 1 ) >> 1;
    }
    return ( sp3 + sq3 ) < ( ( beta * 3 ) >> 5 ) && d < ( beta >> 4 ) && abs( p0 - q0 ) < ( ( tc * 5 + 1 ) >> 1 );
  }
  return dStrong < ( beta >> 3 );
}

int dbCoeff( int n, int pos )
{
  return n == 7 ? kDbCoeffs7[pos] : ( n == 5 ? kDbCoeffs5[pos] : kDbCoeffs3[pos] );
}

// the long filters (xFilteringPandQCore) on a line: p0 .. p(nP - 1), q0 .. q(nQ - 1), nP and nQ 3, 5 or 7
void longLine( inout int s[16], int nP, int nQ, int tc )
{
  int ps[8], qs[8];
  [unroll]
  for( int k = 0; k < 8; k++ )
  {
    ps[k] = s[7 - k];
    qs[k] = s[8 + k];
  }
  const int refP = nP == 7 ? ( ps[6] + ps[7] + 1 ) >> 1 : ( nP == 5 ? ( ps[4] + ps[5] + 1 ) >> 1 : ( ps[2] + ps[3] + 1 ) >> 1 );
  const int refQ = nQ == 7 ? ( qs[6] + qs[7] + 1 ) >> 1 : ( nQ == 5 ? ( qs[4] + qs[5] + 1 ) >> 1 : ( qs[2] + qs[3] + 1 ) >> 1 );
  int       refMiddle;
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
      [unroll]
      for( int k = 0; k < 7; k++ )
      {
        ls[k] = nP == 7 ? ps[k] : qs[k];
      }
      [unroll]
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
  [unroll]
  for( int pos = 0; pos < 7; pos++ )
  {
    if( pos < nP )
    {
      const int src    = ps[pos];
      const int cvalue = ( tc * ( nP == 3 ? kTc3[pos] : kTc7[pos] ) ) >> 1;
      const int c      = dbCoeff( nP, pos );
      s[7 - pos]       = clip3( src - cvalue, src + cvalue, ( refMiddle * c + refP * ( 64 - c ) + 32 ) >> 6 );
    }
    if( pos < nQ )
    {
      const int src    = qs[pos];
      const int cvalue = ( tc * ( nQ == 3 ? kTc3[pos] : kTc7[pos] ) ) >> 1;
      const int c      = dbCoeff( nQ, pos );
      s[8 + pos]       = clip3( src - cvalue, src + cvalue, ( refMiddle * c + refQ * ( 64 - c ) + 32 ) >> 6 );
    }
  }
}

// the normal filters (xPelFilterLumaCorePel) on a line
void normalLine( inout int s[16], int tc, bool sw, int thrCut, bool filterSecondP, bool filterSecondQ, int maxVal )
{
  const int m1 = s[5];
  const int m2 = s[6];
  const int m3 = s[7];
  const int m4 = s[8];
  const int m5 = s[9];
  const int m6 = s[10];
  if( sw )
  {
    const int m0 = s[4];
    const int m7 = s[11];
    s[5]  = clip3( m1 - tc, m1 + tc, ( 2 * m0 + 3 * m1 + m2 + m3 + m4 + 4 ) >> 3 );
    s[6]  = clip3( m2 - 2 * tc, m2 + 2 * tc, ( m1 + m2 + m3 + m4 + 2 ) >> 2 );
    s[7]  = clip3( m3 - 3 * tc, m3 + 3 * tc, ( m1 + 2 * m2 + 2 * m3 + 2 * m4 + m5 + 4 ) >> 3 );
    s[8]  = clip3( m4 - 3 * tc, m4 + 3 * tc, ( m2 + 2 * m3 + 2 * m4 + 2 * m5 + m6 + 4 ) >> 3 );
    s[9]  = clip3( m5 - 2 * tc, m5 + 2 * tc, ( m3 + m4 + m5 + m6 + 2 ) >> 2 );
    s[10] = clip3( m6 - tc, m6 + tc, ( m3 + m4 + m5 + 3 * m6 + 2 * m7 + 4 ) >> 3 );
  }
  else
  {
    int delta = ( 9 * ( m4 - m3 ) - 3 * ( m5 - m2 ) + 8 ) >> 4;
    if( abs( delta ) < thrCut )
    {
      delta         = clip3( -tc, tc, delta );
      const int tc2 = tc >> 1;
      s[7]          = clip3( 0, maxVal, m3 + delta );
      if( filterSecondP )
      {
        s[6] = clip3( 0, maxVal, m2 + clip3( -tc2, tc2, ( ( ( m1 + m3 + 1 ) >> 1 ) - m2 + delta ) >> 1 ) );
      }
      s[8] = clip3( 0, maxVal, m4 - delta );
      if( filterSecondQ )
      {
        s[9] = clip3( 0, maxVal, m5 + clip3( -tc2, tc2, ( ( ( m6 + m4 + 1 ) >> 1 ) - m5 - delta ) >> 1 ) );
      }
    }
  }
}

// The samples of the edge's 4 lines, 8 bytes a load: with DIR 0 (a vertical edge at x), line i is the row y + i,
// v[4 i + g] its samples x - 8 + 4 g .. x - 5 + 4 g (g 1 and 2: p3 .. q3; 0 and 3: the long filters' p7 .. p4 and
// q4 .. q7); with DIR 1 (a horizontal edge at y), v[r] is the row y - 8 + r, sample i of it line i's (the columns x
// .. x + 3). (A thread made some 40 loads of 2 bytes before: VideoQ 4K's vertical edges took 8% longer. What remains
// of their cost, twice the horizontal edges' an edge, is the cache lines: each line of a vertical edge's 4 is in
// another, of which it uses 16 bytes.)
#if DIR == 0
#define SMP( i, k ) sample4( v[4 * ( i ) + ( ( k ) >> 2 )], ( k ) & 3 )
#else
#define SMP( i, k ) sample4( v[k], i )
#endif

[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  if( int( id.x ) >= pc.count )
  {
    return;
  }
  const uint2 edge = pc.edges[id.x];   // qp | bs << 8 | sideMaxFiltLength << 16
  const int   x    = int( edge.x & 0xffff );
  const int   y    = int( edge.x >> 16 );
  const int   bs   = int( edge.y >> 8 ) & 3;
  if( bs == 0 || ctuParam( x, y, 0 ) != 0 )
  {
    return;
  }

  const int qp               = int( int8_t( edge.y & 255 ) );
  const int sideMax          = int( ( edge.y >> 16 ) & 255 );
  const int maxFilterLengthP = ( sideMax >> 4 ) & 7;
  const int maxFilterLengthQ = sideMax & 7;
  bool      largeP           = maxFilterLengthP > 3;
  const bool largeQ          = maxFilterLengthQ > 3;
#if DIR == 1
  if( ( y & ( ( 1 << pc.ctuLog2 ) - 1 ) ) == 0 )
  {
    largeP = false;
  }
#endif

  const int betaOffsetDiv2 = ctuParam( x, y, 1 );
  const int tcOffsetDiv2   = ctuParam( x, y, 2 );
  const int indexTc        = clip3( 0, MAX_QP + 2, qp + 2 * ( bs - 1 ) + tcOffsetDiv2 * 2 );
  const int indexB         = clip3( 0, MAX_QP, qp + betaOffsetDiv2 * 2 );
  const int tc             = tcFromTable( indexTc );
  const int beta           = kBetaTable[indexB] << ( pc.bitDepth - 8 );
  const int sideThreshold  = ( beta + ( beta >> 1 ) ) >> 3;
  const int thrCut         = tc * 10;
  const int maxVal         = ( 1 << pc.bitDepth ) - 1;

  // all the loads at once
  uint2 v[16];
  [unroll]
  for( int a = 0; a < 16; a++ )
  {
    v[a] = uint2( 0, 0 );
  }
#if DIR == 0
  int16_t* at = pc.pic0 + y * pc.stride + x - 8;
  [unroll]
  for( int i = 0; i < 4; i++ )
  {
    v[4 * i + 1] = *(uint2*)( at + i * pc.stride + 4 );
    v[4 * i + 2] = *(uint2*)( at + i * pc.stride + 8 );
    if( largeP )
    {
      v[4 * i] = *(uint2*)( at + i * pc.stride );
    }
    if( largeQ )
    {
      v[4 * i + 3] = *(uint2*)( at + i * pc.stride + 12 );
    }
  }
#else
  int16_t* at = pc.pic0 + ( y - 8 ) * pc.stride + x;
  [unroll]
  for( int r = 4; r < 12; r++ )
  {
    v[r] = *(uint2*)( at + r * pc.stride );
  }
  if( largeP )
  {
    [unroll]
    for( int r = 0; r < 4; r++ )
    {
      v[r] = *(uint2*)( at + r * pc.stride );
    }
  }
  if( largeQ )
  {
    [unroll]
    for( int r = 12; r < 16; r++ )
    {
      v[r] = *(uint2*)( at + r * pc.stride );
    }
  }
#endif

  // the decisions, from lines 0 and 3
  int l0[16], l3[16];
  [unroll]
  for( int k = 0; k < 16; k++ )
  {
    l0[k] = SMP( 0, k );
    l3[k] = SMP( 3, k );
  }
  const int dp0 = abs( l0[5] - 2 * l0[6] + l0[7] );    // |p2 - 2 p1 + p0|
  const int dq0 = abs( l0[8] - 2 * l0[9] + l0[10] );   // |q0 - 2 q1 + q2|
  const int dp3 = abs( l3[5] - 2 * l3[6] + l3[7] );
  const int dq3 = abs( l3[8] - 2 * l3[9] + l3[10] );
  const int d0  = dp0 + dq0;
  const int d3  = dp3 + dq3;

  // what the filters change: 0 nothing, 1 the normal filters, 2 the long ones
  int  mode    = 0;
  int  nP      = 3, nQ = 3;
  bool sw      = false;
  bool filterP = false;
  bool filterQ = false;
  if( largeP || largeQ )
  {
    const int dp0L = largeP ? ( ( dp0 + abs( l0[2] - 2 * l0[3] + l0[4] ) + 1 ) >> 1 ) : dp0;   // |p5 - 2 p4 + p3|
    const int dq0L = largeQ ? ( ( dq0 + abs( l0[11] - 2 * l0[12] + l0[13] ) + 1 ) >> 1 ) : dq0;   // |q3 - 2 q4 + q5|
    const int dp3L = largeP ? ( ( dp3 + abs( l3[2] - 2 * l3[3] + l3[4] ) + 1 ) >> 1 ) : dp3;
    const int dq3L = largeQ ? ( ( dq3 + abs( l3[11] - 2 * l3[12] + l3[13] ) + 1 ) >> 1 ) : dq3;
    const int d0L  = dp0L + dq0L;
    const int d3L  = dp3L + dq3L;
    if( d0L + d3L < beta && strongLine( l0, 2 * d0L, beta, tc, largeP, largeQ, maxFilterLengthP, maxFilterLengthQ )
        && strongLine( l3, 2 * d3L, beta, tc, largeP, largeQ, maxFilterLengthP, maxFilterLengthQ ) )
    {
      mode = 2;
      nP   = largeP ? maxFilterLengthP : 3;
      nQ   = largeQ ? maxFilterLengthQ : 3;
    }
  }
  if( mode == 0 && d0 + d3 < beta )
  {
    mode = 1;
    if( maxFilterLengthP > 1 && maxFilterLengthQ > 1 )
    {
      filterP = dp0 + dp3 < sideThreshold;
      filterQ = dq0 + dq3 < sideThreshold;
    }
    if( maxFilterLengthP > 2 && maxFilterLengthQ > 2 )
    {
      sw = strongLine( l0, 2 * d0, beta, tc, false, false, 7, 7 ) && strongLine( l3, 2 * d3, beta, tc, false, false, 7, 7 );
    }
  }
  if( mode == 0 )
  {
    return;
  }

  // the lines filtered, back into v
  [unroll]
  for( int i = 0; i < 4; i++ )
  {
    int s[16];
    [unroll]
    for( int k = 0; k < 16; k++ )
    {
      s[k] = SMP( i, k );
    }
    if( mode == 2 )
    {
      longLine( s, nP, nQ, tc );
    }
    else
    {
      normalLine( s, tc, sw, thrCut, filterP, filterQ, maxVal );
    }
#if DIR == 0
    [unroll]
    for( int g = 0; g < 4; g++ )
    {
      v[4 * i + g] = pack4( s[4 * g], s[4 * g + 1], s[4 * g + 2], s[4 * g + 3] );
    }
#else
    [unroll]
    for( int k = 0; k < 16; k++ )
    {
      const uint shift = ( i & 1 ) * 16;
      const uint keep  = 0xffffu << ( 16 - shift );
      if( i < 2 )
      {
        v[k].x = ( v[k].x & keep ) | uint( s[k] ) << shift;
      }
      else
      {
        v[k].y = ( v[k].y & keep ) | uint( s[k] ) << shift;
      }
    }
#endif
  }

  // the stores: the samples a filter may change, and around them only those no other edge's filter changes
  // (VVC keeps the edges of a direction apart enough for that)
  const int changedP = mode == 2 ? nP : ( sw ? 3 : ( filterP ? 2 : 1 ) );
  const int changedQ = mode == 2 ? nQ : ( sw ? 3 : ( filterQ ? 2 : 1 ) );
#if DIR == 0
  // a vertical edge: p3 .. p0 in 8 bytes when the P side's filters may reach 3 samples (the next edge is 8 samples
  // or more away then), else p1, p0 in 4 bytes; the long filters' p4 .. p7 as far as they reach (p7 is never
  // changed, nor reached by another edge); the same on the Q side
  [unroll]
  for( int i = 0; i < 4; i++ )
  {
    int16_t* line = at + i * pc.stride;
    if( maxFilterLengthP > 2 )
    {
      *(uint2*)( line + 4 ) = v[4 * i + 1];
    }
    else
    {
      *(uint*)( line + 6 ) = v[4 * i + 1].y;
    }
    if( changedP == 7 )
    {
      *(uint2*)line = v[4 * i];
    }
    else if( changedP == 5 )
    {
      line[3] = int16_t( v[4 * i].y >> 16 );   // p4
    }
    if( maxFilterLengthQ > 2 )
    {
      *(uint2*)( line + 8 ) = v[4 * i + 2];
    }
    else
    {
      *(uint*)( line + 8 ) = v[4 * i + 2].x;
    }
    if( changedQ == 7 )
    {
      *(uint2*)( line + 12 ) = v[4 * i + 3];
    }
    else if( changedQ == 5 )
    {
      line[12] = int16_t( v[4 * i + 3].x & 0xffff );   // q4
    }
  }
#else
  // a horizontal edge: the rows the filters may change, whole (the edge's 4 columns are this thread's)
  [unroll]
  for( int j = 0; j < 7; j++ )
  {
    if( j < changedP )
    {
      *(uint2*)( at + ( 7 - j ) * pc.stride ) = v[7 - j];
    }
    if( j < changedQ )
    {
      *(uint2*)( at + ( 8 + j ) * pc.stride ) = v[8 + j];
    }
  }
#endif
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
[numthreads( 64, 1, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  // the segments xDeblockCtuArea visits: along the edge as many lines as a
  // 4x4 luma block has, across it every 8 chroma samples (at the 4:2:0 rate)
  if( int( id.x ) >= pc.count )
  {
    return;
  }
  const uint2 edge = pc.edges[id.x];   // Cb's qp | Cr's qp << 8 | bs << 16 | flags << 24
  const int   x    = int( edge.x & 0xffff );
  const int   y    = int( edge.x >> 16 );

  const int lx = x << pc.scaleX;
  const int ly = y << pc.scaleY;
  const int bs = int( edge.y >> 16 ) & 255;
  const int bsC[2] = { ( bs >> 2 ) & 3, ( bs >> 4 ) & 3 };
  if( ( bsC[0] == 0 && bsC[1] == 0 ) || ctuParam( lx, ly, 0 ) != 0 )
  {
    return;
  }

  const bool largeBoundary     = ( ( edge.y >> 29 ) & 1 ) != 0;   // flags bit 5
  const bool chromaCtbBoundary = pc.dir == 1 && ( y & ( ( ( 1 << pc.ctuLog2 ) - 1 ) >> pc.scaleY ) ) == 0;
  const int  maxVal            = ( 1 << pc.bitDepth ) - 1;
  const int  off               = pc.dir == 0 ? 1 : pc.stride;
  const int  step              = pc.dir == 0 ? pc.stride : 1;
  const int  base0             = y * pc.stride + x;
  const int  alongLen          = pc.dir == 0 ? ( 4 >> pc.scaleY ) : ( 4 >> pc.scaleX );   // the lines along the edge
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
    const int qp      = int( int8_t( ( edge.y >> ( 8 * c ) ) & 255 ) );
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
