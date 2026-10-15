// The deblocking filter of a picture, in place, as vvdec's LoopFilter applies
// it (xEdgeFilterLuma, xEdgeFilterChroma): all vertical edges (dir 0), then,
// in another dispatch, all horizontal edges (dir 1). VVC keeps the edges of
// one direction far enough apart that each edge can be filtered on its own.
//
// CHROMA=0: a thread filters the 4 lines across one luma edge of a 4x4 block;
//           DIR=0 builds the variant for vertical edges, DIR=1 for horizontal.
// CHROMA=1: a thread filters the lines across one chroma edge segment, for Cb
//           and Cr, at the positions vvdec's xDeblockCtuArea visits.
// The edges arrive as a list of those that need filtering (made by
// edges.slang with the GPU front end, or by GpuRecon::listEdges with the CPU
// front end), each a GpuJobs.h Edge carrying the bytes of vvdec's
// LoopFilterParam that its filter needs (qp[3], bs, sideMaxFiltLength, flags).
// The ctus buffer holds 8 bytes per CTU: deblocking disabled, then the beta
// and tc offsets (div 2) of Y, Cb and Cr. With LADF (luma only), the QP of an
// edge's 4 lines is shifted by the offset of the interval that contains their
// luma level (LoopFilter::deriveLADFShift).

#include "common.slang"

#ifndef DIR
#define DIR 0
#endif

struct Params
{
  int16_t* pic0;        // luma, or Cb
  int16_t* pic1;        // Cr
  int8_t*  ctus;
  uint2*   edges;       // per edge: x | y << 16 (in the plane's samples), then the edge's parameters
  int      stride;
  int      width;       // plane size in samples
  int      height;
  int      ctuLog2;     // log2 of the CTU size in luma samples
  int      widthInCtus;
  int      bitDepth;
  int      dir;         // 0: vertical edges, 1: horizontal
  int      scaleX;      // chroma subsampling shifts
  int      scaleY;
  int      count;       // number of edges; negative: countAt holds it
  int      ladfNum;     // number of LADF intervals (0: no LADF)
  int      ladfOffset0, ladfOffset1, ladfOffset2, ladfOffset3, ladfOffset4;   // QP offset of each interval
  int      ladfBound1, ladfBound2, ladfBound3, ladfBound4;   // lower bound of each interval but the first
  uint*    countAt;     // when count < 0: the number of edges and the index of the first, from the GPU front end
  int16_t* tables;      // LoopFilter's tc table (66 entries), then its beta table (64)
};

int  edgeCount() { return pc.count >= 0 ? pc.count : int( pc.countAt[0] ); }
uint firstEdge() { return pc.count >= 0 ? 0u : pc.countAt[1]; }

[[vk::push_constant]] Params pc;

// LoopFilter's tc and beta tables, read from memory (pc.tables): as the shader's own constant arrays, read with a
// run-time index, Intel's compiler copied each into the thread's scratch memory at each use (21 KB of scratch).
int tcTable( int index )
{
  return int( *at( pc.tables, uint( index ) ) );
}

int betaTable( int index )
{
  return int( *at( pc.tables, 66u + uint( index ) ) );
}

#define MAX_QP 63

// Returns the QP offset of the LADF interval that contains the luma level, as deriveLADFShift does.
int ladfShift( int level )
{
  const int offsets[5] = { pc.ladfOffset0, pc.ladfOffset1, pc.ladfOffset2, pc.ladfOffset3, pc.ladfOffset4 };
  const int bounds[5]  = { 0, pc.ladfBound1, pc.ladfBound2, pc.ladfBound3, pc.ladfBound4 };
  int       shift      = offsets[0];
  for( int k = 1; k < pc.ladfNum; k++ )
  {
    if( level <= bounds[k] )
    {
      break;
    }
    shift = offsets[k];
  }
  return shift;
}

int ctuParam( int x, int y, int i )
{
  return int( pc.ctus[( ( y >> pc.ctuLog2 ) * pc.widthInCtus + ( x >> pc.ctuLog2 ) ) * 8 + i] );
}

int tcFromTable( int index )
{
  const int bd = pc.bitDepth;
  return bd < 10 ? ( ( tcTable( index ) + ( 1 << ( 9 - bd ) ) ) >> ( 10 - bd ) ) : ( tcTable( index ) << ( bd - 10 ) );
}

// Access to a line across the edge: sample k < 0 is on the P side (p0 is k = -1), k >= 0 on the Q side (q0 is k = 0).
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

// In registers, a line across the edge is an array s[16] with s[7 - j] = p_j and s[8 + j] = q_j (p7 .. q7).
// Samples that the filters do not reach are not loaded, and the code never uses their entries.

// Returns sample c of the 4 samples packed in 8 bytes.
int sample4( uint2 u, int c )
{
  return c == 0 ? int( u.x & 0xffff ) : ( c == 1 ? int( u.x >> 16 ) : ( c == 2 ? int( u.y & 0xffff ) : int( u.y >> 16 ) ) );
}

uint2 pack4( int a, int b, int c, int d )
{
  return uint2( uint( a ) | uint( b ) << 16, uint( c ) | uint( d ) << 16 );
}

// xUseStrongFiltering for one line
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

// The long filters (xFilteringPandQCore) on one line: they change p0 .. p(nP - 1) and q0 .. q(nQ - 1), where nP and
// nQ are 3, 5 or 7.
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

// The normal filters (xPelFilterLumaCorePel) on one line.
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

// The samples of the edge's 4 lines are loaded 8 bytes at a time into v[16]. With DIR 0 (a vertical edge at x),
// line i is row y + i, and v[4 i + g] holds its samples x - 8 + 4 g .. x - 5 + 4 g: g = 1 and 2 hold p3 .. q3, and
// g = 0 and 3 the long filters' p7 .. p4 and q4 .. q7. With DIR 1 (a horizontal edge at y), v[r] holds row
// y - 8 + r, and its sample i belongs to line i (columns x .. x + 3). Don't load the samples 2 bytes at a time: some 40
// loads a thread made VideoQ 4K's vertical edges take 8% longer. A vertical edge still costs twice as much as a
// horizontal one, because each of its 4 lines is in a different cache line, of which it uses only 16 bytes.
#if DIR == 0
#define SMP( i, k ) sample4( v[4 * ( i ) + ( ( k ) >> 2 )], ( k ) & 3 )
#else
#define SMP( i, k ) sample4( v[k], i )
#endif

// Loads the edge's samples into v (see above): p3 .. q3, and the long filters' p7 .. p4 or q4 .. q7 on a large side;
// all the loads are issued at once. Returns where the samples start: line 0's x - 8 (DIR 0), or row y - 8 (DIR 1).
int16_t* loadEdge( int x, int y, bool largeP, bool largeQ, out uint2 v[16] )
{
  [unroll]
  for( int a = 0; a < 16; a++ )
  {
    v[a] = uint2( 0, 0 );
  }
#if DIR == 0
  int16_t* at = samplesAt( pc.pic0, y * pc.stride + x - 8 );   // (negative for an edge at x 4 in row 0: see samplesAt)
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
  int16_t* at = samplesAt( pc.pic0, ( y - 8 ) * pc.stride + x );   // (negative for an edge at y 4: see samplesAt)
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
  return at;
}

// How an edge is filtered (xEdgeFilterLuma's decisions, on lines 0 and 3)
struct EdgeFilter
{
  int  mode;                // 0: not filtered, 1: the normal filters, 2: the long filters
  int  nP, nQ;              // the long filters' samples on each side
  bool sw;                  // the normal filters: the strong filter,
  bool filterP, filterQ;    // or the weak one, which also changes the second sample on these sides
  int  tc, thrCut;          // tc, and the weak filter's threshold
  int  changedP, changedQ;  // the samples the filters may change on each side
};

// Decides how the edge is filtered, from its QP and boundary strength (edge.y), its CTU's beta and tc offsets, and
// its lines 0 and 3 (l0, l3; with LADF, the luma level of p0 and q0 there shifts the QP).
EdgeFilter decideEdge( uint2 edge, int x, int y, int bs, int l0[16], int l3[16], bool largeP, bool largeQ, int maxFilterLengthP,
                       int maxFilterLengthQ )
{
  int qp = int( int8_t( edge.y & 255 ) );
  if( pc.ladfNum > 0 )
  {
    qp += ladfShift( ( l0[8] + l3[8] + l0[7] + l3[7] ) >> 2 );
  }
  const int betaOffsetDiv2 = ctuParam( x, y, 1 );
  const int tcOffsetDiv2   = ctuParam( x, y, 2 );
  const int indexTc        = clip3( 0, MAX_QP + 2, qp + 2 * ( bs - 1 ) + tcOffsetDiv2 * 2 );
  const int indexB         = clip3( 0, MAX_QP, qp + betaOffsetDiv2 * 2 );
  const int tc             = tcFromTable( indexTc );
  const int beta           = betaTable( indexB ) << ( pc.bitDepth - 8 );
  const int sideThreshold  = ( beta + ( beta >> 1 ) ) >> 3;
  const int dp0 = abs( l0[5] - 2 * l0[6] + l0[7] );    // |p2 - 2 p1 + p0|
  const int dq0 = abs( l0[8] - 2 * l0[9] + l0[10] );   // |q0 - 2 q1 + q2|
  const int dp3 = abs( l3[5] - 2 * l3[6] + l3[7] );
  const int dq3 = abs( l3[8] - 2 * l3[9] + l3[10] );
  const int d0  = dp0 + dq0;
  const int d3  = dp3 + dq3;

  EdgeFilter f;
  f.mode    = 0;
  f.nP      = 3;
  f.nQ      = 3;
  f.sw      = false;
  f.filterP = false;
  f.filterQ = false;
  f.tc      = tc;
  f.thrCut  = tc * 10;
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
      f.mode = 2;
      f.nP   = largeP ? maxFilterLengthP : 3;
      f.nQ   = largeQ ? maxFilterLengthQ : 3;
    }
  }
  if( f.mode == 0 && d0 + d3 < beta )
  {
    f.mode = 1;
    if( maxFilterLengthP > 1 && maxFilterLengthQ > 1 )
    {
      f.filterP = dp0 + dp3 < sideThreshold;
      f.filterQ = dq0 + dq3 < sideThreshold;
    }
    if( maxFilterLengthP > 2 && maxFilterLengthQ > 2 )
    {
      f.sw = strongLine( l0, 2 * d0, beta, tc, false, false, 7, 7 ) && strongLine( l3, 2 * d3, beta, tc, false, false, 7, 7 );
    }
  }
  f.changedP = f.mode == 2 ? f.nP : ( f.sw ? 3 : ( f.filterP ? 2 : 1 ) );
  f.changedQ = f.mode == 2 ? f.nQ : ( f.sw ? 3 : ( f.filterQ ? 2 : 1 ) );
  return f;
}

// Filters the edge's 4 lines as f says and packs the results back into v.
void filterEdge( inout uint2 v[16], EdgeFilter f, int maxVal )
{
  [unroll]
  for( int i = 0; i < 4; i++ )
  {
    int s[16];
    [unroll]
    for( int k = 0; k < 16; k++ )
    {
      s[k] = SMP( i, k );
    }
    if( f.mode == 2 )
    {
      longLine( s, f.nP, f.nQ, f.tc );
    }
    else
    {
      normalLine( s, f.tc, f.sw, f.thrCut, f.filterP, f.filterQ, maxVal );
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
}

// Stores the samples the filters may change, plus neighbouring samples only where no other edge's filter changes
// them (VVC keeps the edges of one direction far enough apart for that). at is where loadEdge's samples start.
void storeEdge( int16_t* at, uint2 v[16], EdgeFilter f, int maxFilterLengthP, int maxFilterLengthQ )
{
#if DIR == 0
  // Vertical edge: store p3 .. p0 as 8 bytes when the P side's filters may reach 3 samples (the next edge is then at
  // least 8 samples away), else p1 and p0 as 4 bytes. Store the long filters' p4 .. p7 as far as they reach (p7 is
  // never changed, and no other edge reaches it). The Q side works the same way.
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
    if( f.changedP == 7 )
    {
      *(uint2*)line = v[4 * i];
    }
    else if( f.changedP == 5 )
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
    if( f.changedQ == 7 )
    {
      *(uint2*)( line + 12 ) = v[4 * i + 3];
    }
    else if( f.changedQ == 5 )
    {
      line[12] = int16_t( v[4 * i + 3].x & 0xffff );   // q4
    }
  }
#else
  // Horizontal edge: store each row the filters may change, 8 bytes each (the edge's 4 columns are this thread's).
  [unroll]
  for( int j = 0; j < 7; j++ )
  {
    if( j < f.changedP )
    {
      *(uint2*)( at + ( 7 - j ) * pc.stride ) = v[7 - j];
    }
    if( j < f.changedQ )
    {
      *(uint2*)( at + ( 8 + j ) * pc.stride ) = v[8 + j];
    }
  }
#endif
}

// One thread per luma edge of 4 lines (the edges' list from edges.slang): its samples, the filter decisions, the
// filtering and the store.
[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  if( int( id.x ) >= edgeCount() )
  {
    return;
  }
  const uint2 edge = pc.edges[firstEdge() + id.x];   // .y: qp | bs << 8 | sideMaxFiltLength << 16
  const int   x    = int( edge.x & 0xffff );
  const int   y    = int( edge.x >> 16 );
  const int   bs   = int( edge.y >> 8 ) & 3;
  if( bs == 0 || ctuParam( x, y, 0 ) != 0 )
  {
    return;
  }

  const int  sideMax          = int( ( edge.y >> 16 ) & 255 );
  const int  maxFilterLengthP = ( sideMax >> 4 ) & 7;
  const int  maxFilterLengthQ = sideMax & 7;
  bool       largeP           = maxFilterLengthP > 3;
  const bool largeQ           = maxFilterLengthQ > 3;
#if DIR == 1
  if( ( y & ( ( 1 << pc.ctuLog2 ) - 1 ) ) == 0 )
  {
    largeP = false;
  }
#endif

  uint2    v[16];
  int16_t* at = loadEdge( x, y, largeP, largeQ, v );
  // The filter decisions use lines 0 and 3.
  int l0[16], l3[16];
  [unroll]
  for( int k = 0; k < 16; k++ )
  {
    l0[k] = SMP( 0, k );
    l3[k] = SMP( 3, k );
  }
  const EdgeFilter f = decideEdge( edge, x, y, bs, l0, l3, largeP, largeQ, maxFilterLengthP, maxFilterLengthQ );
  if( f.mode == 0 )
  {
    return;
  }
  filterEdge( v, f, ( 1 << pc.bitDepth ) - 1 );
  storeEdge( at, v, f, maxFilterLengthP, maxFilterLengthQ );
}

#else   // CHROMA

// xPelFilterChroma on one line
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
  // Each edge is one segment as xDeblockCtuArea visits them: along the edge, the
  // chroma lines of one 4x4 luma block; across the edges, one segment every 8
  // chroma samples (16 luma samples in 4:2:0).
  if( int( id.x ) >= edgeCount() )
  {
    return;
  }
  const uint2 edge = pc.edges[firstEdge() + id.x];   // .y: Cb's qp | Cr's qp << 8 | bs << 16 | flags << 24
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
  const int  alongLen          = pc.dir == 0 ? ( 4 >> pc.scaleY ) : ( 4 >> pc.scaleX );   // lines along the edge
  // The decisions use line 0 and a second line: line 1 when the edge's chroma is subsampled along it, else line 3.
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
      const int beta   = betaTable( indexB ) * ( 1 << ( pc.bitDepth - 8 ) );
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
