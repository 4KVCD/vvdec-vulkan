// Inter prediction and the reconstruction of inter blocks, as vvdec's
// InterPrediction and InterpolationFilter do it: uni- and bi-prediction with
// the 8-tap luma and 4-tap chroma filters (and AMVR's half-sample filter),
// and decoder-side motion vector refinement (DMVR). A workgroup takes one
// tile of up to 16x16 luma samples of a CU (DMVR's subblock when it refines)
// with its chroma, predicts it, adds the residual the picture holds and
// writes it; its 64 threads share out the tile's samples.
//
// A job, 12 uints (VulkanLib/GpuJobs.h, InterJob):
//   0: x | y << 16 of the tile (luma)      1: width | height << 8 | flags << 16
//   2, 3: L0 motion vector (1/16 luma)     4, 5: L1 motion vector
//   6: x | y << 16 of the CU               7: L0 reference | L1 reference << 8 (slots of the table)
//   8: where DMVR's refinement goes (vvdec's DMVR MV cache)
// flags: 1 L0, 2 L1, 4 DMVR, 8 AMVR half-sample filter, 16/32/64 residual in Y/Cb/Cr

#include "common.slang"

struct RefPlanes
{
  int16_t* y;
  int16_t* cb;
  int16_t* cr;
  int16_t* unused;
};

struct Params
{
  uint*      jobs;
  RefPlanes* refs;
  int16_t*   dst0;
  int16_t*   dst1;
  int16_t*   dst2;
  int*       dmvr;        // vvdec's Mv: hor, ver
  int        stride0;
  int        stride12;
  int        count;
  int        picWidth;    // luma
  int        picHeight;
  int        ctuSize;
  int        bitDepth;
};

[[vk::push_constant]] Params pc;

#define FLAG_L0 1
#define FLAG_L1 2
#define FLAG_DMVR 4
#define FLAG_HPEL 8
#define FLAG_RES_Y 16

#define IF_INTERNAL_OFFS 8192
#define THREADS 64
#define PER_THREAD 4   // a 16x16 tile's luma samples over the threads

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
static const int kBilinear[32] = { 16, 0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

// shared by the workgroup: the horizontal pass of a 2-D filter, per list;
// DMVR's bilinear predictions, its SADs and its result
#define TMP_SIZE ( ( 16 + 7 ) * 16 )
groupshared int  s_tmp[2 * TMP_SIZE];
groupshared int  s_bil0[20 * 20];
groupshared int  s_bil1[20 * 20];
groupshared int  s_sad[25];
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

// a reference sample, its coordinates clamped into the picture: vvdec's border extension
int refAt( int16_t* p, int comp, int x, int y )
{
  const int w = comp == 0 ? pc.picWidth : pc.picWidth >> 1;
  const int h = comp == 0 ? pc.picHeight : pc.picHeight >> 1;
  return int( p[clamp( y, 0, h - 1 ) * ( comp == 0 ? pc.stride0 : pc.stride12 ) + clamp( x, 0, w - 1 )] );
}

// with a window (DMVR's padded prefetch), the reads clamped into it first
int refSample( int16_t* p, int comp, int x, int y, bool windowed, int4 window )
{
  if( windowed )
  {
    x = clamp( x, window.x, window.x + window.z - 1 );
    y = clamp( y, window.y, window.y + window.w - 1 );
  }
  return refAt( p, comp, x, y );
}

// clipMvInPic: the block at (x, y) stays near the picture
int2 clipMv( int2 mv, int x, int y )
{
  const int horMax = ( pc.picWidth + 8 - x - 1 ) * 16;
  const int horMin = ( -pc.ctuSize - 8 - x + 1 ) * 16;
  const int verMax = ( pc.picHeight + 8 - y - 1 ) * 16;
  const int verMin = ( -pc.ctuSize - 8 - y + 1 ) * 16;
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

// xPredInterBlk of a w x h block of a component, the thread's samples
// s = tid + THREADS * k into out[k]: the sample (i, j) filters the reference
// around (baseX + i, baseY + j) with the fractions of mv; bi: the 14-bit
// intermediate for averaging, else the final, clipped value. The workgroup
// calls it together (it may wait for the others).
void predictBlock( int tid, int list, int slot, int comp, int bx, int by, int w, int h, int2 mv, bool bi, bool hpel, bool windowed, int4 window,
                   int2 base, inout int out[PER_THREAD] )
{
  int16_t*  ref      = refPlane( slot, comp );
  const int shift    = comp == 0 ? 4 : 5;
  const int xFrac    = mv.x & ( ( 1 << shift ) - 1 );
  const int yFrac    = mv.y & ( ( 1 << shift ) - 1 );
  const int baseX    = windowed ? base.x : bx + ( mv.x >> shift );
  const int baseY    = windowed ? base.y : by + ( mv.y >> shift );
  const int n        = comp == 0 ? 8 : 4;
  const int half     = n / 2 - 1;
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int log2W    = log2Int( w );

  if( xFrac != 0 && yFrac != 0 )
  {
    // the horizontal pass over the rows the vertical one reads
    const int shift1  = 6 - headRoom;
    const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
    const int at      = list * TMP_SIZE;
    for( int s = tid; s < ( h + n - 1 ) * w; s += THREADS )
    {
      const int r   = s >> log2W;
      const int i   = s & ( w - 1 );
      int       sum = 0;
      for( int k = 0; k < n; k++ )
      {
        sum += coeff( comp, xFrac, hpel, k ) * refSample( ref, comp, baseX + i + k - half, baseY + r - half, windowed, window );
      }
      s_tmp[at + s] = ( sum + offset1 ) >> shift1;
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
        for( int t = 0; t < n; t++ )
        {
          sum += coeff( comp, yFrac, hpel, t ) * s_tmp[at + ( j + t ) * w + i];
        }
        out[k] = bi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
      }
    }
    GroupMemoryBarrierWithGroupSync();   // before the next pass writes the shared memory
    return;
  }

  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int s = tid + THREADS * k;
    if( s >= w * h )
    {
      break;
    }
    const int j = s >> log2W;
    const int i = s & ( w - 1 );
    if( xFrac == 0 && yFrac == 0 )
    {
      const int v = refSample( ref, comp, baseX + i, baseY + j, windowed, window );
      out[k]      = bi ? v * ( 1 << headRoom ) - IF_INTERNAL_OFFS : v;
    }
    else
    {
      const bool vertical = yFrac != 0;
      const int  frac     = vertical ? yFrac : xFrac;
      const int  shiftV   = bi ? 6 - headRoom : 6;
      const int  offset   = bi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
      int        sum      = 0;
      for( int t = 0; t < n; t++ )
      {
        sum += coeff( comp, frac, hpel, t )
               * ( vertical ? refSample( ref, comp, baseX + i, baseY + j + t - half, windowed, window )
                            : refSample( ref, comp, baseX + i + t - half, baseY + j, windowed, window ) );
      }
      const int v = ( sum + offset ) >> shiftV;
      out[k]      = bi ? v : clip3( 0, maxVal, v );
    }
  }
}

// --------------------------------------------------------------------------------------------------------------------
// DMVR (xProcessDMVR): the bilinear predictions around the merge motion, the SADs of the 5x5 integer offsets
// (every other row), the error surface's sub-sample step
// --------------------------------------------------------------------------------------------------------------------

// xinitMC's bilinear prediction of the CU from mvStart (the clipped merge motion minus 2 samples) at the sample
// (u, v) relative to the CU
int bilinearAt( int slot, int cuX, int cuY, int2 mvStart, int u, int v )
{
  int16_t*  ref   = refPlane( slot, 0 );
  const int xFrac = mvStart.x & 15;
  const int yFrac = mvStart.y & 15;
  const int x     = cuX + ( mvStart.x >> 4 ) + u;
  const int y     = cuY + ( mvStart.y >> 4 ) + v;
  const int shift = 4 - ( 10 - pc.bitDepth );
  if( xFrac == 0 && yFrac == 0 )
  {
    return refAt( ref, 0, x, y ) * ( 1 << ( 10 - pc.bitDepth ) );
  }
  if( yFrac == 0 )
  {
    return ( kBilinear[xFrac * 2] * refAt( ref, 0, x, y ) + kBilinear[xFrac * 2 + 1] * refAt( ref, 0, x + 1, y ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  }
  if( xFrac == 0 )
  {
    return ( kBilinear[yFrac * 2] * refAt( ref, 0, x, y ) + kBilinear[yFrac * 2 + 1] * refAt( ref, 0, x, y + 1 ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  }
  const int t0 = ( kBilinear[xFrac * 2] * refAt( ref, 0, x, y ) + kBilinear[xFrac * 2 + 1] * refAt( ref, 0, x + 1, y ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  const int t1 = ( kBilinear[xFrac * 2] * refAt( ref, 0, x, y + 1 ) + kBilinear[xFrac * 2 + 1] * refAt( ref, 0, x + 1, y + 1 ) + ( 1 << ( shift - 1 ) ) ) >> shift;
  return ( kBilinear[yFrac * 2] * t0 + kBilinear[yFrac * 2 + 1] * t1 + 8 ) >> 4;
}

// the SAD of L0 moved by (hor, ver) against L1 moved by (-hor, -ver), every other row
int dmvrSad( int dx, int dy, int hor, int ver )
{
  int sum = 0;
  for( int j = 0; j < dy; j += 2 )
  {
    for( int i = 0; i < dx; i++ )
    {
      sum += abs( s_bil0[( 2 + ver + j ) * 20 + 2 + hor + i] - s_bil1[( 2 - ver + j ) * 20 + 2 - hor + i] );
    }
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

// xSubPelErrorSrfc along one axis: s0 the best, sm the one before it, sp the one after
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

// the refinement of the subblock (in 1/16 luma samples), from the SADs; the centre's cost reduced by a quarter
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

[shader( "compute" )]
[numthreads( THREADS, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int index = int( group.x ) + int( group.y ) * 32768;
  if( index >= pc.count )
  {
    return;   // the whole workgroup
  }
  const int  tid   = int( local.x );
  uint*      job   = pc.jobs + index * 12;
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
  const bool hpel  = ( flags & FLAG_HPEL ) != 0;
  const bool bi    = ( flags & ( FLAG_L0 | FLAG_L1 ) ) == ( FLAG_L0 | FLAG_L1 );
  const bool dmvr  = ( flags & FLAG_DMVR ) != 0;
  const int  maxVal = ( 1 << pc.bitDepth ) - 1;

  // DMVR: the refined motion of the subblock, written for vvdec's TaskFinishMotionInfo
  int2 refined0 = mv0, refined1 = mv1;
  if( dmvr )
  {
    const int2 start0 = clipMv( mv0, cuX, cuY ) - int2( 32, 32 );
    const int2 start1 = clipMv( mv1, cuX, cuY ) - int2( 32, 32 );
    const int  bw     = w + 4;
    for( int s = tid; s < bw * ( h + 4 ); s += THREADS )
    {
      const int v = s / bw;
      const int u = s - v * bw;
      s_bil0[v * 20 + u] = bilinearAt( ref0, cuX, cuY, start0, x - cuX + u, y - cuY + v );
      s_bil1[v * 20 + u] = bilinearAt( ref1, cuX, cuY, start1, x - cuX + u, y - cuY + v );
    }
    GroupMemoryBarrierWithGroupSync();
    if( tid < 25 )
    {
      int sad = dmvrSad( w, h, tid % 5 - 2, tid / 5 - 2 );
      if( tid == 12 )
      {
        sad -= sad >> 2;
      }
      s_sad[tid] = sad;
    }
    GroupMemoryBarrierWithGroupSync();
    if( tid == 0 )
    {
      const int2 delta       = dmvrDelta( w, h );
      const int  index8      = int( job[8] );
      pc.dmvr[index8 * 2]     = delta.x;
      pc.dmvr[index8 * 2 + 1] = delta.y;
      s_delta                = delta;
    }
    GroupMemoryBarrierWithGroupSync();
    refined0 = clipStorage( mv0 + s_delta );
    refined1 = clipStorage( mv1 - s_delta );
  }

  for( int comp = 0; comp < 3; comp++ )
  {
    const int s     = comp == 0 ? 0 : 1;
    const int bx    = x >> s, by = y >> s, bw = w >> s, bh = h >> s;
    const int shift = comp == 0 ? 4 : 5;
    int       pred0[PER_THREAD], pred1[PER_THREAD];

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
      if( dmvr )
      {
        // xFinalPaddedMCForDMVR: clipped at the subblock; when the refinement moves the integer
        // position, the reads stay in the prefetched window of the merge motion (xPrefetchPad)
        clipped = clipMv( mv, x, y );
        const int2 deltaInt = int2( ( mv.x >> shift ) - ( merge.x >> shift ), ( mv.y >> shift ) - ( merge.y >> shift ) );
        if( deltaInt.x != 0 || deltaInt.y != 0 )
        {
          const int  half  = comp == 0 ? 3 : 1;
          const int2 start = clipMv( merge - int2( half << shift, half << shift ), x, y );
          window   = int4( bx + ( start.x >> shift ), by + ( start.y >> shift ), bw + 2 * half + 1, bh + 2 * half + 1 );
          base     = int2( window.x + half + deltaInt.x, window.y + half + deltaInt.y );
          windowed = true;
        }
      }
      else
      {
        clipped = clipMv( mv, cuX, cuY );
      }
      if( list == 0 || ( flags & FLAG_L0 ) == 0 )
      {
        predictBlock( tid, list, slot, comp, bx, by, bw, bh, clipped, bi, hpel, windowed, window, base, pred0 );
      }
      else
      {
        predictBlock( tid, list, slot, comp, bx, by, bw, bh, clipped, bi, hpel, windowed, window, base, pred1 );
      }
    }

    int16_t*   dst      = comp == 0 ? pc.dst0 : ( comp == 1 ? pc.dst1 : pc.dst2 );
    const int  stride   = comp == 0 ? pc.stride0 : pc.stride12;
    const bool residual = ( flags & ( FLAG_RES_Y << comp ) ) != 0;
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
      int       v  = bi ? clip3( 0, maxVal, ( pred0[k] + pred1[k] + offset ) >> shiftNum ) : pred0[k];
      const int at = ( by + j ) * stride + bx + i;
      if( residual )
      {
        v = clip3( 0, maxVal, v + int( dst[at] ) );
      }
      dst[at] = int16_t( v );
    }
  }
}
