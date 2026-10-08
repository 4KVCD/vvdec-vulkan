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
static const int16_t kBilinear[32] = { 16, 0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

// shared by the workgroup: the reference samples a block's filter reads, the
// horizontal pass of a 2-D filter; DMVR's bilinear predictions, its SADs and
// its result
// The filters' and DMVR's arrays share one pool of 16-bit values (each of
// them fits), so that more workgroups fit a compute unit: the filters' the
// reference samples (s_ref) and the horizontal pass (s_tmp), DMVR's bilinear
// predictions (s_bil0, s_bil1), never in use at the same time.
#define REF_AT 0
#define TMP_AT ( ( 16 + 7 ) * ( 16 + 7 ) )
#define BIL0_AT 0
#define BIL1_AT ( 20 * 20 )
groupshared int16_t s_pool[( 16 + 7 ) * ( 16 + 7 ) + ( 16 + 7 ) * 16];
#define s_ref( i ) int( s_pool[REF_AT + ( i )] )
#define s_tmp( i ) int( s_pool[TMP_AT + ( i )] )
#define s_bil0( i ) int( s_pool[BIL0_AT + ( i )] )
#define s_bil1( i ) int( s_pool[BIL1_AT + ( i )] )
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

// xPredInterBlk of a w x h block of a component with an N-tap filter, the
// thread's samples s = tid + THREADS * k into out[k]: the sample (i, j)
// filters the reference around (baseX + i, baseY + j) with the fractions of
// mv; bi: the 14-bit intermediate for averaging, else the final, clipped
// value. The workgroup calls it together: the reference samples the filter
// reads go to shared memory first, clamped (and windowed) once each.
void predictBlock<let N : int>( int tid, int slot, int comp, int bx, int by, int w, int h, int2 mv, bool bi, bool hpel, bool windowed,
                                int4 window, int2 base, inout int out[PER_THREAD] )
{
  int16_t*  ref      = refPlane( slot, comp );
  const int shift    = comp == 0 ? 4 : 5;
  const int xFrac    = mv.x & ( ( 1 << shift ) - 1 );
  const int yFrac    = mv.y & ( ( 1 << shift ) - 1 );
  const int baseX    = windowed ? base.x : bx + ( mv.x >> shift );
  const int baseY    = windowed ? base.y : by + ( mv.y >> shift );
  const int half     = N / 2 - 1;
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int log2W    = log2Int( w );

  int16_t cx[N], cy[N];   // 16 bits: the compiler's full-rate 24-bit multiply-adds then
  [unroll]
  for( int k = 0; k < N; k++ )
  {
    cx[k] = int16_t( coeff( comp, xFrac, hpel, k ) );
    cy[k] = int16_t( coeff( comp, yFrac, hpel, k ) );
  }

  // the samples the filter reads: rw x rh from (baseX - ox, baseY - oy)
  const int ox = xFrac != 0 ? half : 0;
  const int oy = yFrac != 0 ? half : 0;
  const int rw = w + ( xFrac != 0 ? N - 1 : 0 );
  const int rh = h + ( yFrac != 0 ? N - 1 : 0 );
  const int picW = comp == 0 ? pc.picWidth : pc.picWidth >> 1;
  const int picH = comp == 0 ? pc.picHeight : pc.picHeight >> 1;
  if( !windowed && baseX - ox >= 0 && baseX - ox + rw <= picW && baseY - oy >= 0 && baseY - oy + rh <= picH )
  {
    // inside the picture (most blocks): the samples as they are, no clamping
    const int stride = comp == 0 ? pc.stride0 : pc.stride12;
    int16_t*  src    = ref + ( baseY - oy ) * stride + baseX - ox;
    // the thread's sample (c, r), THREADS samples on each step
    const int stepR  = THREADS / rw;
    const int stepC  = THREADS - stepR * rw;
    int       r      = tid / rw;
    int       c      = tid - r * rw;
    for( int s = tid; s < rw * rh; s += THREADS )
    {
      s_pool[REF_AT + s] = src[r * stride + c];
      c += stepC;
      r += stepR;
      if( c >= rw )
      {
        c -= rw;
        r++;
      }
    }
  }
  else
  {
    for( int s = tid; s < rw * rh; s += THREADS )
    {
      const int r = s / rw;
      const int c = s - r * rw;
      s_pool[REF_AT + s] = int16_t( refSample( ref, comp, baseX - ox + c, baseY - oy + r, windowed, window ) );
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if( xFrac != 0 && yFrac != 0 )
  {
    // the horizontal pass over the rows the vertical one reads
    const int shift1  = 6 - headRoom;
    const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
    for( int s = tid; s < rh * w; s += THREADS )
    {
      const int r   = s >> log2W;
      const int i   = s & ( w - 1 );
      int       sum = 0;
      [unroll]
      for( int k = 0; k < N; k++ )
      {
        sum += int( cx[k] ) * s_ref( r * rw + i + k );
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
          sum += int( cy[t] ) * s_tmp( ( j + t ) * w + i );
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
          const int v = s_ref( j * rw + i );
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
              sum += int( cy[t] ) * s_ref( ( j + t ) * rw + i );
            }
          }
          else
          {
            [unroll]
            for( int t = 0; t < N; t++ )
            {
              sum += int( cx[t] ) * s_ref( j * rw + i + t );
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

// bilinearAt where its reads stay inside the picture: src the sample of (u, v)
// = (0, 0), no clamping
int bilinearInside( int16_t* src, int xFrac, int yFrac, int u, int v )
{
  int16_t*  p     = src + v * pc.stride0 + u;
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

// row j of the SAD of L0 moved by (hor, ver) against L1 moved by (-hor, -ver)
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
    // the bilinear predictions of the subblock's (w + 4) x (h + 4) samples
    // from the merge motion minus 2 samples
    const int2 start0 = clipMv( mv0, cuX, cuY ) - int2( 32, 32 );
    const int2 start1 = clipMv( mv1, cuX, cuY ) - int2( 32, 32 );
    const int  bw     = w + 4;
    // the samples the bilinear filters read: (w + 5) x (h + 5) from x0, y0
    const int x00 = x + ( start0.x >> 4 ), y00 = y + ( start0.y >> 4 );
    const int x01 = x + ( start1.x >> 4 ), y01 = y + ( start1.y >> 4 );
    if( x00 >= 0 && x00 + w + 5 <= pc.picWidth && y00 >= 0 && y00 + h + 5 <= pc.picHeight && x01 >= 0 && x01 + w + 5 <= pc.picWidth && y01 >= 0
        && y01 + h + 5 <= pc.picHeight )
    {
      // inside the picture (most subblocks): no clamping
      int16_t*  src0  = refPlane( ref0, 0 ) + y00 * pc.stride0 + x00;
      int16_t*  src1  = refPlane( ref1, 0 ) + y01 * pc.stride0 + x01;
      // the thread's sample (u, v), THREADS samples on each step
      const int stepV = THREADS / bw;
      const int stepU = THREADS - stepV * bw;
      int       v     = tid / bw;
      int       u     = tid - v * bw;
      for( int s = tid; s < bw * ( h + 4 ); s += THREADS )
      {
        s_pool[BIL0_AT + v * 20 + u] = int16_t( bilinearInside( src0, start0.x & 15, start0.y & 15, u, v ) );
        s_pool[BIL1_AT + v * 20 + u] = int16_t( bilinearInside( src1, start1.x & 15, start1.y & 15, u, v ) );
        u += stepU;
        v += stepV;
        if( u >= bw )
        {
          u -= bw;
          v++;
        }
      }
    }
    else
    {
      for( int s = tid; s < bw * ( h + 4 ); s += THREADS )
      {
        const int v = s / bw;
        const int u = s - v * bw;
        s_pool[BIL0_AT + v * 20 + u] = int16_t( bilinearAt( ref0, cuX, cuY, start0, x - cuX + u, y - cuY + v ) );
        s_pool[BIL1_AT + v * 20 + u] = int16_t( bilinearAt( ref1, cuX, cuY, start1, x - cuX + u, y - cuY + v ) );
      }
    }
    if( tid < 25 )
    {
      s_sad[tid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    // the SADs of the 25 offsets on every other row: a row of an offset per
    // step, summed up in shared memory
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
      const int  index8      = int( job[8] );
      pc.dmvr[index8 * 2]     = delta.x;
      pc.dmvr[index8 * 2 + 1] = delta.y;
      s_delta                = delta;
    }
    GroupMemoryBarrierWithGroupSync();
    refined0 = clipStorage( mv0 + s_delta );
    refined1 = clipStorage( mv1 - s_delta );
  }

  [loop]
  for( int comp = 0; comp < 3; comp++ )
  {
    const int s     = comp == 0 ? 0 : 1;
    const int bx    = x >> s, by = y >> s, bw = w >> s, bh = h >> s;
    const int shift = comp == 0 ? 4 : 5;
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
      // one call a filter length: the shader's code stays small (it ran slower
      // when it outgrew the instruction cache)
      int pred[PER_THREAD];
      if( comp == 0 )
      {
        predictBlock<8>( tid, slot, comp, bx, by, bw, bh, clipped, bi, hpel, windowed, window, base, pred );
      }
      else
      {
        predictBlock<4>( tid, slot, comp, bx, by, bw, bh, clipped, bi, hpel, windowed, window, base, pred );
      }
      if( list == 0 || ( flags & FLAG_L0 ) == 0 )
      {
        pred0 = pred;
      }
      else
      {
        pred1 = pred;
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
