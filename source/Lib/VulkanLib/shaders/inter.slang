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
// flags: 1 L0, 2 L1, 4 DMVR, 8 AMVR half-sample filter, 16/32/64 residual in Y/Cb/Cr, 128 LMCS's forward mapping of
// the luma prediction, 256/512 (LMCS's chroma scaling, CIIP) the Cb/Cr prediction kept aside (defer1/2), the residual
// left in the picture for intra.slang's kind 5 or 6, 1024 (CIIP) so the luma's too (defer0)

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
  int*       colMotion;   // the picture's colocated motion: vvdec's ColocatedMotionInfo, 5 ints
  int16_t*   lmcsFwd;     // LMCS: the forward luma mapping, a value each
  int16_t*   defer0;      // LMCS, CIIP: the predictions kept aside, laid out as the planes
  int16_t*   defer1;
  int16_t*   defer2;
  int        stride0;
  int        stride12;
  int        count;
  int        picWidth;    // luma
  int        picHeight;
  int        ctuSize;
  int        bitDepth;
  int        ctuLog2;     // the colocated motion's CTUs
  int        widthInCtus;
  int        num8x8;      // its entries a CTU
  int        wide;        // the planes' rows begin 16-byte aligned, with margins of 16 samples or more: tile16 may load them in 8- and 16-byte pieces
};

[[vk::push_constant]] Params pc;

#define FLAG_L0 1
#define FLAG_L1 2
#define FLAG_DMVR 4
#define FLAG_HPEL 8
#define FLAG_RES_Y 16
#define FLAG_LUMA_MAP 128
#define FLAG_DEFER_CB 256
#define FLAG_DEFER_Y 1024

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
// tile16's: the luma windows of the lists (rows of YS), then a work area for
// DMVR's bilinear predictions or the filters' horizontal passes; the chroma
// windows where the luma ones were, once luma is done
#define YW 23                             // a luma window's side: 16 + 7
#define YS 32                             // its rows here: the 4 aligned loads of 8 samples that cover one
#define CWD 11                            // a chroma window's side: 8 + 3
#define CS 16                             // its rows here: 4 aligned loads of 4 samples
#define WY_AT( l ) ( ( l ) * YW * YS )
#define WORK_AT ( 2 * YW * YS )
#define WC_AT( b ) ( ( b ) * CWD * CS )   // b: 2 * list + chroma component - 1
groupshared int16_t s_pool[WORK_AT + 2 * 20 * 20];
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
    // the thread's sample (c, r) at src[at], THREADS samples on each step
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

// bilinearAt where its reads stay inside the picture: p the sample (u, v), no
// clamping
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

// --------------------------------------------------------------------------------------------------------------------
// tile16: a 16x16 tile whose reference samples all lie inside the picture (nearly every tile). Every global read it
// makes -- each list's luma window of 23 x 23 samples around its motion (with DMVR the merge motion: the search and
// the refined prediction both read inside it), the chroma windows of 11 x 11 and the residuals -- goes out at once,
// in aligned loads of 16 and 8 bytes, before the first wait for the memory; DMVR and the filters then work from
// shared memory. (Read and waited for one sample at a time, a tile waited for the memory some 50 times; the shader
// is latency-bound: with a third fewer waves on a SIMD it ran a third slower.)
// --------------------------------------------------------------------------------------------------------------------

// a window's sample (r, c), at the window's sample (0, 0); clamped into the window when DMVR's refinement moved the
// integer position (xPrefetchPad's padding)
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

// the SAD row of DMVR's search, the bilinear predictions at b0 and b1 (rows of 20)
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

// the 4 luma predictions (j, i0 .. i0 + 3) of a list from its window at (its sample (0, 0)): predictBlock's
// arithmetic; tmp the horizontal pass of a 2-D filter
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

// the 2 chroma predictions (j, i0), (j, i0 + 1) of a list from its window: as lumaPred, 4 taps; tmp the horizontal
// pass, rows of 8
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

// dmvrDelta from the SADs in s_sad, the centre's (reduced by a quarter) given: the threads each work it out
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

// DMVR's bilinear prediction (xinitMC, as bilinearInside) of the window's sample at p in shared memory (rows of YS)
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

// whether tile16 may take the tile: each list's windows inside the picture
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
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  const int headRoom = max( 2, 14 - pc.bitDepth );

  // the lists in slots: the first one used in slot 0; with both, L1 in slot 1
  const int  lists = bi ? 2 : 1;
  const bool l0    = ( flags & FLAG_L0 ) != 0;
  int2       mvs[2]  = { l0 ? mv0 : mv1, mv1 };
  int        refs[2] = { l0 ? ref0 : ref1, bi ? ref1 : ( l0 ? ref0 : ref1 ) };
  // the planes, read once for the workgroup (the loads below pick theirs)
  int16_t*   planeY[2]  = { refPlane( refs[0], 0 ), refPlane( refs[1], 0 ) };
  int16_t*   planeCb[2] = { refPlane( refs[0], 1 ), refPlane( refs[1], 1 ) };
  int16_t*   planeCr[2] = { refPlane( refs[0], 2 ), refPlane( refs[1], 2 ) };
  int        wx[2], wy[2], leadY[2], cxs[2], cys[2], leadC[2];
  [unroll]
  for( int s = 0; s < 2; s++ )
  {
    wx[s]    = x + ( mvs[s].x >> 4 ) - 3;
    wy[s]    = y + ( mvs[s].y >> 4 ) - 3;
    leadY[s] = wx[s] & 7;   // the samples before the window in its aligned 16 bytes
    cxs[s]   = ( x >> 1 ) + ( mvs[s].x >> 5 ) - 1;
    cys[s]   = ( y >> 1 ) + ( mvs[s].y >> 5 ) - 1;
    leadC[s] = cxs[s] & 3;   // in its aligned 8 bytes
  }

  // all of the tile's reads: luma windows, 23 rows of 4 x 16 bytes a list
  uint4 yChunk[3];
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int task = tid + THREADS * k;
    if( task < 92 * lists )
    {
      const int s   = task >= 92 ? 1 : 0;
      const int rem = task - 92 * s;
      // (the slot picked by selects: an array indexed by the lane would go to scratch memory)
      const int row0 = s == 0 ? wy[0] : wy[1];
      const int col0 = s == 0 ? wx[0] - leadY[0] : wx[1] - leadY[1];
      int16_t*  p    = ( s == 0 ? planeY[0] : planeY[1] ) + ( row0 + ( rem >> 2 ) ) * pc.stride0 + col0 + 8 * ( rem & 3 );
      yChunk[k]      = *(uint4*)p;
    }
  }
  // chroma windows, 11 rows of 4 x 8 bytes a component and list
  uint2 cChunk[3];
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
      int16_t*  base = s == 0 ? ( comp == 0 ? planeCb[0] : planeCr[0] ) : ( comp == 0 ? planeCb[1] : planeCr[1] );
      const int row0 = s == 0 ? cys[0] : cys[1];
      const int col0 = s == 0 ? cxs[0] - leadC[0] : cxs[1] - leadC[1];
      int16_t*  p    = base + ( row0 + ( rem2 >> 2 ) ) * pc.stride12 + col0 + 4 * ( rem2 & 3 );
      cChunk[k]      = *(uint2*)p;
    }
  }
  // the residuals the picture holds where the thread's predictions go: luma (j, i0 .. i0 + 3), chroma (cj, ci0 ..
  // ci0 + 1) of the component cc
  const int j   = tid >> 2;
  const int i0  = ( tid & 3 ) << 2;
  const int cc  = tid >> 5;
  const int cj  = ( tid & 31 ) >> 2;
  const int ci0 = ( tid & 3 ) << 1;
  int16_t*  dstY = pc.dst0 + ( y + j ) * pc.stride0 + x + i0;
  int16_t*  dstC = ( cc == 0 ? pc.dst1 : pc.dst2 ) + ( ( y >> 1 ) + cj ) * pc.stride12 + ( x >> 1 ) + ci0;
  uint2     resY = uint2( 0, 0 );
  uint      resC = 0;
  if( ( flags & FLAG_RES_Y ) != 0 )
  {
    resY = *(uint2*)dstY;
  }
  if( ( flags & ( FLAG_RES_Y << ( 1 + cc ) ) ) != 0 )
  {
    resC = *(uint*)dstC;
  }

  // the luma windows into shared memory
  [unroll]
  for( int k = 0; k < 3; k++ )
  {
    const int task = tid + THREADS * k;
    if( task < 92 * lists )
    {
      const int s   = task >= 92 ? 1 : 0;
      const int rem = task - 92 * s;
      const int at  = WY_AT( s ) + ( rem >> 2 ) * YS + 8 * ( rem & 3 );
      const uint4 v = yChunk[k];
      s_pool[at]     = int16_t( v.x );
      s_pool[at + 1] = int16_t( v.x >> 16 );
      s_pool[at + 2] = int16_t( v.y );
      s_pool[at + 3] = int16_t( v.y >> 16 );
      s_pool[at + 4] = int16_t( v.z );
      s_pool[at + 5] = int16_t( v.z >> 16 );
      s_pool[at + 6] = int16_t( v.w );
      s_pool[at + 7] = int16_t( v.w >> 16 );
    }
  }
  if( tid < 25 )
  {
    s_sad[tid] = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  // DMVR, as xProcessDMVR: the centre's SAD first -- when it is small enough (83% of VideoQ's subblocks) there is no
  // refinement, and neither the other 24 SADs nor most bilinear predictions are needed
  int2 mvR[2] = { mvs[0], mvs[1] };
  if( dmvr )
  {
    const int shift = 4 - ( 10 - pc.bitDepth );
    // rows 2, 4 .. 16 and columns 2 .. 17 of the bilinear predictions (the windows' samples from (3, 3)), 2 a thread
    int part = 0;
    [unroll]
    for( int k = 0; k < 2; k++ )
    {
      const int e  = tid + THREADS * k;
      const int rc = ( 3 + 2 * ( e >> 4 ) ) * YS + 3 + ( e & 15 );
      part += abs( bilinearW( WY_AT( 0 ) + leadY[0] + rc, mvs[0].x & 15, mvs[0].y & 15, shift )
                   - bilinearW( WY_AT( 1 ) + leadY[1] + rc, mvs[1].x & 15, mvs[1].y & 15, shift ) );
    }
    const int waveSum = WaveActiveSum( part );
    if( WaveIsFirstLane() )
    {
      InterlockedAdd( s_sad[12], waveSum );
    }
    GroupMemoryBarrierWithGroupSync();
    const bool search = s_sad[12] - ( s_sad[12] >> 2 ) >= 16 * 16;
    if( search )
    {
      // all the bilinear predictions of the 20 x 20 samples from the merge motion minus 2 samples (the windows'
      // from (1, 1)), the other 24 SADs on every other row
      [unroll]
      for( int s = 0; s < 2; s++ )
      {
        const int at = WY_AT( s ) + leadY[s] + YS + 1;
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
    // the refinement, worked out by every thread (no waiting for one), from the SADs with the centre's reduced by a
    // quarter
    const int2 delta = search ? dmvrDeltaFrom( s_sad[12] - ( s_sad[12] >> 2 ) ) : int2( 0, 0 );
    // threads 0 .. 15: the refined motion where TMVP takes it, the 8x8 grid, into the colocated motion the CPU stored
    // (DecCu's storeDmvrMotion), an int each; 16, 17: vvdec's DMVR MV cache
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

  // luma: the horizontal passes of the 2-D filters, 23 rows of 16 in groups of 4
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
      const int at      = WY_AT( s ) + leadY[s];
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
      predY[s] = lumaPred( WY_AT( s ) + leadY[s], WORK_AT + 368 * s, j, i0, rx0[s], ry0[s], clampY[s], mvR[s].x & 15, mvR[s].y & 15, cxY[s], cyY[s], bi );
    }
  }
  GroupMemoryBarrierWithGroupSync();   // the luma windows read: the chroma ones go there

  // chroma: the windows into shared memory, then the horizontal passes of the 2-D filters, 11 rows of 8 in pairs
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
      const int at   = WC_AT( 2 * s + comp ) + ( rem2 >> 2 ) * CS + 4 * ( rem2 & 3 );
      const uint2 v  = cChunk[k];
      s_pool[at]     = int16_t( v.x );
      s_pool[at + 1] = int16_t( v.x >> 16 );
      s_pool[at + 2] = int16_t( v.y );
      s_pool[at + 3] = int16_t( v.y >> 16 );
    }
  }
  GroupMemoryBarrierWithGroupSync();
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
        const int at   = WC_AT( 2 * s + comp ) + leadC[s];
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
          s_pool[WORK_AT + 88 * ( 2 * s + comp ) + r * 8 + gi + o] = int16_t( ( sum + offset1 ) >> shift1 );
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
      predC[s] = chromaPred( WC_AT( 2 * s + cc ) + leadC[s], WORK_AT + 88 * ( 2 * s + cc ), cj, ci0, crx0[s], cry0[s], clampC[s], mvR[s].x & 31,
                             mvR[s].y & 31, cxC[s], cyC[s], bi );
    }
  }

  // the averages, the residuals added, written
  const int shiftNum = headRoom + 1;
  const int offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
  int4      vy;
  [unroll]
  for( int o = 0; o < 4; o++ )
  {
    vy[o] = bi ? clip3( 0, maxVal, ( predY[0][o] + predY[1][o] + offset ) >> shiftNum ) : predY[0][o];
  }
  if( ( flags & FLAG_LUMA_MAP ) != 0 )
  {
    vy = int4( pc.lmcsFwd[vy.x], pc.lmcsFwd[vy.y], pc.lmcsFwd[vy.z], pc.lmcsFwd[vy.w] );
  }
  if( ( flags & FLAG_RES_Y ) != 0 )
  {
    vy = clamp( vy + int4( lo16( resY.x ), hi16( resY.x ), lo16( resY.y ), hi16( resY.y ) ), 0, maxVal );
  }
  int16_t* outY = ( flags & FLAG_DEFER_Y ) != 0 ? pc.defer0 + ( y + j ) * pc.stride0 + x + i0 : dstY;
  *(uint2*)outY  = uint2( uint( vy.x & 0xffff ) | uint( vy.y ) << 16, uint( vy.z & 0xffff ) | uint( vy.w ) << 16 );
  int2 vc;
  [unroll]
  for( int o = 0; o < 2; o++ )
  {
    vc[o] = bi ? clip3( 0, maxVal, ( predC[0][o] + predC[1][o] + offset ) >> shiftNum ) : predC[0][o];
  }
  if( ( flags & ( FLAG_RES_Y << ( 1 + cc ) ) ) != 0 )
  {
    vc = clamp( vc + int2( lo16( resC ), hi16( resC ) ), 0, maxVal );
  }
  int16_t* outC = ( flags & ( FLAG_DEFER_CB << cc ) ) != 0 ? ( cc == 0 ? pc.defer1 : pc.defer2 ) + ( ( y >> 1 ) + cj ) * pc.stride12 + ( x >> 1 ) + ci0
                                                           : dstC;
  *(uint*)outC  = uint( vc.x & 0xffff ) | uint( vc.y ) << 16;
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

  if( pc.wide != 0 && w == 16 && h == 16 && tile16Inside( x, y, flags, mv0, mv1 ) )
  {
    tile16( tid, index, job, x, y, flags, mv0, mv1, ref0, ref1, bi, dmvr, hpel );
    return;
  }

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
      // the thread's sample (u, v): at in the pictures, out in shared
      // memory (rows of 20), THREADS samples on each step
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
      // the refined motion where TMVP takes it, the 8x8 grid, into the
      // colocated motion the CPU stored (DecCu's storeDmvrMotion)
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
      int       v  = bi ? clip3( 0, maxVal, ( pred0[k] + pred1[k] + offset ) >> shiftNum ) : pred0[k];
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
