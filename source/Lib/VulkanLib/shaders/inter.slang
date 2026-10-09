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
//   7, bits 16..19: BCW: L1's weight + 2 (of 8; L0's: 8 - it), with flag 2048
// flags: 1 L0, 2 L1, 4 DMVR, 8 AMVR half-sample filter, 16/32/64 residual in Y/Cb/Cr, 128 LMCS's forward mapping of
// the luma prediction, 256/512 (LMCS's chroma scaling, CIIP) the Cb/Cr prediction kept aside (defer1/2), the residual
// left in the picture for intra.slang's kind 5 or 6, 1024 (CIIP) so the luma's too (defer0), 2048 BCW's weights,
// 4096 BDOF (the variant built with BDOF=1 takes those tiles), 8192 an affine tile (the variant built with AFFINE=1
// takes those, two jobs each: see affineMain)

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
#define FLAG_BCW 2048
#define FLAG_BDOF 4096
#ifndef BDOF
#define BDOF 0
#endif
#ifndef AFFINE
#define AFFINE 0
#endif

// the average of a bi-prediction's two 14-bit predictions (addAvg), or with
// BCW's weights (addWeightedAvg: w1 of 8)
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
  const int bcwW1 = int( ( job[7] >> 16 ) & 15 ) - 2;   // BCW
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
    vy[o] = bi ? biAverage( predY[0][o], predY[1][o], flags, bcwW1, shiftNum, offset, maxVal ) : predY[0][o];
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
    vc[o] = bi ? biAverage( predC[0][o], predC[1][o], flags, bcwW1, shiftNum, offset, maxVal ) : predC[0][o];
  }
  if( ( flags & ( FLAG_RES_Y << ( 1 + cc ) ) ) != 0 )
  {
    vc = clamp( vc + int2( lo16( resC ), hi16( resC ) ), 0, maxVal );
  }
  int16_t* outC = ( flags & ( FLAG_DEFER_CB << cc ) ) != 0 ? ( cc == 0 ? pc.defer1 : pc.defer2 ) + ( ( y >> 1 ) + cj ) * pc.stride12 + ( x >> 1 ) + ci0
                                                           : dstC;
  *(uint*)outC  = uint( vc.x & 0xffff ) | uint( vc.y ) << 16;
}

#if BDOF
// --------------------------------------------------------------------------------------------------------------------
// BDOF, bi-directional optical flow (vvdec's applyBiOptFlow, BiOptFlowCore, gradFilterCore): each list's 14-bit luma
// prediction of the tile with a border of a sample around it, from the reference's integer samples nearest the
// motion (xPredInterBlk), in s_bdof; each 4x4's motion offset (vx, vy) from the gradients and the differences over the
// 6 x 6 around it (the border's own samples and gradients taken from the nearest inside the tile); the samples
// averaged with the offset's correction.
// --------------------------------------------------------------------------------------------------------------------
#define BS 18                                     // a list's rows in s_bdof: the tile, 16 at most, and its border
groupshared int16_t s_bdof[2 * BS * BS];
groupshared int     s_bsum[16 * 5];               // a 4x4's sums: |gx|, |gy|, di by gx's sign, by gy's, gx by gy's sign
groupshared int2    s_bv[16];                     // a 4x4's (vx, vy)
groupshared int     s_minCost;                    // DMVR's best cost (BDOF only where it is 2 w h or more)

int bdofP( int list, int x, int y )   // (x, y): -1 .. w, -1 .. h
{
  return int( s_bdof[list * BS * BS + ( y + 1 ) * BS + x + 1] );
}

// the gradients at (x, y) inside the tile, from the samples around it (the border's too)
int2 bdofGrad( int list, int x, int y )
{
  return int2( ( bdofP( list, x + 1, y ) >> 6 ) - ( bdofP( list, x - 1, y ) >> 6 ), ( bdofP( list, x, y + 1 ) >> 6 ) - ( bdofP( list, x, y - 1 ) >> 6 ) );
}

// a list's prediction into s_bdof: the thread's samples, and the border from the reference: the samples nearest the
// motion (fractions below 8 round down), at 14 bits; at (baseX, baseY) the integer position of the tile's prediction
void bdofStore( int tid, int list, int w, int h, int pred[PER_THREAD], int slot, int baseX, int baseY, int2 mv, bool windowed, int4 window )
{
  const int log2W = log2Int( w );
  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int t = tid + THREADS * k;
    if( t < w * h )
    {
      s_bdof[list * BS * BS + ( ( t >> log2W ) + 1 ) * BS + ( t & ( w - 1 ) ) + 1] = int16_t( pred[k] );
    }
  }
  const int ring  = 2 * ( w + 2 ) + 2 * h;   // the rows above and below, then the columns left and right
  const int xOff  = ( mv.x & 15 ) < 8 ? 1 : 0;
  const int yOff  = ( mv.y & 15 ) < 8 ? 1 : 0;
  const int shift = max( 2, 14 - pc.bitDepth );
  int16_t*  ref   = refPlane( slot, 0 );
  for( int r = tid; r < ring; r += THREADS )
  {
    int bx, by;
    if( r < w + 2 )
    {
      bx = r - 1;
      by = -1;
    }
    else if( r < 2 * ( w + 2 ) )
    {
      bx = r - ( w + 2 ) - 1;
      by = h;
    }
    else if( r < 2 * ( w + 2 ) + h )
    {
      bx = -1;
      by = r - 2 * ( w + 2 );
    }
    else
    {
      bx = w;
      by = r - 2 * ( w + 2 ) - h;
    }
    const int v = refSample( ref, 0, baseX + bx + 1 - xOff, baseY + by + 1 - yOff, windowed, window );
    s_bdof[list * BS * BS + ( by + 1 ) * BS + bx + 1] = int16_t( v * ( 1 << shift ) - IF_INTERNAL_OFFS );
  }
}

// after both lists' bdofStore and a barrier: the thread's samples of the tile, corrected and averaged
void bdofApply( int tid, int w, int h, out int result[PER_THREAD] )
{
  const int nsx = w >> 2;
  const int nsb = nsx * ( h >> 2 );
  // the sums over each 4x4's 6 x 6: four threads a 4x4, nine samples each
  if( tid < 4 * nsb )
  {
    const int sb = tid >> 2;
    const int q  = tid & 3;
    const int sx = sb % nsx;
    const int sy = sb / nsx;
    int       absGX = 0, absGY = 0, dIX = 0, dIY = 0, signGX = 0;
    for( int p = q * 9; p < q * 9 + 9; p++ )
    {
      const int  cx = clamp( 4 * sx - 1 + p % 6, 0, w - 1 );
      const int  cy = clamp( 4 * sy - 1 + p / 6, 0, h - 1 );
      const int2 g0 = bdofGrad( 0, cx, cy );
      const int2 g1 = bdofGrad( 1, cx, cy );
      const int  gx = ( g0.x + g1.x ) >> 1;
      const int  gy = ( g0.y + g1.y ) >> 1;
      const int  di = ( bdofP( 1, cx, cy ) >> 4 ) - ( bdofP( 0, cx, cy ) >> 4 );
      absGX  += abs( gx );
      absGY  += abs( gy );
      dIX    += gx < 0 ? -di : ( gx == 0 ? 0 : di );
      dIY    += gy < 0 ? -di : ( gy == 0 ? 0 : di );
      signGX += gy < 0 ? -gx : ( gy == 0 ? 0 : gx );
    }
    InterlockedAdd( s_bsum[sb * 5], absGX );
    InterlockedAdd( s_bsum[sb * 5 + 1], absGY );
    InterlockedAdd( s_bsum[sb * 5 + 2], dIX );
    InterlockedAdd( s_bsum[sb * 5 + 3], dIY );
    InterlockedAdd( s_bsum[sb * 5 + 4], signGX );
  }
  GroupMemoryBarrierWithGroupSync();
  if( tid < nsb )
  {
    const int absGX  = s_bsum[tid * 5];
    const int absGY  = s_bsum[tid * 5 + 1];
    const int dIX    = s_bsum[tid * 5 + 2];
    const int dIY    = s_bsum[tid * 5 + 3];
    const int signGX = s_bsum[tid * 5 + 4];
    // rightShiftMSB: a shift by the denominator's highest bit
    const int vx   = absGX == 0 ? 0 : clamp( ( dIX * 4 ) >> firstbithigh( uint( absGX ) ), -15, 15 );
    const int data = ( ( vx * ( signGX >> 12 ) ) * 4096 + vx * ( signGX & 4095 ) ) >> 1;
    const int vy   = absGY == 0 ? 0 : clamp( ( dIY * 4 - data ) >> firstbithigh( uint( absGY ) ), -15, 15 );
    s_bv[tid]      = int2( vx, vy );
  }
  GroupMemoryBarrierWithGroupSync();
  const int log2W    = log2Int( w );
  const int shiftNum = 15 - pc.bitDepth;
  const int offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int t = tid + THREADS * k;
    result[k]   = 0;
    if( t < w * h )
    {
      const int  i  = t & ( w - 1 );
      const int  j  = t >> log2W;
      const int2 v  = s_bv[( j >> 2 ) * nsx + ( i >> 2 )];
      const int2 g0 = bdofGrad( 0, i, j );
      const int2 g1 = bdofGrad( 1, i, j );
      const int  b  = v.x * ( g0.x - g1.x ) + v.y * ( g0.y - g1.y );
      result[k]     = clip3( 0, maxVal, int( int16_t( ( bdofP( 0, i, j ) + bdofP( 1, i, j ) + b + offset ) >> shiftNum ) ) );
    }
  }
}
#endif

#if AFFINE
// --------------------------------------------------------------------------------------------------------------------
// Affine motion (xPredAffineBlk): each 4x4 subblock of the tile with motion of its own, worked out from the CU's
// base motion and its steps across and down as setAllAffineMv does (the motion buffer's); the luma with the 6-tap
// filter vvdec takes for 4x4 blocks, refined by PROF; a chroma 4x4 (8x8 luma) with the average of the motion of its
// top-left and bottom-right luma subblocks. A job of 24 words (InterJob::setAffine):
//   0: x | y << 16 of the tile (luma)     1: width | height << 8 | flags << 16 (an InterJob's)
//   2, 3: L0's base motion (x, y; << 7)   4, 5: L1's       6: x | y << 16 of the CU
//   7: L0 reference | L1 reference << 8 | BCW's L1 weight + 2 << 16
//   12: the CU's width | height << 8 | PROF L0 << 16 | PROF L1 << 17 | spread L0 << 18 | spread L1 << 19
//   13..16: L0's steps (across x, y, down x, y; << 7)   17..20: L1's
// --------------------------------------------------------------------------------------------------------------------
static const int kLuma4x4[128] = {
   0, 0,   0, 64,  0,   0,  0,  0,
   0, 1,  -3, 63,  4,  -2,  1,  0,
   0, 1,  -5, 62,  8,  -3,  1,  0,
   0, 2,  -8, 60, 13,  -4,  1,  0,
   0, 3, -10, 58, 17,  -5,  1,  0,
   0, 3, -11, 52, 26,  -8,  2,  0,
   0, 2,  -9, 47, 31, -10,  3,  0,
   0, 3, -11, 45, 34, -10,  3,  0,
   0, 3, -11, 40, 40, -11,  3,  0,
   0, 3, -10, 34, 45, -11,  3,  0,
   0, 3, -10, 31, 47,  -9,  2,  0,
   0, 2,  -8, 26, 52, -11,  3,  0,
   0, 1,  -5, 17, 58, -10,  3,  0,
   0, 1,  -4, 13, 60,  -8,  2,  0,
   0, 1,  -3,  8, 62,  -5,  1,  0,
   0, 1,  -2,  4, 63,  -3,  1,  0
};
groupshared int16_t s_ext[16 * 36];   // PROF: a list's 4x4 predictions with their borders, 6 x 6 each

int roundAffine( int v, int shift )
{
  return ( v + ( 1 << ( shift - 1 ) ) - ( v >= 0 ? 1 : 0 ) ) >> shift;
}

// the motion of the CU's luma subblock (sx, sy)
int2 affineMv( uint job[24], int list, int sx, int sy, int cw, int ch, bool spread )
{
  const int2 base = int2( int( job[2 + 2 * list] ), int( job[3 + 2 * list] ) );
  const int  at   = 13 + 4 * list;
  const int2 hor  = int2( int( job[at] ), int( job[at + 1] ) );
  const int2 ver  = int2( int( job[at + 2] ), int( job[at + 3] ) );
  const int2 pos  = spread ? int2( cw >> 1, ch >> 1 ) : int2( 2 + 4 * sx, 2 + 4 * sy );
  const int2 v    = base + hor * pos.x + ver * pos.y;
  return clipStorage( int2( roundAffine( v.x, 7 ), roundAffine( v.y, 7 ) ) );
}

// a list's luma of the tile: the thread's samples t = tid + THREADS k, 14-bit (bi-prediction, PROF) else final
void affineLuma( int tid, uint job[24], int list, int x, int y, int w, int h, int cuX, int cuY, int cw, int ch, bool spread, bool prof, bool bi,
                 out int pred[PER_THREAD] )
{
  int16_t*  ref      = refPlane( int( ( job[7] >> ( 8 * list ) ) & 255 ), 0 );
  const int nsx      = w >> 2;
  const int nsb      = nsx * ( h >> 2 );
  const int ox       = ( x - cuX ) >> 2;   // the tile's first subblock in the CU
  const int oy       = ( y - cuY ) >> 2;
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;
  const bool hi      = bi || prof;
  const int log2W    = log2Int( w );

  // each subblock's reference samples, 11 x 11 from 3 before its integer position
  for( int s = tid; s < nsb * 121; s += THREADS )
  {
    const int  sb = s / 121;
    const int  r  = ( s % 121 ) / 11;
    const int  c  = s % 11;
    const int  sx = sb % nsx;
    const int  sy = sb / nsx;
    const int2 mv = clipMv( affineMv( job, list, ox + sx, oy + sy, cw, ch, spread ), cuX, cuY );
    s_pool[s]     = int16_t( refAt( ref, 0, cuX + 4 * ( ox + sx ) + ( mv.x >> 4 ) - 3 + c, cuY + 4 * ( oy + sy ) + ( mv.y >> 4 ) - 3 + r ) );
  }
  GroupMemoryBarrierWithGroupSync();
  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int t = tid + THREADS * k;
    pred[k]     = 0;
    if( t >= w * h )
    {
      continue;
    }
    const int  i     = t & ( w - 1 );
    const int  j     = t >> log2W;
    const int  sb    = ( j >> 2 ) * nsx + ( i >> 2 );
    const int  u     = i & 3;
    const int  v     = j & 3;
    const int2 mv    = clipMv( affineMv( job, list, ox + ( i >> 2 ), oy + ( j >> 2 ), cw, ch, spread ), cuX, cuY );
    const int  xFrac = mv.x & 15;
    const int  yFrac = mv.y & 15;
    const int  at    = sb * 121;
    if( xFrac == 0 && yFrac == 0 )
    {
      const int s0 = int( s_pool[at + ( v + 3 ) * 11 + u + 3] );
      pred[k]      = hi ? s0 * ( 1 << headRoom ) - IF_INTERNAL_OFFS : s0;
    }
    else if( xFrac != 0 && yFrac != 0 )
    {
      const int shift1  = 6 - headRoom;
      const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
      int       sum     = 0;
      for( int r = 0; r < 8; r++ )
      {
        int hs = 0;
        for( int c = 0; c < 8; c++ )
        {
          hs += kLuma4x4[xFrac * 8 + c] * int( s_pool[at + ( v + r ) * 11 + u + c] );
        }
        sum += kLuma4x4[yFrac * 8 + r] * ( ( hs + offset1 ) >> shift1 );
      }
      pred[k] = hi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
    }
    else
    {
      const int shiftV = hi ? 6 - headRoom : 6;
      const int offset = hi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
      int       sum    = 0;
      for( int c = 0; c < 8; c++ )
      {
        sum += yFrac != 0 ? kLuma4x4[yFrac * 8 + c] * int( s_pool[at + ( v + c ) * 11 + u + 3] )
                          : kLuma4x4[xFrac * 8 + c] * int( s_pool[at + ( v + 3 ) * 11 + u + c] );
      }
      const int r = ( sum + offset ) >> shiftV;
      pred[k]     = hi ? r : clip3( 0, maxVal, r );
    }
  }
  if( prof )
  {
    // PROF: each 4x4 with a border from the reference's samples nearest its motion (in the window), the gradients
    // inside, each sample moved along them by its offset from the subblock's motion (dMvScaleHor / Ver)
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int t = tid + THREADS * k;
      if( t < w * h )
      {
        const int i = t & ( w - 1 );
        const int j = t >> log2W;
        s_ext[( ( j >> 2 ) * nsx + ( i >> 2 ) ) * 36 + ( ( j & 3 ) + 1 ) * 6 + ( i & 3 ) + 1] = int16_t( pred[k] );
      }
    }
    for( int s = tid; s < nsb * 20; s += THREADS )
    {
      const int sb = s / 20;
      const int r  = s % 20;
      int       bx, by;
      if( r < 6 )
      {
        bx = r - 1;
        by = -1;
      }
      else if( r < 12 )
      {
        bx = r - 7;
        by = 4;
      }
      else if( r < 16 )
      {
        bx = -1;
        by = r - 12;
      }
      else
      {
        bx = 4;
        by = r - 16;
      }
      const int2 mv = clipMv( affineMv( job, list, ox + sb % nsx, oy + sb / nsx, cw, ch, spread ), cuX, cuY );
      const int  sm = int( s_pool[sb * 121 + ( by + ( ( mv.y & 15 ) >> 3 ) + 3 ) * 11 + bx + ( ( mv.x & 15 ) >> 3 ) + 3] );
      s_ext[sb * 36 + ( by + 1 ) * 6 + bx + 1] = int16_t( sm * ( 1 << headRoom ) - IF_INTERNAL_OFFS );
    }
    GroupMemoryBarrierWithGroupSync();
    const int at      = 13 + 4 * list;
    const int quadHX  = int( job[at] ) * 4;
    const int quadHY  = int( job[at + 1] ) * 4;
    const int quadVX  = int( job[at + 2] ) * 4;
    const int quadVY  = int( job[at + 3] ) * 4;
    const int d0H     = ( ( int( job[at] ) + int( job[at + 2] ) ) * 2 ) - ( ( quadHX + quadVX ) * 2 );
    const int d0V     = ( ( int( job[at + 1] ) + int( job[at + 3] ) ) * 2 ) - ( ( quadHY + quadVY ) * 2 );
    const int limit   = 1 << max( pc.bitDepth + 1, 13 );
    const int offset  = ( 1 << ( headRoom - 1 ) ) + IF_INTERNAL_OFFS;
    for( int k = 0; k < PER_THREAD; k++ )
    {
      const int t = tid + THREADS * k;
      if( t < w * h )
      {
        const int i   = t & ( w - 1 );
        const int j   = t >> log2W;
        const int u   = i & 3;
        const int v   = j & 3;
        const int e   = ( ( j >> 2 ) * nsx + ( i >> 2 ) ) * 36 + ( v + 1 ) * 6 + u + 1;
        const int gx  = ( int( s_ext[e + 1] ) >> 6 ) - ( int( s_ext[e - 1] ) >> 6 );
        const int gy  = ( int( s_ext[e + 6] ) >> 6 ) - ( int( s_ext[e - 6] ) >> 6 );
        const int dmH = clamp( roundAffine( d0H + u * quadHX + v * quadVX, 8 ), -31, 31 );
        const int dmV = clamp( roundAffine( d0V + u * quadHY + v * quadVY, 8 ), -31, 31 );
        const int dI  = clamp( dmH * gx + dmV * gy, -limit, limit - 1 );
        const int val = int( int16_t( pred[k] + dI ) );
        pred[k]       = bi ? val : clip3( 0, maxVal, ( val + offset ) >> headRoom );
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();   // before the next writes to the shared memory
}

// a list's chroma of the tile: the thread's sample tid of each component, 14-bit (bi-prediction) else final
void affineChroma( int tid, uint job[24], int list, int x, int y, int w, int h, int cuX, int cuY, int cw, int ch, bool spread, bool bi,
                   out int pred[2] )
{
  const int slot     = int( ( job[7] >> ( 8 * list ) ) & 255 );
  const int cw2      = w >> 1;   // the tile's chroma
  const int ch2      = h >> 1;
  const int ncx      = cw2 >> 2;   // its chroma subblocks
  const int ncb      = ncx * ( ch2 >> 2 );
  const int ocx      = ( x - cuX ) >> 3;   // the tile's first chroma subblock in the CU
  const int ocy      = ( y - cuY ) >> 3;
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;

  // each chroma subblock's motion: the sum of its top-left and bottom-right luma subblocks' rounded, clipped; its
  // reference samples, 7 x 7 from 1 before its integer position, for each component
  for( int s = tid; s < 2 * ncb * 49; s += THREADS )
  {
    const int  comp = 1 + s / ( ncb * 49 );
    const int  rem  = s % ( ncb * 49 );
    const int  cb   = rem / 49;
    const int  r    = ( rem % 49 ) / 7;
    const int  c    = rem % 7;
    const int  csx  = ocx + cb % ncx;
    const int  csy  = ocy + cb / ncx;
    const int2 sum  = affineMv( job, list, 2 * csx, 2 * csy, cw, ch, spread ) + affineMv( job, list, 2 * csx + 1, 2 * csy + 1, cw, ch, spread );
    const int2 mv   = clipMv( int2( roundAffine( sum.x, 1 ), roundAffine( sum.y, 1 ) ), cuX, cuY );
    s_pool[s]       = int16_t( refAt( refPlane( slot, comp ), comp, ( cuX >> 1 ) + 4 * csx + ( mv.x >> 5 ) - 1 + c, ( cuY >> 1 ) + 4 * csy + ( mv.y >> 5 ) - 1 + r ) );
  }
  GroupMemoryBarrierWithGroupSync();
  for( int comp = 1; comp <= 2; comp++ )
  {
    pred[comp - 1] = 0;
    if( tid >= cw2 * ch2 )
    {
      continue;
    }
    const int  i     = tid % cw2;
    const int  j     = tid / cw2;
    const int  cb    = ( j >> 2 ) * ncx + ( i >> 2 );
    const int  csx   = ocx + ( i >> 2 );
    const int  csy   = ocy + ( j >> 2 );
    const int2 sum   = affineMv( job, list, 2 * csx, 2 * csy, cw, ch, spread ) + affineMv( job, list, 2 * csx + 1, 2 * csy + 1, cw, ch, spread );
    const int2 mv    = clipMv( int2( roundAffine( sum.x, 1 ), roundAffine( sum.y, 1 ) ), cuX, cuY );
    const int  xFrac = mv.x & 31;
    const int  yFrac = mv.y & 31;
    const int  at    = ( comp - 1 ) * ncb * 49 + cb * 49;
    const int  u     = i & 3;
    const int  v     = j & 3;
    int        val;
    if( xFrac == 0 && yFrac == 0 )
    {
      const int s0 = int( s_pool[at + ( v + 1 ) * 7 + u + 1] );
      val          = bi ? s0 * ( 1 << headRoom ) - IF_INTERNAL_OFFS : s0;
    }
    else if( xFrac != 0 && yFrac != 0 )
    {
      const int shift1  = 6 - headRoom;
      const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
      int       sum2    = 0;
      for( int r = 0; r < 4; r++ )
      {
        int hs = 0;
        for( int c = 0; c < 4; c++ )
        {
          hs += kChroma[xFrac * 4 + c] * int( s_pool[at + ( v + r ) * 7 + u + c] );
        }
        sum2 += kChroma[yFrac * 4 + r] * ( ( hs + offset1 ) >> shift1 );
      }
      val = bi ? sum2 >> 6 : clip3( 0, maxVal, ( sum2 + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
    }
    else
    {
      const int shiftV = bi ? 6 - headRoom : 6;
      const int offset = bi ? -IF_INTERNAL_OFFS * ( 1 << shiftV ) : 32;
      int       sum2   = 0;
      for( int c = 0; c < 4; c++ )
      {
        sum2 += yFrac != 0 ? kChroma[yFrac * 4 + c] * int( s_pool[at + ( v + c ) * 7 + u + 1] )
                           : kChroma[xFrac * 4 + c] * int( s_pool[at + ( v + 1 ) * 7 + u + c] );
      }
      const int r = ( sum2 + offset ) >> shiftV;
      val         = bi ? r : clip3( 0, maxVal, r );
    }
    pred[comp - 1] = val;
  }
  GroupMemoryBarrierWithGroupSync();
}

void affineMain( int tid, int index )
{
  uint job[24];
  [unroll]
  for( int k = 0; k < 6; k++ )
  {
    const uint4 q    = *(uint4*)( pc.jobs + index * 24 + 4 * k );
    job[4 * k]       = q.x;
    job[4 * k + 1]   = q.y;
    job[4 * k + 2]   = q.z;
    job[4 * k + 3]   = q.w;
  }
  const int  x      = int( job[0] & 0xffff );
  const int  y      = int( job[0] >> 16 );
  const int  w      = int( job[1] & 255 );
  const int  h      = int( ( job[1] >> 8 ) & 255 );
  const int  flags  = int( job[1] >> 16 );
  const int  cuX    = int( job[6] & 0xffff );
  const int  cuY    = int( job[6] >> 16 );
  const int  bcwW1  = int( ( job[7] >> 16 ) & 15 ) - 2;
  const int  cw     = int( job[12] & 255 );
  const int  ch     = int( ( job[12] >> 8 ) & 255 );
  const bool bi     = ( flags & ( FLAG_L0 | FLAG_L1 ) ) == ( FLAG_L0 | FLAG_L1 );
  const int  maxVal = ( 1 << pc.bitDepth ) - 1;

  int pred0[PER_THREAD], pred1[PER_THREAD];
  int predC0[2], predC1[2];
  for( int list = 0; list < 2; list++ )
  {
    if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) == 0 )
    {
      continue;
    }
    const bool prof   = ( ( job[12] >> ( 16 + list ) ) & 1 ) != 0;
    const bool spread = ( ( job[12] >> ( 18 + list ) ) & 1 ) != 0;
    int        p[PER_THREAD];
    int        c[2];
    affineLuma( tid, job, list, x, y, w, h, cuX, cuY, cw, ch, spread, prof, bi, p );
    affineChroma( tid, job, list, x, y, w, h, cuX, cuY, cw, ch, spread, bi, c );
    if( list == 0 || ( flags & FLAG_L0 ) == 0 )
    {
      pred0  = p;
      predC0 = c;
    }
    else
    {
      pred1  = p;
      predC1 = c;
    }
  }

  // the averages (BCW's weights), LMCS's mapping, the residuals added, written
  const int shiftNum = max( 2, 14 - pc.bitDepth ) + 1;
  const int offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
  const int log2W    = log2Int( w );
  for( int k = 0; k < PER_THREAD; k++ )
  {
    const int t = tid + THREADS * k;
    if( t < w * h )
    {
      const int at = ( y + ( t >> log2W ) ) * pc.stride0 + x + ( t & ( w - 1 ) );
      int       v  = bi ? biAverage( pred0[k], pred1[k], flags, bcwW1, shiftNum, offset, maxVal ) : pred0[k];
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
  for( int comp = 1; comp <= 2; comp++ )
  {
    const int cw2 = w >> 1;
    if( tid < cw2 * ( h >> 1 ) )
    {
      int16_t*  dst = comp == 1 ? pc.dst1 : pc.dst2;
      const int at  = ( ( y >> 1 ) + tid / cw2 ) * pc.stride12 + ( x >> 1 ) + tid % cw2;
      int       v   = bi ? biAverage( predC0[comp - 1], predC1[comp - 1], flags, bcwW1, shiftNum, offset, maxVal ) : predC0[comp - 1];
      if( ( flags & ( FLAG_RES_Y << comp ) ) != 0 )
      {
        v = clip3( 0, maxVal, v + int( dst[at] ) );
      }
      ( ( flags & ( FLAG_DEFER_CB << ( comp - 1 ) ) ) != 0 ? ( comp == 1 ? pc.defer1 : pc.defer2 ) : dst )[at] = int16_t( v );
    }
  }
}
#endif

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
#if AFFINE
  affineMain( tid, index );
  return;
#endif
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
  const int  bcwW1 = int( ( job[7] >> 16 ) & 15 ) - 2;   // BCW
  const bool hpel  = ( flags & FLAG_HPEL ) != 0;
  const bool bi    = ( flags & ( FLAG_L0 | FLAG_L1 ) ) == ( FLAG_L0 | FLAG_L1 );
  const bool dmvr  = ( flags & FLAG_DMVR ) != 0;
  const int  maxVal = ( 1 << pc.bitDepth ) - 1;

  if( !BDOF && pc.wide != 0 && w == 16 && h == 16 && tile16Inside( x, y, flags, mv0, mv1 ) )
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
#if BDOF
      // the best cost the search found (or the centre's, where it stops at once)
      int minCost = s_sad[12];
      if( minCost >= w * h )
      {
        for( int i = 0; i < 25; i++ )
        {
          minCost = min( minCost, s_sad[i] );
        }
      }
      s_minCost = minCost;
#endif
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

#if BDOF
  // BDOF: with DMVR, only where the best cost is 2 w h or more (xProcessDMVR)
  const bool bdof = ( flags & FLAG_BDOF ) != 0 && ( !dmvr || s_minCost >= 2 * w * h );
  if( bdof )
  {
    for( int i = tid; i < 16 * 5; i += THREADS )
    {
      s_bsum[i] = 0;   // (predictBlock's barriers before the sums)
    }
  }
#else
  const bool bdof = false;
#endif
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
#if BDOF
      if( bdof && comp == 0 )
      {
        bdofStore( tid, list, bw, bh, pred, slot, windowed ? base.x : bx + ( clipped.x >> shift ), windowed ? base.y : by + ( clipped.y >> shift ),
                   clipped, windowed, window );
      }
#endif
    }
    int bdofOut[PER_THREAD] = { 0, 0, 0, 0 };
#if BDOF
    if( bdof && comp == 0 )
    {
      GroupMemoryBarrierWithGroupSync();
      bdofApply( tid, bw, bh, bdofOut );
    }
#endif

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
      int       v  = bdof && comp == 0 ? bdofOut[k] : ( bi ? biAverage( pred0[k], pred1[k], flags, bcwW1, shiftNum, offset, maxVal ) : pred0[k] );
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
