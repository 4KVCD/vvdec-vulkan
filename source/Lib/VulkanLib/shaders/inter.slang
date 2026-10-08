// Inter prediction and the reconstruction of inter blocks, as vvdec's
// InterPrediction and InterpolationFilter do it: uni- and bi-prediction with
// the 8-tap luma and 4-tap chroma filters (and AMVR's half-sample filter),
// and decoder-side motion vector refinement (DMVR). A thread takes one tile of
// up to 16x16 luma samples of a CU (DMVR's subblock when it refines) with its
// chroma, predicts it, adds the residual the picture holds and writes it.
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

// xPredInterBlk of a w x h block of a component: the sample (i, j) filters
// the reference around (baseX + i, baseY + j) with the fractions xFrac, yFrac;
// bi: the 14-bit intermediate for averaging, else the final, clipped value.
// With a window (DMVR's padded prefetch) the reads are clamped into it first.
static int s_tmp[23 * 16];

void interpolate( int16_t* ref, int comp, int baseX, int baseY, int w, int h, int xFrac, int yFrac, bool bi, bool hpel,
                  bool windowed, int4 window, inout int out[256] )
{
  const int n        = comp == 0 ? 8 : 4;
  const int half     = n / 2 - 1;
  const int headRoom = max( 2, 14 - pc.bitDepth );
  const int maxVal   = ( 1 << pc.bitDepth ) - 1;

  #define REF( px, py ) refAt( ref, comp, windowed ? clamp( px, window.x, window.x + window.z - 1 ) : ( px ), windowed ? clamp( py, window.y, window.y + window.w - 1 ) : ( py ) )

  if( xFrac == 0 && yFrac == 0 )
  {
    for( int j = 0; j < h; j++ )
    {
      for( int i = 0; i < w; i++ )
      {
        const int s = REF( baseX + i, baseY + j );
        out[j * w + i] = bi ? s * ( 1 << headRoom ) - IF_INTERNAL_OFFS : s;
      }
    }
  }
  else if( yFrac == 0 || xFrac == 0 )
  {
    const bool vertical = yFrac != 0;
    const int  frac     = vertical ? yFrac : xFrac;
    const int  shift    = bi ? 6 - headRoom : 6;
    const int  offset   = bi ? -IF_INTERNAL_OFFS * ( 1 << shift ) : 32;
    for( int j = 0; j < h; j++ )
    {
      for( int i = 0; i < w; i++ )
      {
        int sum = 0;
        for( int k = 0; k < n; k++ )
        {
          sum += coeff( comp, frac, hpel, k ) * ( vertical ? REF( baseX + i, baseY + j + k - half ) : REF( baseX + i + k - half, baseY + j ) );
        }
        const int v = ( sum + offset ) >> shift;
        out[j * w + i] = bi ? v : clip3( 0, maxVal, v );
      }
    }
  }
  else
  {
    const int shift1  = 6 - headRoom;
    const int offset1 = -IF_INTERNAL_OFFS * ( 1 << shift1 );
    for( int r = 0; r < h + n - 1; r++ )
    {
      for( int i = 0; i < w; i++ )
      {
        int sum = 0;
        for( int k = 0; k < n; k++ )
        {
          sum += coeff( comp, xFrac, hpel, k ) * REF( baseX + i + k - half, baseY + r - half );
        }
        s_tmp[r * 16 + i] = ( sum + offset1 ) >> shift1;
      }
    }
    for( int j = 0; j < h; j++ )
    {
      for( int i = 0; i < w; i++ )
      {
        int sum = 0;
        for( int k = 0; k < n; k++ )
        {
          sum += coeff( comp, yFrac, hpel, k ) * s_tmp[( j + k ) * 16 + i];
        }
        out[j * w + i] = bi ? sum >> 6 : clip3( 0, maxVal, ( sum + ( 1 << ( 5 + headRoom ) ) + ( IF_INTERNAL_OFFS << 6 ) ) >> ( 6 + headRoom ) );
      }
    }
  }
  #undef REF
}

// the prediction of a block of a component from one list: mv is the motion vector to use (clipped), the
// block at (bx, by) in that component
void predictList( int slot, int comp, int bx, int by, int w, int h, int2 mv, bool bi, bool hpel, bool windowed, int4 window, int2 base,
                  inout int out[256] )
{
  const int shift = comp == 0 ? 4 : 5;
  const int xFrac = mv.x & ( ( 1 << shift ) - 1 );
  const int yFrac = mv.y & ( ( 1 << shift ) - 1 );
  const int baseX = windowed ? base.x : bx + ( mv.x >> shift );
  const int baseY = windowed ? base.y : by + ( mv.y >> shift );
  interpolate( refPlane( slot, comp ), comp, baseX, baseY, w, h, xFrac, yFrac, bi, hpel, windowed, window, out );
}

// --------------------------------------------------------------------------------------------------------------------
// DMVR (xProcessDMVR): the bilinear predictions around the merge motion, the SADs of the 5x5 integer offsets
// (every other row), the error surface's sub-sample step
// --------------------------------------------------------------------------------------------------------------------

static int s_bil0[20 * 20];
static int s_bil1[20 * 20];

// xinitMC's bilinear prediction of the CU from mvStart (the clipped merge motion minus 2 samples), for the
// samples (u, v) relative to the CU in [u0, u0 + w) x [v0, v0 + h)
void bilinear( int slot, int cuX, int cuY, int2 mvStart, int u0, int v0, int w, int h, inout int out[400] )
{
  int16_t*  ref   = refPlane( slot, 0 );
  const int xFrac = mvStart.x & 15;
  const int yFrac = mvStart.y & 15;
  const int bx    = cuX + ( mvStart.x >> 4 );
  const int by    = cuY + ( mvStart.y >> 4 );
  const int shift = 4 - ( 10 - pc.bitDepth );
  for( int v = 0; v < h; v++ )
  {
    for( int u = 0; u < w; u++ )
    {
      const int x = bx + u0 + u;
      const int y = by + v0 + v;
      int       val;
      if( xFrac == 0 && yFrac == 0 )
      {
        val = refAt( ref, 0, x, y ) * ( 1 << ( 10 - pc.bitDepth ) );
      }
      else if( yFrac == 0 )
      {
        val = ( kBilinear[xFrac * 2] * refAt( ref, 0, x, y ) + kBilinear[xFrac * 2 + 1] * refAt( ref, 0, x + 1, y ) + ( 1 << ( shift - 1 ) ) ) >> shift;
      }
      else if( xFrac == 0 )
      {
        val = ( kBilinear[yFrac * 2] * refAt( ref, 0, x, y ) + kBilinear[yFrac * 2 + 1] * refAt( ref, 0, x, y + 1 ) + ( 1 << ( shift - 1 ) ) ) >> shift;
      }
      else
      {
        const int t0 = ( kBilinear[xFrac * 2] * refAt( ref, 0, x, y ) + kBilinear[xFrac * 2 + 1] * refAt( ref, 0, x + 1, y ) + ( 1 << ( shift - 1 ) ) ) >> shift;
        const int t1 = ( kBilinear[xFrac * 2] * refAt( ref, 0, x, y + 1 ) + kBilinear[xFrac * 2 + 1] * refAt( ref, 0, x + 1, y + 1 ) + ( 1 << ( shift - 1 ) ) ) >> shift;
        val = ( kBilinear[yFrac * 2] * t0 + kBilinear[yFrac * 2 + 1] * t1 + 8 ) >> 4;
      }
      out[v * 20 + u] = val;
    }
  }
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

// the refinement of the subblock (in 1/16 luma samples)
int2 dmvrRefine( uint* job, int cuX, int cuY, int sx, int sy, int dx, int dy, int2 mv0, int2 mv1, int ref0, int ref1 )
{
  const int2 start0 = clipMv( mv0, cuX, cuY ) - int2( 32, 32 );
  const int2 start1 = clipMv( mv1, cuX, cuY ) - int2( 32, 32 );
  bilinear( ref0, cuX, cuY, start0, sx - cuX, sy - cuY, dx + 4, dy + 4, s_bil0 );
  bilinear( ref1, cuX, cuY, start1, sx - cuX, sy - cuY, dx + 4, dy + 4, s_bil1 );

  int minCost = dmvrSad( dx, dy, 0, 0 );
  minCost -= minCost >> 2;
  if( minCost < dx * dy )
  {
    return int2( 0, 0 );
  }

  int sads[25];
  sads[12]  = minCost;
  int bestH = 0, bestV = 0;
  for( int ver = -2; ver <= 2; ver++ )
  {
    for( int hor = -2; hor <= 2; hor++ )
    {
      if( ver != 0 || hor != 0 )
      {
        sads[( ver + 2 ) * 5 + hor + 2] = dmvrSad( dx, dy, hor, ver );
      }
      const int cost = sads[( ver + 2 ) * 5 + hor + 2];
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
    delta.x += subPel( sads[c], sads[c - 1], sads[c + 1], 0 );
    delta.y += subPel( sads[c], sads[c - 5], sads[c + 5], 0 );
  }
  return delta;
}

int2 clipStorage( int2 mv )
{
  return int2( clamp( mv.x, -( 1 << 17 ), ( 1 << 17 ) - 1 ), clamp( mv.y, -( 1 << 17 ), ( 1 << 17 ) - 1 ) );
}

[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  if( int( id.x ) >= pc.count )
  {
    return;
  }
  uint*      job   = pc.jobs + int( id.x ) * 12;
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
    const int2 delta = dmvrRefine( job, cuX, cuY, x, y, w, h, mv0, mv1, ref0, ref1 );
    const int  index = int( job[8] );
    pc.dmvr[index * 2]     = delta.x;
    pc.dmvr[index * 2 + 1] = delta.y;
    refined0 = clipStorage( mv0 + delta );
    refined1 = clipStorage( mv1 - delta );
  }

  int pred0[256], pred1[256];
  for( int comp = 0; comp < 3; comp++ )
  {
    const int s  = comp == 0 ? 0 : 1;
    const int bx = x >> s, by = y >> s, bw = w >> s, bh = h >> s;
    const int shift = comp == 0 ? 4 : 5;

    for( int list = 0; list < 2; list++ )
    {
      if( ( flags & ( list == 0 ? FLAG_L0 : FLAG_L1 ) ) == 0 )
      {
        continue;
      }
      const int  slot   = list == 0 ? ref0 : ref1;
      const int2 merge  = list == 0 ? mv0 : mv1;
      const int2 mv     = list == 0 ? refined0 : refined1;
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
        predictList( slot, comp, bx, by, bw, bh, clipped, bi, hpel, windowed, window, base, pred0 );
      }
      else
      {
        predictList( slot, comp, bx, by, bw, bh, clipped, bi, hpel, windowed, window, base, pred1 );
      }
    }

    int16_t*   dst      = comp == 0 ? pc.dst0 : ( comp == 1 ? pc.dst1 : pc.dst2 );
    const int  stride   = comp == 0 ? pc.stride0 : pc.stride12;
    const bool residual = ( flags & ( FLAG_RES_Y << comp ) ) != 0;
    const int  shiftNum = max( 2, 14 - pc.bitDepth ) + 1;
    const int  offset   = ( 1 << ( shiftNum - 1 ) ) + 2 * IF_INTERNAL_OFFS;
    for( int j = 0; j < bh; j++ )
    {
      for( int i = 0; i < bw; i++ )
      {
        int v = bi ? clip3( 0, maxVal, ( pred0[j * bw + i] + pred1[j * bw + i] + offset ) >> shiftNum ) : pred0[j * bw + i];
        const int at = ( by + j ) * stride + bx + i;
        if( residual )
        {
          v = clip3( 0, maxVal, v + int( dst[at] ) );
        }
        dst[at] = int16_t( v );
      }
    }
  }
}
