// Sample adaptive offset (SAO) of one plane of a picture, as vvdec's
// SampleAdaptiveOffset::offsetBlock_core applies it: reads the deblocked
// picture (src) and writes another buffer (dst), one thread per sample, or with
// VEC8 eight samples of a row per thread (sao_vec8: 16-byte loads and stores,
// for planes whose rows start 16-byte aligned).
//
// The ctus buffer holds 8 int16 per CTU and component: the type (-1: off,
// 0-3: edge offset at 0, 90, 135, 45 degrees, 4: band offset), the band
// position, the neighbour availability (bits: left, right, above, below,
// above-left, above-right, below-left, below-right), an unused entry, then
// 4 offsets: for edge classes -2, -1, +1, +2, or for the 4 bands starting at
// the band position.
//
// An edge-offset sample changes only when both samples it is compared with lie
// in CTUs the filter may read (its own CTU or an available neighbour); this is
// what the start and end positions of offsetBlock_core's loops amount to. It
// also stays unchanged when the comparison crosses a virtual boundary, i.e.
// when the sample lies on the boundary or just before it (isProcessDisabled).

#include "common.slang"

struct Params
{
  int16_t* src;
  int16_t* dst;
  int16_t* ctus;
  int      srcStride;
  int      dstStride;
  int      width;    // plane size in samples
  int      height;
  int      ctuWidth;   // CTU size in this plane's samples
  int      ctuHeight;
  int      widthInCtus;
  int      comp;
  int      bitDepth;
  int      scaleX;     // the plane's subsampling shifts
  int      scaleY;
  int      numVbVer;   // number of vertical, then horizontal virtual boundaries; positions in luma samples
  int      numVbHor;
  int      vbVer0, vbVer1, vbVer2;
  int      vbHor0, vbHor1, vbHor2;
};

[[vk::push_constant]] Params pc;

int sgn( int v )
{
  return v > 0 ? 1 : ( v < 0 ? -1 : 0 );
}

// Whether position p of the plane lies on one of the virtual boundaries vb0-vb2
// (luma samples, shifted right by s for this plane) or just before one.
bool onBoundary( int p, int num, int vb0, int vb1, int vb2, int s )
{
  const int vb[3] = { vb0, vb1, vb2 };
  for( int i = 0; i < num; i++ )
  {
    const int v = vb[i] >> s;
    if( p == v || p == v - 1 )
    {
      return true;
    }
  }
  return false;
}

// Whether the filter may read the CTU that holds the sample at (lx, ly), given
// relative to the current CTU of w x h samples.
bool readable( int lx, int ly, int w, int h, int avail )
{
  const int dx = lx < 0 ? 0 : ( lx >= w ? 2 : 1 );
  const int dy = ly < 0 ? 0 : ( ly >= h ? 2 : 1 );
  // Maps [dy][dx] to the bit of avail; -1 marks the current CTU, which is always readable.
  static const int kBit[9] = { 4, 2, 5, 0, -1, 1, 6, 3, 7 };
  const int bit = kBit[dy * 3 + dx];
  return bit < 0 || ( ( avail >> bit ) & 1 ) != 0;
}

#ifndef VEC8
#  define VEC8 0
#endif

#if VEC8

// Sample k (0 to 7) of eight packed in a uint4, two to a word
int sampleOf( uint4 s, int k )
{
  const uint w = k < 2 ? s.x : ( k < 4 ? s.y : ( k < 6 ? s.z : s.w ) );
  return int( ( w >> uint( ( k & 1 ) * 16 ) ) & 0xffffu );
}

// Offset i (0 to 3) of the CTU's parameters, a signed 16-bit value each, two to a word in prm.z and prm.w
int offsetOf( uint4 prm, int i )
{
  const uint w = i < 2 ? prm.z : prm.w;
  return int( w << uint( 16 - ( i & 1 ) * 16 ) ) >> 16;
}

// Eight samples of a row a thread, from (x0, y) on: a workgroup is 8 threads (64 samples) across and 8 rows down.
// A thread's samples lie in one CTU (CTU widths are multiples of 8), whose parameters it reads in one 16-byte load.
// The rows above and below, read whole for edge offsets that look across rows, are shared with the neighbouring
// threads' reads in the caches; the samples just left and right of the eight are read one by one.
[shader( "compute" )]
[numthreads( 8, 8, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  const int x0 = int( id.x ) * 8;
  const int y  = int( id.y );
  if( x0 >= pc.width || y >= pc.height )
  {
    return;
  }
  const int   log2W = firstbithigh( uint( pc.ctuWidth ) );    // (CTU sizes are powers of two)
  const int   log2H = firstbithigh( uint( pc.ctuHeight ) );
  const int   cx    = x0 >> log2W;
  const int   cy    = y >> log2H;
  const uint4 prm   = *(uint4*)( pc.ctus + ( ( cy * pc.widthInCtus + cx ) * 3 + pc.comp ) * 8 );
  const int   type  = int( prm.x << 16 ) >> 16;
  const uint4 cur   = *(uint4*)( pc.src + y * pc.srcStride + x0 );
  const int   maxV  = ( 1 << pc.bitDepth ) - 1;
  int         out[8];
  [unroll]
  for( int k = 0; k < 8; k++ )
  {
    out[k] = sampleOf( cur, k );
  }

  if( type == 4 )
  {
    // band offset: the four bands from the band position get the offsets
    const int bandPos = int( prm.x ) >> 16;
    [unroll]
    for( int k = 0; k < 8; k++ )
    {
      const int band = ( ( out[k] >> ( pc.bitDepth - 5 ) ) - bandPos ) & 31;
      if( band < 4 )
      {
        out[k] = clip3( 0, maxV, out[k] + offsetOf( prm, band ) );
      }
    }
  }
  else if( type >= 0 )
  {
    // edge offset: each sample is compared with (x - dx, y - dy) and (x + dx, y + dy)
    const int dx = type == 1 ? 0 : ( type == 3 ? -1 : 1 );
    const int dy = type == 0 ? 0 : 1;
    // The rows compared with, a and b: the samples from x0 - 1 to x0 + 8 of each. Rows and columns are clamped into
    // the plane: a sample outside it is never used, as its CTU is not available.
    const int ya = clamp( y - dy, 0, pc.height - 1 );
    const int yb = clamp( y + dy, 0, pc.height - 1 );
    const int xl = max( x0 - 1, 0 );
    const int xr = min( x0 + 8, pc.width - 1 );
    const uint4 rowA = dy == 0 ? cur : *(uint4*)( pc.src + ya * pc.srcStride + x0 );
    const uint4 rowB = dy == 0 ? cur : *(uint4*)( pc.src + yb * pc.srcStride + x0 );
    int a[10];
    int b[10];
    a[0] = int( uint( pc.src[ya * pc.srcStride + xl] ) & 0xffffu );
    a[9] = int( uint( pc.src[ya * pc.srcStride + xr] ) & 0xffffu );
    b[0] = int( uint( pc.src[yb * pc.srcStride + xl] ) & 0xffffu );
    b[9] = int( uint( pc.src[yb * pc.srcStride + xr] ) & 0xffffu );
    [unroll]
    for( int k = 0; k < 8; k++ )
    {
      a[k + 1] = sampleOf( rowA, k );
      b[k + 1] = sampleOf( rowB, k );
    }
    // The CTU's size (cut at the plane's edges) and the position of the thread's samples in it
    const int  avail = int( prm.y << 16 ) >> 16;
    const int  w     = min( pc.ctuWidth, pc.width - ( cx << log2W ) );
    const int  h     = min( pc.ctuHeight, pc.height - ( cy << log2H ) );
    const int  lx0   = x0 - ( cx << log2W );
    const int  ly    = y - ( cy << log2H );
    const bool rowOff = dy != 0 && onBoundary( y, pc.numVbHor, pc.vbHor0, pc.vbHor1, pc.vbHor2, pc.scaleY );
    [unroll]
    for( int k = 0; k < 8; k++ )
    {
      const int  lx       = lx0 + k;
      const bool disabled = rowOff || ( dx != 0 && onBoundary( x0 + k, pc.numVbVer, pc.vbVer0, pc.vbVer1, pc.vbVer2, pc.scaleX ) );
      if( !disabled && readable( lx - dx, ly - dy, w, h, avail ) && readable( lx + dx, ly + dy, w, h, avail ) )
      {
        const int v    = out[k];
        const int va   = dx == 0 ? a[k + 1] : ( dx > 0 ? a[k] : a[k + 2] );
        const int vb   = dx == 0 ? b[k + 1] : ( dx > 0 ? b[k + 2] : b[k] );
        const int edge = sgn( v - va ) + sgn( v - vb );
        if( edge != 0 )
        {
          out[k] = clip3( 0, maxV, v + offsetOf( prm, edge < 0 ? edge + 2 : edge + 1 ) );
        }
      }
    }
  }

  const uint4 packed = uint4( uint( out[0] ) | uint( out[1] ) << 16, uint( out[2] ) | uint( out[3] ) << 16, uint( out[4] ) | uint( out[5] ) << 16,
                              uint( out[6] ) | uint( out[7] ) << 16 );
  if( x0 + 8 <= pc.width )
  {
    *(uint4*)( pc.dst + y * pc.dstStride + x0 ) = packed;
  }
  else
  {
    *(uint2*)( pc.dst + y * pc.dstStride + x0 ) = packed.xy;   // (a chroma row may end 4 samples into the eight)
  }
}

#else

[shader( "compute" )]
[numthreads( 16, 16, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  const int x = int( id.x );
  const int y = int( id.y );
  if( x >= pc.width || y >= pc.height )
  {
    return;
  }

  const int cx   = x / pc.ctuWidth;
  const int cy   = y / pc.ctuHeight;
  const int base = ( ( cy * pc.widthInCtus + cx ) * 3 + pc.comp ) * 8;
  const int type = int( pc.ctus[base] );
  const int v    = loadPel( pc.src, pc.srcStride, x, y );
  int       out  = v;

  if( type == 4 )
  {
    const int band = ( ( v >> ( pc.bitDepth - 5 ) ) - int( pc.ctus[base + 1] ) ) & 31;
    if( band < 4 )
    {
      out = clip3( 0, ( 1 << pc.bitDepth ) - 1, v + int( pc.ctus[base + 4 + band] ) );
    }
  }
  else if( type >= 0 )
  {
    int dx = 1, dy = 0;
    if( type == 1 )
    {
      dx = 0; dy = 1;
    }
    else if( type == 2 )
    {
      dx = 1; dy = 1;
    }
    else if( type == 3 )
    {
      dx = -1; dy = 1;
    }
    // The sample is compared with (x - dx, y - dy) and (x + dx, y + dy).
    const int lx    = x - cx * pc.ctuWidth;
    const int ly    = y - cy * pc.ctuHeight;
    const int w     = min( pc.ctuWidth, pc.width - cx * pc.ctuWidth );
    const int h     = min( pc.ctuHeight, pc.height - cy * pc.ctuHeight );
    const int  avail    = int( pc.ctus[base + 2] );
    const bool disabled = ( dx != 0 && onBoundary( x, pc.numVbVer, pc.vbVer0, pc.vbVer1, pc.vbVer2, pc.scaleX ) )
                          || ( dy != 0 && onBoundary( y, pc.numVbHor, pc.vbHor0, pc.vbHor1, pc.vbHor2, pc.scaleY ) );
    if( !disabled && readable( lx - dx, ly - dy, w, h, avail ) && readable( lx + dx, ly + dy, w, h, avail ) )
    {
      const int edge = sgn( v - loadPel( pc.src, pc.srcStride, x - dx, y - dy ) ) + sgn( v - loadPel( pc.src, pc.srcStride, x + dx, y + dy ) );
      if( edge != 0 )
      {
        const int offset = int( pc.ctus[base + 4 + ( edge < 0 ? edge + 2 : edge + 1 )] );
        out = clip3( 0, ( 1 << pc.bitDepth ) - 1, v + offset );
      }
    }
  }

  storePel( pc.dst, pc.dstStride, x, y, out );
}

#endif
