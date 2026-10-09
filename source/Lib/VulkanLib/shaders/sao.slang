// Sample adaptive offset (SAO) of one plane of a picture, as vvdec's
// SampleAdaptiveOffset::offsetBlock_core applies it: reads the deblocked
// picture (src) and writes another buffer (dst), one thread per sample.
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
