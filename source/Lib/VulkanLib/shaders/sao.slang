// Sample adaptive offset (SAO) of one plane of a picture, as vvdec's
// SampleAdaptiveOffset::offsetBlock_core does it, from the deblocked picture
// (src) into another buffer (dst): a thread per sample.
//
// Per CTU and component, 8 int16: type (-1: off, 0-3: edge offset at 0, 90,
// 135, 45 degrees, 4: band offset), band position, neighbour availability
// (bits: left, right, above, below, above-left, above-right, below-left,
// below-right), unused, then 4 offsets: edge classes -2, -1, +1, +2, or the
// bands from the band position on.
//
// An edge-offset sample changes only when both samples it compares with lie in
// CTUs the filter may read (its own, or an available neighbour): that is what
// the start and end positions of offsetBlock_core's loops amount to. Nor where
// it compares across a virtual boundary: on the boundary, or just before it
// (isProcessDisabled).

#include "common.slang"

struct Params
{
  int16_t* src;
  int16_t* dst;
  int16_t* ctus;
  int      srcStride;
  int      dstStride;
  int      width;    // of the plane
  int      height;
  int      ctuWidth;   // of a CTU in this plane
  int      ctuHeight;
  int      widthInCtus;
  int      comp;
  int      bitDepth;
  int      scaleX;     // the plane's subsampling
  int      scaleY;
  int      numVbVer;   // the picture's virtual boundaries (luma samples), vertical, then horizontal
  int      numVbHor;
  int      vbVer0, vbVer1, vbVer2;
  int      vbHor0, vbHor1, vbHor2;
};

[[vk::push_constant]] Params pc;

int sgn( int v )
{
  return v > 0 ? 1 : ( v < 0 ? -1 : 0 );
}

// whether a sample at p of the plane is on a virtual boundary of positions vb
// (luma samples, scaled by s) or just before one
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

// whether the CTU holding the sample at (lx, ly), relative to this CTU of
// w x h samples, may be read
bool readable( int lx, int ly, int w, int h, int avail )
{
  const int dx = lx < 0 ? 0 : ( lx >= w ? 2 : 1 );
  const int dy = ly < 0 ? 0 : ( ly >= h ? 2 : 1 );
  // [dy][dx] -> bit of avail; the CTU itself is always readable
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
    // the two samples compared: (x - dx, y - dy) and (x + dx, y + dy)
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
