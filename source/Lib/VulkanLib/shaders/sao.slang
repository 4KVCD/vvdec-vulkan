// Sample adaptive offset (SAO) of a picture, as vvdec's SampleAdaptiveOffset::offsetBlock_core applies it, in place:
// only the CTUs whose SAO is on change, and few do (a tenth of the luma CTUs of the slower-preset films, a twentieth
// of the chroma ones), so the others are neither read nor written. The list holds the CTUs' components to filter, and
// one workgroup takes each.
//
// The ctus buffer holds 8 int16 per CTU and component: the type (-1: off, 0-3: edge offset at 0, 90, 135, 45 degrees,
// 4: band offset), the band position, the neighbour availability (bits: left, right, above, below, above-left,
// above-right, below-left, below-right), an unused entry, then 4 offsets: for edge classes -2, -1, +1, +2, or for the
// 4 bands starting at the band position.
//
// An edge-offset sample changes only when both samples it is compared with lie in CTUs the filter may read (its own
// CTU or an available neighbour); this is what the start and end positions of offsetBlock_core's loops amount to. It
// also stays unchanged when the comparison crosses a virtual boundary, i.e. when the sample lies on the boundary or
// just before it (isProcessDisabled).
//
// The filter compares with the deblocked samples, while the CTUs around may already be filtered. So two passes run,
// a workgroup per list entry each (entry x + 32768 y, in rows of 32768 workgroups): pass 0 saves the deblocked samples
// around the CTU (the row above it and the row below it, each from the column left of it to the column right of it,
// then the column left of it and the column right of it) into the entry's ring; pass 1 filters the CTU in bands of
// BAND rows: it reads a band and the rows above and below it into shared memory (from the plane, the ring, or for the
// row above, the previous band's last row as it was before being filtered, kept aside), applies SAO there and writes
// the band back. (Bands keep the shared memory small, so that several workgroups fit a compute unit, also beside the
// parse's: the whole CTU took 35.8 KB, one workgroup a compute unit.)

#include "common.slang"

struct Params
{
  int16_t* plane0;     // the planes, filtered in place
  int16_t* plane1;
  int16_t* plane2;
  int16_t* ctus;
  uint*    list;       // the CTUs' components to filter: CTU (raster index) | component << 24
  int      count;      // the list's entries
  int16_t* ring;       // per list entry, ringSlot samples: the deblocked samples around the CTU (pass 0 saves them)
  int      stride0;    // the luma plane's stride, the chroma planes'
  int      stride12;
  int      width;      // the luma plane's size in samples
  int      height;
  int      ctuWidth;   // the CTUs' luma size
  int      ctuHeight;
  int      widthInCtus;
  int      bitDepth;
  int      scaleX;     // the chroma planes' subsampling shifts
  int      scaleY;
  int      ringSlot;
  int      pass;       // 0: save the rings, 1: filter
  int      aligned;    // bit c: plane c's rows start 8-byte aligned (its groups of 4 samples are loaded whole)
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

#define MAX_CTU 128
#define BAND 32
// Pass 1: a band of the CTU's samples, its rows y0 - 1 .. y0 + BAND and columns -4 .. w + 3, in groups of 4 samples
// (two to a word); of the first and last groups of each row, only the samples of columns -1 and w are set
#define TILE_GROUPS ( MAX_CTU / 4 + 2 )
groupshared uint2 s_tile[( BAND + 2 ) * TILE_GROUPS];
// the band's last row as deblocked (its groups 1 .. groups): the next band's row above
groupshared uint2 s_carry[TILE_GROUPS];

// A list entry's plane and CTU: the plane, its stride and size, the CTU's place and size in it (cut at the plane's
// edges), the plane's subsampling shifts and component
struct Ctu
{
  int16_t* plane;
  int      stride;
  int      pw, ph;
  int      x0, y0, w, h;
  int      sx, sy;
  int      comp;
  int      index;   // the CTU's raster index
};

Ctu ctuOf( uint entry )
{
  Ctu c;
  c.index        = int( entry & 0xffffffu );
  c.comp         = int( entry >> 24 );
  c.sx           = c.comp == 0 ? 0 : pc.scaleX;
  c.sy           = c.comp == 0 ? 0 : pc.scaleY;
  c.plane        = c.comp == 0 ? pc.plane0 : ( c.comp == 1 ? pc.plane1 : pc.plane2 );
  c.stride       = c.comp == 0 ? pc.stride0 : pc.stride12;
  c.pw           = pc.width >> c.sx;
  c.ph           = pc.height >> c.sy;
  const int ctuW = pc.ctuWidth >> c.sx;
  const int ctuH = pc.ctuHeight >> c.sy;
  c.x0           = ( c.index % pc.widthInCtus ) * ctuW;
  c.y0           = ( c.index / pc.widthInCtus ) * ctuH;
  c.w            = min( ctuW, c.pw - c.x0 );
  c.h            = min( ctuH, c.ph - c.y0 );
  return c;
}

// The place in the plane of ring sample i of the CTU: the row above (w + 2 samples), the row below, the column left
// (h samples), the column right
int2 ringPlace( Ctu c, int i )
{
  if( i < c.w + 2 )
  {
    return int2( c.x0 - 1 + i, c.y0 - 1 );
  }
  i -= c.w + 2;
  if( i < c.w + 2 )
  {
    return int2( c.x0 - 1 + i, c.y0 + c.h );
  }
  i -= c.w + 2;
  return i < c.h ? int2( c.x0 - 1, c.y0 + i ) : int2( c.x0 + c.w, c.y0 + i - c.h );
}

uint2 packGroup( int v0, int v1, int v2, int v3 )
{
  return packSamples( int4( v0, v1, v2, v3 ) );
}

// Sample k (0 to 3) of a group
int groupSample( uint2 g, int k )
{
  const uint w = k < 2 ? g.x : g.y;
  return int( ( k & 1 ) == 0 ? ( w & 0xffffu ) : ( w >> 16 ) );
}

// Three neighbouring groups of a row of the tile: the samples from 4 before a group to 4 after it
struct Row
{
  uint2 g0, g1, g2;

  // Sample j (0 to 11); j is a constant, so the picks fold into register shifts.
  int at( int j )
  {
    return groupSample( j < 4 ? g0 : ( j < 8 ? g1 : g2 ), j & 3 );
  }
};

// Row y (-1 .. BAND, from the band's first row) of the tile around the CTU's group g (whose first sample is column 4 g)
Row tileRow( int y, int g )
{
  const int at = ( y + 1 ) * TILE_GROUPS + g + 1;
  Row       r;
  r.g0 = s_tile[at - 1];
  r.g1 = s_tile[at];
  r.g2 = s_tile[at + 1];
  return r;
}

// Offset i (0 to 3) of the CTU's parameters, a signed 16-bit value each, two to a word in prm.z and prm.w
int offsetOf( uint4 prm, int i )
{
  const uint w = i < 2 ? prm.z : prm.w;
  return int( w << uint( 16 - ( i & 1 ) * 16 ) ) >> 16;
}

// Pass 1: the band's rows y0 - 1 .. y0 + bh into the tile, whole groups each so that no two threads write the same
// group: rows 0 .. h - 1 from the plane (their columns -1 and w from the ring), the band's row above from the ring
// (the CTU's first band) or as the band before left it in s_carry, the row below from the ring (the CTU's last band)
// or the plane (not filtered yet)
void loadBand( Ctu c, int16_t* ring, int y0, int bh, int groups, bool aligned, int tid )
{
  for( int s = tid; s < ( bh + 2 ) * groups; s += 256 )
  {
    const int r = s / groups;   // tile row r: the band's row r - 1
    const int g = s - r * groups;
    const int y = y0 + r - 1;
    uint2     v;
    if( r == 0 && y0 > 0 )
    {
      v = s_carry[g + 1];
    }
    else if( y < 0 || y >= c.h )
    {
      // the ring's row above or below: its samples 4 g + 1 .. 4 g + 4 (its sample 0 is column -1)
      const int at = y < 0 ? 0 : c.w + 2;
      v            = packGroup( int( ring[at + 4 * g + 1] ), int( ring[at + 4 * g + 2] ), int( ring[at + 4 * g + 3] ), int( ring[at + 4 * g + 4] ) );
    }
    else
    {
      int16_t* p = c.plane + ( c.y0 + y ) * c.stride + c.x0 + 4 * g;
      v          = aligned ? *(uint2*)p : packGroup( int( p[0] ), int( p[1] ), int( p[2] ), int( p[3] ) );
    }
    s_tile[r * TILE_GROUPS + g + 1] = v;
  }
  for( int r = tid; r < bh + 2; r += 256 )
  {
    // the columns left and right (sample 3 of the first group, sample 0 of the last): the ring's corners in its rows
    // above and below, its columns in the CTU's rows
    const int y = y0 + r - 1;
    int       left, right;
    if( y < 0 || y >= c.h )
    {
      const int at = y < 0 ? 0 : c.w + 2;
      left         = int( ring[at] );
      right        = int( ring[at + c.w + 1] );
    }
    else
    {
      left  = int( ring[2 * ( c.w + 2 ) + y] );
      right = int( ring[2 * ( c.w + 2 ) + c.h + y] );
    }
    s_tile[r * TILE_GROUPS]              = packGroup( 0, 0, 0, left );
    s_tile[r * TILE_GROUPS + groups + 1] = packGroup( right, 0, 0, 0 );
  }
}

// Pass 1: SAO on the band's rows y0 .. y0 + bh - 1 of the CTU (in the tile), a group of 4 samples of a row at a time,
// written to the plane
void filterBand( Ctu c, uint4 prm, int type, int y0, int bh, int groups, bool aligned, int tid )
{
  const int maxV  = ( 1 << pc.bitDepth ) - 1;
  const int avail = int( prm.y << 16 ) >> 16;
  const int dx    = type == 1 ? 0 : ( type == 3 ? -1 : 1 );
  const int dy    = type == 0 ? 0 : 1;
  for( int s = tid; s < bh * groups; s += 256 )
  {
    const int r = s / groups;   // the band's row
    const int g = s - r * groups;
    const int y = y0 + r;       // the CTU's row
    const Row r0 = tileRow( r, g );
    int       out[4];
    if( type == 4 )
    {
      // band offset: the four bands from the band position get the offsets
      const int bandPos = int( prm.x ) >> 16;
      [unroll]
      for( int k = 0; k < 4; k++ )
      {
        const int v    = r0.at( 4 + k );
        const int band = ( ( v >> ( pc.bitDepth - 5 ) ) - bandPos ) & 31;
        out[k]         = band < 4 ? clip3( 0, maxV, v + offsetOf( prm, band ) ) : v;
      }
    }
    else
    {
      // edge offset: each sample compared with (x - dx, y - dy) and (x + dx, y + dy), when both lie in CTUs the
      // filter may read and the comparison crosses no virtual boundary (the rows above and below read only for the
      // vertical and diagonal classes)
      const Row  ra     = dy != 0 ? tileRow( r - 1, g ) : r0;
      const Row  rb     = dy != 0 ? tileRow( r + 1, g ) : r0;
      const bool inside = g > 0 && 4 * g + 4 < c.w && y > 0 && y + 1 < c.h;
      const bool rowOff = dy != 0 && onBoundary( c.y0 + y, pc.numVbHor, pc.vbHor0, pc.vbHor1, pc.vbHor2, c.sy );
      [unroll]
      for( int k = 0; k < 4; k++ )
      {
        const int  j        = 4 + k;
        const int  v        = r0.at( j );
        const int  va       = type == 0 ? r0.at( j - 1 ) : ( type == 1 ? ra.at( j ) : ( type == 2 ? ra.at( j - 1 ) : ra.at( j + 1 ) ) );
        const int  vb       = type == 0 ? r0.at( j + 1 ) : ( type == 1 ? rb.at( j ) : ( type == 2 ? rb.at( j + 1 ) : rb.at( j - 1 ) ) );
        const int  lx       = 4 * g + k;
        const bool disabled = rowOff || ( dx != 0 && onBoundary( c.x0 + lx, pc.numVbVer, pc.vbVer0, pc.vbVer1, pc.vbVer2, c.sx ) );
        const bool may      = !disabled && ( inside || ( readable( lx - dx, y - dy, c.w, c.h, avail ) && readable( lx + dx, y + dy, c.w, c.h, avail ) ) );
        const int  edge     = sgn( v - va ) + sgn( v - vb );
        out[k]              = may && edge != 0 ? clip3( 0, maxV, v + offsetOf( prm, edge < 0 ? edge + 2 : edge + 1 ) ) : v;
      }
    }
    int16_t* p = c.plane + ( c.y0 + y ) * c.stride + c.x0 + 4 * g;
    if( aligned )
    {
      *(uint2*)p = packGroup( out[0], out[1], out[2], out[3] );
    }
    else
    {
      [unroll]
      for( int k = 0; k < 4; k++ )
      {
        p[k] = int16_t( out[k] );
      }
    }
  }
}

[shader( "compute" )]
[numthreads( 256, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int  tid = int( local.x );
  const uint e   = group.x + group.y * 32768u;
  if( e >= uint( pc.count ) )
  {
    return;   // (the whole workgroup: the last row of workgroups has more than entries)
  }
  const Ctu c    = ctuOf( pc.list[e] );
  int16_t*   ring = pc.ring + e * uint( pc.ringSlot );
  if( pc.pass == 0 )
  {
    // The ring, read with its places clamped into the plane (a sample beyond the plane's edges is never compared
    // with: its CTU is not available)
    for( int i = tid; i < 2 * ( c.w + 2 ) + 2 * c.h; i += 256 )
    {
      const int2 p = ringPlace( c, i );
      ring[i]      = int16_t( loadClamped( c.plane, c.stride, p.x, p.y, c.pw, c.ph ) );
    }
    return;
  }

  const int   groups  = c.w >> 2;   // (the CTUs' widths, cut at the picture's edge too, are multiples of 4 in every plane)
  const bool  aligned = ( ( pc.aligned >> c.comp ) & 1 ) != 0;
  const uint4 prm     = *(uint4*)( pc.ctus + ( c.index * 3 + c.comp ) * 8 );
  const int   type    = int( prm.x << 16 ) >> 16;
  for( int y0 = 0; y0 < c.h; y0 += BAND )
  {
    const int bh = min( BAND, c.h - y0 );
    loadBand( c, ring, y0, bh, groups, aligned, tid );
    GroupMemoryBarrierWithGroupSync();
    filterBand( c, prm, type, y0, bh, groups, aligned, tid );
    // (the band's last row kept as deblocked; then every thread done with the tile before the next band fills it)
    if( y0 + bh < c.h )
    {
      for( int g = tid; g < groups; g += 256 )
      {
        s_carry[g + 1] = s_tile[bh * TILE_GROUPS + g + 1];
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
}
