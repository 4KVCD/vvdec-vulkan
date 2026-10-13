// The adaptive loop filter (ALF) and the cross-component ALF (CC-ALF) of a
// picture, as vvdec's AdaptiveLoopFilter applies them (deriveClassificationBlk,
// filterBlk, filterBlkCcAlf). The filters see the whole picture, with the
// samples beyond its edges replicated (vvdec's border extension).
//
// BOUNDS=1 is the variant for pictures with virtual boundaries, or with slice,
// tile or subpicture boundaries that the loop filters must not cross
// (AdaptiveLoopFilter::filterCTU). A CTU is split into regions at the virtual
// boundaries inside it, and a sample's filter sees only the sample's region,
// with the samples beyond the region's clipped sides replicated (see Region).
// In this variant each thread reads the samples it needs from the picture.
//
// CHROMA=0: a thread classifies one 4x4 luma block and filters it (7x7
//           diamond), from the workgroup's 32x32 samples and a 3-sample
//           margin, which the workgroup reads into shared memory once.
// CHROMA=1: a thread filters one 4x4 block of Cb and one of Cr (5x5 diamond),
//           then adds the cross-component filter of the co-located luma: in
//           pass 0 the blocks whose CTU has ALF or CC-ALF on that component,
//           from the picture (src) into the scratch planes (dst); in pass 1
//           those blocks back, from the scratch planes (src) into the picture
//           (dst). The other blocks keep their samples where they are (most of
//           the chroma CTUs: copying them through the scratch planes took the
//           pass's time, which reads and writes whole planes at memory's pace).
//
// The ctus buffer holds 8 bytes per CTU: the luma filter set (255: off), the
// chroma APS, the Cb alternative (255: off), the Cr alternative (255: off),
// the Cb and Cr CC-ALF filters as APS * 4 + filter (255: off), the CTU's
// boundaries (BOUNDS only: clipped top 1, bottom 2, left 4, right 8;
// rasterSliceAlfPad << 4), and 1 unused byte.

#include "common.slang"

#define CTU_LUMA_SET 0
#define CTU_CHROMA_APS 1
#define CTU_CB_ALT 2
#define CTU_CR_ALT 3
#define CTU_CC_CB 4
#define CTU_CC_CR 5
#define CTU_BOUNDS 6

#define NUM_CLASSES 25
#define LUMA_COEFF 13
#define CHROMA_COEFF 7
#define LUMA_SET_SIZE ( 4 * NUM_CLASSES * LUMA_COEFF )

struct Params
{
  int16_t* src0;      // luma, or Cb
  int16_t* dst0;
  int16_t* src1;      // Cr
  int16_t* dst1;
  int16_t* srcLuma;   // CHROMA: the luma plane, for CC-ALF
  uint8_t* ctus;
  int16_t* coeff;     // luma: [set][transpose][class][13]; chroma: [aps][alt][7]
  int16_t* clip;      // clipping values, laid out like coeff
  int16_t* ccCoeff;   // [aps][component][filter][8]
  int      srcStride;
  int      dstStride;
  int      lumaStride;
  int      width;     // size of the filtered plane, in samples
  int      height;
  int      lumaWidth;
  int      lumaHeight;
  int      ctuLog2;
  int      widthInCtus;
  int      bitDepth;
  int      scaleX;    // chroma subsampling shifts
  int      scaleY;
  int      numVbVer;  // BOUNDS: number of vertical, then horizontal virtual boundaries; positions in luma samples
  int      numVbHor;
  int      vbVer0, vbVer1, vbVer2;
  int      vbHor0, vbHor1, vbHor2;
  int      aligned;   // CHROMA: nonzero when both planes' rows start 8-byte aligned (a block's rows read and written whole)
  int      pass;      // CHROMA: 0 filters (into the scratch planes), 1 copies back (into the picture)
};

[[vk::push_constant]] Params pc;

int clipAlf( int c, int cur, int a, int b )
{
  return clip3( -c, c, a - cur ) + clip3( -c, c, b - cur );
}

#ifndef BOUNDS
#define BOUNDS 0
#endif

#if BOUNDS
// The region of its CTU that a sample lies in, after filterCTU splits the CTU at
// the virtual boundaries inside it (luma samples; x1 and y1 are exclusive).
// clipL .. clipB tell which of the region's sides clip the filter: a virtual
// boundary, a slice, tile or subpicture boundary that loop filtering must not
// cross (the CTU's, as isClipOrCrossedByVirtualBoundaries finds them), or the
// picture's edge. pad marks the CTU's raster-slice corners (rasterSliceAlfPad:
// 1 the top-left, 2 the bottom-right), for the regions at those corners.
struct Region
{
  int  x0, x1, y0, y1;
  bool clipL, clipR, clipT, clipB;
  int  pad;
  int  cx0, cy0, cx1, cy1;   // the CTU's bounds
};

Region regionOf( int lx, int ly )
{
  const int ctuSize = 1 << pc.ctuLog2;
  Region    r;
  r.cx0             = ( lx >> pc.ctuLog2 ) << pc.ctuLog2;
  r.cy0             = ( ly >> pc.ctuLog2 ) << pc.ctuLog2;
  r.cx1             = min( r.cx0 + ctuSize, pc.lumaWidth );
  r.cy1             = min( r.cy0 + ctuSize, pc.lumaHeight );
  const int flags   = int( pc.ctus[( ( ly >> pc.ctuLog2 ) * pc.widthInCtus + ( lx >> pc.ctuLog2 ) ) * 8 + CTU_BOUNDS] );
  r.x0              = r.cx0;
  r.x1              = r.cx1;
  r.y0              = r.cy0;
  r.y1              = r.cy1;
  r.clipT           = ( flags & 1 ) != 0 || r.y0 == 0;
  r.clipB           = ( flags & 2 ) != 0 || r.y1 == pc.lumaHeight;
  r.clipL           = ( flags & 4 ) != 0 || r.x0 == 0;
  r.clipR           = ( flags & 8 ) != 0 || r.x1 == pc.lumaWidth;
  const int ver[3]  = { pc.vbVer0, pc.vbVer1, pc.vbVer2 };
  const int hor[3]  = { pc.vbHor0, pc.vbHor1, pc.vbHor2 };
  for( int i = 0; i < pc.numVbVer; i++ )
  {
    const int v = ver[i];
    if( r.cx0 < v && v < r.cx1 )
    {
      if( v <= lx )
      {
        r.x0    = max( r.x0, v );
        r.clipL = true;
      }
      else
      {
        r.x1    = min( r.x1, v );
        r.clipR = true;
      }
    }
  }
  for( int i = 0; i < pc.numVbHor; i++ )
  {
    const int v = hor[i];
    if( r.cy0 < v && v < r.cy1 )
    {
      if( v <= ly )
      {
        r.y0    = max( r.y0, v );
        r.clipT = true;
      }
      else
      {
        r.y1    = min( r.y1, v );
        r.clipB = true;
      }
    }
  }
  const int pad = flags >> 4;
  r.pad         = ( ( pad & 1 ) != 0 && r.x0 == r.cx0 && r.y0 == r.cy0 ? 1 : 0 ) | ( ( pad & 2 ) != 0 && r.x1 == r.cx1 && r.y1 == r.cy1 ? 2 : 0 );
  return r;
}

// The same region in a plane subsampled by sx, sy.
Region scaled( Region r, int sx, int sy )
{
  r.x0  >>= sx; r.x1  >>= sx; r.cx0 >>= sx; r.cx1 >>= sx;
  r.y0  >>= sy; r.y1  >>= sy; r.cy0 >>= sy; r.cy1 >>= sy;
  return r;
}

// Reads a sample of a plane (w x h) as the filter of a sample in region r sees
// it, matching filterCTU's copy of the region with its margin: at the raster
// slice's corners, the CTU's edge column is padded outwards (padBorderPel);
// beyond the clipped sides, the region's edge is replicated (extendBorderPel);
// beyond the picture, the picture's edge.
int fetch( int16_t* p, int stride, int x, int y, Region r, int w, int h )
{
  if( ( r.pad & 1 ) != 0 && x < r.cx0 && y < r.cy0 )
  {
    x = r.cx0;
  }
  if( ( r.pad & 2 ) != 0 && x >= r.cx1 && y >= r.cy1 )
  {
    x = r.cx1 - 1;
  }
  x = r.clipL ? max( x, r.x0 ) : x;
  x = r.clipR ? min( x, r.x1 - 1 ) : x;
  y = r.clipT ? max( y, r.y0 ) : y;
  y = r.clipB ? min( y, r.y1 - 1 ) : y;
  return loadClamped( p, stride, x, y, w, h );
}
#endif

#if CHROMA == 0
// The luma filter's steps that both variants (BOUNDS's and the tiled one) share

static const int kTh[16] = { 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4 };
static const int kTransposeTable[8] = { 0, 1, 0, 2, 2, 3, 1, 3 };

// A 4x4 block's class and transpose (AdaptiveLoopFilter::deriveClassificationBlk)
struct AlfClass
{
  int classIdx;
  int transposeIdx;
};

// The block's class and transpose from the Laplacians summed over its window (vertical, horizontal and the two
// diagonals): its activity, and its main direction and how strong it is. byVb: the block's row in its CTU, next to the
// virtual boundary (vbPos) the window has fewer rows and the activity is scaled up.
AlfClass alfClassOf( int sumV, int sumH, int sumD0, int sumD1, int byVb, int vbPos )
{
  const int shift    = pc.bitDepth + 4;
  const int tempAct  = sumV + sumH;
  const int scale    = ( byVb == vbPos - 4 || byVb == vbPos ) ? 96 : 64;
  const int activity = clip3( 0, 15, ( tempAct * scale ) >> shift );
  int       classIdx = kTh[activity];

  int hv1, hv0, d1, d0, dirTempHV, dirTempD;
  if( sumV > sumH )
  {
    hv1 = sumV; hv0 = sumH; dirTempHV = 1;
  }
  else
  {
    hv1 = sumH; hv0 = sumV; dirTempHV = 3;
  }
  if( sumD0 > sumD1 )
  {
    d1 = sumD0; d0 = sumD1; dirTempD = 0;
  }
  else
  {
    d1 = sumD1; d0 = sumD0; dirTempD = 2;
  }
  int hvd1, hvd0, mainDirection, secondaryDirection;
  // 32-bit unsigned products that wrap around, as vvdec's do
  if( uint( d1 ) * uint( hv0 ) > uint( hv1 ) * uint( d0 ) )
  {
    hvd1 = d1; hvd0 = d0; mainDirection = dirTempD; secondaryDirection = dirTempHV;
  }
  else
  {
    hvd1 = hv1; hvd0 = hv0; mainDirection = dirTempHV; secondaryDirection = dirTempD;
  }
  int directionStrength = 0;
  if( hvd1 > 2 * hvd0 )
  {
    directionStrength = 1;
  }
  if( hvd1 * 2 > 9 * hvd0 )
  {
    directionStrength = 2;
  }
  if( directionStrength != 0 )
  {
    classIdx += ( ( ( mainDirection & 1 ) << 1 ) + directionStrength ) * 5;
  }
  AlfClass cls;
  cls.classIdx     = classIdx;
  cls.transposeIdx = kTransposeTable[mainDirection * 2 + ( secondaryDirection >> 1 )];
  return cls;
}

// The 12 coefficients and clipping values of the class's filter in the CTU's filter set
void loadLumaFilter( int set, AlfClass cls, out int c[12], out int k[12] )
{
  const int base = set * LUMA_SET_SIZE + cls.transposeIdx * NUM_CLASSES * LUMA_COEFF + cls.classIdx * LUMA_COEFF;
  for( int i = 0; i < 12; i++ )
  {
    c[i] = int( pc.coeff[base + i] );
    k[i] = int( pc.clip[base + i] );
  }
}

// The rows the filter reads around row y: r1, r3, r5 below and r2, r4, r6 above. Next to the virtual boundary (vbPos
// in each CTU of vbH rows) the rows past it are replaced by the nearest ones before it, and the filter's sum is scaled
// down on the two rows next to it (nearVb).
struct AlfRows
{
  int  r1, r2, r3, r4, r5, r6;
  bool nearVb;
};

AlfRows alfRowsOf( int y, int vbH, int vbPos )
{
  const int yVb = y & ( vbH - 1 );
  AlfRows   r;
  r.r1 = y + 1;
  r.r2 = y - 1;
  r.r3 = y + 2;
  r.r4 = y - 2;
  r.r5 = y + 3;
  r.r6 = y - 3;
  if( yVb < vbPos && yVb >= vbPos - 4 )
  {
    r.r1 = yVb == vbPos - 1 ? y : r.r1;
    r.r3 = yVb >= vbPos - 2 ? r.r1 : r.r3;
    r.r5 = yVb >= vbPos - 3 ? r.r3 : r.r5;
    r.r2 = yVb == vbPos - 1 ? y : r.r2;
    r.r4 = yVb >= vbPos - 2 ? r.r2 : r.r4;
    r.r6 = yVb >= vbPos - 3 ? r.r4 : r.r6;
  }
  else if( yVb >= vbPos && yVb <= vbPos + 3 )
  {
    r.r2 = yVb == vbPos ? y : r.r2;
    r.r4 = yVb <= vbPos + 1 ? r.r2 : r.r4;
    r.r6 = yVb <= vbPos + 2 ? r.r4 : r.r6;
    r.r1 = yVb == vbPos ? y : r.r1;
    r.r3 = yVb <= vbPos + 1 ? r.r1 : r.r3;
    r.r5 = yVb <= vbPos + 2 ? r.r3 : r.r5;
  }
  r.nearVb = yVb == vbPos - 1 || yVb == vbPos;
  return r;
}
#endif

#if CHROMA == 0 && BOUNDS

// A sample of the luma source through the block's region (fetch)
#define S( X, Y ) fetch( pc.src0, pc.srcStride, X, Y, rg, pc.width, pc.height )

// The 4x4 block's Laplacians summed over its window: the 4x4 cells of 2x2 samples from ( bx - 2, by - 2 ), except the
// rows that the virtual boundary hides (where a cell's row past it reads its own row instead), each sample read
// through the block's region
AlfClass classifyBlockBounded( int bx, int by, Region rg, int vbH, int vbPos )
{
  const int byVb = by & ( vbH - 1 );
  int sumV = 0, sumH = 0, sumD0 = 0, sumD1 = 0;
  const int rFirst = byVb == vbPos ? 1 : 0;
  const int rLast  = byVb == vbPos - 4 ? 2 : 3;
  for( int r = rFirst; r <= rLast; r++ )
  {
    const int y0  = by - 2 + 2 * r;
    int       ym1 = y0 - 1;
    int       yp2 = y0 + 2;
    if( y0 > 0 && ( y0 & ( vbH - 1 ) ) == vbPos - 2 )
    {
      yp2 = y0 + 1;
    }
    else if( y0 > 0 && ( y0 & ( vbH - 1 ) ) == vbPos )
    {
      ym1 = y0;
    }
    for( int c = 0; c < 4; c++ )
    {
      const int x0 = bx - 2 + 2 * c;
      const int a  = S( x0, y0 ) << 1;
      const int b  = S( x0 + 1, y0 + 1 ) << 1;
      sumV  += abs( a - S( x0, ym1 ) - S( x0, y0 + 1 ) ) + abs( b - S( x0 + 1, y0 ) - S( x0 + 1, yp2 ) );
      sumH  += abs( a - S( x0 + 1, y0 ) - S( x0 - 1, y0 ) ) + abs( b - S( x0 + 2, y0 + 1 ) - S( x0, y0 + 1 ) );
      sumD0 += abs( a - S( x0 - 1, ym1 ) - S( x0 + 1, y0 + 1 ) ) + abs( b - S( x0, y0 ) - S( x0 + 2, yp2 ) );
      sumD1 += abs( a - S( x0 - 1, y0 + 1 ) - S( x0 + 1, ym1 ) ) + abs( b - S( x0, yp2 ) - S( x0 + 2, y0 ) );
    }
  }
  return alfClassOf( sumV, sumH, sumD0, sumD1, byVb, vbPos );
}

// Filters the 4x4 block at (bx, by) with its class's filter, each sample read through the block's region
void filterBlockBounded( int bx, int by, Region rg, int set, AlfClass cls, int vbH, int vbPos )
{
  int c[12], k[12];
  loadLumaFilter( set, cls, c, k );
  const int maxVal = ( 1 << pc.bitDepth ) - 1;
  for( int dy = 0; dy < 4; dy++ )
  {
    const int     y  = by + dy;
    const AlfRows rs = alfRowsOf( y, vbH, vbPos );
    for( int dx = 0; dx < 4; dx++ )
    {
      const int x   = bx + dx;
      const int cur = S( x, y );
      int       sum = 0;
      sum += c[0]  * clipAlf( k[0],  cur, S( x,     rs.r5 ), S( x,     rs.r6 ) );
      sum += c[1]  * clipAlf( k[1],  cur, S( x + 1, rs.r3 ), S( x - 1, rs.r4 ) );
      sum += c[2]  * clipAlf( k[2],  cur, S( x,     rs.r3 ), S( x,     rs.r4 ) );
      sum += c[3]  * clipAlf( k[3],  cur, S( x - 1, rs.r3 ), S( x + 1, rs.r4 ) );
      sum += c[4]  * clipAlf( k[4],  cur, S( x + 2, rs.r1 ), S( x - 2, rs.r2 ) );
      sum += c[5]  * clipAlf( k[5],  cur, S( x + 1, rs.r1 ), S( x - 1, rs.r2 ) );
      sum += c[6]  * clipAlf( k[6],  cur, S( x,     rs.r1 ), S( x,     rs.r2 ) );
      sum += c[7]  * clipAlf( k[7],  cur, S( x - 1, rs.r1 ), S( x + 1, rs.r2 ) );
      sum += c[8]  * clipAlf( k[8],  cur, S( x - 2, rs.r1 ), S( x + 2, rs.r2 ) );
      sum += c[9]  * clipAlf( k[9],  cur, S( x + 3, y ),     S( x - 3, y ) );
      sum += c[10] * clipAlf( k[10], cur, S( x + 2, y ),     S( x - 2, y ) );
      sum += c[11] * clipAlf( k[11], cur, S( x + 1, y ),     S( x - 1, y ) );
      sum = rs.nearVb ? ( sum + ( 1 << 9 ) ) >> 10 : ( sum + 64 ) >> 7;
      storePel( pc.dst0, pc.dstStride, x, y, clip3( 0, maxVal, sum + cur ) );
    }
  }
}
#undef S

// One thread per 4x4 luma block: classifies the block from the Laplacians of the 2x2 cells around it, then filters
// it (or copies it, where the CTU has ALF off), reading every sample through the block's region (fetch).
[shader( "compute" )]
[numthreads( 8, 8, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  const int bx = int( id.x ) * 4;
  const int by = int( id.y ) * 4;
  if( bx >= pc.width || by >= pc.height )
  {
    return;
  }
  const int ctu = ( by >> pc.ctuLog2 ) * pc.widthInCtus + ( bx >> pc.ctuLog2 );
  const int set = int( pc.ctus[ctu * 8 + CTU_LUMA_SET] );
  if( set == 255 )
  {
    for( int dy = 0; dy < 4; dy++ )
    {
      for( int dx = 0; dx < 4; dx++ )
      {
        storePel( pc.dst0, pc.dstStride, bx + dx, by + dy, loadPel( pc.src0, pc.srcStride, bx + dx, by + dy ) );
      }
    }
    return;
  }
  const Region rg    = regionOf( bx, by );
  const int    vbH   = 1 << pc.ctuLog2;
  const int    vbPos = vbH - 4;
  filterBlockBounded( bx, by, rg, set, classifyBlockBounded( bx, by, rg, vbH, vbPos ), vbH, vbPos );
}

#elif CHROMA == 0

// The workgroup's samples in shared memory: 32 x 32 from (s_x0, s_y0) plus 3
// rows above and below and 4 samples left and right, replicated beyond the
// picture's edges (vvdec's border extension). They are stored in groups of 4
// samples (8 bytes): what a 4x4 block's classification and filter read of a
// row arrives as 3 groups, and the samples are picked out of them in
// registers. Don't read the samples one at a time: a block then reads shared
// memory some 600 times and waits for most of the reads (the groups cut
// VideoQ 4K's ALF luma from 1.55 to 1.33 ms per picture with ALF).
#define TILE 32
#define MARGIN 3                         // rows above and below
#define GROUPS ( TILE / 4 + 2 )          // 4-sample groups per row: the tile's, plus one on each side
#define TILE_H ( TILE + 2 * MARGIN )
groupshared uint2 s_tile[TILE_H * GROUPS];
static int        s_x0, s_y0;
// The classification's Laplacians, computed once per cell of 2x2 samples: a
// block's window has 4x4 cells, and each cell lies in the windows of up to 4
// blocks. Entry (cx, cy) holds V | H << 16 and D0 | D1 << 16 of the cell at
// ( s_x0 - 2 + 2 * cx, s_y0 - 2 + 2 * cy ).
#define CELLS 18
groupshared uint2 s_cls[CELLS * CELLS];

// The samples ( bx - 4 .. bx + 7, y ) around the 4x4 block at bx: the tile's
// group G, which holds the block, and the groups before and after it.
struct Row
{
  uint2 g0, g1, g2;

  // Sample j; j is a constant, so the picks fold into register shifts.
  int at( int j )
  {
    const uint2 g = j < 4 ? g0 : ( j < 8 ? g1 : g2 );
    const uint  w = ( j & 2 ) == 0 ? g.x : g.y;
    return int( ( j & 1 ) == 0 ? ( w & 0xffffu ) : ( w >> 16 ) );
  }
};

// The samples ( s_x0 - 4 + 4 * m .. s_x0 + 3 + 4 * m, y ): the tile's groups m and m + 1.
struct Row2
{
  uint2 g0, g1;

  int at( int j )
  {
    const uint2 g = j < 4 ? g0 : g1;
    const uint  w = ( j & 2 ) == 0 ? g.x : g.y;
    return int( ( j & 1 ) == 0 ? ( w & 0xffffu ) : ( w >> 16 ) );
  }
};

Row2 tileRow2( int y, int m )
{
  const int at = ( y - s_y0 + MARGIN ) * GROUPS + m;
  Row2      r;
  r.g0 = s_tile[at];
  r.g1 = s_tile[at + 1];
  return r;
}

Row tileRow( int y, int G )
{
  const int at = ( y - s_y0 + MARGIN ) * GROUPS + G;
  Row       r;
  r.g0 = s_tile[at - 1];
  r.g1 = s_tile[at];
  r.g2 = s_tile[at + 1];
  return r;
}

// Loads the workgroup's tile (s_x0, s_y0) with its margins into s_tile, in 4-sample groups, by the workgroup's 64
// threads; the samples past the picture's edges are the nearest ones inside. Rows start 8-byte aligned
// (GpuRecon::recordAlf), and so do the groups.
void loadTile( int tid )
{
  for( int s = tid; s < TILE_H * GROUPS; s += 64 )
  {
    const int r = s / GROUPS;
    const int x = s_x0 - 4 + 4 * ( s - r * GROUPS );
    const int y = clamp( s_y0 - MARGIN + r, 0, pc.height - 1 );
    uint2     v;
    if( x >= 0 && x + 3 < pc.width )
    {
      v = *(uint2*)( pc.src0 + y * pc.srcStride + x );
    }
    else
    {
      v.x = ( uint( loadClamped( pc.src0, pc.srcStride, x, y, pc.width, pc.height ) ) & 0xffffu )
            | uint( loadClamped( pc.src0, pc.srcStride, x + 1, y, pc.width, pc.height ) ) << 16;
      v.y = ( uint( loadClamped( pc.src0, pc.srcStride, x + 2, y, pc.width, pc.height ) ) & 0xffffu )
            | uint( loadClamped( pc.src0, pc.srcStride, x + 3, y, pc.width, pc.height ) ) << 16;
    }
    s_tile[s] = v;
  }
}

// Computes the cells' Laplacians into s_cls (see above), two neighbouring cells per iteration (their samples lie in 2
// groups of each tile row), by the workgroup's 64 threads. The rows next to the virtual boundary (vbPos in each CTU of
// vbH rows) read their own row instead of the one past the boundary.
void cellLaplacians( int tid, int vbH, int vbPos )
{
  for( int t = tid; t < CELLS * CELLS / 2; t += 64 )
  {
    const int cy  = t / ( CELLS / 2 );
    const int m   = t - cy * ( CELLS / 2 );   // this iteration's cells: 2 * m and 2 * m + 1 of row cy
    const int y0  = s_y0 - 2 + 2 * cy;
    int       ym1 = y0 - 1;
    int       yp2 = y0 + 2;
    if( y0 > 0 && ( y0 & ( vbH - 1 ) ) == vbPos - 2 )
    {
      yp2 = y0 + 1;
    }
    else if( y0 > 0 && ( y0 & ( vbH - 1 ) ) == vbPos )
    {
      ym1 = y0;
    }
    const Row2 am = tileRow2( ym1, m );
    const Row2 a0 = tileRow2( y0, m );
    const Row2 a1 = tileRow2( y0 + 1, m );
    const Row2 ap = tileRow2( yp2, m );
    [unroll]
    for( int e = 0; e < 2; e++ )
    {
      const int j0 = 2 + 2 * e;   // x0 = s_x0 - 2 + 2 * ( 2 * m + e )
      const int a  = a0.at( j0 ) << 1;       // ( x0, y0 )
      const int b  = a1.at( j0 + 1 ) << 1;   // ( x0 + 1, y0 + 1 )
      const int v  = abs( a - am.at( j0 ) - a1.at( j0 ) ) + abs( b - a0.at( j0 + 1 ) - ap.at( j0 + 1 ) );
      const int h  = abs( a - a0.at( j0 + 1 ) - a0.at( j0 - 1 ) ) + abs( b - a1.at( j0 + 2 ) - a1.at( j0 ) );
      const int d0 = abs( a - am.at( j0 - 1 ) - a1.at( j0 + 1 ) ) + abs( b - a0.at( j0 ) - ap.at( j0 + 2 ) );
      const int d1 = abs( a - a1.at( j0 - 1 ) - am.at( j0 + 1 ) ) + abs( b - ap.at( j0 ) - a0.at( j0 + 2 ) );
      s_cls[cy * CELLS + 2 * m + e] = uint2( uint( v ) | uint( h ) << 16, uint( d0 ) | uint( d1 ) << 16 );
    }
  }
}

// Classifies the 4x4 block at (bx, by): sums the Laplacians over its window, the 4x4 cells from ( bx - 2, by - 2 )
// except the rows that the virtual boundary hides, then works out its activity and directions.
AlfClass classifyBlock( int bx, int by, int vbH, int vbPos )
{
  const int byVb = by & ( vbH - 1 );
  int sumV = 0, sumH = 0, sumD0 = 0, sumD1 = 0;
  const int rFirst = byVb == vbPos ? 1 : 0;
  const int rLast  = byVb == vbPos - 4 ? 2 : 3;
  const int cell0  = ( by - s_y0 ) / 2 * CELLS + ( bx - s_x0 ) / 2;
  [unroll]
  for( int r = 0; r < 4; r++ )
  {
    if( r >= rFirst && r <= rLast )
    {
      [unroll]
      for( int c = 0; c < 4; c++ )
      {
        const uint2 cell = s_cls[cell0 + r * CELLS + c];
        sumV  += int( cell.x & 0xffffu );
        sumH  += int( cell.x >> 16 );
        sumD0 += int( cell.y & 0xffffu );
        sumD1 += int( cell.y >> 16 );
      }
    }
  }

  return alfClassOf( sumV, sumH, sumD0, sumD1, byVb, vbPos );
}

// Filters the 4x4 block at (bx, by), the tile's group G, with the filter of its class in the CTU's filter set, one row
// at a time (alfRowsOf's rows around it): its 4 samples ( bx + i, y ) are sample j = 4 + i of each Row.
void filterBlock( int bx, int by, int G, int set, AlfClass cls, int vbH, int vbPos )
{
  int c[12], k[12];
  loadLumaFilter( set, cls, c, k );
  const int maxVal = ( 1 << pc.bitDepth ) - 1;
  [unroll]
  for( int dy = 0; dy < 4; dy++ )
  {
    const int     y      = by + dy;
    const AlfRows rs     = alfRowsOf( y, vbH, vbPos );
    const bool    nearVb = rs.nearVb;
    const Row     s0     = tileRow( y, G );
    const Row     s1     = tileRow( rs.r1, G );
    const Row     s2     = tileRow( rs.r2, G );
    const Row     s3     = tileRow( rs.r3, G );
    const Row     s4     = tileRow( rs.r4, G );
    const Row     s5     = tileRow( rs.r5, G );
    const Row     s6     = tileRow( rs.r6, G );
    int       out[4];
    [unroll]
    for( int i = 0; i < 4; i++ )
    {
      const int j   = 4 + i;
      const int cur = s0.at( j );
      int sum = 0;
      sum += c[0]  * clipAlf( k[0],  cur, s5.at( j ),     s6.at( j ) );
      sum += c[1]  * clipAlf( k[1],  cur, s3.at( j + 1 ), s4.at( j - 1 ) );
      sum += c[2]  * clipAlf( k[2],  cur, s3.at( j ),     s4.at( j ) );
      sum += c[3]  * clipAlf( k[3],  cur, s3.at( j - 1 ), s4.at( j + 1 ) );
      sum += c[4]  * clipAlf( k[4],  cur, s1.at( j + 2 ), s2.at( j - 2 ) );
      sum += c[5]  * clipAlf( k[5],  cur, s1.at( j + 1 ), s2.at( j - 1 ) );
      sum += c[6]  * clipAlf( k[6],  cur, s1.at( j ),     s2.at( j ) );
      sum += c[7]  * clipAlf( k[7],  cur, s1.at( j - 1 ), s2.at( j + 1 ) );
      sum += c[8]  * clipAlf( k[8],  cur, s1.at( j - 2 ), s2.at( j + 2 ) );
      sum += c[9]  * clipAlf( k[9],  cur, s0.at( j + 3 ), s0.at( j - 3 ) );
      sum += c[10] * clipAlf( k[10], cur, s0.at( j + 2 ), s0.at( j - 2 ) );
      sum += c[11] * clipAlf( k[11], cur, s0.at( j + 1 ), s0.at( j - 1 ) );
      sum    = nearVb ? ( sum + ( 1 << 9 ) ) >> 10 : ( sum + 64 ) >> 7;
      out[i] = clip3( 0, maxVal, sum + cur );
    }
    *(uint2*)( pc.dst0 + y * pc.dstStride + bx ) = uint2( uint( out[0] ) | uint( out[1] ) << 16, uint( out[2] ) | uint( out[3] ) << 16 );
  }
}

// One workgroup per tile of 32 x 32 luma samples, one thread per 4x4 block: the tile and its cells' Laplacians in
// shared memory, then each block's class and filter (or a copy, where the CTU has ALF off).
[shader( "compute" )]
[numthreads( 8, 8, 1 )]
void main( uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  s_x0          = int( group.x ) * TILE;
  s_y0          = int( group.y ) * TILE;
  const int tid = int( local.y * 8 + local.x );
  loadTile( tid );
  GroupMemoryBarrierWithGroupSync();
  const int vbH   = 1 << pc.ctuLog2;
  const int vbPos = vbH - 4;
  cellLaplacians( tid, vbH, vbPos );
  GroupMemoryBarrierWithGroupSync();

  const int bx = int( id.x ) * 4;
  const int by = int( id.y ) * 4;
  if( bx >= pc.width || by >= pc.height )
  {
    return;
  }
  const int G = ( bx - s_x0 ) / 4 + 1;
  // Each tile lies within one CTU (CTUs are 32 luma samples or larger).
  const int ctu = ( by >> pc.ctuLog2 ) * pc.widthInCtus + ( bx >> pc.ctuLog2 );
  const int set = int( pc.ctus[ctu * 8 + CTU_LUMA_SET] );
  if( set == 255 )
  {
    // (ALF off: the block's samples as they are)
    [unroll]
    for( int dy = 0; dy < 4; dy++ )
    {
      *(uint2*)( pc.dst0 + ( by + dy ) * pc.dstStride + bx ) = s_tile[( by + dy - s_y0 + MARGIN ) * GROUPS + G];
    }
    return;
  }
  filterBlock( bx, by, G, set, classifyBlock( bx, by, vbH, vbPos ), vbH, vbPos );
}

#else   // CHROMA

#if BOUNDS
static Region s_region;   // BOUNDS: the luma region of the thread's chroma block
#endif

int lumaPx( int x, int y )
{
#if BOUNDS
  return fetch( pc.srcLuma, pc.lumaStride, x, y, s_region, pc.lumaWidth, pc.lumaHeight );
#else
  return loadClamped( pc.srcLuma, pc.lumaStride, x, y, pc.lumaWidth, pc.lumaHeight );
#endif
}

// Reads a sample of the chroma plane src (with BOUNDS, through the block's region).
int chromaPx( int16_t* src, int x, int y )
{
#if BOUNDS
  return fetch( src, pc.srcStride, x, y, scaled( s_region, pc.scaleX, pc.scaleY ), pc.width, pc.height );
#else
  return loadClamped( src, pc.srcStride, x, y, pc.width, pc.height );
#endif
}

// Applies ALF to one 4x4 block of one chroma plane, into out[].
void alfChroma( int16_t* src, int bx, int by, int aps, int alt, inout int out[16] )
{
  const int vbH   = ( 1 << pc.ctuLog2 ) >> pc.scaleY;
  const int vbPos = vbH - 2;
  const int base  = ( aps * 8 + alt ) * CHROMA_COEFF;
  int c[6], k[6];
  for( int i = 0; i < 6; i++ )
  {
    c[i] = int( pc.coeff[base + i] );
    k[i] = int( pc.clip[base + i] );
  }
  const int maxVal = ( 1 << pc.bitDepth ) - 1;
  for( int y = by; y < by + 4; y++ )
  {
    const int yVb = y & ( vbH - 1 );
    int r1 = y + 1, r2 = y - 1, r3 = y + 2, r4 = y - 2;
    if( yVb < vbPos && yVb >= vbPos - 2 )
    {
      r1 = yVb == vbPos - 1 ? y : r1;
      r3 = r1;
      r2 = yVb == vbPos - 1 ? y : r2;
      r4 = r2;
    }
    else if( yVb >= vbPos && yVb <= vbPos + 1 )
    {
      r2 = yVb == vbPos ? y : r2;
      r4 = r2;
      r1 = yVb == vbPos ? y : r1;
      r3 = r1;
    }
    const bool nearVb = yVb == vbPos - 1 || yVb == vbPos;
    for( int x = bx; x < bx + 4; x++ )
    {
      const int cur = chromaPx( src, x, y );
      int sum = 0;
      sum += c[0] * clipAlf( k[0], cur, chromaPx( src, x,     r3 ), chromaPx( src, x,     r4 ) );
      sum += c[1] * clipAlf( k[1], cur, chromaPx( src, x + 1, r1 ), chromaPx( src, x - 1, r2 ) );
      sum += c[2] * clipAlf( k[2], cur, chromaPx( src, x,     r1 ), chromaPx( src, x,     r2 ) );
      sum += c[3] * clipAlf( k[3], cur, chromaPx( src, x - 1, r1 ), chromaPx( src, x + 1, r2 ) );
      sum += c[4] * clipAlf( k[4], cur, chromaPx( src, x + 2, y ),  chromaPx( src, x - 2, y ) );
      sum += c[5] * clipAlf( k[5], cur, chromaPx( src, x + 1, y ),  chromaPx( src, x - 1, y ) );
      sum = nearVb ? ( sum + ( 1 << 9 ) ) >> 10 : ( sum + 64 ) >> 7;
      out[( y - by ) * 4 + ( x - bx )] = clip3( 0, maxVal, sum + cur );
    }
  }
}

// Adds the CC-ALF correction of one 4x4 chroma block to out[].
void ccAlf( int bx, int by, int filter, int component, inout int out[16] )
{
  const int base = ( ( ( filter >> 2 ) * 2 + component ) * 4 + ( filter & 3 ) ) * 8;
  int c[7];
  for( int i = 0; i < 7; i++ )
  {
    c[i] = int( pc.ccCoeff[base + i] );
  }
  const int vbH    = 1 << pc.ctuLog2;
  const int vbPos  = vbH - 4;
  const int maxVal = ( 1 << pc.bitDepth ) - 1;
  const int offset = ( 1 << pc.bitDepth ) >> 1;
  for( int y = by; y < by + 4; y++ )
  {
    const int ly  = y << pc.scaleY;
    const int pos = ly & ( vbH - 1 );
    if( pc.scaleY == 0 && ( pos == vbPos || pos == vbPos + 1 ) )
    {
      continue;
    }
    int o1 = 1, o2 = -1, o3 = 2;
    if( pos == vbPos - 2 || pos == vbPos + 1 )
    {
      o3 = o1;
    }
    else if( pos == vbPos - 1 || pos == vbPos )
    {
      o1 = 0; o2 = 0; o3 = 0;
    }
    for( int x = bx; x < bx + 4; x++ )
    {
      const int lx  = x << pc.scaleX;
      const int cur = lumaPx( lx, ly );
      int sum = 0;
      sum += c[0] * ( lumaPx( lx,     ly + o2 ) - cur );
      sum += c[1] * ( lumaPx( lx - 1, ly )      - cur );
      sum += c[2] * ( lumaPx( lx + 1, ly )      - cur );
      sum += c[3] * ( lumaPx( lx - 1, ly + o1 ) - cur );
      sum += c[4] * ( lumaPx( lx,     ly + o1 ) - cur );
      sum += c[5] * ( lumaPx( lx + 1, ly + o1 ) - cur );
      sum += c[6] * ( lumaPx( lx,     ly + o3 ) - cur );
      sum = ( sum + 64 ) >> 7;
      sum = clip3( 0, maxVal, sum + offset ) - offset;
      const int i = ( y - by ) * 4 + ( x - bx );
      out[i] = clip3( 0, maxVal, sum + out[i] );
    }
  }
}

void filterPlane( int16_t* src, int16_t* dst, int bx, int by, int aps, int alt, int cc, int component )
{
  int out[16];
  if( alt == 255 && pc.aligned != 0 )
  {
    // Without ALF on this component (the CTU has CC-ALF only), the block is copied before CC-ALF adds to it: its rows
    // are read whole, 8 bytes each.
    [unroll]
    for( int j = 0; j < 4; j++ )
    {
      const uint2 v  = *(uint2*)at( src, uint( ( by + j ) * pc.srcStride + bx ) );
      out[4 * j]     = int( v.x & 0xffffu );
      out[4 * j + 1] = int( v.x >> 16 );
      out[4 * j + 2] = int( v.y & 0xffffu );
      out[4 * j + 3] = int( v.y >> 16 );
    }
  }
  else if( alt == 255 )
  {
    for( int i = 0; i < 16; i++ )
    {
      out[i] = loadPel( src, pc.srcStride, bx + ( i & 3 ), by + ( i >> 2 ) );
    }
  }
  else
  {
    alfChroma( src, bx, by, aps, alt, out );
  }
  if( cc != 255 )
  {
    ccAlf( bx, by, cc, component, out );
  }
  if( pc.aligned != 0 )
  {
    [unroll]
    for( int j = 0; j < 4; j++ )
    {
      *(uint2*)at( dst, uint( ( by + j ) * pc.dstStride + bx ) ) =
        uint2( ( uint( out[4 * j] ) & 0xffffu ) | uint( out[4 * j + 1] ) << 16, ( uint( out[4 * j + 2] ) & 0xffffu ) | uint( out[4 * j + 3] ) << 16 );
    }
  }
  else
  {
    for( int i = 0; i < 16; i++ )
    {
      storePel( dst, pc.dstStride, bx + ( i & 3 ), by + ( i >> 2 ), out[i] );
    }
  }
}

// Copies the 4x4 block at (bx, by) from src to dst.
void copyBlock( int16_t* src, int16_t* dst, int bx, int by )
{
  if( pc.aligned != 0 )
  {
    [unroll]
    for( int j = 0; j < 4; j++ )
    {
      *(uint2*)at( dst, uint( ( by + j ) * pc.dstStride + bx ) ) = *(uint2*)at( src, uint( ( by + j ) * pc.srcStride + bx ) );
    }
    return;
  }
  for( int i = 0; i < 16; i++ )
  {
    storePel( dst, pc.dstStride, bx + ( i & 3 ), by + ( i >> 2 ), loadPel( src, pc.srcStride, bx + ( i & 3 ), by + ( i >> 2 ) ) );
  }
}

[shader( "compute" )]
[numthreads( 8, 8, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  const int bx = int( id.x ) * 4;
  const int by = int( id.y ) * 4;
  if( bx >= pc.width || by >= pc.height )
  {
    return;
  }
  const int ctu   = ( ( by << pc.scaleY ) >> pc.ctuLog2 ) * pc.widthInCtus + ( ( bx << pc.scaleX ) >> pc.ctuLog2 );
  const int cbAlt = int( pc.ctus[ctu * 8 + CTU_CB_ALT] ), cbCc = int( pc.ctus[ctu * 8 + CTU_CC_CB] );
  const int crAlt = int( pc.ctus[ctu * 8 + CTU_CR_ALT] ), crCc = int( pc.ctus[ctu * 8 + CTU_CC_CR] );
  const bool cb   = cbAlt != 255 || cbCc != 255;   // (whether the CTU's Cb is filtered)
  const bool cr   = crAlt != 255 || crCc != 255;
  if( pc.pass == 1 )
  {
    if( cb )
    {
      copyBlock( pc.src0, pc.dst0, bx, by );
    }
    if( cr )
    {
      copyBlock( pc.src1, pc.dst1, bx, by );
    }
    return;
  }
  const int aps = int( pc.ctus[ctu * 8 + CTU_CHROMA_APS] );
#if BOUNDS
  s_region = regionOf( bx << pc.scaleX, by << pc.scaleY );
#endif
  if( cb )
  {
    filterPlane( pc.src0, pc.dst0, bx, by, aps, cbAlt, cbCc, 0 );
  }
  if( cr )
  {
    filterPlane( pc.src1, pc.dst1, bx, by, aps, crAlt, crCc, 1 );
  }
}

#endif
