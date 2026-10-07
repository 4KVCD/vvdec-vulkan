// The adaptive loop filter (ALF) and the cross-component ALF of a picture, as
// vvdec's AdaptiveLoopFilter does it (deriveClassificationBlk, filterBlk,
// filterBlkCcAlf), for pictures that no slice, tile, subpicture or virtual
// boundary splits: there the filters see the whole picture, with the samples
// beyond its edges replicated (vvdec's border extension).
//
// CHROMA=0: a thread classifies one 4x4 luma block and filters it (7x7 diamond).
// CHROMA=1: a thread filters one 4x4 block of Cb and of Cr (5x5 diamond), then
//           adds the cross-component filter of the luma below them.
//
// Per CTU, 8 bytes: luma filter set (255: off), chroma APS, Cb alternative
// (255: off), Cr alternative (255: off), Cb and Cr CC-ALF filter as
// APS * 4 + filter (255: off), 2 unused.

#include "common.slang"

#define CTU_LUMA_SET 0
#define CTU_CHROMA_APS 1
#define CTU_CB_ALT 2
#define CTU_CR_ALT 3
#define CTU_CC_CB 4
#define CTU_CC_CR 5

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
  int16_t* srcLuma;   // chroma: the luma for CC-ALF
  uint8_t* ctus;
  int16_t* coeff;     // luma: [set][transpose][class][13]; chroma: [aps][alt][7]
  int16_t* clip;      // as coeff
  int16_t* ccCoeff;   // [aps][component][filter][8]
  int      srcStride;
  int      dstStride;
  int      lumaStride;
  int      width;     // of the plane filtered
  int      height;
  int      lumaWidth;
  int      lumaHeight;
  int      ctuLog2;
  int      widthInCtus;
  int      bitDepth;
  int      scaleX;    // chroma subsampling
  int      scaleY;
};

[[vk::push_constant]] Params pc;

int clipAlf( int c, int cur, int a, int b )
{
  return clip3( -c, c, a - cur ) + clip3( -c, c, b - cur );
}

#if CHROMA == 0

int px( int x, int y )
{
  return loadClamped( pc.src0, pc.srcStride, x, y, pc.width, pc.height );
}

static const int kTh[16] = { 0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4 };
static const int kTransposeTable[8] = { 0, 1, 0, 2, 2, 3, 1, 3 };

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
    for( int y = by; y < by + 4; y++ )
    {
      for( int x = bx; x < bx + 4; x++ )
      {
        storePel( pc.dst0, pc.dstStride, x, y, loadPel( pc.src0, pc.srcStride, x, y ) );
      }
    }
    return;
  }

  const int vbH   = 1 << pc.ctuLog2;
  const int vbPos = vbH - 4;
  const int byVb  = by & ( vbH - 1 );

  // classification: Laplacians at every other sample of the 8x8 window around the block
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
      const int a  = px( x0, y0 ) << 1;       // y0 (src1)
      const int b  = px( x0 + 1, y0 + 1 ) << 1;  // y0 + 1 (src2)
      sumV  += abs( a - px( x0, ym1 ) - px( x0, y0 + 1 ) ) + abs( b - px( x0 + 1, y0 ) - px( x0 + 1, yp2 ) );
      sumH  += abs( a - px( x0 + 1, y0 ) - px( x0 - 1, y0 ) ) + abs( b - px( x0 + 2, y0 + 1 ) - px( x0, y0 + 1 ) );
      sumD0 += abs( a - px( x0 - 1, ym1 ) - px( x0 + 1, y0 + 1 ) ) + abs( b - px( x0, y0 ) - px( x0 + 2, yp2 ) );
      sumD1 += abs( a - px( x0 - 1, y0 + 1 ) - px( x0 + 1, ym1 ) ) + abs( b - px( x0, yp2 ) - px( x0 + 2, y0 ) );
    }
  }

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
  // 32-bit unsigned products, wrapping as vvdec's do
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
  const int transposeIdx = kTransposeTable[mainDirection * 2 + ( secondaryDirection >> 1 )];

  const int base = set * LUMA_SET_SIZE + transposeIdx * NUM_CLASSES * LUMA_COEFF + classIdx * LUMA_COEFF;
  int c[12], k[12];
  for( int i = 0; i < 12; i++ )
  {
    c[i] = int( pc.coeff[base + i] );
    k[i] = int( pc.clip[base + i] );
  }

  const int maxVal = ( 1 << pc.bitDepth ) - 1;
  for( int y = by; y < by + 4; y++ )
  {
    const int yVb = y & ( vbH - 1 );
    int r1 = y + 1, r2 = y - 1, r3 = y + 2, r4 = y - 2, r5 = y + 3, r6 = y - 3;
    if( yVb < vbPos && yVb >= vbPos - 4 )
    {
      r1 = yVb == vbPos - 1 ? y : r1;
      r3 = yVb >= vbPos - 2 ? r1 : r3;
      r5 = yVb >= vbPos - 3 ? r3 : r5;
      r2 = yVb == vbPos - 1 ? y : r2;
      r4 = yVb >= vbPos - 2 ? r2 : r4;
      r6 = yVb >= vbPos - 3 ? r4 : r6;
    }
    else if( yVb >= vbPos && yVb <= vbPos + 3 )
    {
      r2 = yVb == vbPos ? y : r2;
      r4 = yVb <= vbPos + 1 ? r2 : r4;
      r6 = yVb <= vbPos + 2 ? r4 : r6;
      r1 = yVb == vbPos ? y : r1;
      r3 = yVb <= vbPos + 1 ? r1 : r3;
      r5 = yVb <= vbPos + 2 ? r3 : r5;
    }
    const bool nearVb = yVb == vbPos - 1 || yVb == vbPos;

    for( int x = bx; x < bx + 4; x++ )
    {
      const int cur = px( x, y );
      int sum = 0;
      sum += c[0]  * clipAlf( k[0],  cur, px( x,     r5 ), px( x,     r6 ) );
      sum += c[1]  * clipAlf( k[1],  cur, px( x + 1, r3 ), px( x - 1, r4 ) );
      sum += c[2]  * clipAlf( k[2],  cur, px( x,     r3 ), px( x,     r4 ) );
      sum += c[3]  * clipAlf( k[3],  cur, px( x - 1, r3 ), px( x + 1, r4 ) );
      sum += c[4]  * clipAlf( k[4],  cur, px( x + 2, r1 ), px( x - 2, r2 ) );
      sum += c[5]  * clipAlf( k[5],  cur, px( x + 1, r1 ), px( x - 1, r2 ) );
      sum += c[6]  * clipAlf( k[6],  cur, px( x,     r1 ), px( x,     r2 ) );
      sum += c[7]  * clipAlf( k[7],  cur, px( x - 1, r1 ), px( x + 1, r2 ) );
      sum += c[8]  * clipAlf( k[8],  cur, px( x - 2, r1 ), px( x + 2, r2 ) );
      sum += c[9]  * clipAlf( k[9],  cur, px( x + 3, y ),  px( x - 3, y ) );
      sum += c[10] * clipAlf( k[10], cur, px( x + 2, y ),  px( x - 2, y ) );
      sum += c[11] * clipAlf( k[11], cur, px( x + 1, y ),  px( x - 1, y ) );
      sum = nearVb ? ( sum + ( 1 << 9 ) ) >> 10 : ( sum + 64 ) >> 7;
      storePel( pc.dst0, pc.dstStride, x, y, clip3( 0, maxVal, sum + cur ) );
    }
  }
}

#else   // CHROMA

int lumaPx( int x, int y )
{
  return loadClamped( pc.srcLuma, pc.lumaStride, x, y, pc.lumaWidth, pc.lumaHeight );
}

// ALF of one 4x4 block of one chroma plane into out[]
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
      const int cur = loadClamped( src, pc.srcStride, x, y, pc.width, pc.height );
      int sum = 0;
      sum += c[0] * clipAlf( k[0], cur, loadClamped( src, pc.srcStride, x,     r3, pc.width, pc.height ), loadClamped( src, pc.srcStride, x,     r4, pc.width, pc.height ) );
      sum += c[1] * clipAlf( k[1], cur, loadClamped( src, pc.srcStride, x + 1, r1, pc.width, pc.height ), loadClamped( src, pc.srcStride, x - 1, r2, pc.width, pc.height ) );
      sum += c[2] * clipAlf( k[2], cur, loadClamped( src, pc.srcStride, x,     r1, pc.width, pc.height ), loadClamped( src, pc.srcStride, x,     r2, pc.width, pc.height ) );
      sum += c[3] * clipAlf( k[3], cur, loadClamped( src, pc.srcStride, x - 1, r1, pc.width, pc.height ), loadClamped( src, pc.srcStride, x + 1, r2, pc.width, pc.height ) );
      sum += c[4] * clipAlf( k[4], cur, loadClamped( src, pc.srcStride, x + 2, y,  pc.width, pc.height ), loadClamped( src, pc.srcStride, x - 2, y,  pc.width, pc.height ) );
      sum += c[5] * clipAlf( k[5], cur, loadClamped( src, pc.srcStride, x + 1, y,  pc.width, pc.height ), loadClamped( src, pc.srcStride, x - 1, y,  pc.width, pc.height ) );
      sum = nearVb ? ( sum + ( 1 << 9 ) ) >> 10 : ( sum + 64 ) >> 7;
      out[( y - by ) * 4 + ( x - bx )] = clip3( 0, maxVal, sum + cur );
    }
  }
}

// CC-ALF of one 4x4 chroma block added to out[]
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
  if( alt == 255 )
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
  for( int i = 0; i < 16; i++ )
  {
    storePel( dst, pc.dstStride, bx + ( i & 3 ), by + ( i >> 2 ), out[i] );
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
  const int ctu = ( ( by << pc.scaleY ) >> pc.ctuLog2 ) * pc.widthInCtus + ( ( bx << pc.scaleX ) >> pc.ctuLog2 );
  const int aps = int( pc.ctus[ctu * 8 + CTU_CHROMA_APS] );
  filterPlane( pc.src0, pc.dst0, bx, by, aps, int( pc.ctus[ctu * 8 + CTU_CB_ALT] ), int( pc.ctus[ctu * 8 + CTU_CC_CB] ), 0 );
  filterPlane( pc.src1, pc.dst1, bx, by, aps, int( pc.ctus[ctu * 8 + CTU_CR_ALT] ), int( pc.ctus[ctu * 8 + CTU_CC_CR] ), 1 );
}

#endif
