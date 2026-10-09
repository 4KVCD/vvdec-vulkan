// Film grain synthesis of an output frame, as vvdec's FilmGrain and FilmGrainImpl
// (the VFGS reference model) add it: a workgroup takes 1024 samples of a row of
// one plane, reads them (and one more on each side) into shared memory first
// -- the samples' intensities before any grain, so the row may be written in
// place -- then each sample's grain, a thread four samples, in two steps:
// - the raw grain into shared memory: the frame is cut into blocks of 16x16
//   luma samples; each block takes a random sign and offset from its seed
//   into the 64x64 pattern (the luma's or the chroma's) that its sample's
//   intensity picks (pLUT); on the first two luma rows of a band of blocks
//   (but the frame's first), the block above's pattern is blended in;
// - the output: next to each boundary between two blocks of a row, a
//   sample's grain is [1 3 1] / 4 of the raw grain around it; the grain,
//   scaled by the intensity's factor (sLUT), is added and the sum clipped to
//   [0, 255] << ( bit depth - 8 ).

#include "common.slang"

struct Params
{
  int16_t* src;        // the plane: the picture's samples, at the frame's origin
  uint8_t* dst;        // the output: 2 bytes a sample (may be src) or 1
  int8_t*  pattern;    // FilmGrainImpl's patterns [2][9][64][64]
  uint*    luts;       // sLUT[3][256], then pLUT[3][256] (bytes)
  uint*    seeds;      // each block's seed (FilmGrain::prepareBlockSeeds), a row of blocks after the other
  int      srcStride;  // samples
  int      dstStride;  // bytes
  int      width;      // of the plane in the frame
  int      readWidth;  // the plane's samples there are from the frame's origin (the picture's, past the frame's width)
  int      lumaWidth;  // the frame's
  int      comp;
  int      subx;       // the plane's subsampling: 1 or 2
  int      suby;
  int      bs;         // bit depth - 8: 0 or 2
  int      scaleShift;
  int      dstBytes;
  int      present;    // the component has a model (else the samples as they are)
};

[[vk::push_constant]] Params pc;

groupshared int16_t s_row[1024 + 2];    // the segment's samples, with one on each side
groupshared int16_t s_grain[1024 + 2];  // their raw grain
groupshared uint    s_lut[128];         // the component's sLUT, then its pLUT (bytes, four a word)

int lutS( int i ) { return int( ( s_lut[i >> 2] >> ( 8 * ( i & 3 ) ) ) & 0xff ); }
int lutP( int i ) { return int( ( s_lut[64 + ( i >> 2 )] >> ( 8 * ( i & 3 ) ) ) & 0xff ); }

// get_offset_y / u / v: the block's sign and offset into the pattern from its seed
void offsets( uint val, out int s, out int ox, out int oy )
{
  if( pc.comp == 0 )
  {
    s  = ( ( val >> 31 ) & 1 ) != 0 ? -1 : 1;
    ox = int( ( ( val & 0x3ffu ) * 13 ) >> 10 ) * 4;
    oy = int( ( ( ( val >> 14 ) & 0x3ffu ) * 12 ) >> 10 ) * 4;
  }
  else if( pc.comp == 1 )
  {
    s             = ( ( val >> 2 ) & 1 ) != 0 ? -1 : 1;
    ox            = int( ( ( ( val >> 10 ) & 0x3ffu ) * 13 ) >> 10 ) * ( 4 / pc.subx );
    const uint bf = ( ( val >> 24 ) & 0xffu ) | ( ( val << 8 ) & 0x300u );
    oy            = int( ( bf * 12 ) >> 10 ) * ( 4 / pc.suby );
  }
  else
  {
    s  = ( ( val >> 15 ) & 1 ) != 0 ? -1 : 1;
    ox = int( ( ( ( val >> 20 ) & 0x3ffu ) * 13 ) >> 10 ) * ( 4 / pc.subx );
    oy = int( ( ( ( val >> 4 ) & 0x3ffu ) * 12 ) >> 10 ) * ( 4 / pc.suby );
  }
}

// a block's place in the patterns: its own (row j of its band), and the block above's (the overlap's)
struct Block
{
  int s, ox, oy;
  int sUp, oxUp, oyUp;
};

Block blockAt( int bxi, int band, int blocks, int j, bool overlap )
{
  Block b;
  offsets( pc.seeds[band * blocks + bxi], b.s, b.ox, b.oy );
  b.oy += j / pc.suby;
  b.sUp  = 0;
  b.oxUp = 0;
  b.oyUp = 0;
  if( overlap )
  {
    offsets( pc.seeds[( band - 1 ) * blocks + bxi], b.sUp, b.oxUp, b.oyUp );
    b.oyUp += ( 16 + j ) / pc.suby;
  }
  return b;
}

// make_grain_pattern: the grain of the sample at i in its block, of the intensity given, before the boundaries'
// smoothing (oc1, oc2: the overlap's weights, 0 without)
int grainOf( Block b, int i, int intensity, int oc1, int oc2 )
{
  const int pi   = lutP( intensity ) >> 4;
  const int base = ( pc.comp != 0 ? 9 : 0 ) + pi;
  int       p    = int( pc.pattern[( base * 64 + b.oy ) * 64 + b.ox + i] ) * b.s;
  if( oc1 != 0 )
  {
    p = ( p * oc1 + int( pc.pattern[( base * 64 + b.oyUp ) * 64 + b.oxUp + i] ) * oc2 * b.sUp + 16 ) >> 5;
  }
  return p;
}

[shader( "compute" )]
[numthreads( 256, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int tid    = int( local.x );
  const int seg0   = int( group.x ) * 1024;
  const int row    = int( group.y );
  const int nb     = 16 / pc.subx;   // a block's samples across
  const int blocks = ( pc.lumaWidth + 15 ) >> 4;
  const int end    = blocks * nb;    // the row up to its last block's end

  // the component's LUTs
  if( tid < 64 )
  {
    s_lut[tid] = pc.luts[( pc.comp * 256 ) / 4 + tid];
  }
  else if( tid < 128 )
  {
    s_lut[tid] = pc.luts[( 768 + pc.comp * 256 ) / 4 + tid - 64];
  }
  // the segment's samples (past the frame's width: the picture's, then its
  // last one -- the border extension), with the halo
  int16_t*  src = pc.src + row * pc.srcStride;
  const int x0  = seg0 + 4 * tid;
  for( int k = 0; k < 4; k++ )
  {
    if( x0 + k < end )
    {
      s_row[1 + 4 * tid + k] = src[min( x0 + k, pc.readWidth - 1 )];
    }
  }
  if( tid == 0 && seg0 > 0 )
  {
    s_row[0] = src[min( seg0 - 1, pc.readWidth - 1 )];
  }
  if( tid == 255 && seg0 + 1024 < end )
  {
    s_row[1025] = src[min( seg0 + 1024, pc.readWidth - 1 )];
  }
  GroupMemoryBarrierWithGroupSync();

  const int lumaRow = row * pc.suby;
  const int band    = lumaRow >> 4;
  const int j       = lumaRow & 15;
  int       oc1 = 0, oc2 = 0;
  if( lumaRow > 15 && j == 0 )
  {
    oc1 = pc.suby > 1 ? 20 : 12;
    oc2 = pc.suby > 1 ? 20 : 24;
  }
  else if( lumaRow > 15 && j == 1 )
  {
    oc1 = 24;
    oc2 = 12;
  }
  const bool overlap = oc1 != 0;
  const int  bxi     = x0 / nb;   // the thread's four samples share a block

  // the raw grain: the thread's four samples (their pattern loads together),
  // the halo's on the ends
  if( pc.present != 0 && x0 < end )
  {
    const Block b = blockAt( bxi, band, blocks, j, overlap );
    const int   i = x0 - bxi * nb;
    int         g[4];
    for( int k = 0; k < 4; k++ )
    {
      g[k] = grainOf( b, i + k, int( s_row[1 + 4 * tid + k] ) >> pc.bs, oc1, oc2 );
    }
    for( int k = 0; k < 4; k++ )
    {
      s_grain[1 + 4 * tid + k] = int16_t( g[k] );
    }
    if( tid == 0 && seg0 > 0 )
    {
      const int   xl = seg0 - 1;
      const Block l  = blockAt( xl / nb, band, blocks, j, overlap );
      s_grain[0]     = int16_t( grainOf( l, xl - ( xl / nb ) * nb, int( s_row[0] ) >> pc.bs, oc1, oc2 ) );
    }
    if( tid == 255 && seg0 + 1024 < end )
    {
      const int   xr = seg0 + 1024;
      const Block r  = blockAt( xr / nb, band, blocks, j, overlap );
      s_grain[1025]  = int16_t( grainOf( r, xr - ( xr / nb ) * nb, int( s_row[1025] ) >> pc.bs, oc1, oc2 ) );
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if( x0 >= pc.width )
  {
    return;
  }
  const int maxVal = 255 << pc.bs;
  uint8_t*  dst    = pc.dst + row * pc.dstStride;
  for( int k = 0; k < 4; k++ )
  {
    const int x = x0 + k;
    if( x >= pc.width )
    {
      break;
    }
    const int at  = 1 + 4 * tid + k;   // x's in s_row and s_grain
    const int v   = int( s_row[at] );
    int       out = v;
    if( pc.present != 0 )
    {
      const int i = x - bxi * nb;
      int       g = int( s_grain[at] );
      // scale_and_output's smoothing across the boundaries between blocks
      if( ( i == 0 && bxi > 0 ) || ( i == nb - 1 && bxi + 1 < blocks ) )
      {
        g = ( int( s_grain[at - 1] ) + 3 * g + int( s_grain[at + 1] ) + 2 ) >> 2;
      }
      const int scale = lutS( v >> pc.bs );
      out             = clip3( 0, maxVal, v + ( ( scale * g + ( 1 << ( pc.scaleShift - 1 ) ) ) >> pc.scaleShift ) );
    }
    if( pc.dstBytes == 2 )
    {
      *(int16_t*) ( dst + 2 * x ) = int16_t( out );
    }
    else
    {
      dst[x] = uint8_t( out );
    }
  }
}
