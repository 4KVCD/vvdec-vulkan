// Film grain synthesis for an output frame, as vvdec's FilmGrain and FilmGrainImpl
// (the VFGS reference model) add it. A workgroup takes 1024 samples of a row of
// one plane, four samples per thread. It first reads its samples (and one more
// on each side) into shared memory: these are the intensities before any grain,
// which lets the row be written in place. It then computes each sample's grain
// in two steps:
// - The raw grain, into shared memory. The frame is divided into blocks of 16x16
//   luma samples. From its seed, each block takes a random sign and an offset
//   into a 64x64 pattern (luma or chroma); each sample's intensity picks which
//   pattern (pLUT). On the first two luma rows of every band of blocks except
//   the frame's first band, the pattern of the block above is blended in.
// - The output. Next to each boundary between two blocks of a row, a sample's
//   grain becomes [1 3 1] / 4 of the raw grain around it. The grain, scaled by
//   the intensity's factor (sLUT), is added to the sample, and the sum is
//   clipped to [0, 255] << ( bit depth - 8 ).

#include "common.slang"

struct Params
{
  int16_t* src;        // the picture's plane, pointing at the frame's top-left sample
  uint8_t* dst;        // the output: 2 bytes per sample (may be src) or 1
  int8_t*  pattern;    // FilmGrainImpl's patterns [2][9][64][64]
  uint*    luts;       // sLUT[3][256], then pLUT[3][256], as bytes
  uint*    seeds;      // one seed per block (FilmGrain::prepareBlockSeeds), row of blocks after row of blocks
  int      srcStride;  // samples
  int      dstStride;  // bytes
  int      width;      // plane width within the frame, in samples
  int      readWidth;  // samples the plane holds from the frame's left edge to the picture's right edge
  int      lumaWidth;  // frame width in luma samples
  int      comp;
  int      subx;       // the plane's subsampling factors: 1 or 2
  int      suby;
  int      bs;         // bit depth - 8: 0 or 2
  int      scaleShift;
  int      dstBytes;
  int      present;    // nonzero when the component has a grain model; otherwise the samples pass unchanged
};

[[vk::push_constant]] Params pc;

// A workgroup of kThreads threads takes a segment of kSegment samples of a row, kPerThread samples a thread.
static const int kThreads   = 256;
static const int kPerThread = 4;
static const int kSegment   = kThreads * kPerThread;

// The grain's blocks are kBlockSide x kBlockSide luma samples; its patterns kPatternSide x kPatternSide grain values,
// kPatterns per kind (luma's, then chroma's).
static const int kBlockSide   = 16;
static const int kPatternSide = 64;
static const int kPatterns    = 9;

// A component's LUTs have kLutEntries byte entries, one per 8-bit intensity: pc.luts holds the three components'
// sLUTs, then their pLUTs. s_lut keeps the component's two, four entries a word.
static const int kLutEntries = 256;
static const int kLutWords   = kLutEntries / 4;

groupshared int16_t s_row[kSegment + 2];    // the segment's samples, plus one on each side
groupshared int16_t s_grain[kSegment + 2];  // the raw grain of the samples in s_row
groupshared uint    s_lut[2 * kLutWords];   // the component's sLUT, then its pLUT

int lutS( int i ) { return int( ( s_lut[i >> 2] >> ( 8 * ( i & 3 ) ) ) & 0xff ); }
int lutP( int i ) { return int( ( s_lut[kLutWords + ( i >> 2 )] >> ( 8 * ( i & 3 ) ) ) & 0xff ); }

// Derives a block's sign and offset into the pattern from the block's seed, as get_offset_y / u / v do.
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

// Where a block reads the patterns for row j of its band: its own sign and offsets, and those of the block above
// (for the overlap).
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
    b.oyUp += ( kBlockSide + j ) / pc.suby;
  }
  return b;
}

// Returns the raw grain of the sample at position i of its block for the given intensity, as make_grain_pattern
// computes it, before the smoothing across block boundaries. oc1 and oc2 are the overlap's weights (0 without one).
int grainOf( Block b, int i, int intensity, int oc1, int oc2 )
{
  const int pi   = lutP( intensity ) >> 4;
  const int base = ( pc.comp != 0 ? kPatterns : 0 ) + pi;
  int       p    = int( pc.pattern[( base * kPatternSide + b.oy ) * kPatternSide + b.ox + i] ) * b.s;
  if( oc1 != 0 )
  {
    p = ( p * oc1 + int( pc.pattern[( base * kPatternSide + b.oyUp ) * kPatternSide + b.oxUp + i] ) * oc2 * b.sUp + 16 ) >> 5;
  }
  return p;
}

// Loads the component's LUTs, and the segment's samples of the row (four a thread, up to end, the row's length rounded
// up to whole blocks) with one sample of halo on each side. Past the frame's width these are the picture's samples,
// and past the picture's width its last sample (the border extension).
void loadSegment( int tid, int seg0, int row, int end )
{
  if( tid < kLutWords )
  {
    s_lut[tid] = pc.luts[pc.comp * kLutWords + tid];   // sLUT
  }
  else if( tid < 2 * kLutWords )
  {
    s_lut[tid] = pc.luts[( 3 + pc.comp ) * kLutWords + tid - kLutWords];   // pLUT
  }
  int16_t*  src = pc.src + row * pc.srcStride;
  const int x0  = seg0 + kPerThread * tid;
  for( int k = 0; k < kPerThread; k++ )
  {
    if( x0 + k < end )
    {
      s_row[1 + kPerThread * tid + k] = src[min( x0 + k, pc.readWidth - 1 )];
    }
  }
  if( tid == 0 && seg0 > 0 )
  {
    s_row[0] = src[min( seg0 - 1, pc.readWidth - 1 )];
  }
  if( tid == kThreads - 1 && seg0 + kSegment < end )
  {
    s_row[kSegment + 1] = src[min( seg0 + kSegment, pc.readWidth - 1 )];
  }
}

// The weights that blend the grain of the band above into the first two luma rows of a band of blocks (0, 0 elsewhere)
int2 overlapWeights( int lumaRow )
{
  const int j = lumaRow & ( kBlockSide - 1 );
  if( lumaRow >= kBlockSide && j == 0 )
  {
    return pc.suby > 1 ? int2( 20, 20 ) : int2( 12, 24 );
  }
  if( lumaRow >= kBlockSide && j == 1 )
  {
    return int2( 24, 12 );
  }
  return int2( 0, 0 );
}

// The raw grain of the thread's four samples (their pattern loads are issued together) and of the halo samples at the
// segment's ends, into s_grain; bxi: the block of the thread's samples
void segmentGrain( int tid, int seg0, int x0, int bxi, int nb, int blocks, int end, int lumaRow, int2 oc )
{
  const int  band    = lumaRow / kBlockSide;
  const int  j       = lumaRow & ( kBlockSide - 1 );
  const bool overlap = oc.x != 0;
  if( pc.present == 0 || x0 >= end )
  {
    return;
  }
  const Block b = blockAt( bxi, band, blocks, j, overlap );
  const int   i = x0 - bxi * nb;
  int         g[kPerThread];
  for( int k = 0; k < kPerThread; k++ )
  {
    g[k] = grainOf( b, i + k, int( s_row[1 + kPerThread * tid + k] ) >> pc.bs, oc.x, oc.y );
  }
  for( int k = 0; k < kPerThread; k++ )
  {
    s_grain[1 + kPerThread * tid + k] = int16_t( g[k] );
  }
  if( tid == 0 && seg0 > 0 )
  {
    const int   xl = seg0 - 1;
    const Block l  = blockAt( xl / nb, band, blocks, j, overlap );
    s_grain[0]     = int16_t( grainOf( l, xl - ( xl / nb ) * nb, int( s_row[0] ) >> pc.bs, oc.x, oc.y ) );
  }
  if( tid == kThreads - 1 && seg0 + kSegment < end )
  {
    const int   xr       = seg0 + kSegment;
    const Block r        = blockAt( xr / nb, band, blocks, j, overlap );
    s_grain[kSegment + 1] = int16_t( grainOf( r, xr - ( xr / nb ) * nb, int( s_row[kSegment + 1] ) >> pc.bs, oc.x, oc.y ) );
  }
}

// Writes the thread's four samples of the row (those inside the frame): each plus its grain, smoothed across the
// boundaries between blocks and scaled as scale_and_output does it, at the output's sample size
void writeSegment( int tid, int x0, int bxi, int nb, int blocks, int row )
{
  const int maxVal = 255 << pc.bs;
  uint8_t*  dst    = pc.dst + row * pc.dstStride;
  for( int k = 0; k < kPerThread; k++ )
  {
    const int x = x0 + k;
    if( x >= pc.width )
    {
      break;
    }
    const int at  = 1 + kPerThread * tid + k;   // x's index in s_row and s_grain
    const int v   = int( s_row[at] );
    int       out = v;
    if( pc.present != 0 )
    {
      const int i = x - bxi * nb;
      int       g = int( s_grain[at] );
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

// One workgroup per segment of a row (group.x the segment, group.y the row): the samples, then their grain, then the
// output
[shader( "compute" )]
[numthreads( kThreads, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int tid    = int( local.x );
  const int seg0   = int( group.x ) * kSegment;
  const int row    = int( group.y );
  const int nb     = kBlockSide / pc.subx;   // block width in this plane's samples
  const int blocks = ( pc.lumaWidth + kBlockSide - 1 ) / kBlockSide;
  const int end    = blocks * nb;            // the row's length rounded up to whole blocks
  const int x0     = seg0 + kPerThread * tid;
  const int bxi    = x0 / nb;        // the thread's four samples all lie in this block
  loadSegment( tid, seg0, row, end );
  GroupMemoryBarrierWithGroupSync();
  const int lumaRow = row * pc.suby;
  segmentGrain( tid, seg0, x0, bxi, nb, blocks, end, lumaRow, overlapWeights( lumaRow ) );
  GroupMemoryBarrierWithGroupSync();
  if( x0 < pc.width )
  {
    writeSegment( tid, x0, bxi, nb, blocks, row );
  }
}
