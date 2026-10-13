// Dequantization and the inverse transforms, as vvdec's Quant::dequant,
// TrQuant::invTransformNxN and invTransformICT do them. A workgroup takes one
// transform block of one component, reads the levels that the parser wrote
// (the GPU's parse: the block's rectangle in the picture's levels; the CPU's:
// the picture's planes in host memory), and writes the residual into the
// picture on the GPU.
//
// A job is 10 uints (VulkanLib/GpuJobs.h, TrafoJob):
//   0: x | y << 16 (in the component's samples)
//   1: log2 width | log2 height << 4 | component << 8 | horizontal type << 12 | vertical type << 14 | flags << 16
//      | (joint Cb-Cr mode + 4) << 24                      (types: vvdec's TransType, 0 DCT-II, 1 DCT-VIII, 2 DST-VII)
//   2: last column | last row << 8 of the levels             (the parser wrote only [0, last column] x [0, last row])
//   3: scale   4: right shift (negative: a left shift)   5: the levels' maximum (they are clipped to it)
//   6: LFNST (0: none): index (1, 2) | set << 2 | transposed << 4 | 8x8 << 5 | 8 inputs << 6
//   7: explicit scaling list (0: none): offset of the block's matrix in scaling, plus 1
//   8: where the levels are in parsedLevels   9: the width of their rows (0: the levels are in the planes, levels0..2)
// flags: 1 transform skip, 2 BDPCM horizontal, 4 BDPCM vertical, 8 zeros (no residual)
//
// The threads dequantize the levels into shared memory, run the vertical
// transform into shared memory (rounded and clipped to 16 bits), then run the
// horizontal transform into the picture. With transform skip, each thread
// handles its own samples (with BDPCM, its own rows or columns). With LFNST,
// the top-left coefficients first go through the LFNST matrix
// (TrQuant::xInvLfnst).
//
// A block with only its DC coefficient and DCT-II both ways has one residual
// value, which the threads write 4 samples a store (as they write the zeros of
// blocks without residual). Blocks at least 4 x 4 with rows 8-byte aligned go
// in groups of 4 samples: a thread runs the vertical transform for 4 rows of
// a column, then the horizontal one for 4 samples of a row, which it writes
// with one store, the matrix's entries in its registers. Other blocks take a
// thread per sample.

#include "common.slang"

struct Params
{
  uint*     jobs;
  int16_t*  plane0;
  int16_t*  plane1;
  int16_t*  plane2;
  int16_t*  levels0;    // the CPU parse's levels, in host memory, laid out like the planes
  int16_t*  levels1;
  int16_t*  levels2;
  int16_t*  parsedLevels;   // the GPU parse's levels: each block's rectangle, where its job says (words 8 and 9)
  int16_t*  matrices;   // TrQuant's matrices, laid out as matrixOffset expects (GpuRecon.cpp fills them)
  int8_t*   lfnst;      // LFNST's matrices: vvdec's g_lfnst8x8, then g_lfnst4x4
  uint8_t*  scaling;    // the picture's dequantization matrices from explicit scaling lists (Quant's), 1 byte each
  int       stride0;
  int       stride12;
  int       count;      // number of jobs; negative: countAt holds it
  int       bitDepth;
  uint*     countAt;    // number of jobs and index of the first one, written by the GPU front end
};

[[vk::push_constant]] Params pc;

#define THREADS 64

#define FLAG_TS 1
#define FLAG_BDPCM_HOR 2
#define FLAG_BDPCM_VER 4
#define FLAG_ZERO 8

#define COEFF_MIN ( -32768 )   // maxLog2TrDynamicRange 15
#define COEFF_MAX 32767

// A block has at most 32 x 32 levels (64-point transforms keep only the first 32 coefficients). The coefficients and
// the vertical transform's output are 16-bit (dequantization, LFNST and the vertical transform clip them to 16 bits):
// read back as 16-bit values, they take the GPU's fast 24-bit multiply-add instead of a 32-bit multiply.
groupshared int16_t s_coef[32 * 32];   // [row][column]
groupshared int16_t s_mid[64 * 32];    // the vertical transform's output, [row][column]
groupshared int     s_lfnst[16];       // LFNST's input

// LFNST's input order: the top-left 4x4 coefficients in diagonal scan order, as positions x + 4 y
static const int kLfnstScan[16] = { 0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15 };

int16_t* planeOf( int comp )
{
  return comp == 0 ? pc.plane0 : ( comp == 1 ? pc.plane1 : pc.plane2 );
}

int16_t* levelsOf( int comp )
{
  return comp == 0 ? pc.levels0 : ( comp == 1 ? pc.levels1 : pc.levels2 );
}

int strideOf( int comp )
{
  return comp == 0 ? pc.stride0 : pc.stride12;
}

// Offset of the N x N matrix (N = 1 << log2) of a transform type in matrices,
// which holds DCT-II from 2 to 64 points, then DST-VII and DCT-VIII from 4 to 32.
int matrixOffset( int type, int log2 )
{
  const int first = type == 0 ? 1 : 2;
  int       at    = type == 0 ? 0 : ( type == 2 ? 5460 : 6820 );
  for( int m = first; m < log2; m++ )
  {
    at += 1 << ( 2 * m );
  }
  return at;
}

// The scale of coefficient (x, y) in a block of the given width: the QP's
// scale, times the matrix's factor when the block has an explicit scaling list
// (sl: job word 7), like DeQuantImpl's piDequantCoef[n] * scaleQP.
int coefScale( uint sl, int scale, int width, int x, int y )
{
  return sl != 0 ? int( pc.scaling[sl - 1 + uint( y * width + x )] ) * scale : scale;
}

// Dequantizes one level, as Quant's DeQuantImpl does.
int dequant( int level, int scale, int rightShift, int inputMaximum )
{
  if( level == 0 )
  {
    return 0;
  }
  const int clipped = clip3( -inputMaximum - 1, inputMaximum, level );
  const int value   = rightShift > 0 ? ( clipped * scale + ( 1 << ( rightShift - 1 ) ) ) >> rightShift : ( clipped * scale ) * ( 1 << -rightShift );
  return clip3( COEFF_MIN, COEFF_MAX, value );
}

// Joint Cb-Cr: derives the other chroma component's residual from this component's (TU::getICTMode).
int jointResidual( int ict, int v )
{
  return abs( ict ) == 2 ? ( ict > 0 ? v : -v ) : ( ( ict > 0 ? v : -v ) >> 1 );
}

// Writes v into every sample of the block (2^log2W x 2^log2H at p, rows stride apart). wide: the block's rows start
// 8-byte aligned and are 4 samples wide at least, so that a thread writes 4 samples a store.
void fillBlock( int16_t* p, int stride, int log2W, int log2H, bool wide, int tid, int v )
{
  if( wide )
  {
    const uint pair  = ( uint( v ) & 0xffffu ) * 0x10001u;
    const int  log2G = log2W - 2;   // the 4-sample groups of a row
    for( int s = tid; s < ( 1 << ( log2G + log2H ) ); s += THREADS )
    {
      *(uint2*)at( p, uint( ( s >> log2G ) * stride + ( s & ( ( 1 << log2G ) - 1 ) ) * 4 ) ) = uint2( pair, pair );
    }
  }
  else
  {
    for( int s = tid; s < ( 1 << ( log2W + log2H ) ); s += THREADS )
    {
      *at( p, uint( ( s >> log2W ) * stride + ( s & ( ( 1 << log2W ) - 1 ) ) ) ) = int16_t( v );
    }
  }
}

// Writes 4 residual samples r at (x, y) of the block with one store; with joint Cb-Cr, the other component's too.
void storeGroup( int16_t* block, int16_t* other, int stride, int x, int y, int4 r, int ict )
{
  *(uint2*)at( block, uint( y * stride + x ) ) = packSamples( r );
  if( ict != 0 )
  {
    *(uint2*)at( other, uint( y * stride + x ) ) =
      packSamples( int4( jointResidual( ict, r.x ), jointResidual( ict, r.y ), jointResidual( ict, r.z ), jointResidual( ict, r.w ) ) );
  }
}

// The horizontal transform of a block whose vertical transform's output has NX columns (4 or 8), for 4-sample
// groups of its rows: the thread's group g in rows y0, y0 + rowStep, ... Each thread keeps its group's NX x 4
// matrix entries in registers and writes its 4 samples with one store (with joint Cb-Cr, the other component's too).
void horizontalRows<let NX : int>( int16_t* mHor, int16_t* block, int16_t* other, int stride, int width, int height, int g, int y0,
                                   int rowStep, int shift2, int ict )
{
  int4 m[NX];
  [ForceUnroll]
  for( int k = 0; k < NX; k++ )
  {
    const uint2 w = *(uint2*)at( mHor, uint( k * width + 4 * g ) );
    m[k]          = unpackSamples( w );
  }
  for( int y = y0; y < height; y += rowStep )
  {
    int4 sum = int4( 1 << ( shift2 - 1 ) );
    [ForceUnroll]
    for( int k = 0; k < NX; k++ )
    {
      sum += int( s_mid[y * 32 + k] ) * m[k];
    }
    storeGroup( block, other, stride, 4 * g, y, clamp( sum >> shift2, int4( COEFF_MIN ), int4( COEFF_MAX ) ), ict );
  }
}

// A job's block as the transforms see it (trafoBlock decodes the job's words)
struct TrafoBlock
{
  int      log2W, log2H, width, height;
  int      typeHor, typeVer;    // the transform types (vvdec's TransType)
  int      flags;               // FLAG_*
  int      ict;                 // joint Cb-Cr's mode (TU::getICTMode), 0 without it
  int      lastX, lastY;        // the last column and row the parser wrote levels in (the others are zero)
  int      scale, rightShift, inputMaximum;   // dequantization (Quant's DeQuantImpl)
  uint     sl;                  // the explicit scaling list's matrix offset plus 1 (0: none)
  uint     lfnst;               // LFNST's word (0: none)
  int16_t* block;               // the residual's first sample in the picture
  int16_t* other;               // joint Cb-Cr: the other chroma component's
  int      stride;              // the planes' row stride
  int16_t* levels;              // the levels: the rows of the GPU parse's rectangle, or the plane's
  int      levelStride;
  bool     wide;                // the residual's rows start 8-byte aligned and are 4 samples wide at least: the threads
                                // write 4 samples a store
};

TrafoBlock trafoBlock( uint* job )
{
  TrafoBlock t;
  const int x0   = int( job[0] & 0xffff );
  const int y0   = int( job[0] >> 16 );
  const int comp = int( ( job[1] >> 8 ) & 15 );
  t.log2W        = int( job[1] & 15 );
  t.log2H        = int( ( job[1] >> 4 ) & 15 );
  t.width        = 1 << t.log2W;
  t.height       = 1 << t.log2H;
  t.typeHor      = int( ( job[1] >> 12 ) & 3 );
  t.typeVer      = int( ( job[1] >> 14 ) & 3 );
  t.flags        = int( ( job[1] >> 16 ) & 255 );
  t.ict          = int( job[1] >> 24 ) - 4;
  t.lastX        = int( job[2] & 255 );
  t.lastY        = int( ( job[2] >> 8 ) & 255 );
  t.scale        = int( job[3] );
  t.rightShift   = int( job[4] );
  t.inputMaximum = int( job[5] );
  t.lfnst        = job[6];
  t.sl           = job[7];
  t.stride       = strideOf( comp );
  t.block        = planeOf( comp ) + y0 * t.stride + x0;
  t.other        = planeOf( comp == 1 ? 2 : 1 ) + y0 * t.stride + x0;
  const int rowWidth = int( job[9] );   // (0: the levels are in the planes)
  t.levels       = rowWidth != 0 ? pc.parsedLevels + job[8] : levelsOf( comp ) + y0 * t.stride + x0;
  t.levelStride  = rowWidth != 0 ? rowWidth : t.stride;
  t.wide         = t.width >= 4 && ( t.stride & 3 ) == 0 && ( uint( uint64_t( t.block ) ) & 7u ) == 0
            && ( t.ict == 0 || ( uint( uint64_t( t.other ) ) & 7u ) == 0 );
  return t;
}

// The dequantized level at (x, y) (zero past the last column or row the parser wrote)
int dequantAt( TrafoBlock t, int x, int y )
{
  const int level = x <= t.lastX && y <= t.lastY ? int( *at( t.levels, uint( y * t.levelStride + x ) ) ) : 0;
  return dequant( level, coefScale( t.sl, t.scale, t.width, x, y ), t.rightShift, t.inputMaximum );
}

// Writes a residual sample at (x, y), and with joint Cb-Cr the other component's
void storeResidual( TrafoBlock t, int x, int y, int r )
{
  *at( t.block, uint( y * t.stride + x ) ) = int16_t( r );
  if( t.ict != 0 )
  {
    *at( t.other, uint( y * t.stride + x ) ) = int16_t( jointResidual( t.ict, r ) );
  }
}

// Transform skip: the dequantized levels are the residual. BDPCM first accumulates the levels along its direction
// (invResDPCM). The block's levels go into shared memory (a transform-skip block is 32 x 32 at most), dequantized
// there, or, with BDPCM, accumulated and dequantized there by a thread per row or column; then the residual goes out 4
// samples a store where the block allows.
void transformSkip( int tid, TrafoBlock t )
{
  const bool bdpcm = ( t.flags & ( FLAG_BDPCM_HOR | FLAG_BDPCM_VER ) ) != 0;
  for( int s = tid; s < t.width * t.height; s += THREADS )
  {
    const int x     = s & ( t.width - 1 );
    const int y     = s >> t.log2W;
    const int level = x <= t.lastX && y <= t.lastY ? int( *at( t.levels, uint( y * t.levelStride + x ) ) ) : 0;
    s_coef[y * 32 + x] = int16_t( bdpcm ? level : dequant( level, t.scale, t.rightShift, t.inputMaximum ) );
  }
  GroupMemoryBarrierWithGroupSync();
  if( bdpcm )
  {
    const bool hor = ( t.flags & FLAG_BDPCM_HOR ) != 0;
    for( int line = tid; line < ( hor ? t.height : t.width ); line += THREADS )
    {
      int sum = 0;
      for( int k = 0; k < ( hor ? t.width : t.height ); k++ )
      {
        const int i     = hor ? line * 32 + k : k * 32 + line;
        const int level = int( s_coef[i] );
        sum             = k == 0 ? level : clip3( COEFF_MIN, COEFF_MAX, sum + level );
        s_coef[i]       = int16_t( dequant( sum, t.scale, t.rightShift, t.inputMaximum ) );
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  if( t.wide )
  {
    const int log2G = t.log2W - 2;   // the 4-sample groups of a row
    for( int s = tid; s < t.height << log2G; s += THREADS )
    {
      const int y = s >> log2G;
      const int x = ( s & ( ( 1 << log2G ) - 1 ) ) * 4;
      const int i = y * 32 + x;
      storeGroup( t.block, t.other, t.stride, x, y, int4( int( s_coef[i] ), int( s_coef[i + 1] ), int( s_coef[i + 2] ), int( s_coef[i + 3] ) ), t.ict );
    }
    return;
  }
  for( int s = tid; s < t.width * t.height; s += THREADS )
  {
    const int x = s & ( t.width - 1 );
    const int y = s >> t.log2W;
    storeResidual( t, x, y, int( s_coef[y * 32 + x] ) );
  }
}

// xIT's 1-D transforms (ISP sub-partitions one sample wide or high): a single rounding, with a shift one bit larger
// than the 2-D second shift
void oneDimensionalTransform( int tid, TrafoBlock t, int16_t* m, int shift2 )
{
  const bool ver   = t.width == 1;
  const int  n     = ver ? t.height : t.width;
  const int  count = ver ? t.lastY + 1 : t.lastX + 1;
  const int  shift = shift2 + 1;
  for( int s = tid; s < n; s += THREADS )
  {
    int sum = 0;
    for( int k = 0; k < count; k++ )
    {
      sum += dequant( int( t.levels[ver ? k * t.levelStride : k] ), coefScale( t.sl, t.scale, t.width, ver ? 0 : k, ver ? k : 0 ), t.rightShift,
                      t.inputMaximum )
             * int( m[k * n + s] );
    }
    storeResidual( t, ver ? 0 : s, ver ? s : 0, clip3( COEFF_MIN, COEFF_MAX, ( sum + ( 1 << ( shift - 1 ) ) ) >> shift ) );
  }
}

// xInvLfnst: the set's matrix times the top-left 4x4 coefficients (only 8 of them for 4x4 and 8x8 blocks) gives the
// top-left 4x4 for blocks 4 wide or high, else the top-left 8x8 without its bottom-right 4x4, row by row or
// (transposed) column by column; into s_coef, whose other coefficients (of the top-left 8x8 at most) it zeroes.
void lfnstCoefficients( int tid, TrafoBlock t )
{
  const int  index     = int( t.lfnst & 3 ) - 1;
  const int  set       = int( ( t.lfnst >> 2 ) & 3 );
  const bool transpose = ( t.lfnst & 16 ) != 0;
  const bool block8    = ( t.lfnst & 32 ) != 0;
  const int  inputs    = ( t.lfnst & 64 ) != 0 ? 8 : 16;
  if( tid < 16 )
  {
    s_lfnst[tid] = dequantAt( t, kLfnstScan[tid] & 3, kLfnstScan[tid] >> 2 );
  }
  const int nx = min( t.width, 8 );
  const int ny = min( t.height, 8 );
  for( int s = tid; s < nx * ny; s += THREADS )
  {
    const int k = s / nx;
    s_coef[k * 32 + s - k * nx] = int16_t( 0 );
  }
  GroupMemoryBarrierWithGroupSync();
  if( tid < ( block8 ? 48 : 16 ) )
  {
    int8_t* row = pc.lfnst + ( block8 ? ( ( set * 2 + index ) * 48 + tid ) * 16 : 6144 + ( ( set * 2 + index ) * 16 + tid ) * 16 );
    int     sum = 0;
    for( int i = 0; i < inputs; i++ )
    {
      sum += s_lfnst[i] * int( row[i] );
    }
    // the output's place: in the top-left 4x4, else row by row in the top-left 8x8's top half, then the bottom-left 4x4
    int x, y;
    if( !block8 )
    {
      x = transpose ? tid >> 2 : tid & 3;
      y = transpose ? tid & 3 : tid >> 2;
    }
    else if( tid < 32 )
    {
      x = transpose ? tid >> 3 : tid & 7;
      y = transpose ? tid & 7 : tid >> 3;
    }
    else
    {
      x = transpose ? 4 + ( ( tid - 32 ) >> 2 ) : ( tid - 32 ) & 3;
      y = transpose ? ( tid - 32 ) & 3 : 4 + ( ( tid - 32 ) >> 2 );
    }
    s_coef[y * 32 + x] = int16_t( clip3( COEFF_MIN, COEFF_MAX, ( sum + 64 ) >> 7 ) );
  }
}

// The vertical transform of a block in groups of 4 rows (a thread per column of nx and 4 rows), into s_mid
void verticalGroups( int tid, TrafoBlock t, int16_t* mVer, int nx, int ny )
{
  const int log2Nx = firstbithigh( uint( nx ) );
  for( int i = tid; i < ( t.height >> 2 ) << log2Nx; i += THREADS )
  {
    const int x   = i & ( nx - 1 );
    const int y   = ( i >> log2Nx ) * 4;
    int4      sum = int4( 64 );
    for( int k = 0; k < ny; k += 4 )
    {
      [ForceUnroll]
      for( int j = 0; j < 4; j++ )
      {
        const uint2 w = *(uint2*)at( mVer, uint( ( k + j ) * t.height + y ) );
        sum += int( s_coef[( k + j ) * 32 + x] ) * unpackSamples( w );
      }
    }
    const int4 mid            = clamp( sum >> 7, int4( COEFF_MIN ), int4( COEFF_MAX ) );
    s_mid[y * 32 + x]         = int16_t( mid.x );
    s_mid[( y + 1 ) * 32 + x] = int16_t( mid.y );
    s_mid[( y + 2 ) * 32 + x] = int16_t( mid.z );
    s_mid[( y + 3 ) * 32 + x] = int16_t( mid.w );
  }
}

// The horizontal transform from s_mid in 4-sample groups of rows (a thread per group), written 4 samples a store
void horizontalGroups( int tid, TrafoBlock t, int16_t* mHor, int nx, int shift2 )
{
  const int log2G   = t.log2W - 2;
  const int g       = tid & ( ( 1 << log2G ) - 1 );
  const int y0      = tid >> log2G;
  const int rowStep = THREADS >> log2G;
  if( nx == 4 )
  {
    horizontalRows<4>( mHor, t.block, t.other, t.stride, t.width, t.height, g, y0, rowStep, shift2, t.ict );
    return;
  }
  if( nx == 8 )
  {
    horizontalRows<8>( mHor, t.block, t.other, t.stride, t.width, t.height, g, y0, rowStep, shift2, t.ict );
    return;
  }
  for( int y = y0; y < t.height; y += rowStep )
  {
    int4 sum = int4( 1 << ( shift2 - 1 ) );
    for( int k = 0; k < nx; k++ )
    {
      const uint2 w = *(uint2*)at( mHor, uint( k * t.width + 4 * g ) );
      sum += int( s_mid[y * 32 + k] ) * unpackSamples( w );
    }
    storeGroup( t.block, t.other, t.stride, 4 * g, y, clamp( sum >> shift2, int4( COEFF_MIN ), int4( COEFF_MAX ) ), t.ict );
  }
}

// Both transforms a thread per sample (blocks 2 wide or high, or rows not 8-byte aligned)
void transformSamples( int tid, TrafoBlock t, int16_t* mVer, int16_t* mHor, int nx, int ny, int shift2 )
{
  for( int s = tid; s < nx * t.height; s += THREADS )
  {
    const int y   = s / nx;
    const int x   = s - y * nx;
    int       sum = 0;
    for( int k = 0; k < ny; k++ )
    {
      sum += int( s_coef[k * 32 + x] ) * int( mVer[k * t.height + y] );
    }
    s_mid[y * 32 + x] = int16_t( clip3( COEFF_MIN, COEFF_MAX, ( sum + 64 ) >> 7 ) );
  }
  GroupMemoryBarrierWithGroupSync();
  for( int s = tid; s < t.width * t.height; s += THREADS )
  {
    const int x   = s & ( t.width - 1 );
    const int y   = s >> t.log2W;
    int       sum = 0;
    for( int k = 0; k < nx; k++ )
    {
      sum += int( s_mid[y * 32 + k] ) * int( mHor[k * t.width + x] );
    }
    storeResidual( t, x, y, clip3( COEFF_MIN, COEFF_MAX, ( sum + ( 1 << ( shift2 - 1 ) ) ) >> shift2 ) );
  }
}

[shader( "compute" )]
[numthreads( THREADS, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int index = int( group.x ) + int( group.y ) * 32768;
  const int count = pc.count >= 0 ? pc.count : int( pc.countAt[0] );
  if( index >= count )
  {
    return;   // (the whole workgroup returns here)
  }
  const int        tid = int( local.x );
  const TrafoBlock t   = trafoBlock( pc.jobs + ( ( pc.count >= 0 ? 0 : int( pc.countAt[1] ) ) + index ) * 10 );
  if( ( t.flags & FLAG_ZERO ) != 0 )
  {
    fillBlock( t.block, t.stride, t.log2W, t.log2H, t.wide, tid, 0 );
    return;
  }
  if( ( t.flags & FLAG_TS ) != 0 )
  {
    transformSkip( tid, t );
    return;
  }

  // TrQuant::xIT: the vertical transform (rounded and clipped), then the horizontal one.
  int16_t*  mVer   = pc.matrices + matrixOffset( t.typeVer, t.log2H );
  int16_t*  mHor   = pc.matrices + matrixOffset( t.typeHor, t.log2W );
  const int shift2 = 6 + 15 - 1 - pc.bitDepth;
  if( t.width == 1 || t.height == 1 )
  {
    oneDimensionalTransform( tid, t, t.width == 1 ? mVer : mHor, shift2 );
    return;
  }
  if( t.lfnst == 0 && t.lastX == 0 && t.lastY == 0 && t.typeHor == 0 && t.typeVer == 0 )
  {
    // Only the DC coefficient, DCT-II both ways: DCT-II's first basis function is 64 at every point, so the residual
    // is one value, the DC coefficient through both transforms' rounding and clipping.
    const int dc  = dequantAt( t, 0, 0 );
    const int mid = clip3( COEFF_MIN, COEFF_MAX, ( dc * 64 + 64 ) >> 7 );
    const int r   = clip3( COEFF_MIN, COEFF_MAX, ( mid * 64 + ( 1 << ( shift2 - 1 ) ) ) >> shift2 );
    fillBlock( t.block, t.stride, t.log2W, t.log2H, t.wide, tid, r );
    if( t.ict != 0 )
    {
      fillBlock( t.other, t.stride, t.log2W, t.log2H, t.wide, tid, jointResidual( t.ict, r ) );
    }
    return;
  }

  // The coefficients into s_coef: nx columns and ny rows of them (the others are zero). Blocks 4 high too go in groups
  // of 4 samples (rows in the vertical transform, columns in the horizontal one): the columns rounded up to a power of
  // 2 (4 at least), the rows to a multiple of 4, the added ones zero.
  const bool groups = t.wide && t.height >= 4;
  int        nx     = t.lastX + 1;
  int        ny     = t.lastY + 1;
  if( groups )
  {
    nx = nx <= 4 ? 4 : ( nx <= 8 ? 8 : ( nx <= 16 ? 16 : 32 ) );
    ny = ( ny + 3 ) & ~3;
  }
  if( t.lfnst != 0 )
  {
    lfnstCoefficients( tid, t );
    nx = min( t.width, 8 );
    ny = min( t.height, 8 );
  }
  else
  {
    for( int s = tid; s < nx * ny; s += THREADS )
    {
      const int k = s / nx;
      const int x = s - k * nx;
      s_coef[k * 32 + x] = int16_t( dequantAt( t, x, k ) );
    }
  }
  GroupMemoryBarrierWithGroupSync();

  if( groups )
  {
    verticalGroups( tid, t, mVer, nx, ny );
    GroupMemoryBarrierWithGroupSync();
    horizontalGroups( tid, t, mHor, nx, shift2 );
    return;
  }
  transformSamples( tid, t, mVer, mHor, nx, ny, shift2 );
}
