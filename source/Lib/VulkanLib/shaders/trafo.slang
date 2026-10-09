// Dequantization and the inverse transforms, as vvdec's Quant::dequant,
// TrQuant::invTransformNxN and invTransformICT do them. A workgroup takes one
// transform block of one component, reads the levels that the parser wrote
// into a picture-sized buffer in host memory, and writes the residual into the
// picture on the GPU.
//
// A job is 8 uints (VulkanLib/GpuJobs.h, TrafoJob):
//   0: x | y << 16 (in the component's samples)
//   1: log2 width | log2 height << 4 | component << 8 | horizontal type << 12 | vertical type << 14 | flags << 16
//      | (joint Cb-Cr mode + 4) << 24                      (types: vvdec's TransType, 0 DCT-II, 1 DCT-VIII, 2 DST-VII)
//   2: last column | last row << 8 of the levels             (the parser wrote only [0, last column] x [0, last row])
//   3: scale   4: right shift (negative: a left shift)   5: the levels' maximum (they are clipped to it)
//   6: LFNST (0: none): index (1, 2) | set << 2 | transposed << 4 | 8x8 << 5 | 8 inputs << 6
//   7: explicit scaling list (0: none): offset of the block's matrix in scaling, plus 1
// flags: 1 transform skip, 2 BDPCM horizontal, 4 BDPCM vertical, 8 zeros (no residual)
//
// The threads dequantize the levels into shared memory, run the vertical
// transform into shared memory (rounded and clipped to 16 bits), then run the
// horizontal transform into the picture. With transform skip, each thread
// handles its own samples (with BDPCM, its own rows or columns). With LFNST,
// the top-left coefficients first go through the LFNST matrix
// (TrQuant::xInvLfnst).

#include "common.slang"

struct Params
{
  uint*     jobs;
  int16_t*  plane0;
  int16_t*  plane1;
  int16_t*  plane2;
  int16_t*  levels0;    // the levels, in host memory, laid out like the planes
  int16_t*  levels1;
  int16_t*  levels2;
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

// A block has at most 32 x 32 levels (64-point transforms keep only the first 32 coefficients).
groupshared int s_coef[32 * 32];   // [row][column]
groupshared int s_mid[64 * 32];    // the vertical transform's output, [row][column]
groupshared int s_lfnst[16];       // LFNST's input

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
  const int tid          = int( local.x );
  uint*     job          = pc.jobs + ( ( pc.count >= 0 ? 0 : int( pc.countAt[1] ) ) + index ) * 8;
  const int x0           = int( job[0] & 0xffff );
  const int y0           = int( job[0] >> 16 );
  const int log2W        = int( job[1] & 15 );
  const int log2H        = int( ( job[1] >> 4 ) & 15 );
  const int comp         = int( ( job[1] >> 8 ) & 15 );
  const int typeHor      = int( ( job[1] >> 12 ) & 3 );
  const int typeVer      = int( ( job[1] >> 14 ) & 3 );
  const int flags        = int( ( job[1] >> 16 ) & 255 );
  const int ict          = int( job[1] >> 24 ) - 4;
  const int lastX        = int( job[2] & 255 );
  const int lastY        = int( ( job[2] >> 8 ) & 255 );
  const int scale        = int( job[3] );
  const int rightShift   = int( job[4] );
  const int inputMaximum = int( job[5] );
  const uint sl          = job[7];
  const int width        = 1 << log2W;
  const int height       = 1 << log2H;
  const int stride       = strideOf( comp );
  int16_t*  block        = planeOf( comp ) + y0 * stride + x0;
  int16_t*  levels       = levelsOf( comp ) + y0 * stride + x0;
  int16_t*  other        = planeOf( comp == 1 ? 2 : 1 ) + y0 * stride + x0;   // joint Cb-Cr's other component

  if( ( flags & FLAG_ZERO ) != 0 )
  {
    for( int s = tid; s < width * height; s += THREADS )
    {
      block[( s >> log2W ) * stride + ( s & ( width - 1 ) )] = int16_t( 0 );
    }
    return;
  }

  if( ( flags & FLAG_TS ) != 0 )
  {
    // Transform skip: the dequantized levels are the residual. BDPCM first
    // accumulates the levels along its direction (invResDPCM), one thread per
    // row or column.
    if( ( flags & FLAG_BDPCM_HOR ) != 0 )
    {
      for( int y = tid; y < height; y += THREADS )
      {
        int sum = 0;
        for( int x = 0; x < width; x++ )
        {
          const int level = int( levels[y * stride + x] );
          sum             = x == 0 ? level : clip3( COEFF_MIN, COEFF_MAX, sum + level );
          const int r     = dequant( sum, scale, rightShift, inputMaximum );
          block[y * stride + x] = int16_t( r );
          if( ict != 0 )
          {
            other[y * stride + x] = int16_t( jointResidual( ict, int( int16_t( r ) ) ) );
          }
        }
      }
    }
    else if( ( flags & FLAG_BDPCM_VER ) != 0 )
    {
      for( int x = tid; x < width; x += THREADS )
      {
        int sum = 0;
        for( int y = 0; y < height; y++ )
        {
          const int level = int( levels[y * stride + x] );
          sum             = y == 0 ? level : clip3( COEFF_MIN, COEFF_MAX, sum + level );
          const int r     = dequant( sum, scale, rightShift, inputMaximum );
          block[y * stride + x] = int16_t( r );
          if( ict != 0 )
          {
            other[y * stride + x] = int16_t( jointResidual( ict, int( int16_t( r ) ) ) );
          }
        }
      }
    }
    else
    {
      for( int s = tid; s < width * height; s += THREADS )
      {
        const int x     = s & ( width - 1 );
        const int y     = s >> log2W;
        const int level = x <= lastX && y <= lastY ? int( levels[y * stride + x] ) : 0;
        const int r     = dequant( level, scale, rightShift, inputMaximum );
        block[y * stride + x] = int16_t( r );
        if( ict != 0 )
        {
          other[y * stride + x] = int16_t( jointResidual( ict, int( int16_t( r ) ) ) );
        }
      }
    }
    return;
  }

  // TrQuant::xIT: the vertical transform (rounded and clipped), then the horizontal one.
  int16_t*  mVer   = pc.matrices + matrixOffset( typeVer, log2H );
  int16_t*  mHor   = pc.matrices + matrixOffset( typeHor, log2W );
  const int shift2 = 6 + 15 - 1 - pc.bitDepth;
  int       nx     = lastX + 1;   // columns that hold levels; the others are zero
  int       ny     = lastY + 1;

  if( width == 1 || height == 1 )
  {
    // xIT's 1-D transforms (ISP sub-partitions one sample wide or high): a
    // single rounding, with a shift one bit larger than the 2-D second shift
    const bool ver   = width == 1;
    const int  n     = ver ? height : width;
    const int  count = ver ? ny : nx;
    int16_t*   m     = ver ? mVer : mHor;
    const int  shift = shift2 + 1;
    for( int s = tid; s < n; s += THREADS )
    {
      int sum = 0;
      for( int k = 0; k < count; k++ )
      {
        sum += dequant( int( levels[ver ? k * stride : k] ), coefScale( sl, scale, width, ver ? 0 : k, ver ? k : 0 ), rightShift, inputMaximum )
               * int( m[k * n + s] );
      }
      const int r = clip3( COEFF_MIN, COEFF_MAX, ( sum + ( 1 << ( shift - 1 ) ) ) >> shift );
      block[ver ? s * stride : s] = int16_t( r );
      if( ict != 0 )
      {
        other[ver ? s * stride : s] = int16_t( jointResidual( ict, r ) );
      }
    }
    return;
  }

  const uint lfnst = job[6];
  if( lfnst != 0 )
  {
    // xInvLfnst: the set's matrix times the top-left 4x4 coefficients (only 8
    // of them for 4x4 and 8x8 blocks) gives the top-left 4x4 for blocks 4 wide
    // or high, else the top-left 8x8 without its bottom-right 4x4, row by row
    // or (transposed) column by column. All other coefficients are zero.
    const int  index     = int( lfnst & 3 ) - 1;
    const int  set       = int( ( lfnst >> 2 ) & 3 );
    const bool transpose = ( lfnst & 16 ) != 0;
    const bool block8    = ( lfnst & 32 ) != 0;
    const int  inputs    = ( lfnst & 64 ) != 0 ? 8 : 16;
    if( tid < 16 )
    {
      const int x  = kLfnstScan[tid] & 3;
      const int y  = kLfnstScan[tid] >> 2;
      s_lfnst[tid] = x <= lastX && y <= lastY ? dequant( int( levels[y * stride + x] ), coefScale( sl, scale, width, x, y ), rightShift, inputMaximum ) : 0;
    }
    nx = min( width, 8 );
    ny = min( height, 8 );
    for( int s = tid; s < nx * ny; s += THREADS )
    {
      const int k = s / nx;
      s_coef[k * 32 + s - k * nx] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if( tid < ( block8 ? 48 : 16 ) )
    {
      int8_t*       row = pc.lfnst + ( block8 ? ( ( set * 2 + index ) * 48 + tid ) * 16 : 6144 + ( ( set * 2 + index ) * 16 + tid ) * 16 );
      int           sum = 0;
      for( int i = 0; i < inputs; i++ )
      {
        sum += s_lfnst[i] * int( row[i] );
      }
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
      s_coef[y * 32 + x] = clip3( COEFF_MIN, COEFF_MAX, ( sum + 64 ) >> 7 );
    }
  }
  else
  {
    for( int s = tid; s < nx * ny; s += THREADS )
    {
      const int k = s / nx;
      const int x = s - k * nx;
      s_coef[k * 32 + x] = dequant( int( levels[k * stride + x] ), coefScale( sl, scale, width, x, k ), rightShift, inputMaximum );
    }
  }
  GroupMemoryBarrierWithGroupSync();

  for( int s = tid; s < nx * height; s += THREADS )
  {
    const int y   = s / nx;
    const int x   = s - y * nx;
    int       sum = 0;
    for( int k = 0; k < ny; k++ )
    {
      sum += s_coef[k * 32 + x] * int( mVer[k * height + y] );
    }
    s_mid[y * 32 + x] = clip3( COEFF_MIN, COEFF_MAX, ( sum + 64 ) >> 7 );
  }
  GroupMemoryBarrierWithGroupSync();

  for( int s = tid; s < width * height; s += THREADS )
  {
    const int x   = s & ( width - 1 );
    const int y   = s >> log2W;
    int       sum = 0;
    for( int k = 0; k < nx; k++ )
    {
      sum += s_mid[y * 32 + k] * int( mHor[k * width + x] );
    }
    const int r = clip3( COEFF_MIN, COEFF_MAX, ( sum + ( 1 << ( shift2 - 1 ) ) ) >> shift2 );
    block[y * stride + x] = int16_t( r );
    if( ict != 0 )
    {
      other[y * stride + x] = int16_t( jointResidual( ict, r ) );
    }
  }
}
