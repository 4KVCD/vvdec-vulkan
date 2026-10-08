// Dequantization and the inverse transforms, as vvdec's Quant::dequant and
// TrQuant::invTransformNxN and invTransformICT do them: a workgroup takes one
// transform block of one component, reads the levels the parser left in the
// picture and writes the residual over them.
//
// A job, 8 uints (VulkanLib/GpuJobs.h, TrafoJob):
//   0: x | y << 16 (in samples of its component)
//   1: log2 width | log2 height << 4 | component << 8 | horizontal type << 12 | vertical type << 14 | flags << 16
//      | (joint Cb-Cr mode + 4) << 24                      (types: vvdec's TransType, 0 DCT-II, 1 DCT-VIII, 2 DST-VII)
//   2: last column | last row << 8 of the levels             (the parser wrote [0, last column] x [0, last row])
//   3: scale   4: right shift (negative: a left shift)   5: the levels' maximum
// flags: 1 transform skip, 2 BDPCM horizontal, 4 BDPCM vertical
//
// The threads dequantize the levels into shared memory, run the vertical
// transform (rounded and clipped to 16 bits) into shared memory, then the
// horizontal one into the picture; with transform skip, each takes samples
// (with BDPCM, rows or columns) by itself.

#include "common.slang"

struct Params
{
  uint*     jobs;
  int16_t*  plane0;
  int16_t*  plane1;
  int16_t*  plane2;
  int16_t*  matrices;   // TrQuant's matrices, TrafoJob's order (GpuRecon.cpp)
  int       stride0;
  int       stride12;
  int       count;
  int       bitDepth;
};

[[vk::push_constant]] Params pc;

#define THREADS 64

#define FLAG_TS 1
#define FLAG_BDPCM_HOR 2
#define FLAG_BDPCM_VER 4

#define COEFF_MIN ( -32768 )   // maxLog2TrDynamicRange 15
#define COEFF_MAX 32767

// the levels there are: at most 32 x 32 (64-point transforms keep 32)
groupshared int s_coef[32 * 32];   // [row][column]
groupshared int s_mid[64 * 32];    // the vertical transform's output, [row][column]

int16_t* planeOf( int comp )
{
  return comp == 0 ? pc.plane0 : ( comp == 1 ? pc.plane1 : pc.plane2 );
}

int strideOf( int comp )
{
  return comp == 0 ? pc.stride0 : pc.stride12;
}

// the N x N matrix (N = 1 << log2) of a transform type: DCT-II from 2 to 64
// points, then DST-VII and DCT-VIII from 4 to 32
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

// Quant's DeQuantImpl without scaling lists
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

// joint Cb-Cr: the other chroma component's residual from this one's (TU::getICTMode)
int jointResidual( int ict, int v )
{
  return abs( ict ) == 2 ? ( ict > 0 ? v : -v ) : ( ( ict > 0 ? v : -v ) >> 1 );
}

[shader( "compute" )]
[numthreads( THREADS, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int index = int( group.x ) + int( group.y ) * 32768;
  if( index >= pc.count )
  {
    return;   // the whole workgroup
  }
  const int tid          = int( local.x );
  uint*     job          = pc.jobs + index * 8;
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
  const int width        = 1 << log2W;
  const int height       = 1 << log2H;
  const int stride       = strideOf( comp );
  int16_t*  block        = planeOf( comp ) + y0 * stride + x0;
  int16_t*  other        = planeOf( comp == 1 ? 2 : 1 ) + y0 * stride + x0;   // joint Cb-Cr's other component

  if( ( flags & FLAG_TS ) != 0 )
  {
    // transform skip: the dequantized levels are the residual; BDPCM first
    // adds them up along its direction (invResDPCM), a thread per row or column
    if( ( flags & FLAG_BDPCM_HOR ) != 0 )
    {
      for( int y = tid; y < height; y += THREADS )
      {
        int sum = 0;
        for( int x = 0; x < width; x++ )
        {
          const int level = int( block[y * stride + x] );
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
          const int level = int( block[y * stride + x] );
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
        const int level = x <= lastX && y <= lastY ? int( block[y * stride + x] ) : 0;
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

  // TrQuant::xIT: the vertical transform, rounded and clipped, then the horizontal one
  int16_t*  mVer   = pc.matrices + matrixOffset( typeVer, log2H );
  int16_t*  mHor   = pc.matrices + matrixOffset( typeHor, log2W );
  const int shift2 = 6 + 15 - 1 - pc.bitDepth;
  const int nx     = lastX + 1;   // the columns with levels; the others are zero
  const int ny     = lastY + 1;

  for( int s = tid; s < nx * ny; s += THREADS )
  {
    const int k = s / nx;
    const int x = s - k * nx;
    s_coef[k * 32 + x] = dequant( int( block[k * stride + x] ), scale, rightShift, inputMaximum );
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
