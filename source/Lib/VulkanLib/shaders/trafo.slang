// Dequantization and the inverse transforms, as vvdec's Quant::dequant and
// TrQuant::invTransformNxN and invTransformICT do them: a thread takes one
// transform block of one component, reads the levels the parser left in the
// picture and writes the residual over them, in place.
//
// A job, 8 uints (VulkanLib/GpuJobs.h, TrafoJob):
//   0: x | y << 16 (in samples of its component)
//   1: log2 width | log2 height << 4 | component << 8 | horizontal type << 12 | vertical type << 14 | flags << 16
//      | (joint Cb-Cr mode + 4) << 24                      (types: vvdec's TransType, 0 DCT-II, 1 DCT-VIII, 2 DST-VII)
//   2: last column | last row << 8 of the levels             (the parser wrote [0, last column] x [0, last row])
//   3: scale   4: right shift (negative: a left shift)   5: the levels' maximum
// flags: 1 transform skip, 2 BDPCM horizontal, 4 BDPCM vertical
//
// The transforms run in place: the vertical one column by column (a column's
// levels read first), the horizontal one row by row; their intermediate values
// are clipped to 16 bits, so the picture holds them between the two.

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

#define FLAG_TS 1
#define FLAG_BDPCM_HOR 2
#define FLAG_BDPCM_VER 4

#define COEFF_MIN ( -32768 )   // maxLog2TrDynamicRange 15
#define COEFF_MAX 32767

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

[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  if( int( id.x ) >= pc.count )
  {
    return;
  }
  uint*     job          = pc.jobs + int( id.x ) * 8;
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

  if( ( flags & FLAG_TS ) != 0 )
  {
    // transform skip: the dequantized levels are the residual; BDPCM first
    // adds them up along its direction (invResDPCM)
    if( ( flags & FLAG_BDPCM_HOR ) != 0 )
    {
      for( int y = 0; y < height; y++ )
      {
        int sum = 0;
        for( int x = 0; x < width; x++ )
        {
          const int level = int( block[y * stride + x] );
          sum             = x == 0 ? level : clip3( COEFF_MIN, COEFF_MAX, sum + level );
          block[y * stride + x] = int16_t( dequant( sum, scale, rightShift, inputMaximum ) );
        }
      }
    }
    else if( ( flags & FLAG_BDPCM_VER ) != 0 )
    {
      for( int x = 0; x < width; x++ )
      {
        int sum = 0;
        for( int y = 0; y < height; y++ )
        {
          const int level = int( block[y * stride + x] );
          sum             = y == 0 ? level : clip3( COEFF_MIN, COEFF_MAX, sum + level );
          block[y * stride + x] = int16_t( dequant( sum, scale, rightShift, inputMaximum ) );
        }
      }
    }
    else
    {
      for( int y = 0; y < height; y++ )
      {
        for( int x = 0; x < width; x++ )
        {
          const int level = x <= lastX && y <= lastY ? int( block[y * stride + x] ) : 0;
          block[y * stride + x] = int16_t( dequant( level, scale, rightShift, inputMaximum ) );
        }
      }
    }
  }
  else
  {
    // TrQuant::xIT: the vertical transform, rounded and clipped, then the horizontal one
    int16_t*  mVer   = pc.matrices + matrixOffset( typeVer, log2H );
    int16_t*  mHor   = pc.matrices + matrixOffset( typeHor, log2W );
    const int shift2 = 6 + 15 - 1 - pc.bitDepth;
    int       line[64];

    for( int x = 0; x <= lastX; x++ )
    {
      for( int k = 0; k <= lastY; k++ )
      {
        line[k] = dequant( int( block[k * stride + x] ), scale, rightShift, inputMaximum );
      }
      for( int y = 0; y < height; y++ )
      {
        int sum = 0;
        for( int k = 0; k <= lastY; k++ )
        {
          sum += line[k] * int( mVer[k * height + y] );
        }
        block[y * stride + x] = int16_t( clip3( COEFF_MIN, COEFF_MAX, ( sum + 64 ) >> 7 ) );
      }
    }
    // the columns after lastX are zero
    for( int y = 0; y < height; y++ )
    {
      for( int k = 0; k <= lastX; k++ )
      {
        line[k] = int( block[y * stride + k] );
      }
      for( int x = 0; x < width; x++ )
      {
        int sum = 0;
        for( int k = 0; k <= lastX; k++ )
        {
          sum += line[k] * int( mHor[k * width + x] );
        }
        block[y * stride + x] = int16_t( clip3( COEFF_MIN, COEFF_MAX, ( sum + ( 1 << ( shift2 - 1 ) ) ) >> shift2 ) );
      }
    }
  }

  // joint Cb-Cr: the other chroma component's residual from this one (TU::getICTMode)
  if( ict != 0 )
  {
    int16_t* other = planeOf( comp == 1 ? 2 : 1 ) + y0 * stride + x0;
    for( int y = 0; y < height; y++ )
    {
      for( int x = 0; x < width; x++ )
      {
        const int v = int( block[y * stride + x] );
        const int a = abs( ict );
        const int r = a == 2 ? ( ict > 0 ? v : -v ) : ( ( ict > 0 ? v : -v ) >> 1 );
        other[y * stride + x] = int16_t( r );
      }
    }
  }
}
