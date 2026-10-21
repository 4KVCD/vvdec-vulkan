// LMCS inverse luma mapping, as vvdec's Reshape::rspCtuBcw applies it after a picture's blocks are reconstructed
// (in the mapped domain) and before the in-loop filters: every luma sample of a CTU whose slice uses LMCS goes
// through the inverse table. Each thread maps 4 adjacent samples of a row, so a workgroup maps 256 samples.

#include "common.slang"

struct Params
{
  int16_t* plane;    // luma
  int16_t* lut;      // the inverse table, 1 << bitDepth entries
  uint8_t* ctus;     // one byte per CTU, raster order: 1 when the CTU's slice uses LMCS
  int      stride;
  int      width;    // picture width in luma samples (a multiple of 8)
  int      height;
  int      ctuLog2;
  int      widthInCtus;
};

[[vk::push_constant]] Params pc;

static const int kThreads          = 64;
static const int kSamplesPerThread = 4;   // read and written as one 8-byte word

// The inverse-mapped value of a luma sample
uint inverseMap( uint sample )
{
  return uint( int( pc.lut[sample] ) );
}

bool ctuUsesLmcs( int x, int y )
{
  return pc.ctus[( y >> pc.ctuLog2 ) * pc.widthInCtus + ( x >> pc.ctuLog2 )] != 0;
}

[shader( "compute" )]
[numthreads( kThreads, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int x = ( int( group.x ) * kThreads + int( local.x ) ) * kSamplesPerThread;
  const int y = int( group.y );   // one row of the picture per workgroup row
  if( x >= pc.width || !ctuUsesLmcs( x, y ) )
  {
    return;
  }
  uint2*      samples = (uint2*)( pc.plane + y * pc.stride + x );
  const uint2 packed  = *samples;   // four 16-bit samples, two to a word
  *samples = uint2( inverseMap( packed.x & 0xffff ) | inverseMap( packed.x >> 16 ) << 16,
                    inverseMap( packed.y & 0xffff ) | inverseMap( packed.y >> 16 ) << 16 );
}
