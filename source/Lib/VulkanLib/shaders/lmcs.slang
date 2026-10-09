// LMCS's inverse luma mapping, as vvdec's Reshape::rspCtuBcw does it once a
// picture's blocks are reconstructed (in the mapped domain) and before the
// in-loop filters: every luma sample of a CTU whose slice uses LMCS goes
// through the inverse table. A thread takes 4 samples of a row, a workgroup
// 256 of them.

#include "common.slang"

struct Params
{
  int16_t* plane;    // luma
  int16_t* lut;      // the inverse table, 1 << bitDepth entries
  uint8_t* ctus;     // per CTU, raster order: 1 when its slice uses LMCS
  int      stride;
  int      width;    // of the picture (a multiple of 8)
  int      height;
  int      ctuLog2;
  int      widthInCtus;
};

[[vk::push_constant]] Params pc;

[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int x = ( int( group.x ) * 64 + int( local.x ) ) * 4;
  const int y = int( group.y );
  if( x >= pc.width || pc.ctus[( y >> pc.ctuLog2 ) * pc.widthInCtus + ( x >> pc.ctuLog2 )] == 0 )
  {
    return;
  }
  uint2* p = (uint2*)( pc.plane + y * pc.stride + x );
  const uint2 v = *p;
  const int   a = int( pc.lut[v.x & 0xffff] );
  const int   b = int( pc.lut[v.x >> 16] );
  const int   c = int( pc.lut[v.y & 0xffff] );
  const int   d = int( pc.lut[v.y >> 16] );
  *p = uint2( uint( a ) | uint( b ) << 16, uint( c ) | uint( d ) << 16 );
}
