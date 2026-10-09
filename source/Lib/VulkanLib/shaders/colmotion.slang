// The colocated motion of a picture (vvdec's storeColMotion) from the motion
// the GPU derived (shaders/mider.slang): per CTU, on the 8x8 grid, the motion
// of each 8x8 block's top-left 4x4 block -- what TMVP in the later pictures
// reads (DMVR's refinements are written into it later, by shaders/inter.slang).

#include "common.slang"

struct Params
{
  int* motion;      // mider.slang's MotionInfo of the 4x4 blocks (5 words), CTU by CTU
  int* colMotion;   // the picture's ColocatedMotionInfo (5 words), CTU by CTU, num8x8 a CTU
  int  count;       // the 8x8 blocks of all CTUs
  int  log2Ctu;
  int  num8x8;
};

[[vk::push_constant]] Params pc;

[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 id : SV_DispatchThreadID )
{
  const int i = int( id.y * 65536 + id.x );
  if( i >= pc.count )
  {
    return;
  }
  const int ctu      = i / pc.num8x8;
  const int e        = i - ctu * pc.num8x8;
  const int across8  = 1 << ( pc.log2Ctu - 3 );
  const int ctuSize4 = 1 << ( pc.log2Ctu - 2 );
  const int x4       = ( e & ( across8 - 1 ) ) * 2;
  const int y4       = ( e >> ( pc.log2Ctu - 3 ) ) * 2;
  int*      src      = pc.motion + ( ctu * ctuSize4 * ctuSize4 + y4 * ctuSize4 + x4 ) * 5;
  int*      dst      = pc.colMotion + i * 5;
  for( int k = 0; k < 5; k++ )
  {
    dst[k] = src[k];
  }
}
