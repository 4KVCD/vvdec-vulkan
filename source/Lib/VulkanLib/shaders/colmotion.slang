// Builds a picture's colocated motion, as vvdec's storeColMotion does, from the motion the GPU derived
// (shaders/mider.slang): for each CTU, one entry per 8x8 block, copied from the block's top-left 4x4 block.
// TMVP in later pictures reads it. DMVR's refined motion is written into it afterwards, by shaders/inter.slang.

#include "common.slang"

struct Params
{
  int* motion;      // mider.slang's MotionInfo per 4x4 block (5 words), CTU by CTU
  int* colMotion;   // out: the picture's ColocatedMotionInfo (5 words), CTU by CTU, num8x8 entries per CTU
  int  count;       // number of 8x8 blocks in all CTUs (one thread each)
  int  log2Ctu;
  int  num8x8;      // 8x8 blocks per CTU
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
  const int ctu      = i >> ( 2 * ( pc.log2Ctu - 3 ) );   // (num8x8 == 1 << this shift)
  const int e        = i & ( pc.num8x8 - 1 );
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
