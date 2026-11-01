// The hand-over's pack (GpuHandOver): a picture's plane copied from its twin -- 16-bit samples, rows the margins'
// stride apart, which the decoder keeps for every bit depth -- into another device's buffer (VideoMetricsLab's VMAF
// engine: a slot it exported, imported here), at that buffer's layout: rows pc.dstStride bytes apart, samples of
// pc.dstBytes bytes, the visible picture only (the source starts at its top-left sample: the conformance window's
// and the caller's crop applied). Each sample is shifted left by pc.shift on the way (an 8-bit picture scored beside
// a 10-bit one: 2, as FFmpeg widens it), and stored in one byte (8 bits) or two (little-endian, low bits).
// A thread takes four samples of a row; group.x the row's segment, group.y the row.

#include "common.slang"

struct Params
{
  int16_t* src;        // the twin at the first sample to copy
  uint8_t* dst;        // the destination's first row
  int      srcStride;  // samples
  int      dstStride;  // bytes
  int      width;      // samples a row
  int      height;     // rows
  int      dstBytes;   // 1 or 2
  int      shift;      // 0, or 2 to widen 8-bit samples to 10
};

[[vk::push_constant]] Params pc;

static const int kThreads   = 256;
static const int kPerThread = 4;
static const int kSegment   = kThreads * kPerThread;

[shader( "compute" )]
[numthreads( kThreads, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int row = int( group.y );
  const int x0  = int( group.x ) * kSegment + int( local.x ) * kPerThread;
  if( row >= pc.height )
  {
    return;
  }
  int16_t* src = at( pc.src, uint( row * pc.srcStride ) );
  uint8_t* dst = pc.dst + row * pc.dstStride;
  for( int k = 0; k < kPerThread; k++ )
  {
    const int x = x0 + k;
    if( x >= pc.width )
    {
      break;
    }
    const int v = int( *at( src, uint( x ) ) ) << pc.shift;
    if( pc.dstBytes == 2 )
    {
      *(int16_t*) ( dst + 2 * x ) = int16_t( v );
    }
    else
    {
      dst[x] = uint8_t( v );
    }
  }
}
