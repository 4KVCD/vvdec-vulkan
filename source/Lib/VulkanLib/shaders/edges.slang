// The edges the deblocking filters (GpuRecon::listEdges) on the GPU, from the
// deblocking's parameters (shaders/dbparams.slang): luma's vertical and
// horizontal edges (a 4x4 block's), chroma's (a segment as deblock.slang
// visits them), each list CTU by CTU, a CTU's rows in order, a row's blocks
// left to right -- the CPU's order -- as GpuJobs.h's Edge.
//
// The passes (pc.pass), a workgroup a CTU but in SCAN:
// 0 COUNT: each CTU's edges of each list;
// 1 SCAN:  one workgroup: each CTU's offsets in the lists (each list after
//   the one before), the lists' firsts and counts;
// 2 WRITE: each CTU's edges at its offsets.

#include "common.slang"

struct Params
{
  uint* lfp;         // dbparams.slang's LoopFilterParam, 2 words, per CTU, direction and 4x4 block
  uint* slices;      // mider.slang's slice table
  uint* counts;      // per CTU its edges of each list (4 words), then its offsets
  uint* lists;       // per list its first and count (2 words each), then the count of all
  uint* edges;       // out: Edge (2 words)
  int   widthInCtus;
  int   heightInCtus;
  int   log2Ctu;
  int   width;       // luma samples
  int   height;
  int   chroma;      // a format with chroma
  int   csx;
  int   csy;
  int   pass;
  int   capacity;    // the edges' room
  uint* args;        // SCAN: the front end's arguments (GpuRecon's m_frontArgs): the deblocking's passes
};

[[vk::push_constant]] Params pc;

uint* sliceOfCtu( int ctu ) { return pc.slices + pc.widthInCtus * pc.heightInCtus + pc.slices[ctu] * 64; }
bool  deblockingOff( uint* s ) { return ( s[1] & ( 1u << 23 ) ) != 0; }

groupshared uint s_scan[2][256];

// a 4x4 block's edges of a direction: luma's (bit 0), chroma's (bit 1), and their words
uint blockEdges( int ctu, int dir, int r, int c, out uint2 luma, out uint2 chromaEdge )
{
  const int  n    = 1 << ( pc.log2Ctu - 2 );
  const int  x0   = ( ctu % pc.widthInCtus ) << pc.log2Ctu;
  const int  y0   = ( ctu / pc.widthInCtus ) << pc.log2Ctu;
  uint*      p    = pc.lfp + ( ( ( 2 * ctu + dir ) << ( 2 * ( pc.log2Ctu - 2 ) ) ) + r * n + c ) * 2;
  const uint a    = p[0];
  const uint b    = p[1];
  const uint bs   = a >> 24;
  luma            = uint2( 0 );
  chromaEdge      = uint2( 0 );
  if( bs == 0 )
  {
    return 0;
  }
  // the luma grid of the chroma segments: across the edge every 8 chroma
  // samples (at the 4:2:0 rate), along it as many lines as a 4x4 luma block has
  const int across = dir == 0 ? ( ( 8 << pc.csx ) / 4 * ( 4 >> pc.csx ) ) << pc.csx : ( ( 8 << pc.csy ) / 4 * ( 4 >> pc.csy ) ) << pc.csy;
  const int gridX  = dir == 0 ? across : ( 4 >> pc.csx ) << pc.csx;
  const int gridY  = dir == 0 ? ( 4 >> pc.csy ) << pc.csy : across;
  const int lx     = x0 + c * 4;
  const int ly     = y0 + r * 4;
  uint      which  = 0;
  if( ( bs & 3 ) != 0 )
  {
    which |= 1;
    luma = uint2( uint( lx ) | uint( ly ) << 16, ( a & 0xff ) | bs << 8 | ( b & 0xff ) << 16 );
  }
  if( pc.chroma != 0 && ( ly & ( gridY - 1 ) ) == 0 && ( bs & 0x3c ) != 0 && ( lx & ( gridX - 1 ) ) == 0 )
  {
    which |= 2;
    chromaEdge = uint2( uint( lx >> pc.csx ) | uint( ly >> pc.csy ) << 16, ( ( a >> 8 ) & 0xff ) | ( ( a >> 16 ) & 0xff ) << 8 | bs << 16 | ( ( b >> 8 ) & 0xff ) << 24 );
  }
  return which;
}

[shader( "compute" )]
[numthreads( 256, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int tid  = int( local.x );
  const int ctus = pc.widthInCtus * pc.heightInCtus;
  if( pc.pass == 1 )
  {
    // SCAN: per list, the CTUs' offsets (a thread a run of CTUs), the lists one after the other
    const int per = ( ctus + 255 ) / 256;
    const int c0  = min( tid * per, ctus );
    const int c1  = min( c0 + per, ctus );
    uint      base = 0;
    for( int l = 0; l < 4; l++ )
    {
      uint sum = 0;
      for( int c = c0; c < c1; c++ )
      {
        sum += pc.counts[c * 4 + l];
      }
      s_scan[0][tid] = sum;
      GroupMemoryBarrierWithGroupSync();
      if( tid == 0 )
      {
        uint run = 0;
        for( int i = 0; i < 256; i++ )
        {
          const uint v   = s_scan[0][i];
          s_scan[0][i]   = run;
          run += v;
        }
        pc.lists[l * 2]     = base;
        pc.lists[l * 2 + 1] = run;
        s_scan[1][0]        = run;
        // a thread an edge, workgroups of 64
        pc.args[15 + 3 * l]     = ( run + 63 ) / 64;
        pc.args[15 + 3 * l + 1] = 1;
        pc.args[15 + 3 * l + 2] = 1;
        pc.args[42 + 2 * l]     = run;
        pc.args[42 + 2 * l + 1] = base;
      }
      GroupMemoryBarrierWithGroupSync();
      uint at = base + s_scan[0][tid];
      for( int c = c0; c < c1; c++ )
      {
        const uint v          = pc.counts[c * 4 + l];
        pc.counts[c * 4 + l]  = at;
        at += v;
      }
      base += s_scan[1][0];
      GroupMemoryBarrierWithGroupSync();
    }
    if( tid == 0 )
    {
      pc.lists[8] = base;
    }
    return;
  }

  const int  ctu  = int( group.x );
  const int  n    = 1 << ( pc.log2Ctu - 2 );
  const int  x0   = ( ctu % pc.widthInCtus ) << pc.log2Ctu;
  const int  y0   = ( ctu / pc.widthInCtus ) << pc.log2Ctu;
  const int  cols = min( n, ( pc.width - x0 + 3 ) >> 2 );
  const int  rows = min( n, ( pc.height - y0 + 3 ) >> 2 );
  const bool off  = deblockingOff( sliceOfCtu( ctu ) );
  for( int dir = 0; dir < 2; dir++ )
  {
    // a thread four blocks in a row's order
    const int blocks = off ? 0 : rows * cols;
    uint2     e[4][2];
    uint      which[4];
    uint      cl = 0, cc = 0;
    for( int k = 0; k < 4; k++ )
    {
      const int i = tid * 4 + k;
      which[k]    = 0;
      e[k][0]     = uint2( 0 );
      e[k][1]     = uint2( 0 );
      if( i < blocks )
      {
        uint2 lumaEdge, chromaEdge;
        which[k] = blockEdges( ctu, dir, i / cols, i % cols, lumaEdge, chromaEdge );
        e[k][0]  = lumaEdge;
        e[k][1]  = chromaEdge;
        cl += which[k] & 1;
        cc += ( which[k] >> 1 ) & 1;
      }
    }
    s_scan[0][tid] = cl;
    s_scan[1][tid] = cc;
    GroupMemoryBarrierWithGroupSync();
    if( tid < 2 )
    {
      uint run = 0;
      for( int i = 0; i < 256; i++ )
      {
        const uint v   = s_scan[tid][i];
        s_scan[tid][i] = run;
        run += v;
      }
      if( pc.pass == 0 )
      {
        pc.counts[ctu * 4 + ( tid == 0 ? dir : 2 + dir )] = run;
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if( pc.pass == 2 )
    {
      uint atL = pc.counts[ctu * 4 + dir] + s_scan[0][tid];
      uint atC = pc.counts[ctu * 4 + 2 + dir] + s_scan[1][tid];
      for( int k = 0; k < 4; k++ )
      {
        if( ( which[k] & 1 ) != 0 )
        {
          if( atL < uint( pc.capacity ) )
          {
            pc.edges[atL * 2]     = e[k][0].x;
            pc.edges[atL * 2 + 1] = e[k][0].y;
          }
          atL++;
        }
        if( ( which[k] & 2 ) != 0 )
        {
          if( atC < uint( pc.capacity ) )
          {
            pc.edges[atC * 2]     = e[k][1].x;
            pc.edges[atC * 2 + 1] = e[k][1].y;
          }
          atC++;
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
}
