// Lists the edges that the deblocking filter processes, on the GPU (the CPU's version is GpuRecon::listEdges),
// from the deblocking parameters that shaders/dbparams.slang derived. There are four lists of GpuJobs.h Edge
// records: luma vertical and luma horizontal edges (one per 4x4 block), then chroma vertical and chroma horizontal
// edges (one per segment, as deblock.slang visits them). Each list runs CTU by CTU, through a CTU's rows from top
// to bottom and through each row's blocks from left to right, which is the CPU's order.
//
// The passes (pc.pass) run one workgroup per CTU, except SCAN:
// 0 COUNT: counts each CTU's edges in each list;
// 1 SCAN:  a single workgroup turns the counts into each CTU's offsets in the lists (each list placed after the
//          previous one) and stores each list's first entry and count;
// 2 WRITE: each CTU writes its edges at its offsets.

#include "common.slang"
#include "scan.slang"

struct Params
{
  uint* lfp;         // dbparams.slang's LoopFilterParam, 2 words per 4x4 block, per CTU and direction
  uint* slices;      // mider.slang's slice table
  uint* counts;      // 4 words per CTU: its number of edges in each list; SCAN replaces them with offsets
  uint* lists;       // first entry and count of each list (2 words each), then the total of all lists
  uint* edges;       // out: Edge records (2 words each)
  int   widthInCtus;
  int   heightInCtus;
  int   log2Ctu;
  int   width;       // luma samples
  int   height;
  int   chroma;      // nonzero for a format with chroma
  int   csx;
  int   csy;
  int   pass;
  int   capacity;    // room in edges, in Edge records
  uint  lfpTag;      // the picture's tag in the parameters (dbparams.slang's loadLfp)
  uint* args;        // SCAN: the front end's indirect arguments (GpuRecon's m_frontEnd.args) for the deblocking passes
};

[[vk::push_constant]] Params pc;

uint* sliceOfCtu( int ctu ) { return pc.slices + pc.widthInCtus * pc.heightInCtus + pc.slices[ctu] * 64; }
bool  deblockingOff( uint* s ) { return ( s[1] & ( 1u << 23 ) ) != 0; }

// Returns which edges a 4x4 block has in direction dir (bit 0: a luma edge, bit 1: a chroma segment) and outputs
// their Edge words.
uint blockEdges( int ctu, int dir, int r, int c, out uint2 luma, out uint2 chromaEdge )
{
  const int  n    = 1 << ( pc.log2Ctu - 2 );
  const int  x0   = ( ctu % pc.widthInCtus ) << pc.log2Ctu;
  const int  y0   = ( ctu / pc.widthInCtus ) << pc.log2Ctu;
  uint*      p    = pc.lfp + ( ( ( 2 * ctu + dir ) << ( 2 * ( pc.log2Ctu - 2 ) ) ) + r * n + c ) * 2;
  const uint b    = p[1];
  const uint a    = ( b >> 16 ) == pc.lfpTag ? p[0] : 0u;   // (an entry of another picture: none)
  const uint bs   = a >> 24;
  luma            = uint2( 0 );
  chromaEdge      = uint2( 0 );
  if( bs == 0 )
  {
    return 0;
  }
  // The grid of the chroma segments, in luma samples: across the edges, one
  // every 8 chroma samples (16 luma samples in 4:2:0); along an edge, one per
  // 4x4 luma block (the segment covers that block's lines).
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
    // SCAN: for each list in turn, turn the CTUs' counts into offsets (each thread takes a run of CTUs); each list
    // starts where the previous one ends.
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
      uint       run;
      const uint before = workgroupPrefixSum( sum, tid, run );
      if( tid == 0 )
      {
        pc.lists[l * 2]     = base;
        pc.lists[l * 2 + 1] = run;
        // The deblocking pass for this list runs one thread per edge, in workgroups of 64.
        storePassArgs( pc.args, kPassDeblockLumaVer + l, ( run + 63 ) / 64, 1, run, base );
      }
      uint at = base + before;
      for( int c = c0; c < c1; c++ )
      {
        const uint v          = pc.counts[c * 4 + l];
        pc.counts[c * 4 + l]  = at;
        at += v;
      }
      base += run;
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
  const int   cols    = min( n, ( pc.width - x0 + 3 ) >> 2 );
  const int   rows    = min( n, ( pc.height - y0 + 3 ) >> 2 );
  const bool  off     = deblockingOff( sliceOfCtu( ctu ) );
  const float rcpCols = 1.0f / float( max( cols, 1 ) );
  for( int dir = 0; dir < 2; dir++ )
  {
    // Each thread takes four consecutive 4x4 blocks, in raster order within the CTU.
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
        const int row = int( ( float( i ) + 0.5f ) * rcpCols );   // (exactly i / cols, since i and cols are small)
        which[k]      = blockEdges( ctu, dir, row, i - row * cols, lumaEdge, chromaEdge );
        e[k][0]  = lumaEdge;
        e[k][1]  = chromaEdge;
        cl += which[k] & 1;
        cc += ( which[k] >> 1 ) & 1;
      }
    }
    uint       totalL, totalC;
    const uint beforeL = workgroupPrefixSum( cl, tid, totalL );
    const uint beforeC = workgroupPrefixSum( cc, tid, totalC );
    if( pc.pass == 0 && tid == 0 )
    {
      pc.counts[ctu * 4 + dir]     = totalL;
      pc.counts[ctu * 4 + 2 + dir] = totalC;
    }
    if( pc.pass == 2 )
    {
      uint atL = pc.counts[ctu * 4 + dir] + beforeL;
      uint atC = pc.counts[ctu * 4 + 2 + dir] + beforeC;
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
