// Runs after the GPU's parse (parse.slang), which leaves the CU and TU records staged wherever there was room, each
// with a meta word (its CTU and its index in the CTU). This shader puts them in raster order of the CTUs, the layout
// the host gives the CPU's parse's records (GpuFrontEnd.cpp's writeCuRecords and writeCtuRecords), which the front end
// reads and the check compares word for word:
// - pass 0 (one workgroup of 1024 threads), once every CTU is parsed, writes the CTU records: each CTU's first CU and
//   first TU are the prefix sums of the CTUs' counts, which it works out 1024 CTUs at a time; it also writes the
//   totals, with the parse's counters (the host records it after each round of the parse's dispatches, so it does
//   nothing until the picture is parsed);
// - passes 1 and 2 copy the staged CU and TU records to their places (32 CU records or 64 TU records a workgroup),
//   pass 3 copies the CTU records (256 a workgroup) beside them.
// The CHECK variant compares the coefficient levels the GPU's parse left (where each TU's record says) with those the
// CPU's parse left in the picture, over the rectangles of each TU's coded blocks that the transforms read (one
// workgroup per TU), and counts the differences. GpuReconParse.cpp records both after the parse's own submissions.

#include "common.slang"

#ifndef CHECK
#  define CHECK 0
#endif

#if CHECK
struct Params
{
  uint*         tus;        // the TU records, in raster order of the CTUs (pass 1's output)
  int16_t*      gpu;        // the GPU's levels: each coded block's rectangle where its TU's record says (words 9 to 12)
  int16_t*      cpu0;       // the CPU's: the picture's planes in host memory, where the CPU's parse left the levels
  int16_t*      cpu1;
  int16_t*      cpu2;
  // [0] the levels compared, [1] the levels that differ, [2] the index + 1 of the first TU found to differ (whichever
  // workgroup gets there first), [3] a component of that TU that differs
  Atomic<uint>* counts;
  int           numTus;
  int           cpuStride0;   // the CPU's planes' strides, in samples: luma, chroma
  int           cpuStride12;
  int           width;      // the picture's size, in luma samples
  int           height;
  int           csx;        // the chroma subsampling shifts
  int           csy;
};

[[vk::push_constant]] Params pc;

// Workgroup t compares the coded blocks of TU t
[shader( "compute" )]
[numthreads( 64, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int t = int( group.x ) + int( group.y ) * 32768;
  if( t >= pc.numTus )
  {
    return;
  }
  uint*       r = pc.tus + t * 16;
  uint        compared = 0;
  uint        differ   = 0;
  int         firstComp = -1;
  // Each coded block: its rectangle [0, maxScanPosX] x [0, maxScanPosY], clamped to the block (Cr has none in joint
  // Cb-Cr mode 3)
  for( int comp = 0; comp < 3; comp++ )
  {
    const uint pos  = comp == 0 ? r[0] : r[2];
    const uint size = comp == 0 ? r[1] : r[3];
    if( ( size & 0x10000u ) == 0 || ( ( r[4] >> uint( comp ) ) & 1u ) == 0 || ( comp == 2 && ( ( r[4] >> 3 ) & 3u ) == 3u ) )
    {
      continue;
    }
    const int x0 = int( pos & 0xffffu );
    const int y0 = int( pos >> 16 );
    const int w  = min( int( ( r[6] >> uint( 8 * comp ) ) & 0xffu ), int( size & 0xffu ) - 1 ) + 1;
    const int h  = min( int( ( r[7] >> uint( 8 * comp ) ) & 0xffu ), int( ( size >> 8 ) & 0xffu ) - 1 ) + 1;
    const int pw = comp == 0 ? pc.width : pc.width >> pc.csx;
    const int ph = comp == 0 ? pc.height : pc.height >> pc.csy;
    if( x0 + w > pw || y0 + h > ph )
    {
      // (the record puts the block off the planes, so the record is wrong: count it as a difference)
      differ++;
      firstComp = firstComp < 0 ? comp : firstComp;
      continue;
    }
    const uint at  = r[9 + comp];                                // the GPU's levels of the block, rows of row levels
    const int  row = int( ( r[12] >> uint( 8 * comp ) ) & 0xffu );
    for( int u = int( local.x ); u < w * h; u += 64 )
    {
      const int x  = x0 + u % w;
      const int y  = y0 + u / w;
      const int g  = int( pc.gpu[at + uint( ( u / w ) * row + u % w )] );
      const int c  = comp == 0 ? int( pc.cpu0[y * pc.cpuStride0 + x] ) : ( comp == 1 ? int( pc.cpu1[y * pc.cpuStride12 + x] ) : int( pc.cpu2[y * pc.cpuStride12 + x] ) );
      compared++;
      if( g != c )
      {
        differ++;
        firstComp = firstComp < 0 ? comp : firstComp;
      }
    }
  }
  if( compared != 0 )
  {
    pc.counts[0].add( compared );
  }
  if( differ != 0 )
  {
    pc.counts[1].add( differ );
    uint expected = 0;
    if( pc.counts[2].compareExchange( expected, uint( t + 1 ) ) == 0u )
    {
      pc.counts[3].store( uint( firstComp ), MemoryOrder.Relaxed );
    }
  }
}

#else
struct Params
{
  uint* cusIn;      // parse.slang's staged CU records (32 words each)
  uint* tusIn;      // its staged TU records (16 words each)
  uint* cuMeta;     // per staged CU record: its CTU << kLocalBits | its index among the CTU's CUs
  uint* tuMeta;     // per staged TU record: its CTU << kLocalBits | its index among the CTU's TUs
  uint* ctuOut;     // parse.slang's ctuOut, 16 words per CTU: its numbers of CUs and TUs first
  uint* ctuInfo;    // per CTU, 2 words: its slice; its tile | 1 << 16 if it is in its tile's first column
  uint* sync;       // parse.slang's sync: [2 + CTU] is 1 once the CTU is parsed
  uint* counters;   // parse.slang's counters: the staged CU and TU records, the job lists' room (5 words), the levels
  uint* cus;        // out (pass 1): the CU records in raster order of the CTUs
  uint* tus;        // out (pass 2): the TU records in raster order of the CTUs
  uint* ctus;       // the CTU records (CtuRecord, 4 words each): pass 0 writes them, passes 1 to 3 read them
  uint* ctusOut;    // out (pass 3): a copy of the CTU records
  uint* totals;     // out (pass 0): the numbers of CUs and TUs, then the parse's 8 counters
  int   numCtus;
  int   numCus;     // the CU records staged (passes 1 to 3)
  int   numTus;     // the TU records staged
  int   pass;       // 0: the CTU records and totals; 1: the CU records; 2: the TU records; 3: the CTU records' copy
};

[[vk::push_constant]] Params pc;

static const uint kLocalBits = 12;   // parse.slang's: the bits of a meta word below the CTU

groupshared uint s_cus[1024];   // pass 0: the prefix sums of a chunk of 1024 CTUs' numbers of CUs and TUs
groupshared uint s_tus[1024];
groupshared uint s_parsed;      // pass 0: 1 while every CTU looked at is parsed

[shader( "compute" )]
[numthreads( 1024, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int tid = int( local.x );
  if( pc.pass == 0 )
  {
    // Pass 0, once every CTU is parsed (a substream marks each of its CTUs parsed, also those it gave up on)
    if( tid == 0 )
    {
      s_parsed = 1;
    }
    GroupMemoryBarrierWithGroupSync();
    for( int ctu = tid; ctu < pc.numCtus; ctu += 1024 )
    {
      if( pc.sync[2 + ctu] == 0 )
      {
        s_parsed = 0;
      }
    }
    GroupMemoryBarrierWithGroupSync();
    if( s_parsed == 0 )
    {
      return;
    }
    // One workgroup goes through the CTUs 1024 at a time, one per thread; baseCu and baseTu count the CUs and TUs of
    // the chunks before
    uint baseCu = 0;
    uint baseTu = 0;
    for( int chunk = 0; chunk < pc.numCtus; chunk += 1024 )
    {
      const int  ctu = chunk + tid;
      const uint nc  = ctu < pc.numCtus ? pc.ctuOut[ctu * 16] : 0u;
      const uint nt  = ctu < pc.numCtus ? pc.ctuOut[ctu * 16 + 1] : 0u;
      s_cus[tid]     = nc;
      s_tus[tid]     = nt;
      GroupMemoryBarrierWithGroupSync();
      // Inclusive prefix sums of the chunk's counts (Hillis-Steele)
      for( int d = 1; d < 1024; d <<= 1 )
      {
        const uint ac = tid >= d ? s_cus[tid - d] : 0u;
        const uint at = tid >= d ? s_tus[tid - d] : 0u;
        GroupMemoryBarrierWithGroupSync();
        s_cus[tid] += ac;
        s_tus[tid] += at;
        GroupMemoryBarrierWithGroupSync();
      }
      // The CTU record, laid out as writeCtuRecords lays it out
      if( ctu < pc.numCtus )
      {
        const uint firstCu = baseCu + s_cus[tid] - nc;
        const uint firstTu = baseTu + s_tus[tid] - nt;
        pc.ctus[ctu * 4 + 0] = firstCu;
        pc.ctus[ctu * 4 + 1] = nc | pc.ctuInfo[ctu * 2] << 16;
        pc.ctus[ctu * 4 + 2] = ( ( pc.ctuInfo[ctu * 2 + 1] >> 16 ) & 1u ) | ( pc.ctuInfo[ctu * 2 + 1] & 0xffffu ) << 16;
        pc.ctus[ctu * 4 + 3] = firstTu;
      }
      baseCu += s_cus[1023];
      baseTu += s_tus[1023];
      GroupMemoryBarrierWithGroupSync();
    }
    if( tid == 0 )
    {
      pc.totals[0] = baseCu;
      pc.totals[1] = baseTu;
      for( int i = 0; i < 8; i++ )
      {
        pc.totals[2 + i] = pc.counters[i];
      }
    }
    return;
  }
  const uint first = ( group.x + group.y * 32768u ) * 1024u;   // the workgroup's first word of the records copied
  if( pc.pass == 1 )
  {
    // Pass 1: each 32 threads copy a staged CU record to its place: its CTU's first CU + its index in the CTU
    const uint i = ( first + uint( tid ) ) / 32u;
    if( i < uint( pc.numCus ) )
    {
      const uint meta = pc.cuMeta[i];
      const uint at   = pc.ctus[( meta >> kLocalBits ) * 4u] + ( meta & ( ( 1u << kLocalBits ) - 1u ) );
      pc.cus[at * 32u + uint( tid & 31 )] = pc.cusIn[i * 32u + uint( tid & 31 )];
    }
  }
  else if( pc.pass == 2 )
  {
    // Pass 2: each 16 threads copy a staged TU record to its place: its CTU's first TU + its index in the CTU
    const uint i = ( first + uint( tid ) ) / 16u;
    if( i < uint( pc.numTus ) )
    {
      const uint meta = pc.tuMeta[i];
      const uint at   = pc.ctus[( meta >> kLocalBits ) * 4u + 3u] + ( meta & ( ( 1u << kLocalBits ) - 1u ) );
      pc.tus[at * 16u + uint( tid & 15 )] = pc.tusIn[i * 16u + uint( tid & 15 )];
    }
  }
  else
  {
    // Pass 3: the CTU records copied, a word a thread
    const uint w = first + uint( tid );
    if( w < uint( pc.numCtus ) * 4u )
    {
      pc.ctusOut[w] = pc.ctus[w];
    }
  }
}
#endif
