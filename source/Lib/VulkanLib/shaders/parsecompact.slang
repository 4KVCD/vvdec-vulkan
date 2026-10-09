// Runs after the GPU's parse (parse.slang), which leaves each CTU's CU and TU records in slots reserved for the CTU.
// This shader gathers them in raster order of the CTUs, the layout the host gives the CPU's parse's records
// (GpuFrontEnd.cpp's writeCuRecords and writeCtuRecords), so that the two can be compared word for word:
// - pass 0 (one workgroup of 1024 threads) writes the CTU records: each CTU's first CU and first TU are the prefix
//   sums of the CTUs' counts, which it works out 1024 CTUs at a time; it also writes the totals;
// - pass 1 (one workgroup per CTU) copies the CTU's records from its slots to their place.
// The CHECK variant compares the coefficient levels the GPU's parse left with those the CPU's parse left in the
// picture, over the rectangles of each TU's coded blocks that the transforms read (one workgroup per TU), and counts
// the differences. GpuReconParse.cpp records both after the parse's own submissions.

#include "common.slang"

#ifndef CHECK
#  define CHECK 0
#endif

#if CHECK
struct Params
{
  uint*         tus;        // the TU records, in raster order of the CTUs (pass 1's output)
  int16_t*      gpu0;       // the GPU's planes of levels: Y, Cb, Cr
  int16_t*      gpu1;
  int16_t*      gpu2;
  int16_t*      cpu0;       // the CPU's: the picture's planes in host memory, where the CPU's parse left the levels
  int16_t*      cpu1;
  int16_t*      cpu2;
  // [0] the levels compared, [1] the levels that differ, [2] the index + 1 of the first TU found to differ (whichever
  // workgroup gets there first), [3] a component of that TU that differs
  Atomic<uint>* counts;
  int           numTus;
  int           gpuStride0;   // the planes' strides, in samples: luma, chroma
  int           gpuStride12;
  int           cpuStride0;
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
    for( int u = int( local.x ); u < w * h; u += 64 )
    {
      const int x  = x0 + u % w;
      const int y  = y0 + u / w;
      const int g  = comp == 0 ? int( pc.gpu0[y * pc.gpuStride0 + x] ) : ( comp == 1 ? int( pc.gpu1[y * pc.gpuStride12 + x] ) : int( pc.gpu2[y * pc.gpuStride12 + x] ) );
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
  uint* cusIn;      // parse.slang's CU records, cuSlots slots per CTU
  uint* tusIn;      // parse.slang's TU records, tuSlots slots per CTU
  uint* ctuOut;     // parse.slang's ctuOut, 16 words per CTU: its numbers of CUs and TUs first
  uint* ctuInfo;    // per CTU, 2 words: its slice; its tile | 1 << 16 if it is in its tile's first column
  uint* cus;        // out: the CU records in raster order of the CTUs
  uint* tus;        // out: the TU records in raster order of the CTUs
  uint* ctus;       // out: the CTU records (CtuRecord, 4 words each)
  uint* totals;     // out: the total numbers of CUs and TUs
  int   numCtus;
  int   cuSlots;
  int   tuSlots;
  int   pass;       // 0 or 1
  int   cuCap;      // the room in cus and tus, in records (the CPU's numbers); records past it are dropped
  int   tuCap;
};

[[vk::push_constant]] Params pc;

groupshared uint s_cus[1024];   // pass 0: the prefix sums of a chunk of 1024 CTUs' numbers of CUs and TUs
groupshared uint s_tus[1024];

[shader( "compute" )]
[numthreads( 1024, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int tid = int( local.x );
  if( pc.pass == 0 )
  {
    // Pass 0: one workgroup goes through the CTUs 1024 at a time, one per thread; baseCu and baseTu count the CUs and
    // TUs of the chunks before
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
    }
    return;
  }
  // Pass 1: workgroup i copies CTU i's CU and TU records to their place, as far as cus and tus have room
  const int  ctu     = int( group.x ) + int( group.y ) * 32768;
  if( ctu >= pc.numCtus )
  {
    return;
  }
  const uint nc      = pc.ctuOut[ctu * 16];
  const uint nt      = pc.ctuOut[ctu * 16 + 1];
  const uint firstCu = pc.ctus[ctu * 4];
  const uint firstTu = pc.ctus[ctu * 4 + 3];
  const uint cuEnd   = min( firstCu + nc, uint( pc.cuCap ) );
  const uint tuEnd   = min( firstTu + nt, uint( pc.tuCap ) );
  for( uint i = uint( tid ); firstCu + i / 32u < cuEnd; i += 1024u )
  {
    pc.cus[firstCu * 32u + i] = pc.cusIn[uint( ctu * pc.cuSlots ) * 32u + i];
  }
  for( uint i = uint( tid ); firstTu + i / 16u < tuEnd; i += 1024u )
  {
    pc.tus[firstTu * 16u + i] = pc.tusIn[uint( ctu * pc.tuSlots ) * 16u + i];
  }
}
#endif
