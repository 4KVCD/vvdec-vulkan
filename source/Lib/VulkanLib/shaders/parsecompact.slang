// After the GPU's parse (parse.slang, each CTU's records in its slots): the records gathered in raster order of the
// CTUs, as the host lays out the CPU's parse's (GpuFrontEnd.cpp's writeCuRecords, writeCtuRecords) -- pass 0 the
// CTUs' records (each one's first CU and TU: the counts' prefix sums, one workgroup), pass 1 the copies (a workgroup a
// CTU). Variant CHECK: the levels the GPU's parse left against the CPU's parse's, over the rectangles of each TU's
// coded blocks the transforms read (a workgroup a TU), the differences counted.

#include "common.slang"

#ifndef CHECK
#  define CHECK 0
#endif

#if CHECK
struct Params
{
  uint*         tus;        // the TU records (raster order)
  int16_t*      gpu0;       // the GPU's planes of levels
  int16_t*      gpu1;
  int16_t*      gpu2;
  int16_t*      cpu0;       // the CPU's (the picture in host memory)
  int16_t*      cpu1;
  int16_t*      cpu2;
  Atomic<uint>* counts;     // [0] levels compared, [1] differ, [2] the first differing TU + 1, [3] its component
  int           numTus;
  int           gpuStride0;
  int           gpuStride12;
  int           cpuStride0;
  int           cpuStride12;
  int           width;      // the planes' (luma samples), the chroma's subsampling
  int           height;
  int           csx;
  int           csy;
};

[[vk::push_constant]] Params pc;

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
      // (a block off the planes: a wrong record)
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
  uint* cusIn;      // parse.slang's: kCuSlots records a CTU
  uint* tusIn;      // kTuSlots a CTU
  uint* ctuOut;     // per CTU 16 words: its CUs, its TUs, ...
  uint* ctuInfo;    // per CTU: its slice; its tile | the first column of its tile << 16
  uint* cus;        // out: the CU records in raster order of the CTUs
  uint* tus;        // out: the TU records
  uint* ctus;       // out: the CTU records (CtuRecord, 4 words)
  uint* totals;     // out: the CUs, the TUs
  int   numCtus;
  int   cuSlots;
  int   tuSlots;
  int   pass;
  int   cuCap;      // the room of cus and tus (in records)
  int   tuCap;
};

[[vk::push_constant]] Params pc;

groupshared uint s_cus[1024];
groupshared uint s_tus[1024];

[shader( "compute" )]
[numthreads( 1024, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int tid = int( local.x );
  if( pc.pass == 0 )
  {
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
      // inclusive prefix sums (Hillis-Steele)
      for( int d = 1; d < 1024; d <<= 1 )
      {
        const uint ac = tid >= d ? s_cus[tid - d] : 0u;
        const uint at = tid >= d ? s_tus[tid - d] : 0u;
        GroupMemoryBarrierWithGroupSync();
        s_cus[tid] += ac;
        s_tus[tid] += at;
        GroupMemoryBarrierWithGroupSync();
      }
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
  // pass 1: a CTU's records
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
