// Intra's jobs in the order of their waves (GpuRecon::recordIntra), on the
// GPU: from the jobs shaders/jobs.slang made (CTU by CTU, each with its
// wave), the jobs cut into bands of rows of 256 samples at most, a band a
// workgroup of shaders/intra.slang, in the order of the waves -- within a
// wave in the jobs' order -- and each band's count of the bands of the waves
// before its, which it waits for.
//
// The jobs' indices are sorted by wave with a stable radix sort, two digits
// of 8 bits (waves below 65536), each digit in three passes; then the bands.
// The passes (pc.pass):
// 0 HIST:    a workgroup a block of 1024 jobs: the block's count of each
//            digit (hist, digit-major);
// 1 SCAN:    one workgroup: the digits' and blocks' offsets (in place);
// 2 SCATTER: a lane a block, the block's jobs in order: each at its digit's
//            next place (stable);
// 3 BANDS:   a workgroup a block of 1024 sorted jobs: each one's bands, the
//            block's sums (bandSums);
// 4 BSCAN:   one workgroup: the blocks' band offsets, the count of all;
// 5 PLACE:   a workgroup a block: each sorted job's bands at its place, the
//            first band of each wave where it begins (waveFirst);
// 6 WAITS:   a workgroup a block: each band's wait, its wave's first band.

#include "common.slang"

struct Params
{
  uint*         jobs;       // jobs.slang's intra jobs (12 words, w7 the wave)
  uint*         count;      // the jobs' count (jobs.slang's totals: the intra list's)
  uint*         keysA;      // the jobs' indices: the sort's two buffers (the sorted ones in keysA after the two digits)
  uint*         keysB;
  uint*         hist;       // per digit value (256) and block: counts, then offsets
  uint*         bandAt;     // per sorted job its first band
  uint*         bandSums;   // per block its bands, then their offsets; then the count of all
  uint*         waveFirst;  // per wave its first band (65536)
  uint*         sorted;     // out: the bands (IntraJob, w7 the rows: first | count << 16)
  uint*         waits;      // out: each band's count of the bands of the waves before its
  Atomic<uint>* flags;      // [0]: a wave of 65536 or more
  int           numBlocks;  // of 1024 jobs (the jobs' room)
  int           pass;
  int           shift;      // the digit's: 0, 8
  int           capacity;   // sorted's and waits' room, in bands
  uint*         args;       // BSCAN: the front end's arguments (GpuRecon's m_frontArgs): intra's pass
};

[[vk::push_constant]] Params pc;

static const int kBand  = 256;
static const int kBlock = 1024;

int  bandRows( uint* job ) { return max( 1, kBand / int( job[1] & 0xff ) ); }
int  bandsOf( uint* job ) { const int rows = bandRows( job ); return ( int( ( job[1] >> 8 ) & 0xff ) + rows - 1 ) / rows; }
uint waveOf( uint index ) { return pc.jobs[index * 12 + 7]; }

groupshared uint s_scan[1024];
groupshared uint s_hist[256];

// an exclusive prefix sum of s_scan (1024 entries) in place, its total returned (all threads)
uint scan1024( int tid )
{
  for( int d = 1; d < 1024; d <<= 1 )
  {
    const uint v = tid >= d ? s_scan[tid - d] : 0u;
    GroupMemoryBarrierWithGroupSync();
    s_scan[tid] += v;
    GroupMemoryBarrierWithGroupSync();
  }
  const uint total = s_scan[1023];
  const uint mine  = tid > 0 ? s_scan[tid - 1] : 0u;
  GroupMemoryBarrierWithGroupSync();
  s_scan[tid] = mine;
  GroupMemoryBarrierWithGroupSync();
  return total;
}

[shader( "compute" )]
[numthreads( 1024, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int  tid   = int( local.x );
  const int  block = int( group.x );
  const uint n     = pc.count[0];
  uint*      src   = pc.shift == 0 ? pc.keysA : pc.keysB;   // the digit's input (digit 0: the jobs' order, keysA unused)
  uint*      dst   = pc.shift == 0 ? pc.keysB : pc.keysA;
  const uint i     = uint( block * kBlock + tid );

  if( pc.pass == 0 )
  {
    // HIST
    if( tid < 256 )
    {
      s_hist[tid] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if( i < n )
    {
      const uint index = pc.shift == 0 ? i : src[i];
      const uint wave  = waveOf( index );
      if( wave >= 65536u )
      {
        pc.flags[0].store( 1u );
      }
      InterlockedAdd( s_hist[( wave >> pc.shift ) & 255], 1u );
    }
    GroupMemoryBarrierWithGroupSync();
    if( tid < 256 )
    {
      pc.hist[tid * pc.numBlocks + block] = s_hist[tid];
    }
    return;
  }
  if( pc.pass == 1 )
  {
    // SCAN: the 256 x numBlocks counts, digit-major, a thread a run of them
    const int total = 256 * pc.numBlocks;
    const int per   = ( total + 1023 ) / 1024;
    const int a0    = min( tid * per, total );
    const int a1    = min( a0 + per, total );
    uint      sum   = 0;
    for( int a = a0; a < a1; a++ )
    {
      sum += pc.hist[a];
    }
    s_scan[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    scan1024( tid );
    uint at = s_scan[tid];
    for( int a = a0; a < a1; a++ )
    {
      const uint v = pc.hist[a];
      pc.hist[a]   = at;
      at += v;
    }
    return;
  }
  if( pc.pass == 2 )
  {
    // SCATTER: the block's jobs in their order, each to its digit's next place
    if( tid != 0 )
    {
      return;
    }
    const uint end = min( n, uint( ( block + 1 ) * kBlock ) );
    for( uint j = uint( block * kBlock ); j < end; j++ )
    {
      const uint index = pc.shift == 0 ? j : src[j];
      const uint digit = ( waveOf( index ) >> pc.shift ) & 255;
      const uint at    = pc.hist[digit * pc.numBlocks + block];
      pc.hist[digit * pc.numBlocks + block] = at + 1;
      dst[at]                               = index;
    }
    return;
  }

  // the sorted jobs (keysA)
  if( pc.pass == 3 || pc.pass == 5 )
  {
    // BANDS: each sorted job's bands, the block's prefix; PLACE: its bands at their places
    s_scan[tid] = i < n ? uint( bandsOf( pc.jobs + pc.keysA[i] * 12 ) ) : 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint total = scan1024( tid );
    if( pc.pass == 3 )
    {
      if( tid == 0 )
      {
        pc.bandSums[block] = total;
      }
      return;
    }
    if( i >= n )
    {
      return;
    }
    const uint index = pc.keysA[i];
    uint*      job   = pc.jobs + index * 12;
    const uint first = pc.bandSums[block] + s_scan[tid];
    const uint wave  = waveOf( index );
    pc.bandAt[i]     = first;
    if( ( i == 0 || waveOf( pc.keysA[i - 1] ) != wave ) && wave < 65536u )
    {
      pc.waveFirst[wave] = first;
    }
    const int rows = bandRows( job );
    const int h    = int( ( job[1] >> 8 ) & 0xff );
    uint      at   = first;
    for( int r = 0; r < h; r += rows, at++ )
    {
      if( at < uint( pc.capacity ) )
      {
        uint* band = pc.sorted + at * 12;
        for( int k = 0; k < 12; k++ )
        {
          band[k] = job[k];
        }
        band[7] = uint( r ) | uint( min( rows, h - r ) ) << 16;
      }
    }
    return;
  }
  if( pc.pass == 4 )
  {
    // BSCAN: the blocks' band offsets (a thread a run of blocks), the count of all
    const int per = ( pc.numBlocks + 1023 ) / 1024;
    const int b0  = min( tid * per, pc.numBlocks );
    const int b1  = min( b0 + per, pc.numBlocks );
    uint      sum = 0;
    for( int b = b0; b < b1; b++ )
    {
      sum += pc.bandSums[b];
    }
    s_scan[tid] = sum;
    GroupMemoryBarrierWithGroupSync();
    const uint total = scan1024( tid );
    uint       at    = s_scan[tid];
    for( int b = b0; b < b1; b++ )
    {
      const uint v     = pc.bandSums[b];
      pc.bandSums[b] = at;
      at += v;
    }
    if( tid == 0 )
    {
      pc.bandSums[pc.numBlocks] = total;
      // a workgroup a band, in rows of 32768
      pc.args[12] = min( total, 32768u );
      pc.args[13] = ( total + 32767u ) / 32768u;
      pc.args[14] = 1;
      pc.args[40] = total;
      pc.args[41] = 0;
    }
    return;
  }
  // WAITS: each band's wait
  if( i >= n )
  {
    return;
  }
  const uint index = pc.keysA[i];
  const uint wave  = waveOf( index );
  const uint wait  = wave < 65536u ? pc.waveFirst[wave] : 0u;
  const int  b     = bandsOf( pc.jobs + index * 12 );
  for( int k = 0; k < b; k++ )
  {
    const uint at = pc.bandAt[i] + uint( k );
    if( at < uint( pc.capacity ) )
    {
      pc.waits[at] = wait;
    }
  }
}
