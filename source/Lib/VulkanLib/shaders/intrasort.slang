// On the GPU, puts the intra jobs in the order of their waves (the CPU's version is GpuRecon::recordIntra). The
// input is the jobs that shaders/jobs.slang made, CTU by CTU, each tagged with its wave. Each job is cut into bands
// of rows of at most 256 samples, and each band becomes one workgroup of shaders/intra.slang. The bands are written
// in wave order (within a wave, in the jobs' order), together with each band's wait: the number of bands in all
// earlier waves, which must be done before the band starts.
//
// The job indices are sorted by wave with a stable radix sort of two 8-bit digits (so waves must be below 65536),
// three passes per digit; then the bands are built. The passes (pc.pass):
// 0 HIST:    one workgroup per block of 1024 jobs counts each digit value in the block (hist, digit-major);
// 1 SCAN:    one workgroup turns the counts into offsets per digit value and block, in place;
// 2 SCATTER: one wave per block moves the block's jobs, in order, each to the next free place of its digit value
//            (which keeps the sort stable);
// When every wave is below 256 (HIST of the first digit finds none above), the second digit is 0 everywhere and its
// sort would keep the order: its HIST and SCAN do nothing, and its SCATTER copies the keys.
// 3 BANDS:   one workgroup per block of 1024 sorted jobs counts each job's bands and stores the block's total
//            (bandSums);
// 4 BSCAN:   one workgroup turns the blocks' band totals into offsets and stores the number of all bands;
// 5 PLACE:   one workgroup per block writes each sorted job's bands at the job's offset, and records the first band
//            of each wave where the wave begins (waveFirst);
// 6 WAITS:   one workgroup per block writes each band's wait: the first band of the band's wave.

#include "common.slang"

struct Params
{
  uint*         jobs;       // jobs.slang's intra jobs, 12 words each; word 7 holds the wave
  uint*         count;      // number of intra jobs (the intra list's entry in jobs.slang's totals)
  uint*         keysA;      // job indices: the sort's two buffers; after both digits keysA holds the sorted ones
  uint*         keysB;
  uint*         hist;       // per digit value (256) and block: counts, which SCAN replaces with offsets
  uint*         bandAt;     // first band of each sorted job
  uint*         bandSums;   // bands per block, which BSCAN replaces with offsets; then the number of all bands
  uint*         waveFirst;  // first band of each wave (65536 entries)
  uint*         sorted;     // out: the bands as IntraJob records, word 7 replaced by their rows: first | count << 16
  uint*         waits;      // out: per band, the number of bands in the earlier waves
  Atomic<uint>* flags;      // [0] is set when a wave is 65536 or more, [1] when one is 256 or more
  int           numBlocks;  // blocks of 1024 jobs that the jobs buffer can hold
  int           pass;
  int           shift;      // shift of the current digit: 0 or 8
  int           capacity;   // room in sorted and waits, in bands
  uint*         args;       // BSCAN: the front end's indirect arguments (GpuRecon's m_frontArgs) for intra's pass
};

[[vk::push_constant]] Params pc;

static const int kBand  = 256;
static const int kBlock = 1024;

int  bandRows( uint* job ) { return max( 1, kBand / int( job[1] & 0xff ) ); }
int  bandsOf( uint* job ) { const int rows = bandRows( job ); return ( int( ( job[1] >> 8 ) & 0xff ) + rows - 1 ) / rows; }
uint waveOf( uint index ) { return pc.jobs[index * 12 + 7]; }

groupshared uint s_scan[1024];
groupshared uint s_hist[256];

// Synchronizes one wave's threads on its shared memory: each sees what the others wrote before.
void waveSync()
{
  spirv_asm { OpControlBarrier Subgroup Subgroup AcquireRelease|WorkgroupMemory };
}

// Exclusive prefix sum of the 1024 entries of s_scan, in place; returns the total to every thread.
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
  uint*      src   = pc.shift == 0 ? pc.keysA : pc.keysB;   // this digit's input; digit 0 reads the jobs in order
  uint*      dst   = pc.shift == 0 ? pc.keysB : pc.keysA;
  const uint i     = uint( block * kBlock + tid );

  // (the second digit's passes when it is 0 for every job; SCATTER copies the keys below)
  const bool secondDigitZero = pc.shift != 0 && pc.flags[1].load() == 0;
  if( pc.pass == 0 )
  {
    // HIST
    if( secondDigitZero )
    {
      return;
    }
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
      if( wave >= 256u )
      {
        pc.flags[1].store( 1u );
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
    // SCAN: the 256 x numBlocks counts, digit-major; each thread takes a run of them.
    if( secondDigitZero )
    {
      return;
    }
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
    if( secondDigitZero )
    {
      if( i < n )
      {
        dst[i] = src[i];
      }
      return;
    }
    const uint base = uint( block * kBlock );
    const uint end  = min( n, base + uint( kBlock ) );
    if( base >= end )
    {
      return;
    }
    if( WaveGetLaneCount() != 32 )
    {
      // SCATTER (waves of another size): one thread moves the block's jobs, in order, each to the next place of its
      // digit value.
      if( tid != 0 )
      {
        return;
      }
      for( uint j = base; j < end; j++ )
      {
        const uint index = pc.shift == 0 ? j : src[j];
        const uint digit = ( waveOf( index ) >> pc.shift ) & 255;
        const uint at    = pc.hist[digit * pc.numBlocks + block];
        pc.hist[digit * pc.numBlocks + block] = at + 1;
        dst[at]                               = index;
      }
      return;
    }
    // SCATTER: the first wave moves the block's jobs in order, 32 at a time. Each job goes to the next free place of
    // its digit value after the jobs before it: those of the earlier chunks (the running places in s_hist) and the
    // lower threads of its chunk with the same digit (ballots of the digit's bits find them).
    if( tid >= 32 )
    {
      return;
    }
    for( int d = tid; d < 256; d += 32 )
    {
      s_hist[d] = pc.hist[d * pc.numBlocks + block];
    }
    // (all of the chunks' reads first, so that they wait for memory together)
    uint indices[32], digits[32];
    [unroll]
    for( int c = 0; c < 32; c++ )
    {
      const uint j = min( base + uint( c * 32 + tid ), end - 1 );
      indices[c]   = pc.shift == 0 ? j : src[j];
      digits[c]    = ( waveOf( indices[c] ) >> pc.shift ) & 255;
    }
    waveSync();
    const uint below = ( 1u << uint( tid ) ) - 1u;   // the lower threads
    [unroll]
    for( int c = 0; c < 32; c++ )
    {
      if( base + uint( c * 32 ) >= end )
      {
        break;
      }
      const bool valid = base + uint( c * 32 + tid ) < end;
      const uint digit = digits[c];
      uint       same  = WaveActiveBallot( valid ).x;   // the chunk's threads with this thread's digit
      [unroll]
      for( int b = 0; b < 8; b++ )
      {
        const bool bit = ( ( digit >> b ) & 1u ) != 0;
        const uint set = WaveActiveBallot( bit ).x;
        same &= bit ? set : ~set;
      }
      if( valid )
      {
        dst[s_hist[digit] + countbits( same & below )] = indices[c];
      }
      waveSync();   // (all have read s_hist before the lowest thread of each digit moves its place on)
      if( valid && ( same & below ) == 0 )
      {
        s_hist[digit] += countbits( same );
      }
      waveSync();
    }
    return;
  }

  // The remaining passes read the sorted jobs (keysA).
  if( pc.pass == 3 || pc.pass == 5 )
  {
    // BANDS counts each sorted job's bands and sums them over the block; PLACE writes each job's bands in place.
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
    // BSCAN: turn the blocks' band counts into offsets (each thread takes a run of blocks) and store the total.
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
      // intra.slang runs one workgroup per band, in rows of 32768 workgroups.
      pc.args[12] = min( total, 32768u );
      pc.args[13] = ( total + 32767u ) / 32768u;
      pc.args[14] = 1;
      pc.args[40] = total;
      pc.args[41] = 0;
    }
    return;
  }
  // WAITS: write each band's wait.
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
