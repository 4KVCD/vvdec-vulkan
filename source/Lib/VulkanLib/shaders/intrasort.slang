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
// 7 SMALL:   runs first, one workgroup: with at most 1024 jobs (a B-picture's few intra jobs) it does all of the
//            above itself, and gives passes 0 to 6 (dispatched indirectly) no workgroups; with more, it gives them
//            theirs. (Ten dispatches with a barrier after each took ~0.18 ms of the front end for a few jobs.)

#include "common.slang"
#include "intrajob.slang"

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
  uint*         args;       // the front end's indirect arguments (GpuRecon's m_frontEnd.args): intra's pass (BSCAN,
                            // SMALL), and passes 0 to 6's (SMALL: kArgsBlocks, kArgsSingle)
};

[[vk::push_constant]] Params pc;

static const int kBand  = 256;
static const int kBlock = 1024;

// The passes (pc.pass; see above, and GpuReconFrontEnd.cpp's IntraSortPass)
static const int kPassHist     = 0;
static const int kPassScan     = 1;
static const int kPassScatter  = 2;
static const int kPassBands    = 3;
static const int kPassBandScan = 4;
static const int kPassPlace    = 5;
static const int kPassWaits    = 6;
static const int kPassSmall    = 7;

// Where SMALL writes the indirect arguments of passes 0 to 6 in pc.args (GpuReconFrontEnd.cpp's kSortArgs*): the
// passes of a workgroup per block (HIST, SCATTER, BANDS, PLACE, WAITS), and those of a single workgroup (SCAN, BSCAN)
static const int kArgsBlocks = 50;
static const int kArgsSingle = 53;

// The words of an intra job (intrajob.slang) that this pass uses. Word 1 holds the block's width and height in
// samples (bits 0-7 and 8-15). Word kWaveWord holds the job's wave, which jobs.slang sets; in a band's copy of the job,
// that word holds the band's rows instead (first | count << 16).
static const int kWaveWord = 7;

uint* jobAt( uint index )
{
  return pc.jobs + index * kIntraJobWords;
}

int jobWidth( uint* job )
{
  return int( job[1] & 0xff );
}

int jobHeight( uint* job )
{
  return int( ( job[1] >> 8 ) & 0xff );
}

uint waveOf( uint index )
{
  return jobAt( index )[kWaveWord];
}

// Returns how many of the job's rows each of its bands holds: as many as fit in kBand samples, and at least one.
int bandRows( uint* job )
{
  return max( 1, kBand / jobWidth( job ) );
}

int bandsOf( uint* job )
{
  const int rows = bandRows( job );
  return ( jobHeight( job ) + rows - 1 ) / rows;
}

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

// Sorts the 1024 entries of s_scan in increasing order, in place (a bitonic sort)
void sort1024( int tid )
{
  for( int k = 2; k <= 1024; k <<= 1 )
  {
    for( int j = k >> 1; j > 0; j >>= 1 )
    {
      const int other = tid ^ j;
      if( other > tid )
      {
        const uint a      = s_scan[tid];
        const uint b      = s_scan[other];
        const bool rising = ( tid & k ) == 0;   // the direction of this run of k entries
        if( rising == ( a > b ) )
        {
          s_scan[tid]   = b;
          s_scan[other] = a;
        }
      }
      GroupMemoryBarrierWithGroupSync();
    }
  }
}

// Writes the job's bands into the sorted list, starting at place `first`. Each band is a copy of the job's words with
// the band's rows in word kWaveWord (first | count << 16). With withWaits, it also writes each band's wait (pc.waits).
void writeJobBands( uint* job, uint first, bool withWaits, uint wait )
{
  const int rows = bandRows( job );
  const int h    = jobHeight( job );
  uint      at   = first;
  for( int r = 0; r < h; r += rows, at++ )
  {
    if( at < uint( pc.capacity ) )
    {
      uint* band = pc.sorted + at * kIntraJobWords;
      for( int k = 0; k < kIntraJobWords; k++ )
      {
        band[k] = job[k];
      }
      band[kWaveWord] = uint( r ) | uint( min( rows, h - r ) ) << 16;
      if( withWaits )
      {
        pc.waits[at] = wait;
      }
    }
  }
}

// Stores the total number of bands and the indirect arguments of the intra pass, which runs one workgroup per band.
void storeBandTotal( uint total )
{
  pc.bandSums[pc.numBlocks] = total;
  storeJobPassArgs( pc.args, kPassIntra, total, 0u );
}

// SMALL (pass 7, one workgroup): see the passes above. The jobs are sorted as keys wave << 10 | job, so that the jobs'
// order breaks the ties (a stable sort by wave; with at most 1024 jobs, the waves stay below 1024 too: each wave is one
// above an earlier job's). In sorted order, an exclusive prefix sum of the jobs' bands gives each job's first band, and
// a running maximum of the places where the wave changes gives the first job of each job's wave, whose first band is
// the wait of the job's bands.
void smallSort( int tid, uint n )
{
  const bool small = n <= uint( kBlock );
  if( tid == 0 )
  {
    pc.args[kArgsBlocks]     = small ? 0u : uint( pc.numBlocks );
    pc.args[kArgsBlocks + 1] = 1u;
    pc.args[kArgsBlocks + 2] = 1u;
    pc.args[kArgsSingle]     = small ? 0u : 1u;
    pc.args[kArgsSingle + 1] = 1u;
    pc.args[kArgsSingle + 2] = 1u;
  }
  if( !small )
  {
    return;
  }
  // (the sorted keys' places below n hold the jobs: the other places' keys are the largest)
  const bool valid = uint( tid ) < n;
  if( valid )
  {
    const uint wave = waveOf( uint( tid ) );
    if( wave >= 256u )
    {
      pc.flags[1].store( 1u );   // (as HIST sets it)
    }
    s_scan[tid] = min( wave, 0xffffu ) << 10 | uint( tid );   // (below the padding's keys, whatever the wave)
  }
  else
  {
    s_scan[tid] = 0xffffffffu;
  }
  GroupMemoryBarrierWithGroupSync();
  sort1024( tid );
  const uint key      = s_scan[tid];
  const uint prevWave = tid > 0 ? s_scan[tid - 1] >> 10 : 0xffffffffu;
  GroupMemoryBarrierWithGroupSync();
  const uint index = key & 1023u;
  const uint wave  = key >> 10;

  // each sorted job's first band
  s_scan[tid] = valid ? uint( bandsOf( jobAt( index ) ) ) : 0u;
  GroupMemoryBarrierWithGroupSync();
  const uint total = scan1024( tid );
  const uint first = s_scan[tid];
  GroupMemoryBarrierWithGroupSync();

  // the first job of each job's wave (where its wave begins), and that job's first band
  s_scan[tid] = valid && wave != prevWave ? uint( tid ) : 0u;
  GroupMemoryBarrierWithGroupSync();
  for( int d = 1; d < 1024; d <<= 1 )
  {
    const uint v = tid >= d ? s_scan[tid - d] : 0u;
    GroupMemoryBarrierWithGroupSync();
    s_scan[tid] = max( s_scan[tid], v );
    GroupMemoryBarrierWithGroupSync();
  }
  const uint waveStart = s_scan[tid];
  GroupMemoryBarrierWithGroupSync();
  s_scan[tid] = first;
  GroupMemoryBarrierWithGroupSync();
  const uint wait = s_scan[waveStart];

  // the job's bands (as PLACE writes them) and their waits (as WAITS does)
  if( valid )
  {
    writeJobBands( jobAt( index ), first, true, wait );
  }
  if( tid == 0 )
  {
    storeBandTotal( total );
  }
}

// HIST: counts the block's jobs per value of this digit (pc.shift) into the histogram (pc.hist, digit-major), and flags
// a wave of 65536 or more (pc.flags[0]: too many for the sort) and one of 256 or more (pc.flags[1]: the second digit
// is needed).
void histPass( int tid, int block, uint i, uint n, uint* src )
{
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
}

// Turns the count values into their exclusive prefix sums, in place, by one workgroup: each thread sums a run of
// them, the workgroup scans the runs' sums, and each thread writes its run's sums. Returns the total.
uint prefixSumRuns( uint* values, int count, int tid )
{
  const int per = ( count + 1023 ) / 1024;
  const int a0  = min( tid * per, count );
  const int a1  = min( a0 + per, count );
  uint      sum = 0;
  for( int a = a0; a < a1; a++ )
  {
    sum += values[a];
  }
  s_scan[tid] = sum;
  GroupMemoryBarrierWithGroupSync();
  const uint total = scan1024( tid );
  uint       at    = s_scan[tid];
  for( int a = a0; a < a1; a++ )
  {
    const uint v = values[a];
    values[a]    = at;
    at += v;
  }
  return total;
}

// SCATTER by one thread, for waves of another size than 32: moves the block's jobs (base to end), in order, each to
// the next place of its digit value.
void scatterOneThread( int block, uint base, uint end, uint* src, uint* dst )
{
  for( uint j = base; j < end; j++ )
  {
    const uint index = pc.shift == 0 ? j : src[j];
    const uint digit = ( waveOf( index ) >> pc.shift ) & 255;
    const uint at    = pc.hist[digit * pc.numBlocks + block];
    pc.hist[digit * pc.numBlocks + block] = at + 1;
    dst[at]                               = index;
  }
}

// SCATTER by the first wave: moves the block's jobs (base to end) in order, 32 at a time. Each job goes to the next
// free place of its digit value after the jobs before it: those of the earlier chunks (the running places in s_hist)
// and the lower threads of its chunk with the same digit (ballots of the digit's bits find them).
void scatterByBallots( int tid, int block, uint base, uint end, uint* src, uint* dst )
{
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
}

// SCATTER: moves the block's jobs to their places in this digit's order. When the second digit is 0 for every job,
// the order stays and the keys are copied.
void scatterPass( int tid, int block, uint i, uint n, uint* src, uint* dst, bool secondDigitZero )
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
    if( tid == 0 )
    {
      scatterOneThread( block, base, end, src, dst );
    }
    return;
  }
  if( tid < 32 )
  {
    scatterByBallots( tid, block, base, end, src, dst );
  }
}

// BANDS and PLACE both start with each sorted job's bands, scanned over the block: s_scan then holds each job's first
// band in the block. Returns the block's total.
uint scanBlockBands( int tid, uint i, uint n )
{
  s_scan[tid] = i < n ? uint( bandsOf( jobAt( pc.keysA[i] ) ) ) : 0u;
  GroupMemoryBarrierWithGroupSync();
  return scan1024( tid );
}

// PLACE: writes each sorted job's bands at its place (the block's offset from BSCAN plus the job's in the block),
// records the job's first band (bandAt) and the first band of each wave where the wave begins (waveFirst).
void placeJob( int tid, int block, uint i )
{
  const uint index = pc.keysA[i];
  const uint first = pc.bandSums[block] + s_scan[tid];
  const uint wave  = waveOf( index );
  pc.bandAt[i]     = first;
  if( ( i == 0 || waveOf( pc.keysA[i - 1] ) != wave ) && wave < 65536u )
  {
    pc.waveFirst[wave] = first;
  }
  writeJobBands( jobAt( index ), first, false, 0u );   // (WAITS writes the waits once every wave's first band is known)
}

// WAITS: writes the wait of each of the sorted job's bands: the first band of the job's wave.
void writeJobWaits( uint i )
{
  const uint index = pc.keysA[i];
  const uint wave  = waveOf( index );
  const uint wait  = wave < 65536u ? pc.waveFirst[wave] : 0u;
  const int  b     = bandsOf( jobAt( index ) );
  for( int k = 0; k < b; k++ )
  {
    const uint at = pc.bandAt[i] + uint( k );
    if( at < uint( pc.capacity ) )
    {
      pc.waits[at] = wait;
    }
  }
}

[shader( "compute" )]
[numthreads( 1024, 1, 1 )]
void main( uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID )
{
  const int  tid   = int( local.x );
  const int  block = int( group.x );
  const uint n     = pc.count[0];
  const uint i     = uint( block * kBlock + tid );   // the thread's job in the block (in sorted order from BANDS on)
  if( pc.pass == kPassSmall )
  {
    smallSort( tid, n );
    return;
  }
  if( pc.pass <= kPassScatter )
  {
    // The radix sort's passes for this digit: its input and output keys (digit 0 reads the jobs in order). When the
    // second digit is 0 for every job, HIST and SCAN have nothing to do and SCATTER copies the keys.
    uint*      src             = pc.shift == 0 ? pc.keysA : pc.keysB;
    uint*      dst             = pc.shift == 0 ? pc.keysB : pc.keysA;
    const bool secondDigitZero = pc.shift != 0 && pc.flags[1].load() == 0;
    if( pc.pass == kPassScatter )
    {
      scatterPass( tid, block, i, n, src, dst, secondDigitZero );
    }
    else if( !secondDigitZero )
    {
      if( pc.pass == kPassHist )
      {
        histPass( tid, block, i, n, src );
      }
      else
      {
        prefixSumRuns( pc.hist, 256 * pc.numBlocks, tid );   // SCAN: the counts, digit-major, become places
      }
    }
    return;
  }
  // The band passes read the sorted jobs (keysA).
  if( pc.pass == kPassBands || pc.pass == kPassPlace )
  {
    const uint total = scanBlockBands( tid, i, n );
    if( pc.pass == kPassBands )
    {
      if( tid == 0 )
      {
        pc.bandSums[block] = total;
      }
    }
    else if( i < n )
    {
      placeJob( tid, block, i );
    }
    return;
  }
  if( pc.pass == kPassBandScan )
  {
    // BSCAN: the blocks' band counts become their offsets, and the total is stored.
    const uint total = prefixSumRuns( pc.bandSums, pc.numBlocks, tid );
    if( tid == 0 )
    {
      storeBandTotal( total );
    }
    return;
  }
  if( i < n )   // WAITS
  {
    writeJobWaits( i );
  }
}
