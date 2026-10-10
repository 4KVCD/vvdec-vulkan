// The prefix sums of a workgroup of 256 threads (jobs.slang, edges.slang): each wave sums its threads' values with
// WavePrefixSum, then each thread adds the totals of the waves before its own. Two barriers instead of a thread
// walking 256 entries of shared memory one after another.

groupshared uint s_waveSums[32];   // a wave's total (up to 32 waves)

// The exclusive prefix sum of v over the workgroup's 256 threads in thread order (tid: the thread's index), and in
// total the sum of all their values. All the threads call it, in uniform control flow.
uint workgroupPrefixSum( uint v, int tid, out uint total )
{
  const uint lanes  = WaveGetLaneCount();
  const uint wave   = uint( tid ) / lanes;
  const uint inWave = WavePrefixSum( v );
  if( uint( tid ) % lanes == lanes - 1 )
  {
    s_waveSums[wave] = inWave + v;
  }
  GroupMemoryBarrierWithGroupSync();
  uint before = 0;
  total       = 0;
  for( uint w = 0; w < 256u / lanes; w++ )
  {
    const uint s = s_waveSums[w];
    before += w < wave ? s : 0u;
    total += s;
  }
  GroupMemoryBarrierWithGroupSync();   // (s_waveSums free for the next call)
  return before + inWave;
}
