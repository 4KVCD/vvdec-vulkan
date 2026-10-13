// Helpers shared by the shaders. Every buffer is reached through a device
// address in the push constants; samples are 16-bit (vvdec's Pel).

int clip3( int lo, int hi, int v )
{
  return min( max( v, lo ), hi );
}

// p advanced by index elements, for a pointer the same in all the threads (a push constant's, or uniPtr's) and an
// offset below 2 GB, as every buffer of the decoder's is. So written, AMD's compiler adds the 32-bit offset to the
// pointer in the load or store instruction itself (scalar-base addressing); p + index makes each thread work out a
// 64-bit address instead (a sign extension, a shift and two adds with carry).
T* at<T>( T* p, uint index )
{
  return (T*)( (uint8_t*)p + ( ( index * uint( sizeof( T ) ) ) & 0x7fffffffu ) );
}

// A pointer the same in all the threads, made one the compiler keeps in scalar registers: for a pointer read from
// memory, which the compiler reads with vector loads once the shader may have written memory before (at() could not
// use it as a scalar base then).
T* uniPtr<T>( T* p )
{
  return (T*)WaveReadLaneFirst( (uint64_t)p );
}

// Sample access in a plane of 16-bit samples: (0, 0) is at p, and rows are stride samples apart; (x, y) lies in the
// plane (at()'s offsets are unsigned).
int loadPel( int16_t* p, int stride, int x, int y )
{
  return int( *at( p, uint( y * stride + x ) ) );
}

void storePel( int16_t* p, int stride, int x, int y, int v )
{
  *at( p, uint( y * stride + x ) ) = int16_t( v );
}

// The signed 16-bit halves of a 32-bit word
int lo16( uint v )
{
  return int( v << 16 ) >> 16;
}

int hi16( uint v )
{
  return int( v ) >> 16;
}

// Two 16-bit samples as a word, the first in its low half
uint packPair( int2 v )
{
  return uint( v.x & 0xffff ) | uint( v.y ) << 16;
}

// Four 16-bit samples as two words, the first in the low half of the first word
uint2 packSamples( int4 v )
{
  return uint2( packPair( v.xy ), packPair( v.zw ) );
}

// The four signed 16-bit samples of two words (packSamples' inverse)
int4 unpackSamples( uint2 w )
{
  return int4( lo16( w.x ), hi16( w.x ), lo16( w.y ), hi16( w.y ) );
}

// Reads the sample at (x, y) with the coordinates clamped into the picture,
// which returns what vvdec's border extension of the picture holds there.
int loadClamped( int16_t* p, int stride, int x, int y, int w, int h )
{
  return int( *at( p, uint( clamp( y, 0, h - 1 ) * stride + clamp( x, 0, w - 1 ) ) ) );
}

// A dispatch of a workgroup per job (or per entry, per record...) runs its workgroups in rows of kGroupsPerRow:
// GpuRecon::dispatchJobs, the front end's indirect dispatches (storeJobPassArgs) and GpuParseService's dispatch.
static const uint kGroupsPerRow = 32768;

// The index of the workgroup at group in such a dispatch: its job
uint workgroupIndex( uint3 group )
{
  return group.x + group.y * kGroupsPerRow;
}

// The front end's indirect arguments (GpuRecon's m_frontEnd.args) hold, for pass p of GpuRecon.h's FrontPass, the
// pass's workgroups (x, y, z) at word 3 * p, and its job count and first job at words kFrontCounts + 2 * p and the next.
static const int kFrontCounts        = 32;
static const int kPassTrafo          = 0;
static const int kPassInterAffine    = 3;   // (after kPassInter and kPassInterBdof)
static const int kPassIntra          = 4;
static const int kPassDeblockLumaVer = 5;   // the first of the four deblocking passes, in edges.slang's list order

// Stores pass's arguments: groupsX x groupsY workgroups, which take count jobs from the job first on
void storePassArgs( uint* args, int pass, uint groupsX, uint groupsY, uint count, uint first )
{
  args[3 * pass]                    = groupsX;
  args[3 * pass + 1]                = groupsY;
  args[3 * pass + 2]                = 1;
  args[kFrontCounts + 2 * pass]     = count;
  args[kFrontCounts + 2 * pass + 1] = first;
}

// Stores the arguments of pass, which runs a workgroup per job (in rows of kGroupsPerRow): count jobs from the job
// first on
void storeJobPassArgs( uint* args, int pass, uint count, uint first )
{
  storePassArgs( args, pass, min( count, kGroupsPerRow ), ( count + kGroupsPerRow - 1 ) / kGroupsPerRow, count, first );
}

// Synchronizes the workgroup's threads and makes their writes to global and shared memory visible to each other.
// The barrier has workgroup scope, which suffices for data shared within the workgroup. Data written by other
// workgroups would need AllMemoryBarrierWithGroupSync's device scope, which on AMD GPUs also invalidates the
// compute unit's caches.
void workgroupBarrierAll()
{
  spirv_asm { OpControlBarrier Workgroup Workgroup AcquireRelease|UniformMemory|WorkgroupMemory };
}
