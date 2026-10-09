// Helpers shared by the shaders. Every buffer is reached through a device
// address in the push constants; samples are 16-bit (vvdec's Pel).

int clip3( int lo, int hi, int v )
{
  return min( max( v, lo ), hi );
}

// Sample access in a plane of 16-bit samples: (0, 0) is at p, and rows are stride samples apart.
int loadPel( int16_t* p, int stride, int x, int y )
{
  return int( p[y * stride + x] );
}

void storePel( int16_t* p, int stride, int x, int y, int v )
{
  p[y * stride + x] = int16_t( v );
}

// Reads the sample at (x, y) with the coordinates clamped into the picture,
// which returns what vvdec's border extension of the picture holds there.
int loadClamped( int16_t* p, int stride, int x, int y, int w, int h )
{
  return int( p[clamp( y, 0, h - 1 ) * stride + clamp( x, 0, w - 1 )] );
}

// Synchronizes the workgroup's threads and makes their writes to global and shared memory visible to each other.
// The barrier has workgroup scope, which suffices for data shared within the workgroup. Data written by other
// workgroups would need AllMemoryBarrierWithGroupSync's device scope, which on AMD GPUs also invalidates the
// compute unit's caches.
void workgroupBarrierAll()
{
  spirv_asm { OpControlBarrier Workgroup Workgroup AcquireRelease|UniformMemory|WorkgroupMemory };
}
