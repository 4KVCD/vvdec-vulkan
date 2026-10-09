// Helpers shared by the shaders. Every buffer is reached through a device
// address in the push constants; samples are 16-bit (vvdec's Pel).

int clip3( int lo, int hi, int v )
{
  return min( max( v, lo ), hi );
}

// a plane of 16-bit samples: (0,0) at p, rows stride samples apart
int loadPel( int16_t* p, int stride, int x, int y )
{
  return int( p[y * stride + x] );
}

void storePel( int16_t* p, int stride, int x, int y, int v )
{
  p[y * stride + x] = int16_t( v );
}

// the sample at (x, y) with the coordinates clamped into the picture: what
// vvdec's border extension of the picture gives
int loadClamped( int16_t* p, int stride, int x, int y, int w, int h )
{
  return int( p[clamp( y, 0, h - 1 ) * stride + clamp( x, 0, w - 1 )] );
}

// the workgroup's threads synchronized, their writes to global (and shared) memory visible to each other: a
// workgroup-scope barrier, for a workgroup's own data (AllMemoryBarrierWithGroupSync's device scope, which other
// workgroups' data needs, also invalidates the CU's caches on AMD's GPUs)
void workgroupBarrierAll()
{
  spirv_asm { OpControlBarrier Workgroup Workgroup AcquireRelease|UniformMemory|WorkgroupMemory };
}
