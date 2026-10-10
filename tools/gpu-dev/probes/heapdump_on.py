# heapdump_on.py <repo>: TEMPORARY measurement build. With VVDEC_HEAPDUMP=<n>, vvdecapp walks the process heaps and
# the private memory regions once it has output frame n, and prints the allocations grouped by size (the sizes
# holding the most bytes first) and the committed private regions grouped by size and protection.
import sys

root = sys.argv[1] + '/source/App/vvdecapp/'


def patch(path, subs):
    s = open(root + path, encoding='utf-8', newline='').read()
    crlf = '\r\n' in s
    s = s.replace('\r\n', '\n')
    for old, new in subs:
        assert s.count(old) == 1, (path, old[:80], s.count(old))
        s = s.replace(old, new)
    if crlf:
        s = s.replace('\n', '\r\n')
    open(root + path, 'w', encoding='utf-8', newline='').write(s)


patch('vvdecapp.cpp', [
    ('''#include <io.h>
#include <fcntl.h>
#endif
''', '''#include <io.h>
#include <fcntl.h>
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <map>
#include <vector>
#include <algorithm>
static unsigned heapDumpAt()
{
  char v[32] = {};
  return GetEnvironmentVariableA( "VVDEC_HEAPDUMP", v, sizeof( v ) ) ? unsigned( atoi( v ) ) : 0u;
}
static void dumpHeapHistogram()
{
  HANDLE heaps[256];
  const DWORD n = GetProcessHeaps( 256, heaps );
  std::map<size_t, std::pair<size_t, size_t>> hist;   // size -> count, bytes
  size_t total = 0, blocks = 0;
  for( DWORD h = 0; h < n && h < 256; h++ )
  {
    if( !HeapLock( heaps[h] ) )
    {
      continue;
    }
    PROCESS_HEAP_ENTRY e;
    e.lpData = nullptr;
    while( HeapWalk( heaps[h], &e ) )
    {
      if( e.wFlags & PROCESS_HEAP_ENTRY_BUSY )
      {
        auto& x = hist[e.cbData];
        x.first++;
        x.second += e.cbData;
        total += e.cbData;
        blocks++;
      }
    }
    HeapUnlock( heaps[h] );
  }
  std::vector<std::pair<size_t, std::pair<size_t, size_t>>> v( hist.begin(), hist.end() );
  std::sort( v.begin(), v.end(), []( const auto& a, const auto& b ) { return a.second.second > b.second.second; } );
  fprintf( stderr, "HEAPDUMP %u heaps, %zu blocks, %.1f MB in use\\n", unsigned( n ), blocks, total / 1048576.0 );
  for( size_t i = 0; i < v.size() && i < 45; i++ )
  {
    fprintf( stderr, "HEAPDUMP   size %10zu x %6zu = %8.2f MB\\n", v[i].first, v[i].second.first, v[i].second.second / 1048576.0 );
  }
  // the committed private regions (VirtualAlloc'd memory, the heaps' segments, the GPU driver's mappings)
  std::map<std::pair<size_t, DWORD>, std::pair<size_t, size_t>> regions;   // (size, protect) -> count, bytes
  size_t           privTotal = 0;
  MEMORY_BASIC_INFORMATION mbi;
  for( char* p = nullptr; VirtualQuery( p, &mbi, sizeof( mbi ) ) == sizeof( mbi ); p = (char*) mbi.BaseAddress + mbi.RegionSize )
  {
    if( mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE )
    {
      auto& x = regions[{ mbi.RegionSize, mbi.Protect }];
      x.first++;
      x.second += mbi.RegionSize;
      privTotal += mbi.RegionSize;
    }
    if( (char*) mbi.BaseAddress + mbi.RegionSize < p )
    {
      break;
    }
  }
  std::vector<std::pair<std::pair<size_t, DWORD>, std::pair<size_t, size_t>>> r( regions.begin(), regions.end() );
  std::sort( r.begin(), r.end(), []( const auto& a, const auto& b ) { return a.second.second > b.second.second; } );
  fprintf( stderr, "HEAPDUMP private committed regions: %.1f MB\\n", privTotal / 1048576.0 );
  for( size_t i = 0; i < r.size() && i < 30; i++ )
  {
    fprintf( stderr, "HEAPDUMP   region %10zu prot 0x%03lx x %6zu = %8.2f MB\\n", r[i].first.first, (unsigned long) r[i].first.second, r[i].second.first,
             r[i].second.second / 1048576.0 );
  }
}
#endif
'''),
    ('''    uiFrames++;
    uiFramesTmp++;
''', '''    uiFrames++;
    uiFramesTmp++;
#if defined (_WIN32) || defined (WIN32) || defined (_WIN64) || defined (WIN64)
    if( uiFrames == heapDumpAt() )
    {
      dumpHeapHistogram();
    }
#endif
'''),
])
print('heapdump: vvdecapp prints a heap histogram at frame VVDEC_HEAPDUMP')
