// procstat [--idle <seconds>] [--mark <text>]... -- <command line...>
// Runs the command and reports, on one line (PROCSTAT ...):
//   wall time, CPU user and kernel time, peak working set and peak private commit (MB), page faults,
//   the RAPL energy (J) of the package and of the cores over the run (Energy Meter counters, in pWh),
//   and the time from the launch to the first output line containing each --mark text, with the package's and the
//   cores' energy (J) used by then (markN_pkg_j, markN_cores_j: power between two marks leaves start-up out).
// --idle <seconds>: first measures the package's idle power over that time (printed as IDLE ... W).
// The child's output is passed through.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#pragma comment( lib, "pdh.lib" )
#pragma comment( lib, "psapi.lib" )

static double now( LARGE_INTEGER t0 )
{
  LARGE_INTEGER f, t;
  QueryPerformanceFrequency( &f );
  QueryPerformanceCounter( &t );
  return double( t.QuadPart - t0.QuadPart ) * 1000.0 / double( f.QuadPart );
}

struct Energy
{
  PDH_HQUERY   query = nullptr;
  PDH_HCOUNTER pkg   = nullptr;
  PDH_HCOUNTER cores[16]{};
  int          numCores = 0;

  bool open()
  {
    if( PdhOpenQueryW( nullptr, 0, &query ) != ERROR_SUCCESS )
      return false;
    if( PdhAddEnglishCounterW( query, L"\\Energy Meter(RAPL_Package0_PKG)\\Energy", 0, &pkg ) != ERROR_SUCCESS )
      return false;
    for( int c = 0; c < 16; c++ )
    {
      wchar_t path[128];
      swprintf( path, 128, L"\\Energy Meter(RAPL_Package0_Core%d_CORE)\\Energy", c );
      if( PdhAddEnglishCounterW( query, path, 0, &cores[numCores] ) == ERROR_SUCCESS )
        numCores++;
    }
    return true;
  }
  // the package's and the cores' energy so far, in joules
  bool read( double& package, double& coreSum )
  {
    if( !query || PdhCollectQueryData( query ) != ERROR_SUCCESS )
      return false;
    PDH_RAW_COUNTER raw;
    DWORD           type;
    if( PdhGetRawCounterValue( pkg, &type, &raw ) != ERROR_SUCCESS )
      return false;
    package = double( raw.FirstValue ) * 3.6e-9;   // pWh
    coreSum = 0;
    for( int c = 0; c < numCores; c++ )
    {
      if( PdhGetRawCounterValue( cores[c], &type, &raw ) == ERROR_SUCCESS )
        coreSum += double( raw.FirstValue ) * 3.6e-9;
    }
    return true;
  }
};

int wmain( int argc, wchar_t** argv )
{
  int                       idleSeconds = 0;
  std::vector<std::string>  marks;
  int                       first = 1;
  for( ; first < argc; first++ )
  {
    if( wcscmp( argv[first], L"--idle" ) == 0 && first + 1 < argc )
      idleSeconds = _wtoi( argv[++first] );
    else if( wcscmp( argv[first], L"--mark" ) == 0 && first + 1 < argc )
    {
      char buf[256];
      WideCharToMultiByte( CP_UTF8, 0, argv[++first], -1, buf, sizeof( buf ), nullptr, nullptr );
      marks.push_back( buf );
    }
    else if( wcscmp( argv[first], L"--" ) == 0 )
    {
      first++;
      break;
    }
    else
      break;
  }
  if( first >= argc )
  {
    fprintf( stderr, "usage: procstat [--idle <seconds>] [--mark <text>]... -- <command line...>\n" );
    return 2;
  }
  std::wstring cmd;
  for( int i = first; i < argc; i++ )
  {
    if( i > first )
      cmd += L' ';
    cmd += L'"';
    cmd += argv[i];
    cmd += L'"';
  }

  Energy energy;
  const bool haveEnergy = energy.open();
  double     e0 = 0, c0 = 0, e1 = 0, c1 = 0;
  if( haveEnergy && idleSeconds > 0 )
  {
    LARGE_INTEGER t0;
    QueryPerformanceCounter( &t0 );
    double ia, ib, ja, jb;
    energy.read( ia, ja );
    Sleep( DWORD( idleSeconds * 1000 ) );
    energy.read( ib, jb );
    const double s = now( t0 ) / 1000.0;
    printf( "IDLE package %.2f W cores %.2f W\n", ( ib - ia ) / s, ( jb - ja ) / s );
  }

  SECURITY_ATTRIBUTES sa{ sizeof( sa ), nullptr, TRUE };
  HANDLE              readPipe, writePipe;
  CreatePipe( &readPipe, &writePipe, &sa, 1 << 16 );
  SetHandleInformation( readPipe, HANDLE_FLAG_INHERIT, 0 );
  STARTUPINFOW si{ sizeof( si ) };
  si.dwFlags    = STARTF_USESTDHANDLES;
  si.hStdInput  = GetStdHandle( STD_INPUT_HANDLE );
  si.hStdOutput = writePipe;
  si.hStdError  = writePipe;
  PROCESS_INFORMATION pi{};

  if( haveEnergy )
    energy.read( e0, c0 );
  LARGE_INTEGER t0;
  QueryPerformanceCounter( &t0 );
  if( !CreateProcessW( nullptr, &cmd[0], nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi ) )
  {
    fprintf( stderr, "procstat: CreateProcess failed (%lu)\n", GetLastError() );
    return 2;
  }
  CloseHandle( writePipe );

  std::vector<double> markAt( marks.size(), -1.0 ), markPkg( marks.size(), -1.0 ), markCores( marks.size(), -1.0 );
  std::string         line;
  char                buf[4096];
  DWORD               got;
  while( ReadFile( readPipe, buf, sizeof( buf ), &got, nullptr ) && got > 0 )
  {
    const double t = now( t0 );
    fwrite( buf, 1, got, stdout );
    for( DWORD i = 0; i < got; i++ )
    {
      if( buf[i] == '\n' )
      {
        for( size_t m = 0; m < marks.size(); m++ )
        {
          if( markAt[m] < 0 && line.find( marks[m] ) != std::string::npos )
          {
            markAt[m] = t;
            double pk = 0, co = 0;
            if( haveEnergy && energy.read( pk, co ) )
            {
              markPkg[m]   = pk - e0;
              markCores[m] = co - c0;
            }
          }
        }
        line.clear();
      }
      else
        line += buf[i];
    }
  }
  WaitForSingleObject( pi.hProcess, INFINITE );
  const double wall = now( t0 );
  if( haveEnergy )
    energy.read( e1, c1 );

  FILETIME c, e, k, u;
  GetProcessTimes( pi.hProcess, &c, &e, &k, &u );
  const auto ms = []( FILETIME t ) { return double( ULARGE_INTEGER{ { t.dwLowDateTime, t.dwHighDateTime } }.QuadPart ) / 1e4; };
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof( pmc );
  GetProcessMemoryInfo( pi.hProcess, (PROCESS_MEMORY_COUNTERS*) &pmc, sizeof( pmc ) );
  DWORD code = 0;
  GetExitCodeProcess( pi.hProcess, &code );

  printf( "PROCSTAT exit %lu wall_ms %.1f user_ms %.1f kernel_ms %.1f peak_ws_mb %.1f peak_commit_mb %.1f page_faults %lu", code, wall, ms( u ),
          ms( k ), pmc.PeakWorkingSetSize / 1048576.0, pmc.PeakPagefileUsage / 1048576.0, pmc.PageFaultCount );
  if( haveEnergy )
    printf( " pkg_j %.2f cores_j %.2f", e1 - e0, c1 - c0 );
  for( size_t m = 0; m < marks.size(); m++ )
    printf( " mark%zu_ms %.1f mark%zu_pkg_j %.3f mark%zu_cores_j %.3f", m, markAt[m], m, markPkg[m], m, markCores[m] );
  printf( "\n" );
  CloseHandle( pi.hProcess );
  CloseHandle( pi.hThread );
  return int( code );
}
