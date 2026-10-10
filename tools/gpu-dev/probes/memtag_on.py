# memtag_on.py <repo>: TEMPORARY measurement build. Every GPU buffer is tagged with the file:line that allocated it
# (or took it from a pool); with VVDEC_GPU_PROFILE=1 the decoder prints, for each heap, the live bytes per tag at
# that heap's peak and each tag's own maximum. Restore the files with git checkout afterwards.
import sys

root = sys.argv[1] + '/source/Lib/VulkanLib/'


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


patch('GpuContext.h', [
    ('#include <atomic>\n', '#include <atomic>\n#include <map>\n'),
    ('''  Buffer alloc( size_t size, MemKind kind );
  void   free( Buffer& buf );''', '''  Buffer alloc( size_t size, MemKind kind, const char* file = __builtin_FILE(), int line = __builtin_LINE() );
  void   free( Buffer& buf );
  void   tagMemory( const Buffer& buf, const char* file, int line );'''),
    ('''  Buffer allocPooled( size_t size );''',
     '''  Buffer allocPooled( size_t size, const char* file = __builtin_FILE(), int line = __builtin_LINE() );'''),
    ('''  std::atomic<int64_t>             m_memoryPeak[2] = { { 0 }, { 0 } };''',
     '''  std::atomic<int64_t>             m_memoryPeak[2] = { { 0 }, { 0 } };
  struct MemTag
  {
    std::string label;
    int64_t     bytes;
    int         heap;
  };
  std::mutex                                     m_tagMutex;
  std::map<VkDeviceMemory, MemTag>               m_tags;
  std::map<std::string, std::pair<int64_t, int>> m_byLabel[2], m_peakByLabel[2];
  std::map<std::string, int64_t>                 m_maxByLabel[2];
  int64_t                                        m_tagNow[2] = {}, m_tagPeak[2] = {};
  void tagAdd( VkDeviceMemory mem, const std::string& label, int64_t bytes, int heap );
  void tagRemove( VkDeviceMemory mem );
  void printTags();'''),
])

patch('GpuContext.cpp', [
    ('''Buffer Context::allocPooled( size_t size )
{''', '''static std::string tagLabel( const char* file, int line )
{
  std::string f = file;
  const size_t slash = f.find_last_of( "/\\\\" );
  if( slash != std::string::npos )
  {
    f = f.substr( slash + 1 );
  }
  return line > 0 ? f + ":" + std::to_string( line ) : f;
}

void Context::tagAdd( VkDeviceMemory mem, const std::string& label, int64_t bytes, int heap )
{
  std::lock_guard<std::mutex> lock( m_tagMutex );
  m_tags[mem]      = MemTag{ label, bytes, heap };
  auto& e          = m_byLabel[heap][label];
  e.first         += bytes;
  e.second        += 1;
  int64_t& mx      = m_maxByLabel[heap][label];
  mx               = std::max( mx, e.first );
  m_tagNow[heap]  += bytes;
  if( m_tagNow[heap] > m_tagPeak[heap] )
  {
    m_tagPeak[heap]     = m_tagNow[heap];
    m_peakByLabel[heap] = m_byLabel[heap];
  }
}

void Context::tagRemove( VkDeviceMemory mem )
{
  std::lock_guard<std::mutex> lock( m_tagMutex );
  auto it = m_tags.find( mem );
  if( it == m_tags.end() )
  {
    return;
  }
  auto& e         = m_byLabel[it->second.heap][it->second.label];
  e.first        -= it->second.bytes;
  e.second       -= 1;
  m_tagNow[it->second.heap] -= it->second.bytes;
  m_tags.erase( it );
}

void Context::tagMemory( const Buffer& buf, const char* file, int line )
{
  std::lock_guard<std::mutex> lock( m_tagMutex );
  auto it = m_tags.find( buf.memory );
  if( it == m_tags.end() )
  {
    return;
  }
  const int heap = it->second.heap;
  auto&     o    = m_byLabel[heap][it->second.label];
  o.first       -= it->second.bytes;
  o.second      -= 1;
  it->second.label = tagLabel( file, line );
  auto& e          = m_byLabel[heap][it->second.label];
  e.first         += it->second.bytes;
  e.second        += 1;
  int64_t& mx      = m_maxByLabel[heap][it->second.label];
  mx               = std::max( mx, e.first );
}

void Context::printTags()
{
  std::lock_guard<std::mutex> lock( m_tagMutex );
  for( int heap = 0; heap < 2; heap++ )
  {
    std::vector<std::pair<std::string, std::pair<int64_t, int>>> v( m_peakByLabel[heap].begin(), m_peakByLabel[heap].end() );
    std::sort( v.begin(), v.end(), []( const auto& a, const auto& b ) { return a.second.first > b.second.first; } );
    msg( INFO, "MEMTAG %s peak %.1f MB, at that peak (MB, buffers, label; then the label's own maximum):\\n", heap == 0 ? "device-local" : "host-visible",
         m_tagPeak[heap] / 1048576.0 );
    for( const auto& e: v )
    {
      if( e.second.first > 0 )
      {
        msg( INFO, "MEMTAG   %9.2f %5d  %-34s max %9.2f\\n", e.second.first / 1048576.0, e.second.second, e.first.c_str(),
             m_maxByLabel[heap][e.first] / 1048576.0 );
      }
    }
    for( const auto& e: m_maxByLabel[heap] )
    {
      if( m_peakByLabel[heap].count( e.first ) == 0 || m_peakByLabel[heap][e.first].first == 0 )
      {
        msg( INFO, "MEMTAG   %9.2f     0  %-34s max %9.2f (not at the peak)\\n", 0.0, e.first.c_str(), e.second / 1048576.0 );
      }
    }
  }
}

Buffer Context::allocPooled( size_t size, const char* file, int line )
{'''),
    ('''        const Buffer buf = *it;
        m_pool.erase( it );
        return buf;''', '''        const Buffer buf = *it;
        m_pool.erase( it );
        tagMemory( buf, file, line );
        return buf;'''),
    ('''  return alloc( size, MemKind::Device );
}''', '''  return alloc( size, MemKind::Device, file, line );
}'''),
    ('''  std::lock_guard<std::mutex> lock( m_poolMutex );
  m_pool.push_back( buf );''', '''  std::lock_guard<std::mutex> lock( m_poolMutex );
  tagMemory( buf, "pool (idle)", 0 );
  m_pool.push_back( buf );'''),
    ('''           m_memoryNow[0] / 1048576.0, m_memoryPeak[1] / 1048576.0, m_memoryNow[1] / 1048576.0 );''',
     '''           m_memoryNow[0] / 1048576.0, m_memoryPeak[1] / 1048576.0, m_memoryNow[1] / 1048576.0 );
      printTags();'''),
    ('''Buffer Context::alloc( size_t size, MemKind kind )''',
     '''Buffer Context::alloc( size_t size, MemKind kind, const char* file, int line )'''),
    ('''    return alloc( size, MemKind::HostCached );''', '''    return alloc( size, MemKind::HostCached, file, line );'''),
    ('''  countMemory( buf.kind, int64_t( req.size ) );''',
     '''  tagAdd( buf.memory, tagLabel( file, line ), int64_t( req.size ), buf.kind == MemKind::Device ? 0 : 1 );
  countMemory( buf.kind, int64_t( req.size ) );'''),
    ('''    countMemory( buf.kind, -int64_t( buf.memorySize ) );''', '''    tagRemove( buf.memory );
    countMemory( buf.kind, -int64_t( buf.memorySize ) );'''),
])

patch('GpuParseService.h', [
    ('''  Buffer         takeBuffer( size_t size );''',
     '''  Buffer         takeBuffer( size_t size, const char* file = __builtin_FILE(), int line = __builtin_LINE() );'''),
])

patch('GpuParseService.cpp', [
    ('''static void ensureBuffer( Context* ctx, Buffer& buf, size_t size, MemKind kind )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( ( size + 65535 ) & ~size_t( 65535 ), kind );''',
     '''static void ensureBuffer( Context* ctx, Buffer& buf, size_t size, MemKind kind, const char* file = __builtin_FILE(), int line = __builtin_LINE() )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( ( size + 65535 ) & ~size_t( 65535 ), kind, file, line );'''),
    ('''Buffer ParseService::takeBuffer( size_t size )''', '''Buffer ParseService::takeBuffer( size_t size, const char* file, int line )'''),
    ('''    const Buffer buf = m_bins[bin].back();
    m_bins[bin].pop_back();
    return buf;''', '''    const Buffer buf = m_bins[bin].back();
    m_bins[bin].pop_back();
    m_ctx->tagMemory( buf, file, line );
    return buf;'''),
    ('''  return m_ctx->alloc( size_t( 1 ) << bin, MemKind::Device );''',
     '''  return m_ctx->alloc( size_t( 1 ) << bin, MemKind::Device, file, line );'''),
    ('''  m_bins[bin].push_back( buf );''', '''  m_ctx->tagMemory( buf, "parse bins (idle)", 0 );
  m_bins[bin].push_back( buf );'''),
])

patch('GpuReconFrontEnd.cpp', [
    ('''static void ensureBuffer( Context* ctx, Buffer& buf, size_t size, MemKind kind )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( ( size + 65535 ) & ~size_t( 65535 ), kind );''',
     '''static void ensureBuffer( Context* ctx, Buffer& buf, size_t size, MemKind kind, const char* file = __builtin_FILE(), int line = __builtin_LINE() )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( ( size + 65535 ) & ~size_t( 65535 ), kind, file, line );'''),
])

patch('GpuReconParse.cpp', [
    ('''static void ensureParseBuffer( Context* ctx, Buffer& buf, size_t size, MemKind kind )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( ( size + 65535 ) & ~size_t( 65535 ), kind );''',
     '''static void ensureParseBuffer( Context* ctx, Buffer& buf, size_t size, MemKind kind, const char* file = __builtin_FILE(), int line = __builtin_LINE() )
{
  if( buf.size < size )
  {
    ctx->free( buf );
    buf = ctx->alloc( ( size + 65535 ) & ~size_t( 65535 ), kind, file, line );'''),
])
print('memtag: GPU buffers tagged by where they were allocated')
