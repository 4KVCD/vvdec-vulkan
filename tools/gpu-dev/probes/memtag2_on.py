# memtag2_on.py <repo>: after memtag_on.py, the host planes' tags carry their size (planes and colocated motion share
# that allocator). TEMPORARY measurement build.
import sys

root = sys.argv[1] + '/source/Lib/VulkanLib/'
path = root + 'GpuPictureMemory.cpp'
s = open(path, encoding='utf-8', newline='').read()
old = '  plane->host        = context->alloc( size, MemKind::HostCached );'
new = '''  const std::string label = "hostplane " + std::to_string( size / 1024 ) + " KB";
  plane->host        = context->alloc( size, MemKind::HostCached, label.c_str(), 0 );'''
assert s.count(old) == 1
s = s.replace(old, new)
open(path, 'w', encoding='utf-8', newline='').write(s)
print('memtag2: host planes tagged by size')
