// uiprobe: what the desktop sees of the GPU while something else runs on it. Every 1/60 s it submits, on the first
// graphics queue, a fill of 8 MB (about one 1080p frame of the compositor's), and times each submission from vkQueueSubmit
// to its fence: a compositor whose work waits behind another program's makes the UI lag. At the end it prints the
// latencies' median, 95th and 99th percentiles and maximum, and how many submissions missed a 60 Hz frame.
//
//   uiprobe <seconds> [device index]
#include <vulkan/vulkan.h>
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define LOAD( name ) auto name = (PFN_##name) gipa( inst, #name )

int main( int argc, char** argv )
{
  const double seconds = argc >= 2 ? atof( argv[1] ) : 30.0;
  const int    index   = argc >= 3 ? atoi( argv[2] ) : 0;
  HMODULE      lib     = LoadLibraryA( "vulkan-1.dll" );
  if( !lib )
  {
    return 1;
  }
  auto              gipa           = (PFN_vkGetInstanceProcAddr) GetProcAddress( lib, "vkGetInstanceProcAddr" );
  auto              createInstance = (PFN_vkCreateInstance) gipa( nullptr, "vkCreateInstance" );
  VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
  ici.pApplicationInfo = &app;
  VkInstance inst;
  if( createInstance( &ici, nullptr, &inst ) != VK_SUCCESS )
  {
    return 1;
  }
  LOAD( vkEnumeratePhysicalDevices );
  LOAD( vkGetPhysicalDeviceQueueFamilyProperties );
  LOAD( vkGetPhysicalDeviceMemoryProperties );
  LOAD( vkGetPhysicalDeviceProperties );
  LOAD( vkCreateDevice );
  LOAD( vkGetDeviceQueue );
  LOAD( vkCreateBuffer );
  LOAD( vkGetBufferMemoryRequirements );
  LOAD( vkAllocateMemory );
  LOAD( vkBindBufferMemory );
  LOAD( vkCreateCommandPool );
  LOAD( vkAllocateCommandBuffers );
  LOAD( vkBeginCommandBuffer );
  LOAD( vkCmdFillBuffer );
  LOAD( vkEndCommandBuffer );
  LOAD( vkCreateFence );
  LOAD( vkQueueSubmit );
  LOAD( vkWaitForFences );
  LOAD( vkResetFences );
  LOAD( vkDeviceWaitIdle );
  LOAD( vkDestroyDevice );

  uint32_t n = 0;
  vkEnumeratePhysicalDevices( inst, &n, nullptr );
  std::vector<VkPhysicalDevice> devs( n );
  vkEnumeratePhysicalDevices( inst, &n, devs.data() );
  if( index < 0 || index >= int( n ) )
  {
    return 1;
  }
  VkPhysicalDevice phys = devs[index];
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties( phys, &props );
  vkGetPhysicalDeviceQueueFamilyProperties( phys, &n, nullptr );
  std::vector<VkQueueFamilyProperties> fams( n );
  vkGetPhysicalDeviceQueueFamilyProperties( phys, &n, fams.data() );
  uint32_t family = UINT32_MAX;
  for( uint32_t i = 0; i < n && family == UINT32_MAX; i++ )
  {
    if( fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT )
    {
      family = i;
    }
  }
  if( family == UINT32_MAX )
  {
    return 1;
  }
  const float             priority = 1.0f;
  VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
  qci.queueFamilyIndex = family;
  qci.queueCount       = 1;
  qci.pQueuePriorities = &priority;
  VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos    = &qci;
  VkDevice dev;
  if( vkCreateDevice( phys, &dci, nullptr, &dev ) != VK_SUCCESS )
  {
    return 1;
  }
  VkQueue queue;
  vkGetDeviceQueue( dev, family, 0, &queue );

  const VkDeviceSize size = 8u << 20;
  VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bci.size  = size;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VkBuffer buf;
  vkCreateBuffer( dev, &bci, nullptr, &buf );
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements( dev, buf, &req );
  VkPhysicalDeviceMemoryProperties mem;
  vkGetPhysicalDeviceMemoryProperties( phys, &mem );
  uint32_t type = 0;
  while( type < mem.memoryTypeCount
         && !( ( req.memoryTypeBits >> type & 1 ) && ( mem.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ) ) )
  {
    type++;
  }
  VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
  mai.allocationSize  = req.size;
  mai.memoryTypeIndex = type;
  VkDeviceMemory memory;
  if( type == mem.memoryTypeCount || vkAllocateMemory( dev, &mai, nullptr, &memory ) != VK_SUCCESS )
  {
    return 1;
  }
  vkBindBufferMemory( dev, buf, memory, 0 );

  VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
  pci.queueFamilyIndex = family;
  VkCommandPool pool;
  vkCreateCommandPool( dev, &pci, nullptr, &pool );
  VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  cai.commandPool        = pool;
  cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  vkAllocateCommandBuffers( dev, &cai, &cmd );
  VkCommandBufferBeginInfo cbi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  vkBeginCommandBuffer( cmd, &cbi );
  vkCmdFillBuffer( cmd, buf, 0, size, 0x12345678u );
  vkEndCommandBuffer( cmd );
  VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  VkFence fence;
  vkCreateFence( dev, &fci, nullptr, &fence );

  using Clock          = std::chrono::steady_clock;
  const auto period    = std::chrono::microseconds( 16667 );
  const auto start     = Clock::now();
  auto       next      = start;
  std::vector<double> latencies;
  while( Clock::now() - start < std::chrono::duration<double>( seconds ) )
  {
    std::this_thread::sleep_until( next );
    next += period;
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    const auto t0         = Clock::now();
    vkQueueSubmit( queue, 1, &si, fence );
    vkWaitForFences( dev, 1, &fence, VK_TRUE, UINT64_MAX );
    latencies.push_back( std::chrono::duration<double, std::milli>( Clock::now() - t0 ).count() );
    vkResetFences( dev, 1, &fence );
    if( Clock::now() > next )
    {
      next = Clock::now();   // (a late submission does not make the next ones burst)
    }
  }
  vkDeviceWaitIdle( dev );
  std::vector<double> sorted = latencies;
  std::sort( sorted.begin(), sorted.end() );
  const auto at     = [&]( double q ) { return sorted.empty() ? 0.0 : sorted[size_t( q * ( sorted.size() - 1 ) )]; };
  const long missed = long( std::count_if( sorted.begin(), sorted.end(), []( double v ) { return v > 16.667; } ) );
  printf( "uiprobe %s: %zu submissions, latency ms: median %.2f, p95 %.2f, p99 %.2f, max %.2f; over 16.7 ms: %ld (%.1f%%)\n",
          props.deviceName, sorted.size(), at( 0.5 ), at( 0.95 ), at( 0.99 ), sorted.empty() ? 0.0 : sorted.back(), missed,
          sorted.empty() ? 0.0 : 100.0 * missed / sorted.size() );
  vkDestroyDevice( dev, nullptr );
  return 0;
}
