// vkext: lists the Vulkan device extensions of each device that mention a word (argv[1]), and the driver version
#include <vulkan/vulkan.h>
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <vector>
int main( int argc, char** argv )
{
  HMODULE lib = LoadLibraryA( "vulkan-1.dll" );
  auto gipa = (PFN_vkGetInstanceProcAddr) GetProcAddress( lib, "vkGetInstanceProcAddr" );
  auto createInstance = (PFN_vkCreateInstance) gipa( nullptr, "vkCreateInstance" );
  VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ci{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
  ci.pApplicationInfo = &app;
  VkInstance inst;
  if( createInstance( &ci, nullptr, &inst ) != VK_SUCCESS ) return 1;
  auto enumDevices = (PFN_vkEnumeratePhysicalDevices) gipa( inst, "vkEnumeratePhysicalDevices" );
  auto enumExt = (PFN_vkEnumerateDeviceExtensionProperties) gipa( inst, "vkEnumerateDeviceExtensionProperties" );
  auto getProps = (PFN_vkGetPhysicalDeviceProperties) gipa( inst, "vkGetPhysicalDeviceProperties" );
  uint32_t n = 0;
  enumDevices( inst, &n, nullptr );
  std::vector<VkPhysicalDevice> devs( n );
  enumDevices( inst, &n, devs.data() );
  for( auto d: devs )
  {
    VkPhysicalDeviceProperties p;
    getProps( d, &p );
    printf( "%s driver 0x%x\n", p.deviceName, p.driverVersion );
    uint32_t m = 0;
    enumExt( d, nullptr, &m, nullptr );
    std::vector<VkExtensionProperties> ext( m );
    enumExt( d, nullptr, &m, ext.data() );
    for( auto& e: ext )
      if( argc < 2 || strstr( e.extensionName, argv[1] ) )
        printf( "  %s %u\n", e.extensionName, e.specVersion );
  }
  return 0;
}
