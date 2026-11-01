/* The hand-over's check (vvdec.h's vvdec_handover_*), Windows with Vulkan: decodes a VVC stream twice -- on the CPU,
   and on the GPU with the hand-over on -- and checks that every picture the hand-over copies into another Vulkan
   device's buffer holds exactly the luma samples the CPU decode gives. This program is that other device: it makes
   its own VkDevice on the decoder's GPU (by UUID), exports a host-visible buffer for the decoder to import, imports
   the decoder's timeline semaphore and waits for each copy's value before reading the buffer.

     vvdechandover -b stream.266 [-f frames]

   Run with VVDEC_GPU_PARSE=2 (the hand-over needs the GPU's parse). Exit 0 when every picture is the same. */

#define VK_NO_PROTOTYPES
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_win32.h>

#include "vvdec/vvdec.h"
#include "../vvdecapp/vvdecHelper.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace
{

// FNV-1a over a frame's visible luma, rows packed, samples as 16-bit little-endian
uint64_t hashLuma( const uint8_t* rows, size_t rowBytes, size_t stride, int height )
{
  uint64_t hash = 1469598103934665603ull;
  for( int y = 0; y < height; y++ )
  {
    const uint8_t* row = rows + y * stride;
    for( size_t i = 0; i < rowBytes; i++ )
    {
      hash = ( hash ^ row[i] ) * 1099511628211ull;
    }
  }
  return hash;
}

// Decodes the stream; onFrame gets each picture in output order and may not keep it.
bool decodeStream( const char* path, int maxFrames, bool gpu, const std::function<bool( vvdecDecoder*, vvdecFrame*, int )>& onFrame,
                   const std::function<bool( vvdecDecoder* )>& onOpen )
{
  std::ifstream input( path, std::ios::in | std::ios::binary );
  if( !input )
  {
    fprintf( stderr, "cannot open %s\n", path );
    return false;
  }
  vvdecParams params;
  vvdec_params_default( &params );
  params.threads = -1;
  params.gpu     = gpu ? 1 : 0;
  params.filmGrainSynthesis = false;   // the hand-over takes no grain synthesis: both decodes without the grain
  vvdecDecoder* dec = vvdec_decoder_open( &params );
  if( !dec || ( onOpen && !onOpen( dec ) ) )
  {
    fprintf( stderr, "cannot open the decoder (gpu %d)\n", int( gpu ) );
    return false;
  }
  vvdecAccessUnit* au = vvdec_accessUnit_alloc();
  vvdec_accessUnit_alloc_payload( au, MAX_CODED_PICTURE_SIZE );
  int  frames = 0;
  bool ok     = true;
  auto take   = [&]( vvdecFrame* frame ) {
    if( frame && ok && frames < maxFrames )
    {
      ok = onFrame( dec, frame, frames++ );
    }
    if( frame )
    {
      vvdec_frame_unref( dec, frame );
    }
  };
  while( ok && frames < maxFrames && readBitstreamFromFile( &input, au, false ) > 0 )
  {
    vvdecFrame* frame = nullptr;
    const int   ret   = vvdec_decode( dec, au, &frame );
    if( ret != VVDEC_OK && ret != VVDEC_TRY_AGAIN && ret != VVDEC_EOF )
    {
      fprintf( stderr, "decode error %d: %s\n", ret, vvdec_get_last_error( dec ) );
      ok = false;
    }
    take( frame );
  }
  for( int ret = VVDEC_OK; ok && ret == VVDEC_OK && frames < maxFrames; )
  {
    vvdecFrame* frame = nullptr;
    ret               = vvdec_flush( dec, &frame );
    take( frame );
  }
  vvdec_accessUnit_free( au );
  vvdec_decoder_close( dec );
  return ok;
}

// The other device: a VkDevice on the decoder's GPU with an exportable host-visible buffer and the decoder's timeline
struct OtherDevice
{
  HMODULE                   library = nullptr;
  PFN_vkGetInstanceProcAddr gip     = nullptr;
  VkInstance                instance = VK_NULL_HANDLE;
  VkPhysicalDevice          physical = VK_NULL_HANDLE;
  VkDevice                  device   = VK_NULL_HANDLE;
  VkBuffer                  buffer   = VK_NULL_HANDLE;
  VkDeviceMemory            memory   = VK_NULL_HANDLE;
  uint8_t*                  mapped   = nullptr;
  uint32_t                  memoryType = 0;
  VkSemaphore               timeline   = VK_NULL_HANDLE;
  PFN_vkGetDeviceProcAddr   gdp        = nullptr;

  template<typename F> F instanceFunction( const char* name ) { return (F) gip( instance, name ); }
  template<typename F> F deviceFunction( const char* name ) { return (F) gdp( device, name ); }

  bool open( const uint8_t deviceUuid[VK_UUID_SIZE] )
  {
    library = LoadLibraryA( "vulkan-1.dll" );
    gip     = library ? (PFN_vkGetInstanceProcAddr) GetProcAddress( library, "vkGetInstanceProcAddr" ) : nullptr;
    if( !gip )
    {
      return false;
    }
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo info{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    info.pApplicationInfo = &app;
    if( ( (PFN_vkCreateInstance) gip( nullptr, "vkCreateInstance" ) )( &info, nullptr, &instance ) != VK_SUCCESS )
    {
      return false;
    }
    uint32_t count = 0;
    auto enumerate = instanceFunction<PFN_vkEnumeratePhysicalDevices>( "vkEnumeratePhysicalDevices" );
    enumerate( instance, &count, nullptr );
    std::vector<VkPhysicalDevice> devices( count );
    enumerate( instance, &count, devices.data() );
    auto properties2 = instanceFunction<PFN_vkGetPhysicalDeviceProperties2>( "vkGetPhysicalDeviceProperties2" );
    for( VkPhysicalDevice candidate: devices )
    {
      VkPhysicalDeviceIDProperties ids{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
      VkPhysicalDeviceProperties2  props{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
      props.pNext = &ids;
      properties2( candidate, &props );
      if( !memcmp( ids.deviceUUID, deviceUuid, VK_UUID_SIZE ) )
      {
        physical = candidate;
      }
    }
    if( !physical )
    {
      return false;
    }
    const float             priority = 1.0f;
    VkDeviceQueueCreateInfo queue{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queue.queueCount       = 1;
    queue.pQueuePriorities = &priority;
    const char* extensions[] = { VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME };
    VkPhysicalDeviceVulkan12Features features12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.timelineSemaphore = VK_TRUE;
    VkDeviceCreateInfo deviceInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceInfo.pNext                   = &features12;
    deviceInfo.queueCreateInfoCount    = 1;
    deviceInfo.pQueueCreateInfos       = &queue;
    deviceInfo.enabledExtensionCount   = 2;
    deviceInfo.ppEnabledExtensionNames = extensions;
    if( instanceFunction<PFN_vkCreateDevice>( "vkCreateDevice" )( physical, &deviceInfo, nullptr, &device ) != VK_SUCCESS )
    {
      return false;
    }
    gdp = instanceFunction<PFN_vkGetDeviceProcAddr>( "vkGetDeviceProcAddr" );
    return true;
  }

  // An exportable host-visible buffer of `bytes`; its opaque Win32 handle in *handle
  bool makeBuffer( VkDeviceSize bytes, HANDLE* handle )
  {
    VkExternalMemoryBufferCreateInfo external{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
    external.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkBufferCreateInfo info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    info.pNext = &external;
    info.size  = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if( deviceFunction<PFN_vkCreateBuffer>( "vkCreateBuffer" )( device, &info, nullptr, &buffer ) != VK_SUCCESS )
    {
      return false;
    }
    VkMemoryRequirements req;
    deviceFunction<PFN_vkGetBufferMemoryRequirements>( "vkGetBufferMemoryRequirements" )( device, buffer, &req );
    VkPhysicalDeviceMemoryProperties memProps;
    instanceFunction<PFN_vkGetPhysicalDeviceMemoryProperties>( "vkGetPhysicalDeviceMemoryProperties" )( physical, &memProps );
    const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    memoryType                         = UINT32_MAX;
    for( uint32_t i = 0; i < memProps.memoryTypeCount && memoryType == UINT32_MAX; i++ )
    {
      if( ( req.memoryTypeBits & ( 1u << i ) ) && ( memProps.memoryTypes[i].propertyFlags & wanted ) == wanted )
      {
        memoryType = i;
      }
    }
    if( memoryType == UINT32_MAX )
    {
      return false;
    }
    VkMemoryDedicatedAllocateInfo dedicated{ VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    dedicated.buffer = buffer;
    VkExportMemoryAllocateInfo exported{ VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
    exported.pNext       = &dedicated;
    exported.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    VkMemoryAllocateInfo alloc{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    alloc.pNext           = &exported;
    alloc.allocationSize  = req.size;
    alloc.memoryTypeIndex = memoryType;
    if( deviceFunction<PFN_vkAllocateMemory>( "vkAllocateMemory" )( device, &alloc, nullptr, &memory ) != VK_SUCCESS
        || deviceFunction<PFN_vkBindBufferMemory>( "vkBindBufferMemory" )( device, buffer, memory, 0 ) != VK_SUCCESS
        || deviceFunction<PFN_vkMapMemory>( "vkMapMemory" )( device, memory, 0, VK_WHOLE_SIZE, 0, (void**) &mapped ) != VK_SUCCESS )
    {
      return false;
    }
    VkMemoryGetWin32HandleInfoKHR get{ VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR };
    get.memory     = memory;
    get.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    return deviceFunction<PFN_vkGetMemoryWin32HandleKHR>( "vkGetMemoryWin32HandleKHR" )( device, &get, handle ) == VK_SUCCESS;
  }

  bool importTimeline( HANDLE handle )
  {
    VkSemaphoreTypeCreateInfo type{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo info{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    info.pNext = &type;
    if( deviceFunction<PFN_vkCreateSemaphore>( "vkCreateSemaphore" )( device, &info, nullptr, &timeline ) != VK_SUCCESS )
    {
      return false;
    }
    VkImportSemaphoreWin32HandleInfoKHR import{ VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR };
    import.semaphore  = timeline;
    import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    import.handle     = handle;
    return deviceFunction<PFN_vkImportSemaphoreWin32HandleKHR>( "vkImportSemaphoreWin32HandleKHR" )( device, &import ) == VK_SUCCESS;
  }

  bool waitFor( uint64_t value )
  {
    VkSemaphoreWaitInfo wait{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
    wait.semaphoreCount = 1;
    wait.pSemaphores    = &timeline;
    wait.pValues        = &value;
    return deviceFunction<PFN_vkWaitSemaphores>( "vkWaitSemaphores" )( device, &wait, 10000000000ull ) == VK_SUCCESS;
  }
};

}   // namespace

int main( int argc, char** argv )
{
  const char* stream    = nullptr;
  int         maxFrames = 1 << 30;
  for( int i = 1; i + 1 < argc; i += 2 )
  {
    if( !strcmp( argv[i], "-b" ) )
    {
      stream = argv[i + 1];
    }
    else if( !strcmp( argv[i], "-f" ) )
    {
      maxFrames = atoi( argv[i + 1] );
    }
  }
  if( !stream )
  {
    fprintf( stderr, "usage: vvdechandover -b stream.266 [-f frames]   (with VVDEC_GPU_PARSE=2)\n" );
    return 2;
  }

  // The CPU's pictures: each one's luma hash
  std::vector<uint64_t> expected;
  const bool cpu = decodeStream( stream, maxFrames, false, [&]( vvdecDecoder*, vvdecFrame* frame, int ) {
    const vvdecPlane& y = frame->planes[0];
    expected.push_back( hashLuma( y.ptr, size_t( y.width ) * y.bytesPerSample, y.stride, int( y.height ) ) );
    return true;
  }, nullptr );
  if( !cpu || expected.empty() )
  {
    return 2;
  }

  // The GPU's, handed over into the other device's buffer
  OtherDevice other;
  void*       imported = nullptr;
  int         same = 0, different = 0, notHandedOver = 0;
  auto        onOpen = [&]( vvdecDecoder* dec ) {
    uint8_t deviceUuid[VK_UUID_SIZE], driverUuid[VK_UUID_SIZE];
    HANDLE  memoryHandle = nullptr;
    if( vvdec_handover_enable( dec ) != VVDEC_OK || vvdec_handover_device( dec, deviceUuid, driverUuid ) != VVDEC_OK )
    {
      fprintf( stderr, "no hand-over: %s\n", vvdec_get_last_error( dec ) );
      return false;
    }
    HANDLE timeline = vvdec_handover_timeline( dec );
    if( !other.open( deviceUuid ) || !timeline || !other.importTimeline( timeline ) )
    {
      fprintf( stderr, "the other device could not be made, or the timeline not imported\n" );
      return false;
    }
    CloseHandle( timeline );
    // Large enough for 8K at 16 bits
    const VkDeviceSize bytes = VkDeviceSize( 7680 ) * 4320 * 2;
    if( !other.makeBuffer( bytes, &memoryHandle ) )
    {
      fprintf( stderr, "the other device's buffer could not be made\n" );
      return false;
    }
    imported = vvdec_handover_import( dec, memoryHandle, bytes, other.memoryType, deviceUuid, driverUuid );
    CloseHandle( memoryHandle );
    if( !imported )
    {
      fprintf( stderr, "import failed: %s\n", vvdec_get_last_error( dec ) );
    }
    return imported != nullptr;
  };
  const bool gpu = decodeStream( stream, maxFrames, true, [&]( vvdecDecoder* dec, vvdecFrame* frame, int index ) {
    const int      width = int( frame->width ), height = int( frame->height );
    const int      bytes  = frame->bitDepth > 8 ? 2 : 1;   // as the CPU's frames have them
    const uint32_t stride = uint32_t( width * bytes );
    uint64_t       value  = 0;
    const int      ret    = vvdec_handover_pack( dec, frame, imported, 0, stride, bytes, 0, 0, 0, width, height, &value );
    if( ret == VVDEC_ERR_NOT_SUPPORTED )
    {
      notHandedOver++;   // in host memory (8-bit, film grain): not this check's
      return true;
    }
    if( ret != VVDEC_OK || !other.waitFor( value ) )
    {
      fprintf( stderr, "frame %d: pack %d (%s)\n", index, ret, vvdec_get_last_error( dec ) );
      return false;
    }
    const uint64_t got = hashLuma( other.mapped, size_t( width ) * bytes, stride, height );
    if( index < int( expected.size() ) && got == expected[index] )
    {
      same++;
    }
    else
    {
      different++;
      if( different <= 5 )
      {
        fprintf( stderr, "frame %d: different\n", index );
      }
    }
    return true;
  }, onOpen );
  printf( "%d pictures: %d the same as the CPU's, %d different, %d not handed over\n", int( expected.size() ), same, different,
          notHandedOver );
  return gpu && different == 0 && same + notHandedOver == int( expected.size() ) ? 0 : 1;
}
