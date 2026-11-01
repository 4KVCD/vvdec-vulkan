/* The hand-over's check (vvdec.h's vvdec_handover_*), Windows with Vulkan: decodes a VVC stream twice -- on the CPU,
   and on the GPU with the hand-over on -- and checks that every picture the hand-over copies into another Vulkan
   device's buffer holds exactly the luma samples the CPU decode gives. This program is that other device: it makes
   its own VkDevice on the decoder's GPU (by UUID), exports a host-visible buffer for the decoder to import, imports
   the decoder's timeline semaphore and waits for each copy's value before reading the buffer.

     vvdechandover -b stream.266 [-f frames]

   With --host it checks the hand-over into host memory instead (vvdec_handover_import_host, as Vship's frame buffers
   take it): every picture's Y, U and V, shifted by --shift, in the --crop rectangle (luma samples), against the CPU
   decode, or with --ref-gpu against vvdec-vulkan's own download route (a damaged stream). With --late-wait each copy
   is waited for only after its picture went back to the decoder and it decoded on (the reuse guard's check; at a
   small --parse-delay, so that the picture is reused soon). --threads and --parse-delay set both decoders'.

   The hand-over runs with the CPU's parse (the default) or the GPU's parse ahead (VVDEC_GPU_PARSE=2). Exit 0 when
   every picture is the same. */

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

constexpr uint64_t kHashStart = 1469598103934665603ull;
constexpr uint64_t kWaitNs    = 10000000000ull;   // a copy's wait gives up after 10 s (a hung GPU, not a slow copy)

// FNV-1a over a frame's visible luma, rows packed, samples as 16-bit little-endian
uint64_t hashLuma( const uint8_t* rows, size_t rowBytes, size_t stride, int height )
{
  uint64_t hash = kHashStart;
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

// FNV-1a, continuing `hash`, over the rectangle (x, y, width, height) of a plane of inBytes-byte samples, each as it
// would be stored by the hand-over: shifted left by `shift`, in outBytes bytes (little-endian)
uint64_t hashRect( uint64_t hash, const uint8_t* plane, int inBytes, size_t stride, int x, int y, int width, int height, int shift,
                   int outBytes )
{
  for( int row = y; row < y + height; row++ )
  {
    const uint8_t* samples = plane + row * stride;
    for( int col = x; col < x + width; col++ )
    {
      const unsigned in  = inBytes == 2 ? unsigned( samples[2 * col] | samples[2 * col + 1] << 8 ) : samples[col];
      const unsigned out = in << shift;
      hash               = ( hash ^ ( out & 0xff ) ) * 1099511628211ull;
      if( outBytes == 2 )
      {
        hash = ( hash ^ ( ( out >> 8 ) & 0xff ) ) * 1099511628211ull;
      }
    }
  }
  return hash;
}

// The rectangle a run checks, in luma samples (the whole picture when width is 0)
struct Crop
{
  int x = 0, y = 0, width = 0, height = 0;
};

// FNV-1a over a reference frame's Y, U and V in the crop, stored as the hand-over stores them
uint64_t hashFrame( const vvdecFrame* frame, const Crop& crop, int shift, int outBytes )
{
  uint64_t hash = kHashStart;
  for( uint32_t comp = 0; comp < frame->numPlanes; comp++ )
  {
    const vvdecPlane& p  = frame->planes[comp];
    const int         sx = comp && frame->planes[comp].width < frame->width ? 1 : 0;
    const int         sy = comp && frame->planes[comp].height < frame->height ? 1 : 0;
    hash = hashRect( hash, p.ptr, int( p.bytesPerSample ), p.stride, crop.x >> sx, crop.y >> sy, crop.width >> sx, crop.height >> sy,
                     shift, outBytes );
  }
  return hash;
}

// The decoders' threads and parse delay (--threads, --parse-delay), for both decodes of a run
struct DecoderSettings
{
  int threads    = -1;
  int parseDelay = -1;
};
DecoderSettings s_decoder;

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
  params.threads    = s_decoder.threads;
  params.parseDelay = s_decoder.parseDelay;
  params.gpu        = gpu ? 1 : 0;
  params.filmGrainSynthesis = false;   // the hand-over takes no grain synthesis: both decodes without the grain
  if( getenv( "VVDEC_HANDOVER_TIMES" ) )
  {
    params.logLevel = VVDEC_INFO;   // where the library prints the copies' times
  }
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
    return deviceFunction<PFN_vkWaitSemaphores>( "vkWaitSemaphores" )( device, &wait, kWaitNs ) == VK_SUCCESS;
  }
};

// The rectangle of each picture checked: the crop asked for, or the whole picture
Crop cropOf( const vvdecFrame* frame, const Crop& asked )
{
  return asked.width ? asked : Crop{ 0, 0, int( frame->width ), int( frame->height ) };
}

// --host: the hand-over into imported host memory (vvdec_handover_import_host, as Vship's frame buffers take it): each
// picture's Y, U and V in the crop (vvdec_handover_pack_frame), planes packed one after another, compared with the
// reference decode's stored the same way: the CPU's, or with refGpu vvdec-vulkan's own download route (for a damaged
// stream, whose concealment differs between the two parses). Prints the mean time of a pack and its wait.
// With lateWait each copy is waited for and checked only at the next picture, after its picture went back to the
// decoder, which decoded on (VideoMetricsLab's VMAF route gives a picture back before its copy is done): the decoder
// must not write the picture's host planes again before the copy is done (PicListManager's reuse guard).
int checkHost( const char* stream, int maxFrames, const Crop& crop, int shift, bool refGpu, bool lateWait )
{
  std::vector<uint64_t> expected;
  const bool            reference = decodeStream( stream, maxFrames, refGpu, [&]( vvdecDecoder*, vvdecFrame* frame, int ) {
    const Crop c = cropOf( frame, crop );
    if( c.x < 0 || c.y < 0 || c.width <= 0 || c.height <= 0 || c.x + c.width > int( frame->width )
        || c.y + c.height > int( frame->height ) )
    {
      fprintf( stderr, "the crop %d,%d,%d,%d is not inside the %ux%u picture\n", c.x, c.y, c.width, c.height, frame->width,
               frame->height );
      return false;
    }
    expected.push_back( hashFrame( frame, c, shift, frame->bitDepth > 8 ? 2 : 1 ) );
    return true;
  }, nullptr );
  if( !reference || expected.empty() )
  {
    return 2;
  }
  // Host memory as Vship's frame buffers are: page (here 64 KiB) aligned, large enough for 8K 4:2:0 at 16 bits. With
  // lateWait two take turns, as a copy is checked only once the next one is submitted.
  const size_t bytes      = size_t( 7680 ) * 4320 * 3;
  const int    slots      = lateWait ? 2 : 1;
  uint8_t*     buffers[2] = {};
  void*        imported[2] = {};
  for( int s = 0; s < slots; s++ )
  {
    buffers[s] = static_cast<uint8_t*>( VirtualAlloc( nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE ) );
  }
  auto onOpen = [&]( vvdecDecoder* dec ) {
    if( !buffers[slots - 1] || vvdec_handover_enable( dec ) != VVDEC_OK )
    {
      fprintf( stderr, "no hand-over: %s\n", vvdec_get_last_error( dec ) );
      return false;
    }
    for( int s = 0; s < slots; s++ )
    {
      imported[s] = vvdec_handover_import_host( dec, buffers[s], bytes );
      if( !imported[s] )
      {
        fprintf( stderr, "host import failed: %s\n", vvdec_get_last_error( dec ) );
        return false;
      }
    }
    return true;
  };
  // A copy submitted: where its Y, U and V are (rows packed) and the timeline value it signals
  struct Packed
  {
    int      index = 0, slot = 0, sampleBytes = 0;
    uint64_t value = 0;
    uint64_t offsets[3]{};
    uint32_t strides[3]{};
    int      widths[3]{}, heights[3]{};
  };
  int  same = 0, different = 0;
  auto wait = [&]( vvdecDecoder* dec, const Packed& p ) {
    const int waited = vvdec_handover_wait_value( dec, p.value, kWaitNs );
    if( waited != VVDEC_OK )
    {
      fprintf( stderr, "frame %d: wait %d (%s)\n", p.index, waited, vvdec_get_last_error( dec ) );
    }
    return waited == VVDEC_OK;
  };
  auto verify = [&]( const Packed& p ) {
    uint64_t got = kHashStart;
    for( int comp = 0; comp < 3; comp++ )
    {
      got = hashRect( got, buffers[p.slot] + p.offsets[comp], p.sampleBytes, p.strides[comp], 0, 0, p.widths[comp], p.heights[comp], 0,
                      p.sampleBytes );
    }
    if( p.index < int( expected.size() ) && got == expected[p.index] )
    {
      same++;
    }
    else if( ++different <= 5 )
    {
      fprintf( stderr, "frame %d: different\n", p.index );
    }
  };
  auto check = [&]( vvdecDecoder* dec, const Packed& p ) {
    if( !wait( dec, p ) )
    {
      return false;
    }
    verify( p );
    return true;
  };
  Packed        pending;
  bool          hasPending = false;
  LARGE_INTEGER frequency, packTime{};
  QueryPerformanceFrequency( &frequency );
  const bool gpu = decodeStream( stream, maxFrames, true, [&]( vvdecDecoder* dec, vvdecFrame* frame, int index ) {
    const Crop c  = cropOf( frame, crop );
    const int  sx = frame->planes[1].width < frame->width ? 1 : 0;
    const int  sy = frame->planes[1].height < frame->height ? 1 : 0;
    Packed     p;
    p.index       = index;
    p.slot        = index % slots;
    p.sampleBytes = frame->bitDepth > 8 ? 2 : 1;
    for( int comp = 0; comp < 3; comp++ )
    {
      p.widths[comp]  = comp ? c.width >> sx : c.width;
      p.heights[comp] = comp ? c.height >> sy : c.height;
      p.strides[comp] = uint32_t( p.widths[comp] * p.sampleBytes );
      p.offsets[comp] = comp ? p.offsets[comp - 1] + uint64_t( p.strides[comp - 1] ) * p.heights[comp - 1] : 0;   // Y, U, V
    }
    LARGE_INTEGER start, end;
    QueryPerformanceCounter( &start );
    const int ret = vvdec_handover_pack_frame( dec, frame, imported[p.slot], p.offsets, p.strides, p.sampleBytes, shift, c.x, c.y,
                                               c.width, c.height, &p.value );
    if( ret != VVDEC_OK )
    {
      fprintf( stderr, "frame %d: pack %d (%s)\n", index, ret, vvdec_get_last_error( dec ) );
      return false;
    }
    if( !lateWait )
    {
      if( !wait( dec, p ) )
      {
        return false;
      }
      QueryPerformanceCounter( &end );
      packTime.QuadPart += end.QuadPart - start.QuadPart;
      verify( p );
      return true;
    }
    QueryPerformanceCounter( &end );
    packTime.QuadPart += end.QuadPart - start.QuadPart;
    // The picture goes back to the decoder now, its copy maybe still running; the previous copy is checked, after the
    // decoder has decoded on. The last picture's copy is checked at once (the decoder closes after it).
    bool ok    = !hasPending || check( dec, pending );
    pending    = p;
    hasPending = index + 1 < int( expected.size() );
    if( !hasPending )
    {
      ok = check( dec, p ) && ok;
    }
    return ok;
  }, onOpen );
  const int packed = same + different;
  printf( "%d pictures: %d the same as the reference's, %d different; %s %.3f ms a picture\n", int( expected.size() ), same,
          different, lateWait ? "pack (waited later)" : "pack + wait", packed ? 1000.0 * packTime.QuadPart / frequency.QuadPart / packed : 0.0 );
  for( int s = 0; s < slots; s++ )
  {
    VirtualFree( buffers[s], 0, MEM_RELEASE );
  }
  return gpu && different == 0 && same == int( expected.size() ) ? 0 : 1;
}

}   // namespace

int main( int argc, char** argv )
{
  const char* stream    = nullptr;
  int         maxFrames = 1 << 30;
  bool        host = false, refGpu = false, lateWait = false;
  int         shift = 0;
  Crop        crop;
  for( int i = 1; i < argc; i++ )
  {
    const bool value = i + 1 < argc;
    if( !strcmp( argv[i], "-b" ) && value )
    {
      stream = argv[++i];
    }
    else if( !strcmp( argv[i], "-f" ) && value )
    {
      maxFrames = atoi( argv[++i] );
    }
    else if( !strcmp( argv[i], "--shift" ) && value )
    {
      shift = atoi( argv[++i] );
    }
    else if( !strcmp( argv[i], "--crop" ) && value )
    {
      sscanf( argv[++i], "%d,%d,%d,%d", &crop.x, &crop.y, &crop.width, &crop.height );
    }
    else if( !strcmp( argv[i], "--host" ) )
    {
      host = true;
    }
    else if( !strcmp( argv[i], "--ref-gpu" ) )
    {
      refGpu = true;
    }
    else if( !strcmp( argv[i], "--late-wait" ) )
    {
      lateWait = true;
    }
    else if( !strcmp( argv[i], "--threads" ) && value )
    {
      s_decoder.threads = atoi( argv[++i] );
    }
    else if( !strcmp( argv[i], "--parse-delay" ) && value )
    {
      s_decoder.parseDelay = atoi( argv[++i] );
    }
  }
  if( !stream )
  {
    fprintf( stderr, "usage: vvdechandover -b stream.266 [-f frames] [--threads n] [--parse-delay n]"
                     " [--host [--shift n] [--crop x,y,w,h] [--ref-gpu] [--late-wait]]\n" );
    return 2;
  }
  if( host )
  {
    return checkHost( stream, maxFrames, crop, shift, refGpu, lateWait );
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
