/* -----------------------------------------------------------------------------
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or
other Intellectual Property Rights other than the copyrights concerning
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2018-2026, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The VVdeC Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.


------------------------------------------------------------------------------------------- */

/** \file     GpuContext.cpp
    \brief    the Vulkan device shared by all decoder instances of a process
*/

#include "GpuContext.h"

#include "CommonLib/CommonDef.h"

#include "vvdec_gpu/GpuShaders.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace vvdec
{
namespace gpu
{

std::mutex Context::s_mutex;
Context*   Context::s_context  = nullptr;
int        Context::s_refs     = 0;
void*      Context::s_library  = nullptr;
VkInstance Context::s_instance = VK_NULL_HANDLE;

static constexpr uint32_t kPushConstantSize = 256;

Context* Context::acquire()
{
  std::lock_guard<std::mutex> lock( s_mutex );
  if( !s_context )
  {
    Context* context = new Context();
    if( !context->init() )
    {
      delete context;
      return nullptr;
    }
    s_context = context;
  }
  s_refs++;
  return s_context;
}

void Context::release()
{
  std::lock_guard<std::mutex> lock( s_mutex );
  if( s_refs > 0 && --s_refs == 0 )
  {
    delete s_context;
    s_context = nullptr;
  }
}

Options Options::fromEnvironment()
{
  const auto value = []( const char* name ) -> const char* { return getenv( name ); };
  Options    options;
  options.device        = value( "VVDEC_GPU_DEVICE" ) ? value( "VVDEC_GPU_DEVICE" ) : "";
  options.profile       = value( "VVDEC_GPU_PROFILE" ) && atoi( value( "VVDEC_GPU_PROFILE" ) ) != 0;
  options.isaDir        = value( "VVDEC_GPU_ISA" ) ? value( "VVDEC_GPU_ISA" ) : "";
  options.async         = !( value( "VVDEC_GPU_ASYNC" ) && atoi( value( "VVDEC_GPU_ASYNC" ) ) == 0 );
  options.frontEnd      = !( value( "VVDEC_GPU_FRONT" ) && atoi( value( "VVDEC_GPU_FRONT" ) ) == 0 );
  options.frontEndCheck = value( "VVDEC_GPU_MIDER" ) && atoi( value( "VVDEC_GPU_MIDER" ) ) != 0;
  options.parse         = value( "VVDEC_GPU_PARSE" ) && atoi( value( "VVDEC_GPU_PARSE" ) ) == 1;
  options.parseDrives   = value( "VVDEC_GPU_PARSE" ) && atoi( value( "VVDEC_GPU_PARSE" ) ) == 2;
  return options;
}

bool Context::init()
{
  m_options = Options::fromEnvironment();
#ifdef _WIN32
  if( !s_library )
  {
    s_library = LoadLibraryW( L"vulkan-1.dll" );
  }
  if( !s_library )
  {
    msg( WARNING, "Vulkan: vulkan-1.dll not found\n" );
    return false;
  }
  m_api.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) (void*) GetProcAddress( (HMODULE) s_library, "vkGetInstanceProcAddr" );
#else
  if( !s_library )
  {
    s_library = dlopen( "libvulkan.so.1", RTLD_NOW | RTLD_LOCAL );
  }
  if( !s_library )
  {
    msg( WARNING, "Vulkan: libvulkan.so.1 not found\n" );
    return false;
  }
  m_api.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) dlsym( s_library, "vkGetInstanceProcAddr" );
#endif
  if( !m_api.vkGetInstanceProcAddr )
  {
    return false;
  }

  if( !s_instance )
  {
    auto createInstance = (PFN_vkCreateInstance) m_api.vkGetInstanceProcAddr( VK_NULL_HANDLE, "vkCreateInstance" );
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "vvdec";
    app.apiVersion       = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instanceInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instanceInfo.pApplicationInfo = &app;
    VkInstance instance           = VK_NULL_HANDLE;
    if( !createInstance || createInstance( &instanceInfo, nullptr, &instance ) != VK_SUCCESS )
    {
      msg( WARNING, "Vulkan: no instance\n" );
      return false;
    }
    s_instance = instance;
  }
  m_instance = s_instance;

#define VVDEC_VK_LOAD_INSTANCE( name ) m_api.name = (PFN_##name) m_api.vkGetInstanceProcAddr( m_instance, #name );
  VVDEC_VK_INSTANCE_FUNCS( VVDEC_VK_LOAD_INSTANCE )
#undef VVDEC_VK_LOAD_INSTANCE

  uint32_t count = 0;
  m_api.vkEnumeratePhysicalDevices( m_instance, &count, nullptr );
  std::vector<VkPhysicalDevice> devices( count );
  m_api.vkEnumeratePhysicalDevices( m_instance, &count, devices.data() );
  if( devices.empty() )
  {
    msg( WARNING, "Vulkan: no device\n" );
    return false;
  }

  // VVDEC_GPU_DEVICE picks a device by its index; otherwise the first discrete
  // GPU, then the first integrated one, then the first device
  int chosen = -1;
  if( !m_options.device.empty() )
  {
    const char* index = m_options.device.c_str();
    chosen            = atoi( index );
    if( chosen < 0 || chosen >= (int) devices.size() )
    {
      msg( WARNING, "Vulkan: VVDEC_GPU_DEVICE=%s is not a device index\n", index );
      return false;
    }
  }
  for( VkPhysicalDeviceType wanted: { VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU } )
  {
    for( int i = 0; i < (int) devices.size() && chosen < 0; i++ )
    {
      VkPhysicalDeviceProperties props;
      m_api.vkGetPhysicalDeviceProperties( devices[i], &props );
      if( props.deviceType == wanted && props.apiVersion >= VK_API_VERSION_1_2 )
      {
        chosen = i;
      }
    }
  }
  if( chosen < 0 )
  {
    chosen = 0;
  }
  m_physical = devices[chosen];

  VkPhysicalDeviceProperties props;
  m_api.vkGetPhysicalDeviceProperties( m_physical, &props );
  m_deviceName      = props.deviceName;
  m_timestampPeriod = props.limits.timestampPeriod;
  if( props.apiVersion < VK_API_VERSION_1_2 || props.limits.maxPushConstantsSize < kPushConstantSize )
  {
    msg( WARNING, "Vulkan: %s lacks Vulkan 1.2 or 256 bytes of push constants\n", props.deviceName );
    return false;
  }

  VkPhysicalDeviceVulkan12Features features12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
  VkPhysicalDeviceVulkan11Features features11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
  VkPhysicalDeviceSubgroupSizeControlFeatures sizeControl{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES };
  features12.pNext  = &features11;
  features11.pNext  = &sizeControl;
  VkPhysicalDeviceFeatures2 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
  features.pNext = &features12;
  m_api.vkGetPhysicalDeviceFeatures2( m_physical, &features );
  if( !features.features.shaderInt16 || !features.features.shaderInt64 || !features12.bufferDeviceAddress
      || !features12.storageBuffer8BitAccess || !features12.shaderInt8 || !features11.storageBuffer16BitAccess )
  {
    msg( WARNING, "Vulkan: %s lacks 8/16/64-bit integers or buffer device addresses\n", props.deviceName );
    return false;
  }

  // a queue with compute: the first family that has it
  m_api.vkGetPhysicalDeviceQueueFamilyProperties( m_physical, &count, nullptr );
  std::vector<VkQueueFamilyProperties> families( count );
  m_api.vkGetPhysicalDeviceQueueFamilyProperties( m_physical, &count, families.data() );
  m_queueFamily = UINT32_MAX;
  for( uint32_t i = 0; i < count && m_queueFamily == UINT32_MAX; i++ )
  {
    if( ( families[i].queueFlags & VK_QUEUE_COMPUTE_BIT ) && families[i].timestampValidBits )
    {
      m_queueFamily = i;
    }
  }
  if( m_queueFamily == UINT32_MAX )
  {
    msg( WARNING, "Vulkan: %s has no compute queue with timestamps\n", props.deviceName );
    return false;
  }

  // and a family for copies only (the copy engine), the pictures' downloads
  // there run alongside the compute queue's work
  for( uint32_t i = 0; i < count && m_transferFamily == UINT32_MAX; i++ )
  {
    if( ( families[i].queueFlags & VK_QUEUE_TRANSFER_BIT ) && !( families[i].queueFlags & ( VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT ) ) )
    {
      m_transferFamily     = i;
      m_transferTimestamps = families[i].timestampValidBits != 0;
    }
  }

  // and a family for compute only (an async compute engine), with timestamps: the GPU's front end there runs
  // alongside the compute queue's work (VVDEC_GPU_ASYNC=0: not)
  for( uint32_t i = 0; i < count && m_asyncFamily == UINT32_MAX && features12.timelineSemaphore && m_options.async; i++ )
  {
    if( i != m_queueFamily && ( families[i].queueFlags & VK_QUEUE_COMPUTE_BIT ) && !( families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT )
        && families[i].timestampValidBits )
    {
      m_asyncFamily = i;
    }
  }

  const float priority = 1.0f;
  const float asyncPriorities[kMaxAsync + 1] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  m_asyncCount = m_asyncFamily != UINT32_MAX ? int( std::min<uint32_t>( families[m_asyncFamily].queueCount, uint32_t( kMaxAsync ) ) ) : 0;
  // the parse queue: one more of the family's when it has one (AMD's has 8)
  const bool ownParseQueue = m_asyncCount > 0 && families[m_asyncFamily].queueCount > uint32_t( m_asyncCount );
  VkDeviceQueueCreateInfo queueInfos[3] = { { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO }, { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO },
                                            { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO } };
  uint32_t numQueueInfos = 0;
  for( const uint32_t family: { m_queueFamily, m_transferFamily, m_asyncFamily } )
  {
    if( family != UINT32_MAX )
    {
      queueInfos[numQueueInfos].queueFamilyIndex = family;
      queueInfos[numQueueInfos].queueCount       = family == m_asyncFamily ? uint32_t( m_asyncCount + ( ownParseQueue ? 1 : 0 ) ) : 1;
      queueInfos[numQueueInfos].pQueuePriorities = family == m_asyncFamily ? asyncPriorities : &priority;
      numQueueInfos++;
    }
  }

  // waves of 32 threads where the shaders run better so (GPUs with both
  // sizes, AMD's): the size, subgroup size control
  VkPhysicalDeviceSubgroupSizeControlProperties sizes{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES };
  VkPhysicalDeviceProperties2                   props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
  props2.pNext = &sizes;
  m_api.vkGetPhysicalDeviceProperties2( m_physical, &props2 );
  m_wave32 = props.apiVersion >= VK_API_VERSION_1_3 && sizeControl.subgroupSizeControl && sizes.minSubgroupSize <= 32 && sizes.maxSubgroupSize >= 32
             && ( sizes.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT );

  VkPhysicalDeviceVulkan12Features enable12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
  VkPhysicalDeviceVulkan11Features enable11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
  VkPhysicalDeviceSubgroupSizeControlFeatures enableSize{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES };
  enableSize.subgroupSizeControl    = VK_TRUE;
  enable11.pNext                    = m_wave32 ? &enableSize : nullptr;
  enable12.pNext                    = &enable11;
  enable12.bufferDeviceAddress      = VK_TRUE;
  enable12.storageBuffer8BitAccess  = VK_TRUE;
  enable12.shaderInt8               = VK_TRUE;
  enable12.timelineSemaphore        = asyncQueue() ? VK_TRUE : VK_FALSE;
  enable11.storageBuffer16BitAccess = VK_TRUE;
  VkPhysicalDeviceFeatures2 enable{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
  enable.pNext                = &enable12;
  enable.features.shaderInt16 = VK_TRUE;
  enable.features.shaderInt64 = VK_TRUE;

  // VVDEC_GPU_PROFILE=1: the GPU time of the passes, and with
  // VK_AMD_shader_info the registers and memory each shader takes
  m_profiling = m_options.profile;
  const char* shaderInfo = "VK_AMD_shader_info";
  const char* shaderClock = "VK_KHR_shader_clock";
  bool        hasShaderInfo = false;
  bool        hasShaderClock = false;
  if( m_profiling )
  {
    uint32_t extensionCount = 0;
    m_api.vkEnumerateDeviceExtensionProperties( m_physical, nullptr, &extensionCount, nullptr );
    std::vector<VkExtensionProperties> extensions( extensionCount );
    m_api.vkEnumerateDeviceExtensionProperties( m_physical, nullptr, &extensionCount, extensions.data() );
    for( const VkExtensionProperties& extension: extensions )
    {
      hasShaderInfo |= strcmp( extension.extensionName, shaderInfo ) == 0;
      hasShaderClock |= strcmp( extension.extensionName, shaderClock ) == 0;
    }
  }
  // and the subgroup's clock, for the parse's profile (shaders/parse.slang's PROFILE)
  VkPhysicalDeviceShaderClockFeaturesKHR enableClock{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR };
  enableClock.shaderSubgroupClock = VK_TRUE;
  if( hasShaderClock )
  {
    enableClock.pNext = enable.pNext;
    enable.pNext      = &enableClock;
  }
  m_shaderClock = hasShaderClock;
  std::vector<const char*> deviceExtensions;
  if( hasShaderInfo )
  {
    deviceExtensions.push_back( shaderInfo );
  }
  if( hasShaderClock )
  {
    deviceExtensions.push_back( shaderClock );
  }

  VkDeviceCreateInfo deviceInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  deviceInfo.pNext                = &enable;
  deviceInfo.queueCreateInfoCount = numQueueInfos;
  deviceInfo.pQueueCreateInfos    = queueInfos;
  deviceInfo.enabledExtensionCount   = uint32_t( deviceExtensions.size() );
  deviceInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();
  if( m_api.vkCreateDevice( m_physical, &deviceInfo, nullptr, &m_device ) != VK_SUCCESS )
  {
    msg( WARNING, "Vulkan: creating the device on %s failed\n", props.deviceName );
    return false;
  }

#define VVDEC_VK_LOAD_DEVICE( name ) m_api.name = (PFN_##name) m_api.vkGetDeviceProcAddr( m_device, #name );
  VVDEC_VK_DEVICE_FUNCS( VVDEC_VK_LOAD_DEVICE )
#undef VVDEC_VK_LOAD_DEVICE
  if( hasShaderInfo )
  {
    m_getShaderInfo = (PFN_vkGetShaderInfoAMD) m_api.vkGetDeviceProcAddr( m_device, "vkGetShaderInfoAMD" );
  }

  m_api.vkGetDeviceQueue( m_device, m_queueFamily, 0, &m_queue );
  if( transferQueue() )
  {
    m_api.vkGetDeviceQueue( m_device, m_transferFamily, 0, &m_transferQueue );
  }
  if( asyncQueue() )
  {
    for( int i = 0; i < m_asyncCount; i++ )
    {
      m_api.vkGetDeviceQueue( m_device, m_asyncFamily, uint32_t( i ), &m_asyncQueues[i] );
    }
    if( ownParseQueue )
    {
      m_api.vkGetDeviceQueue( m_device, m_asyncFamily, uint32_t( m_asyncCount ), &m_parseQueue );
    }
    else
    {
      m_parseQueue = m_asyncQueues[m_asyncCount - 1];
      m_parseMutex = &m_asyncMutexes[m_asyncCount - 1];
    }
    VkSemaphoreTypeCreateInfo timelineType{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    timelineType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo timelineInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    timelineInfo.pNext = &timelineType;
    if( m_api.vkCreateSemaphore( m_device, &timelineInfo, nullptr, &m_timeline ) != VK_SUCCESS
        || m_api.vkCreateSemaphore( m_device, &timelineInfo, nullptr, &m_parseTimeline ) != VK_SUCCESS )
    {
      THROW_FATAL( "Vulkan: vkCreateSemaphore failed (the timeline semaphores)" );
    }
  }
  m_api.vkGetPhysicalDeviceMemoryProperties( m_physical, &m_memProps );

  VkPushConstantRange range{ VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushConstantSize };
  VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges    = &range;
  if( m_api.vkCreatePipelineLayout( m_device, &layoutInfo, nullptr, &m_layout ) != VK_SUCCESS )
  {
    return false;
  }

  m_pipelines = new VkPipeline[kShaderCount];
  for( int i = 0; i < kShaderCount; i++ )
  {
    m_pipelines[i] = VK_NULL_HANDLE;
  }

  msg( INFO, "Vulkan: %s\n", m_deviceName.c_str() );
  return true;
}

void Context::addTime( const char* label, double milliseconds )
{
  std::lock_guard<std::mutex> lock( m_timeMutex );
  for( Time& time: m_times )
  {
    if( time.label == label )
    {
      time.milliseconds += milliseconds;
      time.count++;
      return;
    }
  }
  m_times.push_back( Time{ label, milliseconds, 1 } );
}

Buffer Context::allocPooled( size_t size )
{
  const VkDeviceSize rounded = ( size + 255 ) & ~size_t( 255 );
  {
    std::lock_guard<std::mutex> lock( m_poolMutex );
    for( auto it = m_pool.begin(); it != m_pool.end(); ++it )
    {
      if( it->size == rounded )
      {
        const Buffer buf = *it;
        m_pool.erase( it );
        return buf;
      }
    }
    // none of the size: those of the sizes not wanted any more go, once the
    // work queued is done (the scratch planes come back while their work is
    // queued)
    if( !m_pool.empty() && m_poolSizes.count( rounded ) == 0 )
    {
      {
        std::lock_guard<std::mutex> queueLock( m_queueMutex );
        m_api.vkQueueWaitIdle( m_queue );
      }
      for( Buffer& buf: m_pool )
      {
        free( buf );
      }
      m_pool.clear();
      m_poolSizes.clear();
    }
    m_poolSizes.insert( rounded );
  }
  return alloc( size, MemKind::Device );
}

void Context::recycle( Buffer& buf )
{
  if( !buf.valid() )
  {
    return;
  }
  std::lock_guard<std::mutex> lock( m_poolMutex );
  m_pool.push_back( buf );
  buf = Buffer();
}

void Context::countMemory( MemKind kind, int64_t bytes )
{
  const int     heap = kind == MemKind::Device ? 0 : 1;
  const int64_t now  = m_memoryNow[heap] += bytes;
  int64_t       peak = m_memoryPeak[heap];
  while( now > peak && !m_memoryPeak[heap].compare_exchange_weak( peak, now ) )
  {
  }
}

Context::~Context()
{
  if( m_profiling && !m_times.empty() )
  {
    msg( INFO, "GPU profile (milliseconds):\n" );
    for( const Time& time: m_times )
    {
      msg( INFO, "  %-16s total %10.1f  count %7ld  mean %8.3f\n", time.label.c_str(), time.milliseconds, time.count,
           time.milliseconds / double( time.count ) );
    }
  }

  if( m_device )
  {
    m_api.vkDeviceWaitIdle( m_device );
    for( Buffer& buf: m_pool )
    {
      free( buf );
    }
    m_pool.clear();
    if( m_profiling )
    {
      msg( INFO, "GPU memory (MB): device-local peak %.1f, at the end %.1f; host-visible peak %.1f, at the end %.1f\n", m_memoryPeak[0] / 1048576.0,
           m_memoryNow[0] / 1048576.0, m_memoryPeak[1] / 1048576.0, m_memoryNow[1] / 1048576.0 );
    }
    if( m_pipelines )
    {
      for( int i = 0; i < kShaderCount; i++ )
      {
        if( m_pipelines[i] )
        {
          m_api.vkDestroyPipeline( m_device, m_pipelines[i], nullptr );
        }
      }
    }
    if( m_layout )
    {
      m_api.vkDestroyPipelineLayout( m_device, m_layout, nullptr );
    }
    if( m_timeline )
    {
      m_api.vkDestroySemaphore( m_device, m_timeline, nullptr );
    }
    if( m_parseTimeline )
    {
      m_api.vkDestroySemaphore( m_device, m_parseTimeline, nullptr );
    }
    m_api.vkDestroyDevice( m_device, nullptr );
  }
  delete[] m_pipelines;
  // the instance and the Vulkan library stay (s_instance)
}

int Context::memoryType( uint32_t typeBits, MemKind kind ) const
{
  // flags a type must have, flags it should have, and flags it must not have
  // (AMD's uncached device-coherent types are slow for ordinary use)
  VkMemoryPropertyFlags required = 0, preferred = 0;
  const VkMemoryPropertyFlags avoid = VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
  switch( kind )
  {
  case MemKind::Device:
    required  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    break;
  case MemKind::HostCached:
    required  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    preferred = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    break;
  case MemKind::Upload:
    required  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    preferred = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    break;
  }
  for( VkMemoryPropertyFlags want: { required | preferred, required } )
  {
    for( uint32_t i = 0; i < m_memProps.memoryTypeCount; i++ )
    {
      const VkMemoryPropertyFlags flags = m_memProps.memoryTypes[i].propertyFlags;
      if( ( typeBits & ( 1u << i ) ) && ( flags & want ) == want && !( flags & avoid ) )
      {
        // for device memory, a type the CPU cannot see is the one that is not a scarce BAR window
        if( kind == MemKind::Device && ( flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ) && want == required )
        {
          continue;
        }
        return int( i );
      }
    }
  }
  if( kind == MemKind::Device )
  {
    for( uint32_t i = 0; i < m_memProps.memoryTypeCount; i++ )
    {
      if( ( typeBits & ( 1u << i ) ) && ( m_memProps.memoryTypes[i].propertyFlags & required ) == required
          && !( m_memProps.memoryTypes[i].propertyFlags & avoid ) )
      {
        return int( i );
      }
    }
  }
  return -1;
}

Buffer Context::alloc( size_t size, MemKind kind )
{
  Buffer buf;
  buf.kind = kind;
  buf.size = ( size + 255 ) & ~size_t( 255 );

  VkBufferCreateInfo info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  info.size  = buf.size;
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
               | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  uint32_t families[3]  = { m_queueFamily };
  uint32_t numFamilies  = 1;
  for( const uint32_t family: { m_transferFamily, m_asyncFamily } )
  {
    if( family != UINT32_MAX )
    {
      families[numFamilies++] = family;
    }
  }
  if( numFamilies > 1 )
  {
    info.sharingMode           = VK_SHARING_MODE_CONCURRENT;   // the compute queue's, the copy engine's, the async compute queue's
    info.queueFamilyIndexCount = numFamilies;
    info.pQueueFamilyIndices   = families;
  }
  if( m_api.vkCreateBuffer( m_device, &info, nullptr, &buf.buffer ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateBuffer failed for " << size << " bytes" );
  }

  VkMemoryRequirements req;
  m_api.vkGetBufferMemoryRequirements( m_device, buf.buffer, &req );
  const int type = memoryType( req.memoryTypeBits, kind );
  if( type < 0 )
  {
    m_api.vkDestroyBuffer( m_device, buf.buffer, nullptr );
    THROW_FATAL( "Vulkan: no memory type for a buffer of " << size << " bytes" );
  }

  VkMemoryAllocateFlagsInfo flags{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
  allocInfo.pNext           = &flags;
  allocInfo.allocationSize  = req.size;
  allocInfo.memoryTypeIndex = uint32_t( type );
  VkResult result = m_api.vkAllocateMemory( m_device, &allocInfo, nullptr, &buf.memory );
  if( result != VK_SUCCESS && kind == MemKind::Upload )
  {
    // the BAR window is small: fall back to plain host memory
    m_api.vkDestroyBuffer( m_device, buf.buffer, nullptr );
    return alloc( size, MemKind::HostCached );
  }
  if( result != VK_SUCCESS )
  {
    m_api.vkDestroyBuffer( m_device, buf.buffer, nullptr );
    THROW_FATAL( "Vulkan: vkAllocateMemory failed for " << size << " bytes (" << result << ")" );
  }
  m_api.vkBindBufferMemory( m_device, buf.buffer, buf.memory, 0 );
  buf.memorySize = req.size;
  countMemory( buf.kind, int64_t( req.size ) );

  if( m_memProps.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT )
  {
    void* mapped = nullptr;
    m_api.vkMapMemory( m_device, buf.memory, 0, VK_WHOLE_SIZE, 0, &mapped );
    buf.mapped = static_cast<uint8_t*>( mapped );
  }

  VkBufferDeviceAddressInfo addrInfo{ VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
  addrInfo.buffer = buf.buffer;
  buf.addr        = m_api.vkGetBufferDeviceAddress( m_device, &addrInfo );
  return buf;
}

void Context::free( Buffer& buf )
{
  if( buf.buffer )
  {
    m_api.vkDestroyBuffer( m_device, buf.buffer, nullptr );
  }
  if( buf.memory )
  {
    m_api.vkFreeMemory( m_device, buf.memory, nullptr );
    countMemory( buf.kind, -int64_t( buf.memorySize ) );
  }
  buf = Buffer();
}

VkPipeline Context::pipeline( int shaderId )
{
  std::lock_guard<std::mutex> lock( m_pipelineMutex );
  if( m_pipelines[shaderId] )
  {
    return m_pipelines[shaderId];
  }

  VkShaderModuleCreateInfo moduleInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  moduleInfo.codeSize = kShaders[shaderId].size;
  moduleInfo.pCode    = kShaders[shaderId].words;
  VkShaderModule module;
  if( m_api.vkCreateShaderModule( m_device, &moduleInfo, nullptr, &module ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: the shader " << kShaders[shaderId].name << " was refused" );
  }

  VkComputePipelineCreateInfo info{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
  info.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  info.stage.module = module;
  info.stage.pName  = "main";
  info.layout       = m_layout;
  // waves of 32 threads, but for inter (VideoQ 4K on a Radeon 780M: intra
  // 1.50 -> 1.31 ms, ALF luma 1.54 -> 1.36, the residuals 0.33 -> 0.28, but
  // inter 4.4 -> 5.2 ms, where the driver's choice, waves of 64, is better)
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo size{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
  size.requiredSubgroupSize = 32;
  if( m_wave32 && shaderId != kShader_inter )
  {
    info.stage.pNext = &size;
  }
  VkPipeline pipe   = VK_NULL_HANDLE;
  const VkResult result = m_api.vkCreateComputePipelines( m_device, VK_NULL_HANDLE, 1, &info, nullptr, &pipe );
  m_api.vkDestroyShaderModule( m_device, module, nullptr );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: the pipeline of " << kShaders[shaderId].name << " failed (" << result << ")" );
  }
  m_pipelines[shaderId] = pipe;

  if( m_getShaderInfo )
  {
    VkShaderStatisticsInfoAMD stats{};
    size_t                    size = sizeof( stats );
    if( m_getShaderInfo( m_device, pipe, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_INFO_TYPE_STATISTICS_AMD, &size, &stats ) == VK_SUCCESS )
    {
      msg( INFO, "GPU shader %-16s VGPRs %3u (of %3u)  SGPRs %3u  shared memory %5zu bytes  scratch %5zu bytes\n", kShaders[shaderId].name,
           stats.resourceUsage.numUsedVgprs, stats.numAvailableVgprs, stats.resourceUsage.numUsedSgprs, stats.resourceUsage.ldsUsageSizeInBytes,
           stats.resourceUsage.scratchMemUsageInBytes );
    }
    // VVDEC_GPU_ISA=<directory>: the shader's machine code to <directory>/<name>.isa
    if( !m_options.isaDir.empty() )
    {
      const char* dir = m_options.isaDir.c_str();
      size_t textSize = 0;
      if( m_getShaderInfo( m_device, pipe, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &textSize, nullptr ) == VK_SUCCESS )
      {
        std::vector<char> text( textSize );
        if( m_getShaderInfo( m_device, pipe, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_INFO_TYPE_DISASSEMBLY_AMD, &textSize, text.data() ) == VK_SUCCESS )
        {
          const std::string path = std::string( dir ) + "/" + kShaders[shaderId].name + ".isa";
          if( FILE* file = fopen( path.c_str(), "wb" ) )
          {
            fwrite( text.data(), 1, textSize, file );
            fclose( file );
          }
        }
      }
    }
  }
  return pipe;
}

uint64_t Context::submitFirst( VkCommandBuffer first, VkFence firstDone, VkSemaphore waitFront )
{
  VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  info.commandBufferCount = 1;
  info.pCommandBuffers    = &first;
  // (the front end's results: its buffers, the indirect dispatches' arguments)
  const VkPipelineStageFlags    stage       = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
  const uint64_t                waitValue   = 0;   // (a binary semaphore's)
  uint64_t                      signalValue = 0;
  VkTimelineSemaphoreSubmitInfo timeline{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
  if( waitFront )
  {
    info.waitSemaphoreCount          = 1;
    info.pWaitSemaphores             = &waitFront;
    info.pWaitDstStageMask           = &stage;
    timeline.waitSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues    = &waitValue;
  }
  std::lock_guard<std::mutex> lock( m_queueMutex );
  if( m_timeline )
  {
    signalValue                        = ++m_timelineValue;
    info.signalSemaphoreCount          = 1;
    info.pSignalSemaphores             = &m_timeline;
    timeline.signalSemaphoreValueCount = 1;
    timeline.pSignalSemaphoreValues    = &signalValue;
    info.pNext                         = &timeline;
  }
  const VkResult result = m_api.vkQueueSubmit( m_queue, 1, &info, firstDone );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkQueueSubmit failed (" << result << ")" );
  }
  return signalValue;
}

void Context::submitFront( int queue, VkCommandBuffer cmd, uint64_t waitFirst, VkSemaphore done, uint64_t waitParse )
{
  VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  info.commandBufferCount   = 1;
  info.pCommandBuffers      = &cmd;
  info.signalSemaphoreCount = 1;
  info.pSignalSemaphores    = &done;
  const VkPipelineStageFlags    stages[2]   = { VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
  const uint64_t                signalValue = 0;   // (a binary semaphore's)
  VkTimelineSemaphoreSubmitInfo timeline{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
  timeline.signalSemaphoreValueCount = 1;
  timeline.pSignalSemaphoreValues    = &signalValue;
  // the waits: the colocated pictures' first parts, the picture's parse
  VkSemaphore waits[2];
  uint64_t    waitValues[2];
  uint32_t    numWaits = 0;
  if( waitFirst > 0 )
  {
    waits[numWaits]      = m_timeline;
    waitValues[numWaits] = waitFirst;
    numWaits++;
  }
  if( waitParse > 0 )
  {
    waits[numWaits]      = m_parseTimeline;
    waitValues[numWaits] = waitParse;
    numWaits++;
  }
  info.waitSemaphoreCount          = numWaits;
  info.pWaitSemaphores             = waits;
  info.pWaitDstStageMask           = stages;
  timeline.waitSemaphoreValueCount = numWaits;
  timeline.pWaitSemaphoreValues    = waitValues;
  info.pNext = &timeline;
  std::lock_guard<std::mutex> lock( m_asyncMutexes[queue] );
  const VkResult result = m_api.vkQueueSubmit( m_asyncQueues[queue], 1, &info, VK_NULL_HANDLE );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkQueueSubmit failed on the async compute queue (" << result << ")" );
  }
}

uint64_t Context::firstPartsDone() const
{
  uint64_t value = 0;
  if( m_timeline && m_api.vkGetSemaphoreCounterValue( m_device, m_timeline, &value ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkGetSemaphoreCounterValue failed" );
  }
  return value;
}

uint64_t Context::submitParse( VkCommandBuffer cmd, VkFence done )
{
  VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  info.commandBufferCount   = 1;
  info.pCommandBuffers      = &cmd;
  info.signalSemaphoreCount = 1;
  info.pSignalSemaphores    = &m_parseTimeline;
  VkTimelineSemaphoreSubmitInfo timeline{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
  std::lock_guard<std::mutex>   lock( *m_parseMutex );
  const uint64_t                value = ++m_parseValue;
  timeline.signalSemaphoreValueCount = 1;
  timeline.pSignalSemaphoreValues    = &value;
  info.pNext                         = &timeline;
  const VkResult result = m_api.vkQueueSubmit( m_parseQueue, 1, &info, done );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkQueueSubmit failed on the parse queue (" << result << ")" );
  }
  return value;
}

void Context::submitRest( VkCommandBuffer rest, VkFence restDone, VkSemaphore semaphore, VkCommandBuffer transfer, VkFence transferDone,
                          VkSemaphore waitFront )
{
  {
    VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    info.commandBufferCount   = 1;
    info.pCommandBuffers      = &rest;
    info.signalSemaphoreCount = transfer ? 1 : 0;
    info.pSignalSemaphores    = &semaphore;
    // (the front end's second part: intra's waves and order, the deblocking's parameters and edges, their arguments)
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if( waitFront )
    {
      info.waitSemaphoreCount = 1;
      info.pWaitSemaphores    = &waitFront;
      info.pWaitDstStageMask  = &stage;
    }
    std::lock_guard<std::mutex> lock( m_queueMutex );
    const VkResult result = m_api.vkQueueSubmit( m_queue, 1, &info, restDone );
    if( result != VK_SUCCESS )
    {
      THROW_FATAL( "Vulkan: vkQueueSubmit failed (" << result << ")" );
    }
  }
  if( transfer )
  {
    const VkPipelineStageFlags  stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo                info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    info.commandBufferCount = 1;
    info.pCommandBuffers    = &transfer;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores    = &semaphore;
    info.pWaitDstStageMask  = &stage;
    std::lock_guard<std::mutex> lock( m_transferMutex );
    const VkResult              result = m_api.vkQueueSubmit( m_transferQueue, 1, &info, transferDone );
    if( result != VK_SUCCESS )
    {
      THROW_FATAL( "Vulkan: vkQueueSubmit (copies) failed (" << result << ")" );
    }
  }
}

VkCommandPool Context::createCommandPool( uint32_t family )
{
  VkCommandPoolCreateInfo info{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
  info.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  info.queueFamilyIndex = family;
  VkCommandPool pool    = VK_NULL_HANDLE;
  if( m_api.vkCreateCommandPool( m_device, &info, nullptr, &pool ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateCommandPool failed" );
  }
  return pool;
}

void Context::allocateCommandBuffers( VkCommandPool pool, uint32_t count, VkCommandBuffer* out )
{
  VkCommandBufferAllocateInfo info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  info.commandPool        = pool;
  info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  info.commandBufferCount = count;
  if( m_api.vkAllocateCommandBuffers( m_device, &info, out ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkAllocateCommandBuffers failed" );
  }
}

VkFence Context::createFence()
{
  VkFenceCreateInfo info{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  VkFence           fence = VK_NULL_HANDLE;
  if( m_api.vkCreateFence( m_device, &info, nullptr, &fence ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateFence failed" );
  }
  return fence;
}

VkSemaphore Context::createSemaphore()
{
  VkSemaphoreCreateInfo info{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
  VkSemaphore           semaphore = VK_NULL_HANDLE;
  if( m_api.vkCreateSemaphore( m_device, &info, nullptr, &semaphore ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateSemaphore failed" );
  }
  return semaphore;
}

VkQueryPool Context::createTimestampQueries( uint32_t count )
{
  VkQueryPoolCreateInfo info{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
  info.queryType      = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount     = count;
  VkQueryPool queries = VK_NULL_HANDLE;
  if( m_api.vkCreateQueryPool( m_device, &info, nullptr, &queries ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateQueryPool failed" );
  }
  return queries;
}

void Context::destroyCommandPool( VkCommandPool pool )
{
  if( pool )
  {
    m_api.vkDestroyCommandPool( m_device, pool, nullptr );
  }
}

void Context::destroyFence( VkFence fence )
{
  if( fence )
  {
    m_api.vkDestroyFence( m_device, fence, nullptr );
  }
}

void Context::destroySemaphore( VkSemaphore semaphore )
{
  if( semaphore )
  {
    m_api.vkDestroySemaphore( m_device, semaphore, nullptr );
  }
}

void Context::destroyQueries( VkQueryPool queries )
{
  if( queries )
  {
    m_api.vkDestroyQueryPool( m_device, queries, nullptr );
  }
}

}   // namespace gpu
}   // namespace vvdec
