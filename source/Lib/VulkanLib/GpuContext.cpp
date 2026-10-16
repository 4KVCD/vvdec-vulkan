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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <sys/stat.h>
#  include <unistd.h>
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
static constexpr uint32_t kVendorNvidia     = 0x10de;   // VkPhysicalDeviceProperties::vendorID
static constexpr uint32_t kVendorIntel      = 0x8086;

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
  options.pipelineCache = value( "VVDEC_GPU_CACHE" ) ? value( "VVDEC_GPU_CACHE" ) : "";
  options.async         = !( value( "VVDEC_GPU_ASYNC" ) && atoi( value( "VVDEC_GPU_ASYNC" ) ) == 0 );
  options.frontEnd      = !( value( "VVDEC_GPU_FRONT" ) && atoi( value( "VVDEC_GPU_FRONT" ) ) == 0 );
  options.frontEndSet   = value( "VVDEC_GPU_FRONT" ) != nullptr;
  options.frontEndCheck = value( "VVDEC_GPU_MIDER" ) && atoi( value( "VVDEC_GPU_MIDER" ) ) != 0;
  options.parse         = value( "VVDEC_GPU_PARSE" ) && atoi( value( "VVDEC_GPU_PARSE" ) ) == 1;
  options.parseDrives   = value( "VVDEC_GPU_PARSE" ) && atoi( value( "VVDEC_GPU_PARSE" ) ) == 2;
  options.precompile    = !( value( "VVDEC_GPU_PRECOMPILE" ) && atoi( value( "VVDEC_GPU_PRECOMPILE" ) ) == 0 );
  options.waveSize      = value( "VVDEC_GPU_WAVE" ) ? atoi( value( "VVDEC_GPU_WAVE" ) ) : -1;
  options.waveSizeInter = value( "VVDEC_GPU_WAVE_INTER" ) ? atoi( value( "VVDEC_GPU_WAVE_INTER" ) ) : -1;
  options.lowPriority   = !( value( "VVDEC_GPU_PRIORITY" ) && strcmp( value( "VVDEC_GPU_PRIORITY" ), "normal" ) == 0 );
  options.engine        = value( "VVDEC_GPU_ENGINE" ) ? value( "VVDEC_GPU_ENGINE" ) : "";
  return options;
}

// The device extensions the decoder uses where the device has them
static constexpr const char* kExtShaderInfo     = "VK_AMD_shader_info";              // the shaders' registers, machine code
static constexpr const char* kExtShaderClock    = "VK_KHR_shader_clock";             // the parse's profile
static constexpr const char* kExtSubgroupBallot = "VK_EXT_shader_subgroup_ballot";   // the parse's OpSubgroupReadInvocationKHR
static constexpr const char* kExtMaintenance5   = "VK_KHR_maintenance5";             // (the pipelines' binaries need it)
static constexpr const char* kExtPipelineBinary = "VK_KHR_pipeline_binary";          // the pipelines' binaries kept in a file
static constexpr const char* kExtExecutables    = "VK_KHR_pipeline_executable_properties";   // the shaders' statistics

bool Context::init()
{
  m_options = Options::fromEnvironment();
  if( !loadLibrary() || !createInstance() )
  {
    return false;
  }
  VkPhysicalDeviceProperties props;
  bool                       timelineSemaphore = false, subgroupSizeControl = false;
  if( !chooseDevice( props ) || !checkFeatures( props, timelineSemaphore, subgroupSizeControl ) )
  {
    return false;
  }
  if( m_intel && !m_options.frontEndSet && !m_options.parseDrives )
  {
    // Intel's GPUs: the CPU front end (VVDEC_GPU_FRONT=1: the GPU's). The GPU front end's motion derivation runs a
    // CTU row per 32-thread workgroup as a wavefront, a chain of CTUs that leaves most of an integrated GPU idle
    // (mider: 7.9 ms of a VideoQ 4K picture's 27 on the 285K's Intel Graphics). VideoQ 4K 35.4 -> 55.3 fps, 1080p
    // 90.2 -> 206.5 fps. The GPU parse (VVDEC_GPU_PARSE=2) needs the GPU front end: with it, the front end stays there.
    m_options.frontEnd = false;
  }
  uint32_t count = 0;
  m_api.vkGetPhysicalDeviceQueueFamilyProperties( m_physical, &count, nullptr );
  std::vector<VkQueueFamilyProperties> families( count );
  m_api.vkGetPhysicalDeviceQueueFamilyProperties( m_physical, &count, families.data() );
  if( !chooseQueueFamilies( families, timelineSemaphore, props.deviceName ) )
  {
    return false;
  }
  const AsyncQueueLayout async = chooseAsyncQueues( families );
  if( !createDevice( props, subgroupSizeControl, async ) )
  {
    return false;
  }
  getQueues( async );
  m_api.vkGetPhysicalDeviceMemoryProperties( m_physical, &m_memProps );
  if( !createPipelineLayout() )
  {
    return false;
  }
  msg( INFO, "Vulkan: %s\n", m_deviceName.c_str() );
  if( m_options.precompile )
  {
    startPrecompile();
  }
  return true;
}

bool Context::loadLibrary()
{
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
  return m_api.vkGetInstanceProcAddr != nullptr;
}

bool Context::createInstance()
{
  if( !s_instance )
  {
    auto createInstance = (PFN_vkCreateInstance) m_api.vkGetInstanceProcAddr( VK_NULL_HANDLE, "vkCreateInstance" );
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "vvdec";
    app.apiVersion       = VK_API_VERSION_1_2;
    // Vulkan 1.3 when the loader has it (VK_KHR_maintenance5, which the pipelines' binaries need, wants it)
    auto     enumerateVersion = (PFN_vkEnumerateInstanceVersion) m_api.vkGetInstanceProcAddr( VK_NULL_HANDLE, "vkEnumerateInstanceVersion" );
    uint32_t loaderVersion    = 0;
    if( enumerateVersion && enumerateVersion( &loaderVersion ) == VK_SUCCESS && loaderVersion >= VK_API_VERSION_1_3 )
    {
      app.apiVersion = VK_API_VERSION_1_3;
    }
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
  return true;
}

bool Context::chooseDevice( VkPhysicalDeviceProperties& props )
{
  uint32_t count = 0;
  m_api.vkEnumeratePhysicalDevices( m_instance, &count, nullptr );
  std::vector<VkPhysicalDevice> devices( count );
  m_api.vkEnumeratePhysicalDevices( m_instance, &count, devices.data() );
  if( devices.empty() )
  {
    msg( WARNING, "Vulkan: no device\n" );
    return false;
  }

  // VVDEC_GPU_DEVICE picks a device by its index; otherwise the first discrete GPU, then the first integrated one,
  // then the first device
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
      VkPhysicalDeviceProperties p;
      m_api.vkGetPhysicalDeviceProperties( devices[i], &p );
      if( p.deviceType == wanted && p.apiVersion >= VK_API_VERSION_1_2 )
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

  m_api.vkGetPhysicalDeviceProperties( m_physical, &props );
  m_deviceName      = props.deviceName;
  m_timestampPeriod = props.limits.timestampPeriod;
  m_discrete         = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  m_splitStores      = props.vendorID == kVendorNvidia;
  m_integerAddresses = props.vendorID == kVendorIntel;
  m_coherentReads    = props.vendorID == kVendorNvidia;
  m_intel            = props.vendorID == kVendorIntel;
  // The compute queue on the GPU's compute engine (a compute-only family) instead of the graphics engine, where the
  // desktop's compositor runs, on Intel's GPUs (VVDEC_GPU_ENGINE=compute or 3d chooses). Intel Graphics (285K, driver
  // 101.8991), VideoQ 4K beside a 60 Hz probe of the graphics queue: its submissions late for a frame 16.3% -> 0.1%;
  // VideoQ 1080p 94.4 -> 89.5 fps (the front end then shares the compute queue: on another queue of the compute
  // family, 85.8 fps)
  m_computeEngine = m_options.engine == "compute" || ( m_options.engine != "3d" && props.vendorID == kVendorIntel );
  if( props.apiVersion < VK_API_VERSION_1_2 || props.limits.maxPushConstantsSize < kPushConstantSize )
  {
    msg( WARNING, "Vulkan: %s lacks Vulkan 1.2 or 256 bytes of push constants\n", props.deviceName );
    return false;
  }
  return true;
}

bool Context::checkFeatures( const VkPhysicalDeviceProperties& props, bool& timelineSemaphore, bool& subgroupSizeControl )
{
  VkPhysicalDeviceVulkan12Features features12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
  VkPhysicalDeviceVulkan11Features features11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
  VkPhysicalDeviceSubgroupSizeControlFeatures sizeControl{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES };
  features12.pNext = &features11;
  features11.pNext = &sizeControl;
  VkPhysicalDeviceFeatures2 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
  features.pNext = &features12;
  m_api.vkGetPhysicalDeviceFeatures2( m_physical, &features );
  if( !features.features.shaderInt16 || !features.features.shaderInt64 || !features12.bufferDeviceAddress
      || !features12.storageBuffer8BitAccess || !features12.shaderInt8 || !features11.storageBuffer16BitAccess )
  {
    msg( WARNING, "Vulkan: %s lacks 8/16/64-bit integers or buffer device addresses\n", props.deviceName );
    return false;
  }
  timelineSemaphore   = features12.timelineSemaphore != VK_FALSE;
  subgroupSizeControl = sizeControl.subgroupSizeControl != VK_FALSE;
  return true;
}

bool Context::chooseQueueFamilies( const std::vector<VkQueueFamilyProperties>& families, bool timelineSemaphore, const char* deviceName )
{
  // a queue with compute: the first family that has it; with m_computeEngine the first without graphics, if any
  const uint32_t count = uint32_t( families.size() );
  m_queueFamily        = UINT32_MAX;
  for( uint32_t i = 0; i < count && m_queueFamily == UINT32_MAX && m_computeEngine; i++ )
  {
    if( ( families[i].queueFlags & VK_QUEUE_COMPUTE_BIT ) && !( families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT ) && families[i].timestampValidBits )
    {
      m_queueFamily = i;
    }
  }
  for( uint32_t i = 0; i < count && m_queueFamily == UINT32_MAX; i++ )
  {
    if( ( families[i].queueFlags & VK_QUEUE_COMPUTE_BIT ) && families[i].timestampValidBits )
    {
      m_queueFamily = i;
    }
  }
  if( m_queueFamily == UINT32_MAX )
  {
    msg( WARNING, "Vulkan: %s has no compute queue with timestamps\n", deviceName );
    return false;
  }

  // and a family for copies only (the copy engine): the pictures' downloads there run alongside the compute queue's
  // work
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
  for( uint32_t i = 0; i < count && m_asyncFamily == UINT32_MAX && timelineSemaphore && m_options.async; i++ )
  {
    if( i != m_queueFamily && ( families[i].queueFlags & VK_QUEUE_COMPUTE_BIT ) && !( families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT )
        && families[i].timestampValidBits )
    {
      m_asyncFamily = i;
    }
  }
  return true;
}

Context::AsyncQueueLayout Context::chooseAsyncQueues( const std::vector<VkQueueFamilyProperties>& families )
{
  AsyncQueueLayout layout;
  layout.familySize = m_asyncFamily != UINT32_MAX ? families[m_asyncFamily].queueCount : 0;
  m_asyncCount      = int( std::min<uint32_t>( layout.familySize, uint32_t( kMaxAsync ) ) );
  // the parse queue: one more of the family's when it has one (AMD's has 8); then the parse's copies' queue, one more
  layout.ownParseQueue = m_asyncCount > 0 && layout.familySize > uint32_t( m_asyncCount );
  layout.ownCopyQueue  = layout.ownParseQueue && layout.familySize > uint32_t( m_asyncCount + 1 );
  // Where the family has twice the front ends' queues (AMD's 8), the front ends take the odd queues and the parse and
  // its copies the even ones. Windows runs AMD's async family on two compute engines, each queue on the engine of its
  // index's parity, and the queues of an engine take turns at their submissions: a parse step (several ms) held up the
  // front ends that shared its engine (Top Gun slower 1080p +1.6 to +2.5%, Beekeeper 4K +1.0%).
  layout.split = layout.ownCopyQueue && m_asyncCount == kMaxAsync && layout.familySize >= uint32_t( 2 * kMaxAsync );
  for( int i = 0; i < m_asyncCount; i++ )
  {
    layout.frontIndex[i] = layout.split ? uint32_t( 2 * i + 1 ) : uint32_t( i );
  }
  layout.parseIndex = layout.split ? 0u : uint32_t( m_asyncCount );
  layout.copyIndex  = layout.split ? 2u : uint32_t( m_asyncCount + 1 );
  layout.grainIndex = 4u;   // (with split: one of the even queues the parse leaves, the film grain's)
  layout.queues     = layout.split ? uint32_t( 2 * kMaxAsync )
                                   : uint32_t( m_asyncCount + ( layout.ownParseQueue ? 1 : 0 ) + ( layout.ownCopyQueue ? 1 : 0 ) );
  return layout;
}

Context::DeviceExtensions Context::findExtensions() const
{
  DeviceExtensions found;
  // VVDEC_GPU_PROFILE=1: the GPU time of the passes, and with VK_AMD_shader_info the registers and memory each shader
  // takes; VVDEC_GPU_ISA=<directory> needs VK_AMD_shader_info too, for the shaders' machine code
  const bool wantShaderInfo = m_options.profile || !m_options.isaDir.empty();
  uint32_t   count          = 0;
  m_api.vkEnumerateDeviceExtensionProperties( m_physical, nullptr, &count, nullptr );
  std::vector<VkExtensionProperties> extensions( count );
  m_api.vkEnumerateDeviceExtensionProperties( m_physical, nullptr, &count, extensions.data() );
  for( const VkExtensionProperties& extension: extensions )
  {
    found.shaderInfo |= wantShaderInfo && strcmp( extension.extensionName, kExtShaderInfo ) == 0;
    found.shaderClock |= m_options.profile && strcmp( extension.extensionName, kExtShaderClock ) == 0;
    found.subgroupBallot |= strcmp( extension.extensionName, kExtSubgroupBallot ) == 0;
    found.maintenance5 |= strcmp( extension.extensionName, kExtMaintenance5 ) == 0;
    found.pipelineBinary |= strcmp( extension.extensionName, kExtPipelineBinary ) == 0;
    found.executables |= m_options.profile && strcmp( extension.extensionName, kExtExecutables ) == 0;
    if( strcmp( extension.extensionName, "VK_KHR_global_priority" ) == 0 )
    {
      found.globalPriority = "VK_KHR_global_priority";
    }
    else if( !found.globalPriority && strcmp( extension.extensionName, "VK_EXT_global_priority" ) == 0 )
    {
      found.globalPriority = "VK_EXT_global_priority";
    }
  }
  return found;
}

// The queues to create into infos: the compute queue, the copy queue, and the async family's (async.queues). Returns
// their number.
uint32_t Context::queueCreateInfos( const AsyncQueueLayout& async, VkDeviceQueueCreateInfo infos[3] ) const
{
  // every queue gets the same priority
  static const float kPriorities[2 * kMaxAsync] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
  uint32_t           count                      = 0;
  for( const uint32_t family: { m_queueFamily, m_transferFamily, m_asyncFamily } )
  {
    if( family != UINT32_MAX )
    {
      infos[count]                  = VkDeviceQueueCreateInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
      infos[count].queueFamilyIndex = family;
      infos[count].queueCount       = family == m_asyncFamily ? async.queues : 1;
      infos[count].pQueuePriorities = kPriorities;
      count++;
    }
  }
  return count;
}

// Whether the shaders may ask for waves of 32 threads, on GPUs that run both sizes (AMD's), where the shaders run
// better so: it takes Vulkan 1.3's subgroup size control for compute shaders, with 32 among the sizes
bool Context::wave32Possible( const VkPhysicalDeviceProperties& props, bool subgroupSizeControl )
{
  VkPhysicalDeviceSubgroupSizeControlProperties sizes{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES };
  VkPhysicalDeviceProperties2                   props2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
  props2.pNext = &sizes;
  m_api.vkGetPhysicalDeviceProperties2( m_physical, &props2 );
  if( props.apiVersion >= VK_API_VERSION_1_3 && subgroupSizeControl && ( sizes.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT ) )
  {
    m_minWave = sizes.minSubgroupSize;
    m_maxWave = sizes.maxSubgroupSize;
  }
  return props.apiVersion >= VK_API_VERSION_1_3 && subgroupSizeControl && sizes.minSubgroupSize <= 32 && sizes.maxSubgroupSize >= 32
         && ( sizes.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT );
}

// Whether the device has VK_KHR_shader_clock's subgroup clock and device clock (for the parse's profile:
// shaders/parse.slang's PROFILE)
bool Context::hasShaderClocks() const
{
  VkPhysicalDeviceShaderClockFeaturesKHR clock{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR };
  VkPhysicalDeviceFeatures2              features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
  features.pNext = &clock;
  m_api.vkGetPhysicalDeviceFeatures2( m_physical, &features );
  return clock.shaderSubgroupClock && clock.shaderDeviceClock;
}

// Whether the device can give out the compiled pipelines' binaries (VK_KHR_pipeline_binary, which needs
// VK_KHR_maintenance5)
bool Context::hasPipelineBinaries() const
{
  VkPhysicalDeviceMaintenance5FeaturesKHR   maintenance5{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR };
  VkPhysicalDevicePipelineBinaryFeaturesKHR binaries{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR };
  VkPhysicalDeviceFeatures2                 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
  maintenance5.pNext = &binaries;
  features.pNext     = &maintenance5;
  m_api.vkGetPhysicalDeviceFeatures2( m_physical, &features );
  return maintenance5.maintenance5 && binaries.pipelineBinaries;
}

// The features vkCreateDevice turns on, chained from enable, and the names of the device extensions it turns on. The
// chain points into the structure, so it is never copied.
struct DeviceFeatures
{
  VkPhysicalDeviceFeatures2                   enable{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
  VkPhysicalDeviceVulkan12Features            enable12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
  VkPhysicalDeviceVulkan11Features            enable11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
  VkPhysicalDeviceSubgroupSizeControlFeatures subgroupSize{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES };
  VkPhysicalDeviceShaderClockFeaturesKHR      clock{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR };
  VkPhysicalDeviceMaintenance5FeaturesKHR     maintenance5{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR };
  VkPhysicalDevicePipelineBinaryFeaturesKHR   binaries{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR };
  VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR executables{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR };
  std::vector<const char*>                    extensions;

  // The features the decoder needs (Context::checkFeatures): timeline semaphores for the async queues, and subgroup size
  // control for waves of 32 threads
  DeviceFeatures( bool timelineSemaphore, bool subgroupSizeControl )
  {
    enable.features.shaderInt16       = VK_TRUE;
    enable.features.shaderInt64       = VK_TRUE;
    enable12.bufferDeviceAddress      = VK_TRUE;
    enable12.storageBuffer8BitAccess  = VK_TRUE;
    enable12.shaderInt8               = VK_TRUE;
    enable12.timelineSemaphore        = timelineSemaphore ? VK_TRUE : VK_FALSE;
    enable11.storageBuffer16BitAccess = VK_TRUE;
    subgroupSize.subgroupSizeControl  = VK_TRUE;
    if( subgroupSizeControl )
    {
      chain( subgroupSize );
    }
    chain( enable11 );
    chain( enable12 );
  }
  DeviceFeatures( const DeviceFeatures& )            = delete;
  DeviceFeatures& operator=( const DeviceFeatures& ) = delete;

  // VK_KHR_shader_clock's subgroup and device clocks
  void enableShaderClock()
  {
    clock.shaderSubgroupClock = VK_TRUE;
    clock.shaderDeviceClock   = VK_TRUE;
    chain( clock );
    extensions.push_back( kExtShaderClock );
  }

  // VK_KHR_pipeline_executable_properties' statistics of the pipelines
  void enableExecutableStatistics()
  {
    executables.pipelineExecutableInfo = VK_TRUE;
    chain( executables );
    extensions.push_back( kExtExecutables );
  }

  // VK_KHR_pipeline_binary, with the VK_KHR_maintenance5 it needs
  void enablePipelineBinaries()
  {
    binaries.pipelineBinaries = VK_TRUE;
    maintenance5.maintenance5 = VK_TRUE;
    chain( binaries );
    chain( maintenance5 );
    extensions.push_back( kExtMaintenance5 );
    extensions.push_back( kExtPipelineBinary );
  }

  // Puts feature (with nothing chained to it yet) at the front of the chain.
  template<typename Feature>
  void chain( Feature& feature )
  {
    feature.pNext = enable.pNext;
    enable.pNext  = &feature;
  }
};

bool Context::createDevice( const VkPhysicalDeviceProperties& props, bool subgroupSizeControl, const AsyncQueueLayout& async )
{
  VkDeviceQueueCreateInfo queueInfos[3];
  const uint32_t          numQueueInfos = queueCreateInfos( async, queueInfos );
  m_wave32                              = wave32Possible( props, subgroupSizeControl );

  // The features the decoder needs, then the extensions it uses where the device has them: VVDEC_GPU_PROFILE=1's and
  // VVDEC_GPU_ISA's (findExtensions), the parse's OpSubgroupReadInvocationKHR (GpuContext.h's subgroupBallot), and the
  // compiled pipelines' binaries kept in a file (PipelineBinaries), unless VVDEC_GPU_CACHE=0.
  DeviceFeatures         features( asyncQueue(), m_wave32 );
  const DeviceExtensions found = findExtensions();
  const bool binaries = found.maintenance5 && found.pipelineBinary && props.apiVersion >= VK_API_VERSION_1_3 && m_options.pipelineCache != "0"
                        && hasPipelineBinaries();
  m_profiling      = m_options.profile;
  m_shaderClock    = found.shaderClock && hasShaderClocks();
  m_subgroupBallot = found.subgroupBallot;
  if( found.shaderInfo )
  {
    features.extensions.push_back( kExtShaderInfo );
  }
  // (VVDEC_GPU_PROFILE=1 without VK_AMD_shader_info: the shaders' statistics, VK_KHR_pipeline_executable_properties)
  const bool executables = found.executables && !found.shaderInfo;
  if( executables )
  {
    features.enableExecutableStatistics();
  }
  if( m_shaderClock )
  {
    features.enableShaderClock();
  }
  if( m_subgroupBallot )
  {
    features.extensions.push_back( kExtSubgroupBallot );
  }
  if( binaries )
  {
    features.enablePipelineBinaries();
  }

  // Every queue at low global priority (VVDEC_GPU_PRIORITY=normal: not), so that other programs' GPU work, the
  // desktop's, goes first; a device that refuses it is made without. Intel Graphics, VideoQ 4K beside a 60 Hz probe
  // of the graphics queue, the compute queue on the graphics engine: probe submissions late for a frame 16.3% -> 7.6%,
  // the decode's speed unchanged
  VkDeviceQueueGlobalPriorityCreateInfoKHR lowPriority{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR };
  lowPriority.globalPriority = VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR;
  const bool priority        = m_options.lowPriority && found.globalPriority;
  if( priority )
  {
    for( uint32_t i = 0; i < numQueueInfos; i++ )
    {
      queueInfos[i].pNext = &lowPriority;
    }
    features.extensions.push_back( found.globalPriority );
  }

  VkDeviceCreateInfo deviceInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  deviceInfo.pNext                   = &features.enable;
  deviceInfo.queueCreateInfoCount    = numQueueInfos;
  deviceInfo.pQueueCreateInfos       = queueInfos;
  deviceInfo.enabledExtensionCount   = uint32_t( features.extensions.size() );
  deviceInfo.ppEnabledExtensionNames = features.extensions.empty() ? nullptr : features.extensions.data();
  VkResult created                   = m_api.vkCreateDevice( m_physical, &deviceInfo, nullptr, &m_device );
  if( created != VK_SUCCESS && priority )
  {
    for( uint32_t i = 0; i < numQueueInfos; i++ )
    {
      queueInfos[i].pNext = nullptr;
    }
    features.extensions.pop_back();
    deviceInfo.enabledExtensionCount   = uint32_t( features.extensions.size() );
    deviceInfo.ppEnabledExtensionNames = features.extensions.empty() ? nullptr : features.extensions.data();
    created                            = m_api.vkCreateDevice( m_physical, &deviceInfo, nullptr, &m_device );
    msg( WARNING, "Vulkan: %s refused low queue priorities\n", props.deviceName );
  }
  if( created != VK_SUCCESS )
  {
    msg( WARNING, "Vulkan: creating the device on %s failed\n", props.deviceName );
    return false;
  }

#define VVDEC_VK_LOAD_DEVICE( name ) m_api.name = (PFN_##name) m_api.vkGetDeviceProcAddr( m_device, #name );
  VVDEC_VK_DEVICE_FUNCS( VVDEC_VK_LOAD_DEVICE )
#undef VVDEC_VK_LOAD_DEVICE
  if( found.shaderInfo )
  {
    m_getShaderInfo = (PFN_vkGetShaderInfoAMD) m_api.vkGetDeviceProcAddr( m_device, "vkGetShaderInfoAMD" );
  }
  if( executables )
  {
    m_getExecutables = (PFN_vkGetPipelineExecutablePropertiesKHR) m_api.vkGetDeviceProcAddr( m_device, "vkGetPipelineExecutablePropertiesKHR" );
    m_getExecutableStatistics
      = (PFN_vkGetPipelineExecutableStatisticsKHR) m_api.vkGetDeviceProcAddr( m_device, "vkGetPipelineExecutableStatisticsKHR" );
  }
  if( binaries )
  {
    openPipelineBinaries( props );
  }
  return true;
}

void Context::getQueues( const AsyncQueueLayout& async )
{
  m_api.vkGetDeviceQueue( m_device, m_queueFamily, 0, &m_queue );
  if( transferQueue() )
  {
    m_api.vkGetDeviceQueue( m_device, m_transferFamily, 0, &m_transferQueue );
  }
  if( !asyncQueue() )
  {
    return;
  }
  for( int i = 0; i < m_asyncCount; i++ )
  {
    m_api.vkGetDeviceQueue( m_device, m_asyncFamily, async.frontIndex[i], &m_asyncQueues[i] );
  }
  if( async.ownParseQueue )
  {
    m_api.vkGetDeviceQueue( m_device, m_asyncFamily, async.parseIndex, &m_parseQueue );
  }
  else
  {
    m_parseQueue = m_asyncQueues[m_asyncCount - 1];
    m_parseMutex = &m_asyncMutexes[m_asyncCount - 1];
  }
  if( async.ownCopyQueue )
  {
    m_api.vkGetDeviceQueue( m_device, m_asyncFamily, async.copyIndex, &m_parseCopyQueue );
  }
  else
  {
    m_parseCopyQueue = m_parseQueue;
    m_parseCopyMutex = m_parseMutex;
  }
  // Film grain gets a queue of its own only where the queues are split; elsewhere it uses the compute queue.
  if( async.split )
  {
    m_api.vkGetDeviceQueue( m_device, m_asyncFamily, async.grainIndex, &m_grainQueue );
  }
  VkSemaphoreTypeCreateInfo timelineType{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
  timelineType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo timelineInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
  timelineInfo.pNext = &timelineType;
  if( m_api.vkCreateSemaphore( m_device, &timelineInfo, nullptr, &m_timeline ) != VK_SUCCESS
      || m_api.vkCreateSemaphore( m_device, &timelineInfo, nullptr, &m_parseTimeline ) != VK_SUCCESS
      || m_api.vkCreateSemaphore( m_device, &timelineInfo, nullptr, &m_parseCopyTimeline ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateSemaphore failed (the timeline semaphores)" );
  }
}

bool Context::createPipelineLayout()
{
  VkPushConstantRange        range{ VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushConstantSize };
  VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  layoutInfo.pushConstantRangeCount = 1;
  layoutInfo.pPushConstantRanges    = &range;
  if( m_api.vkCreatePipelineLayout( m_device, &layoutInfo, nullptr, &m_layout ) != VK_SUCCESS )
  {
    return false;
  }
  m_pipelines    = new VkPipeline[kShaderCount];
  m_pipelineOnce = new std::once_flag[kShaderCount];
  for( int i = 0; i < kShaderCount; i++ )
  {
    m_pipelines[i] = VK_NULL_HANDLE;
  }
  return true;
}

// ====================================================================================================================
// The compiled pipelines' binaries
// ====================================================================================================================
// The driver keeps the pipelines it compiled in a cache of its own, but per program. So the first run of a program it
// has not seen (a new or updated one, or any program after a driver update) compiled every shader again on first use,
// and VideoQ 1080p showed its first frame 12.3 s after the start instead of 0.85 s. A VkPipelineCache of the decoder's own does not
// help on AMD's driver (it compiles again, the cache's entries being per program too). With VK_KHR_pipeline_binary the
// decoder keeps each pipeline's binaries itself and makes the pipeline from them, whatever the program: a set of
// shaders is compiled once per driver. The binaries are kept in a file per device, with the driver's global key
// (binaries of another driver do not fit) and the hash of the decoder's shaders (a new set starts the file anew, so it
// does not grow with every version); it is rewritten when binaries were added.

struct Context::PipelineBinaries
{
  PFN_vkGetPipelineKeyKHR              getKey  = nullptr;
  PFN_vkCreatePipelineBinariesKHR      create  = nullptr;
  PFN_vkDestroyPipelineBinaryKHR       destroy = nullptr;
  PFN_vkGetPipelineBinaryDataKHR       getData = nullptr;
  PFN_vkReleaseCapturedPipelineDataKHR release = nullptr;

  struct Binary
  {
    std::vector<uint8_t> key, data;
  };
  std::map<std::vector<uint8_t>, std::vector<Binary>> pipelines;   // a pipeline's key -> its binaries
  std::vector<uint8_t> globalKey;                                  // the driver's
  uint64_t             shadersHash = 0;
  std::string          path;
  bool                 added = false;   // binaries added since the file was read

  // the binaries in the file, when it is whole and of this driver and these shaders (else none)
  std::map<std::vector<uint8_t>, std::vector<Binary>> read() const;
  void                                                write() const;
};

static constexpr uint32_t kBinariesMagic   = 0x42445656;   // "VVDB"
static constexpr uint32_t kBinariesVersion = 2;

static constexpr uint64_t kHashSeed = 0x9e3779b97f4a7c15ull;

// h updated with a shader's SPIR-V words
static uint64_t hashShader( uint64_t h, int shaderId )
{
  for( size_t w = 0; w < kShaders[shaderId].size / sizeof( uint32_t ); w++ )
  {
    h = ( h ^ kShaders[shaderId].words[w] ) * 0xff51afd7ed558ccdull;
    h = ( h << 29 ) | ( h >> 35 );
  }
  return h;
}

// A hash of all the shaders' SPIR-V words (a millisecond or so)
static uint64_t shadersHash()
{
  uint64_t h = kHashSeed;
  for( int i = 0; i < kShaderCount; i++ )
  {
    h = hashShader( h, i );
  }
  return h;
}

// A pipeline's key in the binaries' file: its shader (its name and its SPIR-V's hash), the wave size it asks for (0:
// none) and the work-arounds set (kSplitStores, kIntegerAddresses, kCoherentReads). (Not the driver's pipeline
// key, vkGetPipelineKeyKHR: AMD's makes it per program, as its shader cache.)
static std::vector<uint8_t> pipelineKeyOf( int shaderId, uint32_t waveSize, bool splitStores, bool integerAddresses, bool coherentReads )
{
  const char*          name = kShaders[shaderId].name;
  std::vector<uint8_t> key( name, name + strlen( name ) );
  const uint64_t       h = hashShader( kHashSeed, shaderId );
  key.insert( key.end(), reinterpret_cast<const uint8_t*>( &h ), reinterpret_cast<const uint8_t*>( &h ) + sizeof( h ) );
  key.push_back( uint8_t( ( splitStores ? 1 : 0 ) | ( integerAddresses ? 2 : 0 ) | ( coherentReads ? 4 : 0 ) ) );
  key.push_back( uint8_t( waveSize ) );
  return key;
}

// The file name in the user's cache directory (its directory made when missing), or empty when there is none
static std::string defaultCacheFile( const char* name )
{
#ifdef _WIN32
  const char* base = getenv( "LOCALAPPDATA" );
  if( !base || !*base )
  {
    return "";
  }
  const std::string dir = std::string( base ) + "\\vvdec";
  CreateDirectoryA( dir.c_str(), nullptr );   // (fails when it exists)
  return dir + "\\" + name;
#else
  const char*       xdg  = getenv( "XDG_CACHE_HOME" );
  const char*       home = getenv( "HOME" );
  const std::string base = xdg && *xdg ? std::string( xdg ) : home && *home ? std::string( home ) + "/.cache" : std::string();
  if( base.empty() )
  {
    return "";
  }
  mkdir( base.c_str(), 0755 );   // (fails when it exists)
  const std::string dir = base + "/vvdec";
  mkdir( dir.c_str(), 0755 );
  return dir + "/" + name;
#endif
}

std::map<std::vector<uint8_t>, std::vector<Context::PipelineBinaries::Binary>> Context::PipelineBinaries::read() const
{
  std::map<std::vector<uint8_t>, std::vector<Binary>> read;
  FILE*                                               f = fopen( path.c_str(), "rb" );
  if( !f )
  {
    return read;
  }
  std::vector<uint8_t> file;
  uint8_t              chunk[65536];
  size_t               got;
  while( ( got = fread( chunk, 1, sizeof( chunk ), f ) ) > 0 )
  {
    file.insert( file.end(), chunk, chunk + got );
  }
  fclose( f );
  // (each read checked against the file's end: a cut or damaged file leaves no binaries)
  size_t     at = 0;
  const auto take = [&]( void* to, size_t size ) {
    if( at + size > file.size() )
    {
      return false;
    }
    memcpy( to, file.data() + at, size );
    at += size;
    return true;
  };
  const auto takeBytes = [&]( std::vector<uint8_t>& to, size_t size ) {
    if( at + size > file.size() )
    {
      return false;
    }
    to.assign( file.begin() + at, file.begin() + at + size );
    at += size;
    return true;
  };
  uint32_t             magic = 0, version = 0, keySize = 0, count = 0;
  uint64_t             hash = 0;
  std::vector<uint8_t> fileGlobalKey;
  if( !take( &magic, 4 ) || magic != kBinariesMagic || !take( &version, 4 ) || version != kBinariesVersion || !take( &hash, 8 ) || hash != shadersHash
      || !take( &keySize, 4 ) || keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR || !takeBytes( fileGlobalKey, keySize ) || fileGlobalKey != globalKey
      || !take( &count, 4 ) )
  {
    return read;
  }
  for( uint32_t p = 0; p < count; p++ )
  {
    std::vector<uint8_t> key;
    uint32_t             binaryCount = 0;
    if( !take( &keySize, 4 ) || keySize > 256 || !takeBytes( key, keySize ) || !take( &binaryCount, 4 ) || binaryCount > 64 )
    {
      return {};
    }
    std::vector<Binary>& binaries = read[key];
    for( uint32_t b = 0; b < binaryCount; b++ )
    {
      Binary   binary;
      uint64_t dataSize = 0;
      if( !take( &keySize, 4 ) || keySize > VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR || !takeBytes( binary.key, keySize ) || !take( &dataSize, 8 )
          || dataSize > file.size() || !takeBytes( binary.data, size_t( dataSize ) ) )
      {
        return {};
      }
      binaries.push_back( std::move( binary ) );
    }
  }
  return read;
}

// Writes the binaries beside the file, then puts them in its place at once (another decoder may be reading it)
void Context::PipelineBinaries::write() const
{
#ifdef _WIN32
  const std::string temp = path + ".tmp" + std::to_string( GetCurrentProcessId() );
#else
  const std::string temp = path + ".tmp" + std::to_string( getpid() );
#endif
  FILE* f = fopen( temp.c_str(), "wb" );
  if( !f )
  {
    return;
  }
  bool       ok  = true;
  const auto put = [&]( const void* from, size_t size ) { ok = ok && fwrite( from, 1, size, f ) == size; };
  const uint32_t keySize = uint32_t( globalKey.size() ), count = uint32_t( pipelines.size() );
  put( &kBinariesMagic, 4 );
  put( &kBinariesVersion, 4 );
  put( &shadersHash, 8 );
  put( &keySize, 4 );
  put( globalKey.data(), globalKey.size() );
  put( &count, 4 );
  for( const auto& pipeline: pipelines )
  {
    const uint32_t pipelineKeySize = uint32_t( pipeline.first.size() ), binaryCount = uint32_t( pipeline.second.size() );
    put( &pipelineKeySize, 4 );
    put( pipeline.first.data(), pipeline.first.size() );
    put( &binaryCount, 4 );
    for( const Binary& binary: pipeline.second )
    {
      const uint32_t binaryKeySize = uint32_t( binary.key.size() );
      const uint64_t dataSize      = binary.data.size();
      put( &binaryKeySize, 4 );
      put( binary.key.data(), binary.key.size() );
      put( &dataSize, 8 );
      put( binary.data.data(), binary.data.size() );
    }
  }
  ok = fclose( f ) == 0 && ok;
#ifdef _WIN32
  if( !ok || !MoveFileExA( temp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING ) )
#else
  if( !ok || rename( temp.c_str(), path.c_str() ) != 0 )
#endif
  {
    remove( temp.c_str() );
  }
}

void Context::openPipelineBinaries( const VkPhysicalDeviceProperties& props )
{
  PipelineBinaries* b = new PipelineBinaries;
  b->getKey           = (PFN_vkGetPipelineKeyKHR) m_api.vkGetDeviceProcAddr( m_device, "vkGetPipelineKeyKHR" );
  b->create           = (PFN_vkCreatePipelineBinariesKHR) m_api.vkGetDeviceProcAddr( m_device, "vkCreatePipelineBinariesKHR" );
  b->destroy          = (PFN_vkDestroyPipelineBinaryKHR) m_api.vkGetDeviceProcAddr( m_device, "vkDestroyPipelineBinaryKHR" );
  b->getData          = (PFN_vkGetPipelineBinaryDataKHR) m_api.vkGetDeviceProcAddr( m_device, "vkGetPipelineBinaryDataKHR" );
  b->release          = (PFN_vkReleaseCapturedPipelineDataKHR) m_api.vkGetDeviceProcAddr( m_device, "vkReleaseCapturedPipelineDataKHR" );
  VkPipelineBinaryKeyKHR globalKey{ VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR };
  if( !b->getKey || !b->create || !b->destroy || !b->getData || !b->release || b->getKey( m_device, nullptr, &globalKey ) != VK_SUCCESS )
  {
    delete b;
    return;
  }
  b->globalKey.assign( globalKey.key, globalKey.key + globalKey.keySize );
  b->shadersHash = shadersHash();
  char name[64];
  snprintf( name, sizeof( name ), "vulkan-pipelines-%04x-%04x.bin", props.vendorID, props.deviceID );
  b->path = m_options.pipelineCache.empty() ? defaultCacheFile( name ) : m_options.pipelineCache;
  if( !b->path.empty() )
  {
    b->pipelines = b->read();
  }
  m_binaries = b;
}

void Context::closePipelineBinaries()
{
  if( m_binaries && m_binaries->added && !m_binaries->path.empty() )
  {
    // with the binaries another decoder added to the file meanwhile (decoders running side by side)
    for( auto& pipeline: m_binaries->read() )
    {
      m_binaries->pipelines.insert( std::move( pipeline ) );   // (ours kept where both have one)
    }
    m_binaries->write();
  }
  delete m_binaries;
  m_binaries = nullptr;
}

VkResult Context::createPipeline( int shaderId, uint32_t waveSize, VkComputePipelineCreateInfo& info, VkPipeline& pipe )
{
  if( !m_binaries )
  {
    return m_api.vkCreateComputePipelines( m_device, VK_NULL_HANDLE, 1, &info, nullptr, &pipe );
  }
  const std::vector<uint8_t> key = pipelineKeyOf( shaderId, waveSize, m_splitStores, m_integerAddresses, m_coherentReads );
  if( createFromBinaries( key, info, pipe ) )
  {
    return VK_SUCCESS;
  }
  // Compiled, its binaries captured for the file
  VkPipelineCreateFlags2CreateInfoKHR flags{ VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR };
  flags.flags = VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR;
  flags.pNext = info.pNext;
  info.pNext  = &flags;
  const VkResult result = m_api.vkCreateComputePipelines( m_device, VK_NULL_HANDLE, 1, &info, nullptr, &pipe );
  info.pNext            = flags.pNext;
  if( result == VK_SUCCESS )
  {
    captureBinaries( key, pipe );
  }
  return result;
}

// Creates the pipeline from the binaries the file has for key, if any; binaries the driver refuses are dropped (the
// pipeline is then compiled).
bool Context::createFromBinaries( const std::vector<uint8_t>& key, VkComputePipelineCreateInfo& info, VkPipeline& pipe )
{
  PipelineBinaries*                     b = m_binaries;
  std::vector<PipelineBinaries::Binary> binaries;
  {
    std::lock_guard<std::mutex> lock( m_pipelineMutex );
    const auto                  found = b->pipelines.find( key );
    if( found == b->pipelines.end() || found->second.empty() )
    {
      return false;
    }
    binaries = found->second;
  }
  std::vector<VkPipelineBinaryKeyKHR>  keys;
  std::vector<VkPipelineBinaryDataKHR> data;
  for( const PipelineBinaries::Binary& binary: binaries )
  {
    VkPipelineBinaryKeyKHR k{ VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR };
    k.keySize = uint32_t( binary.key.size() );
    memcpy( k.key, binary.key.data(), binary.key.size() );
    keys.push_back( k );
    data.push_back( VkPipelineBinaryDataKHR{ binary.data.size(), const_cast<uint8_t*>( binary.data.data() ) } );
  }
  VkPipelineBinaryKeysAndDataKHR keysAndData{ uint32_t( keys.size() ), keys.data(), data.data() };
  VkPipelineBinaryCreateInfoKHR  createInfo{ VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR };
  createInfo.pKeysAndDataInfo = &keysAndData;
  std::vector<VkPipelineBinaryKHR> handles( keys.size(), VK_NULL_HANDLE );
  VkPipelineBinaryHandlesInfoKHR   handlesInfo{ VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR };
  handlesInfo.pipelineBinaryCount = uint32_t( handles.size() );
  handlesInfo.pPipelineBinaries   = handles.data();
  VkResult result = b->create( m_device, &createInfo, nullptr, &handlesInfo );
  if( result == VK_SUCCESS )
  {
    VkPipelineBinaryInfoKHR binaryInfo{ VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR };
    binaryInfo.binaryCount       = uint32_t( handles.size() );
    binaryInfo.pPipelineBinaries = handles.data();
    binaryInfo.pNext             = info.pNext;
    info.pNext                   = &binaryInfo;
    result                       = m_api.vkCreateComputePipelines( m_device, VK_NULL_HANDLE, 1, &info, nullptr, &pipe );
    info.pNext                   = binaryInfo.pNext;
  }
  for( VkPipelineBinaryKHR handle: handles )
  {
    if( handle )
    {
      b->destroy( m_device, handle, nullptr );
    }
  }
  if( result != VK_SUCCESS )
  {
    std::lock_guard<std::mutex> lock( m_pipelineMutex );
    b->pipelines.erase( key );
  }
  return result == VK_SUCCESS;
}

// Keeps the binaries of pipe (created with VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR) under key, for the file, and
// releases the captured data
void Context::captureBinaries( const std::vector<uint8_t>& key, VkPipeline pipe )
{
  PipelineBinaries*             b = m_binaries;
  VkPipelineBinaryCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR };
  createInfo.pipeline = pipe;
  VkPipelineBinaryHandlesInfoKHR handlesInfo{ VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR };
  if( b->create( m_device, &createInfo, nullptr, &handlesInfo ) == VK_SUCCESS && handlesInfo.pipelineBinaryCount > 0 )
  {
    std::vector<VkPipelineBinaryKHR> handles( handlesInfo.pipelineBinaryCount, VK_NULL_HANDLE );
    handlesInfo.pPipelineBinaries = handles.data();
    if( b->create( m_device, &createInfo, nullptr, &handlesInfo ) == VK_SUCCESS )
    {
      std::vector<PipelineBinaries::Binary> binaries;
      for( VkPipelineBinaryKHR handle: handles )
      {
        VkPipelineBinaryDataInfoKHR dataInfo{ VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR };
        dataInfo.pipelineBinary = handle;
        VkPipelineBinaryKeyKHR binaryKey{ VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR };
        size_t                 size = 0;
        if( b->getData( m_device, &dataInfo, &binaryKey, &size, nullptr ) == VK_SUCCESS && size > 0 )
        {
          PipelineBinaries::Binary binary;
          binary.data.resize( size );
          if( b->getData( m_device, &dataInfo, &binaryKey, &size, binary.data.data() ) == VK_SUCCESS )
          {
            binary.key.assign( binaryKey.key, binaryKey.key + binaryKey.keySize );
            binary.data.resize( size );
            binaries.push_back( std::move( binary ) );
          }
        }
        b->destroy( m_device, handle, nullptr );
      }
      if( binaries.size() == handles.size() )
      {
        std::lock_guard<std::mutex> lock( m_pipelineMutex );
        b->pipelines[key] = std::move( binaries );
        b->added          = true;
      }
    }
  }
  VkReleaseCapturedPipelineDataInfoKHR releaseInfo{ VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR };
  releaseInfo.pipeline = pipe;
  b->release( m_device, &releaseInfo, nullptr );
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
  const VkDeviceSize rounded = alignUp( size, 256 );
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

Buffer Context::allocHost( size_t size, MemKind kind )
{
  {
    std::lock_guard<std::mutex> lock( m_hostCacheMutex );
    // the smallest that fits, so that a larger one stays for a larger picture
    auto best = m_hostCache.end();
    for( auto it = m_hostCache.begin(); it != m_hostCache.end(); ++it )
    {
      const bool fits = it->kind == kind && it->size >= size && it->size <= 2 * size;
      if( fits && ( best == m_hostCache.end() || it->size < best->size ) )
      {
        best = it;
      }
    }
    if( best != m_hostCache.end() )
    {
      const Buffer buf = *best;
      m_hostCacheBytes -= size_t( buf.size );
      m_hostCache.erase( best );
      return buf;
    }
  }
  return alloc( size, kind );
}

void Context::keepHost( Buffer& buf )
{
  if( !buf.valid() )
  {
    return;
  }
  std::vector<Buffer> evicted;
  {
    std::lock_guard<std::mutex> lock( m_hostCacheMutex );
    m_hostCache.push_back( buf );
    m_hostCacheBytes += size_t( buf.size );
    while( m_hostCacheBytes > kHostCacheBytes )
    {
      m_hostCacheBytes -= size_t( m_hostCache.front().size );
      evicted.push_back( m_hostCache.front() );
      m_hostCache.pop_front();
    }
  }
  for( Buffer& old: evicted )
  {
    free( old );   // (outside the lock: freeing memory takes time)
  }
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
  stopPrecompile();
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
    for( Buffer& buf: m_hostCache )
    {
      free( buf );
    }
    m_hostCache.clear();
    closePipelineBinaries();
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
    if( m_parseCopyTimeline )
    {
      m_api.vkDestroySemaphore( m_device, m_parseCopyTimeline, nullptr );
    }
    m_api.vkDestroyDevice( m_device, nullptr );
  }
  delete[] m_pipelines;
  delete[] m_pipelineOnce;
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
  buf.size = alignUp( size, 256 );

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
  CHECK_FATAL( buf.isRange(), "Vulkan: a range of a larger buffer freed as a buffer (RangeArena::give takes it back)" );
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

void Context::ensure( Buffer& buf, size_t size, MemKind kind )
{
  static constexpr size_t kGranule = 65536;
  if( buf.size < size )
  {
    free( buf );
    buf = alloc( alignUp( size, kGranule ), kind );
  }
}

void Context::freeAfterQueued( Buffer& buf )
{
  if( !buf.valid() )
  {
    return;
  }
  {
    // (under the queue's lock nothing is queued meanwhile: the wait is for the work queued so far)
    std::lock_guard<std::mutex> lock( m_queueMutex );
    const VkResult result = m_api.vkQueueWaitIdle( m_queue );
    if( result != VK_SUCCESS )
    {
      THROW_FATAL( "Vulkan: waiting for the compute queue failed (" << result << ")" );
    }
  }
  free( buf );
}

// Whether the shader is one of inter's variants, which run in waves of 64 threads (the driver's choice). VideoQ 4K on
// a Radeon 780M with all the shaders in waves of 32: intra 1.50 -> 1.31 ms, ALF luma 1.54 -> 1.36, the residuals
// 0.33 -> 0.28, but inter 4.4 -> 5.2 ms; Beekeeper 4K slower preset: BDOF's tiles 1.31 -> 1.23 ms in waves of 64.
static bool interShader( int shaderId )
{
  return shaderId == kShader_inter || shaderId == kShader_inter_bdof || shaderId == kShader_inter_affine || shaderId == kShader_inter_wide
         || shaderId == kShader_inter_bdof_wide || shaderId == kShader_inter_affine_wide;
}

// The shaders that work with waves of any size from 16 up (no ballots or lane reads that take 32 threads): they may
// run in the size VVDEC_GPU_WAVE (VVDEC_GPU_WAVE_INTER: inter's variants) asks for
static bool anyWaveSize( int shaderId )
{
  switch( shaderId )
  {
  case kShader_alf_luma: case kShader_alf_chroma: case kShader_alf_luma_bounds: case kShader_alf_chroma_bounds:
  case kShader_sao: case kShader_deblock_luma_ver: case kShader_deblock_luma_hor: case kShader_deblock_chroma:
  case kShader_intra: case kShader_trafo: case kShader_lmcs: case kShader_filmgrain: case kShader_dbparams:
  case kShader_dbparams_sub: case kShader_edges:
    return true;
  default:
    return false;
  }
}

// The wave size a shader's pipeline asks for (0: the driver's choice): 32, but for inter's variants (interShader), and
// on Intel's GPUs 16 for the shaders that work with any (anyWaveSize), unless VVDEC_GPU_WAVE or VVDEC_GPU_WAVE_INTER asks
// for another. Intel's compiler gives a thread of a 32-wide wave half the registers of a 16-wide one's; at 32 the
// heavy shaders spill. Intel Graphics (285K, driver 101.8991), VideoQ 4K, ms a picture at 32 -> 16: ALF luma 5.66 ->
// 0.90, deblocking 1.31 -> 0.86, dbparams 1.61 -> 0.82, edges 1.79 -> 1.41; 29.3 -> 35.4 fps (GPU front end). Inter's
// variants keep the driver's choice there (16: the same; 32: 3.4 -> 4.7 ms at 1080p).
uint32_t Context::waveSizeOf( int shaderId ) const
{
  int asked = interShader( shaderId ) ? m_options.waveSizeInter : anyWaveSize( shaderId ) ? m_options.waveSize : -1;
  if( asked < 0 && m_intel && anyWaveSize( shaderId ) )
  {
    asked = 16;
  }
  if( asked >= 0 && ( asked == 0 || ( m_minWave <= uint32_t( asked ) && uint32_t( asked ) <= m_maxWave ) ) )
  {
    return uint32_t( asked );
  }
  return m_wave32 && !interShader( shaderId ) ? 32u : 0u;
}

VkPipeline Context::pipeline( int shaderId )
{
  std::call_once( m_pipelineOnce[shaderId], [this, shaderId]() { m_pipelines[shaderId] = createPipelineOf( shaderId ); } );
  return m_pipelines[shaderId];
}

// Which shaders the background compiles, and in what order: the two parse variants a stream may take first (they
// compile longest), then the shaders of a picture's work in the order the first picture uses them, then the variants
// other streams take. The profiling and shuffle variants of the parse and the parse check only when the configuration
// is set up for them, last.
static std::vector<int> precompileOrder( const Options& options, bool subgroupBallot )
{
  static const char* const kFirstUse[] = { "parse",           "parse_ctu64",     "mider",          "jobs",        "dbparams",        "inter_wide",
                                           "inter_bdof_wide", "intra",           "jobs_count",     "dbparams_sub", "trafo",          "intrasort",
                                           "deblock_luma_ver", "deblock_luma_hor", "deblock_chroma", "inter_affine_wide", "sao",     "alf_luma",
                                           "alf_chroma",      "jobs_waves",      "edges",          "parsecompact", "mider_cumap",     "lmcs",
                                           "parse_ctu64_lite", "mider_lite",     "inter",          "inter_bdof",   "inter_affine",   "alf_luma_bounds",
                                           "alf_chroma_bounds", "filmgrain" };
  std::vector<int>         order;
  const auto               add = [&]( const std::string& name ) {
    for( int i = 0; i < kShaderCount; i++ )
    {
      if( name == kShaders[i].name )
      {
        order.push_back( i );
      }
    }
  };
  for( const char* name: kFirstUse )
  {
    add( name );
  }
  for( int i = 0; i < kShaderCount; i++ )
  {
    const std::string name    = kShaders[i].name;
    const bool        profile = name.find( "_prof" ) != std::string::npos;
    const bool        shuffle = name == "parse_shuffle";
    const bool        check   = name == "parsecheck";
    const bool        wanted  = ( !profile || options.profile ) && ( !shuffle || !subgroupBallot ) && ( !check || options.parse );
    if( wanted && std::find( order.begin(), order.end(), i ) == order.end() )
    {
      order.push_back( i );
    }
  }
  return order;
}

void Context::startPrecompile()
{
  m_precompileOrder = precompileOrder( m_options, m_subgroupBallot );
  // Compiles running side by side slow each other (the parse shader: 4.5 s alone, 7.1 s among 8), so a few threads
  // serve better than many.
  const unsigned cores   = std::thread::hardware_concurrency();
  const int      threads = int( std::min( 4u, std::max( 2u, cores / 4 ) ) );
  for( int t = 0; t < threads; t++ )
  {
    m_precompileThreads.emplace_back( [this]() {
#ifdef _WIN32
      SetThreadPriority( GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL );   // (the decode's threads come first)
#endif
      while( !m_precompileStop.load( std::memory_order_relaxed ) )
      {
        const int k = m_precompileNext.fetch_add( 1 );
        if( k >= int( m_precompileOrder.size() ) )
        {
          break;
        }
        try
        {
          pipeline( m_precompileOrder[k] );
        }
        catch( ... )
        {
          // (the decode thread that needs this pipeline will meet the failure itself)
        }
      }
    } );
  }
}

void Context::stopPrecompile()
{
  m_precompileStop.store( true );
  for( std::thread& thread: m_precompileThreads )
  {
    thread.join();
  }
  m_precompileThreads.clear();
}

VkPipeline Context::createPipelineOf( int shaderId )
{

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
  if( m_getExecutableStatistics )
  {
    info.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  }
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo size{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
  size.requiredSubgroupSize = waveSizeOf( shaderId );
  if( size.requiredSubgroupSize != 0 )
  {
    info.stage.pNext = &size;
  }
  // The work-arounds of drivers' bugs, for the shaders that have them: kSplitStores (constant 0), kIntegerAddresses (1)
  // and kCoherentReads (2), 32-bit bools
  const uint32_t                 values[3]  = { m_splitStores ? 1u : 0u, m_integerAddresses ? 1u : 0u, m_coherentReads ? 1u : 0u };
  const VkSpecializationMapEntry entries[3] = { { 0, 0, sizeof( uint32_t ) }, { 1, sizeof( uint32_t ), sizeof( uint32_t ) },
                                                { 2, 2 * sizeof( uint32_t ), sizeof( uint32_t ) } };
  const VkSpecializationInfo     specialization{ 3, entries, sizeof( values ), values };
  if( m_splitStores || m_integerAddresses || m_coherentReads )
  {
    info.stage.pSpecializationInfo = &specialization;
  }
  VkPipeline     pipe   = VK_NULL_HANDLE;
  const VkResult result = createPipeline( shaderId, size.requiredSubgroupSize, info, pipe );
  m_api.vkDestroyShaderModule( m_device, module, nullptr );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: the pipeline of " << kShaders[shaderId].name << " failed (" << result << ")" );
  }
  reportShader( shaderId, pipe );
  return pipe;
}

// With VK_AMD_shader_info (profiling, or VVDEC_GPU_ISA): the shader's registers and shared memory, and with
// VVDEC_GPU_ISA=<directory> its machine code into <directory>/<name>.isa
void Context::reportShader( int shaderId, VkPipeline pipe ) const
{
  if( m_getExecutables && m_getExecutableStatistics )
  {
    // each executable's statistics (Intel's: its SIMD width, registers, spills and fills, scratch, instructions)
    VkPipelineInfoKHR pipeInfo{ VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR };
    pipeInfo.pipeline = pipe;
    uint32_t count    = 0;
    m_getExecutables( m_device, &pipeInfo, &count, nullptr );
    std::vector<VkPipelineExecutablePropertiesKHR> props( count, VkPipelineExecutablePropertiesKHR{ VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR } );
    m_getExecutables( m_device, &pipeInfo, &count, props.data() );
    for( uint32_t e = 0; e < count; e++ )
    {
      VkPipelineExecutableInfoKHR exe{ VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR };
      exe.pipeline        = pipe;
      exe.executableIndex = e;
      uint32_t n          = 0;
      m_getExecutableStatistics( m_device, &exe, &n, nullptr );
      std::vector<VkPipelineExecutableStatisticKHR> stats( n, VkPipelineExecutableStatisticKHR{ VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR } );
      m_getExecutableStatistics( m_device, &exe, &n, stats.data() );
      std::string line;
      for( const VkPipelineExecutableStatisticKHR& st: stats )
      {
        char value[64];
        switch( st.format )
        {
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: snprintf( value, sizeof( value ), "%s", st.value.b32 ? "yes" : "no" ); break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: snprintf( value, sizeof( value ), "%lld", (long long) st.value.i64 ); break;
        case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: snprintf( value, sizeof( value ), "%llu", (unsigned long long) st.value.u64 ); break;
        default: snprintf( value, sizeof( value ), "%g", st.value.f64 ); break;
        }
        line += std::string( "  " ) + st.name + " " + value;
      }
      msg( INFO, "GPU shader %-16s %s (subgroup %u):%s\n", kShaders[shaderId].name, props[e].name, props[e].subgroupSize, line.c_str() );
    }
  }
  if( !m_getShaderInfo )
  {
    return;
  }
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
        else
        {
          msg( WARNING, "Vulkan: cannot write %s (does the directory exist?)\n", path.c_str() );
        }
      }
    }
  }
}

uint64_t Context::submitFirst( VkCommandBuffer first, VkFence firstDone, VkSemaphore waitFront, uint64_t waitParse )
{
  VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  info.commandBufferCount = 1;
  info.pCommandBuffers    = &first;
  // the waits: the front end's results (its buffers, the indirect dispatches' arguments), the picture's parsed records
  const VkPipelineStageFlags    stages[2]   = { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT };
  VkSemaphore                   waits[2];
  uint64_t                      waitValues[2];
  uint32_t                      numWaits    = 0;
  uint64_t                      signalValue = 0;
  VkTimelineSemaphoreSubmitInfo timeline{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
  if( waitFront )
  {
    waits[numWaits]      = waitFront;
    waitValues[numWaits] = 0;   // (a binary semaphore's)
    numWaits++;
  }
  if( waitParse > 0 )
  {
    waits[numWaits]      = m_parseCopyTimeline;
    waitValues[numWaits] = waitParse;
    numWaits++;
  }
  if( numWaits > 0 )
  {
    info.waitSemaphoreCount          = numWaits;
    info.pWaitSemaphores             = waits;
    info.pWaitDstStageMask           = stages + ( waitFront ? 0 : 1 );
    timeline.waitSemaphoreValueCount = numWaits;
    timeline.pWaitSemaphoreValues    = waitValues;
    info.pNext                       = &timeline;
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
  // the waits: the colocated pictures' first parts, the picture's parsed records
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
    waits[numWaits]      = m_parseCopyTimeline;
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

uint64_t Context::submitParseCopy( VkCommandBuffer cmd, VkFence done, uint64_t waitParse )
{
  VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  info.commandBufferCount = 1;
  info.pCommandBuffers    = &cmd;
  // the wait: the parse's steps that wrote what the copy reads (done already: the parse's thread saw them done)
  const VkPipelineStageFlags    stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkTimelineSemaphoreSubmitInfo timeline{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
  if( waitParse > 0 )
  {
    info.waitSemaphoreCount          = 1;
    info.pWaitSemaphores             = &m_parseTimeline;
    info.pWaitDstStageMask           = &stage;
    timeline.waitSemaphoreValueCount = 1;
    timeline.pWaitSemaphoreValues    = &waitParse;
  }
  info.signalSemaphoreCount = 1;
  info.pSignalSemaphores    = &m_parseCopyTimeline;
  std::lock_guard<std::mutex> lock( *m_parseCopyMutex );
  const uint64_t              value = ++m_parseCopyValue;
  timeline.signalSemaphoreValueCount = 1;
  timeline.pSignalSemaphoreValues    = &value;
  info.pNext                         = &timeline;
  const VkResult result = m_api.vkQueueSubmit( m_parseCopyQueue, 1, &info, done );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkQueueSubmit failed on the parse's copies' queue (" << result << ")" );
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

void Context::submitGrain( VkCommandBuffer cmd, VkFence done )
{
  if( !m_grainQueue )
  {
    submit( cmd, done );
    return;
  }
  VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  info.commandBufferCount = 1;
  info.pCommandBuffers    = &cmd;
  std::lock_guard<std::mutex> lock( m_grainMutex );
  const VkResult              result = m_api.vkQueueSubmit( m_grainQueue, 1, &info, done );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkQueueSubmit (film grain) failed (" << result << ")" );
  }
}

void Context::submitCopy( VkCommandBuffer cmd, VkFence done )
{
  VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  info.commandBufferCount = 1;
  info.pCommandBuffers    = &cmd;
  std::lock_guard<std::mutex> lock( transferQueue() ? m_transferMutex : m_queueMutex );
  const VkResult              result = m_api.vkQueueSubmit( transferQueue() ? m_transferQueue : m_queue, 1, &info, done );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkQueueSubmit (copies) failed (" << result << ")" );
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
