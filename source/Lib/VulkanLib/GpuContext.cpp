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
Context*   Context::s_context = nullptr;
int        Context::s_refs    = 0;

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

bool Context::init()
{
#ifdef _WIN32
  HMODULE library = LoadLibraryW( L"vulkan-1.dll" );
  m_library       = library;
  if( !library )
  {
    msg( WARNING, "Vulkan: vulkan-1.dll not found\n" );
    return false;
  }
  m_api.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) (void*) GetProcAddress( library, "vkGetInstanceProcAddr" );
#else
  m_library = dlopen( "libvulkan.so.1", RTLD_NOW | RTLD_LOCAL );
  if( !m_library )
  {
    msg( WARNING, "Vulkan: libvulkan.so.1 not found\n" );
    return false;
  }
  m_api.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr) dlsym( m_library, "vkGetInstanceProcAddr" );
#endif
  if( !m_api.vkGetInstanceProcAddr )
  {
    return false;
  }

  auto createInstance = (PFN_vkCreateInstance) m_api.vkGetInstanceProcAddr( VK_NULL_HANDLE, "vkCreateInstance" );
  VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
  app.pApplicationName = "vvdec";
  app.apiVersion       = VK_API_VERSION_1_2;
  VkInstanceCreateInfo instanceInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
  instanceInfo.pApplicationInfo = &app;
  if( !createInstance || createInstance( &instanceInfo, nullptr, &m_instance ) != VK_SUCCESS )
  {
    msg( WARNING, "Vulkan: no instance\n" );
    return false;
  }

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
  if( const char* index = getenv( "VVDEC_GPU_DEVICE" ) )
  {
    chosen = atoi( index );
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
  features12.pNext = &features11;
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

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
  queueInfo.queueFamilyIndex = m_queueFamily;
  queueInfo.queueCount       = 1;
  queueInfo.pQueuePriorities = &priority;

  VkPhysicalDeviceVulkan12Features enable12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
  VkPhysicalDeviceVulkan11Features enable11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
  enable12.pNext                    = &enable11;
  enable12.bufferDeviceAddress      = VK_TRUE;
  enable12.storageBuffer8BitAccess  = VK_TRUE;
  enable12.shaderInt8               = VK_TRUE;
  enable11.storageBuffer16BitAccess = VK_TRUE;
  VkPhysicalDeviceFeatures2 enable{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
  enable.pNext                = &enable12;
  enable.features.shaderInt16 = VK_TRUE;
  enable.features.shaderInt64 = VK_TRUE;

  VkDeviceCreateInfo deviceInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  deviceInfo.pNext                = &enable;
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos    = &queueInfo;
  if( m_api.vkCreateDevice( m_physical, &deviceInfo, nullptr, &m_device ) != VK_SUCCESS )
  {
    msg( WARNING, "Vulkan: creating the device on %s failed\n", props.deviceName );
    return false;
  }

#define VVDEC_VK_LOAD_DEVICE( name ) m_api.name = (PFN_##name) m_api.vkGetDeviceProcAddr( m_device, #name );
  VVDEC_VK_DEVICE_FUNCS( VVDEC_VK_LOAD_DEVICE )
#undef VVDEC_VK_LOAD_DEVICE

  m_api.vkGetDeviceQueue( m_device, m_queueFamily, 0, &m_queue );
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

Context::~Context()
{
  if( m_device )
  {
    m_api.vkDeviceWaitIdle( m_device );
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
    m_api.vkDestroyDevice( m_device, nullptr );
  }
  delete[] m_pipelines;
  if( m_instance )
  {
    m_api.vkDestroyInstance( m_instance, nullptr );
  }
#ifdef _WIN32
  if( m_library )
  {
    FreeLibrary( (HMODULE) m_library );
  }
#else
  if( m_library )
  {
    dlclose( m_library );
  }
#endif
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
  VkPipeline pipe   = VK_NULL_HANDLE;
  const VkResult result = m_api.vkCreateComputePipelines( m_device, VK_NULL_HANDLE, 1, &info, nullptr, &pipe );
  m_api.vkDestroyShaderModule( m_device, module, nullptr );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: the pipeline of " << kShaders[shaderId].name << " failed (" << result << ")" );
  }
  m_pipelines[shaderId] = pipe;
  return pipe;
}

void Context::submit( VkCommandBuffer cmd, VkFence fence )
{
  VkSubmitInfo info{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
  info.commandBufferCount = 1;
  info.pCommandBuffers    = &cmd;
  std::lock_guard<std::mutex> lock( m_queueMutex );
  const VkResult result = m_api.vkQueueSubmit( m_queue, 1, &info, fence );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkQueueSubmit failed (" << result << ")" );
  }
}

}   // namespace gpu
}   // namespace vvdec
