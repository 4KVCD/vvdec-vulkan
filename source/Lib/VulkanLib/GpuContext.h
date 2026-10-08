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

/** \file     GpuContext.h
    \brief    the Vulkan device shared by all decoder instances of a process

    Every buffer the shaders use is reached through its device address (push
    constants carry the addresses), so there are no descriptor sets: one
    pipeline layout with 256 bytes of push constants serves every shader.
*/

#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vvdec
{
namespace gpu
{

#define VVDEC_VK_INSTANCE_FUNCS( X )                                                                                    \
  X( vkDestroyInstance ) X( vkEnumeratePhysicalDevices ) X( vkGetPhysicalDeviceProperties )                              \
  X( vkGetPhysicalDeviceProperties2 ) X( vkGetPhysicalDeviceFeatures2 ) X( vkGetPhysicalDeviceMemoryProperties )         \
  X( vkGetPhysicalDeviceQueueFamilyProperties ) X( vkCreateDevice ) X( vkGetDeviceProcAddr )                            \
  X( vkEnumerateDeviceExtensionProperties )

#define VVDEC_VK_DEVICE_FUNCS( X )                                                                                      \
  X( vkDestroyDevice ) X( vkGetDeviceQueue ) X( vkCreateBuffer ) X( vkDestroyBuffer ) X( vkGetBufferMemoryRequirements ) \
  X( vkAllocateMemory ) X( vkFreeMemory ) X( vkBindBufferMemory ) X( vkMapMemory ) X( vkUnmapMemory )                    \
  X( vkGetBufferDeviceAddress ) X( vkCreateShaderModule ) X( vkDestroyShaderModule ) X( vkCreatePipelineLayout )        \
  X( vkDestroyPipelineLayout ) X( vkCreateComputePipelines ) X( vkDestroyPipeline ) X( vkCreateCommandPool )            \
  X( vkDestroyCommandPool ) X( vkResetCommandPool ) X( vkAllocateCommandBuffers ) X( vkFreeCommandBuffers )             \
  X( vkBeginCommandBuffer ) X( vkEndCommandBuffer ) X( vkCmdBindPipeline ) X( vkCmdPushConstants ) X( vkCmdDispatch )   \
  X( vkCmdDispatchIndirect ) X( vkCmdPipelineBarrier ) X( vkCmdCopyBuffer ) X( vkCmdFillBuffer ) X( vkCreateFence )     \
  X( vkDestroyFence ) X( vkWaitForFences ) X( vkResetFences ) X( vkGetFenceStatus ) X( vkQueueSubmit )                  \
  X( vkQueueWaitIdle ) X( vkDeviceWaitIdle ) X( vkCreateQueryPool ) X( vkDestroyQueryPool ) X( vkCmdResetQueryPool )    \
  X( vkCmdWriteTimestamp ) X( vkGetQueryPoolResults ) X( vkCreateSemaphore ) X( vkDestroySemaphore )

struct Api
{
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define VVDEC_VK_DECLARE( name ) PFN_##name name = nullptr;
  VVDEC_VK_INSTANCE_FUNCS( VVDEC_VK_DECLARE )
  VVDEC_VK_DEVICE_FUNCS( VVDEC_VK_DECLARE )
#undef VVDEC_VK_DECLARE
};

// where a buffer lives
enum class MemKind
{
  Device,       // GPU memory, not visible to the CPU
  HostCached,   // CPU memory the CPU reads and writes at full speed and the GPU reaches through the bus
  Upload,       // memory the CPU writes in order and the GPU reads (device local if the BAR allows)
};

struct Buffer
{
  VkBuffer        buffer = VK_NULL_HANDLE;
  VkDeviceMemory  memory = VK_NULL_HANDLE;
  VkDeviceSize    size   = 0;
  uint8_t*        mapped = nullptr;   // null for MemKind::Device
  VkDeviceAddress addr   = 0;
  MemKind         kind   = MemKind::Device;

  bool valid() const { return buffer != VK_NULL_HANDLE; }
};

class Context
{
public:
  // the process-wide context: created by the first acquire(), destroyed with
  // the last release(); acquire() returns null when no usable Vulkan device exists
  static Context* acquire();
  static void     release();

  const Api& api() const { return m_api; }
  VkDevice   device() const { return m_device; }

  Buffer alloc( size_t size, MemKind kind );   // throws on failure
  void   free( Buffer& buf );

  VkPipeline       pipeline( int shaderId );   // created on first use
  VkPipelineLayout layout() const { return m_layout; }

  // the queue is shared by the decoder instances of the process; signal (when
  // given) is signalled once the commands are done
  void submit( VkCommandBuffer cmd, VkFence fence, VkSemaphore signal = VK_NULL_HANDLE );
  // a queue for copies only (the GPU's copy engine), when the device has one:
  // its copies run alongside the compute queue's work; the commands wait for
  // wait first. The buffers are shared by both queues' families.
  bool transferQueue() const { return m_transferFamily != UINT32_MAX; }
  void submitTransfer( VkCommandBuffer cmd, VkSemaphore wait, VkFence fence );

  uint32_t           queueFamily()     const { return m_queueFamily; }
  uint32_t           transferFamily()  const { return m_transferFamily; }
  bool               transferTimestamps() const { return m_transferTimestamps; }
  double             timestampPeriod() const { return m_timestampPeriod; }
  const std::string& deviceName()      const { return m_deviceName; }

  // VVDEC_GPU_PROFILE=1: the decoders add up the time of their passes, printed
  // when the context goes
  bool profiling() const { return m_profiling; }
  void addTime( const char* label, double milliseconds );

private:
  Context() = default;
  ~Context();
  bool init();
  int  memoryType( uint32_t typeBits, MemKind kind ) const;

  Api                              m_api;
  void*                            m_library  = nullptr;
  VkInstance                       m_instance = VK_NULL_HANDLE;
  VkPhysicalDevice                 m_physical = VK_NULL_HANDLE;
  VkDevice                         m_device   = VK_NULL_HANDLE;
  VkQueue                          m_queue    = VK_NULL_HANDLE;
  uint32_t                         m_queueFamily = 0;
  VkQueue                          m_transferQueue  = VK_NULL_HANDLE;
  uint32_t                         m_transferFamily = UINT32_MAX;
  bool                             m_transferTimestamps = false;
  std::mutex                       m_transferMutex;
  VkPhysicalDeviceMemoryProperties m_memProps{};
  VkPipelineLayout                 m_layout = VK_NULL_HANDLE;
  VkPipeline*                      m_pipelines = nullptr;
  double                           m_timestampPeriod = 1.0;
  std::string                      m_deviceName;
  std::mutex                       m_queueMutex;
  std::mutex                       m_pipelineMutex;

  struct Time
  {
    std::string label;
    double      milliseconds;
    long        count;
  };
  bool                             m_profiling = false;
  PFN_vkGetShaderInfoAMD           m_getShaderInfo = nullptr;   // with VVDEC_GPU_PROFILE, where VK_AMD_shader_info is
  bool                             m_wave32 = false;            // the shaders may ask for waves of 32 threads
  std::vector<Time>                m_times;
  std::mutex                       m_timeMutex;

  static std::mutex s_mutex;
  static Context*   s_context;
  static int        s_refs;
};

}   // namespace gpu
}   // namespace vvdec
