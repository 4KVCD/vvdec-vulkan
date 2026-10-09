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
    \brief    The Vulkan device that all decoder instances of a process share

    The context owns the device, its queues, the memory allocation and the shaders' pipelines. Shaders reach every
    buffer through its device address, which the push constants carry, so there are no descriptor sets: one
    pipeline layout with 256 bytes of push constants serves every shader. See README.md for an overview of the GPU
    path.
*/

#pragma once

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace vvdec
{
namespace gpu
{

// The Vulkan functions the decoder uses, loaded at run time from vulkan-1.dll (or libvulkan.so).
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

// Where a buffer's memory lives.
enum class MemKind
{
  Device,       // GPU memory that the CPU cannot see
  HostCached,   // CPU memory that the CPU reads and writes at full speed; the GPU reaches it over the bus
  Upload,       // memory the CPU writes once, in order, and the GPU reads (GPU memory when the BAR allows it)
};

struct Buffer
{
  VkBuffer        buffer = VK_NULL_HANDLE;
  VkDeviceMemory  memory = VK_NULL_HANDLE;
  VkDeviceSize    size   = 0;
  uint8_t*        mapped = nullptr;   // the CPU's view of the buffer; null for MemKind::Device
  VkDeviceAddress addr   = 0;         // the address the shaders use
  MemKind         kind   = MemKind::Device;
  VkDeviceSize    memorySize = 0;     // the size of the memory allocated for the buffer (at least size)

  bool valid() const { return buffer != VK_NULL_HANDLE; }
};

// The GPU path's settings from the environment variables (README.md), read once when the context is made.
struct Options
{
  std::string device;                  // VVDEC_GPU_DEVICE: the index of the Vulkan device (empty: the first discrete GPU,
                                       // else the first integrated one)
  bool        profile       = false;   // VVDEC_GPU_PROFILE=1: the GPU time of each pass, each shader's registers
  std::string isaDir;                  // VVDEC_GPU_ISA: with profile on AMD, where each shader's machine code is written
  bool        async         = true;    // VVDEC_GPU_ASYNC=0: the front end on the compute queue, not an async queue
  bool        frontEnd      = true;    // VVDEC_GPU_FRONT=0: the CPU front end drives the pictures
  bool        frontEndCheck = false;   // VVDEC_GPU_MIDER=1: the CPU front end drives, the GPU front end is compared with it

  static Options fromEnvironment();
};

class Context
{
public:
  // The process-wide context. The first acquire() creates it and the last release() destroys it, except for the
  // Vulkan instance, which stays for the rest of the process (see s_instance). acquire() returns null when no
  // usable Vulkan device exists.
  static Context* acquire();
  static void     release();

  const Api&     api() const { return m_api; }
  VkDevice       device() const { return m_device; }
  const Options& options() const { return m_options; }

  // Allocates a buffer (throws on failure) and frees one.
  Buffer alloc( size_t size, MemKind kind );
  void   free( Buffer& buf );

  // A pool for the device-local buffers that come and go with the pictures: their planes on the GPU and the scratch
  // planes of their filters. allocPooled() returns a buffer of the same size given back earlier, or a new one.
  // Returned buffers stay in the pool for later pictures, because freeing and allocating them during decoding costs
  // too much time (VideoQ 1080p: -15%), until buffers of another size are requested (a new resolution). A buffer
  // may be returned while queued work still uses it, as long as whatever takes it next is queued after that work.
  Buffer allocPooled( size_t size );
  void   recycle( Buffer& buf );

  // The compute pipeline of a shader (kShader_...), created on first use, and the layout they all share.
  VkPipeline       pipeline( int shaderId );
  VkPipelineLayout layout() const { return m_layout; }

  // Vulkan objects with the settings the decoder uses. The create functions throw on failure; the destroy functions
  // ignore null handles. (The destroy functions have distinct names because on 32-bit builds all these handle types
  // are uint64_t.) Command pools are transient: their command buffers are recorded for one submission at a time.
  VkCommandPool createCommandPool( uint32_t family );
  void          allocateCommandBuffers( VkCommandPool pool, uint32_t count, VkCommandBuffer* out );
  VkFence       createFence();
  VkSemaphore   createSemaphore();
  VkQueryPool   createTimestampQueries( uint32_t count );
  void          destroyCommandPool( VkCommandPool pool );
  void          destroyFence( VkFence fence );
  void          destroySemaphore( VkSemaphore semaphore );
  void          destroyQueries( VkQueryPool queries );

  // --- The compute queue, which all decoder instances of the process share ---
  // A picture's work is submitted in two parts. submitFirst() submits the first part, which signals firstDone.
  // With the async queues, the first part also waits for waitFront (the semaphore of the picture's front end) and
  // signals the timeline semaphore; the value it signals is returned (0 without the async queues).
  uint64_t submitFirst( VkCommandBuffer first, VkFence firstDone, VkSemaphore waitFront = VK_NULL_HANDLE );
  // submitRest() submits the rest, which waits for waitFront (the front end's second submission) when given and
  // signals restDone. With a copy queue it also submits the picture's download (transfer), which waits for the
  // rest's semaphore and signals transferDone.
  void submitRest( VkCommandBuffer rest, VkFence restDone, VkSemaphore semaphore, VkCommandBuffer transfer, VkFence transferDone,
                   VkSemaphore waitFront = VK_NULL_HANDLE );
  // Other work on the compute queue (film grain on the output frames), which signals done.
  void submit( VkCommandBuffer cmd, VkFence done ) { submitFirst( cmd, done ); }

  // --- The copy queue ---
  // A queue for copies only (the GPU's copy engine), when the device has one. Its copies run alongside the compute
  // queue's work. Buffers are shared by both queues' families.
  bool transferQueue() const { return m_transferFamily != UINT32_MAX; }

  // --- The async compute queues ---
  // Compute-only queues (async compute engines), when the device has them and VVDEC_GPU_ASYNC is not 0. The
  // pictures' front ends run there, alongside the compute queue's work. Each decoder instance takes its own queue
  // (takeAsyncQueue), so the front ends of different instances also run at the same time. submitFront() submits a
  // front end's commands: they wait until the compute queue's timeline semaphore reaches waitFirst (the first parts
  // of the picture's colocated pictures) and then signal done.
  bool     asyncQueue() const { return m_asyncFamily != UINT32_MAX; }
  uint32_t asyncFamily() const { return m_asyncFamily; }
  int      takeAsyncQueue() { return int( m_asyncNext++ % uint32_t( m_asyncCount ) ); }
  void     submitFront( int queue, VkCommandBuffer cmd, uint64_t waitFirst, VkSemaphore done );

  uint32_t           queueFamily()     const { return m_queueFamily; }
  uint32_t           transferFamily()  const { return m_transferFamily; }
  bool               transferTimestamps() const { return m_transferTimestamps; }
  double             timestampPeriod() const { return m_timestampPeriod; }
  const std::string& deviceName()      const { return m_deviceName; }

  // --- Profiling (VVDEC_GPU_PROFILE=1) ---
  // The decoder instances add up the GPU time of their passes (addTime); the totals are printed when the context
  // is destroyed, with the memory in use now and at its peak: heap 0 is the GPU's own memory, heap 1 the
  // host-visible memory, in bytes.
  bool    profiling() const { return m_profiling; }
  int64_t memoryNow( int heap ) const { return m_memoryNow[heap]; }
  int64_t memoryPeak( int heap ) const { return m_memoryPeak[heap]; }
  void    addTime( const char* label, double milliseconds );

private:
  Context() = default;
  ~Context();
  bool init();
  int  memoryType( uint32_t typeBits, MemKind kind ) const;
  void countMemory( MemKind kind, int64_t bytes );

  Api                              m_api;
  Options                          m_options;
  VkInstance                       m_instance = VK_NULL_HANDLE;   // s_instance
  VkPhysicalDevice                 m_physical = VK_NULL_HANDLE;
  VkDevice                         m_device   = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties m_memProps{};
  double                           m_timestampPeriod = 1.0;
  std::string                      m_deviceName;
  bool                             m_wave32 = false;   // the shaders may ask for waves of 32 threads

  // the compute queue
  VkQueue                          m_queue       = VK_NULL_HANDLE;
  uint32_t                         m_queueFamily = 0;
  std::mutex                       m_queueMutex;
  VkSemaphore                      m_timeline      = VK_NULL_HANDLE;   // signalled by the pictures' first parts
  uint64_t                         m_timelineValue = 0;                // the last value signalled (guarded by m_queueMutex)

  // the copy queue
  VkQueue                          m_transferQueue      = VK_NULL_HANDLE;
  uint32_t                         m_transferFamily     = UINT32_MAX;
  bool                             m_transferTimestamps = false;
  std::mutex                       m_transferMutex;

  // the async compute queues
  static constexpr int             kMaxAsync = 4;
  VkQueue                          m_asyncQueues[kMaxAsync] = {};
  std::mutex                       m_asyncMutexes[kMaxAsync];
  int                              m_asyncCount  = 0;
  std::atomic<uint32_t>            m_asyncNext{ 0 };
  uint32_t                         m_asyncFamily = UINT32_MAX;

  // the pipelines
  VkPipelineLayout                 m_layout    = VK_NULL_HANDLE;
  VkPipeline*                      m_pipelines = nullptr;
  std::mutex                       m_pipelineMutex;

  // the buffer pool (allocPooled, recycle)
  std::mutex                       m_poolMutex;
  std::vector<Buffer>              m_pool;
  std::set<VkDeviceSize>           m_poolSizes;   // the sizes requested since the size last changed

  // profiling
  struct Time
  {
    std::string label;
    double      milliseconds;
    long        count;
  };
  bool                             m_profiling = false;
  std::vector<Time>                m_times;
  std::mutex                       m_timeMutex;
  std::atomic<int64_t>             m_memoryNow[2]  = { { 0 }, { 0 } };
  std::atomic<int64_t>             m_memoryPeak[2] = { { 0 }, { 0 } };
  PFN_vkGetShaderInfoAMD           m_getShaderInfo = nullptr;   // VK_AMD_shader_info, when profiling on AMD

  static std::mutex s_mutex;
  static Context*   s_context;
  static int        s_refs;
  // The Vulkan library and instance are created by the first init() and kept for the rest of the process (not
  // destroyed at exit either, when the driver may already be unloading). The loader unloads the driver together
  // with the last instance, and AMD's driver (32.0.31041) leaks 1.5 MB every time it is loaded again: a process that
  // opened and closed 200 decoders one after another grew by 300 MB.
  static void*      s_library;
  static VkInstance s_instance;
};

}   // namespace gpu
}   // namespace vvdec
