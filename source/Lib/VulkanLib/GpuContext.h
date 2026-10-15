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
#include <deque>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace vvdec
{
namespace gpu
{

// v rounded up to a multiple of alignment (a power of 2)
constexpr size_t alignUp( size_t v, size_t alignment )
{
  return ( v + alignment - 1 ) & ~( alignment - 1 );
}

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
  X( vkCmdWriteTimestamp ) X( vkGetQueryPoolResults ) X( vkCreateSemaphore ) X( vkDestroySemaphore )                 \
  X( vkGetSemaphoreCounterValue )

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
  Upload,       // GPU memory the CPU can map when the BAR allows it, else host memory: for what the CPU writes once, in
                // order, and the GPU reads, and for what the GPU alone uses but the CPU must reach (colocated motion)
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
  // Where the buffer's bytes start in buffer: 0, but for a range of a larger buffer (RangeArena), which owns no memory
  VkDeviceSize    offset = 0;

  bool valid() const { return buffer != VK_NULL_HANDLE; }
  bool isRange() const { return buffer != VK_NULL_HANDLE && memory == VK_NULL_HANDLE; }
};

// The GPU path's settings from the environment variables (README.md), read once when the context is made.
struct Options
{
  std::string device;                  // VVDEC_GPU_DEVICE: the index of the Vulkan device (empty: the first discrete GPU,
                                       // else the first integrated one)
  bool        profile       = false;   // VVDEC_GPU_PROFILE=1: the GPU time of each pass, each shader's registers
  std::string isaDir;                  // VVDEC_GPU_ISA: on AMD, the directory where each shader's machine code is written
  std::string pipelineCache;           // VVDEC_GPU_CACHE: the file of the compiled pipelines' binaries (Context's
                                       // PipelineBinaries; empty: in the user's cache directory, "0": none)
  bool        async         = true;    // VVDEC_GPU_ASYNC=0: the front end on the compute queue, not an async queue
  bool        frontEnd      = true;    // VVDEC_GPU_FRONT=0: the CPU front end drives the pictures (on Intel's GPUs the
                                       // default; VVDEC_GPU_FRONT=1: the GPU front end)
  bool        frontEndSet   = false;   // VVDEC_GPU_FRONT is set
  bool        frontEndCheck = false;   // VVDEC_GPU_MIDER=1: the CPU front end drives, the GPU front end is compared with it
  bool        parse         = false;   // VVDEC_GPU_PARSE=1: the GPU parses the slice data too, compared with the CPU's parse
  bool        parseDrives   = false;   // VVDEC_GPU_PARSE=2: the GPU parses the slice data instead of the CPU
  bool        precompile    = true;    // VVDEC_GPU_PRECOMPILE=0: the pipelines are not compiled in the background at start-up
  int         waveSize      = -1;      // VVDEC_GPU_WAVE=<n>: the wave size of the shaders that work with any (0: the
                                       // driver's choice; -1: the default)
  int         waveSizeInter = -1;      // VVDEC_GPU_WAVE_INTER=<n>: the same for inter's variants
  bool        lowPriority   = true;    // VVDEC_GPU_PRIORITY=normal: the queues not at low global priority
  std::string engine;                  // VVDEC_GPU_ENGINE=compute or 3d: the compute queue in a compute-only family (the
                                       // GPU's compute engine) or in the graphics family (empty: per GPU, m_computeEngine)

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

  // Allocates a buffer (throws on failure) and frees one (not a range of a RangeArena's: give() takes those back).
  Buffer alloc( size_t size, MemKind kind );
  void   free( Buffer& buf );
  // Makes buf hold at least size bytes of kind: when it is smaller, frees it and allocates one of size rounded up to
  // 64 KB (so that a buffer whose need grows a little at a time is not allocated again each time).
  void   ensure( Buffer& buf, size_t size, MemKind kind );
  // Frees a buffer once the work queued on the compute queue so far is done (it waits for that work): for a buffer that
  // other threads' queued work may still use, such as the instances' shared planes. (Freed while queued work still
  // uses it, its memory would go while the GPU reads it: a GPU fault.)
  void   freeAfterQueued( Buffer& buf );

  // A pool for the device-local buffers that come and go with the pictures: their planes on the GPU and the scratch
  // planes of their filters. allocPooled() returns a buffer of the same size given back earlier, or a new one.
  // Returned buffers stay in the pool for later pictures, because freeing and allocating them during decoding costs
  // too much time (VideoQ 1080p: -15%), until buffers of another size are requested (a new resolution). A buffer
  // may be returned while queued work still uses it, as long as whatever takes it next is queued after that work.
  Buffer allocPooled( size_t size );
  void   recycle( Buffer& buf );

  // A cache for the host-memory buffers of the pictures' planes (hostPictureAllocator). A picture whose size differs
  // from that of the picture whose buffers it reuses is created anew, and allocating host memory the GPU can reach
  // takes ~0.3 ms a plane (BOUNDARY_A_3, whose pictures change size one after the other, spent half its decode there;
  // the CPU's malloc takes microseconds). allocHost() returns a buffer given back earlier, of kind and of at least size
  // bytes but at most twice that, or a new one; keepHost() gives one back. The cache holds kHostCacheBytes at most, and
  // frees the buffers given back longest ago beyond that.
  Buffer allocHost( size_t size, MemKind kind );
  void   keepHost( Buffer& buf );

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
  // for the parse's copies' timeline to reach waitParse (the picture's parsed records, when its front end is in its
  // first part: submitParseCopy), and signals the timeline semaphore; the value it signals is returned (0 without the
  // async queues).
  uint64_t submitFirst( VkCommandBuffer first, VkFence firstDone, VkSemaphore waitFront = VK_NULL_HANDLE, uint64_t waitParse = 0 );
  // submitRest() submits the rest, which waits for waitFront (the front end's second submission) when given and
  // signals restDone. With a copy queue it also submits the picture's download (transfer), which waits for the
  // rest's semaphore and signals transferDone.
  void submitRest( VkCommandBuffer rest, VkFence restDone, VkSemaphore semaphore, VkCommandBuffer transfer, VkFence transferDone,
                   VkSemaphore waitFront = VK_NULL_HANDLE );
  // Other work on the compute queue, which signals done.
  void submit( VkCommandBuffer cmd, VkFence done ) { submitFirst( cmd, done ); }

  // --- The film grain's queue ---
  // The film grain on the output frames (GpuFilmGrain) needs only the downloaded frame. So it runs on a queue of its
  // own where the async compute family has one that nothing else takes (AMD's queue 4 of 8), and there it does not
  // wait for the decoding queued on the compute queue. Without such a queue it runs on the compute queue. Its command
  // pools are of grainFamily(); submitGrain() submits its commands, which signal done.
  uint32_t grainFamily() const { return m_grainQueue ? m_asyncFamily : m_queueFamily; }
  void     submitGrain( VkCommandBuffer cmd, VkFence done );

  // --- The copy queue ---
  // A queue for copies only (the GPU's copy engine), when the device has one. Its copies run alongside the compute
  // queue's work. Buffers are shared by both queues' families.
  bool transferQueue() const { return m_transferFamily != UINT32_MAX; }
  // Submits commands that only copy (a picture's download: PlanePool), which signal done: on the copy queue, or on the
  // compute queue without one. Their command pool is of copyFamily().
  void     submitCopy( VkCommandBuffer cmd, VkFence done );
  uint32_t copyFamily() const { return transferQueue() ? m_transferFamily : m_queueFamily; }

  // --- The async compute queues ---
  // Compute-only queues (async compute engines), when the device has them and VVDEC_GPU_ASYNC is not 0. The
  // pictures' front ends run there, alongside the compute queue's work. Each decoder instance takes its own queue
  // (takeAsyncQueue), so the front ends of different instances also run at the same time. submitFront() submits a
  // front end's commands: they wait until the compute queue's timeline semaphore reaches waitFirst (the first parts
  // of the picture's colocated pictures) and the parse's copies' reaches waitParse (the picture's parsed records in
  // place: submitParseCopy), then signal done.
  bool     asyncQueue() const { return m_asyncFamily != UINT32_MAX; }
  uint32_t asyncFamily() const { return m_asyncFamily; }
  int      takeAsyncQueue() { return int( m_asyncNext++ % uint32_t( m_asyncCount ) ); }
  void     submitFront( int queue, VkCommandBuffer cmd, uint64_t waitFirst, VkSemaphore done, uint64_t waitParse = 0 );

  // --- The parse queue ---
  // With the async compute queues, the GPU's parse of the pictures ahead of their reconstruction (VVDEC_GPU_PARSE=2;
  // ParseService) runs on a queue of its own: an async compute queue that no decoder instance takes when the family
  // has one more, else the last one. submitParse() submits the parse's commands, which signal the parse's timeline
  // semaphore and done; the value they signal is returned, for the copies that read what they wrote.
  bool     parseQueue() const { return m_parseQueue != VK_NULL_HANDLE; }
  uint64_t submitParse( VkCommandBuffer cmd, VkFence done );
  // The copies of the parsed pictures' records (ParseService) run on a queue of their own, one more of the family's
  // when it has one (else the parse queue): the parse's next steps do not wait for them. submitParseCopy() submits a
  // copy, which waits until the parse's timeline reaches waitParse (the steps that parsed its pictures), then
  // signals the copies' timeline semaphore and done; the value it signals is returned, for the front ends.
  uint64_t submitParseCopy( VkCommandBuffer cmd, VkFence done, uint64_t waitParse );
  // VK_EXT_shader_subgroup_ballot: the parse shader reads a thread's value with OpSubgroupReadInvocationKHR, a
  // v_readlane on AMD (shaders/parse.slang's laneValue); without it, the parse_shuffle variant
  bool     subgroupBallot() const { return m_subgroupBallot; }

  uint32_t           queueFamily()     const { return m_queueFamily; }
  uint32_t           transferFamily()  const { return m_transferFamily; }
  bool               transferTimestamps() const { return m_transferTimestamps; }
  double             timestampPeriod() const { return m_timestampPeriod; }
  const std::string& deviceName()      const { return m_deviceName; }
  // A discrete GPU: its shaders reach host memory across the bus, slowly (GpuFilmGrain copies its frames instead)
  bool               discrete()        const { return m_discrete; }

  // --- Profiling (VVDEC_GPU_PROFILE=1) ---
  // The decoder instances add up the GPU time of their passes (addTime); the totals are printed when the context
  // is destroyed, with the memory in use now and at its peak: heap 0 is the GPU's own memory, heap 1 the
  // host-visible memory, in bytes.
  bool    profiling() const { return m_profiling; }
  bool    shaderClock() const { return m_shaderClock; }   // VK_KHR_shader_clock's subgroup clock (with VVDEC_GPU_PROFILE)
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
  bool                             m_discrete = false;   // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
  bool                             m_wave32 = false;   // the shaders may ask for waves of 32 threads
  bool                             m_computeEngine = false;   // the compute queue in a compute-only family (chooseDevice)
  bool                             m_intel         = false;   // an Intel GPU
  uint32_t                         m_minWave = 0, m_maxWave = 0;   // the wave sizes a shader may ask for (subgroup size control)
  bool                             m_splitStores      = false;   // the shaders' kSplitStores is set (NVIDIA's GPUs)
  bool                             m_integerAddresses = false;   // their kIntegerAddresses is set (Intel's GPUs)
  bool                             m_coherentReads    = false;   // their kCoherentReads is set (NVIDIA's GPUs)
  bool                             m_subgroupBallot = false;   // VK_EXT_shader_subgroup_ballot is on

  // the compute queue
  VkQueue                          m_queue       = VK_NULL_HANDLE;
  uint32_t                         m_queueFamily = 0;
  std::mutex                       m_queueMutex;
  VkSemaphore                      m_timeline      = VK_NULL_HANDLE;   // signalled by the pictures' first parts
  uint64_t                         m_timelineValue = 0;                // the last value signalled (guarded by m_queueMutex)

  // the film grain's queue (one of the async family's), when it has its own, and its lock
  VkQueue                          m_grainQueue = VK_NULL_HANDLE;
  std::mutex                       m_grainMutex;

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

  // init's steps: the Vulkan loader (s_library); the instance (s_instance); the device (VVDEC_GPU_DEVICE's, else the
  // first discrete GPU, then the first integrated one) and the features the decoder needs; the queue families; the
  // async family's queues; the device with its queues and the extensions it has that the decoder uses; the layout of
  // the pipelines
  bool loadLibrary();
  bool createInstance();
  bool chooseDevice( VkPhysicalDeviceProperties& props );
  bool checkFeatures( const VkPhysicalDeviceProperties& props, bool& timelineSemaphore, bool& subgroupSizeControl );
  bool chooseQueueFamilies( const std::vector<VkQueueFamilyProperties>& families, bool timelineSemaphore, const char* deviceName );
  // Which of the async family's queues the front ends, the parse, its copies and film grain take
  struct AsyncQueueLayout
  {
    uint32_t familySize    = 0;       // the family's queues
    bool     ownParseQueue = false;   // the parse has a queue of its own (else the last front end's)
    bool     ownCopyQueue  = false;   // and its copies one more (else the parse's)
    bool     split         = false;   // the front ends on the odd queues; the parse, its copies and film grain on even ones
    uint32_t frontIndex[kMaxAsync] = {};
    uint32_t parseIndex = 0, copyIndex = 0, grainIndex = 0;
    uint32_t queues     = 0;          // the family's queues to create
  };
  AsyncQueueLayout chooseAsyncQueues( const std::vector<VkQueueFamilyProperties>& families );
  // The extensions the device has, of those the decoder uses
  struct DeviceExtensions
  {
    bool        shaderInfo = false, shaderClock = false, subgroupBallot = false, maintenance5 = false, pipelineBinary = false;
    const char* globalPriority = nullptr;   // VK_KHR_global_priority, else VK_EXT_global_priority, if the device has one
    bool        executables    = false;     // VK_KHR_pipeline_executable_properties (when profiling)
  };
  DeviceExtensions findExtensions() const;
  // createDevice's steps: the queues to create (returns their number), whether the shaders may run in waves of 32
  // threads, and whether the device has the optional features the decoder uses
  uint32_t queueCreateInfos( const AsyncQueueLayout& async, VkDeviceQueueCreateInfo infos[3] ) const;
  bool     wave32Possible( const VkPhysicalDeviceProperties& props, bool subgroupSizeControl );
  bool     hasShaderClocks() const;
  bool     hasPipelineBinaries() const;
  bool createDevice( const VkPhysicalDeviceProperties& props, bool subgroupSizeControl, const AsyncQueueLayout& async );
  void getQueues( const AsyncQueueLayout& async );
  bool createPipelineLayout();

  // the parse queue (one of the async family's), its lock (its own, or the async queue's it shares) and timeline
  VkQueue                          m_parseQueue    = VK_NULL_HANDLE;
  std::mutex                       m_parseOwnMutex;
  std::mutex*                      m_parseMutex    = &m_parseOwnMutex;
  VkSemaphore                      m_parseTimeline = VK_NULL_HANDLE;
  uint64_t                         m_parseValue    = 0;   // the last value signalled (guarded by *m_parseMutex)
  // the queue of the parse's copies (one more of the async family's, or the parse queue), its lock and timeline
  VkQueue                          m_parseCopyQueue    = VK_NULL_HANDLE;
  std::mutex                       m_parseCopyOwnMutex;
  std::mutex*                      m_parseCopyMutex    = &m_parseCopyOwnMutex;
  VkSemaphore                      m_parseCopyTimeline = VK_NULL_HANDLE;
  uint64_t                         m_parseCopyValue    = 0;   // the last value signalled (guarded by *m_parseCopyMutex)

  // the pipelines
  VkPipelineLayout                 m_layout    = VK_NULL_HANDLE;
  VkPipeline*                      m_pipelines = nullptr;
  std::once_flag*                  m_pipelineOnce = nullptr;   // per shader: its pipeline created once (pipeline)
  std::mutex                       m_pipelineMutex;            // guards the binaries map (PipelineBinaries::pipelines)
  VkPipeline                       createPipelineOf( int shaderId );   // the shader's module and pipeline (compiled, or from its binaries)
  // The pipelines this configuration is likely to use, compiled in the background from the end of init, biggest first
  // (startPrecompile), by threads that stop taking shaders when the context ends (stopPrecompile)
  void                             startPrecompile();
  void                             stopPrecompile();
  std::vector<std::thread>         m_precompileThreads;
  std::vector<int>                 m_precompileOrder;
  std::atomic<int>                 m_precompileNext{ 0 };
  std::atomic<bool>                m_precompileStop{ false };
  // the compiled pipelines' binaries, kept in a file from one run to the next (VK_KHR_pipeline_binary: GpuContext.cpp's
  // PipelineBinaries; null without it)
  struct PipelineBinaries;
  PipelineBinaries*                m_binaries = nullptr;
  // Creates the compute pipeline of info (waveSize: the wave size it asks for, 0: none): from its binaries when the file has
  // them, else compiled, its binaries kept for the file
  VkResult                         createPipeline( int shaderId, uint32_t waveSize, VkComputePipelineCreateInfo& info, VkPipeline& pipe );
  uint32_t                         waveSizeOf( int shaderId ) const;
  bool                             createFromBinaries( const std::vector<uint8_t>& key, VkComputePipelineCreateInfo& info, VkPipeline& pipe );
  void                             captureBinaries( const std::vector<uint8_t>& key, VkPipeline pipe );   // (pipe compiled to capture them)
  void                             reportShader( int shaderId, VkPipeline pipe ) const;   // its registers, its machine code (VVDEC_GPU_ISA)
  void                             openPipelineBinaries( const VkPhysicalDeviceProperties& props );
  void                             closePipelineBinaries();   // (the file rewritten when binaries were added)

  // the buffer pool (allocPooled, recycle)
  std::mutex                       m_poolMutex;
  std::vector<Buffer>              m_pool;
  std::set<VkDeviceSize>           m_poolSizes;   // the sizes requested since the size last changed

  // the host buffers' cache (allocHost, keepHost): the buffers given back, the oldest first, and their bytes
  static constexpr size_t          kHostCacheBytes = size_t( 128 ) << 20;
  std::mutex                       m_hostCacheMutex;
  std::deque<Buffer>               m_hostCache;
  size_t                           m_hostCacheBytes = 0;

  // profiling
  struct Time
  {
    std::string label;
    double      milliseconds;
    long        count;
  };
  bool                             m_profiling = false;
  bool                             m_shaderClock = false;
  std::vector<Time>                m_times;
  std::mutex                       m_timeMutex;
  std::atomic<int64_t>             m_memoryNow[2]  = { { 0 }, { 0 } };
  std::atomic<int64_t>             m_memoryPeak[2] = { { 0 }, { 0 } };
  PFN_vkGetShaderInfoAMD           m_getShaderInfo = nullptr;   // VK_AMD_shader_info, on AMD when profiling or writing machine code
  // VK_KHR_pipeline_executable_properties, when profiling without VK_AMD_shader_info: the shaders' statistics
  PFN_vkGetPipelineExecutablePropertiesKHR m_getExecutables          = nullptr;
  PFN_vkGetPipelineExecutableStatisticsKHR m_getExecutableStatistics = nullptr;

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
