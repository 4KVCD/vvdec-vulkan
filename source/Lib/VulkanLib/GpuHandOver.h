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
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

------------------------------------------------------------------------------------------- */

/** \file     GpuHandOver.h
    \brief    Decoded pictures copied into another Vulkan device's buffers on the same GPU
*/

#pragma once

#include "GpuContext.h"

#include "CommonLib/CommonDef.h"

#include <chrono>
#include <mutex>
#include <vector>

namespace vvdec
{
struct Picture;
struct FilmGrainFrame;

namespace gpu
{
class PlanePool;

// The hand-over: a decoded picture's planes copied from its twins straight into another Vulkan device's buffers on
// this GPU (Context::importBuffer; VideoMetricsLab's VMAF engine) or into imported host memory
// (Context::importHostBuffer; Vship's frame buffers), by one compute pass a plane (shaders/handover.slang) on the
// compute queue, each copy signalling the context's hand-over timeline (Context::submitHandOver) that the other
// device waits for. With the GPU's parse ahead the picture is not downloaded to host memory: its twins hold it until
// the application gives it back. With the CPU's parse the GPU downloads every picture to its host planes, which the
// copies read; a copy that read them keeps them from being written again until it is done (Picture::handOverValue).
// A frame with film grain gets its grain in the copy: the grain shader (GpuFilmGrain::record) reads the frame -- the
// twins, or the host planes (on a discrete GPU a copy of them in its memory first) -- and writes it with its grain
// into a scratch in GPU memory, which the copy then reads.
// One HandOver per decoder; its copies may be asked for from any one thread at a time.
class HandOver
{
public:
  explicit HandOver( Context* context );
  ~HandOver();   // waits for its copies
  HandOver( const HandOver& )            = delete;
  HandOver& operator=( const HandOver& ) = delete;

  // One plane of a copy: the rectangle (x, y, width, height) of the picture's plane comp -- in the plane's samples,
  // the conformance window already applied by the caller -- into dst at dstOffset, rows dstStride bytes apart.
  struct Plane
  {
    ComponentID  comp;
    VkDeviceSize dstOffset;
    uint32_t     dstStride;
    int          x, y, width, height;
  };
  // Copies `count` planes of the picture into dst, samples of dstBytes bytes (1 or 2), each shifted left by shift, in
  // one submission. A picture on the GPU is read from its twins; one held in host planes (gpuNeedsDownload clear: an
  // error picture the CPU filled, or one the CPU parsed, downloaded at submit) from those, which the GPU reaches too.
  // pool: the decoder's PlanePool, whose lock guards the planes' records (null without one). grain: the frame's film
  // grain, added in the copy (null: none); the planes are then the components 0 to count - 1. Returns the hand-over
  // timeline's value the copy signals; its writes are visible to the host once that value is reached.
  uint64_t pack( Picture& pic, PlanePool* pool, const Plane* planes, int count, const Buffer& dst, int dstBytes, int shift,
                 const FilmGrainFrame* grain = nullptr );
  // Waits for every copy submitted (before the pictures' twins may be freed: a seek, the decoder closing).
  void wait();
  // Waits until the copy that signals value (from pack) and the copies before it are done, at most timeoutNs:
  // VK_SUCCESS, VK_TIMEOUT or the failure (VK_ERROR_DEVICE_LOST). Callable while pack runs on another thread.
  VkResult waitFor( uint64_t value, uint64_t timeoutNs );

private:
  using Clock = std::chrono::steady_clock;

  // A copy's commands; the ring takes them in turn, each waited for before it is recorded again
  struct Copy
  {
    VkCommandPool   pool    = VK_NULL_HANDLE;
    VkCommandBuffer cmd     = VK_NULL_HANDLE;
    VkFence         fence   = VK_NULL_HANDLE;
    bool            pending = false;
    uint64_t        value   = 0;   // the timeline's value it signals
    // With the timing on: its GPU timestamps (its start, after its first barrier, its end), its submission, and the
    // time from its submission to the end of a waitFor that saw it done (negative without one)
    VkQueryPool       queries = VK_NULL_HANDLE;
    Clock::time_point submitted;
    double            waitedMs = -1;
    // With film grain: its tables and the frame's block seeds (Upload), written once the copy before is done
    Buffer grainTables;
  };
  static constexpr size_t kCopies = 8;

  // The timing (VVDEC_HANDOVER_TIMES=1), printed every kTimesEvery copies and when the HandOver closes: the means of
  // the time a copy waited on the GPU for earlier work on its queue, of the pack itself, and of the time before the
  // GPU started it (waitFor's time minus the GPU's: the queue, its priority, other processes' work)
  struct Times
  {
    long   copies   = 0;
    double queuedMs = 0;
    double packMs   = 0;
    long   waited   = 0;   // the copies a waitFor timed
    double delayMs  = 0;
  };
  static constexpr long kTimesEvery = 100;
  void addTimes( Copy& copy );   // a done copy's times, with m_mutex held
  void printTimes( const char* when ) const;
  void waitPendingLocked();      // wait(), with m_mutex held
  // A plane of the frame as its film grain reads it: the address of the frame's top-left sample and the stride
  // (samples), the samples a row has from there to the picture's right edge, and the buffer it is in with that
  // sample's offset there
  struct GrainSource
  {
    VkDeviceAddress addr      = 0;
    int             stride    = 0;
    int             readWidth = 0;
    VkBuffer        buffer    = VK_NULL_HANDLE;
    VkDeviceSize    offset    = 0;
  };
  // The plane comp of the frame as its grain reads it, from the plane's origin (planeOrigin) and stride. With the
  // planes' lock held.
  static GrainSource grainSource( const Picture& pic, ComponentID comp, VkDeviceAddress origin, int stride, const FilmGrainFrame& grain );
  // Records the frame's grain into the copy's commands, from src (count planes: components 0 to count - 1) into
  // m_grainAfter; afterAt: where each plane's top-left sample is there. With m_mutex held.
  void recordGrain( Copy& copy, const FilmGrainFrame& grain, const GrainSource* src, int count, bool fromHost, VkDeviceAddress* afterAt );

  Context*          m_ctx;
  std::mutex        m_mutex;
  std::vector<Copy> m_copies;
  size_t            m_next  = 0;
  bool              m_timed = false;
  Times             m_times;
  // The film grain's scratch in GPU memory, shared by the copies (each copy's start barrier orders it after the one
  // before): the frame's planes with their grain (after), and on a discrete GPU the frame's planes copied from host
  // memory (before). Grown only once every copy is done.
  Buffer            m_grainBefore, m_grainAfter;
};

}   // namespace gpu
}   // namespace vvdec
