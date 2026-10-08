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

/** \file     GpuRecon.h
    \brief    the stages of a picture's reconstruction that run on the GPU

    The pictures live in host memory the GPU reaches (hostPictureAllocator),
    where vvdec's CPU code reads and writes them, and each has a twin in the
    GPU's own memory, where the GPU finishes it and keeps it as a reference.
*/

#pragma once

#include <chrono>

#include "GpuContext.h"
#include "GpuJobs.h"

#include "CommonLib/Common.h"
#include "CommonLib/Buffer.h"

namespace vvdec
{

class CodingStructure;
class AdaptiveLoopFilter;
class SampleAdaptiveOffset;
class Mv;
class SPS;
struct Picture;

namespace gpu
{

// UserAllocator callbacks putting the planes of the pictures in host memory
// the GPU reaches; each plane's allocator handle is its HostPlane (GpuRecon.cpp)
UserAllocator hostPictureAllocator( Context* context );
// a reconstructed picture no picture will reference any more gives back its
// planes on the GPU (to Context's pool); not gpuCurrent then
void releaseTwins( Picture& pic );

class GpuRecon
{
public:
  explicit GpuRecon( Context* context );
  ~GpuRecon();
  GpuRecon( const GpuRecon& )            = delete;
  GpuRecon& operator=( const GpuRecon& ) = delete;

  // whether the GPU's ALF covers the picture: no CTU of it is clipped by a
  // slice, tile or subpicture boundary the filters may not cross, nor by a
  // virtual boundary
  static bool alfSupported( const CodingStructure& cs );
  // whether the GPU's SAO covers the picture: no virtual boundaries
  static bool saoSupported( const CodingStructure& cs );
  // whether the GPU covers all the in-loop filters of the picture: deblocking
  // (not with luma-adaptive QP offsets), SAO and ALF
  static bool filtersSupported( const CodingStructure& cs );

  // whether the GPU predicts the intra and IBC blocks of the picture: 4:2:0,
  // none of the tools it lacks (MIP, ISP, multiple reference lines, CIIP, ACT,
  // LMCS, reference wraparound)
  static bool intraSupported( const CodingStructure& cs );
  static bool intraSupported( const SPS& sps );
  // whether the GPU predicts the inter blocks of the pictures of the sequence:
  // what intraSupported needs, 10 bits at most, none of affine, BDOF, GPM, BCW,
  // SbTMVP, subpictures
  static bool interSupported( const SPS& sps );
  // and of the picture: no weighted prediction, no scaled references
  static bool interSupported( const CodingStructure& cs );
  // whether the GPU dequantizes and inverse transforms the residuals of the
  // picture: 10 bits at most, no LFNST, no scaling lists
  static bool residualSupported( const CodingStructure& cs );

  // the deblocking parameters of the current picture (vvdec's LF_INIT writes
  // them): count of them, in host memory the GPU reads
  LoopFilterParam* loopFilterParams( size_t count );
  // the edges of a CTU the deblocking filters, from its deblocking parameters
  // (after LF_INIT): the GPU's deblocking takes only those
  static void      listEdges( const CodingStructure& cs, int ctuRsAddr, EdgeLists& edges );
  // vvdec's DMVR MV cache of the current picture, count of them, in host memory the GPU writes
  Mv*              dmvrMvCache( size_t count );

  // the rest of the picture in pic, on top of what the CPU reconstructed: the
  // residuals of jobs.trafo, the inter blocks of jobs.inter, from the
  // reference pictures of jobs.refs, the intra and IBC blocks of jobs.intra,
  // then the in-loop filters: deblocking (when deblock), SAO (when sao is
  // given), ALF with CC-ALF (when alf is given); the result stays on the GPU as
  // a reference (the picture is gpuCurrent then) and comes back to pic.
  // startPicture queues the work, in two parts: the first, the residuals and
  // the inter blocks, the rest. waitFirst: it returns once the first part is
  // done, DMVR's refinements in host memory (dmvrMvCache). endPicture waits
  // for the rest, the download of pic included.
  void startPicture( CodingStructure& cs, const PictureJobs& jobs, bool deblock, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf,
                     PelStorage& pic, bool waitFirst );
  void endPicture();

private:
  void     beginCommands();
  void     computeBarrier();
  void     mark( const char* label );   // with VVDEC_GPU_PROFILE: the GPU time since the last mark counts as label
  void     splitCommands();   // the first part's commands end and go to the GPU, the rest's begin
  // the rest's commands queued, and the download of pic: on the copy engine
  // (Context::transferQueue), after the rest's m_semaphore, else at the end of
  // the rest
  void     submitPicture( const PelStorage& pic, bool waitFirst );
  void     readTimestamps( double waited );
  uint8_t* params( size_t size );   // the host-visible parameter buffer, at least size bytes

  // the planes of a picture as the shaders see them
  struct Planes
  {
    VkDeviceAddress addr[MAX_NUM_COMPONENT];
    int32_t         stride[MAX_NUM_COMPONENT];
  };
  enum { kHost, kTwin, kScratch };
  Planes        hostPlanes( const PelStorage& pic );     // the picture in host memory
  Planes        twinPlanes( const PelStorage& pic );     // the picture on the GPU (its memory made when missing)
  Planes        scratchPlanes( const PelStorage& pic );  // GPU memory laid out as pic's buffers
  void          copyPlanes( const PelStorage& pic, int from, int to );
  void          transferBarrier( bool toTransfer );

  static size_t deblockParamSize( const CodingStructure& cs );
  static size_t alfParamSize( const CodingStructure& cs );
  static size_t saoParamSize( const CodingStructure& cs );
  bool          prepareDeblock( const CodingStructure& cs, uint8_t* params );   // false: no slice deblocks
  void          prepareAlf( const CodingStructure& cs, const AdaptiveLoopFilter& alf, uint8_t* params );
  bool          prepareSao( CodingStructure& cs, SampleAdaptiveOffset& sao, uint8_t* params );   // false: no CTU uses SAO
  void          recordDeblock( const CodingStructure& cs, const Planes& planes, VkDeviceAddress params, const std::vector<EdgeLists>& edges );
  void          recordIntra( const CodingStructure& cs, const std::vector<std::vector<IntraJob>>& jobs, const Planes& planes );
  void          recordInter( const CodingStructure& cs, const std::vector<std::vector<InterJob>>& jobs, const Planes& planes, VkDeviceAddress refTable );
  // the levels from the planes of levels (the picture in host memory, where the parser left them), the residuals to planes
  void          recordTrafo( const CodingStructure& cs, const std::vector<std::vector<TrafoJob>>& jobs, const Planes& levels, const Planes& planes );
  void          recordAlf( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params );
  void          recordSao( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params );

  static constexpr uint32_t kMaxMarks = 64;

  Context*        m_ctx;
  VkCommandPool   m_pool    = VK_NULL_HANDLE;
  VkCommandBuffer m_cmd     = VK_NULL_HANDLE;   // the one being recorded: m_cmdFirst or m_cmdRest
  VkCommandBuffer m_cmdFirst = VK_NULL_HANDLE;  // a picture's first part
  VkCommandBuffer m_cmdRest  = VK_NULL_HANDLE;  // and the rest
  VkFence         m_fenceFirst = VK_NULL_HANDLE;   // the first part done
  VkFence         m_fence   = VK_NULL_HANDLE;   // all done
  std::chrono::steady_clock::time_point m_submitted;
  VkQueryPool     m_queries = VK_NULL_HANDLE;   // the profiling timestamps
  VkCommandPool   m_transferPool    = VK_NULL_HANDLE;   // the copy engine's commands
  VkCommandBuffer m_transferCmd     = VK_NULL_HANDLE;
  VkSemaphore     m_semaphore       = VK_NULL_HANDLE;   // the compute commands done, for the copies
  VkQueryPool     m_transferQueries = VK_NULL_HANDLE;
  const char*     m_labels[kMaxMarks];
  uint32_t        m_marks   = 0;
  Buffer          m_params;
  Buffer          m_scratch[MAX_NUM_COMPONENT];     // a picture's worth of GPU memory (SAO's output)
  Buffer          m_lfp;                            // the deblocking parameters
  Buffer          m_dmvr;                           // DMVR's refinements
  Buffer          m_intraJobs;                      // the intra jobs of a picture, wave by wave
  Buffer          m_interJobs;
  Buffer          m_trafoJobs;
  Buffer          m_edges;                          // the edges the deblocking filters
  Buffer          m_matrices;                       // TrQuant's transform matrices
};

}   // namespace gpu
}   // namespace vvdec
