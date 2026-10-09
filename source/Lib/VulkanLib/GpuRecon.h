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

#include <atomic>
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
  // none of the tools it lacks (ACT, reference wraparound)
  static bool intraSupported( const CodingStructure& cs );
  static bool intraSupported( const SPS& sps );
  // whether the GPU predicts the inter blocks of the pictures of the sequence:
  // what intraSupported needs, 10 bits at most, none of affine, BDOF, GPM, BCW,
  // SbTMVP, subpictures
  static bool interSupported( const SPS& sps );
  // and of the picture: no weighted prediction, no scaled references
  static bool interSupported( const CodingStructure& cs );
  // whether the GPU dequantizes and inverse transforms the residuals of the
  // picture: 10 bits at most, no scaling lists
  static bool residualSupported( const CodingStructure& cs );

  // the edges of a CTU the deblocking filters, with the bytes of their
  // deblocking parameters the shaders need (after LF_INIT): the GPU's
  // deblocking takes only those
  static void      listEdges( const CodingStructure& cs, int ctuRsAddr, EdgeLists& edges );
  // vvdec's DMVR MV cache of the current picture, count of them, in host memory the GPU writes
  Mv*              dmvrMvCache( size_t count );

  // the parameters of the picture's filters (deblocking, SAO, ALF), worked out
  // before its work is recorded -- a task of its own once the picture is
  // parsed, while its CTU tasks run (else startPicture does it)
  void prepareFilters( CodingStructure& cs, bool deblock, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf );
  // the residuals (jobs.trafo) and the inter blocks (jobs.inter, from the
  // reference pictures of jobs.refs) of the picture's first ctus CTUs queued
  // ahead of the rest of its work: once their CTU tasks are done, while the
  // CPU still makes the jobs of the CTUs below, the GPU starts on them.
  // startPicture then queues the rest (jobs.bandCtus = ctus).
  void startBand( CodingStructure& cs, const PictureJobs& jobs, size_t ctus, PelStorage& pic );
  // the rest of the picture in pic, on top of what the CPU reconstructed: the
  // residuals of jobs.trafo, the inter blocks of jobs.inter, from the
  // reference pictures of jobs.refs, the intra and IBC blocks of jobs.intra,
  // then the in-loop filters: deblocking (when deblock), SAO (when sao is
  // given), ALF with CC-ALF (when alf is given); the result stays on the GPU as
  // a reference (the picture is gpuCurrent then) and comes back to pic.
  // startPicture queues the work, in two parts: the first, the residuals and
  // the inter blocks (those of the CTUs after a band startBand queued), the
  // rest. waitFirst: it returns once the first part is
  // done, DMVR's refinements in host memory (dmvrMvCache). It returns the
  // picture's download, for endDownload (-1: none of its own).
  // endCompute waits for the rest (without the copy engine, the download
  // included): the next picture's work may be recorded then, while the
  // download runs. endDownload waits for that, before the work of the picture
  // after the next is recorded (the downloads take turns between two sets of
  // commands).
  int  startPicture( CodingStructure& cs, const PictureJobs& jobs, bool deblock, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf,
                     PelStorage& pic, bool waitFirst );
  // the picture whose work comes next, when the GPU predicts its intra
  // blocks; set before its work is recorded (its band's too), once the
  // instance's picture before is done. LMCS (tables: null without it): its
  // luma predicted and reconstructed in the mapped domain, inter's luma
  // forward-mapped, the chroma residuals scaled (tables->chroma), then the
  // luma mapped back before the in-loop filters. defer: inter keeps some of
  // its predictions aside, for intra jobs to finish (LMCS's chroma scaling,
  // CIIP). scalingMatrices: the picture's dequantisation matrices of its
  // explicit scaling list (TrQuant::getDequantCoeffBuf, scalingCount of them;
  // null without)
  void setPicture( const PelStorage& pic, const LmcsTables* tables, bool defer, const int* scalingMatrices = nullptr, size_t scalingCount = 0 );
  void endCompute();
  void endDownload( int download );

private:
  void     beginCommands( bool resetPool = true );
  void     computeBarrier();
  void     mark( const char* label );   // with VVDEC_GPU_PROFILE: the GPU time since the last mark counts as label
  void     splitCommands();   // the first part's commands end and go to the GPU, the rest's begin
  // the rest's commands queued, and the download of pic: on the copy engine
  // (Context::transferQueue), after the rest's semaphore (the download's
  // returned), else at the end of the rest (-1 returned)
  int      submitPicture( const PelStorage& pic, bool waitFirst );
  void     readTimestamps( double waited );
  // the reference pictures' planes on the GPU into m_refs (laid out as pic's);
  // uploads: those the CPU reconstructed (or filled in), to copy there first
  void     writeRefTable( const RefTable& refs, const PelStorage& pic, std::vector<const Picture*>& uploads );
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
  Planes        scratchPlanes( const PelStorage& pic );  // GPU memory laid out as pic's buffers (from the pool)
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
  // the jobs of the CTUs ctuBegin .. ctuEnd - 1, copied into buf
  void          recordInter( const CodingStructure& cs, const std::vector<std::vector<InterJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf,
                             const Planes& planes );
  // the levels from the planes of levels (the picture in host memory, where the parser left them), the residuals to planes
  void          recordTrafo( const CodingStructure& cs, const std::vector<std::vector<TrafoJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf,
                             const Planes& levels, const Planes& planes );
  void          recordAlf( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params );
  void          recordLmcs( const CodingStructure& cs, const Planes& planes, VkDeviceAddress ctus );   // the inverse mapping
  void          recordSao( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params );

  static constexpr uint32_t kMaxMarks = 64;
  static constexpr uint32_t kBandMarks = 8;

  Context*        m_ctx;
  VkCommandPool   m_pool    = VK_NULL_HANDLE;
  VkCommandBuffer m_cmd     = VK_NULL_HANDLE;   // the one being recorded: m_cmdBand, m_cmdFirst or m_cmdRest
  VkCommandBuffer m_cmdFirst = VK_NULL_HANDLE;  // a picture's first part
  VkCommandBuffer m_cmdRest  = VK_NULL_HANDLE;  // and the rest
  VkCommandBuffer m_cmdBand  = VK_NULL_HANDLE;  // its top band, queued ahead (startBand)
  VkFence         m_fenceFirst = VK_NULL_HANDLE;   // the first part done
  VkFence         m_fence   = VK_NULL_HANDLE;   // the rest done (without the copy engine, the download too)
  std::chrono::steady_clock::time_point m_submitted;
  VkQueryPool     m_queries = VK_NULL_HANDLE;   // the profiling timestamps
  struct Download                               // a picture's download on the copy engine
  {
    VkCommandPool     pool      = VK_NULL_HANDLE;
    VkCommandBuffer   cmd       = VK_NULL_HANDLE;
    VkSemaphore       semaphore = VK_NULL_HANDLE;   // the rest done, for the copies
    VkFence           fence     = VK_NULL_HANDLE;   // the copies done
    VkQueryPool       queries   = VK_NULL_HANDLE;
    std::atomic<bool> pending{ false };             // queued, not yet waited for (endDownload)
  } m_downloads[2];
  int             m_download = 0;   // the next picture's
  const char*     m_labels[kMaxMarks];
  uint32_t        m_marks   = 0;
  uint32_t        m_gapMark = 0;   // the mark that ends the time between the picture's parts (0: none)
  VkQueryPool     m_bandQueries = VK_NULL_HANDLE;   // the band's profiling timestamps, its marks counted as the rest's
  const char*     m_bandLabels[kBandMarks];
  uint32_t        m_bandMarks = 0;
  struct Prepared                  // prepareFilters' results
  {
    const Picture* picture = nullptr;
    bool           deblock = false, sao = false;
    size_t         dbSize = 0, alfSize = 0, saoSize = 0, lmcsSize = 0;
  } m_prepared;
  // the buffers the CPU only writes, in order, and the shaders read -- the
  // filters' parameters, the reference table, the jobs, the edges, the
  // transform matrices -- are MemKind::Upload: in the GPU's own memory where
  // the BAR allows (VideoQ 4K: the GPU's time a picture -1%; the CPU's
  // write-combined copies of the jobs no slower)
  Buffer          m_params;
  Buffer          m_refs;                           // the reference table: the reference pictures' planes on the GPU
  Buffer          m_scratch[MAX_NUM_COMPONENT];     // a picture's worth of GPU memory (SAO's or ALF's output), from the pool while its work is recorded
  Buffer          m_dmvr;                           // DMVR's refinements
  Buffer          m_intraJobs;                      // the intra jobs of a picture, wave by wave, then what each waits for
  Buffer          m_intraSync;                      // shaders/intra.slang's counts of the jobs taken and done
  Buffer          m_interJobs;
  Buffer          m_trafoJobs;
  Buffer          m_bandInterJobs;                  // the top band's
  Buffer          m_bandTrafoJobs;
  Buffer          m_edges;                          // the edges the deblocking filters
  Buffer          m_matrices;                       // TrQuant's transform matrices
  Buffer          m_mip;                            // MIP's matrices
  Buffer          m_lfnst;                          // LFNST's matrices
  Buffer          m_geo;                            // GPM's masks of weights
  // LMCS (setPicture): its tables (the forward and inverse luma mappings,
  // int16, the chroma scales, int32); the planes inter's predictions are
  // kept aside in (LMCS's chroma scaling, CIIP)
  bool            m_lmcsOn    = false;
  size_t          m_lmcsCount = 0;                  // the values a table
  Buffer          m_lmcs;
  Buffer          m_defer;
  Planes          m_deferPlanes{};
  // the dequantisation matrices of explicit scaling lists (setPicture), a byte a factor
  bool            m_scalingOn = false;
  Buffer          m_scaling;
};

}   // namespace gpu
}   // namespace vvdec
