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
    \brief    Records and submits a picture's GPU work: uploads, the front end, the reconstruction passes, the in-loop
              filters and the download

    There is one GpuRecon per decoder instance. Its member functions are defined in several files, by area:
    - GpuRecon.cpp: construction, commands and submissions, the picture's planes and tables, and the order of a
      picture's work (startEarlyRows, startPicture);
    - GpuReconFrontEnd.cpp: the GPU front end (motion, deblocking parameters, jobs, intra order, deblocking edges);
    - GpuReconPrediction.cpp: the residuals, inter prediction and intra prediction;
    - GpuReconFilters.cpp: LMCS's inverse mapping, deblocking, SAO and ALF;
    - GpuFrontEndCheck.cpp: the check of the front end against the CPU's (VVDEC_GPU_MIDER=1).
    README.md describes the overall flow.
*/

#pragma once

#include <atomic>
#include <chrono>
#include <functional>

#include "GpuContext.h"
#include "GpuFrontEnd.h"
#include "GpuFrontEndCheck.h"
#include "GpuJobs.h"
#include "GpuParse.h"
#include "GpuParseService.h"
#include "GpuPictureMemory.h"

#include "CommonLib/Buffer.h"
#include "CommonLib/Common.h"

namespace vvdec
{

class CodingStructure;
class AdaptiveLoopFilter;
class SampleAdaptiveOffset;
class Mv;
class SPS;
struct Picture;
struct MotionInfo;
struct LoopFilterParam;

namespace gpu
{

// The passes whose job counts the GPU front end works out (jobs.slang, intrasort.slang, edges.slang). For pass p,
// the front end's argument buffer holds the pass's indirect dispatch (3 words) at word 3 * p, and its job count and
// first job at words 32 + 2 * p and 33 + 2 * p.
enum FrontPass
{
  kPassTrafo            = 0,
  kPassInter            = 1,   // inter's regular tiles
  kPassInterBdof        = 2,   // the BDOF tiles
  kPassInterAffine      = 3,   // the affine pairs
  kPassIntra            = 4,   // the intra bands
  kPassDeblockLumaVer   = 5,
  kPassDeblockLumaHor   = 6,
  kPassDeblockChromaVer = 7,
  kPassDeblockChromaHor = 8,
};

class GpuRecon
{
public:
  explicit GpuRecon( Context* context );
  ~GpuRecon();
  GpuRecon( const GpuRecon& )            = delete;
  GpuRecon& operator=( const GpuRecon& ) = delete;

  // --- What the GPU covers ---
  // The GPU predicts the intra and IBC blocks of a picture unless it uses ACT (the adaptive colour transform).
  static bool intraSupported( const CodingStructure& cs );
  static bool intraSupported( const SPS& sps );
  // The GPU predicts the inter blocks of a sequence's pictures when it predicts their intra blocks and the bit depth
  // is at most 10.
  static bool interSupported( const SPS& sps );
  static bool interSupported( const CodingStructure& cs );
  // The GPU makes the residuals (dequantization and inverse transforms) of pictures with a bit depth of at most 10.
  static bool residualSupported( const CodingStructure& cs );

  // --- Helpers for the CTU tasks (CPU front end) ---
  // Lists the edges of a CTU that the deblocking filters, with the bytes of their deblocking parameters that
  // deblock.slang needs (after vvdec's LF_INIT stage).
  static void listEdges( const CodingStructure& cs, int ctuRsAddr, EdgeLists& edges );
  // vvdec's cache of DMVR refinements for the current picture, with room for count, in host memory the GPU writes.
  Mv* dmvrMvCache( size_t count );

  // --- A picture's work, in the order of the calls ---
  // Sets up the picture whose work comes next, once the instance's previous picture is done. tables: the picture's
  // LMCS tables (null without LMCS). With LMCS, the luma is predicted and reconstructed in the mapped domain, inter's
  // luma prediction is forward-mapped, the chroma residuals are scaled (tables->chroma), and the luma is mapped back
  // before the in-loop filters. defer: inter prediction keeps some predictions aside for intra jobs to finish
  // (LMCS's chroma scaling, CIIP). scalingMatrices: the dequantization matrices of the picture's explicit scaling
  // lists (TrQuant::getDequantCoeffBuf, scalingCount values; null without).
  void setPicture( const PelStorage& pic, const LmcsTables* tables, bool defer, const int* scalingMatrices = nullptr, size_t scalingCount = 0 );
  // For the GPU front end's jobs: where each explicit scaling list's dequantization matrix starts among the
  // picture's (Quant::getDequantCoeff), plus one, per list (6), log2 width and log2 height (0..6); 0 where none.
  void setDequantOffsets( const std::vector<uint32_t>& offsets ) { m_dequantOffsets = offsets; }
  // Works out the parameters of the picture's in-loop filters (deblocking, SAO, ALF). This runs as a task of its
  // own once the picture is parsed, while its CTU tasks run; startPicture calls it if that task did not run.
  void prepareFilters( CodingStructure& cs, bool deblock, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf );
  // CPU front end only: submits the residuals (jobs.trafo) and inter blocks (jobs.inter) of the picture's first ctus
  // CTUs, the early rows, as soon as their CTU tasks are done, while the CPU still works on the rows below.
  // startPicture then submits the rest (with jobs.earlyCtus = ctus).
  void startEarlyRows( CodingStructure& cs, const PictureJobs& jobs, size_t ctus, PelStorage& pic );
  // Records and submits the picture's work on top of what the CPU reconstructed in pic: the residuals, inter
  // prediction, intra and IBC prediction, then the in-loop filters (deblocking with deblock, SAO when sao is given,
  // ALF and CC-ALF when alf is given). The result stays on the GPU as a reference picture (the picture is then
  // gpuCurrent) and is downloaded to pic. The work is submitted in two parts (see README.md): the first part (the
  // residuals and inter blocks) and the rest. With waitFirst, returns once the first part is done, with DMVR's
  // refinements in dmvrMvCache. Returns the picture's download for endDownload (-1: none of its own).
  int startPicture( CodingStructure& cs, const PictureJobs& jobs, bool deblock, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf,
                    PelStorage& pic, bool waitFirst );
  // Waits for the rest of the picture's work (without a copy queue, for its download too). The next picture's work
  // may be recorded after this, while the download runs.
  void endCompute();
  // Waits for a picture's download. The downloads take turns between two sets of commands, so this must be called
  // before the work of the picture after the next one is recorded.
  void endDownload( int download );

  // --- The check of the GPU front end (VVDEC_GPU_MIDER=1; GpuFrontEndCheck.cpp) ---
  // After endCompute, compares the GPU front end's results with the CPU's: cpu is the picture's motion as vvdec
  // keeps it, lfp its deblocking parameters, jobs the CTU tasks' jobs and edges.
  void                 compareFrontEnd( const CodingStructure& cs, const MotionInfo* cpu, const LoopFilterParam* lfp, const PictureJobs* jobs );
  const FrontEndStats& frontEndStats() const { return m_check.stats; }

  // --- The GPU parse instead of the CPU's (VVDEC_GPU_PARSE=2; GpuReconParse.cpp, GpuParseService.h) ---
  // Whether the GPU parses the picture (sps: its slices') instead of the CPU: only when the GPU front end takes it.
  // The ParseService parses it then, and the front end takes its records from the picture's ParseSlot
  // (PictureJobs::parsed).
  static bool parsesPicture( const Picture& pic, const SPS& sps );

  // --- The check of the GPU parse (VVDEC_GPU_PARSE=1; GpuReconParse.cpp) ---
  // After endCompute, compares the GPU's parse of the picture (PictureJobs::parse) with the CPU's: the records (cus,
  // tus), the SAO and ALF parameters (cs.m_gpuSaoAlf) and the room of the front end's job lists.
  void              compareParse( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, const std::vector<std::vector<TuRecord>>& tus );
  const ParseStats& parseStats() const { return m_parse.stats; }

private:
  // The planes of a picture as the shaders see them: each component's sample (0, 0) and stride, in samples.
  struct Planes
  {
    VkDeviceAddress addr[MAX_NUM_COMPONENT];
    int32_t         stride[MAX_NUM_COMPONENT];
  };
  // Where a picture's planes are, for copyPlanes: in host memory, in the twin, or in the scratch planes.
  enum { kHost, kTwin, kScratch };

  // --- Commands and submissions (GpuRecon.cpp) ---
  // Begins recording m_cmd. With resetPool, the command pool is reset first, and the commands start with a barrier
  // over the work submitted before.
  void beginCommands( bool resetPool = true );
  void beginOneTime( VkCommandBuffer cmd );   // begins recording cmd for a single submission
  // Ends the first part's commands and submits them, then begins the rest's.
  void splitCommands();
  // Ends and submits the rest's commands, and the download of pic: on the copy queue after the rest's semaphore
  // (the download's turn is returned), or at the end of the rest (-1 is returned). With waitFirst, waits for the
  // first part to finish.
  int  submitPicture( const PelStorage& pic, bool waitFirst );
  // Records the download of pic on the copy queue into the commands of the download whose turn it is.
  void recordDownload( const PelStorage& pic, int turn );
  // VVDEC_GPU_PROFILE=1: the GPU time since the previous mark counts as label's.
  void mark( const char* label );
  // VVDEC_GPU_PROFILE=1: adds up the picture's marks; waited is how long the CPU waited for the GPU, in ms.
  void readTimestamps( double waited );

  // Barriers between the commands recorded into m_cmd (or cmd)
  void memoryBarrier( VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess );
  void memoryBarrier( VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage, VkAccessFlags dstAccess );
  void fullBarrier();                // all earlier writes visible to all later commands
  void computeBarrier();             // compute shaders' writes visible to later compute shaders
  void transferToComputeBarrier();   // the writes of copies and fills visible to later compute shaders
  void computeToTransferBarrier();   // compute shaders' writes visible to later copies
  void hostReadBarrier();            // compute shaders' and copies' writes visible to the CPU
  void indirectBarrier();            // the front end's dispatch arguments and job counts visible to the passes

  // Dispatches: bind() binds a shader's pipeline (kShader_...), pushConstants() sets its push constants.
  void bind( int shader );
  void pushConstants( const void* data, size_t size );
  template<class Push>
  void pushConstants( const Push& push )
  {
    pushConstants( &push, sizeof( Push ) );
  }
  void dispatchGroups( uint32_t groupsX, uint32_t groupsY = 1 );
  void dispatchJobs( size_t count );         // one workgroup per job, in rows of 32768 workgroups
  void dispatchIndirect( FrontPass pass );   // with the front end's dispatch arguments for pass
  // Binds shader, sets push and dispatches groupsX x groupsY workgroups.
  template<class Push>
  void dispatch( int shader, const Push& push, uint32_t groupsX, uint32_t groupsY = 1 )
  {
    bind( shader );
    pushConstants( push );
    dispatchGroups( groupsX, groupsY );
  }
  // The address of a pass's job count in the front end's argument buffer (FrontPass)
  VkDeviceAddress frontEndCount( FrontPass pass ) const;

  // --- The picture's planes, tables and parameters (GpuRecon.cpp) ---
  Planes   hostPlanes( const PelStorage& pic );      // the picture in host memory
  Planes   twinPlanes( const PelStorage& pic );      // the picture on the GPU (its twin is allocated when missing)
  Planes   scratchPlanes( const PelStorage& pic );   // scratch planes laid out as pic's (from the pool)
  void     copyPlanes( const PelStorage& pic, int from, int to );
  // Writes the reference table into m_refs: the reference pictures' planes on the GPU, laid out as pic's. Appends
  // to uploads the reference pictures the CPU reconstructed (or filled in), which must be copied to the GPU first.
  void     writeRefTable( const CodingStructure& cs, const RefTable& refs, const PelStorage& pic, std::vector<const Picture*>& uploads );
  // With subpictures treated as pictures: writes each CTU's rectangle (its subpicture's or the picture's) into
  // m_subpics and returns its address. Otherwise returns 0.
  VkDeviceAddress subpicRects( const CodingStructure& cs );
  // The upload buffer of the in-loop filters' parameters, with room for size bytes. It holds the deblocking
  // parameters, then ALF's, SAO's and LMCS's (prepareFilters); these return where each starts on the GPU.
  uint8_t*        params( size_t size );
  VkDeviceAddress deblockParams() const { return m_params.addr; }
  VkDeviceAddress alfParams() const { return m_params.addr + m_prepared.dbSize; }
  VkDeviceAddress saoParams() const { return m_params.addr + m_prepared.dbSize + m_prepared.alfSize; }
  VkDeviceAddress lmcsParams() const { return m_params.addr + m_prepared.dbSize + m_prepared.alfSize + m_prepared.saoSize; }

  // --- The order of a picture's work (GpuRecon.cpp) ---
  // The first part: the picture's planes uploaded (when the CPU made the residuals), the front end (without the
  // async queue), the residuals and inter prediction.
  void recordFirstPart( const CodingStructure& cs, const PictureJobs& jobs, const PelStorage& pic, const Planes& planes, bool frontEndAsync );
  // The in-loop filters, from planes (the picture's twin) on: deblocking in place, then SAO and ALF, each from one
  // set of planes to the other (the twin and the scratch planes). Returns whether the result ended in the scratch
  // planes.
  bool recordInLoopFilters( const CodingStructure& cs, const PictureJobs& jobs, const PelStorage& pic, const Planes& planes, bool alf );

  // --- The GPU front end (GpuReconFrontEnd.cpp) ---
  // Records the front end on the async compute queue, in two submissions: part 1 (motion and jobs), which the
  // picture's first part waits for, and part 2 (intra order, deblocking parameters and edges), which its rest waits
  // for. Part 1 waits for the first parts of the picture's colocated pictures.
  void recordFrontEndAsync( const CodingStructure& cs, const PictureJobs& jobs );
  // Ends and submits the front end's part 1 and begins part 2 (recordJobs calls it after the jobs).
  void splitFrontEnd();
  // Records the whole front end into m_cmd: the records uploaded, the motion derivation, the colocated motion, the
  // jobs, the intra order, the deblocking parameters and edges. In the check mode (without m_useFrontEnd), the
  // results are also copied to host memory for compareFrontEnd.
  void recordFrontEnd( const CodingStructure& cs, const PictureJobs& jobs );
  size_t uploadRecords( const CodingStructure& cs, const PictureJobs& jobs );   // returns the picture's CU count
  // With the GPU's parse: records the scatter of the picture's records from its parse slot into the front end's
  // buffers. Returns the picture's CU count.
  size_t useParsedRecords( const CodingStructure& cs, const ParseSlot& parsed );
  // The front end's buffers for a picture of numCus CUs and numTus TUs
  void   ensureFrontEndBuffers( const CodingStructure& cs, size_t numCus, size_t numTus );
  void   recordMotion( const CodingStructure& cs, size_t numCus );
  void   recordColMotion( const CodingStructure& cs );
  void   recordDeblockParams( const CodingStructure& cs, size_t numCus );
  void   recordJobs( const CodingStructure& cs, const PictureJobs& jobs, size_t numCus );
  void   allocateJobBuffers( const CodingStructure& cs, const PictureJobs& jobs, size_t numCus );
  void   recordIntraOrder( const PictureJobs& jobs );
  void   copyJobsForCheck( const PictureJobs& jobs );
  void   recordEdges( const CodingStructure& cs, const PictureJobs& jobs );

  // --- The check of the GPU parse (VVDEC_GPU_PARSE=1; GpuReconParse.cpp) ---
  // Parses the picture's slice data on the GPU in submissions of its own until every substream is parsed, then
  // records into m_cmd the copy of the parse's records to raster order in host memory and the check of its
  // coefficient levels against the CPU's parse's (cpuLevels: the picture in host memory, where the CPU's parse left
  // them).
  void     recordParse( const CodingStructure& cs, const PictureJobs& jobs, const Planes& cpuLevels );
  // Runs the slot's parse on the compute queue, a few dispatches per submission, until every substream is parsed or
  // none advances any more. Returns the number of submissions.
  uint32_t runParse( ParseSlot& slot );
  // Records the commands record() records into the parse's own command buffer, submits them and waits for them.
  void     submitParseCommands( const std::function<void()>& record );

  // --- Reconstruction (GpuReconPrediction.cpp) ---
  // The residuals: the coefficient levels from levels (the picture in host memory, where the parse left them), the
  // residuals into planes. Without the front end, the jobs of CTUs ctuBegin .. ctuEnd - 1 are uploaded into buf.
  void recordTrafo( const CodingStructure& cs, const std::vector<std::vector<TrafoJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf,
                    const Planes& levels, const Planes& planes );
  // Inter prediction into planes. Without the front end, the jobs of CTUs ctuBegin .. ctuEnd - 1 are uploaded into
  // buf: the regular tiles, then BDOF's, then the affine pairs, each set run by its own shader variant.
  void recordInter( const CodingStructure& cs, const std::vector<std::vector<InterJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf,
                    const Planes& planes );
  struct InterCounts
  {
    size_t regular = 0, bdof = 0, affine = 0;   // affine counts jobs: two per affine tile
  };
  InterCounts uploadInterJobs( const std::vector<std::vector<InterJob>>& jobs, size_t ctuBegin, size_t ctuEnd, Buffer& buf );
  // Intra and IBC prediction in dependency waves, in place in planes: with the front end, its bands in wave order;
  // otherwise the CTU tasks' jobs, cut into bands and sorted by wave here.
  void   recordIntra( const CodingStructure& cs, const std::vector<std::vector<IntraJob>>& jobs, const Planes& planes );
  size_t uploadIntraJobs( const std::vector<std::vector<IntraJob>>& jobs );   // returns the number of bands

  // --- The in-loop filters (GpuReconFilters.cpp) ---
  static size_t deblockParamSize( const CodingStructure& cs );
  static size_t alfParamSize( const CodingStructure& cs );
  static size_t saoParamSize( const CodingStructure& cs );
  bool prepareDeblock( const CodingStructure& cs, uint8_t* params );                              // returns false when no slice deblocks
  bool prepareAlf( const CodingStructure& cs, const AdaptiveLoopFilter& alf, uint8_t* params );   // returns true when a CTU has boundaries
  bool prepareSao( CodingStructure& cs, SampleAdaptiveOffset& sao, uint8_t* params );             // returns false when no CTU uses SAO
  // LMCS's inverse luma mapping, in place; ctus: per CTU, whether its slice uses LMCS
  void recordLmcs( const CodingStructure& cs, const Planes& planes, VkDeviceAddress ctus );
  // Deblocking, in place. edges: the CPU front end's edge lists per CTU (unused with the GPU front end).
  void recordDeblock( const CodingStructure& cs, const Planes& planes, VkDeviceAddress params, const std::vector<EdgeLists>& edges );
  void uploadCpuEdges( const std::vector<EdgeLists>& edges, size_t first[5] );   // first: each list's first edge, then the total
  void recordSao( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params );
  void recordAlf( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params );

  // --- State ---
  static constexpr uint32_t kMaxMarks       = 64;   // profiling marks per picture
  static constexpr uint32_t kEarlyRowsMarks = 8;    // and in the early rows' submission

  Context* m_ctx;

  // The compute queue's commands
  VkCommandPool   m_pool         = VK_NULL_HANDLE;
  VkCommandBuffer m_cmd          = VK_NULL_HANDLE;   // the command buffer being recorded
  VkCommandBuffer m_cmdFirst     = VK_NULL_HANDLE;   // the picture's first part
  VkCommandBuffer m_cmdRest      = VK_NULL_HANDLE;   // its rest
  VkCommandBuffer m_cmdEarlyRows = VK_NULL_HANDLE;   // its early rows, submitted ahead (startEarlyRows)
  VkFence         m_fenceFirst   = VK_NULL_HANDLE;   // signalled when the first part is done
  VkFence         m_fence        = VK_NULL_HANDLE;   // signalled when the rest is done (without a copy queue, the download too)
  uint64_t        m_firstValue   = 0;                // the timeline value the first part signals (Context::submitFirst)
  std::chrono::steady_clock::time_point m_submitted;   // when the first part was submitted

  // The front end on an async compute queue (recordFrontEndAsync), in two submissions
  VkCommandPool   m_poolFrontEnd  = VK_NULL_HANDLE;
  int             m_asyncQueue    = 0;                // which async queue (Context::takeAsyncQueue)
  VkCommandBuffer m_cmdFrontEnd1  = VK_NULL_HANDLE;   // part 1: motion and jobs
  VkCommandBuffer m_cmdFrontEnd2  = VK_NULL_HANDLE;   // part 2: intra order, deblocking parameters and edges
  VkSemaphore     m_frontEnd1Done = VK_NULL_HANDLE;
  VkSemaphore     m_frontEnd2Done = VK_NULL_HANDLE;
  VkSemaphore     m_waitFrontEnd1 = VK_NULL_HANDLE;   // what the next first part waits for (part 1's semaphore)
  VkSemaphore     m_waitFrontEnd2 = VK_NULL_HANDLE;   // what the next rest waits for (part 2's semaphore)
  uint64_t        m_frontEndWait  = 0;                // the timeline value part 1 waits for: the colocated pictures' first parts
  uint64_t        m_frontEndParseWait = 0;            // the parse queue's timeline value part 1 waits for: the picture's parse
  bool            m_splitFrontEnd = false;            // recordJobs ends part 1 after the jobs (splitFrontEnd)

  // The downloads on the copy queue: two sets of commands that take turns
  struct Download
  {
    VkCommandPool     pool      = VK_NULL_HANDLE;
    VkCommandBuffer   cmd       = VK_NULL_HANDLE;
    VkSemaphore       semaphore = VK_NULL_HANDLE;   // signalled by the rest, for the copies
    VkFence           fence     = VK_NULL_HANDLE;   // signalled when the copies are done
    VkQueryPool       queries   = VK_NULL_HANDLE;
    std::atomic<bool> pending{ false };             // submitted, and not yet waited for (endDownload)
  } m_downloads[2];
  int m_download = 0;   // the next picture's turn

  // Profiling (VVDEC_GPU_PROFILE=1): timestamps after the passes
  VkQueryPool m_queries = VK_NULL_HANDLE;
  const char* m_labels[kMaxMarks];
  uint32_t    m_marks   = 0;
  uint32_t    m_gapMark = 0;                         // the mark that ends the time between the picture's parts (0: none)
  VkQueryPool m_earlyRowsQueries = VK_NULL_HANDLE;   // the early rows'; their passes count with the rest's of the same name
  const char* m_earlyRowsLabels[kEarlyRowsMarks];
  uint32_t    m_earlyRowsMarks = 0;

  // The picture
  bool m_useFrontEnd   = false;   // the passes take their jobs from the GPU front end (PictureJobs::useFrontEnd)
  bool m_frontEndTrafo = false;   // and the front end makes the residual jobs (PictureJobs::frontEndTrafo)
  bool m_frontEndInter = false;   // and the inter jobs (PictureJobs::frontEndInter)
  struct Prepared                 // prepareFilters' results
  {
    const Picture* picture   = nullptr;
    bool           deblock   = false, sao = false;
    bool           alfBounds = false;   // ALF's BOUNDS variants are needed (a CTU is split or clipped)
    size_t         dbSize = 0, alfSize = 0, saoSize = 0, lmcsSize = 0;
  } m_prepared;
  std::vector<uint32_t> m_dequantOffsets;   // setDequantOffsets
  // LMCS (setPicture): its tables (the forward and inverse luma mappings, int16_t, then the chroma scales, int32_t)
  bool   m_lmcsOn    = false;
  size_t m_lmcsCount = 0;   // the entries of a table
  Buffer m_lmcs;
  // The planes inter prediction keeps some predictions aside in, for intra jobs (LMCS's chroma scaling, CIIP)
  Buffer m_defer;
  Planes m_deferPlanes{};
  // The dequantization matrices of explicit scaling lists (setPicture), a byte per factor
  bool   m_scalingOn = false;
  Buffer m_scaling;

  // Buffers. Those the CPU writes once, in order, and the shaders only read (the parameters, the reference table,
  // the jobs, the edges, the tables) are MemKind::Upload: in the GPU's own memory when the BAR allows it (VideoQ 4K:
  // the GPU's time per picture -1%; the CPU's write-combined copies of the jobs are no slower).
  Buffer m_params;                       // the in-loop filters' parameters (params())
  Buffer m_refs;                         // the reference table: the reference pictures' planes on the GPU
  Buffer m_scratch[MAX_NUM_COMPONENT];   // a picture's worth of GPU memory for SAO's or ALF's output, from the pool while its work is recorded
  Buffer m_dmvr;                         // DMVR's refinements (dmvrMvCache)
  Buffer m_intraSync;                    // intra.slang's counts of the jobs taken and done
  Buffer m_subpics;                      // subpictures treated as pictures: each CTU's rectangle (subpicRects)

  // The constant tables the shaders read, uploaded once
  struct Tables
  {
    Buffer transform;    // TrQuant's transform matrices (trafo.slang)
    Buffer lfnst;        // LFNST's matrices (trafo.slang)
    Buffer mip;          // MIP's matrices (intra.slang)
    Buffer geoWeights;   // GPM's weight masks, g_globalGeoWeights (inter.slang)
  } m_tables;

  // The CPU front end's jobs and edges, uploaded per picture
  struct CpuJobs
  {
    Buffer trafo;
    Buffer inter;
    Buffer intra;        // the bands in wave order, then each band's wait (recordIntra)
    Buffer earlyTrafo;   // the early rows'
    Buffer earlyInter;
    Buffer edges;        // the deblocking edges, list after list
  } m_cpuJobs;

  // The GPU front end's buffers (GpuReconFrontEnd.cpp). The records are uploaded; the rest is GPU memory.
  struct FrontEndBuffers
  {
    Buffer cus, tus, ctus;   // the parse's records (GpuFrontEnd.h)
    // Motion derivation (mider.slang): per 4x4 block, its CU in the luma tree and in the chroma tree; per CU, its
    // motion; per 4x4 block, its motion (5 words); per CTU row, its progress
    Buffer cuMapLuma, cuMapChroma, cuMotion, motion, rowProgress;
    // Deblocking parameters (dbparams.slang): per CTU, its subpicture's index | whether loop filtering crosses the
    // subpicture's boundaries << 16; per 4x4 block and direction, the parameters (2 words)
    Buffer ctuInfo, lfp;
    // Jobs (jobs.slang): its tables (writeJobTables); the slices' words (writeJobSlices); the scaling lists' matrix
    // offsets; per CU, its job counts; per block of 256 CUs, their sums, then the totals; the job lists (residual,
    // inter, intra); each 4x4 block's wave; each CTU row's progress
    Buffer jobTables, jobSlices, jobScaling, jobCounts, jobBlocks, jobs[3], waveMap, jobProgress;
    size_t jobCapacity[3] = { 0, 0, 0 };   // the room of the job lists, in jobs
    int    jobBlockCount  = 0;
    Buffer args;   // the passes' indirect dispatch arguments and job counts (FrontPass)
    // Intra order (intrasort.slang): a radix sort of the intra jobs by wave (the keys, two halves, and the
    // histograms), then the jobs cut into bands (each job's first band, the blocks' band sums, each wave's first
    // band), the bands in wave order with each band's wait, and flags (a wave past 65535)
    Buffer sortKeys, sortHistogram, bandAt, bandSums, waveFirst, sortedBands, bandWaits, sortFlags;
    size_t sortCapacity = 0;   // the room for bands
    int    sortBlocks   = 0;   // blocks of 1024 jobs
    // Deblocking edges (edges.slang): per CTU, its edge counts; the four lists' first edge and count; the edges
    Buffer edgeCounts, edgeLists, edges;
    size_t edgeCapacity = 0;
    FrontCapacities capacities;   // the picture's (frontCapacities)

    void release( Context* ctx );
  } m_frontEnd;

  // The check of the GPU parse (VVDEC_GPU_PARSE=1; GpuReconParse.cpp): its parse of the picture, what it needs
  // besides, its commands
  struct ParseCheck
  {
    std::unique_ptr<ParseSlot> slot;   // the picture's parse
    bool       recorded = false;       // compareParse has results to compare
    int        contexts = 0;           // parsectx.slang's context sets checked against vvdec's: 1 alike, -1 not
    // the levels the parse writes (the CPU's parse has its own in the picture), the records in raster order and the
    // levels check's counts in host memory, for compareParse
    Buffer levels[MAX_NUM_COMPONENT], outCus, outTus, outCtus, check;
    Buffer prof;   // VVDEC_GPU_PROFILE with the shader clock: parse_prof's cycle counts
    bool   profRead = false;
    // the parse's own commands on the compute queue
    VkCommandPool   pool  = VK_NULL_HANDLE;
    VkCommandBuffer cmd   = VK_NULL_HANDLE;
    VkFence         fence = VK_NULL_HANDLE;
    ParseStats      stats;

    void release( Context* ctx );
  } m_parse;

  // The front end's results copied to host memory for the check (VVDEC_GPU_MIDER=1; GpuFrontEndCheck.cpp), and
  // whether the picture recorded each copy
  struct FrontEndCheck
  {
    Buffer        motion, lfp, jobBlocks, jobs[3], sortedBands, bandWaits, sortInfo, edges, edgeLists;
    bool          lfpRecorded = false, jobsRecorded = false, edgesRecorded = false;
    FrontEndStats stats;

    void release( Context* ctx );
  } m_check;
};

}   // namespace gpu
}   // namespace vvdec
