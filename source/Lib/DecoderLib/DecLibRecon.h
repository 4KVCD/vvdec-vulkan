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

/** \file     DecLibRecon.h
    \brief    decoder class (header)
*/

#pragma once

#include "CommonLib/CommonDef.h"
#include "CommonLib/AdaptiveLoopFilter.h"
#include "CommonLib/LoopFilter.h"
#include "CommonLib/InterPrediction.h"
#include "CommonLib/IntraPrediction.h"
#include "CommonLib/Picture.h"
#include "CommonLib/RdCost.h"
#include "CommonLib/Reshape.h"
#include "CommonLib/SampleAdaptiveOffset.h"
#include "CommonLib/TrQuant.h"

#include "DecCu.h"
#include "Utilities/ThreadPool.h"

#if VVDEC_VULKAN
#  include "VulkanLib/GpuRecon.h"
#endif


namespace vvdec
{

class DecLibRecon;
class InterPrediction;
class TrQuant;

//! \ingroup DecoderLib
//! \{
// ====================================================================================================================
// Class definition
// ====================================================================================================================


enum TaskType
{
  /*TRAFO=-1,*/ MIDER, MIDER_cont, LF_INIT, INTER, INTRA, INTRA_cont, RSP, LF_V, LF_V_cont, LF_H, PRESAO, SAO, SAO_cont, ALF, DONE, DMVR
};
using CtuState = std::atomic<TaskType>;

struct CommonTaskParam
{
  DecLibRecon&             decLib;
  CodingStructure*         cs           = nullptr;
  std::vector<CtuState>    ctuStates;
  std::vector<MotionHist>  perLineMiHist;

  bool                     doALF        = true;
  bool                     doSAO        = true;   // the CTU tasks apply SAO (not when the GPU does)
  bool                     doDeblock    = true;   // the CTU tasks deblock (not when the GPU does)
  bool                     gpuIntra     = false;  // the GPU predicts the intra and IBC blocks: the CTU tasks make its jobs
  bool                     gpuInter     = false;  // and the inter blocks
  bool                     gpuResidual  = false;  // and the residuals
  bool                     gpuFront     = false;  // the GPU's front end drives the picture: the CTU tasks only parse

  explicit CommonTaskParam( DecLibRecon* dec ) : decLib( *dec ) {}
  void reset( CodingStructure& cs, TaskType ctuStartState, int tasksPerLine, bool doALF, bool doSAO, bool doDeblock, bool gpuIntra, bool gpuInter,
              bool gpuResidual );
};

struct LineTaskParam
{
  CommonTaskParam& common;
  int              line;
};

struct CtuTaskParam
{
  CommonTaskParam& common;
  int              taskLine;
  int              taskCol;
  int              ctuStart;
  int              ctuEnd;
  int              numColPerTask;
  int              numTasksPerLine;
};

struct FinishPicTaskParam
{
  DecLibRecon* decLib;
  Picture*     pic;
#if VVDEC_VULKAN
  int          download = -1;   // the picture's download on the GPU (GpuRecon::startPicture)
#endif

  FinishPicTaskParam() : decLib( nullptr ), pic( nullptr ) {}
  FinishPicTaskParam( DecLibRecon* _dec, Picture* _pic ) : decLib( _dec ), pic( _pic ) {}
};

struct PerThreadResource
{
  IntraPrediction m_cIntraPred;
  InterPrediction m_cInterPred;
  TrQuant         m_cTrQuant;
  Reshape         m_cReshaper;
  DecCu           m_cCuDecoder;

  explicit PerThreadResource() : m_cTrQuant( &m_cInterPred ) {}
  explicit PerThreadResource( TrQuant& trQuant0 ) : m_cTrQuant( &m_cInterPred, &trQuant0 ) {}
};

/// decoder class
class DecLibRecon
{
private:
  PerThreadResource  **m_pcThreadResource;
  RdCost               m_cRdCost;
  LoopFilter           m_cLoopFilter;
  SampleAdaptiveOffset m_cSAO;
  AdaptiveLoopFilter   m_cALF;

  int                  m_numDecThreads = 0;
  ThreadPool*          m_decodeThreadPool;
  bool                 m_upscaleOutputEnabled = false;

  Picture*             m_currDecompPic = nullptr;
#if TRACE_ENABLE_ITT
  __itt_domain*        m_itt_decInst = nullptr;
#endif

  std::unique_ptr<Pel[], AlignedDeleter<Pel>> 
                       m_predBuf;
  ptrdiff_t            m_predBufSize     = 0;
  Mv*                  m_dmvrMvCache     = nullptr;
  size_t               m_dmvrMvCacheSize = 0;
  ptrdiff_t            m_num4x4Elements  = 0;
  LoopFilterParam*     m_loopFilterParam = nullptr;
  MotionInfo*          m_motionInfo      = nullptr;

  PelStorage           m_fltBuf;

  bool                 m_doALF     = false;   // the in-loop filters of the current picture
  bool                 m_doSAO     = false;
  bool                 m_doDeblock = true;

#if VVDEC_VULKAN
  gpu::Context*                  m_gpu = nullptr;   // the Vulkan device, when the pictures are reconstructed on the GPU
  std::unique_ptr<gpu::GpuRecon> m_gpuRecon;        // then, from the instance's first picture on
  bool                           m_gpuFilters = false;   // the current picture's in-loop filters run on the GPU
  bool                           m_gpuIntra   = false;   // and its intra and IBC blocks
  bool                           m_gpuInter   = false;   // and its inter blocks
  bool                           m_gpuResidual = false;  // and its residuals
  gpu::WaveMap                   m_waveMap;
  std::vector<std::vector<gpu::IntraJob>> m_gpuIntraJobs;   // per CTU
  std::vector<std::vector<gpu::InterJob>> m_gpuInterJobs;   // per CTU
  std::vector<std::vector<gpu::TrafoJob>> m_gpuTrafoJobs;   // per CTU
  std::vector<gpu::EdgeLists>    m_gpuEdges;               // per CTU: the edges the GPU deblocks
  gpu::RefTable                  m_refTable;               // the reference pictures of the current picture
  gpu::LmcsTables                m_lmcsTables;             // the current picture's LMCS (GpuRecon::setLmcs)
  int                            m_filterPicturesGpu = 0;
  int                            m_filterPicturesCpu = 0;
  int                            m_intraPicturesGpu  = 0;
  int                            m_intraPicturesCpu  = 0;
  int                            m_interPicturesGpu  = 0;
  int                            m_interPicturesCpu  = 0;
  int                            m_residualPicturesGpu = 0;
  int                            m_residualPicturesCpu = 0;
  // VVDEC_GPU_MIDER=1: the GPU derives the motion too, from the parse's CUs,
  // and its motion is compared with the CPU's (GpuRecon::compareMider)
  bool                           m_gpuMider = false;
  // the GPU's front end drives the pictures (all their stages on the GPU; VVDEC_GPU_FRONT=0: not): the CTU tasks parse and
  // write the records, the GPU does the rest
  bool                           m_gpuFrontOn = false;
  bool                           m_gpuFront   = false;   // this picture's
  std::vector<std::vector<gpu::CuRecord>> m_gpuCuRecords;   // per CTU
  std::vector<std::vector<gpu::TuRecord>> m_gpuTuRecords;   // per CTU
#endif

  CommonTaskParam            commonTaskParam{ this };
  std::vector<LineTaskParam> tasksFinishMotion;
  std::vector<CtuTaskParam>  tasksCtu;
  // the pictures' in turn: a picture's last tasks may still run while the
  // next picture's are queued (releasePrevDecompressedPic)
  FinishPicTaskParam         taskFinishPic[2];
  int                        m_taskFinishPic = 0;
#if VVDEC_VULKAN
  bool                       m_currDecompReleasable = false;   // the current picture's GPU work ends with a download alone (gpuComputeDone)
  Barrier                    m_gpuFiltersPrepared;   // the picture's filter parameters for the GPU worked out (GpuRecon::prepareFilters)
  // the picture's top rows of CTUs whose residuals and inter blocks go to the
  // GPU ahead of the rest (0: none), their CTU tasks done, their jobs queued
  int                        m_gpuBandRows = 0;
  std::atomic<bool>          m_gpuBandDone{ false };
  Barrier                    m_gpuBandQueued;
#endif
  CBarrierVec                picBarriers;

public:
  DecLibRecon();
  ~DecLibRecon() = default;
  CLASS_COPY_MOVE_DELETE( DecLibRecon )

#if VVDEC_VULKAN
  void create( ThreadPool* threadPool, unsigned instanceId, bool upscaleOutputEnabled, gpu::Context* gpu = nullptr );
#else
  void create( ThreadPool* threadPool, unsigned instanceId, bool upscaleOutputEnabled );
#endif
  void destroy();

  void     decompressPicture( Picture* pcPic );
  Picture* waitForPrevDecompressedPic();
  // as waitForPrevDecompressedPic, but a picture whose work on the GPU is done
  // but for its download comes back then, not yet reconstructed: the instance
  // may take the next picture while it downloads. waitForReleasedPic then
  // waits for it.
  Picture* releasePrevDecompressedPic();
  static void waitForReleasedPic( Picture* pic );
  void     cleanupOnException();
  Picture* getCurrPic() const { return m_currDecompPic; }
  void     swapBufs( CodingStructure& cs );
#if VVDEC_VULKAN
  int      filterPicturesGpu() const { return m_filterPicturesGpu; }
  int      filterPicturesCpu() const { return m_filterPicturesCpu; }
  int      intraPicturesGpu()  const { return m_intraPicturesGpu; }
  int      intraPicturesCpu()  const { return m_intraPicturesCpu; }
  int      interPicturesGpu()  const { return m_interPicturesGpu; }
  int      interPicturesCpu()  const { return m_interPicturesCpu; }
  int      residualPicturesGpu() const { return m_residualPicturesGpu; }
  int      residualPicturesCpu() const { return m_residualPicturesCpu; }
  const gpu::MiderStats* miderStats() const { return m_gpuMider && m_gpuRecon ? &m_gpuRecon->miderStats() : nullptr; }
#endif


private:
  void borderExtPic       ( Picture* pic, const Picture* currPic );
  void createSubPicRefBufs( Picture* pic, const Picture* currPic );

  template<bool checkReadyState=false>
  static bool ctuTask( int tid, void* task_param );
};

}
