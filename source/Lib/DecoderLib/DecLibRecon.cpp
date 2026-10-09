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

/** \file     DecLibRecon.cpp
    \brief    decoder class
*/

#include "DecLib.h"

#include "CommonLib/TrQuant.h"
#if ENABLE_SIMD_TCOEFF_OPS
#include "CommonLib/TrQuant_EMT.h"
#endif
#include "CommonLib/InterPrediction.h"
#include "CommonLib/IntraPrediction.h"
#include "CommonLib/Unit.h"
#include "CommonLib/Buffer.h"
#include "CommonLib/UnitTools.h"

#include "CommonLib/dtrace_next.h"
#include "CommonLib/dtrace_buffer.h"

namespace vvdec
{

#ifdef TRACE_ENABLE_ITT
extern __itt_domain*              itt_domain_dec;
extern std::vector<__itt_domain*> itt_domain_decInst;

extern __itt_string_handle* itt_handle_alf;
extern __itt_string_handle* itt_handle_presao;
extern __itt_string_handle* itt_handle_sao;
extern __itt_string_handle* itt_handle_lfl;
extern __itt_string_handle* itt_handle_intra;
extern __itt_string_handle* itt_handle_inter;
extern __itt_string_handle* itt_handle_mider;
extern __itt_string_handle* itt_handle_lfcl;
extern __itt_string_handle* itt_handle_ext;
extern __itt_string_handle* itt_handle_dmvr;
extern __itt_string_handle* itt_handle_rsp;

extern __itt_string_handle* itt_handle_schedTasks;
extern __itt_string_handle* itt_handle_waitTasks;

// create global domain for DecLib
extern __itt_domain* itt_domain_glb;
// create a global counter
extern __itt_counter itt_frame_counter;

#define ITT_TASKSTART( d, t ) __itt_task_begin( ( d ), __itt_null, __itt_null, ( t ) )
#define ITT_TASKEND( d, t )   __itt_task_end  ( ( d ) )
#else
#define ITT_TASKSTART( d, t )
#define ITT_TASKEND( d, t )
#endif

//! \ingroup DecoderLib
//! \{

// VVDEC_DEBUG_NO_FILTERS_POC=<poc>: no in-loop filters for that picture, on the
// CPU or on the GPU, so their reconstructions can be compared
static int debugNoFiltersPoc()
{
  static const int poc = getenv( "VVDEC_DEBUG_NO_FILTERS_POC" ) ? atoi( getenv( "VVDEC_DEBUG_NO_FILTERS_POC" ) ) : INT_MIN;
  return poc;
}

void CommonTaskParam::reset( CodingStructure& cs, TaskType ctuStartState, int tasksPerLine, bool _doALF, bool _doSAO, bool _doDeblock, bool _gpuIntra,
                             bool _gpuInter, bool _gpuResidual )
{
  this->cs = &cs;

  const int heightInCtus = cs.pcv->heightInCtus;
  CHECKD( !ctuStates.empty() && std::any_of( ctuStates.begin(), ctuStates.end(), []( CtuState& s ) { return s != DONE; } ), "some CTUs of previous pic not done" );
  ctuStates = std::vector<CtuState>( heightInCtus * tasksPerLine );
  for( auto& ctu: ctuStates )
  {
    ctu.store( ctuStartState );
  }
  perLineMiHist = std::vector<MotionHist>( heightInCtus );
  doALF         = _doALF;
  doSAO         = _doSAO;
  doDeblock     = _doDeblock;
  gpuIntra      = _gpuIntra;
  gpuInter      = _gpuInter;
  gpuResidual   = _gpuResidual;
}

DecLibRecon::DecLibRecon()
{
#if ENABLE_SIMD_OPT_BUFFER
#  if defined( TARGET_SIMD_X86 )
  g_pelBufOP.initPelBufOpsX86();
#  endif
#  if defined( TARGET_SIMD_ARM )
  g_pelBufOP.initPelBufOpsARM();
#  endif
#endif
#if ENABLE_SIMD_TCOEFF_OPS && defined( TARGET_SIMD_X86 )
  g_tCoeffOps.initTCoeffOpsX86();
#endif
#if ENABLE_SIMD_TCOEFF_OPS && defined( TARGET_SIMD_ARM )
  g_tCoeffOps.initTCoeffOpsARM();
#endif
}

#if VVDEC_VULKAN
void DecLibRecon::create( ThreadPool* threadPool, unsigned instanceId, bool upscaleOutputEnabled, gpu::Context* gpu )
#else
void DecLibRecon::create( ThreadPool* threadPool, unsigned instanceId, bool upscaleOutputEnabled )
#endif
{
  // run constructor again to ensure all variables, especially in DecLibParser have been reset
  this->~DecLibRecon();
  new( this ) DecLibRecon;

#if VVDEC_VULKAN
  m_gpu = gpu;   // m_gpuRecon made with the instance's first picture (decompressPicture)
#endif


#if TRACE_ENABLE_ITT
  if( itt_domain_decInst.size() < instanceId + 1 )
  {
    std::string name( "DecLibRecon " + std::to_string( instanceId ) );
    itt_domain_decInst.push_back( __itt_domain_create( name.c_str() ) );
    itt_domain_decInst.back()->flags = 1;

    CHECK_FATAL( itt_domain_decInst.back() != itt_domain_decInst[instanceId], "current decLibRecon ITT-Domain is not the last in vector. Instances created in the wrong order?" );
  }
  m_itt_decInst = itt_domain_decInst[instanceId];
#endif

  m_decodeThreadPool = threadPool;
  m_numDecThreads    = std::max( 1, threadPool ? threadPool->numThreads() : 1 );

  m_upscaleOutputEnabled = upscaleOutputEnabled;
  m_predBufSize     = 0;
  m_dmvrMvCacheSize = 0;
  m_dmvrMvCache     = nullptr;

  m_num4x4Elements   = 0;
  m_loopFilterParam  = nullptr;
  m_motionInfo       = nullptr;

  m_pcThreadResource = nullptr;   // with the instance's first picture (decompressPicture)
}

void DecLibRecon::destroy()
{
  m_decodeThreadPool = nullptr;

  if( m_predBuf )
  {
    m_predBuf.reset();
    m_predBufSize = 0;
  }

  if( m_dmvrMvCache )
  {
    free( m_dmvrMvCache );
    m_dmvrMvCache = nullptr;
    m_dmvrMvCacheSize = 0;
  }

  if( m_loopFilterParam )
  {
    free( m_loopFilterParam );
    m_loopFilterParam = nullptr;
  }

  if( m_motionInfo )
  {
    free( m_motionInfo );
    m_motionInfo = nullptr;
  }

  m_num4x4Elements = 0;

  if( m_pcThreadResource )
  {
    for( int i = 0; i < m_numDecThreads; i++ ) delete m_pcThreadResource[i];
    delete[] m_pcThreadResource; m_pcThreadResource = nullptr;
  }

  // the filter buffer may be in the GPU's memory: free it while the device is there
  m_fltBuf.destroy();
#if VVDEC_VULKAN
  m_gpuRecon.reset();
#endif
}


static void getCompatibleBuffer( const CodingStructure& cs, const CPelUnitBuf& srcBuf, PelStorage& destBuf, const UserAllocator* userAllocator )
{
  if( !destBuf.bufs.empty() )
  {
    bool compat = false;
    if( destBuf.chromaFormat == srcBuf.chromaFormat )
    {
      compat = true;
      const uint32_t numCh = getNumberValidComponents( srcBuf.chromaFormat );
      for( uint32_t i = 0; i < numCh; i++ )
      {
        // check this otherwise it would turn out to get very weird
        compat &= destBuf.get( ComponentID( i ) )         == srcBuf.get( ComponentID( i ) );
        compat &= destBuf.get( ComponentID( i ) ).stride  == srcBuf.get( ComponentID( i ) ).stride;
        compat &= destBuf.get( ComponentID( i ) ).width   == srcBuf.get( ComponentID( i ) ).width;
        compat &= destBuf.get( ComponentID( i ) ).height  == srcBuf.get( ComponentID( i ) ).height;
      }
    }
    if( !compat )
    {
      destBuf.destroy();
    }
  }
  if( destBuf.bufs.empty() )
  {
    destBuf.create( cs.picture->chromaFormat, cs.picture->lumaSize(), cs.pcv->maxCUWidth, cs.picture->margin, MEMORY_ALIGN_DEF_SIZE, true, userAllocator );
  }
}

void DecLibRecon::borderExtPic( Picture* pic, const Picture* currPic )
{
  // we block and wait here, so the exceptions from the reference pic don't propagate to the current picture
  pic->waitForAllTasks();
  if( pic->progress < Picture::reconstructed )   // an exception must have happended in the picture, so we need to clean it up
  {
    CHECK_FATAL( pic->progress < Picture::parsing, "Slice parsing should have started, so all structures are there" );
    try
    {
      pic->reconDone.checkAndRethrowException();
      pic->parseDone.checkAndRethrowException();  // when the error happened in the slice parsing tasks, there might not be an exception in recon done, so check parseDone also
    }
    catch( ... )
    {
      pic->error = true;
      pic->reconDone.clearException();
      // TODO: for now we set it on parseDone, so we can handle it outside:
      if( !pic->parseDone.hasException() )
      {
        pic->parseDone.setException( std::current_exception() );
      }

      pic->fillGrey( currPic->cs->sps.get() );
    }
  }

  pic->borderExtStarted = true;

  const bool wrapAround = pic->cs->sps->getUseWrapAround();
  if( wrapAround )
  {
    // copy reconstruction buffer to wrapAround buffer. All other border-extension tasks depend on this task.
    static auto copyTask = []( int, void* task_param )
    {
      ITT_TASKSTART( itt_domain_dec, itt_handle_ext );
      Picture* picture = static_cast<Picture*>( task_param );
      picture->getRecoBuf( true ).copyFrom( picture->getRecoBuf() );
      ITT_TASKEND( itt_domain_dec, itt_handle_ext );
      return true;
    };
    pic->m_copyWrapBufDone.lock();
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string( currPic->poc ) + " copyTask Ref-POC:" + std::to_string( pic->poc ) )
                                        copyTask,
                                        pic,
                                        &pic->m_borderExtTaskCounter,
                                        &pic->m_copyWrapBufDone,
                                        { &pic->reconDone } );
  }

  // start actual border extension tasks
  {
    static auto task = []( int, void* task_param )
    {
      ITT_TASKSTART( itt_domain_dec, itt_handle_ext );
      Picture* picture = static_cast<Picture*>( task_param );
      picture->extendPicBorder( true, false, false, false );
      ITT_TASKEND( itt_domain_dec, itt_handle_ext );
      return true;
    };
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string(currPic->poc) + " borderExtTask T Ref-POC:" + std::to_string(pic->poc) )
                                        task,
                                        pic,
                                        &pic->m_borderExtTaskCounter,
                                        nullptr,
                                        { wrapAround ? &pic->m_copyWrapBufDone : &pic->reconDone } );
  }

  {
    static auto task = []( int, void* task_param )
    {
      ITT_TASKSTART( itt_domain_dec, itt_handle_ext );
      Picture* picture = static_cast<Picture*>( task_param );
      picture->extendPicBorder( false, true, false, false );
      ITT_TASKEND( itt_domain_dec, itt_handle_ext );
      return true;
    };
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string(currPic->poc) + " borderExtTask B Ref-POC:" + std::to_string(pic->poc) )
                                        task,
                                        pic,
                                        &pic->m_borderExtTaskCounter,
                                        nullptr,
                                        { wrapAround ? &pic->m_copyWrapBufDone : &pic->reconDone } );
  }

  {
    static auto task = []( int, void* task_param )
    {
      ITT_TASKSTART( itt_domain_dec, itt_handle_ext );
      Picture* picture = static_cast<Picture*>( task_param );
      picture->extendPicBorder( false, false, true, false, CH_L );
      ITT_TASKEND( itt_domain_dec, itt_handle_ext );
      return true;
    };
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string(currPic->poc) + " borderExtTask ltT Ref-POC:" + std::to_string(pic->poc) )
                                        task,
                                        pic,
                                        &pic->m_borderExtTaskCounter,
                                        nullptr,
                                        { wrapAround ? &pic->m_copyWrapBufDone : &pic->reconDone } );
  }
  {
    static auto task = []( int, void* task_param )
    {
      ITT_TASKSTART( itt_domain_dec, itt_handle_ext );
      Picture* picture = static_cast<Picture*>( task_param );
      picture->extendPicBorder( false, false, false, true, CH_L );
      ITT_TASKEND( itt_domain_dec, itt_handle_ext );
      return true;
    };
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string(currPic->poc) + " borderExtTask lrB Y Ref-POC:" + std::to_string(pic->poc) )
                                        task,
                                        pic,
                                        &pic->m_borderExtTaskCounter,
                                        nullptr,
                                        { wrapAround ? &pic->m_copyWrapBufDone : &pic->reconDone } );
  }

  {
    static auto task = []( int, void* task_param )
    {
      ITT_TASKSTART( itt_domain_dec, itt_handle_ext );
      Picture* picture = static_cast<Picture*>( task_param );
      picture->extendPicBorder( false, false, true, false, CH_C );
      ITT_TASKEND( itt_domain_dec, itt_handle_ext );
      return true;
    };
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string(currPic->poc) + " borderExtTask lrB UV Ref-POC:" + std::to_string(pic->poc) )
                                        task,
                                        pic,
                                        &pic->m_borderExtTaskCounter,
                                        nullptr,
                                        { wrapAround ? &pic->m_copyWrapBufDone : &pic->reconDone } );
  }
  {
    static auto task = []( int, void* task_param )
    {
      ITT_TASKSTART( itt_domain_dec, itt_handle_ext );
      Picture* picture = static_cast<Picture*>( task_param );
      picture->extendPicBorder( false, false, false, true, CH_C );
      ITT_TASKEND( itt_domain_dec, itt_handle_ext );
      return true;
    };
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string(currPic->poc) + " borderExtTask lrB UV Ref-POC:" + std::to_string(pic->poc) )
                                        task,
                                        pic,
                                        &pic->m_borderExtTaskCounter,
                                        nullptr,
                                        { wrapAround ? &pic->m_copyWrapBufDone : &pic->reconDone } );
  }
}

void DecLibRecon::createSubPicRefBufs( Picture* pic, const Picture* currPic )
{
  pic->subPicExtStarted = true;

  const PPS* pps       = pic->cs->pps.get();
  const SPS* sps       = pic->cs->sps.get();
  const int  numSubPic = pps->getNumSubPics();

  pic->m_subPicRefBufs .resize( numSubPic );
  pic->m_subPicExtTasks.resize( numSubPic );
  for( int i = 0; i < numSubPic; ++i )
  {
    const SubPic& currSubPic = pps->getSubPic( i );
    const Area    subPicArea( currSubPic.getSubPicLeft(),
                              currSubPic.getSubPicTop(),
                              currSubPic.getSubPicWidthInLumaSample(),
                              currSubPic.getSubPicHeightInLumaSample() );

    pic->m_subPicRefBufs[i].create( pic->chromaFormat, Size( subPicArea ), sps->getMaxCUWidth(), pic->margin, MEMORY_ALIGN_DEF_SIZE );

    static auto task = []( int, void* task_param )
    {
      SubPicExtTask* t = static_cast<SubPicExtTask*>( task_param );
      t->subPicBuf->copyFrom( t->picture->getRecoBuf().subBuf( t->subPicArea ) );
      t->picture->extendPicBorderBuf( *t->subPicBuf );
      return true;
    };
    pic->m_subPicExtTasks[i] = SubPicExtTask{ pic, &pic->m_subPicRefBufs[i], subPicArea };
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string( currPic->poc ) + " subPicBorderExtTask refPOC:" + std::to_string( pic->poc ) )
                                        task,
                                        &pic->m_subPicExtTasks[i],
                                        &pic->m_borderExtTaskCounter,
                                        nullptr,
                                        { &pic->reconDone } );
  }
}

void DecLibRecon::swapBufs( CodingStructure& cs )
{
  cs.picture->m_bufs[PIC_RECONSTRUCTION].swap( m_fltBuf );
  cs.rebindPicBufs();   // ensure the recon buf in the coding structure points to the correct buffer
}

void DecLibRecon::decompressPicture( Picture* pcPic )
{
  m_currDecompPic = pcPic;
#if VVDEC_VULKAN
  m_currDecompReleasable = false;
#endif
  FinishPicTaskParam& finishParam = taskFinishPic[m_taskFinishPic ^= 1];

  CodingStructure& cs = *pcPic->cs;

  pcPic->progress = Picture::reconstructing;

#ifdef TRACE_ENABLE_ITT
  // mark start of frame
    pcPic->m_itt_decLibInst = m_itt_decInst;
  __itt_frame_begin_v3( pcPic->m_itt_decLibInst, nullptr );
#endif

  // Initialise the various objects for the new set of settings
  const SPS * sps = cs.sps.get();
  const PPS * pps = cs.pps.get();

  // the threads' resources and the GPU's commands, made with the instance's
  // first picture: an instance left idle (with the GPU, the pictures up to
  // 1080p take two of the four) goes without them
#if VVDEC_VULKAN
  if( m_gpu && !m_gpuRecon )
  {
    m_gpuRecon = std::make_unique<gpu::GpuRecon>( m_gpu );
  }
#endif
  if( !m_pcThreadResource )
  {
    m_pcThreadResource    = new PerThreadResource*[m_numDecThreads];
    m_pcThreadResource[0] = new PerThreadResource();
    for( int i = 1; i < m_numDecThreads; i++ )
    {
      m_pcThreadResource[i] = new PerThreadResource( m_pcThreadResource[0]->m_cTrQuant );
    }
  }

  for( int i = 0; i < m_numDecThreads; i++ )
  {
    if( sps->getUseReshaper() )
    {
      m_pcThreadResource[i]->m_cReshaper.createDec( sps->getBitDepth() );
      m_pcThreadResource[i]->m_cReshaper.initSlice( pcPic->slices[0]->getNalUnitLayerId(), *pcPic->slices[0]->getPicHeader(), pcPic->slices[0]->getVPS_nothrow() );
    }

    m_pcThreadResource[i]->m_cIntraPred.init( sps->getChromaFormatIdc(), sps->getBitDepth() );
    m_pcThreadResource[i]->m_cInterPred.init( &m_cRdCost, sps->getChromaFormatIdc(), sps->getMaxCUHeight() );

    // Recursive structure
    m_pcThreadResource[i]->m_cTrQuant.init( pcPic );
    m_pcThreadResource[i]->m_cCuDecoder.init( &m_pcThreadResource[i]->m_cIntraPred, &m_pcThreadResource[i]->m_cInterPred, &m_pcThreadResource[i]->m_cReshaper, &m_pcThreadResource[i]->m_cTrQuant );
  }

  const bool isIntra = std::all_of( pcPic->slices.begin(), pcPic->slices.end(), []( const Slice* pcSlice ) { return pcSlice->isIntra(); } );
#if VVDEC_VULKAN
  // the GPU applies the in-loop filters to the whole picture once its CTUs are
  // reconstructed (the finish tasks)
  m_gpuFilters = m_gpuRecon && gpu::GpuRecon::filtersSupported( cs );
  // and, before the filters, the intra and IBC blocks, in waves of blocks the
  // CTU tasks list (TaskGpuCtu)
  m_gpuIntra   = m_gpuFilters && gpu::GpuRecon::intraSupported( cs );
  // and, before those, the inter blocks, from the reference pictures on the GPU
  m_gpuInter   = m_gpuIntra && !isIntra && gpu::GpuRecon::interSupported( cs );
  // and, first of all, the residuals, when the CPU predicts nothing
  m_gpuResidual = m_gpuIntra && ( isIntra || m_gpuInter ) && gpu::GpuRecon::residualSupported( cs );
  pcPic->gpuEarlyRef = false;
  // the picture-sized buffers of what the CPU does itself, the filters' output
  // (SAO, ALF) and the prediction, allocated only for that (an instance's: 4K
  // 28 + 25 MB)
  const bool cpuFilters  = !m_gpuFilters;
  const bool cpuPredicts = !m_gpuIntra || ( !isIntra && !m_gpuInter );
#else
  const bool cpuFilters  = true;
  const bool cpuPredicts = true;
#endif
  if( cpuFilters )
  {
    getCompatibleBuffer( *pcPic->cs, pcPic->cs->getRecoBuf(), m_fltBuf, pcPic->getUserAllocator() );
  }

  const uint32_t  log2SaoOffsetScale = (uint32_t) std::max(0, sps->getBitDepth() - MAX_SAO_TRUNCATED_BITDEPTH);
  const int maxDepth = getLog2(sps->getMaxCUWidth()) - pps->pcv->minCUWidthLog2;
  m_cSAO.create( pps->getPicWidthInLumaSamples(),
                 pps->getPicHeightInLumaSamples(),
                 sps->getChromaFormatIdc(),
                 sps->getMaxCUWidth(),
                 sps->getMaxCUHeight(),
                 maxDepth,
                 log2SaoOffsetScale,
                 m_fltBuf
               );

  if( sps->getUseALF() )
  {
    m_cALF.create( cs.picHeader.get(), sps, pps, m_numDecThreads, m_fltBuf );
  }

  const PreCalcValues* pcv = cs.pcv;

  // set reconstruction buffers in CodingStructure
  const ptrdiff_t ctuSampleSizeL = pcv->maxCUHeight * pcv->maxCUWidth;
  const ptrdiff_t ctuSampleSizeC = isChromaEnabled( pcv->chrFormat ) ? ( ctuSampleSizeL >> ( getChannelTypeScaleX( CH_C, pcv->chrFormat ) + getChannelTypeScaleY( CH_C, pcv->chrFormat ) ) ) : 0;
  const ptrdiff_t ctuSampleSize  = ctuSampleSizeL + 2 * ctuSampleSizeC;
  const size_t    predBufSize    = ctuSampleSize * pcv->sizeInCtus;
  if( cpuPredicts && predBufSize != m_predBufSize )
  {
    m_predBuf.reset( ( Pel* ) xMalloc( Pel, predBufSize ) );
    m_predBufSize = predBufSize;
  }

  pcPic->cs->m_predBuf = cpuPredicts ? m_predBuf.get() : nullptr;

  // for the worst case of all PUs being 8x8 and using DMVR
  const size_t _maxNumDmvrMvs = pcv->num8x8CtuBlks * pcv->sizeInCtus;
  if( _maxNumDmvrMvs != m_dmvrMvCacheSize )
  {
    if( m_dmvrMvCache ) free( m_dmvrMvCache );
    m_dmvrMvCacheSize = _maxNumDmvrMvs;
    m_dmvrMvCache     = ( Mv* ) malloc( sizeof( Mv ) * _maxNumDmvrMvs );
  }

  pcPic->cs->m_dmvrMvCache = m_dmvrMvCache;

  if( m_num4x4Elements != cs.pcv->num4x4CtuBlks * cs.pcv->sizeInCtus )
  {
    if( m_loopFilterParam ) free( m_loopFilterParam );
    if( m_motionInfo      ) free( m_motionInfo );

    m_num4x4Elements = cs.pcv->num4x4CtuBlks * cs.pcv->sizeInCtus;

    // (with the GPU too: its deblocking takes the parameters it needs from the
    // edge lists, GpuRecon::listEdges)
    m_loopFilterParam = ( LoopFilterParam* ) malloc( sizeof( LoopFilterParam ) * m_num4x4Elements * 2 );
    m_motionInfo      = ( MotionInfo* )      malloc( sizeof( MotionInfo      ) * m_num4x4Elements );
  }
  // finished

  const int widthInCtus  = cs.pcv->widthInCtus;
  const int heightInCtus = cs.pcv->heightInCtus;

  pcPic->startProcessingTimer();

  if( m_decodeThreadPool->numThreads() > 0 )
  {
    ITT_TASKSTART( itt_domain_dec, itt_handle_schedTasks );
  }

  picBarriers.clear();
#if ALLOW_MIDER_LF_DURING_PICEXT
  CBarrierVec  picExtBarriers;
#else
  CBarrierVec &picExtBarriers = picBarriers;
#endif

  const int numSubPic = cs.pps->getNumSubPics();

  std::vector<Picture*> borderExtRefPics( pcPic->buildAllRefPicsVec() );
  for( Picture* refPic : borderExtRefPics )
  {
#if VVDEC_VULKAN
    // inter prediction on the GPU reads the reference pictures there (no border
    // extension in host memory): the CTU tasks wait for a reference the GPU
    // reconstructs only until it can be referenced (its motion for TMVP final,
    // its GPU work queued), not for the whole picture
    if( m_gpuInter && refPic->gpuEarlyRef && !refPic->gpuRefReady.hasException() )
    {
      if( refPic->gpuRefReady.isBlocked() && std::find( picBarriers.cbegin(), picBarriers.cend(), &refPic->gpuRefReady ) == picBarriers.cend() )
      {
        picBarriers.push_back( &refPic->gpuRefReady );
      }
      continue;
    }
#endif
    if( !refPic->borderExtStarted )
    {
      // TODO: (GH) Can we bypass this border extension, when all subpics (>1) are treated as pics?
      borderExtPic( refPic, pcPic );
    }

    if( !refPic->subPicExtStarted && numSubPic > 1 && refPic->m_subPicRefBufs.size() != numSubPic )
    {
      CHECK( !refPic->m_subPicRefBufs.empty(), "Wrong number of subpics already present in reference picture" );
      CHECK( cs.sps->getUseWrapAround(), "Wraparound + subpics not implemented" );

      createSubPicRefBufs( refPic, pcPic );
    }

    if( refPic->m_borderExtTaskCounter.isBlocked() &&
        std::find( picExtBarriers.cbegin(), picExtBarriers.cend(), refPic->m_borderExtTaskCounter.donePtr() ) == picExtBarriers.cend() )
    {
      picExtBarriers.push_back( refPic->m_borderExtTaskCounter.donePtr() );
    }
  }

  if( m_decodeThreadPool->numThreads() == 0 && (
       std::any_of( picExtBarriers.cbegin(), picExtBarriers.cend(), []( const Barrier* b ) { return b->isBlocked(); } ) ||
       std::any_of( picBarriers   .cbegin(), picBarriers   .cend(), []( const Barrier* b ) { return b->isBlocked(); } ) ) )
  {
    m_decodeThreadPool->processTasksOnMainThread();
  }

  const int numColPerTask = std::max( std::min( widthInCtus, ( widthInCtus / std::max( m_numDecThreads * ( isIntra ? 2 : 1 ), 1 ) ) + ( isIntra ? 0 : 1 ) ), 1 );
  const int numTasksPerLine = widthInCtus / numColPerTask + !!( widthInCtus % numColPerTask );

#if ALLOW_MIDER_LF_DURING_PICEXT
  pcPic->refPicExtDepBarriers = std::move( picExtBarriers );
#endif
#if !RECO_WHILE_PARSE
  picBarriers.push_back( &cs.picture->parseDone );
#endif

  const TaskType ctuStartState = MIDER;
  const bool     noFilters     = pcPic->poc == debugNoFiltersPoc();
  const bool     doALF         = cs.sps->getUseALF() && !AdaptiveLoopFilter::getAlfSkipPic( cs ) && !noFilters;
  const bool     doSAO         = cs.sps->getUseSAO() && !noFilters;
  m_doALF                      = doALF;
  m_doSAO                      = doSAO;
  m_doDeblock                  = !noFilters;
#if VVDEC_VULKAN
  commonTaskParam.reset( cs, ctuStartState, numTasksPerLine, doALF && !m_gpuFilters, doSAO && !m_gpuFilters, m_doDeblock && !m_gpuFilters, m_gpuIntra,
                         m_gpuInter, m_gpuResidual );
  if( m_gpuRecon )
  {
    ( m_gpuFilters ? m_filterPicturesGpu : m_filterPicturesCpu )++;
    ( m_gpuIntra ? m_intraPicturesGpu : m_intraPicturesCpu )++;
    if( !isIntra )
    {
      ( m_gpuInter ? m_interPicturesGpu : m_interPicturesCpu )++;
    }
    ( m_gpuResidual ? m_residualPicturesGpu : m_residualPicturesCpu )++;
  }
  if( m_gpuFilters )
  {
    m_gpuEdges.resize( pcv->sizeInCtus );
  }
  if( m_gpuResidual )
  {
    m_gpuTrafoJobs.resize( pcv->sizeInCtus );
    for( auto& jobs: m_gpuTrafoJobs )
    {
      jobs.clear();
    }
  }
  if( m_gpuInter )
  {
    m_refTable.clear();
    for( const Slice* slice: pcPic->slices )
    {
      for( int list = 0; list < NUM_REF_PIC_LIST_01; list++ )
      {
        for( int refIdx = 0; refIdx < slice->getNumRefIdx( RefPicList( list ) ); refIdx++ )
        {
          m_refTable.add( slice->getRefPic( RefPicList( list ), refIdx ) );
        }
      }
    }
    CHECK( m_refTable.pictures().size() > 255, "too many reference pictures" );
    m_gpuInterJobs.resize( pcv->sizeInCtus );
    for( auto& jobs: m_gpuInterJobs )
    {
      jobs.clear();
    }
    // DMVR's refinements: the GPU's, in memory it writes
    pcPic->cs->m_dmvrMvCache = m_gpuRecon->dmvrMvCache( m_dmvrMvCacheSize );
  }
  if( m_gpuIntra )
  {
    m_waveMap.reset( pcv->lumaWidth, pcv->lumaHeight, getChannelTypeScaleX( CH_C, pcv->chrFormat ), getChannelTypeScaleY( CH_C, pcv->chrFormat ),
                     isChromaEnabled( pcv->chrFormat ) );
    m_gpuIntraJobs.resize( pcv->sizeInCtus );
    for( auto& jobs: m_gpuIntraJobs )
    {
      jobs.clear();
    }
  }
#else
  commonTaskParam.reset( cs, ctuStartState, numTasksPerLine, doALF, doSAO, m_doDeblock, false, false, false );
#endif

  tasksFinishMotion = std::vector<LineTaskParam>( heightInCtus, LineTaskParam{ commonTaskParam, -1 } );
  tasksCtu          = std::vector<CtuTaskParam >( heightInCtus * numTasksPerLine, CtuTaskParam{ commonTaskParam, -1, -1, {} } );

  pcPic->reconDone.lock();
#if VVDEC_VULKAN
  // the top half of the picture's rows of CTUs: their residuals and inter
  // blocks go to the GPU as soon as their CTU tasks are done (the rows finish
  // in order, the top half's well before the last row), while the CPU makes
  // the bottom half's jobs (VideoQ 1080p +3.5 to +5.9%, 4K 0 to +1%; 3 to 5
  // bands no faster: more submissions)
  m_gpuBandRows = m_gpuFilters && m_gpuResidual && m_gpuIntra && heightInCtus >= 4 ? heightInCtus / 2 : 0;
  m_gpuBandDone.store( false, std::memory_order_relaxed );
#endif

#if 0
  // schedule in raster scan order
  for( int line = 0; line < heightInCtus; ++line )
  {
    for( int col = 0; col < widthInCtus;  ++col )
    {
#else
  // schedule in zig-zag scan order
  for( int i = 0; i < numTasksPerLine + heightInCtus; ++i )
  {
    int line = 0;
    for( int col = i; col >= 0; --col, ++line )
    {
#endif
      if( line < heightInCtus && col < numTasksPerLine )
      {
        CBarrierVec ctuBarriers = picBarriers;
        const int   ctuStart    = col * numColPerTask;
        const int   ctuEnd      = std::min( ctuStart + numColPerTask, widthInCtus );

#if RECO_WHILE_PARSE
        if( pcPic->parseDone.isBlocked() )
        {
          // wait for the last CTU in the current line to be parsed
          ctuBarriers.push_back( &pcPic->ctuParsedBarrier[( line + 1 ) * widthInCtus - 1] );
        }

#endif
        CtuTaskParam* param    = &tasksCtu[line * numTasksPerLine + col];
        param->taskLine        = line;
        param->taskCol         = col;
        param->ctuEnd          = ctuEnd;
        param->ctuStart        = ctuStart;
        param->numColPerTask   = numColPerTask;
        param->numTasksPerLine = numTasksPerLine;

        m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string(pcPic->poc) + " ctuTask:" + std::to_string( col ) + "," + std::to_string( line ) )
                                            ctuTask<false>,
                                            param,
                                            &pcPic->m_ctuTaskCounter,
                                            nullptr,
                                            std::move( ctuBarriers ),
                                            ctuTask<true> );
      }
    }
  }

#if VVDEC_VULKAN
  if( m_gpuFilters )
  {
    // with the GPU, three tasks. The first queues the picture's work on the
    // GPU, and when DMVR refines motion there, waits for that part of the work,
    // which stores the refined motion for TMVP; then (gpuRefReady) the pictures
    // whose inter prediction the GPU makes can reference this one, their work
    // queued after its. The second waits for the rest of the work on the
    // compute queue: then (gpuComputeDone) the instance may take the next
    // picture while this one downloads (releasePrevDecompressedPic: VideoQ
    // 1080p +1.4%). The third waits for the download.
    static auto queueGpuTask = []( int tid, void* task_param )
    {
      FinishPicTaskParam* param = static_cast<FinishPicTaskParam*>( task_param );
      CodingStructure&    cs    = *param->pic->cs;
      DecLibRecon*        dec   = param->decLib;

      gpu::PictureJobs jobs;
      jobs.trafo = dec->m_gpuResidual ? &dec->m_gpuTrafoJobs : nullptr;
      jobs.inter = dec->m_gpuInter ? &dec->m_gpuInterJobs : nullptr;
      jobs.intra = dec->m_gpuIntra ? &dec->m_gpuIntraJobs : nullptr;
      jobs.refs  = &dec->m_refTable;
      jobs.edges = &dec->m_gpuEdges;
      jobs.bandCtus = size_t( dec->m_gpuBandRows ) * cs.pcv->widthInCtus;
      // with DMVR, the GPU stores the refined motion for TMVP in the next
      // pictures in the first part (the CTU tasks stored the rest)
      const bool refine = dec->m_gpuInter && cs.picture->stillReferenced;
      param->download   = dec->m_gpuRecon->startPicture( cs, jobs, dec->m_doDeblock, dec->m_doSAO ? &dec->m_cSAO : nullptr,
                                                         dec->m_doALF ? &dec->m_cALF : nullptr, cs.picture->m_bufs[PIC_RECONSTRUCTION], refine );
      cs.picture->gpuCurrent = true;   // for the work queued after this
      return true;
    };
    static auto computeDoneTask = []( int tid, void* task_param )
    {
      FinishPicTaskParam* param = static_cast<FinishPicTaskParam*>( task_param );
      param->decLib->m_gpuRecon->endCompute();
      return true;
    };
    static auto finishGpuTask = []( int tid, void* task_param )
    {
      FinishPicTaskParam* param = static_cast<FinishPicTaskParam*>( task_param );
      param->decLib->m_gpuRecon->endDownload( param->download );
      param->pic->cs->deallocTempInternals();

#ifdef TRACE_ENABLE_ITT
      // mark end of frame
      __itt_frame_end_v3( param->pic->m_itt_decLibInst, nullptr );
#endif
      param->pic->stopProcessingTimer();

      param->pic->progress = Picture::reconstructed;
      return true;
    };

    // the parameters of the filters on the GPU, from the parsed picture, while
    // its CTU tasks run: not on the way from them to the GPU
    static auto filterParamsTask = []( int tid, void* task_param )
    {
      FinishPicTaskParam* param = static_cast<FinishPicTaskParam*>( task_param );
      DecLibRecon*        dec   = param->decLib;
      dec->m_gpuRecon->prepareFilters( *param->pic->cs, dec->m_doDeblock, dec->m_doSAO ? &dec->m_cSAO : nullptr, dec->m_doALF ? &dec->m_cALF : nullptr );
      return true;
    };

    finishParam = FinishPicTaskParam( this, pcPic );
    m_gpuFiltersPrepared.clearException();
    m_gpuFiltersPrepared.lock();
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string( pcPic->poc ) + " filterParamsTask" )
                                        filterParamsTask,
                                        &finishParam,
                                        &pcPic->m_divTasksCounter,
                                        &m_gpuFiltersPrepared,
                                        { &pcPic->parseDone } );
    CBarrierVec queueBarriers{ pcPic->m_ctuTaskCounter.donePtr(), &m_gpuFiltersPrepared };
    if( m_gpuBandRows > 0 )
    {
      // the top band's jobs to the GPU: a task that waits for the band's CTU
      // tasks (its ready check: no barrier, so that a CTU task's failure fails
      // it too rather than leaving it waiting)
      static auto bandReady = []( int tid, void* task_param )
      {
        FinishPicTaskParam* param = static_cast<FinishPicTaskParam*>( task_param );
        if( param->pic->m_ctuTaskCounter.hasException() )
        {
          std::rethrow_exception( param->pic->m_ctuTaskCounter.getException() );
        }
        return param->decLib->m_gpuBandDone.load( std::memory_order_acquire );
      };
      static auto bandTask = []( int tid, void* task_param )
      {
        FinishPicTaskParam* param = static_cast<FinishPicTaskParam*>( task_param );
        CodingStructure&    cs    = *param->pic->cs;
        DecLibRecon*        dec   = param->decLib;
        gpu::PictureJobs    jobs;
        jobs.trafo = &dec->m_gpuTrafoJobs;
        jobs.inter = dec->m_gpuInter ? &dec->m_gpuInterJobs : nullptr;
        jobs.refs  = &dec->m_refTable;
        dec->m_gpuRecon->startBand( cs, jobs, size_t( dec->m_gpuBandRows ) * cs.pcv->widthInCtus, cs.picture->m_bufs[PIC_RECONSTRUCTION] );
        return true;
      };
      m_gpuBandQueued.clearException();
      m_gpuBandQueued.lock();
      m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string( pcPic->poc ) + " bandTask" )
                                          bandTask,
                                          &finishParam,
                                          &pcPic->m_divTasksCounter,
                                          &m_gpuBandQueued,
                                          {},
                                          bandReady );
      queueBarriers.push_back( &m_gpuBandQueued );
    }
    pcPic->gpuRefReady.lock();
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string( pcPic->poc ) + " queueGpuTask" )
                                        queueGpuTask,
                                        &finishParam,
                                        &pcPic->m_divTasksCounter,
                                        &pcPic->gpuRefReady,
                                        std::move( queueBarriers ) );
    pcPic->gpuComputeDone.clearException();
    pcPic->gpuComputeDone.lock();
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string( pcPic->poc ) + " computeDoneTask" )
                                        computeDoneTask,
                                        &finishParam,
                                        &pcPic->m_divTasksCounter,
                                        &pcPic->gpuComputeDone,
                                        { &pcPic->gpuRefReady } );
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string( pcPic->poc ) + " finishPicTask" )
                                        finishGpuTask,
                                        &finishParam,
                                        &pcPic->m_divTasksCounter,
                                        &pcPic->reconDone,
                                        { &pcPic->gpuComputeDone } );
    // from now on, the next pictures may reference this one early
    pcPic->gpuEarlyRef     = true;
    m_currDecompReleasable = true;
  }
  else
#endif
  {
    static auto finishReconTask = []( int tid, void* task_param )
    {
      FinishPicTaskParam* param = static_cast<FinishPicTaskParam*>( task_param );
      CodingStructure& cs = *param->pic->cs;

      if( param->decLib->m_doALF )
      {
        param->decLib->swapBufs( cs );
      }

      cs.deallocTempInternals();

#ifdef TRACE_ENABLE_ITT
      // mark end of frame
      __itt_frame_end_v3( param->pic->m_itt_decLibInst, nullptr );
#endif
      param->pic->stopProcessingTimer();

      param->pic->progress = Picture::reconstructed;
      return true;
    };

    finishParam = FinishPicTaskParam( this, pcPic );
    m_decodeThreadPool->addBarrierTask( TP_TASK_NAME_ARG( "POC:" + std::to_string( pcPic->poc ) + " finishPicTask" )
                                        finishReconTask,
                                        &finishParam,
                                        &pcPic->m_divTasksCounter,
                                        &pcPic->reconDone,
                                        { pcPic->m_ctuTaskCounter.donePtr() } );
  }

  if( m_decodeThreadPool->numThreads() == 0 )
  {
  }
  else
  {
    ITT_TASKEND( itt_domain_dec, itt_handle_schedTasks );
  }
}

Picture* DecLibRecon::waitForPrevDecompressedPic()
{
  if( !m_currDecompPic )
    return nullptr;

  ITT_TASKSTART( itt_domain_dec, itt_handle_waitTasks );
  try
  {
    if( m_decodeThreadPool->numThreads() == 0 )
    {
      m_decodeThreadPool->processTasksOnMainThread();
      CHECK_FATAL( m_currDecompPic->reconDone.isBlocked(),
                   "can't make progress reconstructing POC " << m_currDecompPic->poc << ". some dependency has not been finished." );
    }

    const Slice*   lastSlice           = m_currDecompPic->slices.back();
    const unsigned lastSliceLastCtuIdx = lastSlice->getCtuAddrInSlice( lastSlice->getNumCtuInSlice() - 1 );
    CHECK( lastSliceLastCtuIdx != m_currDecompPic->cs->pcv->sizeInCtus - 1, "Picture incomplete. A slice was probably lost." );

    m_currDecompPic->reconDone.wait();
  }
  catch( ... )
  {
    m_currDecompPic->error = true;
    m_currDecompPic->reconDone.setException( std::current_exception() );
  }

  // also check error flag, which can have been set earlier (e.g., when trying to use the picture as reference)
  if( m_currDecompPic->error || m_currDecompPic->reconDone.hasException() )
  {
    // ensure all tasks are cleared from declibRecon
    cleanupOnException();
  }

  ITT_TASKEND( itt_domain_dec, itt_handle_waitTasks );

  return std::exchange( m_currDecompPic, nullptr );
}

Picture* DecLibRecon::releasePrevDecompressedPic()
{
#if VVDEC_VULKAN
  // the picture's work on the GPU done but for its download: the instance is
  // free (what the next picture's work takes from it, the picture's work
  // there no longer needs, the download's commands aside: GpuRecon)
  if( m_currDecompPic && m_currDecompReleasable && m_decodeThreadPool->numThreads() > 0 )
  {
    bool released = false;
    try
    {
      const Slice*   lastSlice           = m_currDecompPic->slices.back();
      const unsigned lastSliceLastCtuIdx = lastSlice->getCtuAddrInSlice( lastSlice->getNumCtuInSlice() - 1 );
      CHECK( lastSliceLastCtuIdx != m_currDecompPic->cs->pcv->sizeInCtus - 1, "Picture incomplete. A slice was probably lost." );

      m_currDecompPic->gpuComputeDone.wait();
      released = !m_currDecompPic->error;
    }
    catch( ... )
    {
      // as waitForPrevDecompressedPic handles it
    }
    if( released )
    {
      return std::exchange( m_currDecompPic, nullptr );
    }
  }
#endif
  return waitForPrevDecompressedPic();
}

void DecLibRecon::waitForReleasedPic( Picture* pic )
{
  try
  {
    pic->reconDone.wait();
  }
  catch( ... )
  {
    pic->error = true;
    pic->reconDone.setException( std::current_exception() );
  }
  if( pic->error || pic->reconDone.hasException() )
  {
    // its tasks cleared from the thread pool (not those of the instance's next
    // picture: cleanupOnException's)
    pic->waitForAllTasks();
  }
}

void DecLibRecon::cleanupOnException()
{
  if( m_decodeThreadPool->numThreads() == 0 )
  {
    CHECK_FATAL( m_decodeThreadPool->processTasksOnMainThread() != true, "can't make progress: stuck tasks." );
  }

  // there was an exception anywhere in m_currDecompPic
  // => we need to wait for all tasks to be cleared from the thread pool
  m_currDecompPic->waitForAllTasks();

  commonTaskParam.ctuStates.clear();
}

template<bool onlyCheckReadyState>
bool DecLibRecon::ctuTask( int tid, void* task_param )
{
  CtuTaskParam* param = static_cast<CtuTaskParam*>( task_param );

  const int       taskCol      = param->taskCol;
  const int       line         = param->taskLine;
  const int       col          = taskCol;

  auto&           cs           = *param->common.cs;
  auto&           decLib       = param->common.decLib;
  const int       tasksPerLine = param->numTasksPerLine;
  const int       heightInCtus = cs.pcv->heightInCtus;

  CtuState&       thisCtuState =  param->common.ctuStates[line * tasksPerLine + taskCol];
  const CtuState* thisLine     = &param->common.ctuStates[line * tasksPerLine];
  const CtuState* lineAbove    = thisLine - tasksPerLine;
  const CtuState* lineBelow    = thisLine + tasksPerLine;

  const int       ctuStart     = param->ctuStart;
  const int       ctuEnd       = param->ctuEnd;

  if( cs.picture->m_ctuTaskCounter.hasException() )
  {
    std::rethrow_exception( cs.picture->m_ctuTaskCounter.getException() );
  }

  switch( thisCtuState.load() )
  {
    // all case statements fall through to continue with next task, unless they return false due to unsatisfied preconditions

  case MIDER:
  {
    if( col > 0 && thisLine[col - 1] <= MIDER_cont )
      return false;
    if( line > 0 )
    {
      if( col + 1 < tasksPerLine )
      {
        if( lineAbove[col + 1] <= MIDER )
          return false;
      }
      else
      {
        if( lineAbove[col] <= MIDER_cont )
          return false;
      }
    }
    if( onlyCheckReadyState )
      return true;

    ITT_TASKSTART( itt_domain_dec, itt_handle_mider );

    for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
    {
      const int ctuRsAddr = ctu + line * cs.pcv->widthInCtus;
      CtuData& ctuData    = cs.getCtuData( ctuRsAddr );
      ctuData.motion      = &decLib.m_motionInfo[cs.pcv->num4x4CtuBlks * ctuRsAddr];

      if( !ctuData.slice->isIntra() || cs.sps->getIBCFlag() )
      {
        const UnitArea ctuArea = getCtuArea( cs, ctu, line, true );
        decLib.m_pcThreadResource[tid]->m_cCuDecoder.TaskDeriveCtuMotionInfo( cs, ctuRsAddr, ctuArea, param->common.perLineMiHist[line] );
      }
      else
      {
        memset( NO_WARNING_class_memaccess( ctuData.motion ), MI_NOT_VALID, sizeof( MotionInfo ) * cs.pcv->num4x4CtuBlks );
      }

      thisCtuState = MIDER_cont;
    }

    thisCtuState = LF_INIT;

    ITT_TASKEND( itt_domain_dec, itt_handle_mider );
  }

  case LF_INIT:
  {
    if( onlyCheckReadyState )
      return true;

    ITT_TASKSTART( itt_domain_dec, itt_handle_lfcl );

    for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
    {
      const int ctuRsAddr = ctu + line * cs.pcv->widthInCtus;
      CtuData& ctuData    = cs.getCtuData( ctuRsAddr );
      ctuData.lfParam[0]  = &decLib.m_loopFilterParam[cs.pcv->num4x4CtuBlks * ( 2 * ctuRsAddr + 0 )];
      ctuData.lfParam[1]  = &decLib.m_loopFilterParam[cs.pcv->num4x4CtuBlks * ( 2 * ctuRsAddr + 1 )];
      memset( ctuData.lfParam[0], 0, sizeof( LoopFilterParam ) * 2 * cs.pcv->num4x4CtuBlks );

      decLib.m_cLoopFilter.calcFilterStrengthsCTU( cs, ctuRsAddr );
#if VVDEC_VULKAN
      if( decLib.m_gpuFilters )
      {
        gpu::GpuRecon::listEdges( cs, ctuRsAddr, decLib.m_gpuEdges[ctuRsAddr] );
      }
#endif
    }

    thisCtuState = INTER;

    ITT_TASKEND( itt_domain_dec, itt_handle_lfcl );
  }

  case INTER:
  {
    if( std::all_of( cs.picture->slices.begin(), cs.picture->slices.end(), []( const Slice* pcSlice ) { return pcSlice->isIntra(); } ) )
    {
      // not really necessary, but only for optimizing the wave-fronts
      if( col > 1 && thisLine[col - 2] <= INTER )
        return false;
      if( line > 0 && lineAbove[col] <= INTER )
        return false;
    }

    if( std::any_of( cs.picture->refPicExtDepBarriers.cbegin(), cs.picture->refPicExtDepBarriers.cend(), []( const Barrier* b ) { return b->isBlocked(); } ) )
    {
      return false;
    }

    if( onlyCheckReadyState )
      return true;

    ITT_TASKSTART( itt_domain_dec, itt_handle_inter );

    for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
    {
      const int ctuRsAddr    = ctu + line * cs.pcv->widthInCtus;
      const UnitArea ctuArea = getCtuArea( cs, ctu, line, true );
      const CtuData& ctuData = cs.getCtuData( ctuRsAddr );

#if VVDEC_VULKAN
      if( !param->common.gpuResidual )   // else its jobs (TaskGpuCtu)
#endif
      decLib.m_pcThreadResource[tid]->m_cCuDecoder.TaskTrafoCtu( cs, ctuRsAddr, ctuArea );

#if VVDEC_VULKAN
      // with the GPU, its inter jobs (TaskGpuCtu); the motion for TMVP here, but
      // that of the CUs DMVR refines once the GPU has refined it (finishReconTask)
      if( !ctuData.slice->isIntra() && param->common.gpuInter )
      {
        if( cs.picture->stillReferenced )
        {
          decLib.m_pcThreadResource[tid]->m_cCuDecoder.TaskColMotion( cs, ctuRsAddr );
        }
      }
      else
#endif
      if( !ctuData.slice->isIntra() )
      {
        decLib.m_pcThreadResource[tid]->m_cCuDecoder.TaskInterCtu( cs, ctuRsAddr, ctuArea );

        if( cs.picture->stillReferenced )
        {
          decLib.m_pcThreadResource[tid]->m_cCuDecoder.TaskFinishMotionInfo( cs, ctuRsAddr, ctu, line );
        }
      }
    }

    thisCtuState = INTRA;

    ITT_TASKEND( itt_domain_dec, itt_handle_inter );
  }

  case INTRA:
  {
    if( col > 0 && thisLine[col - 1] <= INTRA_cont )
      return false;

    if( line > 0 )
    {
      if( col + 1 < tasksPerLine )
      {
        if( lineAbove[col + 1] <= INTRA )
          return false;
      }
      else
      {
        if( lineAbove[col] <= INTRA_cont )
          return false;
      }
    }
    if( onlyCheckReadyState )
      return true;

    ITT_TASKSTART( itt_domain_dec, itt_handle_intra );

    for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
    {
      const int ctuRsAddr    = ctu + line * cs.pcv->widthInCtus;
      const UnitArea ctuArea = getCtuArea( cs, ctu, line, true );
#if VVDEC_VULKAN
      if( param->common.gpuIntra )
      {
        gpu::CtuJobs jobs;
        jobs.trafo = param->common.gpuResidual ? &decLib.m_gpuTrafoJobs[ctuRsAddr] : nullptr;
        jobs.inter = param->common.gpuInter ? &decLib.m_gpuInterJobs[ctuRsAddr] : nullptr;
        jobs.intra = &decLib.m_gpuIntraJobs[ctuRsAddr];
        jobs.waves = &decLib.m_waveMap;
        jobs.refs  = &decLib.m_refTable;
        decLib.m_pcThreadResource[tid]->m_cCuDecoder.TaskGpuCtu( cs, ctuRsAddr, ctuArea, jobs );
      }
      else
#endif
      decLib.m_pcThreadResource[tid]->m_cCuDecoder.TaskCriticalIntraKernel( cs, ctuRsAddr, ctuArea );

      thisCtuState = INTRA_cont;
    }

#if VVDEC_VULKAN
    // the GPU reconstructs the picture from these jobs on (intra there: no
    // reshaper, the filters there): the later stages have nothing to do, and
    // their waits for the neighbours (RSP's for the line below, PRESAO's for
    // whole lines) would only hold the jobs' recording back
    if( param->common.gpuIntra )
    {
      if( line == decLib.m_gpuBandRows - 1 && col == tasksPerLine - 1 )
      {
        // the band's last task: the others of its line are done (each waited
        // for the one before), and the lines above (this one waited for the
        // last task of the line above)
        decLib.m_gpuBandDone.store( true, std::memory_order_release );
      }
      thisCtuState = DONE;
      ITT_TASKEND( itt_domain_dec, itt_handle_intra );
      return true;
    }
#endif
    thisCtuState = RSP;

    ITT_TASKEND( itt_domain_dec, itt_handle_intra );
  }

  case RSP:
  {
    // RIRZIIIII
    // IIIIIXXXX
    //
    // - Z can be reshaped when it is no more an intra prediction source for X in the next line


    if     ( line + 1 < heightInCtus && col + 1 < tasksPerLine && lineBelow[col + 1] < INTRA_cont )
      return false;
    else if( line + 1 < heightInCtus &&                           lineBelow[col]     < RSP )
      return false;
    else if(                            col + 1 < tasksPerLine && thisLine [col + 1] < INTRA_cont ) // need this for the last line
      return false;

    if( onlyCheckReadyState )
      return true;

    ITT_TASKSTART( itt_domain_dec, itt_handle_rsp );

    for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
    {
      decLib.m_pcThreadResource[tid]->m_cReshaper.rspCtuBcw( cs, ctu, line );
    }

    ITT_TASKEND( itt_domain_dec, itt_handle_rsp );

    thisCtuState = LF_V;
  }

  case LF_V:
  {
    if( col > 0 && thisLine[col - 1] < LF_V )
      return false;
    if( onlyCheckReadyState )
      return true;

    ITT_TASKSTART( itt_domain_dec, itt_handle_lfl );

    for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
    {
      if( param->common.doDeblock )
      {
        decLib.m_cLoopFilter.loopFilterCTU( cs, MAX_NUM_CHANNEL_TYPE, ctu, line, EDGE_VER );
      }

      thisCtuState = LF_V_cont;
    }

    thisCtuState = LF_H;

    ITT_TASKEND( itt_domain_dec, itt_handle_lfl );
  }

  case LF_H:
  {
    if( line > 0 && lineAbove[col] < LF_H )
      return false;

    if( line > 0 && col + 1 < tasksPerLine && lineAbove[col + 1] < LF_V_cont )
      return false;

    if(             col + 1 < tasksPerLine && thisLine[col + 1] < LF_V_cont )
      return false;

    if( onlyCheckReadyState )
      return true;

    ITT_TASKSTART( itt_domain_dec, itt_handle_lfl );

    for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
    {
      if( param->common.doDeblock )
      {
        decLib.m_cLoopFilter.loopFilterCTU( cs, MAX_NUM_CHANNEL_TYPE, ctu, line, EDGE_HOR );
      }
    }

    thisCtuState = PRESAO;

    ITT_TASKEND( itt_domain_dec, itt_handle_lfl );
  }

  case PRESAO:
  {
    // only last CTU processes full line
    if( col == tasksPerLine - 1 )
    {
      if( line > 0 && lineAbove[col] <= PRESAO )
        return false;

      for( int c = 0; c < tasksPerLine; ++c )
      {
        if( thisLine[c] < PRESAO )
          return false;

        if( line + 1 < heightInCtus && lineBelow[c] < PRESAO )
          return false;
      }
      if( onlyCheckReadyState )
        return true;

      ITT_TASKSTART( itt_domain_dec, itt_handle_presao );

      if( param->common.doSAO )
      {
        decLib.m_cSAO.SAOPrepareCTULine( cs, getLineArea( cs, line, true ) );
      }

      ITT_TASKEND( itt_domain_dec, itt_handle_presao );
    }
    else if( thisLine[tasksPerLine - 1] <= PRESAO )   // wait for last CTU to finish PRESAO
    {
      return false;
    }
    if( onlyCheckReadyState )
      return true;

    thisCtuState = SAO;
  }

  case SAO:
  {
    if( onlyCheckReadyState )
      return true;

    // only last CTU processes full line
    if( param->common.doSAO )
    {
      ITT_TASKSTART( itt_domain_dec, itt_handle_sao );

      for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
      {
        const UnitArea  ctuArea = getCtuArea( cs, ctu, line, true );
        decLib.m_cSAO.SAOProcessCTU( cs, ctuArea );
      }

      ITT_TASKEND( itt_domain_dec, itt_handle_sao );
    }
    if( param->common.doALF )
    {
      ITT_TASKSTART( itt_domain_dec, itt_handle_alf );

      for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
      {
        AdaptiveLoopFilter::prepareCTU( cs, ctu, line );

        thisCtuState = SAO_cont;
      }

      ITT_TASKEND( itt_domain_dec, itt_handle_alf );
    }

    thisCtuState = ALF;
  }

  case ALF:
  {
    if( param->common.doALF )
    {
      const bool a = line > 0;
      const bool b = line + 1 < heightInCtus;
      const bool c = col > 0;
      const bool d = col + 1 < tasksPerLine;

      if( a )
      {
        if( c && lineAbove[col - 1] < ALF ) return false;
        if(      lineAbove[col    ] < ALF ) return false;
        if( d && lineAbove[col + 1] < SAO_cont ) return false;
      }

      if( b )
      {
        if( c && lineBelow[col - 1] < ALF ) return false;
        if(      lineBelow[col    ] < ALF ) return false;
        if( d && lineBelow[col + 1] < SAO_cont ) return false;
      }

      if( c && thisLine[col - 1] < ALF ) return false;
      if( d && thisLine[col + 1] < SAO_cont ) return false;

      if( onlyCheckReadyState )
        return true;

      ITT_TASKSTART( itt_domain_dec, itt_handle_alf );
      for( int ctu = ctuStart; ctu < ctuEnd; ctu++ )
      {
        decLib.m_cALF.processCTU( cs, ctu, line, tid );
      }
      ITT_TASKEND( itt_domain_dec, itt_handle_alf );
    }
    else if( onlyCheckReadyState )
      return true;

    thisCtuState = DONE;
  }

  default:
    CHECKD( thisCtuState != DONE, "Wrong CTU state" );
  }   // end switch

  return true;
}

}
