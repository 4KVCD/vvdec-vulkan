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

/** \file     DecLib.cpp
    \brief    decoder wrapper class
*/

#include "DecLib.h"

#include "CommonLib/dtrace_next.h"
#include "CommonLib/dtrace_buffer.h"
#include "CommonLib/TimeProfiler.h"

#include "CommonLib/arm/CommonDefARM.h"
#include "CommonLib/x86/CommonDefX86.h"

#include "NALread.h"


namespace vvdec
{

#ifdef TRACE_ENABLE_ITT
std::vector<__itt_domain*> itt_domain_decInst;
__itt_domain* itt_domain_dec           = __itt_domain_create( "Decode" );
__itt_domain* itt_domain_prs           = __itt_domain_create( "Parse" );
__itt_domain* itt_domain_oth           = __itt_domain_create( "Other" );

__itt_string_handle* itt_handle_alf    = __itt_string_handle_create( "ALF_CTU" );
__itt_string_handle* itt_handle_presao = __itt_string_handle_create( "PreSAO_Line" );
__itt_string_handle* itt_handle_sao    = __itt_string_handle_create( "SAO_CTU" );
__itt_string_handle* itt_handle_lfl    = __itt_string_handle_create( "LFL_CTU" );
__itt_string_handle* itt_handle_intra  = __itt_string_handle_create( "Intra_CTU" );
__itt_string_handle* itt_handle_inter  = __itt_string_handle_create( "Inter_CTU" );
__itt_string_handle* itt_handle_mider  = __itt_string_handle_create( "MI-Der_CTU" );
__itt_string_handle* itt_handle_lfcl   = __itt_string_handle_create( "Prep_ClearLF" );
__itt_string_handle* itt_handle_ext    = __itt_string_handle_create( "Prep_ExtBrdr" );
__itt_string_handle* itt_handle_dmvr   = __itt_string_handle_create( "MI-DMVR" );
__itt_string_handle* itt_handle_rsp    = __itt_string_handle_create( "Reshape_CTU" );

__itt_string_handle* itt_handle_parse  = __itt_string_handle_create( "Parse_Slice" );

__itt_string_handle* itt_handle_start  = __itt_string_handle_create( "Start_Pic" );
__itt_string_handle* itt_handle_done   = __itt_string_handle_create( "Pic_Done" );
__itt_string_handle* itt_handle_finish = __itt_string_handle_create( "Finish_Pic" );

__itt_string_handle* itt_handle_schedTasks = __itt_string_handle_create( "Scheduling_Tasks" );
__itt_string_handle* itt_handle_waitTasks  = __itt_string_handle_create( "Wait_for_Dec_Tasks" );


// create global domain for DecLib
__itt_domain* itt_domain_glb    = __itt_domain_create ( "Global" );
// create a global counter
__itt_counter itt_frame_counter = __itt_counter_create( "FrameNumber", "Global" );

#define ITT_TASKSTART( d, t ) __itt_task_begin( ( d ), __itt_null, __itt_null, ( t ) )
#define ITT_TASKEND( d, t )   __itt_task_end  ( ( d ) )
#else
#define ITT_TASKSTART( d, t )
#define ITT_TASKEND( d, t )
#endif

//! \ingroup DecoderLib
//! \{

DecLib::DecLib()
{
#ifdef TRACE_ENABLE_ITT
  itt_domain_dec->flags = 1;
  itt_domain_prs->flags = 1;
  itt_domain_glb->flags = 1;
  itt_domain_oth->flags = 1;
#endif
}

void DecLib::create( int numDecThreads, int parserFrameDelay, const UserAllocator& userAllocator, ErrHandlingFlags errHandlingFlags, bool useGpu )
{
  // run constructor again to ensure all variables, especially in DecLibParser have been reset
  this->~DecLib();
  new( this ) DecLib;

  if( numDecThreads < 0 )
  {
    numDecThreads = std::thread::hardware_concurrency();
  }

  const bool defaultParseDelay = parserFrameDelay < 0;
  if( defaultParseDelay )
  {
    CHECK_FATAL( numDecThreads < 0, "invalid number of threads" );
    parserFrameDelay = std::min<int>( ( numDecThreads * DEFAULT_PARSE_DELAY_FACTOR ) >> 4, DEFAULT_PARSE_DELAY_MAX );
  }
  m_parseFrameDelay = parserFrameDelay;

  UserAllocator* gpuAllocator = nullptr;
#if VVDEC_VULKAN
  if( useGpu )
  {
    if( userAllocator.enabled )
    {
      msg( WARNING, "GPU reconstruction off: it needs the decoder's own picture buffers, not an external allocator\n" );
    }
    else if( ( m_gpu = gpu::Context::acquire() ) == nullptr )
    {
      msg( WARNING, "GPU reconstruction off: no usable Vulkan device\n" );
    }
    else
    {
      m_gpuAllocator = gpu::hostPictureAllocator( m_gpu );
      gpuAllocator   = &m_gpuAllocator;
    }
  }
#else
  if( useGpu )
  {
    msg( WARNING, "GPU reconstruction off: this vvdec is built without Vulkan\n" );
  }
#endif

#if VVDEC_VULKAN
  // with the GPU, 12 pictures parsed ahead at most by default (the CPU's
  // default, 1.5 a thread, is 24 at 16 threads): as fast (VideoQ, 16 threads:
  // 1080p 642.4 -> 653.1 fps, 4K 177.7 -> 179.4), and each picture parsed
  // ahead holds its buffers (1080p 13 MB, 4K 44 MB): peak commit 1080p 793 ->
  // 633 MB, 4K 2546 -> 2014 MB. 8 saves 60 MB more at 1080p but is slower
  // there (634 vs 657 fps).
  if( m_gpu && defaultParseDelay )
  {
    parserFrameDelay  = std::min( parserFrameDelay, 12 );
    m_parseFrameDelay = parserFrameDelay;
  }
#endif

#if VVDEC_VULKAN
  // with the GPU, the CPU waits for it a lot: its idle threads block at once
  // instead of spinning (which on an APU takes power from the GPU too)
  m_decodeThreadPool.reset( new ThreadPool( numDecThreads, "DecThread", m_gpu != nullptr ) );
#else
  m_decodeThreadPool.reset( new ThreadPool( numDecThreads, "DecThread" ) );
#endif

#if VVDEC_VULKAN
  // with the GPU, more pictures in reconstruction at a time: the GPU finishes
  // one while the CPU readies the next ones (VideoQ 4K: 2 -> 4 instances,
  // 107 -> 126 fps; 5 no better, 8 worse)
  if( m_gpu )
  {
    m_decLibRecon.resize( 4 );
  }
#endif
  bool upscalingEnabled = false;
  m_picListManager.create( m_parseFrameDelay, (int) m_decLibRecon.size(), userAllocator, gpuAllocator );
#if VVDEC_VULKAN
  if( gpuAllocator )
  {
    m_picListManager.setReleaseGpuCopy( gpu::releaseTwins );
  }
#endif
  m_decLibParser.create  ( m_decodeThreadPool.get(), m_parseFrameDelay, (int) m_decLibRecon.size(), numDecThreads, errHandlingFlags );

  int id=0;
  for( auto &dec: m_decLibRecon )
  {
#if VVDEC_VULKAN
    dec.create( m_decodeThreadPool.get(), id++, upscalingEnabled, m_gpu );
#else
    dec.create( m_decodeThreadPool.get(), id++, upscalingEnabled );
#endif
  }
#if VVDEC_VULKAN
  // VVDEC_GPU_PARSE=2: the GPU parses the pictures ahead of their reconstruction, on the parse queue
  const gpu::Options* options = m_gpu ? &m_gpu->options() : nullptr;
  if( options && options->parseDrives && options->frontEnd && !options->frontEndCheck && m_gpu->parseQueue() )
  {
    ThreadPool* pool  = m_decodeThreadPool.get();
    m_gpuParseService = std::make_unique<gpu::ParseService>( m_gpu, [pool]() { pool->wakeUp(); } );
    m_decLibParser.setGpuParseService( m_gpuParseService.get() );
    for( auto& dec: m_decLibRecon )
    {
      dec.setGpuParseService( m_gpuParseService.get() );
    }
  }
#endif

  std::stringstream cssCap;
  cssCap << "THREADS="     << numDecThreads << "; "
         << "PARSE_DELAY=" << parserFrameDelay << "; ";
#if VVDEC_VULKAN
  if( m_gpu )
  {
    cssCap << "GPU=" << m_gpu->deviceName() << "; ";
  }
#endif
#if ENABLE_SIMD_OPT
#  if defined( TARGET_SIMD_ARM )
  cssCap << "SIMD=" << read_arm_extension_name();
#  elif defined( TARGET_SIMD_X86 )
  cssCap << "SIMD=" << read_x86_extension_name();
#  else
  cssCap << "SIMD=SCALAR";
#  endif
#else
  cssCap << "SIMD=NONE";
#endif

  m_sDecoderCapabilities = cssCap.str();

  DTRACE_UPDATE( g_trace_ctx, std::make_pair( "final", 1 ) );
}

void DecLib::destroy()
{
#if VVDEC_VULKAN
  // the GPU's parse ahead stops (its last submission done) after the thread pool's threads, which hand it pictures,
  // and before the pool, which it wakes, and the pictures it writes into go
  const bool      parseAheadOn = m_gpuParseService != nullptr;
  gpu::ParseStats parseAhead;
  const auto      stopParseAhead = [&]() {
    if( m_gpuParseService )
    {
      parseAhead = m_gpuParseService->stats();
      m_gpuParseService.reset();
    }
  };
#endif
  if( m_decodeThreadPool )
  {
    m_decodeThreadPool->shutdown( true );
#if VVDEC_VULKAN
    stopParseAhead();
#endif
    m_decodeThreadPool.reset();
  }
#if VVDEC_VULKAN
  stopParseAhead();   // (without a thread pool)
#endif

  m_decLibParser.destroy();
#if VVDEC_VULKAN
  if( m_gpu )
  {
    int filtersGpu = 0, filtersCpu = 0, intraGpu = 0, intraCpu = 0, interGpu = 0, interCpu = 0, residualGpu = 0, residualCpu = 0;
    for( auto& dec: m_decLibRecon )
    {
      filtersGpu += dec.filterPicturesGpu();
      filtersCpu += dec.filterPicturesCpu();
      intraGpu   += dec.intraPicturesGpu();
      intraCpu   += dec.intraPicturesCpu();
      interGpu   += dec.interPicturesGpu();
      interCpu   += dec.interPicturesCpu();
      residualGpu += dec.residualPicturesGpu();
      residualCpu += dec.residualPicturesCpu();
    }
    msg( INFO,
         "GPU: residuals of %d pictures on the GPU, %d on the CPU; inter prediction of %d on the GPU, %d on the CPU; intra prediction of %d on the GPU, "
         "%d on the CPU; in-loop filters of %d on the GPU, %d on the CPU\n",
         residualGpu, residualCpu, interGpu, interCpu, intraGpu, intraCpu, filtersGpu, filtersCpu );
    gpu::ParseStats parse;
    bool            parseOn = false;
    for( auto& dec: m_decLibRecon )
    {
      if( const gpu::ParseStats* s = dec.parseStats() )
      {
        parseOn = true;
        parse.pictures += s->pictures;
        parse.errors += s->errors;
        parse.cusCompared += s->cusCompared;
        parse.cusDiffer += s->cusDiffer;
        parse.tusCompared += s->tusCompared;
        parse.tusDiffer += s->tusDiffer;
        parse.ctusCompared += s->ctusCompared;
        parse.ctusDiffer += s->ctusDiffer;
        parse.levelsCompared += s->levelsCompared;
        parse.levelsDiffer += s->levelsDiffer;
        parse.substreams += s->substreams;
        parse.runs += s->runs;
        parse.submissions += s->submissions;
        parse.capsCompared += s->capsCompared;
        parse.capsDiffer += s->capsDiffer;
        for( int i = 0; i < 16; i++ )
        {
          parse.prof[i] += s->prof[i];
        }
      }
    }
    if( parseAheadOn )
    {
      parse = parseAhead;
    }
    if( parseOn || parseAheadOn )
    {
      if( parse.prof[0] != 0 )
      {
        // the parts in the device clock's ticks (100 MHz on AMD: shown as ms), decodeBin's in the subgroup's cycles
        const auto ms = [&]( int k ) { return double( parse.prof[k] ) / 1e5; };
        msg( INFO,
             "GPU: parse's profile, summed over the substreams: dispatches %.1f ms (their starts %.1f ms), CTUs %.1f ms, of which CUs %.1f ms "
             "(residuals %.1f ms, of which levels written %.1f ms), records %.1f ms, SAO/ALF %.1f ms; decodeBin %llu cycles, decodeBinEP %llu; "
             "bins %llu context-coded, %llu bypass; CUs %llu, TUs %llu, coded blocks %llu, levels %llu" "\n",
             ms( 14 ), ms( 15 ), ms( 0 ), ms( 1 ), ms( 2 ), ms( 3 ), ms( 4 ), ms( 5 ), (unsigned long long) parse.prof[6],
             (unsigned long long) parse.prof[7], (unsigned long long) parse.prof[8], (unsigned long long) parse.prof[9],
             (unsigned long long) parse.prof[10], (unsigned long long) parse.prof[11], (unsigned long long) parse.prof[12],
             (unsigned long long) parse.prof[13] );
      }
    }
    if( parseOn )
    {
      msg( INFO,
           "GPU: parse (VVDEC_GPU_PARSE) of %lld pictures: CUs %lld compared with the CPU's, %lld differ; TUs %lld, %lld; CTUs (with SAO, ALF) %lld, "
           "%lld; levels %lld, %lld; substreams %lld in %lld runs, %lld submissions" "\n",
           (long long) parse.pictures, (long long) parse.cusCompared, (long long) parse.cusDiffer, (long long) parse.tusCompared,
           (long long) parse.tusDiffer, (long long) parse.ctusCompared, (long long) parse.ctusDiffer, (long long) parse.levelsCompared,
           (long long) parse.levelsDiffer, (long long) parse.substreams, (long long) parse.runs, (long long) parse.submissions );
      msg( INFO, "GPU: parse's job lists' room (VVDEC_GPU_PARSE): %lld pictures compared with the CPU's records' (gpu::frontCapacities), %lld differ\n",
           (long long) parse.capsCompared, (long long) parse.capsDiffer );
    }
    if( parseAheadOn )
    {
      msg( INFO,
           "GPU: parse ahead (VVDEC_GPU_PARSE=2) of %lld pictures: %lld submissions with %.1f pictures each on average, %lld pictures parsed again "
           "for room, %lld failed; substreams %lld in %lld runs; at most %lld CU and %lld TU records a picture\n",
           (long long) parse.pictures, (long long) parse.submissions,
           parse.submissions > 0 ? double( parse.parsingSum ) / double( parse.submissions ) : 0.0, (long long) parse.restarts,
           (long long) parse.errors, (long long) parse.substreams, (long long) parse.runs, (long long) parse.maxCus, (long long) parse.maxTus );
    }
    gpu::FrontEndStats check;
    bool               checkOn = false;
    for( auto& dec: m_decLibRecon )
    {
      if( const gpu::FrontEndStats* s = dec.frontEndStats() )
      {
        checkOn = true;
        check.pictures += s->pictures;
        check.compared += s->compared;
        check.mismatched += s->mismatched;
        check.lfpCompared += s->lfpCompared;
        check.lfpMismatched += s->lfpMismatched;
        check.lfpEdges += s->lfpEdges;
        for( int k = 0; k < 3; k++ )
        {
          check.jobsCompared[k] += s->jobsCompared[k];
          check.jobsMismatched[k] += s->jobsMismatched[k];
        }
        check.edgesCompared += s->edgesCompared;
        check.sortedCompared += s->sortedCompared;
        check.sortedMismatched += s->sortedMismatched;
        check.edgesMismatched += s->edgesMismatched;
      }
    }
    if( checkOn )
    {
      msg( INFO, "GPU: motion derivation (VVDEC_GPU_MIDER) of %lld pictures: %lld 4x4 blocks compared with the CPU's, %lld differ\n",
           (long long) check.pictures, (long long) check.compared, (long long) check.mismatched );
      msg( INFO, "GPU: deblocking parameters (VVDEC_GPU_MIDER): %lld compared with the CPU's (%lld with a boundary strength), %lld differ\n",
           (long long) check.lfpCompared, (long long) check.lfpEdges, (long long) check.lfpMismatched );
      msg( INFO, "GPU: jobs (VVDEC_GPU_MIDER): residuals' %lld compared with the CPU's, %lld differ; inter's %lld, %lld; intra's %lld, %lld\n",
           (long long) check.jobsCompared[0], (long long) check.jobsMismatched[0], (long long) check.jobsCompared[1],
           (long long) check.jobsMismatched[1], (long long) check.jobsCompared[2], (long long) check.jobsMismatched[2] );
      msg( INFO, "GPU: deblocking edges (VVDEC_GPU_MIDER): %lld compared with the CPU's, %lld differ\n", (long long) check.edgesCompared,
           (long long) check.edgesMismatched );
      msg( INFO, "GPU: intra's order (VVDEC_GPU_MIDER): %lld bands compared with the CPU's, %lld differ\n", (long long) check.sortedCompared,
           (long long) check.sortedMismatched );
    }
  }
#endif
  for( auto &dec: m_decLibRecon )
  {
    dec.destroy();
  }

  m_picListManager.deleteBuffers();

#if VVDEC_VULKAN
  // after the last picture buffer is gone
  if( m_gpu )
  {
    gpu::Context::release();
    m_gpu = nullptr;
  }
#endif
}

Picture* DecLib::decode( InputNALUnit& nalu )
{
  PROFILER_SCOPE_AND_STAGE( 1, g_timeProfiler, P_NALU_SLICE_PIC_HL );

  bool newPic = false;
  if( m_iMaxTemporalLayer < 0 || nalu.m_temporalId <= m_iMaxTemporalLayer )
  {
    newPic = m_decLibParser.parse( nalu );
  }

  if( newPic )
  {
    Picture* pcParsedPic = m_decLibParser.getNextDecodablePicture();

    while( pcParsedPic && ( pcParsedPic->error || pcParsedPic->wasLost || pcParsedPic->parseDone.hasException() ) )
    {
      CHECK_FATAL( pcParsedPic->progress >= Picture::reconstructing, "The error picture shouldn't be in reconstructing state yet." );

      sanitizeBrokenPicture( pcParsedPic );

      // try again to get a picture, that we can reconstruct now
      pcParsedPic = m_decLibParser.getNextDecodablePicture();
    }

    if( pcParsedPic )
    {
      reconPicture( pcParsedPic );
    }
  }

  if( newPic || nalu.m_nalUnitType == NAL_UNIT_EOS )
  {
    Picture* outPic = getNextOutputPic( false );
    if( outPic )
    {
      CHECK_WARN( outPic->progress < Picture::finished, "Picture should have been finished by now. Blocking and finishing..." );
      if( outPic->progress < Picture::finished )
      {
        blockAndFinishPictures( outPic );

        CHECK( outPic->progress < Picture::finished, "Picture still not finished. Something is really broken." );
      }

      m_checkMissingOutput = true;
    }

    // warn if we don't produce an output picture for every incoming picture
    CHECK_WARN( m_checkMissingOutput && !outPic, "missing output picture" ); // this CHECK_FATAL is not needed in flushPic(), because in flushPic() the nullptr signals the end of the bitstream
    return outPic;
  }

  return nullptr;
}

Picture* DecLib::flushPic()
{
  Picture* outPic = nullptr;
  try
  {
    m_decLibParser.failIncompleteParsePic();   // no more slices are coming for the last picture

    // at end of file, fill the decompression queue and decode pictures until the next output-picture is finished
    while( Picture* pcParsedPic = m_decLibParser.getNextDecodablePicture() )
    {
      if( pcParsedPic->error || pcParsedPic->wasLost || pcParsedPic->parseDone.hasException() )
      {
        CHECK_FATAL( pcParsedPic->progress >= Picture::reconstructing, "The error picture shouldn't be in reconstructing state yet." );

        sanitizeBrokenPicture( pcParsedPic );

        // try again to get a picture, that we can reconstruct now
        continue;
      }

      // reconPicture() blocks and finishes one picture on each call
      reconPicture( pcParsedPic );

      if( !outPic )
      {
        outPic = getNextOutputPic( false );
      }
      if( outPic && outPic->progress == Picture::finished )
      {
        return outPic;
      }
    }

    // no more pictures to finish decoding, but we might still have finished pictures in the recon queue
    if( !outPic )
    {
      outPic = getNextOutputPic( false );
    }
    if( outPic && outPic->progress == Picture::finished )
    {
      return outPic;
    }

    // if all pictures have been parsed, but not finished, iteratively wait for and finish next pictures
    blockAndFinishPictures( outPic );
    if( !outPic )
    {
      outPic = getNextOutputPic( false );
    }
    if( outPic && outPic->progress == Picture::finished )
    {
      return outPic;
    }

    CHECK( outPic, "we shouldn't be holding an output picture here" );
    // flush remaining pictures without considering num reorder pics
    outPic = getNextOutputPic( true );
    if( outPic )
    {
      CHECK( outPic->progress != Picture::finished, "all pictures should have been finished by now" );
      // outPic->referenced = false;
      return outPic;
    }
  }
  catch( ... )
  {
    m_picListManager.releasePicture( outPic );
    throw;
  }

  // At the very end reset parser state
  InputNALUnit eosNAL;
  eosNAL.m_nalUnitType = NAL_UNIT_EOS;
  m_decLibParser.parse( eosNAL );
  m_checkMissingOutput = false;

  return nullptr;
}

void DecLib::sanitizeBrokenPicture( Picture* pcParsedPic )
{
  std::exception_ptr parsing_exception = pcParsedPic->parseDone.hasException() ? pcParsedPic->parseDone.getException() : nullptr;
  if( parsing_exception )
  {   // the exception has not been thrown out of the library. Do that after preparing this picture for referencing
    pcParsedPic->error = true;
    pcParsedPic->waitForAllTasks();
    pcParsedPic->parseDone.clearException();
  }

  pcParsedPic->waitForAllTasks();

  if( pcParsedPic->progress < Picture::parsing )
  {
    // we don't know if all structures are there yet, so we init them
    pcParsedPic->ensureUsableAsRef();
  }
  pcParsedPic->fillGrey( m_decLibParser.getParameterSetManager().getFirstSPS() );

  // need to finish picture here, because it won't go through declibRecon
  finishPicture( pcParsedPic );

  // this exception has not been thrown outside (error must have happened in slice parsing task)
  if( parsing_exception )
  {
    CHECK_FATAL( pcParsedPic->exceptionThrownOut, "The exception shouldn't have been thrown out already." );
    pcParsedPic->exceptionThrownOut = true;
    std::rethrow_exception( parsing_exception );
  }
}

#if JVET_R0270
int DecLib::finishPicture( Picture* pcPic, MsgLevel msgl, bool associatedWithNewClvs )
#else
int DecLib::finishPicture( Picture* pcPic, MsgLevel msgl )
#endif
{
#ifdef TRACE_ENABLE_ITT
  // increment Framecounter
  __itt_counter_inc( itt_frame_counter );
#endif

  Slice*  pcSlice = pcPic->slices[0];
  if( pcPic->wasLost || pcPic->error || pcPic->reconDone.hasException() || pcPic->parseDone.hasException() )
  {
    msg( msgl, "POC %4d LId: %2d TId: %1d %s\n", pcPic->poc, pcPic->layerId, pcSlice->getTLayer(), pcPic->wasLost ? "LOST" : "ERROR" );
    pcPic->fillGrey( pcPic->cs->sps.get() );
    pcPic->progress = Picture::finished;

    // if the picture has an exception set (originating from thread-pool tasks), don't return here, but rethrow the exception
    try
    {
      pcPic->parseDone.checkAndRethrowException();
      pcPic->reconDone.checkAndRethrowException();
    }
    catch( ... )
    {
      pcPic->waitForAllTasks();

      // need to clear exception so we can use it as reference picture
      pcPic->reconDone.clearException();
      pcPic->reconDone.unlock();
      pcPic->error = true;
      if( !pcPic->exceptionThrownOut )
      {
        pcPic->exceptionThrownOut = true;
        throw;
      }
    }

    return pcPic->poc;
  }

  ITT_TASKSTART( itt_domain_oth, itt_handle_finish );

  char c = ( pcSlice->isIntra() ? 'I' : pcSlice->isInterP() ? 'P' : 'B' );
  if( !pcPic->isReferencePic )
  {
    c += 32;  // tolower
  }

  //-- For time output for each slice
  msg( msgl, "POC %4d LId: %2d TId: %1d ( %c-SLICE, QP%3d ) ", pcPic->poc, pcPic->layerId,
         pcSlice->getTLayer(),
         c,
         pcSlice->getSliceQp() );
  msg( msgl, "[DT %6.3f] ", pcPic->getProcessingTime() );

  for (int iRefList = 0; iRefList < 2; iRefList++)
  {
    msg( msgl, "[L%d ", iRefList);
    for (int iRefIndex = 0; iRefIndex < pcSlice->getNumRefIdx(RefPicList(iRefList)); iRefIndex++)
    {
#if RPR_OUTPUT_MSG
      const std::pair<int, int>& scaleRatio = pcSlice->getScalingRatio( RefPicList( iRefList ), iRefIndex );
      
      if( pcSlice->getPicHeader()->getEnableTMVPFlag() && pcSlice->getColFromL0Flag() == bool(1 - iRefList) && pcSlice->getColRefIdx() == iRefIndex )
      {
        if( scaleRatio.first != 1 << SCALE_RATIO_BITS || scaleRatio.second != 1 << SCALE_RATIO_BITS )
        {
          msg( msgl, "%dc(%1.2lfx, %1.2lfx) ", pcSlice->getRefPOC( RefPicList( iRefList ), iRefIndex ), double( scaleRatio.first ) / ( 1 << SCALE_RATIO_BITS ), double( scaleRatio.second ) / ( 1 << SCALE_RATIO_BITS ) );
        }
        else
        {
          msg( msgl, "%dc ", pcSlice->getRefPOC( RefPicList( iRefList ), iRefIndex ) );
        }
      }
      else
      {
        if( scaleRatio.first != 1 << SCALE_RATIO_BITS || scaleRatio.second != 1 << SCALE_RATIO_BITS )
        {
          msg( msgl, "%d(%1.2lfx, %1.2lfx) ", pcSlice->getRefPOC( RefPicList( iRefList ), iRefIndex ), double( scaleRatio.first ) / ( 1 << SCALE_RATIO_BITS ), double( scaleRatio.second ) / ( 1 << SCALE_RATIO_BITS ) );
        }
        else
        {
      msg( msgl, "%d ", pcSlice->getRefPOC(RefPicList(iRefList), iRefIndex));
    }
      }
#else
      msg( msgl, "%d ", pcSlice->getRefPOC(RefPicList(iRefList), iRefIndex));
#endif
    }
    msg( msgl, "] ");
  }

  msg( msgl, "\n");

//  pcPic->neededForOutput = (pcSlice->getPicHeader()->getPicOutputFlag() ? true : false);
#if JVET_R0270
  if (associatedWithNewClvs && pcPic->neededForOutput)
  {
    if (!pcSlice->getPPS()->getMixedNaluTypesInPicFlag() && pcSlice->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL)
    {
      pcPic->neededForOutput = false;
    }
    else if (pcSlice->getPPS()->getMixedNaluTypesInPicFlag())
    {
      bool isRaslPic = true;
      for (int i = 0; isRaslPic && i < pcPic->numSlices; i++)
      {
        if (!(pcSlice->getNalUnitType() == NAL_UNIT_CODED_SLICE_RASL || pcSlice->getNalUnitType() == NAL_UNIT_CODED_SLICE_RADL))
        {
          isRaslPic = false;
        }
      }
      if (isRaslPic)
      {
        pcPic->neededForOutput = false;
      }
    }
  }
#endif

  m_picListManager.markUnusedPicturesReusable();

  if( m_parseFrameDelay > 0 )
  {
    checkPictureHashSEI( pcPic );
  }

  ITT_TASKEND( itt_domain_oth, itt_handle_finish );

  pcPic->progress = Picture::finished;

  return pcSlice->getPOC();
}

void DecLib::checkPictureHashSEI( Picture* pcPic )
{
  if( !m_decodedPictureHashSEIEnabled )
  {
    return;
  }
  if( !pcPic->neededForOutput || pcPic->picCheckedDPH )
  {
    return;
  }

  CHECK( pcPic->progress < Picture::reconstructed, "picture not reconstructed" );

  seiMessages pictureHashes = SEI_internal::getSeisByType( pcPic->seiMessageList, VVDEC_DECODED_PICTURE_HASH );
  if( !pictureHashes.empty() )
  {
    if( pictureHashes.size() > 1 )
    {
      msg( WARNING, "Warning: Got multiple decoded picture hash SEI messages. Using first." );
    }

    const vvdecSEIDecodedPictureHash* hash = (vvdecSEIDecodedPictureHash*)pictureHashes.front()->payload;

    msg( INFO, "         " );
    const int hashErrors              = calcAndPrintHashStatus( pcPic->getRecoBuf(), hash, pcPic->cs->sps->getBitDepths(), INFO );
    m_numberOfChecksumErrorsDetected += hashErrors;
    pcPic->dphMismatch                = !!hashErrors;
    pcPic->picCheckedDPH              = true;
    msg( INFO, "\n" );
  }
  else
  {
    if( pcPic->subPictures.empty() )
    {
      msg( WARNING, "Warning: missing decoded picture hash SEI message.\n" );
      return;
    }

    seiMessages scalableNestingSeis = SEI_internal::getSeisByType( pcPic->seiMessageList, VVDEC_SCALABLE_NESTING );
    for( auto* seiIt: scalableNestingSeis )
    {
      CHECK( seiIt->payloadType != VVDEC_SCALABLE_NESTING, "expected nesting SEI" );

      const vvdecSEIScalableNesting* nestingSei = (vvdecSEIScalableNesting*) seiIt->payload;
      if( !nestingSei->snSubpicFlag )
      {
        continue;
      }

      for( int i = 0; i < nestingSei->snNumSEIs; ++i )
      {
        auto& nestedSei = nestingSei->nestedSEIs[i];
        CHECK( nestedSei == nullptr, "missing nested sei" );
        if( nestedSei && nestedSei->payloadType != VVDEC_DECODED_PICTURE_HASH )
          continue;

        const vvdecSEIDecodedPictureHash* hash = (vvdecSEIDecodedPictureHash*) nestedSei->payload;

        if( pcPic->subpicsCheckedDPH.empty() )
        {
          pcPic->subpicsCheckedDPH.resize( pcPic->subPictures.size(), false );
        }
        else
        {
          CHECK( pcPic->subpicsCheckedDPH.size() != pcPic->subPictures.size(), "Picture::subpicsCheckedDPH not properly initialized" );
        }

        for( int j = 0; j < nestingSei->snNumSubpics; ++j )
        {
          uint32_t subpicId = nestingSei->snSubpicId[j];

          for( auto& subPic: pcPic->subPictures )
          {
            if( subPic.getSubPicID() != subpicId )
              continue;

            auto subPicIdx = subPic.getSubPicIdx();
            if( pcPic->subpicsCheckedDPH[subPicIdx] )
              continue;

            const UnitArea area                 = UnitArea( pcPic->chromaFormat, subPic.getLumaArea() );
            const int      hashErrors           = calcAndPrintHashStatus( pcPic->cs->getRecoBuf( area ), hash, pcPic->cs->sps->getBitDepths(), INFO );
            m_numberOfChecksumErrorsDetected   += hashErrors;
            pcPic->dphMismatch                 |= !!hashErrors;
            pcPic->subpicsCheckedDPH[subPicIdx] = true;
            msg( INFO, "\n" );
          }
        }
      }
    }

    size_t checkedSubpicCount = std::count( pcPic->subpicsCheckedDPH.cbegin(), pcPic->subpicsCheckedDPH.cend(), true );
    pcPic->picCheckedDPH      = ( checkedSubpicCount == pcPic->subPictures.size() );   // mark when all subpics have been checked

    if( m_parseFrameDelay )
    {
      // this warning is only enabled, when running with parse delay enabled, because otherwise we don't know here if the last DPH Suffix-SEI has already been
      // parsed
      if( checkedSubpicCount != pcPic->subPictures.size() )
      {
        msg( WARNING, "Warning: missing decoded picture hash SEI message for SubPics (%u/%u).\n", checkedSubpicCount, pcPic->subPictures.size() );
      }
    }
  }
}

Picture* DecLib::getNextOutputPic( bool bFlush )
{
  if( m_picListManager.getFrontPic() == nullptr )
  {
    return nullptr;
  }

  const SPS* activeSPS      = m_picListManager.getFrontPic()->cs->sps.get();
  const int  maxNrSublayers = activeSPS->getMaxTLayers();

  int numReorderPicsHighestTid;
  int maxDecPicBufferingHighestTid;
  if( m_iMaxTemporalLayer == -1 || m_iMaxTemporalLayer >= maxNrSublayers )
  {
    numReorderPicsHighestTid     = activeSPS->getNumReorderPics( maxNrSublayers - 1 );
    maxDecPicBufferingHighestTid = activeSPS->getMaxDecPicBuffering( maxNrSublayers - 1 );
  }
  else
  {
    numReorderPicsHighestTid     = activeSPS->getNumReorderPics( m_iMaxTemporalLayer );
    maxDecPicBufferingHighestTid = activeSPS->getMaxDecPicBuffering( m_iMaxTemporalLayer );
  }

  return m_picListManager.getNextOutputPic( numReorderPicsHighestTid, maxDecPicBufferingHighestTid, bFlush );
}

void DecLib::reconPicture( Picture* pcPic )
{
  CHECK_FATAL( std::any_of( m_decLibRecon.begin(), m_decLibRecon.end(), [=]( auto& rec ) { return rec.getCurrPic() == pcPic; } ),
         "(Reused) Picture structure is still in progress in decLibRecon." );

  // the instances in turn: the first takes the picture and goes behind the
  // last of those in turn
  size_t inTurn = m_decLibRecon.size();
#if VVDEC_VULKAN
  // with the GPU and the CPU's front end, the pictures up to 1080p take the
  // first two only: as fast as with four (VideoQ 1080p 685 vs 675 fps; two
  // films' 1080p within 0.5%), with two pictures fewer in reconstruction and
  // two instances' buffers not allocated (VideoQ 1080p: peak commit 580 -> 477
  // MB). Larger pictures take all four (3840x1608 and 4K: 1.3% and 2.0% faster
  // than with two), and with the GPU's front end all pictures do: a picture's
  // front end waits for its colocated picture's first part, more pictures in
  // flight keep the GPU busy (VideoQ 1080p +9%, peak commit +140 MB).
  const bool gpuFrontEnd = m_gpu && m_gpu->options().frontEnd && !m_gpu->options().frontEndCheck;
  if( m_gpu && pcPic->lumaSize().area() <= 1920 * 1088 && !gpuFrontEnd )
  {
    inTurn = std::min<size_t>( inTurn, 2 );
  }
  if( inTurn < m_reconInTurn )
  {
    // the pictures of the instances leaving the turns finished now
    auto leaving = m_decLibRecon.begin();
    std::advance( leaving, inTurn );
    for( ; leaving != m_decLibRecon.end(); ++leaving )
    {
      if( Picture* donePic = leaving->waitForPrevDecompressedPic() )
      {
        finishPicture( donePic );
      }
    }
  }
  m_reconInTurn = inTurn;
#endif
  DecLibRecon* reconInstance = &m_decLibRecon.front();
  auto         behind        = m_decLibRecon.begin();
  std::advance( behind, inTurn );
  m_decLibRecon.splice( behind, m_decLibRecon, m_decLibRecon.begin() );

  // the instance's previous picture: with the GPU, possibly still downloading
  // while this one starts
  Picture* donePic = reconInstance->releasePrevDecompressedPic();
  try
  {
    reconInstance->decompressPicture( pcPic );
  }
  catch( ... )
  {
    pcPic->reconDone.setException( std::current_exception() );
    pcPic->error = true;
  }

  if( donePic )
  {
    DecLibRecon::waitForReleasedPic( donePic );
    finishPicture( donePic );
  }
}

void DecLib::blockAndFinishPictures( Picture* pcPic )
{
  // find Recon instance where current picture (if not null) is active and ensure the picture gets finished
  // otherwise all pictures get finished
  for( auto& recon: m_decLibRecon )
  {
    if( pcPic && recon.getCurrPic() != pcPic )
    {
      continue;
    }

    if( Picture* donePic = recon.waitForPrevDecompressedPic() )
    {
      finishPicture( donePic );
    }
  }
}

void DecLib::checkNalUnitConstraints( uint32_t naluType )
{
  const ConstraintInfo *cInfo = NULL;
  if (m_decLibParser.getParameterSetManager().getActiveSPS() != NULL && m_decLibParser.getParameterSetManager().getActiveSPS()->getProfileTierLevel() != NULL)
  {
    cInfo = m_decLibParser.getParameterSetManager().getActiveSPS()->getProfileTierLevel()->getConstraintInfo();
    if( cInfo != NULL )
    {
      xCheckNalUnitConstraintFlags( cInfo, naluType );
    }
  }
}

void DecLib::xCheckNalUnitConstraintFlags( const ConstraintInfo *cInfo, uint32_t naluType )
{
  if( cInfo != NULL )
  {
    CHECK( cInfo->getNoTrailConstraintFlag() && naluType == NAL_UNIT_CODED_SLICE_TRAIL,
           "Non-conforming bitstream. no_trail_constraint_flag is equal to 1 but bitstream contains NAL unit of type TRAIL_NUT." );
    CHECK( cInfo->getNoStsaConstraintFlag()  && naluType == NAL_UNIT_CODED_SLICE_STSA,
           "Non-conforming bitstream. no_stsa_constraint_flag is equal to 1 but bitstream contains NAL unit of type STSA_NUT." );
    CHECK( cInfo->getNoRaslConstraintFlag()  && naluType == NAL_UNIT_CODED_SLICE_RASL,
           "Non-conforming bitstream. no_rasl_constraint_flag is equal to 1 but bitstream contains NAL unit of type RASL_NUT." );
    CHECK( cInfo->getNoRadlConstraintFlag()  && naluType == NAL_UNIT_CODED_SLICE_RADL,
           "Non-conforming bitstream. no_radl_constraint_flag is equal to 1 but bitstream contains NAL unit of type RADL_NUT." );
    CHECK( cInfo->getNoIdrConstraintFlag()   && naluType == NAL_UNIT_CODED_SLICE_IDR_W_RADL,
           "Non-conforming bitstream. no_idr_constraint_flag is equal to 1 but bitstream contains NAL unit of type IDR_W_RADL." );
    CHECK( cInfo->getNoIdrConstraintFlag()   && naluType == NAL_UNIT_CODED_SLICE_IDR_N_LP,
           "Non-conforming bitstream. no_idr_constraint_flag is equal to 1 but bitstream contains NAL unit of type IDR_N_LP." );
    CHECK( cInfo->getNoCraConstraintFlag()   && naluType == NAL_UNIT_CODED_SLICE_CRA,
           "Non-conforming bitstream. no_cra_constraint_flag is equal to 1 but bitstream contains NAL unit of type CRA_NUT." );
    CHECK( cInfo->getNoGdrConstraintFlag()   && naluType == NAL_UNIT_CODED_SLICE_GDR,
           "Non-conforming bitstream. no_gdr_constraint_flag is equal to 1 but bitstream contains NAL unit of type GDR_NUT." );
    CHECK( cInfo->getNoApsConstraintFlag()   && naluType == NAL_UNIT_PREFIX_APS,
           "Non-conforming bitstream. no_aps_constraint_flag is equal to 1 but bitstream contains NAL unit of type APS_PREFIX_NUT." );
    CHECK( cInfo->getNoApsConstraintFlag()   && naluType == NAL_UNIT_SUFFIX_APS,
           "Non-conforming bitstream. no_aps_constraint_flag is equal to 1 but bitstream contains NAL unit of type APS_SUFFIX_NUT." );
  }
}

#define SEI_REPETITION_CONSTRAINT_LIST_SIZE  21

/**
 - Count the number of identical SEI messages in the current picture
 */
void DecLib::checkSeiInPictureUnit()
{
  std::vector<std::tuple<int, uint32_t, uint8_t*>> seiList;

  // payload types subject to constrained SEI repetition
  int picUnitRepConSeiList[SEI_REPETITION_CONSTRAINT_LIST_SIZE] = { 0, 1, 19, 45, 129, 132, 133, 137, 144, 145, 147, 148, 149, 150, 153, 154, 155, 156, 168, 203, 204};
  
  // extract SEI messages from NAL units
  for( auto &sei : m_pictureSeiNalus )
  {
    InputBitstream bs = sei.getBitstream();

    do
    {
      int payloadType = 0;
      uint32_t val = 0;

      do
      {
        val = bs.readByte();
        payloadType += val;
      } while (val==0xFF);

      uint32_t payloadSize = 0;
      do
      {
        val = bs.readByte();
        payloadSize += val;
      } while (val==0xFF);

      uint8_t *payload = new uint8_t[payloadSize];
      for( uint32_t i = 0; i < payloadSize; i++ )
      {
        val = bs.readByte();
        payload[i] = (uint8_t)val;
      }
      seiList.push_back(std::tuple<int, uint32_t, uint8_t*>(payloadType, payloadSize, payload));
    }
    while( bs.getNumBitsLeft() > 8 );
  }

  // count repeated messages in list
  for( uint32_t i = 0; i < seiList.size(); i++ )
  {
    int      k, count = 1;
    int      payloadType1 = std::get<0>(seiList[i]);
    uint32_t payloadSize1 = std::get<1>(seiList[i]);
    uint8_t  *payload1    = std::get<2>(seiList[i]);

    // only consider SEI payload types in the PicUnitRepConSeiList
    for( k=0; k<SEI_REPETITION_CONSTRAINT_LIST_SIZE; k++ )
    {
      if( payloadType1 == picUnitRepConSeiList[k] )
      {
        break;
      }
    }
    if( k >= SEI_REPETITION_CONSTRAINT_LIST_SIZE )
    {
      continue;
    }

    // compare current SEI message with remaining messages in the list
    for( uint32_t j = i+1; j < seiList.size(); j++ )
    {
      int      payloadType2 = std::get<0>(seiList[j]);
      uint32_t payloadSize2 = std::get<1>(seiList[j]);
      uint8_t  *payload2    = std::get<2>(seiList[j]);
      
      // check for identical SEI type, size, and payload
      if( payloadType1 == payloadType2 && payloadSize1 == payloadSize2 )
      {
        if( memcmp(payload1, payload2, payloadSize1*sizeof(uint8_t)) == 0 )
        {
          count++;
        }
      }
    }
    CHECK(count > 4, "There shall be less than or equal to 4 identical sei_payload( ) syntax structures within a picture unit.");
  }

  // free SEI message list memory
  for( uint32_t i = 0; i < seiList.size(); i++ )
  {
    uint8_t *payload = std::get<2>(seiList[i]);
    delete   payload;
  }
  seiList.clear();
}

/**
 - Reset list of SEI NAL units from the current picture
 */
void DecLib::resetPictureSeiNalus()
{
  m_pictureSeiNalus.clear();
}

void DecLib::checkAPSInPictureUnit()
{
  bool firstVCLFound = false;
  bool suffixAPSFound = false;
  for (auto &nalu : m_pictureUnitNals)
  {
    if (NALUnit::isVclNalUnitType(nalu))
    {
      firstVCLFound = true;
      CHECK( suffixAPSFound, "When any suffix APS NAL units are present in a PU, they shall follow the last VCL unit of the PU" );
    }
    else if (nalu == NAL_UNIT_PREFIX_APS)
    {
      CHECK( firstVCLFound, "When any prefix APS NAL units are present in a PU, they shall precede the first VCL unit of the PU");
    }
    else if (nalu == NAL_UNIT_SUFFIX_APS)
    {
      suffixAPSFound = true;
    }
  }
}

}
