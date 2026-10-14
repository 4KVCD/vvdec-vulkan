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

/** \file     vvdecbatch.cpp
    \brief    Decodes a list of bitstreams in one process and prints each output frame's per-plane MD5: a test aid for
              the Vulkan port's fast checks. With the GPU, a process per bitstream spends about a second setting up
              the Vulkan device and its pipelines and tearing them down; here they stay up for the whole list.

    vvdecbatch <list file>    (or - : the jobs from stdin, each decoded as its line comes, for a runner that hands
                               out jobs one by one)
    Each line of the list: <bitstream> <frames> <threads> <gpu> <film grain>
      frames:     stop once that many frames are output, without flushing the rest (0: all frames);
      threads:    vvdecParams::threads; gpu: vvdecParams::gpu (0 or 1); film grain: vvdecParams::filmGrainSynthesis.
    The decoder's own environment variables (VVDEC_GPU_PARSE, ...) apply to every job; VVDECBATCH_NOHASH=1 leaves the
    hashes out (a frame's line ends after its number), for timing.
    Out, for job i (counted from 0): "job i frame k [MD5:y,u,v]" for each output frame, as vvdecapp -dph 11 prints
    it, then "job i end OK <seconds> open <s> first <s> close <s>" (the seconds of the whole job, of opening the
    decoder, until its first frame, of closing it), or "job i end ERROR <message>", or "job i end FIELDS" for a stream of field
    pictures (whose frames this does not hash).
    From the first job on the GPU to the end, one more decoder holds the GPU's context, so that the jobs share it.
*/

#include "vvdec/vvdec.h"

#include "../vvdecapp/vvdecHelper.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

struct Job
{
  std::string bitstream;
  unsigned    frames    = 0;
  int         threads   = 16;
  int         gpu       = 1;
  int         filmGrain = 0;
};

// The library's messages: errors to stderr, the rest dropped
void logErrors( void*, int level, const char* fmt, va_list args )
{
  if( level <= VVDEC_ERROR )
  {
    vfprintf( stderr, fmt, args );
  }
}

// The decoder's last error with its details (an exception's message, which says where it came from), on one line
std::string lastError( vvdecDecoder* dec )
{
  std::string error   = vvdec_get_last_error( dec );
  const char* details = vvdec_get_last_additional_error( dec );
  if( details && *details )
  {
    error += std::string( ": " ) + details;
  }
  for( char& c: error )
  {
    c = c == '\n' || c == '\r' ? ' ' : c;   // (the job's end line must stay one line)
  }
  return error;
}

// Whether the environment variable is set (MSVC refuses getenv)
static bool environmentSet( const char* name )
{
#ifdef _MSC_VER
  char*      value = nullptr;
  size_t     size  = 0;
  const bool set   = _dupenv_s( &value, &size, name ) == 0 && value != nullptr;
  free( value );
  return set;
#else
  return getenv( name ) != nullptr;
#endif
}

// Prints an output frame's per-plane MD5 and gives it back to the decoder. Returns false for a field picture.
bool handleFrame( vvdecDecoder* dec, vvdecFrame* frame, int index, unsigned& frames )
{
  if( frame->frameFormat != VVDEC_FF_PROGRESSIVE )
  {
    vvdec_frame_unref( dec, frame );
    return false;
  }
  // (VVDECBATCH_NOHASH: no hash, for timing: a 4K frame's MD5 takes some 37 ms, more than decoding it)
  static const bool noHash = environmentSet( "VVDECBATCH_NOHASH" );
  if( noHash )
  {
    std::cout << "job " << index << " frame " << frames << "\n";
  }
  else
  {
    std::vector<uint8_t> hash;
    const int            chars = calc_md5( frame, hash );
    std::cout << "job " << index << " frame " << frames << " [MD5:" << hashToString( hash, chars ) << "]\n";
  }
  frames++;
  vvdec_frame_unref( dec, frame );
  return true;
}

// Decodes one job: its frames' hashes to stdout, then its end line
void decodeJob( int index, const Job& job )
{
  const auto  start = std::chrono::steady_clock::now();
  vvdecParams params;
  vvdec_params_default( &params );
  params.threads            = job.threads;
  params.gpu                = job.gpu;
  params.filmGrainSynthesis = job.filmGrain != 0;
  params.logLevel           = VVDEC_ERROR;
  std::ifstream file( job.bitstream, std::ios::binary );
  if( !file )
  {
    std::cout << "job " << index << " end ERROR cannot open " << job.bitstream << std::endl;
    return;
  }
  vvdecDecoder* dec    = vvdec_decoder_open( &params );
  const auto    opened = std::chrono::steady_clock::now();
  if( !dec )
  {
    std::cout << "job " << index << " end ERROR cannot open the decoder" << std::endl;
    return;
  }
  vvdec_set_logging_callback( dec, logErrors );
  vvdecAccessUnit* au = vvdec_accessUnit_alloc();
  vvdec_accessUnit_alloc_payload( au, MAX_CODED_PICTURE_SIZE );
  au->cts      = 0;
  au->ctsValid = true;
  au->dts      = 0;
  au->dtsValid = true;

  // vvdecapp's input loop: the access units' time stamps count the pictures (a picture's first slice), so that the
  // frames come out with valid time stamps
  unsigned     frames          = 0;
  double       firstFrame      = -1;   // seconds from the start to the first frame
  bool         fields          = false;
  bool         stop            = false;
  std::string  error;
  vvdecNalType sliceType       = VVC_NAL_UNIT_INVALID;
  bool         multipleSlices  = false;
  int          read            = 0;
  do
  {
    read                       = readBitstreamFromFile( &file, au, false );
    const vvdecNalType nalType = vvdec_get_nal_unit_type( au );
    multipleSlices             = multipleSlices || nalType == VVC_NAL_UNIT_PH;
    bool isSlice               = vvdec_is_nal_unit_slice( nalType );
    if( isSlice && multipleSlices )
    {
      if( sliceType == VVC_NAL_UNIT_INVALID )
      {
        sliceType = nalType;
      }
      else
      {
        isSlice = false;   // (not the picture's first slice)
      }
    }
    if( sliceType != VVC_NAL_UNIT_INVALID && nalType != sliceType )
    {
      sliceType = VVC_NAL_UNIT_INVALID;
    }
    vvdecFrame* frame = nullptr;
    const int   ret   = vvdec_decode( dec, au, &frame );
    if( isSlice )
    {
      au->cts++;
      au->dts++;
    }
    // as vvdecapp: a broken input unit is only a warning; the end of the input still comes with a frame
    if( ret != VVDEC_OK && ret != VVDEC_TRY_AGAIN && ret != VVDEC_EOF && ret != VVDEC_ERR_DEC_INPUT )
    {
      error = lastError( dec );
      break;
    }
    if( frame && frame->ctsValid )
    {
      firstFrame = firstFrame < 0 ? std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count() : firstFrame;
      fields     = !handleFrame( dec, frame, index, frames ) || fields;
      stop       = job.frames && frames >= job.frames;
    }
    if( ret == VVDEC_EOF )
    {
      break;
    }
  } while( read > 0 && !stop && !fields );

  // the frames left, unless the job's frames are out
  while( error.empty() && !stop && !fields )
  {
    vvdecFrame* frame = nullptr;
    const int   ret   = vvdec_flush( dec, &frame );
    if( ret != VVDEC_OK && ret != VVDEC_EOF )
    {
      error = lastError( dec );
      break;
    }
    // as vvdecapp: a frame can come with the end; the flush is over when none comes
    if( !frame || !frame->ctsValid )
    {
      break;
    }
    fields = !handleFrame( dec, frame, index, frames ) || fields;
    stop   = job.frames && frames >= job.frames;
  }
  vvdec_accessUnit_free( au );
  const auto closing = std::chrono::steady_clock::now();
  vvdec_decoder_close( dec );

  const double seconds = std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
  std::cout << "job " << index << " end ";
  if( fields )
  {
    std::cout << "FIELDS";
  }
  else if( !error.empty() )
  {
    std::cout << "ERROR " << error;
  }
  else
  {
    // the seconds, then where they went: opening the decoder, until the first frame, closing it
    std::cout << "OK " << seconds << " open " << std::chrono::duration<double>( opened - start ).count() << " first " << firstFrame << " close "
              << std::chrono::duration<double>( std::chrono::steady_clock::now() - closing ).count();
  }
  std::cout << std::endl;
}

}   // namespace

int main( int argc, char* argv[] )
{
  if( argc < 2 )
  {
    std::cerr << "usage: vvdecbatch <list file, or - for one job a line on stdin>: lines of <bitstream> <frames> <threads> <gpu> <film grain>"
              << std::endl;
    return 1;
  }
  std::ifstream file;
  const bool    fromStdin = std::string( argv[1] ) == "-";
  if( !fromStdin )
  {
    file.open( argv[1] );
  }
  std::istream& list = fromStdin ? std::cin : file;

  // each job as its line comes (with stdin, the next line once this job's end line is out); the decoder that holds
  // the GPU's context from the first job on the GPU to the end
  vvdecDecoder* keeper = nullptr;
  std::string   line;
  int           index = 0;
  while( std::getline( list, line ) )
  {
    std::istringstream fields( line );
    Job                job;
    if( !( fields >> job.bitstream >> job.frames >> job.threads >> job.gpu >> job.filmGrain ) )
    {
      continue;
    }
    if( job.gpu && !keeper )
    {
      vvdecParams params;
      vvdec_params_default( &params );
      params.threads  = 1;
      params.gpu      = 1;
      params.logLevel = VVDEC_ERROR;
      keeper          = vvdec_decoder_open( &params );
    }
    decodeJob( index++, job );
  }
  if( keeper )
  {
    vvdec_decoder_close( keeper );
  }
  return 0;
}
