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

/** \file     GpuFrontEndCheck.h
    \brief    The check of the GPU front end against the CPU's results (VVDEC_GPU_MIDER=1)

    With VVDEC_GPU_MIDER=1 the CPU front end drives the pictures and the GPU front end runs beside it on the same
    records. After each picture, GpuRecon::compareFrontEnd (GpuFrontEndCheck.cpp) compares the two, and DecLib
    prints the totals when the decoder closes. It is a debugging aid.
*/

#pragma once

#include <cstdint>

namespace vvdec
{
namespace gpu
{

// The totals of the check over a decoder instance's pictures. The first 20 differences of each kind are printed.
struct FrontEndStats
{
  int64_t pictures = 0;
  // motion: 4x4 blocks compared, and those that differ
  int64_t compared   = 0;
  int64_t mismatched = 0;
  int     reported   = 0;
  // deblocking parameters: per 4x4 block and edge direction
  int64_t lfpCompared   = 0;
  int64_t lfpMismatched = 0;
  int64_t lfpEdges      = 0;   // the compared ones the CPU gave a boundary strength
  int     lfpReported   = 0;
  // jobs (DecCu's xGpu*Jobs against jobs.slang): residual, inter and intra jobs
  int64_t jobsCompared[3]   = { 0, 0, 0 };
  int64_t jobsMismatched[3] = { 0, 0, 0 };
  int     jobsReported      = 0;
  // intra bands in wave order (GpuRecon::recordIntra against intrasort.slang)
  int64_t sortedCompared   = 0;
  int64_t sortedMismatched = 0;
  // deblocking edges (GpuRecon::listEdges against edges.slang)
  int64_t edgesCompared   = 0;
  int64_t edgesMismatched = 0;
  int     edgesReported   = 0;
};

}   // namespace gpu
}   // namespace vvdec
