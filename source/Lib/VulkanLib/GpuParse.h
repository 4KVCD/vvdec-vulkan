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
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

------------------------------------------------------------------------------------------- */

#pragma once

// The GPU's parse of the slice data (shaders/parse.slang): its inputs, built from a picture's slices (their slice
// data as the CPU's parse task captures it, the substreams, the CTUs in decoding order, the slices' parameters and
// initial contexts -- vvdec's own Ctx::init), and the comparison of what it leaves with the CPU's parse
// (VVDEC_GPU_PARSE=1).

#include "GpuFrontEnd.h"

#include <cstdint>
#include <vector>

namespace vvdec
{

class CodingStructure;

namespace gpu
{

// the layout of shaders/parse.slang's tables
static constexpr int kParsePicWords    = 160 + 3 * 128;
static constexpr int kParseSliceWords  = 16;
static constexpr int kParseStreamWords = 8;
static constexpr int kParseCtuOutWords = 16;
static constexpr int kParseScanTab     = 961;   // the scans' table: the diagonal scans of the blocks 1..16 x 1..16

// the inputs of a picture's parse, as parse.slang reads them
struct ParseInput
{
  std::vector<uint8_t>  bits;      // the slices' data, one after the other
  std::vector<uint32_t> streams;   // kParseStreamWords a substream
  std::vector<uint32_t> ctuList;   // 2 words a CTU, in decoding order
  std::vector<uint32_t> ctuInfo;   // 2 words a CTU, raster order
  std::vector<uint32_t> slices;    // a slice's words and initial contexts
  std::vector<int32_t>  pic;       // kParsePicWords
  uint32_t              numStreams = 0;
  uint32_t              cuSlots    = 0;   // a CTU's slots of CU records
  uint32_t              tuSlots    = 0;
  uint32_t              wppSlots   = 0;   // WPP's saved contexts
  uint32_t              contexts   = 0;   // the context models (CTX_COUNT)
};

// VVDEC_GPU_PARSE=1: the GPU parses the pictures too, compared with the CPU's parse (the CPU's parse task keeps the
// slice data for it)
bool parseShadowRequested();

// the picture's inputs; false when a slice's data was not captured
bool buildParseInput( const CodingStructure& cs, ParseInput& in );
// the diagonal scans of the blocks 1..16 x 1..16 (vvdec's g_scanOrder[SCAN_UNGROUPED]): per position its raster
// position | the scan index of the raster position << 16, kParseScanTab words
void writeParseScans( uint32_t* out );
// parsectx.slang's context sets against vvdec's (Ctx) and vvdec's grouped scans against parse.slang's composition of
// them: false when one differs (printed)
bool checkParseContexts();
// a CTU's SAO and ALF parameters as the CPU's parse left them, as parse.slang writes them (its ctuOut's words 2..8)
void captureSaoAlf( const CodingStructure& cs, int ctuRsAddr, uint32_t out[7] );

// the comparison of the GPU's parse with the CPU's
struct ParseStats
{
  int64_t pictures       = 0;
  int64_t errors         = 0;   // pictures the GPU's parse gave up on (a CTU's slots full, ...)
  int64_t cusCompared    = 0;
  int64_t cusDiffer      = 0;
  int64_t tusCompared    = 0;
  int64_t tusDiffer      = 0;
  int64_t ctusCompared   = 0;
  int64_t ctusDiffer     = 0;   // CTU records and SAO / ALF parameters
  int64_t levelsCompared = 0;
  int64_t levelsDiffer   = 0;
  int     reported       = 0;
  int64_t substreams     = 0;
  int64_t runs           = 0;    // the substreams' runs: one per substream and dispatch, until parsed
  int64_t submissions    = 0;    // the parse's submissions (GpuRecon::runParse)
  uint64_t prof[16]      = {};   // VVDEC_GPU_PROFILE: parse_prof's sums (parse.slang's kProf...)
};
// gpu*: the GPU's records in raster order of the CTUs, its CTU records and ctuOut words, the totals; cpu*: the CPU's
// records CTU by CTU and its SAO / ALF parameters (captureSaoAlf's, 7 words a CTU); levels: the check's counts (levels
// compared, differ, the first differing TU + 1, its component); errors: parse.slang's
void compareParse( const CodingStructure& cs, const uint32_t* gpuCus, const uint32_t* gpuTus, const uint32_t* gpuCtus, const uint32_t* gpuCtuOut,
                   const uint32_t* gpuTotals, const std::vector<std::vector<CuRecord>>& cpuCus, const std::vector<std::vector<TuRecord>>& cpuTus,
                   const std::vector<uint32_t>& cpuSaoAlf, const uint32_t* levels, uint32_t errors, ParseStats& stats );

}   // namespace gpu
}   // namespace vvdec
