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

// The host side of the GPU's parse of the slice data (the CABAC parse in shaders/parse.slang and the shaders it
// includes). With VVDEC_GPU_PARSE=1 the GPU parses every picture as well, and the result is compared with the CPU's
// parse; the CPU's parse still drives the decoding. This header declares:
// - buildParseInput, which lays out a picture's slices as parse.slang reads them: the slice data after each slice
//   header (which the CPU's parse task keeps for this), the substreams the entry points split it into, the CTUs in
//   decoding order, each slice's parameters and initial contexts (computed with vvdec's own Ctx::init), and the
//   picture's parameters;
// - writeParseScans and checkParseContexts, for the tables the shader shares with vvdec;
// - captureSaoAlf and compareParse, which compare what the GPU's parse leaves with what the CPU's parse left.
// GpuRecon (GpuReconParse.cpp) uploads the inputs, runs the shaders and calls compareParse.

#include "GpuFrontEnd.h"

#include <cstdint>
#include <vector>

namespace vvdec
{

class CodingStructure;

namespace gpu
{

// The layout of parse.slang's tables (keep in step with its constants)
static constexpr int kParsePicWords    = 160 + 3 * 128;   // the picture's: 160 parameter words, 3 chroma QP maps of 128
static constexpr int kParseSliceWords  = 16;    // a slice's parameter words (its initial contexts follow them)
static constexpr int kParseStreamWords = 8;     // a substream's words
static constexpr int kParseCtuOutWords = 16;    // the words parse.slang writes per CTU besides the records (ctuOut)
static constexpr int kParseScanTab     = 961;   // the scan table's words (writeParseScans)

// The inputs of a picture's parse, laid out as parse.slang reads them
struct ParseInput
{
  std::vector<uint8_t>  bits;      // the slices' data (each from the end of its slice header), one after another
  // Per substream, kParseStreamWords words: its first byte in bits, its size in bytes, its first CTU in ctuList, its
  // number of CTUs and the index of its slice (words 5..7 are unused)
  std::vector<uint32_t> streams;
  // Per CTU in decoding order, 2 words: its raster-scan address; its flags (parse.slang's kCtu...) | its WPP context
  // slot << 8 (CTU row * tile columns + tile column)
  std::vector<uint32_t> ctuList;
  // Per CTU in raster order, 2 words: the index of its slice; the index of its tile | 1 << 16 if the CTU is in its
  // tile's first column
  std::vector<uint32_t> ctuInfo;
  // Per slice, kParseSliceWords parameter words, then its contexts' initial states and their rates (CTX_COUNT words
  // each)
  std::vector<uint32_t> slices;
  std::vector<int32_t>  pic;       // the picture's parameters, kParsePicWords words
  uint32_t              numStreams = 0;   // the number of substreams in streams
  uint32_t              cuSlots    = 0;   // CU record slots per CTU: the most CUs a CTU can have
  uint32_t              tuSlots    = 0;   // TU record slots per CTU (as many)
  uint32_t              wppSlots   = 0;   // slots for WPP's saved contexts: one per CTU row and tile column
  uint32_t              contexts   = 0;   // the number of context models (CTX_COUNT)
  // Whether the picture leaves off every tool parse.slang's LITE variants leave out (its SPS: SBT, affine, BCW, CIIP,
  // GPM, MMVD, SMVD, ISP, MRL, MIP, LFNST, explicit MTS, ACT, joint CbCr; its slices: dependent quantization)
  bool                  lite = false;
};

// Whether VVDEC_GPU_PARSE=1 asks for the GPU's parse. With it, the GPU parses every picture as well, the result is
// compared with the CPU's parse, and the CPU's parse task keeps each slice's data for the GPU.
bool parseShadowRequested();

// Fills in with the inputs of the picture's parse. Returns false when a slice's data was not kept; the picture is then
// not parsed on the GPU.
bool buildParseInput( const CodingStructure& cs, ParseInput& in );
// Writes the scan table, kParseScanTab words: the diagonal scans of the blocks 1, 2, 4, 8 and 16 samples wide and
// high (vvdec's g_scanOrder[SCAN_UNGROUPED]). Each block's scan takes w * h words from where scanTabOffset says; word
// k holds the raster position of scan position k | the scan position of raster position k << 16.
void writeParseScans( uint32_t* out );
// Checks parsectx.slang's context sets against vvdec's (Ctx), and the grouped scans parse.slang composes from the
// scan table against vvdec's. Returns false, after printing what differs, when anything does.
bool checkParseContexts();
// Writes a CTU's SAO and ALF parameters as the CPU's parse left them, in the 7 words that parse.slang writes for them
// (words 2..8 of the CTU's ctuOut).
void captureSaoAlf( const CodingStructure& cs, int ctuRsAddr, uint32_t out[7] );

// The comparison of the GPU's parse with the CPU's, summed over the pictures
struct ParseStats
{
  int64_t pictures       = 0;
  int64_t errors         = 0;   // pictures with a substream that gave up (slots full, ...) or was left unparsed
  int64_t cusCompared    = 0;
  int64_t cusDiffer      = 0;
  int64_t tusCompared    = 0;
  int64_t tusDiffer      = 0;
  int64_t ctusCompared   = 0;
  int64_t ctusDiffer     = 0;   // CTUs whose record or SAO / ALF parameters differ
  int64_t levelsCompared = 0;
  int64_t levelsDiffer   = 0;
  int     reported       = 0;   // the differences printed so far (compareParse prints at most 20)
  int64_t substreams     = 0;
  int64_t runs           = 0;    // the substreams' runs: one per substream and dispatch until the substream is parsed
  int64_t submissions    = 0;    // the parse's submissions (GpuRecon::runParse)
  uint64_t prof[16]      = {};   // VVDEC_GPU_PROFILE: the sums of parse_prof's counters (parse.slang's kProf...)
  // the check of the room of the front end's job lists the parse adds up, against gpu::frontCapacities
  int64_t capsCompared   = 0;
  int64_t capsDiffer     = 0;
  // the parse ahead of the reconstruction (VVDEC_GPU_PARSE=2: ParseService)
  int64_t restarts       = 0;   // pictures parsed again, with more room for their records
  int64_t parsingSum     = 0;   // the pictures in each submission, added up
  int64_t maxCus         = 0;   // the most CU records a picture had
  int64_t maxTus         = 0;
};
// Compares the GPU's parse of a picture with the CPU's, adds the counts to stats and prints the first differences.
// gpuCus, gpuTus: the GPU's CU and TU records in raster order of the CTUs (gathered by parsecompact.slang); gpuCtus:
// its CTU records; gpuCtuOut: parse.slang's ctuOut words.
// cpuCus, cpuTus: the CPU's records, CTU by CTU; cpuSaoAlf: the CPU's SAO and ALF parameters, 7 words per CTU
// (captureSaoAlf's; empty when they were not captured). levels: the levels check's counts (the levels compared, the
// levels that differ, the index + 1 of the first TU found to differ, a component of it that differs). errors: the
// number of substreams that gave up or were left unparsed.
void compareParse( const CodingStructure& cs, const uint32_t* gpuCus, const uint32_t* gpuTus, const uint32_t* gpuCtus, const uint32_t* gpuCtuOut,
                   const std::vector<std::vector<CuRecord>>& cpuCus, const std::vector<std::vector<TuRecord>>& cpuTus,
                   const std::vector<uint32_t>& cpuSaoAlf, const uint32_t* levels, uint32_t errors, ParseStats& stats );

}   // namespace gpu
}   // namespace vvdec
