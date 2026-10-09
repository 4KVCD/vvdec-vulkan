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

/** \file     GpuFrontEnd.h
    \brief    the parse's CUs as the GPU's front end takes them (motion derivation, shaders/mider.slang)
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "GpuJobs.h"

namespace vvdec
{
class CodingStructure;
struct MotionInfo;
struct LoopFilterParam;

namespace gpu
{

// a CU as the parse leaves it -- before the motion derivation, which turns
// the motion vector differences into motion vectors -- for the front end's
// shaders (shaders/mider.slang, dbparams.slang): 32 words
struct CuRecord
{
  enum Flags : uint32_t
  {
    kMerge   = 1,
    kSkip    = 2,
    kMmvd    = 4,
    kAffine  = 8,
    kGeo     = 16,
    kCiip    = 32,
    kChroma  = 64,    // a CU of the chroma tree
    kAffine6 = 128,   // the 6-parameter model
  };

  // w0: the luma position, x | y << 16
  // w1: the luma size, width | height << 8; the prediction mode << 16 (vvdec's PredMode); the flags << 18
  // w2: merge index | MMVD index << 3 | GPM split << 9 | GPM indices << 15, << 18 | imv << 21 | BCW index << 23 |
  //     inter direction << 26 | MVP indices << 28, << 29
  // w3: reference indices (int8_t) | << 8; QP << 16 (int8_t); chroma QP adjustment << 24 (int8_t)
  // w4..w15: the motion vector differences as parsed: list l's k-th at w[4 + 6 * l + 2 * k], (hor, ver)
  // w16: its first TU (in its CTU's) | their count << 16
  // w17: chType | treeType << 1 | modeType << 3 | ISP << 5 | LFNST index << 7 | MIP << 9 | MIP transposed << 10 |
  //      MRL index << 11 | colour transform << 13 | BDPCM luma << 14 | chroma << 16 | root CBF << 18 | SBT info
  //      << 19 (8 bits) | the planes' CBFs << 27
  // w18: intra direction luma (uint8_t) | chroma << 8 | QT depth << 16 | depth << 20
  // w19: the chroma block (chroma samples), x | y << 16; w20: its width | height << 8 | chroma there << 16 |
  //      luma there << 17
  // w21: DMVR's refinements' place (CodingUnit::mvdL0SubPuOff, in its CTU's) | left neighbour there << 16 | above
  //      << 17 (CodingUnit::left, above) | SMVD << 18
  // w22..w31: reserved
  uint32_t w[32];
};

// a TU of the parse: 16 words
// w0, w1: its luma block x | y << 16, width | height << 8 | there << 16
// w2, w3: its chroma block (chroma samples), the same
// w4: CBFs (3 bits) | joint Cb-Cr << 3 | MTS index luma << 5 (3 bits) | Cb << 8 | Cr << 9 | chType << 10
// w5: the chroma QPs (int8_t) | << 8
// w6, w7: the highest scan positions, x and y, per component (a byte each)
// w8: its CU (in its CTU's)
// w9..w15: reserved
struct TuRecord
{
  uint32_t w[16];
};

// a CTU's: its first CU (the picture's CUs CTU by CTU, each CTU's in decoding
// order), their count, its slice (the index in the picture's slices), its
// tile and whether it begins a tile's row (the history of motion begins
// anew), its first TU: 4 words
struct CtuRecord
{
  uint32_t w[4];
};

// the CU and TU records of a CTU, in decoding order, appended to cus and tus
void writeCuRecords( CodingStructure& cs, int ctuRsAddr, std::vector<CuRecord>& cus, std::vector<TuRecord>& tus );
// the CTU records of a picture whose CUs' and TUs' records are cus and tus, CTU by CTU
void writeCtuRecords( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, const std::vector<std::vector<TuRecord>>& tus,
                      CtuRecord* out );

// a picture's slice table: per CTU its slice's index, then the slices'
// records (64 words each: their POC, reference lists, colocated picture --
// its motion and slice table, for TMVP in the motion derivation, which reads
// the slice table of the colocated picture's CTUs): its words
size_t sliceTableWords( const CodingStructure& cs );
void   writeSliceTable( const CodingStructure& cs, uint32_t* table );

// the comparison of the GPU's motion derivation with the CPU's (VVDEC_GPU_MIDER=1)
struct MiderStats
{
  int64_t pictures   = 0;
  int64_t compared   = 0;   // 4x4 blocks
  int64_t mismatched = 0;
  int64_t skipped    = 0;   // of the CUs the GPU doesn't derive yet
  int64_t lfpCompared   = 0;   // the deblocking's parameters (4x4 blocks and directions)
  int64_t lfpMismatched = 0;
  int64_t lfpEdges      = 0;   // of them with a boundary strength (the CPU's)
  int     lfpReported   = 0;
  int     reported   = 0;
  int64_t jobsCompared[3]   = { 0, 0, 0 };   // the jobs (DecCu's xGpu*Jobs; shaders/jobs.slang): residuals', inter's, intra's
  int64_t jobsMismatched[3] = { 0, 0, 0 };
  int     jobsReported      = 0;
  int64_t sortedCompared    = 0;   // intra's bands in the order of the waves (GpuRecon::recordIntra; shaders/intrasort.slang)
  int64_t sortedMismatched  = 0;
  int64_t edgesCompared     = 0;   // the deblocking's edges (GpuRecon::listEdges; shaders/edges.slang)
  int64_t edgesMismatched   = 0;
  int     edgesReported     = 0;
};
// gpu: the motion of each 4x4 block as the GPU wrote it (shaders/mider.slang's), cpu: as vvdec keeps it
void compareMotion( const CodingStructure& cs, const int32_t* gpu, const MotionInfo* cpu, MiderStats& stats );
// gpu: the deblocking's parameters as the GPU wrote them (shaders/dbparams.slang's, 2 words each), cpu: vvdec's
void compareLfp( const CodingStructure& cs, const uint32_t* gpu, const LoopFilterParam* cpu, MiderStats& stats );

// the jobs' tables (shaders/jobs.slang's T_*): vvdec's, kJobTableWords words
static constexpr size_t kJobTableWords = 2048;
void writeJobTables( uint32_t* out );
// per slice of the picture 256 words for its jobs (shaders/jobs.slang's JobSlice): w0: the slice type | LMCS << 2 |
// dependent quantisation << 3 | explicit scaling lists << 4; w32 + 16 * list + refIdx: the reference's slot in refs |
// scaled (RPR) << 8 | of the picture's size << 9 | weights present (Y, Cb, Cr) << 10; w64 + ( 16 * list + refIdx ) *
// 3 + component: its weight (int16_t) | offset << 16 (int8_t) | log2 denominator << 24
void writeJobSlices( const CodingStructure& cs, const RefTable& refs, uint32_t* out );
// gpu: the lists the GPU made (count jobs each, words a job), cpu: the CTU tasks' (list 0 residuals', 1 inter's, 2
// intra's)
// gpu: intra's bands in the order of the waves as the GPU made them (12 words each) and each one's wait, count of them;
// cpu: the CTU tasks' jobs, which recordIntra puts in that order
void compareSorted( const CodingStructure& cs, const uint32_t* gpu, const uint32_t* waits, size_t count, const std::vector<std::vector<IntraJob>>& cpu,
                    MiderStats& stats );
// gpu: the edges the GPU listed (2 words each) and its lists' firsts and counts (2 words each), cpu: the CTU tasks'
void compareEdges( const CodingStructure& cs, const uint32_t* gpu, const uint32_t* lists, size_t capacity, const std::vector<EdgeLists>& cpu,
                   MiderStats& stats );
template<typename Job>
void compareJobs( const CodingStructure& cs, int list, const uint32_t* gpu, size_t count, const std::vector<std::vector<Job>>& cpu, MiderStats& stats );

}   // namespace gpu
}   // namespace vvdec
