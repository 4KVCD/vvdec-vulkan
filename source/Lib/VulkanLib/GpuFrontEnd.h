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

namespace vvdec
{
class CodingStructure;
struct MotionInfo;

namespace gpu
{

// a CU as the parse leaves it -- before the motion derivation, which turns
// the motion vector differences into motion vectors -- for the front end's
// shaders (shaders/mider.slang): 16 words
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
  // w3: reference indices (int8_t) | << 8
  // w4..w15: the motion vector differences as parsed: list l's k-th at w[4 + 6 * l + 2 * k], (hor, ver)
  uint32_t w[16];
};

// a CTU's: its first CU (the picture's CUs CTU by CTU, each CTU's in decoding
// order), their count, its slice (the index in the picture's slices), its
// tile and whether it begins a tile's row (the history of motion begins
// anew): 4 words
struct CtuRecord
{
  uint32_t w[4];
};

// the CU records of a CTU, in decoding order, appended to out
void writeCuRecords( CodingStructure& cs, int ctuRsAddr, std::vector<CuRecord>& out );
// the CTU records of a picture whose CUs' records are cus, CTU by CTU
void writeCtuRecords( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, CtuRecord* out );

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
  int     reported   = 0;
};
// gpu: the motion of each 4x4 block as the GPU wrote it (shaders/mider.slang's), cpu: as vvdec keeps it
void compareMotion( const CodingStructure& cs, const int32_t* gpu, const MotionInfo* cpu, MiderStats& stats );

}   // namespace gpu
}   // namespace vvdec
