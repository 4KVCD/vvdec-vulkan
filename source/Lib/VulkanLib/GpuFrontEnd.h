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
    \brief    The records the CPU's parse writes for the GPU's front end, and the tables its shaders read

    With the GPU front end, the CPU's CTU tasks only parse. For every CU, TU and CTU they write a fixed-size record
    of words that the front end's shaders read (mider.slang, dbparams.slang, jobs.slang). The records hold what the
    parse produced and nothing derived from it: a CU's motion vector differences, for example, not its motion
    vectors, which the GPU derives. See README.md for the overall flow.
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "GpuJobs.h"

namespace vvdec
{
class CodingStructure;

namespace gpu
{

// A CU as the parse leaves it: 32 words. Fields packed into a word are listed from the lowest bit up.
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
    kChroma  = 64,    // the CU belongs to the chroma tree (dual tree)
    kAffine6 = 128,   // the affine model has 6 parameters
  };

  // w0:  luma position: x (16 bits), y (16 bits)
  // w1:  luma size: width (8 bits), height (8 bits); prediction mode (2 bits, vvdec's PredMode); flags (Flags)
  // w2:  merge index (3 bits), MMVD index (6 bits), GPM split direction (6 bits), GPM merge indices (3 + 3 bits),
  //      AMVR index imv (2 bits), BCW index (3 bits), inter direction (2 bits), MVP index of list 0 and list 1
  // w3:  reference index of list 0 and list 1 (int8_t each), QP (int8_t), chroma QP adjustment (int8_t)
  // w4..w15: motion vector differences as parsed: list l's k-th (k = 0..2 for the affine control points) at
  //      w[4 + 6 * l + 2 * k] (horizontal) and the next word (vertical)
  // w16: index of the CU's first TU within its CTU (16 bits), number of TUs (16 bits)
  // w17: channel type (1 bit), tree type (2), mode type (2), ISP mode (2), LFNST index (2), MIP (1),
  //      MIP transposed (1), MRL index (2), colour transform (1), BDPCM luma (2), BDPCM chroma (2), root CBF (1),
  //      SBT info (8), CBFs of the planes (from bit 27)
  // w18: intra luma direction (8 bits), intra chroma direction (8 bits), QT depth (4 bits), depth (from bit 20)
  // w19: chroma block position in chroma samples: x (16 bits), y (16 bits)
  // w20: chroma block width (8 bits), height (8 bits); bit 16: the CU has a chroma block; bit 17: a luma block
  // w21: where the CU's DMVR refinements start in its CTU's share of the picture's refinements
  //      (CodingUnit::mvdL0SubPuOff minus the CTU's start, 16 bits); bit 16: the left neighbour is available; bit 17:
  //      the above neighbour is available (CodingUnit::left, above); bit 18: SMVD
  // w22..w31: reserved
  uint32_t w[32];
};

// A TU as the parse leaves it: 16 words.
// w0:  luma block position: x (16 bits), y (16 bits)
// w1:  luma block width (8 bits), height (8 bits); bit 16: the TU has a luma block
// w2, w3: the chroma block, the same in chroma samples
// w4:  CBFs (3 bits), joint Cb-Cr (2 bits), luma's MTS index (3 bits), Cb's and Cr's transform skip (1 bit each),
//      channel type (1 bit)
// w5:  chroma QPs of Cb and Cr (int8_t each)
// w6, w7: per component, the largest x (w6) and y (w7) of a nonzero coefficient, one byte per component
//      (vvdec's maxScanPosX, maxScanPosY)
// w8:  index of the TU's CU within its CTU
// w9, w10, w11: with the GPU's parse, where Y's, Cb's and Cr's levels are in the picture's levels (the index of their
//      first one; ParsedPicture::levels), each coded block's rectangle with its rows one after the other
// w12: the width of those rows, a byte per component (0: the CPU's parse, whose levels are in the picture's planes)
// w13..w15: reserved
struct TuRecord
{
  uint32_t w[16];
};

// A CTU: 4 words.
// w0: index of the CTU's first CU (the picture's CUs are stored CTU by CTU, each CTU's in decoding order)
// w1: number of CUs (16 bits), index of the CTU's slice in the picture's slices (16 bits)
// w2: bit 0: the CTU starts a row of its tile (the history of motion candidates starts anew); the tile's index
//     (from bit 16)
// w3: index of the CTU's first TU
struct CtuRecord
{
  uint32_t w[4];
};

// Appends the CU and TU records of a CTU, in decoding order, to cus and tus.
void writeCuRecords( CodingStructure& cs, int ctuRsAddr, std::vector<CuRecord>& cus, std::vector<TuRecord>& tus );
// Writes the CTU records of a picture, given its CU and TU records CTU by CTU.
void writeCtuRecords( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, const std::vector<std::vector<TuRecord>>& tus,
                      CtuRecord* out );

// A picture's slice table: one word per CTU with the index of its slice, then one record of 64 words per slice
// (its POC, reference lists and colocated picture). For TMVP, a slice record points to the colocated picture's
// motion and slice table. sliceTableWords() returns the table's size in words.
size_t sliceTableWords( const CodingStructure& cs );
void   writeSliceTable( const CodingStructure& cs, uint32_t* table );

// The most room each of the front end's lists of a picture can need (in jobs, intra bands and deblocking edges),
// worked out from its records.
struct FrontCapacities
{
  size_t trafo = 0, inter = 0, intra = 0, bands = 0, edges = 0;
};
FrontCapacities frontCapacities( const CodingStructure& cs, const std::vector<std::vector<CuRecord>>& cus, const std::vector<std::vector<TuRecord>>& tus );

// The constant tables that jobs.slang reads (its T_* tables, taken from vvdec): kJobTableWords words.
static constexpr size_t kJobTableWords = 2048;
void writeJobTables( uint32_t* out );
// The per-slice words that jobs.slang reads (its JobSlice), 256 words per slice of the picture:
// w0: slice type (2 bits); bit 2: LMCS; bit 3: dependent quantisation; bit 4: explicit scaling lists
// w[32 + 16 * list + refIdx]: the reference picture's slot in refs (8 bits); bit 8: scaled (RPR); bit 9: the same
//     size as the picture; bits 10..12: weights present for Y, Cb, Cr
// w[64 + 3 * (16 * list + refIdx) + component]: weighted prediction's weight (int16_t), offset (int8_t, bits 16..23)
//     and log2 denominator (from bit 24)
void writeJobSlices( const CodingStructure& cs, const RefTable& refs, uint32_t* out );

}   // namespace gpu
}   // namespace vvdec
