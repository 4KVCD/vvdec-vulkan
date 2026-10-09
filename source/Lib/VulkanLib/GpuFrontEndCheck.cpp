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

/** \file     GpuFrontEndCheck.cpp
    \brief    The check of the GPU front end against the CPU's results (VVDEC_GPU_MIDER=1)
*/

#include "GpuFrontEndCheck.h"
#include "GpuRecon.h"

#include "CommonLib/CodingStructure.h"
#include "CommonLib/LoopFilter.h"
#include "CommonLib/MotionInfo.h"
#include "CommonLib/Picture.h"
#include "CommonLib/Slice.h"
#include "CommonLib/Unit.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>

namespace vvdec
{
namespace gpu
{

static constexpr int kMaxReports = 20;   // differences printed per kind

// Compares the motion of each 4x4 block as mider.slang wrote it (5 words per block) with vvdec's MotionInfo.
static void compareMotion( const CodingStructure& cs, const int32_t* gpu, const MotionInfo* cpu, FrontEndStats& stats )
{
  const PreCalcValues& pcv     = *cs.pcv;
  const int            ctuSize = int( pcv.maxCUWidth );
  const int            size4   = ctuSize >> 2;
  stats.pictures++;
  for( int ctu = 0; ctu < (int) pcv.sizeInCtus; ctu++ )
  {
    const int x0 = ( ctu % pcv.widthInCtus ) * ctuSize;
    const int y0 = ( ctu / pcv.widthInCtus ) * ctuSize;
    for( int y = 0; y < size4; y++ )
    {
      for( int x = 0; x < size4; x++ )
      {
        if( x0 + 4 * x >= int( pcv.lumaWidth ) || y0 + 4 * y >= int( pcv.lumaHeight ) )
        {
          continue;
        }
        const int         at   = ctu * size4 * size4 + y * size4 + x;
        const int32_t*    g    = gpu + at * 5;
        const MotionInfo& c    = cpu[at];
        const uint32_t    w    = uint32_t( g[4] );
        const int         kind = int( ( w >> 16 ) & 0xff );
        stats.compared++;
        const int ref[2] = { int( int8_t( w & 0xff ) ), int( int8_t( ( w >> 8 ) & 0xff ) ) };
        bool      same   = true;
        for( int l = 0; l < 2; l++ )
        {
          const int cref = c.miRefIdx[l] < 0 ? -1 : c.miRefIdx[l];
          same &= ref[l] == cref;
          // (IBC has a block vector without a reference index)
          if( ref[l] >= 0 || ( kind == 3 && l == 0 ) )
          {
            same &= g[2 * l] == c.mv[l].getHor() && g[2 * l + 1] == c.mv[l].getVer();
          }
        }
        if( !same )
        {
          stats.mismatched++;
          if( stats.reported < kMaxReports )
          {
            stats.reported++;
            msg( WARNING,
                 "GPU motion derivation: POC %d at (%d,%d) kind %d: GPU ref %d,%d mv (%d,%d) (%d,%d), CPU ref %d,%d mv (%d,%d) (%d,%d)\n",
                 cs.picture->poc, x0 + 4 * x, y0 + 4 * y, kind, ref[0], ref[1], g[0], g[1], g[2], g[3], c.miRefIdx[0], c.miRefIdx[1],
                 c.mv[0].getHor(), c.mv[0].getVer(), c.mv[1].getHor(), c.mv[1].getVer() );
            // the CU there, as the CPU left it
            if( const CodingUnit* cu = cs.getCU( Position( x0 + 4 * x, y0 + 4 * y ), CHANNEL_TYPE_LUMA ) )
            {
              msg( WARNING,
                   "  CU (%d,%d) %dx%d mode %d merge %d idx %d mmvd %d (%d) affine %d type %d geo %d ciip %d mrgType %d interDir %d refIdx %d,%d "
                   "mvp %d,%d imv %d bcw %d slice %d\n",
                   cu->lx(), cu->ly(), cu->lwidth(), cu->lheight(), cu->predMode(), cu->mergeFlag(), cu->mergeIdx(), cu->mmvdFlag(), cu->mmvdIdx,
                   cu->affineFlag(), cu->affineType(), cu->geoFlag(), cu->ciipFlag(), cu->mergeType(), cu->interDir(), cu->refIdx[0], cu->refIdx[1],
                   cu->mvpIdx[0], cu->mvpIdx[1], cu->imv(), cu->BcwIdx(), cu->slice->getIndependentSliceIdx() );
            }
          }
        }
      }
    }
  }
}

// Compares the deblocking parameters as dbparams.slang wrote them (2 words per 4x4 block and direction) with vvdec's
// LoopFilterParam.
static void compareLfp( const CodingStructure& cs, const uint32_t* gpu, const LoopFilterParam* cpu, FrontEndStats& stats )
{
  const PreCalcValues& pcv = *cs.pcv;
  const size_t         n   = size_t( pcv.sizeInCtus ) * 2 * pcv.num4x4CtuBlks;
  const int            per = int( pcv.num4x4CtuBlks );
  const int            s4  = int( pcv.maxCUWidth ) >> 2;
  for( size_t i = 0; i < n; i++ )
  {
    const LoopFilterParam& c = cpu[i];
    const uint32_t         a = gpu[2 * i];
    const uint32_t         b = gpu[2 * i + 1];
    stats.lfpCompared++;
    stats.lfpEdges += c.bs != 0 ? 1 : 0;
    const bool same = int8_t( a & 0xff ) == c.qp[0] && int8_t( ( a >> 8 ) & 0xff ) == c.qp[1] && int8_t( ( a >> 16 ) & 0xff ) == c.qp[2]
                      && uint8_t( a >> 24 ) == c.bs && uint8_t( b & 0xff ) == c.sideMaxFiltLength && uint8_t( ( b >> 8 ) & 0xff ) == c.flags;
    if( !same )
    {
      stats.lfpMismatched++;
      if( stats.lfpReported < kMaxReports )
      {
        stats.lfpReported++;
        const int ctu = int( i / ( 2 * per ) );
        const int dir = int( ( i / per ) & 1 );
        const int pos = int( i % per );
        const int x   = ( ctu % int( pcv.widthInCtus ) ) * int( pcv.maxCUWidth ) + ( pos % s4 ) * 4;
        const int y   = ( ctu / int( pcv.widthInCtus ) ) * int( pcv.maxCUHeight ) + ( pos / s4 ) * 4;
        msg( WARNING,
             "GPU deblocking parameters: POC %d at (%d,%d) dir %d: GPU qp %d,%d,%d bs %02x len %02x flags %02x, CPU qp %d,%d,%d bs %02x len %02x flags %02x\n",
             cs.picture->poc, x, y, dir, int8_t( a & 0xff ), int8_t( ( a >> 8 ) & 0xff ), int8_t( ( a >> 16 ) & 0xff ), a >> 24, b & 0xff,
             ( b >> 8 ) & 0xff, c.qp[0], c.qp[1], c.qp[2], c.bs, c.sideMaxFiltLength, c.flags );
      }
    }
  }
}

// Compares one of the GPU's job lists (count jobs) with the CTU tasks' (list 0: residual, 1: inter, 2: intra jobs).
template<typename Job>
static void compareJobs( const CodingStructure& cs, int list, const uint32_t* gpu, size_t count, const std::vector<std::vector<Job>>& cpu,
                         FrontEndStats& stats )
{
  static const char* const names[3] = { "residuals'", "inter's", "intra's" };
  constexpr size_t         words    = sizeof( Job ) / sizeof( uint32_t );
  size_t                   total    = 0;
  for( const auto& ctu: cpu )
  {
    total += ctu.size();
  }
  if( total != count )
  {
    stats.jobsMismatched[list] += std::max( total, count );
    if( stats.jobsReported < kMaxReports )
    {
      stats.jobsReported++;
      msg( WARNING, "GPU jobs: POC %d: %s %zu on the GPU, %zu on the CPU\n", cs.picture->poc, names[list], count, total );
    }
  }
  size_t at = 0;
  for( size_t ctu = 0; ctu < cpu.size(); ctu++ )
  {
    for( const Job& job: cpu[ctu] )
    {
      stats.jobsCompared[list]++;
      if( at < count && memcmp( &job, gpu + at * words, sizeof( Job ) ) != 0 )
      {
        stats.jobsMismatched[list]++;
        if( stats.jobsReported < kMaxReports )
        {
          stats.jobsReported++;
          std::string g, c;
          for( size_t w = 0; w < words; w++ )
          {
            char buf[16];
            snprintf( buf, sizeof( buf ), " %08x", gpu[at * words + w] );
            g += buf;
            snprintf( buf, sizeof( buf ), " %08x", job.w[w] );
            c += buf;
          }
          msg( WARNING, "GPU jobs: POC %d CTU %zu %s job %zu: GPU%s, CPU%s\n", cs.picture->poc, ctu, names[list], at, g.c_str(), c.c_str() );
        }
      }
      at++;
    }
  }
}

// Compares the intra bands in wave order as intrasort.slang made them (12 words each, count bands, each band's wait
// in waits) with the CTU tasks' jobs, put into that order as GpuRecon::recordIntra does.
static void compareSorted( const CodingStructure& cs, const uint32_t* gpu, const uint32_t* waits, size_t count, const std::vector<std::vector<IntraJob>>& cpu,
                           FrontEndStats& stats )
{
  static constexpr int kBand = 256;
  const auto    bandRows = []( const IntraJob& job ) { return std::max( 1, kBand / job.width() ); };
  const auto    bands    = [&]( const IntraJob& job ) { return ( job.height() + bandRows( job ) - 1 ) / bandRows( job ); };
  int           maxWave  = 0;
  size_t        total    = 0;
  for( const auto& ctu: cpu )
  {
    for( const IntraJob& job: ctu )
    {
      maxWave = std::max( maxWave, job.wave() );
      total += bands( job );
    }
  }
  if( total != count )
  {
    stats.sortedMismatched += std::max( total, count );
    if( stats.jobsReported < kMaxReports )
    {
      stats.jobsReported++;
      msg( WARNING, "GPU intra order: POC %d: %zu bands on the GPU, %zu on the CPU\n", cs.picture->poc, count, total );
    }
    return;
  }
  std::vector<uint32_t> first( size_t( maxWave ) + 2, 0 );
  for( const auto& ctu: cpu )
  {
    for( const IntraJob& job: ctu )
    {
      first[job.wave() + 1] += bands( job );
    }
  }
  for( int wave = 1; wave <= maxWave + 1; wave++ )
  {
    first[wave] += first[wave - 1];
  }
  std::vector<uint32_t> next( first.begin(), first.end() - 1 );
  for( const auto& ctu: cpu )
  {
    for( const IntraJob& job: ctu )
    {
      const int rows = bandRows( job );
      for( int row = 0; row < job.height(); row += rows )
      {
        const uint32_t at   = next[job.wave()]++;
        IntraJob       band = job;
        band.setRows( row, std::min( rows, job.height() - row ) );
        stats.sortedCompared++;
        if( memcmp( &band, gpu + size_t( at ) * 12, sizeof( IntraJob ) ) != 0 || waits[at] != first[job.wave()] )
        {
          stats.sortedMismatched++;
          if( stats.jobsReported < kMaxReports )
          {
            stats.jobsReported++;
            msg( WARNING, "GPU intra order: POC %d band %u (wave %d): GPU %08x %08x %08x wait %u, CPU %08x %08x %08x wait %u\n", cs.picture->poc, at,
                 job.wave(), gpu[at * 12], gpu[at * 12 + 1], gpu[at * 12 + 7], waits[at], band.w[0], band.w[1], band.w[7], first[job.wave()] );
          }
        }
      }
    }
  }
}

// Compares the deblocking edges edges.slang listed (2 words each, capacity of them) with the CTU tasks' lists.
// lists holds each of the four lists' first edge and count.
static void compareEdges( const CodingStructure& cs, const uint32_t* gpu, const uint32_t* lists, size_t capacity, const std::vector<EdgeLists>& cpu,
                          FrontEndStats& stats )
{
  for( int l = 0; l < 4; l++ )
  {
    size_t total = 0;
    for( const EdgeLists& ctu: cpu )
    {
      total += ctu.lists[l].size();
    }
    if( total != lists[2 * l + 1] )
    {
      stats.edgesMismatched += std::max<size_t>( total, lists[2 * l + 1] );
      if( stats.edgesReported < kMaxReports )
      {
        stats.edgesReported++;
        msg( WARNING, "GPU edges: POC %d list %d: %u on the GPU, %zu on the CPU\n", cs.picture->poc, l, lists[2 * l + 1], total );
      }
      continue;
    }
    size_t at = lists[2 * l];
    for( size_t ctu = 0; ctu < cpu.size(); ctu++ )
    {
      for( const Edge& e: cpu[ctu].lists[l] )
      {
        stats.edgesCompared++;
        if( at >= capacity || gpu[2 * at] != e.xy || gpu[2 * at + 1] != e.params )
        {
          stats.edgesMismatched++;
          if( stats.edgesReported < kMaxReports )
          {
            stats.edgesReported++;
            msg( WARNING, "GPU edges: POC %d CTU %zu list %d edge %zu: GPU %08x %08x, CPU %08x %08x\n", cs.picture->poc, ctu, l, at,
                 at < capacity ? gpu[2 * at] : 0u, at < capacity ? gpu[2 * at + 1] : 0u, e.xy, e.params );
          }
        }
        at++;
      }
    }
  }
}

void GpuRecon::compareFrontEnd( const CodingStructure& cs, const MotionInfo* cpu, const LoopFilterParam* lfp, const PictureJobs* jobs )
{
  compareMotion( cs, reinterpret_cast<const int32_t*>( m_check.motion.mapped ), cpu, m_check.stats );
  if( m_check.lfpRecorded && lfp )
  {
    compareLfp( cs, reinterpret_cast<const uint32_t*>( m_check.lfp.mapped ), lfp, m_check.stats );
  }
  m_check.lfpRecorded = false;
  if( m_check.jobsRecorded && jobs )
  {
    // jobs.slang's totals of the residual, inter (its three parts) and intra jobs, after its blocks' counts
    const uint32_t* lists     = reinterpret_cast<const uint32_t*>( m_check.jobBlocks.mapped ) + size_t( m_frontEnd.jobBlockCount ) * 5;
    const uint32_t  totals[3] = { lists[0], lists[1] + lists[3] + lists[4], lists[2] };
    const auto      count     = [&]( int k ) { return std::min<size_t>( totals[k], m_frontEnd.jobCapacity[k] ); };
    if( jobs->trafo )
    {
      compareJobs( cs, 0, reinterpret_cast<const uint32_t*>( m_check.jobs[0].mapped ), totals[0] > m_frontEnd.jobCapacity[0] ? totals[0] : count( 0 ),
                   *jobs->trafo, m_check.stats );
    }
    if( jobs->inter )
    {
      // the CTU tasks' inter jobs in recordInter's order: the regular jobs, then BDOF's, then the affine pairs
      std::vector<std::vector<InterJob>> parts( 1 );
      for( int part = 0; part < 3; part++ )
      {
        for( const auto& ctu: *jobs->inter )
        {
          for( size_t i = 0; i < ctu.size(); i++ )
          {
            const uint32_t flags = ctu[i].w[1] >> 16;
            const int      in    = flags & InterJob::kAffine ? 2 : flags & InterJob::kBdof ? 1 : 0;
            if( in == part )
            {
              parts[0].push_back( ctu[i] );
              if( flags & InterJob::kAffine )
              {
                parts[0].push_back( ctu[i + 1] );
              }
            }
            i += flags & InterJob::kAffine ? 1 : 0;
          }
        }
      }
      compareJobs( cs, 1, reinterpret_cast<const uint32_t*>( m_check.jobs[1].mapped ), totals[1] > m_frontEnd.jobCapacity[1] ? totals[1] : count( 1 ),
                   parts, m_check.stats );
    }
    if( jobs->intra )
    {
      compareJobs( cs, 2, reinterpret_cast<const uint32_t*>( m_check.jobs[2].mapped ), totals[2] > m_frontEnd.jobCapacity[2] ? totals[2] : count( 2 ),
                   *jobs->intra, m_check.stats );
      // intrasort.slang's band count, and its flag for a wave past 65535
      const uint32_t* info = reinterpret_cast<const uint32_t*>( m_check.sortInfo.mapped );
      if( info[2] != 0 )
      {
        msg( WARNING, "GPU intra order: POC %d: a wave past 65535\n", cs.picture->poc );
        m_check.stats.sortedMismatched++;
      }
      else
      {
        compareSorted( cs, reinterpret_cast<const uint32_t*>( m_check.sortedBands.mapped ), reinterpret_cast<const uint32_t*>( m_check.bandWaits.mapped ),
                       std::min<size_t>( info[0], m_frontEnd.sortCapacity ), *jobs->intra, m_check.stats );
      }
    }
  }
  m_check.jobsRecorded = false;
  if( m_check.edgesRecorded && jobs && jobs->edges )
  {
    compareEdges( cs, reinterpret_cast<const uint32_t*>( m_check.edges.mapped ), reinterpret_cast<const uint32_t*>( m_check.edgeLists.mapped ),
                  m_frontEnd.edgeCapacity, *jobs->edges, m_check.stats );
  }
  m_check.edgesRecorded = false;
}

}   // namespace gpu
}   // namespace vvdec
