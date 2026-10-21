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

/** \file     CodingStructure.h
 *  \brief    A class managing the coding information for a specific image part
 */

#pragma once

#include "Rom.h"
#include "Unit.h"
#include "Buffer.h"
#include "CommonDef.h"
#include "UnitPartitioner.h"
#include "Slice.h"

#include <vector>
#include <mutex>

namespace vvdec
{

struct Picture;

enum PictureType
{
  PIC_RECONSTRUCTION,
  PIC_RECON_WRAP,
  NUM_PIC_TYPES
};

#define NUM_PARTS_IN_CTU ( MAX_CU_SIZE * MAX_CU_SIZE ) >> (   MIN_CU_LOG2       << 1 )
// num collocated motion
#define NUM_COMOT_IN_CTU ( MAX_CU_SIZE * MAX_CU_SIZE ) >> ( ( MIN_CU_LOG2 + 1 ) << 1 )


struct CtuAlfData
{
  uint8_t ccAlfFilterControl[MAX_NUM_COMPONENT - 1];
  uint8_t alfCtuEnableFlag  [MAX_NUM_COMPONENT];
  uint8_t alfCtuAlternative [MAX_NUM_COMPONENT - 1];
  short   alfCtbFilterIndex;

  CtuAlfData() : ccAlfFilterControl{ 0, 0 }, alfCtuEnableFlag{ 0, 0, 0 } {}
};

struct CtuData
{
  SAOBlkParam           saoParam;
  CtuAlfData            alfParam;
  const Slice*          slice;
  const PPS*            pps;
  const SPS*            sps;
  const PicHeader*      ph;
  int                   lineIdx, colIdx, ctuIdx;

  CodingUnit           *firstCU, *lastCU;
  unsigned              numCUs, numTUs;

  ptrdiff_t             predBufOffset;
  ptrdiff_t             dmvrMvCacheOffset;

  CodingUnit**          cuPtr  [MAX_NUM_CHANNEL_TYPE];
  LoopFilterParam*      lfParam[NUM_EDGE_DIR];
  MotionInfo*           motion;
  ColocatedMotionInfo*  colMotion;
};
// ---------------------------------------------------------------------------
// coding structure
// ---------------------------------------------------------------------------

// The CU maps of the pictures' coding structures (a CodingUnit pointer per
// 4x4 block, luma and chroma: 8 MB at 4K), kept for the next pictures. A
// picture gives its map back zeroed once it is reconstructed
// (deallocTempInternals, on a worker thread), and the next picture set up
// takes it (allocTempInternals). Don't allocate a map anew per picture: its
// fresh pages are zeroed on the parser's thread, which takes most of a
// picture's set-up (4K: 4.4 ms).
class CuMapCache
{
public:
  std::unique_ptr<CodingUnit*[]> get( ptrdiff_t size )   // zeroed
  {
    {
      std::lock_guard<std::mutex> lock( m_mutex );
      if( size == m_size && !m_maps.empty() )
      {
        std::unique_ptr<CodingUnit*[]> map = std::move( m_maps.back() );
        m_maps.pop_back();
        return map;
      }
    }
    return std::make_unique<CodingUnit*[]>( size );
  }
  void put( std::unique_ptr<CodingUnit*[]> map, ptrdiff_t size )   // zeroed
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    if( size != m_size )
    {
      m_maps.clear();   // of another picture size
      m_size = size;
    }
    m_maps.push_back( std::move( map ) );
  }

  // The CTU data of the pictures the GPU parses (CodingStructure::attachCtuData), kept the same way: zeroed
  std::unique_ptr<CtuData[]> getCtuData( size_t count )
  {
    {
      std::lock_guard<std::mutex> lock( m_mutex );
      if( count == m_ctuCount && !m_ctuData.empty() )
      {
        std::unique_ptr<CtuData[]> data = std::move( m_ctuData.back() );
        m_ctuData.pop_back();
        return data;
      }
    }
    return std::make_unique<CtuData[]>( count );
  }
  void putCtuData( std::unique_ptr<CtuData[]> data, size_t count )   // (zeroed here)
  {
    memset( NO_WARNING_class_memaccess( data.get() ), 0, sizeof( CtuData ) * count );
    std::lock_guard<std::mutex> lock( m_mutex );
    if( count != m_ctuCount )
    {
      m_ctuData.clear();   // of another picture size
      m_ctuCount = count;
    }
    m_ctuData.push_back( std::move( data ) );
  }

private:
  std::mutex                                  m_mutex;
  ptrdiff_t                                   m_size = 0;
  std::vector<std::unique_ptr<CodingUnit*[]>> m_maps;
  size_t                                      m_ctuCount = 0;
  std::vector<std::unique_ptr<CtuData[]>>     m_ctuData;
};

class CodingStructure
{
public:

  UnitArea         area;

  Picture         *picture;

  UnitScale        unitScale[MAX_NUM_COMPONENT];
  int              chromaQpAdj;
  std::shared_ptr<const VPS> vps;
  std::shared_ptr<const SPS> sps;
  std::shared_ptr<const PPS> pps;
  std::shared_ptr<PicHeader> picHeader;
  std::shared_ptr<const APS> alfApss[ALF_CTB_MAX_NUM_APS];
  std::shared_ptr<const APS> lmcsAps;
  const PreCalcValues*       pcv;

  std::unique_ptr<CtuData[]> m_ctuData;
  size_t                     m_ctuDataSize;
  // A picture the GPU parses has its CTU data only while it is reconstructed (attachCtuData to detachCtuData: from
  // m_cuMapCache's pool); until then the CTUs its slices claimed are marked here (DecSlice::claimSlice)
  bool                       m_ctuDataPooled = false;
  std::vector<uint8_t>       m_ctuClaimed;
  bool                       hasCtuData() const { return m_ctuData != nullptr; }
  void                       attachCtuData();   // the CTU data, from the pool, with each CTU's slice
  void                       detachCtuData();   // back to the pool (nothing when not from it)

  // data for which memory is partially borrowed from DecLibRecon
  Pel*              m_predBuf;
  Mv*               m_dmvrMvCache;
  // end of partially borrowed data
  
  CodingStructure( CUChunkCache* cuChunkCache, TUChunkCache* tuChunkCache, CuMapCache* cuMapCache = nullptr );
  ~CodingStructure() { destroy(); }

  void create(const UnitArea &_unit);
  void create(const ChromaFormat &_chromaFormat, const Area& _area);
  void destroy();

  void resetForUse();
  void rebindPicBufs();

  // ---------------------------------------------------------------------------
  // global accessors
  // ---------------------------------------------------------------------------

#if _DEBUG
  const CodingUnit*     getCU(Position pos, ChannelType _chType) const
  {
    if( area.blocks[_chType].contains( pos ) )
    {
      int rsAddr = ctuRsAddr( pos, _chType );
      int inCtu  = inCtuPos ( pos, _chType );
      return getCtuData( rsAddr ).cuPtr[_chType][inCtu];
    }
    else return nullptr;
  }

  CodingUnit*     getCU(Position pos, ChannelType _chType)
  {
    if( area.blocks[_chType].contains( pos ) )
    {
      int rsAddr = ctuRsAddr( pos, _chType );
      int inCtu  = inCtuPos ( pos, _chType );
      return getCtuData( rsAddr ).cuPtr[_chType][inCtu];
    }
    else return nullptr;
  }
#else
  const CodingUnit*     getCU(Position pos, ChannelType _chType) const { if( area.blocks[_chType].contains( pos ) ) return getCtuData( ctuRsAddr( pos, _chType ) ).cuPtr[_chType][inCtuPos( pos, _chType )]; else return nullptr; }
        CodingUnit*     getCU(Position pos, ChannelType _chType)       { if( area.blocks[_chType].contains( pos ) ) return getCtuData( ctuRsAddr( pos, _chType ) ).cuPtr[_chType][inCtuPos( pos, _chType )]; else return nullptr; }
#endif

  const CodingUnit*     getCURestricted(const Position &pos, const Position curPos, const unsigned curSliceIdx, const unsigned curTileIdx, const ChannelType _chType) const;
  const CodingUnit*     getCURestricted(const Position &pos, const CodingUnit& curCu,                                                      const ChannelType _chType, const CodingUnit* guess = nullptr) const;

  CodingUnit&     addCU(const UnitArea &unit, const ChannelType _chType, const TreeType treeType, const ModeType modeType, const CodingUnit* cuLeft, const CodingUnit* cuAbove );
  TransformUnit&  addTU(const UnitArea &unit, const ChannelType _chType, CodingUnit &cu);
  void            addEmptyTUs(Partitioner &partitioner, CodingUnit& cu);
  CUTraverser     traverseCUs(const int ctuRsAddr);

  // clearColMotion: the colocated motion too, else the picture's first parse
  // task clears it (clearColMotion(), off the parser's thread)
  void initStructData( bool clearColMotion = true );
  void initCtuData();   // sets each CTU's place, maps and offsets in m_ctuData, which must be zeroed before
  void clearColMotion();

  // cuMap: the CU map (a CodingUnit pointer a 4x4 block), which a picture the GPU parses never uses; colMotion: the
  // colocated motion, which a picture the GPU parses gets later (attachColMotion); ctuData: the CTU data, which a
  // picture the GPU parses gets later too (attachCtuData)
  void allocTempInternals( bool cuMap = true, bool colMotion = true, bool ctuData = true );
  void deallocTempInternals();

  void createInternals(const UnitArea& _unit);

  CUCache    m_cuCache;
  TUCache    m_tuCache;

  PelStorage m_reco;
  PelStorage m_rec_wrap;

  unsigned int         m_widthInCtus;
  PosType              m_ctuSizeMask[2];
  PosType              m_ctuWidthLog2[2];

  std::unique_ptr<CodingUnit*[]>         m_cuMap;
  ptrdiff_t                              m_cuMapSize;
  CuMapCache*                            m_cuMapCache;   // where m_cuMap comes from and goes back to (or null)
  bool                                   m_cuMapZero = false;   // m_cuMap zeroed (from the cache or new): initStructData need not
  ColocatedMotionInfo*                   m_colMiMap;
  ptrdiff_t                              m_colMiMapSize;
#if VVDEC_VULKAN
  // the colocated motion in memory from colMiAllocator, when it is set before
  // allocTempInternals: the GPU's, which writes DMVR's refinements there
  // (gpu::GpuRecon); m_colMiAllocation the allocator's handle of it
  const UserAllocator*                   colMiAllocator = nullptr;
  void*                                  m_colMiAllocation = nullptr;
  // the picture's slice table for the GPU's motion derivation
  // (gpu::writeSliceTable), in memory from colMiAllocator: its own, then as
  // the colocated picture's
  void*                                  m_gpuSliceAllocation = nullptr;
  uint32_t*                              m_gpuSliceTable      = nullptr;
  size_t                                 m_gpuSliceTableWords = 0;
  uint32_t*                              gpuSliceTable( size_t words );   // room for words of it
  void                                   freeGpuSliceTable();
  // VVDEC_GPU_PARSE=2: a picture the GPU parses has its colocated motion only while its work is on the GPU
  // (gpu::PlanePool): map, of colMotionEntries() entries, from colMiAllocator (allocation: its handle), set; or given
  // back (the handle; null when there is none)
  ptrdiff_t                              colMotionEntries() const;
  void                                   attachColMotion( ColocatedMotionInfo* map, void* allocation );
  void*                                  detachColMotion();
  // VVDEC_GPU_PARSE: each CTU's SAO and ALF parameters as the CPU's parse left them (gpu::captureSaoAlf, 7 words a CTU)
  std::vector<uint32_t>                  m_gpuSaoAlf;
#endif
  void                                   allocColMiMap( ptrdiff_t size );
  void                                   freeColMiMap();

public:

  // in CTU coordinates
  int ctuRsAddr( int col, int line ) const { return col + ( line * m_widthInCtus ); }
  // in sample coordinates
  int ctuRsAddr( Position pos, ChannelType chType ) const { Position posL = recalcPosition( area.chromaFormat, chType, CH_L, pos ); return ctuRsAddr( posL.x >> pcv->maxCUWidthLog2, posL.y >> pcv->maxCUHeightLog2 ); }
  // 4x4 luma block position within the CTU
  int inCtuPos ( Position pos, ChannelType chType ) const { return ( unitScale[chType].scaleHor( pos.x ) & m_ctuSizeMask[chType] ) + ( ( unitScale[chType].scaleVer( pos.y ) & m_ctuSizeMask[chType] ) << m_ctuWidthLog2[chType] ); }
  // 8x8 luma block position within the CTU
  int colMotPos( Position pos )                     const { return ( g_colMiScaling.scaleHor( pos.x ) & ( m_ctuSizeMask[CH_L] >> 1 ) ) + ( ( g_colMiScaling.scaleVer( pos.y ) & ( m_ctuSizeMask[CH_L] >> 1 ) ) << ( m_ctuWidthLog2[CH_L] - 1 ) ); }

        CtuData& getCtuData( int col, int line )       { return m_ctuData[ctuRsAddr( col, line )]; }
  const CtuData& getCtuData( int col, int line ) const { return m_ctuData[ctuRsAddr( col, line )]; }

        CtuData& getCtuData( int addr )       { return m_ctuData[addr]; }
  const CtuData& getCtuData( int addr ) const { return m_ctuData[addr]; }

  int m_IBCBufferWidth;
  std::vector<PelStorage> m_virtualIBCbuf;
  std::vector<char>       hasIbcBlock;
  void initVIbcBuf( int numCtuLines, ChromaFormat chromaFormatIDC, int ctuSize );
  void fillIBCbuffer( CodingUnit &cu, int lineIdx );

  MotionBuf getMotionBuf( const     Area& _area );
  MotionBuf getMotionBuf( const UnitArea& _area ) { return getMotionBuf( _area.Y() ); }

  const CMotionBuf getMotionBuf( const     Area& _area ) const;
  const CMotionBuf getMotionBuf( const UnitArea& _area ) const { return getMotionBuf( _area.Y() ); }

        MotionInfo& getMotionInfo( const Position& pos )       { return getCtuData( ctuRsAddr( pos, CH_L ) ).motion[inCtuPos( pos, CH_L )]; }
  const MotionInfo& getMotionInfo( const Position& pos ) const { return getCtuData( ctuRsAddr( pos, CH_L ) ).motion[inCtuPos( pos, CH_L )]; }
  

  const ColocatedMotionInfo& getColInfo( const Position &pos, const Slice*& pColSlice ) const;

  LoopFilterParam const* getLFPMapPtr   ( const DeblockEdgeDir edgeDir, ptrdiff_t _ctuRsAddr ) const { return m_ctuData[_ctuRsAddr].lfParam[edgeDir]; }
  LoopFilterParam      * getLFPMapPtr   ( const DeblockEdgeDir edgeDir, ptrdiff_t _ctuRsAddr )       { return m_ctuData[_ctuRsAddr].lfParam[edgeDir]; }
  ptrdiff_t              get4x4MapStride() const { return ( ptrdiff_t( 1 ) << m_ctuWidthLog2[CH_L] ); }

  UnitScale getScaling( const UnitScale::ScaliningType type, const ChannelType chType = CH_L ) const
  {
    return type == UnitScale::MI_MAP ? g_miScaling : unitScale[chType];
  }

public:

         PelBuf       getRecoBuf(const CompArea &blk)        { return m_reco.bufs[blk.compID()].subBuf( blk ); }
  const CPelBuf       getRecoBuf(const CompArea &blk) const  { return m_reco.bufs[blk.compID()].subBuf( blk ); }
         PelUnitBuf   getRecoBuf(const UnitArea &unit)       { return m_reco.subBuf( unit ); }
  const CPelUnitBuf   getRecoBuf(const UnitArea &unit) const { return m_reco.subBuf( unit ); }



  // reco buffer
         PelBuf       getRecoBuf(const ComponentID compID, bool wrap=false)         { return wrap ? m_rec_wrap.get(compID) : m_reco.get(compID); }
  const CPelBuf       getRecoBuf(const ComponentID compID, bool wrap=false)   const { return wrap ? m_rec_wrap.get(compID) : m_reco.get(compID); }
         PelUnitBuf   getRecoBuf(bool wrap=false)                                   { return wrap ? m_rec_wrap : m_reco; }
  const CPelUnitBuf   getRecoBuf(bool wrap=false)                             const { return wrap ? m_rec_wrap : m_reco; }

         PelUnitBuf   getPredBuf(const CodingUnit &cu);
  const CPelUnitBuf   getPredBuf(const CodingUnit &cu) const;
};

}
