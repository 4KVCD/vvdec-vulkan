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

/** \file     GpuRecon.cpp
    \brief    the stages of a picture's reconstruction that run on the GPU
*/

#include "GpuRecon.h"

#include "vvdec_gpu/GpuShaders.h"

#include "CommonLib/AdaptiveLoopFilter.h"
#include "CommonLib/CodingStructure.h"
#include "CommonLib/Picture.h"
#include "CommonLib/SampleAdaptiveOffset.h"
#include "CommonLib/Slice.h"

#include <chrono>
#include <cstring>
#include <utility>

namespace vvdec
{
namespace gpu
{

// ====================================================================================================================
// picture buffers
// ====================================================================================================================

static void* createHostPlane( void* opaque, vvdecComponentType, uint32_t size, uint32_t, void** allocator )
{
  Context* context = static_cast<Context*>( opaque );
  Buffer*  buf     = new Buffer( context->alloc( size, MemKind::HostCached ) );
  if( !buf->mapped )
  {
    context->free( *buf );
    delete buf;
    return nullptr;
  }
  *allocator = buf;
  return buf->mapped;
}

static void unrefHostPlane( void* opaque, void* allocator )
{
  Buffer* buf = static_cast<Buffer*>( allocator );
  static_cast<Context*>( opaque )->free( *buf );
  delete buf;
}

UserAllocator hostPictureAllocator( Context* context )
{
  return UserAllocator( createHostPlane, unrefHostPlane, context );
}

// the GPU address of a plane's sample (0, 0)
static VkDeviceAddress planeAddress( const PelStorage& storage, ComponentID comp )
{
  const Buffer* buf = static_cast<const Buffer*>( storage.getBufAllocator( comp ) );
  CHECK_FATAL( !buf || !storage.isExternAllocator(), "Vulkan: a picture buffer the GPU cannot reach" );
  const uint8_t* origin = reinterpret_cast<const uint8_t*>( storage.bufs[comp].buf );
  CHECK_FATAL( origin < buf->mapped || origin >= buf->mapped + buf->size, "Vulkan: a plane outside its buffer" );
  return buf->addr + VkDeviceAddress( origin - buf->mapped );
}

// ====================================================================================================================
// GpuRecon
// ====================================================================================================================

GpuRecon::GpuRecon( Context* context ) : m_ctx( context )
{
  const Api& vk = m_ctx->api();

  VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
  poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
  poolInfo.queueFamilyIndex = m_ctx->queueFamily();
  if( vk.vkCreateCommandPool( m_ctx->device(), &poolInfo, nullptr, &m_pool ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateCommandPool failed" );
  }

  VkCommandBufferAllocateInfo cmdInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
  cmdInfo.commandPool        = m_pool;
  cmdInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdInfo.commandBufferCount = 1;
  if( vk.vkAllocateCommandBuffers( m_ctx->device(), &cmdInfo, &m_cmd ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkAllocateCommandBuffers failed" );
  }

  VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
  if( vk.vkCreateFence( m_ctx->device(), &fenceInfo, nullptr, &m_fence ) != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: vkCreateFence failed" );
  }

  if( m_ctx->profiling() )
  {
    VkQueryPoolCreateInfo queryInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    queryInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = kMaxMarks;
    if( vk.vkCreateQueryPool( m_ctx->device(), &queryInfo, nullptr, &m_queries ) != VK_SUCCESS )
    {
      THROW_FATAL( "Vulkan: vkCreateQueryPool failed" );
    }
  }
}

GpuRecon::~GpuRecon()
{
  const Api& vk = m_ctx->api();
  if( m_queries )
  {
    vk.vkDestroyQueryPool( m_ctx->device(), m_queries, nullptr );
  }
  if( m_fence )
  {
    vk.vkDestroyFence( m_ctx->device(), m_fence, nullptr );
  }
  if( m_pool )
  {
    vk.vkDestroyCommandPool( m_ctx->device(), m_pool, nullptr );
  }
  m_ctx->free( m_params );
  for( auto& mirror: m_mirror )
  {
    for( Buffer& buf: mirror )
    {
      m_ctx->free( buf );
    }
  }
}

uint8_t* GpuRecon::params( size_t size )
{
  if( m_params.size < size )
  {
    m_ctx->free( m_params );
    m_params = m_ctx->alloc( size, MemKind::HostCached );
  }
  return m_params.mapped;
}

void GpuRecon::beginCommands()
{
  const Api& vk = m_ctx->api();
  vk.vkResetCommandPool( m_ctx->device(), m_pool, 0 );
  VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vk.vkBeginCommandBuffer( m_cmd, &begin );

  m_marks = 0;
  if( m_queries )
  {
    vk.vkCmdResetQueryPool( m_cmd, m_queries, 0, kMaxMarks );
    vk.vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, m_queries, m_marks++ );
  }
}

void GpuRecon::mark( const char* label )
{
  if( m_queries && m_marks < kMaxMarks )
  {
    m_labels[m_marks] = label;
    m_ctx->api().vkCmdWriteTimestamp( m_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_queries, m_marks++ );
  }
}

void GpuRecon::computeBarrier()
{
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  m_ctx->api().vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                                     nullptr );
}

void GpuRecon::submitAndWait()
{
  const Api& vk = m_ctx->api();

  // the shaders' writes, made visible to the CPU
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
  vk.vkEndCommandBuffer( m_cmd );

  vk.vkResetFences( m_ctx->device(), 1, &m_fence );
  const auto start = std::chrono::steady_clock::now();
  m_ctx->submit( m_cmd, m_fence );
  const VkResult result = vk.vkWaitForFences( m_ctx->device(), 1, &m_fence, VK_TRUE, UINT64_MAX );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: waiting for the GPU failed (" << result << ")" );
  }

  if( m_queries && m_marks > 1 )
  {
    m_ctx->addTime( "wait (CPU)", std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now() - start ).count() );
    uint64_t stamps[kMaxMarks];
    vk.vkGetQueryPoolResults( m_ctx->device(), m_queries, 0, m_marks, sizeof( stamps ), stamps, sizeof( uint64_t ),
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT );
    const double toMs = m_ctx->timestampPeriod() * 1e-6;
    for( uint32_t i = 1; i < m_marks; i++ )
    {
      m_ctx->addTime( m_labels[i], double( stamps[i] - stamps[i - 1] ) * toMs );
    }
    m_ctx->addTime( "submission (GPU)", double( stamps[m_marks - 1] - stamps[0] ) * toMs );
  }
}

// ====================================================================================================================
// SAO and ALF
// ====================================================================================================================

// the push constants of shaders/alf.slang
struct AlfPush
{
  VkDeviceAddress src0, dst0, src1, dst1, srcLuma, ctus, coeff, clip, ccCoeff;
  int32_t         srcStride, dstStride, lumaStride, width, height, lumaWidth, lumaHeight, ctuLog2, widthInCtus, bitDepth, scaleX, scaleY;
};

// the push constants of shaders/sao.slang
struct SaoPush
{
  VkDeviceAddress src, dst, ctus;
  int32_t         srcStride, dstStride, width, height, ctuWidth, ctuHeight, widthInCtus, comp, bitDepth;
};

// the ALF parameters: the filter tables, luma sets [16 fixed + 8 APS][transpose][class][13],
// chroma [APS][alternative][7], CC-ALF [APS][component][filter][8], then 8 bytes per CTU
constexpr size_t kAlfSetSize     = MAX_NUM_ALF_TRANSPOSE_ID * MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF;
constexpr size_t kAlfChromaSize  = ALF_CTB_MAX_NUM_APS * MAX_NUM_ALF_ALTERNATIVES_CHROMA * MAX_NUM_ALF_CHROMA_COEFF;
constexpr size_t kAlfCcSize      = ALF_CTB_MAX_NUM_APS * 2 * MAX_NUM_CC_ALF_FILTERS * MAX_NUM_CC_ALF_CHROMA_COEFF;
constexpr size_t kAlfLumaCoeff   = 0;
constexpr size_t kAlfLumaClip    = kAlfLumaCoeff + NUM_TOTAL_FILTER_SETS * kAlfSetSize * sizeof( int16_t );
constexpr size_t kAlfChromaCoeff = kAlfLumaClip + NUM_TOTAL_FILTER_SETS * kAlfSetSize * sizeof( int16_t );
constexpr size_t kAlfChromaClip  = kAlfChromaCoeff + kAlfChromaSize * sizeof( int16_t );
constexpr size_t kAlfCcCoeff     = kAlfChromaClip + kAlfChromaSize * sizeof( int16_t );
constexpr size_t kAlfCtus        = ( kAlfCcCoeff + kAlfCcSize * sizeof( int16_t ) + 255 ) & ~size_t( 255 );
static_assert( kAlfSetSize == 4 * 25 * 13, "the layout of shaders/alf.slang" );

bool GpuRecon::alfSupported( const CodingStructure& cs )
{
  if( cs.picHeader->getVirtualBoundariesPresentFlag() )
  {
    return false;
  }
  if( cs.sps->getSubPicInfoPresentFlag() )
  {
    for( int i = 0; i < cs.pps->getNumSubPics(); i++ )
    {
      if( !cs.pps->getSubPic( i ).getloopFilterAcrossSubPicEnabledFlag() )
      {
        return false;
      }
    }
  }
  if( cs.pps->getNumTiles() > 1 && !cs.pps->getLoopFilterAcrossTilesEnabledFlag() )
  {
    return false;
  }
  if( !cs.pps->getLoopFilterAcrossSlicesEnabledFlag() )
  {
    for( const Slice* slice: cs.picture->slices )
    {
      if( slice->getNumCtuInSlice() != cs.pcv->sizeInCtus )
      {
        return false;
      }
    }
  }
  return true;
}

bool GpuRecon::saoSupported( const CodingStructure& cs )
{
  return !cs.picHeader->getVirtualBoundariesPresentFlag();
}

size_t GpuRecon::alfParamSize( const CodingStructure& cs )
{
  return ( kAlfCtus + size_t( cs.pcv->sizeInCtus ) * 8 + 255 ) & ~size_t( 255 );
}

size_t GpuRecon::saoParamSize( const CodingStructure& cs )
{
  return ( size_t( cs.pcv->sizeInCtus ) * 3 * 8 * sizeof( int16_t ) + 255 ) & ~size_t( 255 );
}

void GpuRecon::prepareAlf( const CodingStructure& cs, const AdaptiveLoopFilter& alfFilter, uint8_t* base )
{
  const PreCalcValues& pcv = *cs.pcv;

  int16_t* lumaCoeff   = reinterpret_cast<int16_t*>( base + kAlfLumaCoeff );
  int16_t* lumaClip    = reinterpret_cast<int16_t*>( base + kAlfLumaClip );
  int16_t* chromaCoeff = reinterpret_cast<int16_t*>( base + kAlfChromaCoeff );
  int16_t* chromaClip  = reinterpret_cast<int16_t*>( base + kAlfChromaClip );
  int16_t* ccCoeff     = reinterpret_cast<int16_t*>( base + kAlfCcCoeff );
  uint8_t* ctus        = base + kAlfCtus;

  for( int set = 0; set < NUM_FIXED_FILTER_SETS; set++ )
  {
    memcpy( lumaCoeff + set * kAlfSetSize, alfFilter.getFixedFilterSetCoeff( set ), kAlfSetSize * sizeof( int16_t ) );
    memcpy( lumaClip  + set * kAlfSetSize, alfFilter.getClipDefault(),               kAlfSetSize * sizeof( int16_t ) );
  }
  for( const Slice* slice: cs.picture->slices )
  {
    if( slice->getAlfEnabledFlag( COMPONENT_Y ) )
    {
      for( int i = 0; i < slice->getNumAlfAps(); i++ )
      {
        const int  apsId = slice->getAlfApsIdsLuma()[i];
        const APS* aps   = slice->getAlfAPS( apsId );
        CHECK( aps == nullptr, "invalid APS" );
        const AlfSliceParam& alfParam = aps->getAlfAPSParam();
        memcpy( lumaCoeff + ( NUM_FIXED_FILTER_SETS + apsId ) * kAlfSetSize, alfParam.lumaCoeffFinal, kAlfSetSize * sizeof( int16_t ) );
        memcpy( lumaClip  + ( NUM_FIXED_FILTER_SETS + apsId ) * kAlfSetSize, alfParam.lumaClippFinal, kAlfSetSize * sizeof( int16_t ) );
      }
    }
    if( slice->getAlfEnabledFlag( COMPONENT_Cb ) || slice->getAlfEnabledFlag( COMPONENT_Cr ) )
    {
      const int  apsId = slice->getAlfApsIdChroma();
      const APS* aps   = slice->getAlfAPS( apsId );
      CHECK( aps == nullptr, "invalid APS" );
      const AlfSliceParam& alfParam = aps->getAlfAPSParam();
      constexpr size_t     perAps   = MAX_NUM_ALF_ALTERNATIVES_CHROMA * MAX_NUM_ALF_CHROMA_COEFF;
      memcpy( chromaCoeff + apsId * perAps, alfParam.chromaCoeff,    perAps * sizeof( int16_t ) );
      memcpy( chromaClip  + apsId * perAps, alfParam.chrmClippFinal, perAps * sizeof( int16_t ) );
    }
    for( int comp = 1; comp < 3; comp++ )
    {
      if( slice->getCcAlfEnabledFlag( comp ) )
      {
        const int  apsId = comp == 1 ? slice->getCcAlfCbApsId() : slice->getCcAlfCrApsId();
        const APS* aps   = slice->getAlfAPS( apsId );
        CHECK( aps == nullptr, "invalid APS" );
        constexpr size_t perComp = MAX_NUM_CC_ALF_FILTERS * MAX_NUM_CC_ALF_CHROMA_COEFF;
        memcpy( ccCoeff + ( apsId * 2 + comp - 1 ) * perComp, aps->getCcAlfAPSParam().ccAlfCoeff[comp - 1], perComp * sizeof( int16_t ) );
      }
    }
  }

  for( int ctuIdx = 0; ctuIdx < (int) pcv.sizeInCtus; ctuIdx++ )
  {
    const CtuData&    ctuData = cs.getCtuData( ctuIdx );
    const Slice*      slice   = ctuData.cuPtr[0][0]->slice;
    const CtuAlfData& alf     = ctuData.alfParam;
    uint8_t*          ctu     = ctus + ctuIdx * 8;

    ctu[0] = 255;
    if( alf.alfCtuEnableFlag[COMPONENT_Y] )
    {
      const int index = alf.alfCtbFilterIndex;
      ctu[0] = uint8_t( index < NUM_FIXED_FILTER_SETS ? index : NUM_FIXED_FILTER_SETS + slice->getAlfApsIdsLuma()[index - NUM_FIXED_FILTER_SETS] );
    }
    ctu[1] = uint8_t( slice->getAlfApsIdChroma() & ( ALF_CTB_MAX_NUM_APS - 1 ) );
    ctu[2] = alf.alfCtuEnableFlag[COMPONENT_Cb] ? alf.alfCtuAlternative[0] : 255;
    ctu[3] = alf.alfCtuEnableFlag[COMPONENT_Cr] ? alf.alfCtuAlternative[1] : 255;
    ctu[4] = slice->getCcAlfEnabledFlag( 1 ) && alf.ccAlfFilterControl[0] ? uint8_t( slice->getCcAlfCbApsId() * 4 + alf.ccAlfFilterControl[0] - 1 ) : 255;
    ctu[5] = slice->getCcAlfEnabledFlag( 2 ) && alf.ccAlfFilterControl[1] ? uint8_t( slice->getCcAlfCrApsId() * 4 + alf.ccAlfFilterControl[1] - 1 ) : 255;
    ctu[6] = 0;
    ctu[7] = 0;
  }
}

void GpuRecon::recordAlf( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params )
{
  const PreCalcValues& pcv = *cs.pcv;
  const Api&           vk  = m_ctx->api();

  AlfPush push{};
  push.ctus        = params + kAlfCtus;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  push.bitDepth    = cs.sps->getBitDepth();
  push.lumaWidth   = pcv.lumaWidth;
  push.lumaHeight  = pcv.lumaHeight;
  push.lumaStride  = src.stride[COMPONENT_Y];

  // luma
  push.src0      = src.addr[COMPONENT_Y];
  push.dst0      = dst.addr[COMPONENT_Y];
  push.coeff     = params + kAlfLumaCoeff;
  push.clip      = params + kAlfLumaClip;
  push.srcStride = src.stride[COMPONENT_Y];
  push.dstStride = dst.stride[COMPONENT_Y];
  push.width     = pcv.lumaWidth;
  push.height    = pcv.lumaHeight;
  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_alf_luma ) );
  vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
  vk.vkCmdDispatch( m_cmd, ( ( push.width + 3 ) / 4 + 7 ) / 8, ( ( push.height + 3 ) / 4 + 7 ) / 8, 1 );
  mark( "alf luma" );

  if( isChromaEnabled( pcv.chrFormat ) )
  {
    CHECK( src.stride[COMPONENT_Cb] != src.stride[COMPONENT_Cr] || dst.stride[COMPONENT_Cb] != dst.stride[COMPONENT_Cr],
           "the chroma planes differ in stride" );
    push.src0      = src.addr[COMPONENT_Cb];
    push.dst0      = dst.addr[COMPONENT_Cb];
    push.src1      = src.addr[COMPONENT_Cr];
    push.dst1      = dst.addr[COMPONENT_Cr];
    push.srcLuma   = src.addr[COMPONENT_Y];
    push.coeff     = params + kAlfChromaCoeff;
    push.clip      = params + kAlfChromaClip;
    push.ccCoeff   = params + kAlfCcCoeff;
    push.srcStride = src.stride[COMPONENT_Cb];
    push.dstStride = dst.stride[COMPONENT_Cb];
    push.scaleX    = getComponentScaleX( COMPONENT_Cb, pcv.chrFormat );
    push.scaleY    = getComponentScaleY( COMPONENT_Cb, pcv.chrFormat );
    push.width     = pcv.lumaWidth >> push.scaleX;
    push.height    = pcv.lumaHeight >> push.scaleY;
    vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_alf_chroma ) );
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    vk.vkCmdDispatch( m_cmd, ( ( push.width + 3 ) / 4 + 7 ) / 8, ( ( push.height + 3 ) / 4 + 7 ) / 8, 1 );
    mark( "alf chroma" );
  }
}

bool GpuRecon::prepareSao( CodingStructure& cs, SampleAdaptiveOffset& sao, uint8_t* base )
{
  const int numCtus = (int) cs.pcv->sizeInCtus;
  const int numComp = (int) getNumberValidComponents( cs.pcv->chrFormat );
  int16_t*  ctus    = reinterpret_cast<int16_t*>( base );
  bool      any     = false;

  // in raster order: a merged CTU copies the parameters of its left or above neighbour
  for( int ctuIdx = 0; ctuIdx < numCtus; ctuIdx++ )
  {
    sao.reconstructCtuParam( cs, ctuIdx );
  }

  for( int ctuIdx = 0; ctuIdx < numCtus; ctuIdx++ )
  {
    const SAOBlkParam& param = cs.getCtuData( ctuIdx ).saoParam;
    int16_t            avail = -1;   // derived when a component of the CTU uses SAO
    for( int comp = 0; comp < 3; comp++ )
    {
      int16_t*         ctu    = ctus + ( ctuIdx * 3 + comp ) * 8;
      const SAOOffset& offset = param[comp];
      memset( ctu, 0, 8 * sizeof( int16_t ) );
      ctu[0] = -1;
      if( comp >= numComp || offset.modeIdc == SAO_MODE_OFF )
      {
        continue;
      }
      CHECK( offset.modeIdc != SAO_MODE_NEW || offset.typeIdc < SAO_TYPE_START_EO || offset.typeIdc > SAO_TYPE_BO, "unresolved SAO parameters" );
      if( avail < 0 )
      {
        avail = sao.neighbourAvailability( cs, ctuIdx );
      }
      any    = true;
      ctu[0] = int16_t( offset.typeIdc );
      ctu[2] = avail;
      if( offset.typeIdc == SAO_TYPE_BO )
      {
        ctu[1] = int16_t( offset.typeAuxInfo );
        for( int i = 0; i < 4; i++ )
        {
          ctu[4 + i] = int16_t( offset.offset[( offset.typeAuxInfo + i ) % NUM_SAO_BO_CLASSES] );
        }
      }
      else
      {
        ctu[4] = int16_t( offset.offset[0] );
        ctu[5] = int16_t( offset.offset[1] );
        ctu[6] = int16_t( offset.offset[3] );
        ctu[7] = int16_t( offset.offset[4] );
      }
    }
  }
  return any;
}

void GpuRecon::recordSao( const CodingStructure& cs, const Planes& src, const Planes& dst, VkDeviceAddress params )
{
  const PreCalcValues& pcv = *cs.pcv;
  const Api&           vk  = m_ctx->api();

  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_sao ) );
  for( int comp = 0; comp < (int) getNumberValidComponents( pcv.chrFormat ); comp++ )
  {
    const ComponentID compID = ComponentID( comp );
    SaoPush           push{};
    push.src         = src.addr[compID];
    push.dst         = dst.addr[compID];
    push.ctus        = params;
    push.srcStride   = src.stride[compID];
    push.dstStride   = dst.stride[compID];
    push.width       = pcv.lumaWidth >> getComponentScaleX( compID, pcv.chrFormat );
    push.height      = pcv.lumaHeight >> getComponentScaleY( compID, pcv.chrFormat );
    push.ctuWidth    = pcv.maxCUWidth >> getComponentScaleX( compID, pcv.chrFormat );
    push.ctuHeight   = pcv.maxCUHeight >> getComponentScaleY( compID, pcv.chrFormat );
    push.widthInCtus = pcv.widthInCtus;
    push.comp        = comp;
    push.bitDepth    = cs.sps->getBitDepth();
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    vk.vkCmdDispatch( m_cmd, ( push.width + 15 ) / 16, ( push.height + 15 ) / 16, 1 );
  }
  mark( "sao" );
}

GpuRecon::Planes GpuRecon::mirrorPlanes( int mirror, const PelStorage& pic )
{
  Planes planes{};
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    const Buffer* host = static_cast<const Buffer*>( pic.getBufAllocator( comp ) );
    Buffer&       dev  = m_mirror[mirror][comp];
    if( dev.size != host->size )
    {
      m_ctx->free( dev );
      dev = m_ctx->alloc( size_t( host->size ), MemKind::Device );
    }
    planes.addr[comp]   = dev.addr + ( planeAddress( pic, ComponentID( comp ) ) - host->addr );
    planes.stride[comp] = int32_t( pic.bufs[comp].stride );
  }
  return planes;
}

void GpuRecon::copyPlanes( const PelStorage& pic, int mirror, bool toMirror )
{
  const Api& vk = m_ctx->api();
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    const Buffer* host = static_cast<const Buffer*>( pic.getBufAllocator( comp ) );
    const Buffer& dev  = m_mirror[mirror][comp];
    VkBufferCopy  region{ 0, 0, host->size };
    vk.vkCmdCopyBuffer( m_cmd, toMirror ? host->buffer : dev.buffer, toMirror ? dev.buffer : host->buffer, 1, &region );
  }
  if( toMirror )
  {
    // the shaders after an upload wait for it
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
    mark( "upload" );
  }
  else
  {
    mark( "download" );
  }
}

void GpuRecon::loopFilters( CodingStructure& cs, SampleAdaptiveOffset* sao, const AdaptiveLoopFilter* alf, PelStorage& pic )
{
  const size_t alfSize = alf ? alfParamSize( cs ) : 0;
  const size_t saoSize = sao ? saoParamSize( cs ) : 0;
  uint8_t*     base    = params( alfSize + saoSize );

  const bool doSao = sao && prepareSao( cs, *sao, base + alfSize );
  if( alf )
  {
    prepareAlf( cs, *alf, base );
  }
  if( !doSao && !alf )
  {
    return;
  }

  // in the GPU's own memory, which its shaders reach faster than the host's:
  // the picture goes in, is filtered there, and comes back
  beginCommands();
  const Planes planes[2] = { mirrorPlanes( 0, pic ), mirrorPlanes( 1, pic ) };
  int          cur       = 0;
  copyPlanes( pic, 0, true );
  if( doSao )
  {
    recordSao( cs, planes[cur], planes[1 - cur], m_params.addr + alfSize );
    cur = 1 - cur;
  }
  if( alf )
  {
    if( doSao )
    {
      computeBarrier();
    }
    recordAlf( cs, planes[cur], planes[1 - cur], m_params.addr );
    cur = 1 - cur;
  }
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  m_ctx->api().vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                                     nullptr );
  copyPlanes( pic, cur, false );
  submitAndWait();
}

}   // namespace gpu
}   // namespace vvdec
