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

// a plane of a picture: in host memory the GPU reaches, where vvdec's CPU
// code reads and writes it, and its twin in the GPU's own memory, where the
// GPU reconstructs the picture and keeps it as a reference (made when first used)
struct HostPlane
{
  Buffer host;
  Buffer twin;
};

static void* createHostPlane( void* opaque, vvdecComponentType, uint32_t size, uint32_t, void** allocator )
{
  Context*   context = static_cast<Context*>( opaque );
  HostPlane* plane   = new HostPlane;
  plane->host        = context->alloc( size, MemKind::HostCached );
  if( !plane->host.mapped )
  {
    context->free( plane->host );
    delete plane;
    return nullptr;
  }
  *allocator = plane;
  return plane->host.mapped;
}

static void unrefHostPlane( void* opaque, void* allocator )
{
  HostPlane* plane = static_cast<HostPlane*>( allocator );
  static_cast<Context*>( opaque )->free( plane->host );
  static_cast<Context*>( opaque )->free( plane->twin );
  delete plane;
}

static HostPlane* hostPlane( const PelStorage& storage, ComponentID comp )
{
  HostPlane* plane = static_cast<HostPlane*>( storage.getBufAllocator( comp ) );
  CHECK_FATAL( !plane || !storage.isExternAllocator(), "Vulkan: a picture buffer the GPU cannot reach" );
  return plane;
}

UserAllocator hostPictureAllocator( Context* context )
{
  return UserAllocator( createHostPlane, unrefHostPlane, context );
}

// where a plane's sample (0, 0) is in its buffer
static VkDeviceSize planeOffset( const PelStorage& storage, ComponentID comp )
{
  const Buffer&  host   = hostPlane( storage, comp )->host;
  const uint8_t* origin = reinterpret_cast<const uint8_t*>( storage.bufs[comp].buf );
  CHECK_FATAL( origin < host.mapped || origin >= host.mapped + host.size, "Vulkan: a plane outside its buffer" );
  return VkDeviceSize( origin - host.mapped );
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
  m_ctx->free( m_lfp );
  m_ctx->free( m_dmvr );
  m_ctx->free( m_intraJobs );
  m_ctx->free( m_interJobs );
  for( Buffer& buf: m_scratch )
  {
    m_ctx->free( buf );
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

  // what earlier submissions wrote (the reference pictures), visible to this one
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
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

  // the writes of the shaders and the copies, made visible to the CPU
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vk.vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                           nullptr, 0, nullptr );
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

GpuRecon::Planes GpuRecon::twinPlanes( const PelStorage& pic )
{
  Planes planes{};
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    HostPlane* plane = hostPlane( pic, ComponentID( comp ) );
    if( !plane->twin.valid() )
    {
      plane->twin = m_ctx->alloc( size_t( plane->host.size ), MemKind::Device );
    }
    planes.addr[comp]   = plane->twin.addr + planeOffset( pic, ComponentID( comp ) );
    planes.stride[comp] = int32_t( pic.bufs[comp].stride );
  }
  return planes;
}

GpuRecon::Planes GpuRecon::scratchPlanes( const PelStorage& pic )
{
  Planes planes{};
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    const HostPlane* plane = hostPlane( pic, ComponentID( comp ) );
    if( m_scratch[comp].size != plane->host.size )
    {
      m_ctx->free( m_scratch[comp] );
      m_scratch[comp] = m_ctx->alloc( size_t( plane->host.size ), MemKind::Device );
    }
    planes.addr[comp]   = m_scratch[comp].addr + planeOffset( pic, ComponentID( comp ) );
    planes.stride[comp] = int32_t( pic.bufs[comp].stride );
  }
  return planes;
}

void GpuRecon::transferBarrier( bool toTransfer )
{
  VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
  barrier.srcAccessMask = toTransfer ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = toTransfer ? VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  m_ctx->api().vkCmdPipelineBarrier( m_cmd, toTransfer ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     toTransfer ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0,
                                     nullptr );
}

void GpuRecon::copyPlanes( const PelStorage& pic, int from, int to )
{
  const Api& vk = m_ctx->api();
  for( int comp = 0; comp < (int) pic.bufs.size(); comp++ )
  {
    const HostPlane* plane    = hostPlane( pic, ComponentID( comp ) );
    const Buffer*    bufs[3]  = { &plane->host, &plane->twin, &m_scratch[comp] };
    VkBufferCopy     region{ 0, 0, plane->host.size };
    vk.vkCmdCopyBuffer( m_cmd, bufs[from]->buffer, bufs[to]->buffer, 1, &region );
  }
  mark( from == kHost ? "upload" : ( to == kHost ? "download" : "copy" ) );
}

// the push constants of shaders/deblock.slang
struct DeblockPush
{
  VkDeviceAddress pic0, pic1, lfp, ctus;
  int32_t         stride, width, height, ctuLog2, widthInCtus, bitDepth, num4x4, dir, scaleX, scaleY;
};

bool GpuRecon::filtersSupported( const CodingStructure& cs )
{
  return alfSupported( cs ) && saoSupported( cs ) && !cs.sps->getLadfEnabled();
}

LoopFilterParam* GpuRecon::loopFilterParams( size_t count )
{
  const size_t size = count * sizeof( LoopFilterParam );
  if( m_lfp.size < size )
  {
    m_ctx->free( m_lfp );
    m_lfp = m_ctx->alloc( size, MemKind::HostCached );
  }
  return reinterpret_cast<LoopFilterParam*>( m_lfp.mapped );
}

size_t GpuRecon::deblockParamSize( const CodingStructure& cs )
{
  return ( size_t( cs.pcv->sizeInCtus ) * 8 + 255 ) & ~size_t( 255 );
}

bool GpuRecon::prepareDeblock( const CodingStructure& cs, uint8_t* base )
{
  bool any = false;
  for( int ctuIdx = 0; ctuIdx < (int) cs.pcv->sizeInCtus; ctuIdx++ )
  {
    const Slice& slice = *cs.getCtuData( ctuIdx ).slice;
    int8_t*      ctu   = reinterpret_cast<int8_t*>( base + ctuIdx * 8 );
    ctu[0] = slice.getDeblockingFilterDisable() ? 1 : 0;
    ctu[1] = int8_t( slice.getDeblockingFilterBetaOffsetDiv2() );
    ctu[2] = int8_t( slice.getDeblockingFilterTcOffsetDiv2() );
    ctu[3] = int8_t( slice.getDeblockingFilterCbBetaOffsetDiv2() );
    ctu[4] = int8_t( slice.getDeblockingFilterCbTcOffsetDiv2() );
    ctu[5] = int8_t( slice.getDeblockingFilterCrBetaOffsetDiv2() );
    ctu[6] = int8_t( slice.getDeblockingFilterCrTcOffsetDiv2() );
    ctu[7] = 0;
    any |= !slice.getDeblockingFilterDisable();
  }
  return any;
}

void GpuRecon::recordDeblock( const CodingStructure& cs, const Planes& planes, VkDeviceAddress params )
{
  const PreCalcValues& pcv = *cs.pcv;
  const Api&           vk  = m_ctx->api();
  const bool           chroma = isChromaEnabled( pcv.chrFormat );

  DeblockPush push{};
  push.lfp         = m_lfp.addr;
  push.ctus        = params;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  push.bitDepth    = cs.sps->getBitDepth();
  push.num4x4      = int32_t( pcv.num4x4CtuBlks );
  push.scaleX      = chroma ? getComponentScaleX( COMPONENT_Cb, pcv.chrFormat ) : 0;
  push.scaleY      = chroma ? getComponentScaleY( COMPONENT_Cb, pcv.chrFormat ) : 0;

  for( int dir = 0; dir < 2; dir++ )
  {
    if( dir == 1 )
    {
      computeBarrier();   // the horizontal edges take what the vertical ones left
    }
    push.dir = dir;

    push.pic0   = planes.addr[COMPONENT_Y];
    push.pic1   = 0;
    push.stride = planes.stride[COMPONENT_Y];
    push.width  = pcv.lumaWidth;
    push.height = pcv.lumaHeight;
    vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_deblock_luma ) );
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    vk.vkCmdDispatch( m_cmd, ( ( push.width + 3 ) / 4 + 7 ) / 8, ( ( push.height + 3 ) / 4 + 7 ) / 8, 1 );

    if( chroma )
    {
      CHECK( planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );
      push.pic0   = planes.addr[COMPONENT_Cb];
      push.pic1   = planes.addr[COMPONENT_Cr];
      push.stride = planes.stride[COMPONENT_Cb];
      push.width  = pcv.lumaWidth >> push.scaleX;
      push.height = pcv.lumaHeight >> push.scaleY;
      // as shaders/deblock.slang walks the segments
      const int along  = dir == 0 ? ( 4 >> push.scaleY ) : ( 4 >> push.scaleX );
      const int across = dir == 0 ? ( ( 8 << push.scaleX ) / 4 * ( 4 >> push.scaleX ) ) : ( ( 8 << push.scaleY ) / 4 * ( 4 >> push.scaleY ) );
      const int threadsX = dir == 0 ? ( push.width + across - 1 ) / across : ( push.width + along - 1 ) / along;
      const int threadsY = dir == 0 ? ( push.height + along - 1 ) / along : ( push.height + across - 1 ) / across;
      vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_deblock_chroma ) );
      vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
      vk.vkCmdDispatch( m_cmd, ( threadsX + 7 ) / 8, ( threadsY + 7 ) / 8, 1 );
    }
    mark( dir == 0 ? "deblock ver" : "deblock hor" );
  }
}

// the push constants of shaders/intra.slang
struct IntraPush
{
  VkDeviceAddress plane0, plane1, plane2, jobs;
  int32_t         stride0, stride12, first, count, bitDepth, ctuLog2, scaleX, scaleY, collocated;
};

bool GpuRecon::intraSupported( const CodingStructure& cs )
{
  return intraSupported( *cs.sps );
}

bool GpuRecon::intraSupported( const SPS& sps )
{
  return sps.getChromaFormatIdc() == CHROMA_420 && !sps.getUseMIP() && !sps.getUseISP() && !sps.getUseMRL() && !sps.getUseCiip()
         && !sps.getUseColorTrans() && !sps.getUseReshaper() && !sps.getUseWrapAround();
}

void GpuRecon::recordIntra( const CodingStructure& cs, const std::vector<std::vector<IntraJob>>& jobs, const Planes& planes )
{
  // the jobs, wave by wave
  int    maxWave = 0;
  size_t total   = 0;
  for( const auto& ctu: jobs )
  {
    for( const IntraJob& job: ctu )
    {
      maxWave = std::max( maxWave, job.wave() );
    }
    total += ctu.size();
  }
  if( total == 0 )
  {
    return;
  }
  std::vector<uint32_t> first( size_t( maxWave ) + 2, 0 );
  for( const auto& ctu: jobs )
  {
    for( const IntraJob& job: ctu )
    {
      first[job.wave() + 1]++;
    }
  }
  for( int wave = 1; wave <= maxWave + 1; wave++ )
  {
    first[wave] += first[wave - 1];
  }
  const size_t size = total * sizeof( IntraJob );
  if( m_intraJobs.size < size )
  {
    m_ctx->free( m_intraJobs );
    m_intraJobs = m_ctx->alloc( size * 2, MemKind::HostCached );
  }
  IntraJob*             sorted = reinterpret_cast<IntraJob*>( m_intraJobs.mapped );
  std::vector<uint32_t> next( first.begin(), first.end() - 1 );
  for( const auto& ctu: jobs )
  {
    for( const IntraJob& job: ctu )
    {
      sorted[next[job.wave()]++] = job;
    }
  }

  const PreCalcValues& pcv = *cs.pcv;
  const Api&           vk  = m_ctx->api();
  IntraPush            push{};
  push.plane0     = planes.addr[COMPONENT_Y];
  push.plane1     = planes.addr[COMPONENT_Cb];
  push.plane2     = planes.addr[COMPONENT_Cr];
  push.jobs       = m_intraJobs.addr;
  push.stride0    = planes.stride[COMPONENT_Y];
  push.stride12   = planes.stride[COMPONENT_Cb];
  push.bitDepth   = cs.sps->getBitDepth();
  push.ctuLog2    = getLog2( pcv.maxCUWidth );
  push.scaleX     = getComponentScaleX( COMPONENT_Cb, pcv.chrFormat );
  push.scaleY     = getComponentScaleY( COMPONENT_Cb, pcv.chrFormat );
  push.collocated = cs.sps->getCclmCollocatedChromaFlag() ? 1 : 0;
  CHECK( planes.stride[COMPONENT_Cb] != planes.stride[COMPONENT_Cr], "the chroma planes differ in stride" );
  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_intra ) );
  for( int wave = 1; wave <= maxWave; wave++ )
  {
    push.first = int32_t( first[wave] );
    push.count = int32_t( first[wave + 1] - first[wave] );
    if( push.count == 0 )
    {
      continue;
    }
    computeBarrier();   // the waves before
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    vk.vkCmdDispatch( m_cmd, ( push.count + 63 ) / 64, 1, 1 );
  }
  mark( "intra" );
}

// the push constants of shaders/inter.slang
struct InterPush
{
  VkDeviceAddress jobs, refs, dst0, dst1, dst2, dmvr;
  int32_t         stride0, stride12, count, picWidth, picHeight, ctuSize, bitDepth;
};

bool GpuRecon::interSupported( const SPS& sps )
{
  return intraSupported( sps ) && sps.getBitDepth() <= 10 && !sps.getUseAffine() && !sps.getUseBIO() && !sps.getUseGeo() && !sps.getUseBcw()
         && !sps.getSBTMVPEnabledFlag() && !sps.getSubPicInfoPresentFlag();
}

bool GpuRecon::interSupported( const CodingStructure& cs )
{
  const PPS& pps = *cs.pps;
  if( !interSupported( *cs.sps ) || pps.getUseWP() || pps.getWPBiPred() )
  {
    return false;
  }
  for( const Slice* slice: cs.picture->slices )
  {
    for( int list = 0; list < 2; list++ )
    {
      for( int refIdx = 0; refIdx < slice->getNumRefIdx( RefPicList( list ) ); refIdx++ )
      {
        if( slice->getRefPic( RefPicList( list ), refIdx )->isRefScaled( &pps ) )
        {
          return false;
        }
      }
    }
  }
  return true;
}

Mv* GpuRecon::dmvrMvCache( size_t count )
{
  const size_t size = count * sizeof( Mv );
  if( m_dmvr.size < size )
  {
    m_ctx->free( m_dmvr );
    m_dmvr = m_ctx->alloc( size, MemKind::HostCached );
  }
  return reinterpret_cast<Mv*>( m_dmvr.mapped );
}

void GpuRecon::recordInter( const CodingStructure& cs, const std::vector<std::vector<InterJob>>& jobs, const Planes& planes, VkDeviceAddress refTable )
{
  size_t total = 0;
  for( const auto& ctu: jobs )
  {
    total += ctu.size();
  }
  if( total == 0 )
  {
    return;
  }
  const size_t size = total * sizeof( InterJob );
  if( m_interJobs.size < size )
  {
    m_ctx->free( m_interJobs );
    m_interJobs = m_ctx->alloc( size * 2, MemKind::HostCached );
  }
  InterJob* all = reinterpret_cast<InterJob*>( m_interJobs.mapped );
  for( const auto& ctu: jobs )
  {
    memcpy( all, ctu.data(), ctu.size() * sizeof( InterJob ) );
    all += ctu.size();
  }

  const PreCalcValues& pcv = *cs.pcv;
  InterPush            push{};
  push.jobs      = m_interJobs.addr;
  push.refs      = refTable;
  push.dst0      = planes.addr[COMPONENT_Y];
  push.dst1      = planes.addr[COMPONENT_Cb];
  push.dst2      = planes.addr[COMPONENT_Cr];
  push.dmvr      = m_dmvr.addr;
  push.stride0   = planes.stride[COMPONENT_Y];
  push.stride12  = planes.stride[COMPONENT_Cb];
  push.count     = int32_t( total );
  push.picWidth  = pcv.lumaWidth;
  push.picHeight = pcv.lumaHeight;
  push.ctuSize   = pcv.maxCUWidth;
  push.bitDepth  = cs.sps->getBitDepth();

  const Api& vk = m_ctx->api();
  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_inter ) );
  vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
  vk.vkCmdDispatch( m_cmd, uint32_t( ( total + 63 ) / 64 ), 1, 1 );
  mark( "inter" );
}

void GpuRecon::finishPicture( CodingStructure& cs, const std::vector<std::vector<IntraJob>>* intraJobs,
                              const std::vector<std::vector<InterJob>>* interJobs, const RefTable& refs, bool deblock, SampleAdaptiveOffset* sao,
                              const AdaptiveLoopFilter* alf, PelStorage& pic )
{
  const size_t dbSize  = deblockParamSize( cs );
  const size_t alfSize = alf ? alfParamSize( cs ) : 0;
  const size_t saoSize = sao ? saoParamSize( cs ) : 0;
  const size_t refSize = ( refs.pictures().size() * 4 * sizeof( VkDeviceAddress ) + 255 ) & ~size_t( 255 );
  uint8_t*     base    = params( dbSize + alfSize + saoSize + refSize );
  const size_t refAt   = dbSize + alfSize + saoSize;

  // the reference pictures: their planes on the GPU, where a picture the CPU
  // reconstructed alone (or filled in) goes first
  std::vector<const Picture*> uploads;
  if( interJobs )
  {
    VkDeviceAddress* table = reinterpret_cast<VkDeviceAddress*>( base + refAt );
    for( const void* ref: refs.pictures() )
    {
      const Picture* refPic = static_cast<const Picture*>( ref );
      const Planes   planes = twinPlanes( refPic->m_bufs[PIC_RECONSTRUCTION] );
      table[0]              = planes.addr[COMPONENT_Y];
      table[1]              = planes.addr[COMPONENT_Cb];
      table[2]              = planes.addr[COMPONENT_Cr];
      table[3]              = 0;
      CHECK( planes.stride[COMPONENT_Y] != pic.bufs[COMPONENT_Y].stride || planes.stride[COMPONENT_Cb] != pic.bufs[COMPONENT_Cb].stride,
             "Vulkan: a reference picture laid out differently" );
      table += 4;
      if( !refPic->gpuCurrent )
      {
        uploads.push_back( refPic );
      }
    }
  }

  const bool doDeblock = deblock && prepareDeblock( cs, base );
  const bool doSao     = sao && prepareSao( cs, *sao, base + dbSize + alfSize );
  if( alf )
  {
    prepareAlf( cs, *alf, base + dbSize );
  }
  // in the GPU's own memory, which its shaders reach faster than the host's:
  // the picture goes in (what the CPU made of it), is finished there, stays
  // there as a reference, and comes back for the output
  beginCommands();
  for( const Picture* refPic: uploads )
  {
    copyPlanes( refPic->m_bufs[PIC_RECONSTRUCTION], kHost, kTwin );
  }
  const Planes planes[2] = { twinPlanes( pic ), scratchPlanes( pic ) };
  int          cur       = 0;
  copyPlanes( pic, kHost, kTwin );
  transferBarrier( false );
  if( interJobs )
  {
    recordInter( cs, *interJobs, planes[cur], m_params.addr + refAt );   // in place
    computeBarrier();
  }
  if( intraJobs )
  {
    recordIntra( cs, *intraJobs, planes[cur] );   // in place
    computeBarrier();
  }
  if( doDeblock )
  {
    recordDeblock( cs, planes[cur], m_params.addr );   // in place
  }
  if( doSao )
  {
    if( doDeblock )
    {
      computeBarrier();
    }
    recordSao( cs, planes[cur], planes[1 - cur], m_params.addr + dbSize + alfSize );
    cur = 1 - cur;
  }
  if( alf )
  {
    if( doDeblock || doSao )
    {
      computeBarrier();
    }
    recordAlf( cs, planes[cur], planes[1 - cur], m_params.addr + dbSize );
    cur = 1 - cur;
  }
  transferBarrier( true );
  if( cur == 1 )
  {
    // the result to the picture's own planes on the GPU
    copyPlanes( pic, kScratch, kTwin );
    VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    m_ctx->api().vkCmdPipelineBarrier( m_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr );
  }
  copyPlanes( pic, kTwin, kHost );
  submitAndWait();
  for( const Picture* refPic: uploads )
  {
    refPic->gpuCurrent = true;
  }
}

}   // namespace gpu
}   // namespace vvdec
