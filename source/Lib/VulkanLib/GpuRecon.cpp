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
#include "CommonLib/Slice.h"

#include <cstring>

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
}

GpuRecon::~GpuRecon()
{
  const Api& vk = m_ctx->api();
  if( m_fence )
  {
    vk.vkDestroyFence( m_ctx->device(), m_fence, nullptr );
  }
  if( m_pool )
  {
    vk.vkDestroyCommandPool( m_ctx->device(), m_pool, nullptr );
  }
  m_ctx->free( m_params );
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
  m_ctx->submit( m_cmd, m_fence );
  const VkResult result = vk.vkWaitForFences( m_ctx->device(), 1, &m_fence, VK_TRUE, UINT64_MAX );
  if( result != VK_SUCCESS )
  {
    THROW_FATAL( "Vulkan: waiting for the GPU failed (" << result << ")" );
  }
}

// ====================================================================================================================
// ALF
// ====================================================================================================================

// the push constants of shaders/alf.slang
struct AlfPush
{
  VkDeviceAddress src0, dst0, src1, dst1, srcLuma, ctus, coeff, clip, ccCoeff;
  int32_t         srcStride, dstStride, lumaStride, width, height, lumaWidth, lumaHeight, ctuLog2, widthInCtus, bitDepth, scaleX, scaleY;
};

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

void GpuRecon::alf( const CodingStructure& cs, const AdaptiveLoopFilter& alfFilter, const PelStorage& src, PelStorage& dst )
{
  const PreCalcValues& pcv = *cs.pcv;

  // the filter tables: luma sets [16 fixed + 8 APS][transpose][class][13], chroma
  // [APS][alternative][7], CC-ALF [APS][component][filter][8]; then the CTUs
  constexpr size_t kSetSize       = MAX_NUM_ALF_TRANSPOSE_ID * MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF;
  constexpr size_t kChromaSize    = ALF_CTB_MAX_NUM_APS * MAX_NUM_ALF_ALTERNATIVES_CHROMA * MAX_NUM_ALF_CHROMA_COEFF;
  constexpr size_t kCcSize        = ALF_CTB_MAX_NUM_APS * 2 * MAX_NUM_CC_ALF_FILTERS * MAX_NUM_CC_ALF_CHROMA_COEFF;
  constexpr size_t kLumaCoeff     = 0;
  constexpr size_t kLumaClip      = kLumaCoeff + NUM_TOTAL_FILTER_SETS * kSetSize * sizeof( int16_t );
  constexpr size_t kChromaCoeff   = kLumaClip + NUM_TOTAL_FILTER_SETS * kSetSize * sizeof( int16_t );
  constexpr size_t kChromaClip    = kChromaCoeff + kChromaSize * sizeof( int16_t );
  constexpr size_t kCcCoeff       = kChromaClip + kChromaSize * sizeof( int16_t );
  constexpr size_t kCtus          = ( kCcCoeff + kCcSize * sizeof( int16_t ) + 255 ) & ~size_t( 255 );
  static_assert( MAX_NUM_ALF_TRANSPOSE_ID * MAX_NUM_ALF_CLASSES * MAX_NUM_ALF_LUMA_COEFF == 4 * 25 * 13, "the layout of shaders/alf.slang" );

  uint8_t* base        = params( kCtus + size_t( pcv.sizeInCtus ) * 8 );
  int16_t* lumaCoeff   = reinterpret_cast<int16_t*>( base + kLumaCoeff );
  int16_t* lumaClip    = reinterpret_cast<int16_t*>( base + kLumaClip );
  int16_t* chromaCoeff = reinterpret_cast<int16_t*>( base + kChromaCoeff );
  int16_t* chromaClip  = reinterpret_cast<int16_t*>( base + kChromaClip );
  int16_t* ccCoeff     = reinterpret_cast<int16_t*>( base + kCcCoeff );
  uint8_t* ctus        = base + kCtus;

  for( int set = 0; set < NUM_FIXED_FILTER_SETS; set++ )
  {
    memcpy( lumaCoeff + set * kSetSize, alfFilter.getFixedFilterSetCoeff( set ), kSetSize * sizeof( int16_t ) );
    memcpy( lumaClip  + set * kSetSize, alfFilter.getClipDefault(),               kSetSize * sizeof( int16_t ) );
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
        memcpy( lumaCoeff + ( NUM_FIXED_FILTER_SETS + apsId ) * kSetSize, alfParam.lumaCoeffFinal, kSetSize * sizeof( int16_t ) );
        memcpy( lumaClip  + ( NUM_FIXED_FILTER_SETS + apsId ) * kSetSize, alfParam.lumaClippFinal, kSetSize * sizeof( int16_t ) );
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

  const Api& vk = m_ctx->api();
  beginCommands();

  AlfPush push{};
  push.ctus        = m_params.addr + kCtus;
  push.ctuLog2     = getLog2( pcv.maxCUWidth );
  push.widthInCtus = pcv.widthInCtus;
  push.bitDepth    = cs.sps->getBitDepth();
  push.lumaWidth   = pcv.lumaWidth;
  push.lumaHeight  = pcv.lumaHeight;
  push.lumaStride  = int32_t( src.bufs[COMPONENT_Y].stride );

  // luma
  push.src0      = planeAddress( src, COMPONENT_Y );
  push.dst0      = planeAddress( dst, COMPONENT_Y );
  push.coeff     = m_params.addr + kLumaCoeff;
  push.clip      = m_params.addr + kLumaClip;
  push.srcStride = int32_t( src.bufs[COMPONENT_Y].stride );
  push.dstStride = int32_t( dst.bufs[COMPONENT_Y].stride );
  push.width     = pcv.lumaWidth;
  push.height    = pcv.lumaHeight;
  vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_alf_luma ) );
  vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
  vk.vkCmdDispatch( m_cmd, ( ( push.width + 3 ) / 4 + 7 ) / 8, ( ( push.height + 3 ) / 4 + 7 ) / 8, 1 );

  if( isChromaEnabled( pcv.chrFormat ) )
  {
    CHECK( src.bufs[COMPONENT_Cb].stride != src.bufs[COMPONENT_Cr].stride || dst.bufs[COMPONENT_Cb].stride != dst.bufs[COMPONENT_Cr].stride,
           "the chroma planes differ in stride" );
    push.src0      = planeAddress( src, COMPONENT_Cb );
    push.dst0      = planeAddress( dst, COMPONENT_Cb );
    push.src1      = planeAddress( src, COMPONENT_Cr );
    push.dst1      = planeAddress( dst, COMPONENT_Cr );
    push.srcLuma   = planeAddress( src, COMPONENT_Y );
    push.coeff     = m_params.addr + kChromaCoeff;
    push.clip      = m_params.addr + kChromaClip;
    push.ccCoeff   = m_params.addr + kCcCoeff;
    push.srcStride = int32_t( src.bufs[COMPONENT_Cb].stride );
    push.dstStride = int32_t( dst.bufs[COMPONENT_Cb].stride );
    push.scaleX    = getComponentScaleX( COMPONENT_Cb, pcv.chrFormat );
    push.scaleY    = getComponentScaleY( COMPONENT_Cb, pcv.chrFormat );
    push.width     = pcv.lumaWidth >> push.scaleX;
    push.height    = pcv.lumaHeight >> push.scaleY;
    vk.vkCmdBindPipeline( m_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->pipeline( kShader_alf_chroma ) );
    vk.vkCmdPushConstants( m_cmd, m_ctx->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( push ), &push );
    vk.vkCmdDispatch( m_cmd, ( ( push.width + 3 ) / 4 + 7 ) / 8, ( ( push.height + 3 ) / 4 + 7 ) / 8, 1 );
  }

  submitAndWait();
}

}   // namespace gpu
}   // namespace vvdec
