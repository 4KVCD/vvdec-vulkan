# The Vulkan shaders of source/Lib/VulkanLib: each entry of VVDEC_GPU_SHADERS is
# "name|file|DEFINE=value|...", compiled from shaders/<file>.slang with slangc
# (VVDEC_SLANGC) to SPIR-V, and all of them embedded in the header
# ${VVDEC_GPU_INCLUDE_DIR}/vvdec_gpu/GpuShaders.h, with their ids in GpuShaderIds.h
# next to it (target vvdec_gpu_shaders).

set( VVDEC_GPU_SHADERS
  "alf_luma|alf|CHROMA=0"
  "alf_chroma|alf|CHROMA=1"
  "alf_luma_bounds|alf|CHROMA=0|BOUNDS=1"
  "alf_chroma_bounds|alf|CHROMA=1|BOUNDS=1"
  "sao|sao"
  "deblock_luma_ver|deblock|CHROMA=0|DIR=0"
  "deblock_luma_hor|deblock|CHROMA=0|DIR=1"
  "deblock_chroma|deblock|CHROMA=1"
  "intra|intra"
  "inter|inter"
  "inter_bdof|inter|BDOF=1"
  "inter_affine|inter|AFFINE=1"
  "trafo|trafo"
  "lmcs|lmcs"
  "filmgrain|filmgrain"
  "mider|mider"
  "mider_lite|mider|LITE=1"
  "mider_cumap|mider|CUMAP=1"
  "dbparams|dbparams"
  "jobs|jobs"
  "jobs_waves|jobs|WAVES=1"
  "edges|edges"
  "intrasort|intrasort"
  "colmotion|colmotion"
  "parse|parse"
  "parse_prof|parse|PROFILE=1"
  "parse_ctu64|parse|MAX_CTU=64"
  "parse_prof_ctu64|parse|PROFILE=1|MAX_CTU=64"
  "parse_shuffle|parse|READ_INVOCATION=0"
  "parsecompact|parsecompact"
  "parsecheck|parsecompact|CHECK=1"
)

if( NOT VVDEC_SLANGC OR NOT EXISTS "${VVDEC_SLANGC}" )
  message( FATAL_ERROR "VVDEC_ENABLE_VULKAN needs VVDEC_SLANGC, the path of slangc (the Slang shader compiler)" )
endif()
if( NOT VVDEC_VULKAN_HEADERS OR NOT EXISTS "${VVDEC_VULKAN_HEADERS}/vulkan/vulkan.h" )
  message( FATAL_ERROR "VVDEC_ENABLE_VULKAN needs VVDEC_VULKAN_HEADERS, the include directory of the Vulkan headers" )
endif()

set( _shader_dir        "${CMAKE_CURRENT_LIST_DIR}/../../source/Lib/VulkanLib/shaders" )
set( _spirv_dir         "${CMAKE_BINARY_DIR}/vvdec_gpu/spirv" )
set( VVDEC_GPU_INCLUDE_DIR "${CMAKE_BINARY_DIR}/vvdec_gpu/include" )
set( _header            "${VVDEC_GPU_INCLUDE_DIR}/vvdec_gpu/GpuShaders.h" )
set( _ids_header        "${VVDEC_GPU_INCLUDE_DIR}/vvdec_gpu/GpuShaderIds.h" )

file( GLOB _shader_sources CONFIGURE_DEPENDS "${_shader_dir}/*.slang" )
file( MAKE_DIRECTORY "${_spirv_dir}" "${VVDEC_GPU_INCLUDE_DIR}/vvdec_gpu" )

set( _spirv_files )
set( _names )
foreach( _entry IN LISTS VVDEC_GPU_SHADERS )
  string( REPLACE "|" ";" _parts "${_entry}" )
  list( GET _parts 0 _name )
  list( GET _parts 1 _file )
  list( LENGTH _parts _count )
  set( _defines )
  if( _count GREATER 2 )
    list( SUBLIST _parts 2 -1 _definitions )
    foreach( _definition IN LISTS _definitions )
      list( APPEND _defines "-D${_definition}" )
    endforeach()
  endif()
  set( _spirv "${_spirv_dir}/${_name}.spv" )
  add_custom_command( OUTPUT  "${_spirv}"
                      COMMAND "${VVDEC_SLANGC}" "${_shader_dir}/${_file}.slang" -target spirv -profile spirv_1_5
                              -entry main -stage compute -O2 ${_defines} -o "${_spirv}"
                      DEPENDS ${_shader_sources}
                      COMMENT "Compiling the shader ${_name}"
                      VERBATIM )
  list( APPEND _spirv_files "${_spirv}" )
  list( APPEND _names "${_name}" )
endforeach()

string( REPLACE ";" "," _names_argument "${_names}" )
add_custom_command( OUTPUT  "${_header}" "${_ids_header}"
                    COMMAND "${CMAKE_COMMAND}" "-DOUTPUT=${_header}" "-DNAMES=${_names_argument}" "-DDIR=${_spirv_dir}"
                            -P "${CMAKE_CURRENT_LIST_DIR}/vvdecEmbedSpirv.cmake"
                    DEPENDS ${_spirv_files} "${CMAKE_CURRENT_LIST_DIR}/vvdecEmbedSpirv.cmake"
                    COMMENT "Embedding the shaders"
                    VERBATIM )
add_custom_target( vvdec_gpu_shaders DEPENDS "${_header}" "${_ids_header}" )
