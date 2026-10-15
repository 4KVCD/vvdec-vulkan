# Checks that no shader's push constants (its struct Params) hold an array: Intel's compiler (driver 101.8991) read an
# array of the push constants indexed at run time wrong (dbparams.slang's virtual boundaries, which deblocked the GDR
# streams' boundaries). Arrays go in buffers, or as scalars, as dbparams.slang's, sao.slang's and alf.slang's virtual
# boundaries are.
#
#   cmake -DSHADER_DIR=<source/Lib/VulkanLib/shaders> -P vvdecCheckPushConstants.cmake
if( NOT SHADER_DIR )
  message( FATAL_ERROR "SHADER_DIR: the directory of the .slang shaders" )
endif()

file( GLOB _shaders "${SHADER_DIR}/*.slang" )
set( _found 0 )
set( _bad "" )
foreach( _shader IN LISTS _shaders )
  file( READ "${_shader}" _text )
  string( REGEX MATCH "struct Params[ \t\r\n]*{[^}]*}" _params "${_text}" )
  if( NOT _params )
    continue()
  endif()
  math( EXPR _found "${_found} + 1" )
  string( REGEX REPLACE "//[^\n]*" "" _params "${_params}" )
  string( REGEX MATCHALL "[A-Za-z_][A-Za-z0-9_]*[ \t]*\\[[^;\n]*\\][ \t]*[;,]" _arrays "${_params}" )
  if( _arrays )
    get_filename_component( _name "${_shader}" NAME )
    string( APPEND _bad "  ${_name}: ${_arrays}\n" )
  endif()
endforeach()

if( _found EQUAL 0 )
  message( FATAL_ERROR "no struct Params found in ${SHADER_DIR}" )
endif()
if( _bad )
  message( FATAL_ERROR "push constants with arrays (read them as scalars or from a buffer):\n${_bad}" )
endif()
message( STATUS "push constants of ${_found} shaders: no arrays" )
