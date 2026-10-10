# executable: vvdecbatch, a test aid of the Vulkan port (decodes a list of bitstreams in one process and prints each
# output frame's per-plane MD5; see vvdecbatch.cpp)
set( EXE_NAME vvdecbatch )

add_executable( ${EXE_NAME} vvdecbatch.cpp )
set_target_properties( ${EXE_NAME} PROPERTIES RELEASE_POSTFIX        "${CMAKE_RELEASE_POSTFIX}" )
set_target_properties( ${EXE_NAME} PROPERTIES DEBUG_POSTFIX          "${CMAKE_DEBUG_POSTFIX}" )
set_target_properties( ${EXE_NAME} PROPERTIES RELWITHDEBINFO_POSTFIX "${CMAKE_RELWITHDEBINFO_POSTFIX}" )
set_target_properties( ${EXE_NAME} PROPERTIES MINSIZEREL_POSTFIX     "${CMAKE_MINSIZEREL_POSTFIX}" )
if( VVDEC_LIBRARY_ONLY )
  set_target_properties( ${EXE_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE )
endif()

target_link_libraries( ${EXE_NAME} Threads::Threads vvdec )
target_include_directories( ${EXE_NAME} PRIVATE ../../Lib/libmd5 )

set_target_properties( ${EXE_NAME} PROPERTIES FOLDER app )
