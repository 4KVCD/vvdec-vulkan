# vvdechandover: the hand-over's check (vvdec_handover_*), Windows with Vulkan only

set( EXE_NAME vvdechandover )

add_executable( ${EXE_NAME} vvdechandover.cpp )
set_target_properties( ${EXE_NAME} PROPERTIES RELEASE_POSTFIX        "${CMAKE_RELEASE_POSTFIX}" )
set_target_properties( ${EXE_NAME} PROPERTIES DEBUG_POSTFIX          "${CMAKE_DEBUG_POSTFIX}" )
set_target_properties( ${EXE_NAME} PROPERTIES RELWITHDEBINFO_POSTFIX "${CMAKE_RELWITHDEBINFO_POSTFIX}" )
set_target_properties( ${EXE_NAME} PROPERTIES MINSIZEREL_POSTFIX     "${CMAKE_MINSIZEREL_POSTFIX}" )
if( VVDEC_LIBRARY_ONLY )
  set_target_properties( ${EXE_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE )
endif()

target_link_libraries( ${EXE_NAME} Threads::Threads vvdec )
target_include_directories( ${EXE_NAME} PRIVATE ../../Lib/libmd5 "${VVDEC_VULKAN_HEADERS}" )
