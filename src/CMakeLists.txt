add_library(nncore)
add_library(NNKernels::nncore ALIAS nncore)

# Library sources
target_sources(nncore
  PRIVATE
    core/Tensor.cpp
    core/Device.cpp
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
      core/Tensor.h
      core/Device.h
)

# Include directories
target_include_directories(nncore
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Compiler definitions based on features
if(NNK_WITH_CUDA)
  target_compile_definitions(nncore PUBLIC NNK_WITH_CUDA=1)

  # Link CUDA libraries if available
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    target_link_libraries(nncore PUBLIC CUDA::cudart)
  endif()
else()
  target_compile_definitions(nncore PUBLIC NNK_WITH_CUDA=0)
endif()

# Set target properties for professional library
set_target_properties(nncore PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
  EXPORT_NAME nncore
  OUTPUT_NAME nnk_core
)

# Conditionally add subdirectories for additional modules
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/CMakeLists.txt)
  add_subdirectory(kernels)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/layers/CMakeLists.txt)
  add_subdirectory(layers)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/profiler/CMakeLists.txt)
  add_subdirectory(profiler)
endif()

# Installation configuration
install(TARGETS nncore
  EXPORT NNKernelsTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets for find_package()
install(EXPORT NNKernelsTargets
  FILE NNKernelsTargets.cmake
  NAMESPACE NNKernels::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NNKernels
)


