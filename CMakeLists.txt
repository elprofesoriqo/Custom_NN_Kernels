cmake_minimum_required(VERSION 3.20)

# Project definition with version for packaging
project(NNKernels
  VERSION 1.0.0
  DESCRIPTION "Neural Network and Graph Neural Network Kernels Library"
  LANGUAGES CXX
)

# Build options
option(NNK_WITH_CUDA "Enable CUDA backend" OFF)
option(NNK_BUILD_TESTS "Build unit tests" ON)
option(NNK_BUILD_DOCS "Build documentation" OFF)
option(NNK_BUILD_EXAMPLES "Build example programs" ON)

# C++ standard and compilation flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable CUDA if requested
if(NNK_WITH_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
endif()

# Add compile definitions for feature detection
add_compile_definitions(
  NNK_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  NNK_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  NNK_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Include GNUInstallDirs for standard installation paths
include(GNUInstallDirs)

# Add main library
add_subdirectory(src)

# Optional components
if(NNK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(NNK_BUILD_EXAMPLES)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt)
    add_subdirectory(examples)
  endif()
endif()

# Documentation with Doxygen
if(NNK_BUILD_DOCS)
  find_package(Doxygen QUIET)
  if(DOXYGEN_FOUND)
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    add_custom_target(docs ALL
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM
    )
  else()
    message(WARNING "Doxygen not found, documentation will not be generated")
  endif()
endif()

# Export and install configuration
include(CMakePackageConfigHelpers)

# Create config files for find_package()
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NNKernelsConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/NNKernelsConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NNKernels
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/NNKernelsConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Installation of config files
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NNKernelsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NNKernelsConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NNKernels
)


