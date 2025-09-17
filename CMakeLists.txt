cmake_minimum_required(VERSION 3.20)

project(nn_kernels_tt LANGUAGES CXX)

option(TT_WITH_CUDA "Enable CUDA backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(TT_WITH_CUDA)
  enable_language(CUDA)
endif()

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)


