add_library(nncore)

target_sources(nncore
  PRIVATE
    core/Tensor.cpp
)

target_include_directories(nncore
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(TT_WITH_CUDA)
  target_compile_definitions(nncore PUBLIC TT_WITH_CUDA=1)
else()
  target_compile_definitions(nncore PUBLIC TT_WITH_CUDA=0)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/kernels/CMakeLists.txt)
  add_subdirectory(kernels)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/layers/CMakeLists.txt)
  add_subdirectory(layers)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/profiler/CMakeLists.txt)
  add_subdirectory(profiler)
endif()


