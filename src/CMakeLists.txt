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

add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(profiler)


