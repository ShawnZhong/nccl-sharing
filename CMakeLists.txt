set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
cmake_minimum_required(VERSION 3.10)
project(NCCL_SCHED CUDA)

set(CMAKE_CUDA_ARCHITECTURES "native")

set(NCCL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/nccl/build/)

add_executable(nccl_test main.cu)
target_link_libraries(nccl_test ${NCCL_PATH}/lib/libnccl.so)
target_include_directories(nccl_test PRIVATE ${NCCL_PATH}/include)
