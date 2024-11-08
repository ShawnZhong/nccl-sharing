#!/usr/bin/env sh

set -x
set -e

base_dir=$(dirname "$0")

make -C "${base_dir}"/nccl -j"$(nproc)" NVCC_GENCODE="-arch=native"
mkdir -p "${base_dir}"/build
cmake -B "${base_dir}"/build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build "${base_dir}"/build -- -j"$(nproc)"
NCCL_DEBUG_SUBSYS=INIT,GRAPH,TUNING,ENV NCCL_P2P_DISABLE=1 NCCL_DEBUG=INFO NCCL_MIN_NCHANNELS=8 nvprof --print-gpu-trace "${base_dir}"/build/nccl_test