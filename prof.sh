#!/usr/bin/env sh

set -x
set -e

base_dir=$(dirname "$0")

make -C "${base_dir}"/nccl
mkdir -p "${base_dir}"/build
cmake -B "${base_dir}"/build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build "${base_dir}"/build -- -j"$(nproc)"
nvprof --print-gpu-trace "${base_dir}"/build/nccl_test