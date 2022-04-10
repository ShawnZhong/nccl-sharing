#include "cuda_runtime.h"
#include "nccl.h"
#include <chrono>
#include <cstdio>
#include <cuda_profiler_api.h>
#include <string>

#define CUDACHECK(cmd)                                         \
    do {                                                       \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess) {                                \
            printf("Failed: Cuda error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#define NCCLCHECK(cmd)                                         \
    do {                                                       \
        ncclResult_t r = cmd;                                  \
        if (r != ncclSuccess) {                                \
            printf("Failed, NCCL error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

__global__ void kernel(float *buf, int sz) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= sz)
        return;
    for (int j = 0; j < sz; ++j) {
        buf[i] += buf[j];
    }
}

int main(int argc, char *argv[]) {
    ncclComm_t comms[2];

    //managing 2 devices
    int nDev = 2;
    size_t comp_size = 50 * 1024;
    int nccl_size = 10 * 1024 * 1024;
    int devs[2] = {0, 1};
    int num_threads = 1024;

    //allocating and initializing device buffers
    auto nccl_buff = new float *[nDev];
    auto comp_buff = new float *[nDev];
    auto nccl_streams = new cudaStream_t[nDev];
    auto comp_streams = new cudaStream_t[nDev];

    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(&nccl_buff[i], nccl_size * sizeof(float)));
        CUDACHECK(cudaMalloc(&comp_buff[i], comp_size * sizeof(float)));
        CUDACHECK(cudaStreamCreate(&nccl_streams[i]));
        CUDACHECK(cudaStreamCreate(&comp_streams[i]));
    }

    //initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    cudaProfilerStart();
    printf("start timing\n");
    auto start_ts = std::chrono::high_resolution_clock::now();

    //calling NCCL communication API. Group API is required when using
    //multiple devices per thread
    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
        NCCLCHECK(ncclAllReduce(nccl_buff[i], nccl_buff[i], nccl_size, ncclFloat, ncclSum, comms[i], nccl_streams[i], 2, 288));
    }

    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        kernel<<<comp_size / num_threads, num_threads, 0, comp_streams[i]>>>(comp_buff[i], comp_size);
        CUDACHECK(cudaPeekAtLastError());
    }
    NCCLCHECK(ncclGroupEnd());

    //synchronizing on CUDA streams to wait for completion of NCCL operation
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(nccl_streams[i]));
        CUDACHECK(cudaStreamSynchronize(comp_streams[i]));
    }

    auto end_ts = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ts - start_ts).count();
    printf("Time taken: %ld milliseconds\n", duration);

    cudaProfilerStop();

    //finalizing NCCL
    for (int i = 0; i < nDev; ++i) {
        ncclCommDestroy(comms[i]);
    }

    //free device buffers
    for (int i = 0; i < nDev; ++i) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(nccl_buff[i]));
        CUDACHECK(cudaFree(comp_buff[i]));
    }
}