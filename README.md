# nccl-sched
 
```
make -C nccl NVCC_GENCODE="-arch=native" -j
make -C nccl-tests NCCL_HOME=$(readlink -f nccl/build) -j
NCCL_P2P_DISABLE=1 LD_PRELOAD=$(readlink -f nccl/build/lib/libnccl.so) ./nccl-tests/build/all_reduce_perf -g 2 -b 8 -e 100M -f 2
```

# Commands
- Profiling
```
sudo PATH=$PATH /usr/local/cuda/bin/nvprof \
    --devices 0 \
    --profile-child-processes \
    --metrics inst_per_warp,inst_executed,ipc \
    --print-gpu-trace \
    python launch.py op --op nop -c 1,256 2,128 2,256 -n 1 --comm_size 1
```
