# nccl-sched
 
# Commands

Communication-only benchmark: 

```sh
python launch.py op --world_size 4 --op nop --configs all
```


Build and run `all_reduce_perf`: 
```sh
make -C nccl NVCC_GENCODE="-arch=native" -j"$(nproc)"
make -C nccl-tests NCCL_HOME=$(readlink -f nccl/build) -j"$(nproc)"
NCCL_P2P_DISABLE=1 LD_PRELOAD=$(readlink -f nccl/build/lib/libnccl.so) ./nccl-tests/build/all_reduce_perf -g 2 -b 8 -e 100M -f 2
```

Trace: 
```sh
nvprof --print-gpu-trace --profile-child-processes python launch.py op --world_size 4 --op nop --configs 1,64
```

Profiling: 
```sh
sudo PATH=$PATH /usr/local/cuda/bin/nvprof \
    --devices 0 \
    --profile-child-processes \
    --metrics inst_per_warp,inst_executed,ipc \
    -o result-%p.nvvp \
    --print-gpu-trace \
    python launch.py op --op nop -c 1,256 2,128 2,256 -n 1 --comm_size 1
```

```sh
sudo PATH=$PATH /usr/local/cuda/bin/nvprof \
    --profile-child-processes \
    -o result-%p.nvvp \
    python {args}
```