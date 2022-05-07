# nccl-sched
 
```
make -C nccl NVCC_GENCODE="-arch=native" -j
make -C nccl-tests NCCL_HOME=$(readlink -f nccl/build) -j
NCCL_P2P_DISABLE=1 LD_PRELOAD=$(readlink -f nccl/build/lib/libnccl.so) ./nccl-tests/build/all_reduce_perf -g 2 -b 8 -e 100M -f 2
```



NCCL_MIN_NCHANNELS
NCCL_MAX_NCHANNELS
NCCL_NTHREADS
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

# install java 8 and run nvvp
sudo apt install openjdk-8-jdk
nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java python main.py

# profile main.py
nvprof python main.py
nvprof --profile-child-processes -s -f -o profile-%p.nvvp {cmd}
```
