import argparse
import os

from torchvision import models
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
import torch.distributed as dist
import torch.multiprocessing as mp

from main import print_profiling_info, run_configs, add_common_args


world_size = 2

def test(rank, args):
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:23456", world_size=world_size, rank=rank)
    resnet18 = models.resnet18().cuda(rank)
    comm_tensor = torch.zeros(100, 1024, 1024, device=rank)
    if args.op == "conv":
        comp = resnet18.layer1[0].conv1
        comp_tensor = torch.zeros(256, 64, 224, 224, device=rank)
    elif args.op == "bn":
        comp = resnet18.layer1[0].bn1
        comp_tensor = torch.zeros(256, 64, 224, 224, device=rank)
    elif args.op == "relu":
        comp = resnet18.layer1[0].relu
        comp_tensor = torch.zeros(256, 64, 224, 224, device=rank)
    elif args.op == "avg_pool":
        comp = resnet18.avgpool
        comp_tensor = torch.zeros(256, 64, 224, 224, device=rank)
    elif args.op == "fc":
        comp = resnet18.fc
        comp_tensor = torch.zeros(1000, device=rank)
    else:
        raise ValueError(f"Unknown op: {args.op}")
    
    
    niter = 2
    for i in range(niter):
        with profile(activities=[ProfilerActivity.CUDA]) as perf:
            print(f"Iteration {i} before all_reduce")
            handle = dist.all_reduce(comm_tensor, async_op=True)
            print(f"Iteration {i} after all_reduce")
            comp(comp_tensor)
            print(f"Iteration {i} after comp")
            handle.wait()
            print(f"Iteration {i} after wait")
        if i != 0 and rank == 0:
            print_profiling_info(perf)
            # print(perf.key_averages().table(sort_by="cuda_time_total"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default="conv")
    add_common_args(parser)
    args = parser.parse_args()
    run_configs(test, args)