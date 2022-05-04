import argparse
import os

from torchvision import models
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
import torch.distributed as dist

from main import print_profiling_info, run_configs, add_common_args


def test(rank, args):
    dist.init_process_group(
        backend="nccl", 
        init_method="tcp://localhost:23456", 
        world_size=args.world_size, 
        rank=rank
    )

    resnet18 = models.resnet18().cuda(rank)
    comm_tensor = torch.zeros(100, 1024, 1024, device=rank)
    comp_dict = {
        "conv": resnet18.layer1[0].conv1,
        "bn": resnet18.layer1[0].bn1,
        "relu": resnet18.layer1[0].relu,
        "avgpool": resnet18.avgpool,
        "fc": resnet18.fc,
    }
    comp = comp_dict[args.op]

    if args.op in ["conv", "bn", "relu", "avgpool"]:
        comp_tensor = torch.zeros(256, 64, 224, 224, device=rank)
    elif args.op == "fc":
        comp_tensor = torch.zeros(256, 512, device=rank)
    
    niter = 2
    for i in range(niter):
        with profile(activities=[ProfilerActivity.CUDA]) as perf:
            if args.nchannels == 0 or args.nthreads == 0:
                comp(comp_tensor)
            else:
                handle = dist.all_reduce(comm_tensor, async_op=True)
                comp(comp_tensor)
                handle.wait()
        if i != 0 and rank == 0:
            print_profiling_info(perf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str)
    add_common_args(parser)
    args = parser.parse_args()
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    run_configs(test, args)