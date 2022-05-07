import argparse
import time
import os

from torchvision import models
import torch
from torch.profiler import profile, ProfilerActivity
import torch.distributed as dist

from model import get_profiling_info, run_configs, add_common_args, parse_args

comp_dict = {
    "conv": models.resnet18().layer1[0].conv1,
    "bn": models.resnet18().layer1[0].bn1,
    "relu": models.resnet18().layer1[0].relu,
    "avgpool": models.resnet18().avgpool,
    "fc": models.resnet18().fc,
}

all_ops = list(comp_dict.keys()) + ["nop"]


def test(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:23456",
        world_size=args.world_size,
        rank=rank
    )
    comp = comp_dict[args.op].cuda(rank)
    if args.op in ["conv", "bn", "relu", "avgpool"]:
        comp_tensor = torch.zeros(256, 64, 224, 224, device=rank)
    elif args.op == "fc":
        comp_tensor = torch.zeros(256, 512, device=rank)

    comm_tensor = torch.zeros(100, 1024, 1024, device=rank)

    for i in range(args.niter):
        with profile(activities=[ProfilerActivity.CUDA]) as perf:
            start_ts = time.time()
            if args.comp_only or args.nchannels == 0 or args.nthreads == 0:
                comp(comp_tensor)
            elif args.comm_only:
                dist.all_reduce(comm_tensor)
            else:
                handle = dist.all_reduce(comm_tensor, async_op=True)
                comp(comp_tensor)
                handle.wait()
            torch.cuda.synchronize(rank)
            end_ts = time.time()
        if rank == 0 and i == args.niter - 1:
            cpu_time = (end_ts - start_ts) * 1e3
            comp_time, comm_time, overlap_time = get_profiling_info(perf)
            print(
                f"{args.op:8}, "
                f"{args.nchannels}, "
                f"{args.nthreads:4}, "
                f"{comp_time:8.3f}, "
                f"{comm_time:8.3f}, "
                f"{overlap_time:8.3f}, "
                f"{cpu_time:8.3f}"
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops", nargs="+", default=all_ops)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--comp_only", action="store_true")
    group.add_argument("--comm_only", action="store_true")
    add_common_args(parser)
    args = parse_args(parser)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    for op in args.ops:
        args.op = op
        run_configs(test, args)
