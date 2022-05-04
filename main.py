"""
nvprof --profile-child-processes -s -f -o profile-%p.nvvp python main.py
"""

import os
import time
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models
import torch.multiprocessing as mp
from torch.profiler import profile, ProfilerActivity


def print_profiling_info(prof):
    # print(prof.key_averages(group_by_stack_n=2).table(sort_by="cuda_time_total", row_limit=10))
    comm_events = []
    comp_events = []

    for event in prof.events():
        if event.cuda_time_total == 0:
            continue
        if "nccl" in event.name:
            comm_events.append(event)
        else:
            comp_events.append(event)

    comp_time = sum(e.cuda_time_total for e in comp_events) / 1e3
    comm_time = sum(e.cuda_time_total for e in comm_events) / 1e3

    overlap_time = 0
    for comm_event in comm_events:
        for comp_event in comp_events:
            if comm_event.time_range.start > comp_event.time_range.end:
                continue
            if comm_event.time_range.end < comp_event.time_range.start:
                continue
            min_end = min(comm_event.time_range.end, comp_event.time_range.end)
            max_start = max(comm_event.time_range.start,
                            comp_event.time_range.start)
            overlap_time += (min_end - max_start) / 1e3

    print(f"{comp_time:6.3f}, {comm_time:6.3f}, {overlap_time:6.3f}")


def train(rank, args):
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:23456",
        world_size=args.world_size,
        rank=rank,
    )

    model = getattr(models, args.model_name)().to(rank)
    ddp_model = DDP(model)
    data = torch.randn(args.batch_size, 3, 224, 224, device=rank)
    target = torch.randint(0, 1000, (args.batch_size,), device=rank)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    for i in range(args.niter):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            start_ts = time.time()
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            end_ts = time.time()

        if rank == 0 and i == args.niter - 1:
            print_profiling_info(prof)

        throughput = args.batch_size / (end_ts - start_ts)
        cpu_time = end_ts - start_ts
        print(f"[{rank}] iter {i}: {throughput:.2f} it/sec, {cpu_time:.3f} sec")


def add_common_args(parser):
    parser.add_argument("-w", "--world_size", type=int, default=2)
    parser.add_argument("-c", "--configs", nargs="+", default=["2,256"])
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-g", "--grid", action="store_true")


def run_configs(fn, args):
    if args.grid:
        args.configs = ["0,0"]
        args.configs += [f"{i},{2 ** j}" for i in range(1, 5) for j in range(6, 10)]
    print(args)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    if args.debug:
            os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
            os.environ["NCCL_DEBUG"] = "INFO"
    for config in args.configs:
        nchannels, nthreads = map(int, config.split(","))
        args.nchannels = nchannels
        args.nthreads = nthreads
        if nchannels != 0 and nthreads != 0:
            os.environ["NCCL_MIN_NCHANNELS"] = str(nchannels)
            os.environ["NCCL_MAX_NCHANNELS"] = str(nchannels)
            os.environ["NCCL_NTHREADS"] = str(nthreads)
        print(f"{nchannels},{nthreads:4}, ", end="")
        mp.spawn(fn, args=(args,), nprocs=args.world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, default="resnet50")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-n", "--niter", type=int, default=3)
    add_common_args(parser)
    args = parser.parse_args()
    run_configs(train, args)
