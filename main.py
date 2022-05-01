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
from torch.profiler import profile, record_function, ProfilerActivity


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

    comp_time = sum(e.cuda_time_total for e in comp_events) / 1e6
    comm_time = sum(e.cuda_time_total for e in comm_events) / 1e6

    overlap_time = 0
    for comm_event in comm_events:
        for comp_event in comp_events:
            if (comm_event.time_range.start > comp_event.time_range.end or comm_event.time_range.end < comp_event.time_range.start):
                continue
            overlap_time += min(comm_event.time_range.end, comp_event.time_range.end) - max(comm_event.time_range.start, comp_event.time_range.start)
    overlap_time /= 1e6

    print(f"comp time: {comp_time:.3f}, comm time: {comm_time:.3f}, cuda time:{comp_time + comm_time:.3f}, overlap time: {overlap_time:.3f}")


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
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = F.cross_entropy(output, target)
        start_ts = time.time()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            loss.backward()
        end_ts = time.time()

        if rank == 0 and i == 2:
            print_profiling_info(prof)
        optimizer.step()
        throughput = args.batch_size / (end_ts - start_ts)
        print(f"[{rank}] iter {i}: {throughput:.2f} it/sec, {end_ts - start_ts:.3f} sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--world_size", type=int, default=2)
    parser.add_argument("-m", "--model_name", type=str, default="resnet50")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-c", "--configs", type=str,
                        nargs="+", default=["2,256"])
    parser.add_argument("-n", "--niter", type=int, default=3)
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()
    print(args)
    for config in args.configs:
        nchannels, nthreads = map(int, config.split(","))
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_MIN_NCHANNELS"] = str(nchannels)
        os.environ["NCCL_MAX_NCHANNELS"] = str(nchannels)
        os.environ["NCCL_NTHREADS"] = str(nthreads)
        if args.debug:
            os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
            os.environ["NCCL_DEBUG"] = "INFO"
        print(f"nchannels = {nchannels}, nthreads = {nthreads}")
        mp.spawn(train, args=(args, ), nprocs=args.world_size)
