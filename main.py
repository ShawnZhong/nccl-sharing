"""
nvprof --profile-child-processes -s -f -o profile-%p.nvvp python main.py
"""

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models
import torch.multiprocessing as mp

def main(rank, world_size, model_name, batch_size, nchannels, nthreads):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_MIN_NCHANNELS"] = str(nchannels)
    os.environ["NCCL_MAX_NCHANNELS"] = str(nchannels)
    os.environ["NCCL_NTHREADS"] = str(nthreads)

    dist.set_debug_level(dist.DebugLevel.DETAIL)
    print(dist.get_debug_level())

    dist.init_process_group(
        backend='nccl', 
        init_method='tcp://localhost:23456', 
        world_size=world_size, 
        rank=rank
    )

    model = getattr(models, model_name)().to(rank)
    ddp_model = DDP(model)
    data = torch.randn(batch_size, 3, 224, 224, device=rank)
    target = torch.randint(0, 1000, (batch_size,), device=rank)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    start_ts = time.time()
    for i in range(10):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    end_ts = time.time()
    print(f"Rank {rank} took {end_ts - start_ts} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--world_size", type=int, default=2)
    parser.add_argument("-m", "--model_name", type=str, default="resnet50")
    parser.add_argument("-b", "--batch_size", type=int, default=256)
    parser.add_argument("-c", "--nchannels", type=int, nargs="+", default=[2])
    parser.add_argument("-t", "--nthreads", type=int, nargs="+", default=[256])
    args = parser.parse_args()
    for nchannels in args.nchannels:
        for nthreads in args.nthreads:
            print(f"nchannels = {nchannels}, nthreads = {nthreads}")
            mp.spawn(main, args=(args.world_size, args.model_name, args.batch_size, nchannels, nthreads), nprocs=args.world_size)