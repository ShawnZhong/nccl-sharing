import argparse
import time
import os
import sys
import contextlib

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from utils import (
    add_common_args,
    setup_args,
    get_profiling_info,
    init_torch_dist,
)
from plot import plot_op_results


all_ops = ["nop", "conv", "bn", "relu", "avgpool", "fc"]


def bench_op(op, nchannels, nthreads, comp, comp_tensor, comm_tensor, fout, args):
    comm_enabled = nchannels != 0 and nthreads != 0
    if comm_enabled:
        os.environ["NCCL_LOCAL_NCHANNELS"] = str(nchannels)
        os.environ["NCCL_LOCAL_NTHREADS"] = str(nthreads)

    for i in range(args.niter + args.nwarmup):
        if args.profile:
            from torch.profiler import profile, ProfilerActivity

            cm = profile(activities=[ProfilerActivity.CUDA])
        else:
            cm = contextlib.nullcontext()
        torch.cuda.synchronize(args.local_rank)
        with cm as perf:
            start_ts = time.time()
            if comm_enabled:
                handle = dist.all_reduce(comm_tensor, async_op=True)
            if op != "nop":
                comp(comp_tensor)
            if comm_enabled:
                handle.wait()
            torch.cuda.synchronize(args.local_rank)
            end_ts = time.time()
        if i < args.nwarmup:
            continue
        cpu_time = (end_ts - start_ts) * 1e3
        comp_time, comm_time, overlap_time = get_profiling_info(perf)
        for file in [fout, sys.stdout]:
            print(
                args.global_rank,
                op,
                nchannels,
                nthreads,
                comp_time,
                comm_time,
                overlap_time,
                cpu_time,
                sep="\t",
                file=file,
                flush=True,
            )


def bench_ops(args, fout):
    comp_dict = {
        "conv": nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False),
        "bn": nn.BatchNorm2d(64),
        "relu": nn.ReLU(inplace=True),
        "avgpool": nn.AdaptiveAvgPool2d((1, 1)),
        "fc": nn.Linear(224 * 224, 64),
    }

    comm_tensor = torch.zeros(args.comm_size, 1024, 1024, device="cuda")
    for op in args.ops:
        comp = comp_dict[op].cuda() if op != "nop" else None
        comp_tensor = torch.zeros(args.batch_size, 64, 224, 224, device="cuda")
        if op == "fc":
            comp_tensor = comp_tensor.reshape(args.batch_size, 64, -1)
        for nchannels, nthreads in args.configs:
            bench_op(
                op=op,
                nchannels=nchannels,
                nthreads=nthreads,
                comp=comp,
                comp_tensor=comp_tensor,
                comm_tensor=comm_tensor,
                fout=fout,
                args=args,
            )


def main(local_rank, args):
    init_torch_dist(local_rank, args)
    with open(args.output_dir / f"result-{args.global_rank}.csv", "a") as fout:
        print(
            "rank",
            "op",
            "nchannels",
            "nthreads",
            "comp_time",
            "comm_time",
            "overlap_time",
            "cpu_time",
            sep="\t",
            file=fout,
            flush=True,
        )
        bench_ops(args, fout)

    if local_rank == 0:
        plot_op_results(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.set_defaults(configs=["0,0", "1-4,64", "1-4,128", "1-4,256", "1-4,512"])
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=all_ops,
        default=all_ops,
        help="operations to benchmark",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="the local batch size for each op",
    )
    parser.add_argument(
        "-cs",
        "--comm_size",
        type=int,
        default=300,
        help="the size of NCCL AllReduce tensor",
    )
    args = parser.parse_args()

    setup_args(args)

    mp.spawn(main, args=(args,), nprocs=args.nprocs)
