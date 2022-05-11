import argparse
import time
import os
import sys

import pandas as pd

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity
import torch.distributed as dist
import torch.multiprocessing as mp

from common import add_common_args
from worker import set_worker_env, get_profiling_info

pd.options.display.expand_frame_repr = False

all_ops = ["nop", "conv", "bn", "relu", "avgpool", "fc"]


def bench_op(op, nchannels, nthreads, comp, comp_tensor, comm_tensor, args):
    comm_enabled = nchannels != 0 and nthreads != 0
    if comm_enabled:
        os.environ["NCCL_LOCAL_NCHANNELS"] = str(nchannels)
        os.environ["NCCL_LOCAL_NTHREADS"] = str(nthreads)

    for i in range(args.niter + args.nwarmup):
        with profile(activities=[ProfilerActivity.CUDA]) as perf:
            start_ts = time.time()
            if comm_enabled:
                handle = dist.all_reduce(comm_tensor, async_op=True)
            if op != "nop":
                comp(comp_tensor)
            if comm_enabled:
                handle.wait()
            torch.cuda.synchronize()
            end_ts = time.time()
        if i < args.nwarmup:
            continue
        cpu_time = (end_ts - start_ts) * 1e3
        comp_time, comm_time, overlap_time = get_profiling_info(perf)
        for file in [args.fout, sys.stdout]:
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


def bench_ops(args):
    comp_dict = {
        "conv": nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False),
        "bn": nn.BatchNorm2d(64),
        "relu": nn.ReLU(inplace=True),
        "avgpool": nn.AdaptiveAvgPool2d((1, 1)),
        "fc": nn.Linear(224 * 224, 64),
    }

    torch.cuda.set_device(args.local_rank)

    comm_tensor = torch.zeros(args.comm_size, 1024, 1024).cuda()
    for op in args.ops:
        comp = comp_dict[op].cuda() if op != "nop" else None
        comp_tensor = torch.zeros(args.batch_size, 64, 224, 224).cuda()
        if op == "fc":
            comp_tensor = comp_tensor.reshape(args.batch_size, 64, -1)
        for config in args.configs:
            nchannels, nthreads = map(int, config.split(","))
            bench_op(
                op=op,
                nchannels=nchannels,
                nthreads=nthreads,
                comp=comp,
                comp_tensor=comp_tensor,
                comm_tensor=comm_tensor,
                args=args,
            )


def main(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.group_rank * args.nprocs + local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}",
        world_size=args.nnodes * args.nprocs,
        rank=args.global_rank,
    )
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
        args.fout = fout
        bench_ops(args)

    if local_rank == 0:
        files = args.output_dir.glob("*.csv")
        df = pd.concat(
            [pd.read_csv(f, sep="\t") for f in files], ignore_index=True, sort=False
        )
        print(
            df.pivot_table(
                index=["op", "nchannels"],
                columns="nthreads",
                values=["comp_time", "comm_time"],
                aggfunc="median",
                sort=False,
            ).reindex(["comp_time", "comm_time"], axis=1, level=0)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        "--ops", nargs="+", choices=all_ops + ["all"], default=["all"],
    )
    parser.add_argument(
        "--bs", "--batch_size", dest="batch_size", type=int, default=64,
    )
    parser.add_argument(
        "--comm_size", type=int, default=300, help="Communication size in MB"
    )
    args = parser.parse_args()

    if "all" in args.ops:
        args.ops = all_ops

    if "all" in args.configs:
        args.configs = ["0,0"]
        args.configs += [f"{i},{2 ** j}" for i in range(1, 9) for j in range(6, 10)]

    print(args)
    set_worker_env(args)

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    mp.spawn(main, args=(args,), nprocs=args.nprocs)
