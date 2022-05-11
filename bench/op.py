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


def bench_op(
    rank, op, nchannels, nthreads, comp, comp_tensor, comm_tensor, niter, nwarmup, fout
):
    comm_enabled = nchannels != 0 and nthreads != 0
    if comm_enabled:
        os.environ["NCCL_LOCAL_NCHANNELS"] = str(nchannels)
        os.environ["NCCL_LOCAL_NTHREADS"] = str(nthreads)

    for i in range(niter + nwarmup):
        with profile(activities=[ProfilerActivity.CUDA]) as perf:
            start_ts = time.time()
            if comm_enabled:
                handle = dist.all_reduce(comm_tensor, async_op=True)
            if op != "nop":
                comp(comp_tensor)
            if comm_enabled:
                handle.wait()
            torch.cuda.synchronize(rank)
            end_ts = time.time()
        if i < nwarmup:
            continue
        cpu_time = (end_ts - start_ts) * 1e3
        comp_time, comm_time, overlap_time = get_profiling_info(perf)
        for file in [fout, sys.stdout]:
            print(
                rank,
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


def bench_ops(rank, args, fout):
    comp_dict = {
        "conv": nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False),
        "bn": nn.BatchNorm2d(64),
        "relu": nn.ReLU(inplace=True),
        "avgpool": nn.AdaptiveAvgPool2d((1, 1)),
        "fc": nn.Linear(224 * 224, 64),
    }

    comm_tensor = torch.zeros(args.comm_size, 1024, 1024, device=rank)
    for op in args.ops:
        comp = comp_dict[op].cuda(rank) if op != "nop" else None
        comp_tensor = torch.zeros(args.batch_size, 64, 224, 224, device=rank)
        if op == "fc":
            comp_tensor = comp_tensor.reshape(args.batch_size, 64, -1)
        for config in args.configs:
            nchannels, nthreads = map(int, config.split(","))
            bench_op(
                rank=rank,
                op=op,
                nchannels=nchannels,
                nthreads=nthreads,
                comp=comp,
                comp_tensor=comp_tensor,
                comm_tensor=comm_tensor,
                fout=fout,
                niter=args.niter,
                nwarmup=args.nwarmup,
            )


def main(rank, args):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:23456",
        world_size=args.world_size,
        rank=rank,
    )

    result_csv = args.output_dir / f"result.csv"
    with open(result_csv, "a") as fout:
        bench_ops(rank, args, fout)

    if rank == 0:
        df = pd.read_csv(result_csv, sep="\t")
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
        args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "result.csv"
    if not result_path.exists():
        with open(result_path, "w") as fout:
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
            )

    if args.spawn:
        mp.spawn(main, args=(args,), nprocs=args.world_size)
    else:
        if "LOCAL_RANK" not in os.environ:
            raise ValueError(f"LOCAL_RANK not set. {parser.format_usage()}")
        rank = int(os.environ["LOCAL_RANK"])
        main(rank, args)
