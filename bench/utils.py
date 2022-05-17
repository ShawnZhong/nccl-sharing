import os
import argparse
import time
import platform
from pathlib import Path
from typing import List
from itertools import product


def get_default_group_rank():
    hostname = platform.node()
    simple_name = hostname.split(".")[0]
    if simple_name.startswith("node"):
        return int(simple_name.removeprefix("node"))
    return 0


def gen_ts_str():
    return time.strftime("%Y%m%d-%H%M%S")


def parse_range(range_str):
    if ":" in range_str:
        range_str, step_str = range_str.split(":")
        step = int(step_str)
    else:
        step = 1

    if "-" in range_str:
        start, end = map(int, range_str.split("*"))
        return range(start, end + step, step)
    else:
        return [int(range_str)]


def parse_config(config: str):
    nc_str, nt_str = config.split(",")
    nchannels = parse_range(nc_str)
    nthreads = parse_range(nt_str)
    return list(product(nchannels, nthreads))


def parse_configs(configs):
    return [config for c in configs for config in parse_config(c)]


def add_common_args(parser):
    parser.add_argument(
        "-np",
        "--nprocs",
        type=int,
        help="the number of processes (or GPUs) for each node",
    )
    parser.add_argument(
        "-nn",
        "--nnodes",
        type=int,
        default=1,
        help="the number of nodes in the cluster",
    )
    parser.add_argument(
        "-gr",
        "--group_rank",
        type=int,
        default=get_default_group_rank(),
        help="rank of the current node",
    )
    parser.add_argument(
        "-ma",
        "--master_addr",
        type=str,
        default="localhost:23456",
        help="master address",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="debug mode",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("results") / gen_ts_str(),
        help="output directory",
    )
    parser.add_argument(
        "-c",
        "--configs",
        nargs="+",
        type=str,
        help="configs. e.g., 1-2,32-64:32 means 1-2 channels, 32-64 threads",
    )
    parser.add_argument(
        "-mc",
        "--max_nc",
        type=int,
        default=16,
        help="max number of channels created during ncclInit",
    )
    parser.add_argument(
        "-ni",
        "--niter",
        type=int,
        default=5,
        help="number of training iterations per config",
    )
    parser.add_argument(
        "-nw",
        "--nwarmup",
        type=int,
        default=1,
        help="number of warmup iterations per config",
    )
    parser.add_argument(
        "--p2p",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use p2p",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether or not to gather profiling info",
    )


def get_profiling_info(prof):
    if prof is None:
        return 0, 0, 0

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
            max_start = max(comm_event.time_range.start, comp_event.time_range.start)
            overlap_time += (min_end - max_start) / 1e3

    return comp_time, comm_time, overlap_time


def get_nccl_path():
    root_dir = Path(__file__).parent.parent
    nccl_path = root_dir / "nccl" / "build" / "lib" / "libnccl.so"
    if not nccl_path.exists():
        raise FileNotFoundError(f"{nccl_path} not found")
    return nccl_path


def setup_args(args):
    args.configs = parse_configs(args.configs)

    print(args)

    if not args.p2p:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    if args.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_MIN_NCHANNELS"] = str(args.max_nc)
    os.environ["NCCL_MAX_NCHANNELS"] = str(args.max_nc)
    os.environ["LD_PRELOAD"] = str(get_nccl_path())

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)


def init_torch_dist(local_rank, args):
    import torch

    args.local_rank = local_rank
    args.global_rank = args.group_rank * args.nprocs + local_rank
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}",
        world_size=args.nnodes * args.nprocs,
        rank=args.global_rank,
    )
    torch.cuda.set_device(local_rank)
