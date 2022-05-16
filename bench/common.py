import os
import argparse
import time
import platform
from pathlib import Path

def get_default_group_rank():
    hostname = platform.node()
    simple_name = hostname.split(".")[0]
    if simple_name.startswith("node"):
        return int(simple_name.removeprefix("node"))
    return 0


def gen_ts_str():
    return time.strftime("%Y%m%d-%H%M%S")

def parse_configs(configs):
    result = []
    for config in configs:
        nchannels, nthreads = config.split(",")
        # config could be "nchannels,nthreads" 

def add_common_args(parser):
    """
    arguments shared by launcher and all workers
    """
    parser.add_argument("-np", "--nprocs", type=int)
    parser.add_argument("-nn", "--nnodes", type=int, default=1)
    parser.add_argument("-gr", "--group_rank", type=int, default=get_default_group_rank())
    parser.add_argument("-m", "--master_addr", type=str, default="localhost:23456")
    parser.add_argument("-d", "--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("results")/gen_ts_str())
    parser.add_argument("-c", "--configs", nargs="+", default=["all"])
    parser.add_argument("-mc", "--max_nc", type=int, default=16)
    parser.add_argument("-ni", "--niter", type=int, default=5)
    parser.add_argument("-nw", "--nwarmup", type=int, default=1)
    parser.add_argument("--p2p", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
