import os
import argparse
import time
from pathlib import Path

def gen_ts_str():
    return time.strftime("%Y%m%d-%H%M%S")

def add_common_args(parser):
    """
    arguments shared by launcher and all workers
    """
    parser.add_argument("-np", "--nprocs", type=int)
    parser.add_argument("-nn", "--nnodes", type=int, default=1)
    parser.add_argument("-gr", "--group_rank", type=int, default=0)
    parser.add_argument("-m", "--master_addr", type=str, default="localhost:23456")
    parser.add_argument("-d", "--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("results")/gen_ts_str())
    parser.add_argument("-c", "--configs", nargs="+", default=["all"])
    parser.add_argument("--niter", type=int, default=5)
    parser.add_argument("--nwarmup", type=int, default=1)
    parser.add_argument("--p2p", action=argparse.BooleanOptionalAction, default=True)
