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
    parser.add_argument("-c", "--configs", nargs="+", default=["all"])
    parser.add_argument("-w", "--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 2)))
    parser.add_argument("-d", "--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-o", "--output_dir", type=Path, default=Path("results")/gen_ts_str())
    parser.add_argument("-n", "--niter", type=int, default=5)
    parser.add_argument("--nwarmup", type=int, default=1)
    parser.add_argument("--p2p", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spawn", action=argparse.BooleanOptionalAction, default=True)

