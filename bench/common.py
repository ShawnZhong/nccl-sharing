import os
import argparse

def add_common_args(parser):
    parser.add_argument("-w", "--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 2)))
    parser.add_argument("-d", "--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-n", "--niter", type=int, default=10)