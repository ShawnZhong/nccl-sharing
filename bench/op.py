import argparse
import time
import os


from model import get_profiling_info, get_config, add_worker_args, set_worker_env

all_ops = ["nop", "conv", "bn", "relu", "avgpool", "fc"]

def bench_op(rank, op, args, fout):
    import torch
    import torch.nn as nn
    from torch.profiler import profile, ProfilerActivity
    import torch.distributed as dist

    comp_dict = {
        "conv": nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False),
        "bn": nn.BatchNorm2d(64),
        "relu": nn.ReLU(inplace=True),
        "avgpool": nn.AdaptiveAvgPool2d((1, 1)),
        "fc": nn.Linear(224 * 224, 64),
    }

    if op != "nop":
        comp = comp_dict[op].cuda(rank)
    if op in ["conv", "bn", "relu", "avgpool"]:
        comp_tensor = torch.zeros(args.batch_size, 64, 224, 224, device=rank)
    elif op == "fc":
        comp_tensor = torch.zeros(args.batch_size, 64, 224 * 224, device=rank)

    nchannels, nthreads = get_config()
    comm_enabled = nchannels != 0 and nthreads != 0
    if comm_enabled:
        comm_tensor = torch.zeros(args.comm_size, 1024, 1024, device=rank)

    for i in range(args.niter):
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
        cpu_time = (end_ts - start_ts) * 1e3
        comp_time, comm_time, overlap_time = get_profiling_info(perf)
        msg = (
            f"{rank}, "
            f"{op:8}, "
            f"{nchannels}, "
            f"{nthreads:4}, "
            f"{comp_time:8.3f}, "
            f"{comm_time:8.3f}, "
            f"{overlap_time:8.3f}, "
            f"{cpu_time:8.3f}"
        )
        print(msg)
        fout.write(msg + "\n")


def main(rank, args):
    import torch.distributed as dist

    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:23456",
        world_size=args.world_size,
        rank=rank
    )

    with open(args.output_dir / "result.csv", "a") as fout:
        for op in args.ops:
            bench_op(rank, op, args, fout)


def add_op_args(parser):
    parser.add_argument(
        "--ops", nargs="+", 
        choices=all_ops, 
        default=all_ops
    )
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=64,
    )
    parser.add_argument(
        "--comm_size", type=int, default=25,
        help="Communication size in MB"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_worker_args(parser)
    add_op_args(parser)
    args = parser.parse_args()
    print(args)
    set_worker_env(args)

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "result.csv"
    if not result_path.exists():
        with open(result_path, "w") as fout:
            fout.write(
                "rank, op, nchannels, nthreads, comp_time, comm_time, overlap_time, cpu_time\n")

    if args.spawn:
        import torch.multiprocessing as mp
        mp.spawn(main, args=(args,), nprocs=args.world_size)
    else:
        if "LOCAL_RANK" not in os.environ:
            raise ValueError(f"LOCAL_RANK not set. {parser.format_usage()}")
        rank = int(os.environ["LOCAL_RANK"])
        main(rank, args)
