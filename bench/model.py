import argparse
import sys
import time

import torch
import torch.nn.functional as F
import torchvision

from args import add_common_args, setup_args
from utils import get_profiling_info, init_torch_dist, get_profiler, set_config_env


def get_model_and_optimizer(framework, model_name, rank):
    model = getattr(torchvision.models, model_name)()
    if framework == "torch":
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model.to(rank))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif framework == "hvd":
        import horovod.torch as hvd

        model = model.to(rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters()
        )
    return model, optimizer


def bench_model(
    rank,
    args,
    model_name,
    model,
    optimizer,
    batch_size,
    data,
    target,
    nchannels,
    nthreads,
    fout,
):
    set_config_env(nchannels, nthreads)
    for i in range(args.niter + args.nwarmup):
        with get_profiler(args) as prof:
            start_ts = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(rank)
            end_ts = time.time()
        if i < args.nwarmup:
            continue
        cpu_time = (end_ts - start_ts) * 1e3
        throughput = batch_size / (end_ts - start_ts)
        comp_time, comm_time, overlap_time = get_profiling_info(prof)
        for file in [fout, sys.stdout]:
            print(
                rank,
                args.framework,
                model_name,
                batch_size,
                nchannels,
                nthreads,
                comp_time,
                comm_time,
                overlap_time,
                cpu_time,
                throughput,
                file=file,
                flush=True,
                sep="\t",
            )


def bench_models(rank, args, fout):
    for model_name in args.model_names:
        model, optimizer = get_model_and_optimizer(args.framework, model_name, rank)
        for batch_size in args.batch_sizes:
            data = torch.randn(batch_size, 3, 224, 224, device="cuda")
            target = torch.randint(0, 1000, (batch_size,), device="cuda")
            for nchannels, nthreads in args.configs:
                bench_model(
                    rank=rank,
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    fout=fout,
                    batch_size=batch_size,
                    data=data,
                    target=target,
                    nchannels=nchannels,
                    nthreads=nthreads,
                    model_name=model_name,
                )


def main(rank, args):
    if args.framework == "torch":
        init_torch_dist(rank, args)
    elif args.framework == "hvd":
        import horovod.torch as hvd

        hvd.init()
        rank = hvd.local_rank()
    torch.cuda.set_device(rank)

    with open(args.output_dir / "result.csv", "a") as fout:
        print(
            "rank",
            "framework",
            "model_name",
            "batch_size",
            "nchannels",
            "nthreads",
            "comp_time",
            "comm_time",
            "overlap_time",
            "cpu_time",
            "throughput",
            sep="\t",
            file=fout,
            flush=True,
        )
        bench_models(rank, args, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Benchmark Model",
        usage="""
    Using torch.DDP for distributed training:
        python model.py -np 2
    Using horovod for distributed training:
        python model.py -f hvd -w 2
        horovodrun -np 2 python model.py --no-spawn -f hvd
    """,
    )
    add_common_args(parser)
    parser.set_defaults(configs=["1-4,64", "1-4,128", "1-4,256", "1-4,512"])
    parser.add_argument(
        "-f",
        "--framework",
        type=str,
        default="torch",
        choices=["torch", "hvd"],
        help="framework used for distributed training",
    )
    parser.add_argument(
        "-m",
        "--model_names",
        type=str,
        nargs="+",
        default=["resnet18", "resnet50"],
        help="model names",
    )
    parser.add_argument(
        "-b",
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="local batch sizes",
    )
    parser.add_argument(
        "--spawn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="spawn processes for distributed training",
    )

    args = parser.parse_args()

    setup_args(args)

    if args.framework == "torch":
        if args.spawn:
            import torch.multiprocessing as mp

            mp.spawn(main, args=(args,), nprocs=args.nprocs)
    elif args.framework == "hvd":
        if args.spawn:
            import horovod

            horovod.run(main, args=(-1, args,), np=args.nprocs)
        else:
            main(-1, args)
