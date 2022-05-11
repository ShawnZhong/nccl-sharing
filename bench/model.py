import os
import time
import argparse

from common import add_common_args





def get_config():
    nminchannels = os.environ.get("NCCL_MIN_NCHANNELS") or -1
    nmaxchannels = os.environ.get("NCCL_MAX_NCHANNELS") or -1
    nthreads = os.environ.get("NCCL_NTHREADS") or -1
    if nminchannels != nmaxchannels:
        raise ValueError("MAX_NCHANNELS and MIN_NCHANNELS must be equal")
    return int(nminchannels), int(nthreads)





def main(rank, args):
    import torch
    import torch.nn.functional as F
    from torch.profiler import profile, ProfilerActivity
    import torchvision

    model = getattr(torchvision.models, args.model_name)()

    if args.framework == "torch":
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP

        dist.init_process_group(backend="nccl", rank=rank)
        torch.cuda.set_device(rank)
        model = DDP(model.to(rank))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif args.framework == "hvd":
        import horovod.torch as hvd

        hvd.init()
        rank = hvd.local_rank()
        torch.cuda.set_device(rank)
        model = model.to(rank)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters()
        )

    args.output_dir.mkdir(exist_ok=True, parents=True)
    with open(args.output_dir / "result.csv", "a") as fout:
        data = torch.randn(args.batch_size, 3, 224, 224, device=rank)
        target = torch.randint(0, 1000, (args.batch_size,), device=rank)
        for i in range(args.niter + args.nwarmup):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
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
            throughput = args.batch_size / (end_ts - start_ts)
            comp_time, comm_time, overlap_time = get_profiling_info(prof)
            nchannels, nthreads = get_config()
            msg = (
                f"{rank}, "
                f"{args.framework}, "
                f"{args.model_name}, "
                f"{args.batch_size}, "
                f"{nchannels}, "
                f"{nthreads:4}, "
                f"{comp_time:8.3f}, "
                f"{comm_time:8.3f}, "
                f"{overlap_time:8.3f}, "
                f"{cpu_time:8.3f}, "
                f"{throughput:8.3f}"
            )
            print(msg)
            fout.write(msg + "\n")


def add_worker_args(parser):
    add_common_args(parser)
    parser.add_argument("-nc", "--nchannels", type=int, default=2)
    parser.add_argument("-nt", "--nthreads", type=int, default=256)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark Model", usage="""
    Using torch.DDP for distributed training:
        python model.py -f torch -w 2
        torchrun torchrun --nproc_per_node 2 model.py --no-spawn -f torch
        LOCAL_RANK=0 python model.py --no-spawn -f torch & LOCAL_RANK=1 python model.py --no-spawn -f torch
    Using horovod for distributed training:
        python model.py -f hvd -w 2
        horovodrun -np 2 python model.py --no-spawn -f hvd
    """)
    add_worker_args(parser)
    parser.add_argument("-m", "--model_name", type=str, default="resnet50")
    parser.add_argument("-f", "--framework", type=str,
                        default="torch", choices=["torch", "hvd"])
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    args = parser.parse_args()
    print(args)
    set_worker_env(args)

    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "result.csv"
    if not result_path.exists():
        with open(result_path, "w") as fout:
            fout.write("rank, framework, model_name, batch_size, nchannels, nthreads, comp_time, comm_time, overlap_time, cpu_time, throughput\n")

    if args.framework == "torch":
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12345"

        if args.spawn:
            import torch.multiprocessing as mp
            mp.spawn(main, args=(args,), nprocs=args.world_size)
        else:
            if "LOCAL_RANK" not in os.environ:
                raise ValueError(f"LOCAL_RANK not set. {parser.format_usage()}")
            rank = int(os.environ["LOCAL_RANK"])
            main(rank, args)
    elif args.framework == "hvd":
        if args.spawn:
            import horovod

            horovod.run(main, args=(-1, args,), np=args.world_size)
        else:
            main(-1, args)
