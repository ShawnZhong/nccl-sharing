"""
nvprof --profile-child-processes -s -f -o profile-%p.nvvp python main.py
"""

import os
import time
import argparse

from common import add_common_args


def get_profiling_info(prof):
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
            max_start = max(comm_event.time_range.start,
                            comp_event.time_range.start)
            overlap_time += (min_end - max_start) / 1e3

    return comp_time, comm_time, overlap_time


def get_config():
    nminchannels = os.environ.get("NCCL_MIN_NCHANNELS") or -1
    nmaxchannels = os.environ.get("NCCL_MAX_NCHANNELS") or -1
    nthreads = os.environ.get("NCCL_NTHREADS") or -1
    if nminchannels != nmaxchannels:
        raise ValueError("MAX_NCHANNELS and MIN_NCHANNELS must be equal")
    return int(nminchannels), int(nthreads)


def run(rank, model, optimizer, batch_size, niter, framework, model_name, **kwargs):
    import torch
    import torch.nn.functional as F
    from torch.profiler import profile, ProfilerActivity

    data = torch.randn(batch_size, 3, 224, 224, device=rank)
    target = torch.randint(0, 1000, (batch_size,), device=rank)
    for i in range(niter):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            start_ts = time.time()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(rank)
            end_ts = time.time()

        cpu_time = (end_ts - start_ts) * 1e3
        throughput = batch_size / (end_ts - start_ts)
        comp_time, comm_time, overlap_time = get_profiling_info(prof)
        nchannels, nthreads = get_config()
        print(
            f"{rank}, "
            f"{framework}, "
            f"{model_name}, "
            f"{batch_size}, "
            f"{nchannels}, "
            f"{nthreads:4}, "
            f"{comp_time:8.3f}, "
            f"{comm_time:8.3f}, "
            f"{overlap_time:8.3f}, "
            f"{cpu_time:8.3f}, "
            f"{throughput:8.3f}"
        )


def add_worker_args(parser):
    add_common_args(parser)
    parser.add_argument("-nc", "--nchannels", type=int, default=2)
    parser.add_argument("-nt", "--nthreads", type=int, default=256)


def set_worker_env(args):
    os.environ["NCCL_P2P_DISABLE"] = "1"
    if args.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["NCCL_MIN_NCHANNELS"] = str(args.nchannels)
    os.environ["NCCL_MAX_NCHANNELS"] = str(args.nchannels)
    os.environ["NCCL_NTHREADS"] = str(args.nthreads)


def main():
    import torch
    import torchvision

    parser = argparse.ArgumentParser()
    add_worker_args(parser)
    parser.add_argument("-m", "--model_name", type=str, default="resnet50")
    parser.add_argument("-f", "--framework", type=str,
                        default="torch", choices=["torch", "hvd"])
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    args = parser.parse_args()
    print(args)
    set_worker_env(args)

    # initialize distributed backend
    if args.framework == "torch":
        import torch.distributed as dist
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12345"
        rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", rank=rank)
    elif args.framework == "hvd":
        import horovod.torch as hvd
        hvd.init()
        rank = hvd.local_rank()

    torch.cuda.set_device(rank)
    model = getattr(torchvision.models, args.model_name)().to(rank)

    if args.framework == "torch":
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif args.framework == "hvd":
        import horovod.torch as hvd
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters()
        )

    run(rank, model, optimizer, **vars(args))


if __name__ == "__main__":
    main()
