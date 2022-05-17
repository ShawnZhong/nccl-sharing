def get_profiler(args):
    if not args.profile:
        import contextlib

        return contextlib.nullcontext()
    else:
        from torch.profiler import profile, ProfilerActivity

        return profile(activities=[ProfilerActivity.CUDA])


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


def set_config_env(nchannels, nthreads):
    import os

    os.environ["NCCL_LOCAL_NCHANNELS"] = str(nchannels)
    os.environ["NCCL_LOCAL_NTHREADS"] = str(nthreads)
