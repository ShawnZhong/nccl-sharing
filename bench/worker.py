from pathlib import Path
import os

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

def get_nccl_path():
    root_dir = Path(__file__).parent.parent
    nccl_path = root_dir / "nccl" / "build" / "lib" / "libnccl.so"
    if not nccl_path.exists():
        raise FileNotFoundError(f"{nccl_path} not found")
    return nccl_path

def set_worker_env(args):
    if not args.p2p:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    if args.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["NCCL_MIN_NCHANNELS"] = "8"
    os.environ["NCCL_MAX_NCHANNELS"] = "8"
    os.environ["LD_PRELOAD"] = str(get_nccl_path())