import argparse
import os
import itertools

from tqdm import tqdm

from common import add_common_args


def system(cmd):
    print(f"Running command: `{cmd}`")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Command `{cmd}` failed with exit code {ret}")


def arg_to_str(k, v):
    if isinstance(v, list):
        return f"--{k} {' '.join(map(str, v))}"
    elif isinstance(v, bool):
        return f"--{k}" if v else f"--no-{k}"
    else:
        return f"--{k} {v}"


def run(file, **kwargs):
    configs = kwargs.pop("configs")
    model_names = kwargs.pop("model_names")
    batch_sizes = kwargs.pop("batch_sizes")
    frameworks = kwargs.pop("frameworks")
    combinations = itertools.product(model_names, frameworks, batch_sizes, configs)
    for model_name, framework, batch_size, config in tqdm(list(combinations), colour="green"):
        kwargs["model_name"] = model_name
        kwargs["framework"] = framework
        kwargs["batch_size"] = batch_size
        kwargs["nchannels"], kwargs["nthreads"] = map(int, config.split(","))

        world_size = kwargs["world_size"]
        args = " ".join(map(arg_to_str, kwargs.keys(), kwargs.values()))
        if framework == "torch":
            os.environ["OMP_NUM_THREADS"] = "1"
            system(f"torchrun --nproc_per_node {world_size} {file} {args}")
        elif framework == "hvd":
            system(f"horovodrun -np {world_size} python {file} {args}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="file", required=True)

    # model launcher
    model_parser = subparsers.add_parser("model")
    model_parser.add_argument("-m", "--model_names",
                              nargs="+", default=["resnet50"])
    model_parser.add_argument("-b", "--batch_sizes",
                              nargs="+", default=[32])
    model_parser.add_argument("-f", "--frameworks",
                              nargs="+", default=["torch"], 
                              choices=["torch", "hvd", "all"])

    # op launcher
    op_subparser = subparsers.add_parser("op")

    for subparser in [model_parser, op_subparser]:
        add_common_args(subparser)
        subparser.add_argument("-c", "--configs", nargs="+", default=["2,256"])

    args = parser.parse_args()
    if "all" in args.configs:
        args.configs = [
            f"{i},{2 ** j}"
            for i in range(1, 5)
            for j in range(6, 10)
        ]
    if "all" in args.batch_sizes:
        args.batch_sizes = [16, 32, 64, 128]
    if "all" in args.frameworks:
        args.frameworks = ["torch", "hvd"]
    if "all" in args.model_names:
        args.model_names = ["resnet18", "resnet50", "resnet101", "vgg16", "alexnet", "vit_b_16"]
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    with open(args.output_dir / "args.txt", "w") as f:
        f.write(str(args))

    if args.file == "model":
        with open(args.output_dir / "result.csv", "w") as f:
            f.write("rank, framework, model_name, batch_size, nchannels, nthreads, comp_time, comm_time, overlap_time, cpu_time, throughput\n")
    
    args.file += ".py"
    print(args)
    run(**vars(args))
