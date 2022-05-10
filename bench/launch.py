import argparse
import os
import itertools

from tqdm import tqdm

from common import add_common_args
from op import add_op_args


def system(cmd, log_path=None):
    if log_path is not None:
        cmd = f"{cmd} | tee -a {log_path}"
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


def gen_cmd_args(kwargs):
    return " ".join(map(arg_to_str, kwargs.keys(), kwargs.values()))


def bench_model(configs, model_names, batch_sizes, frameworks, log_path, bench, **kwargs):
    combinations = itertools.product(
        model_names, frameworks, batch_sizes, configs)
    for model_name, framework, batch_size, config in tqdm(list(combinations), colour="green"):
        kwargs["model_name"] = model_name
        kwargs["framework"] = framework
        kwargs["batch_size"] = batch_size
        kwargs["nchannels"], kwargs["nthreads"] = map(int, config.split(","))

        world_size = kwargs["world_size"]
        args = gen_cmd_args(kwargs)
        if framework == "torch":
            os.environ["OMP_NUM_THREADS"] = "1"
            system(
                f"torchrun --nproc_per_node {world_size} model.py --no-spawn {args}",
                log_path
            )
        elif framework == "hvd":
            system(
                f"horovodrun -np {world_size} python model.py --no-spawn {args}",
                log_path
            )


def bench_op(configs, ops, log_path, bench, **kwargs):
    combinations = itertools.product(ops, configs)
    for op, config in tqdm(list(combinations), colour="green"):
        kwargs["op"] = op
        kwargs["nchannels"], kwargs["nthreads"] = map(int, config.split(","))
        args = gen_cmd_args(kwargs)
        system(f"python op.py {args} --spawn", log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="bench", required=True)

    # model launcher
    model_subparser = subparsers.add_parser("model")
    model_subparser.add_argument("-m", "--model_names",
                              nargs="+", default=["resnet50"])
    model_subparser.add_argument("-b", "--batch_sizes",
                              nargs="+", default=[32])
    model_subparser.add_argument("-f", "--frameworks",
                              nargs="+", default=["torch"],
                              choices=["torch", "hvd", "all"])

    # op launcher
    op_subparser = subparsers.add_parser("op")
    add_op_args(op_subparser)

    for subparser in [model_subparser, op_subparser]:
        add_common_args(subparser)
        subparser.add_argument("-c", "--configs", nargs="+", default=["2,256"])

    args = parser.parse_args()
    if hasattr(args, "configs") and "all" in args.configs:
        args.configs = [
            f"{i},{2 ** j}"
            for i in range(1, 5)
            for j in range(6, 10)
        ]
        if args.bench == "op":
            args.configs.insert(0, "0,0")
    if hasattr(args, "batch_sizes") and "all" in args.batch_sizes:
        args.batch_sizes = [16, 32, 64, 128]
    if hasattr(args, "frameworks") and "all" in args.frameworks:
        args.frameworks = ["torch", "hvd"]
    if hasattr(args, "model_names") and "all" in args.model_names:
        args.model_names = ["resnet18", "resnet50", "resnet101",
                            "vgg16", "alexnet", "vit_b_16"]

    print(args)

    args.output_dir.mkdir(exist_ok=True, parents=True)
    args.log_path = args.output_dir / "log.txt"
    with open(args.log_path, "w") as f:
        f.write(str(args) + "\n")

    if args.bench == "model":
        bench_model(**vars(args))
    elif args.bench == "op":
        bench_op(**vars(args))
