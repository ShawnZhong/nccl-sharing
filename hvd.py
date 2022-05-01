import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import horovod.torch as hvd
import timeit
import numpy as np
import os
from torch.profiler import profile, record_function, ProfilerActivity
from main import print_profiling_info

def main(model_name, batch_size, num_iters, num_warmup_iters):
    hvd.init()
    rank = hvd.local_rank()
    torch.cuda.set_device(rank)

    # Set up standard model.
    model = getattr(models, model_name)()
    model.cuda()

    # By default, Adasum doesn't need scaling up learning rate.
    lr_scaler = hvd.size()
    optimizer = optim.SGD(model.parameters(), lr=0.01 * lr_scaler)

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                        named_parameters=model.named_parameters(),
                                        op=hvd.Average)

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # Set up fixed fake data
    data = torch.randn(batch_size, 3, 224, 224, device=rank)
    target = torch.randint(0, 1000, (batch_size,), device=rank)


    def benchmark_step():
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            loss.backward()
        print_profiling_info(prof)
        optimizer.step()


    def log(s, nl=True):
        if hvd.rank() != 0:
            return
        print(s, end='\n' if nl else '')


    log('Model: %s' % model_name)
    log('Batch size: %d' % batch_size)
    log('Number of GPUs: %d' % (hvd.size()))

    # Warm-up
    log('Running warmup...')
    timeit.timeit(benchmark_step, number=num_warmup_iters)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(num_iters):
        time = timeit.timeit(benchmark_step, number=1)
        img_sec = batch_size / time
        log('Iter #%d: %.1f img/sec per GPU' % (x, img_sec))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per GPU: %.1f +-%.1f' % (img_sec_mean, img_sec_conf))
    log('Total img/sec on %d GPU(s): %.1f +-%.1f' %
        (hvd.size(), hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

if __name__ == "__main__":
    # Benchmark settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet50',
                        help='model to benchmark')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size')

    parser.add_argument('--num-warmup-iters', type=int, default=0,
                        help='number of warm-up batches that don\'t count towards benchmark')
    parser.add_argument('--num-iters', type=int, default=10,
                        help='number of benchmark iterations')

    # parser.add_argument("--world_size", default=2)

    args = parser.parse_args()
    os.environ["NCCL_P2P_DISABLE"] = "1"
    main(**vars(args))