import os, sys
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

# from train import demo_basic
from mnist_train import demo_basic

def run(rank, size):
    '''Distributed function to be implemented later.'''
    pass

def init_process(rank, size, func, backend='nccl'):
    '''Initialize the distributed environment.'''
    # os.environ['MASTER_ADDR'] = '10.0.0.101'
    # os.environ['MASTER_PORT'] = '8900'

    try:
        print('rank', rank, 'is listening ')
        dist.init_process_group(backend, init_method='tcp://127.0.0.1:8900',
            rank=rank, world_size=size)
        print('rank', rank, 'is starting ')

        func(rank, size)
    finally:
        cleanup()

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    size = 3
    func = demo_basic
    processes = []

    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
