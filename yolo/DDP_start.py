import torch
import torch.distributed as dist
from torch.multiprocessing import Process

from DDP_yolo import infrence

def init_process(rank, size, backend='nccl'):
    '''Initialize the distributed environment.'''
    # os.environ['MASTER_ADDR'] = '10.0.0.101'
    # os.environ['MASTER_PORT'] = '8900'

    try:
        print('rank', rank, 'is listening')
        dist.init_process_group(backend, init_method='tcp://127.0.0.1:8901',
            rank=rank, world_size=size)
        print('rank', rank, 'is starting')

        # RUN the main function
        infrence(rank, size)
    finally:
        cleanup()

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    size = 3
    processes = []

    for rank in range(size):
        p = Process(target=init_process, args=(rank, size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()