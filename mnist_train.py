import sys, os
import time
#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
#
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#
from data_partition import get_mnist
from models import (
    LeNet,
    resnet18, resnet34, resnet50
)

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group('nccl',
        init_method='tcp://127.0.0.1:8900',
        rank=rank, world_size=world_size
    )

def demo_basic(rank, world_size, max_epochs=5, verbose=False):
    # map rank [0, 1, 2] => ['cuda:1', 'cuda:2', 'cuda:3']
    gpu_rank = rank + 1
    # create model and move it to GPU with id rank
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_rank))
    else:
        device = torch.device('cpu')

    model = resnet34().to(device)

    if torch.cuda.is_available():
        ddp_model = DDP(model, device_ids=[gpu_rank])
    else:
        ddp_model = DDP(model)

    data = get_mnist('~/data', rank, world_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.005)

    t = time.time()

    for epoch in range(1, max_epochs + 1):
        loss_list = []
        total_count = 0
        acc_count = 0

        loader = data
        if rank == 0 or verbose:
            loader = tqdm(data, total=len(data))
        for image, label in loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_count += outputs.shape[0]
            correct = torch.argmax(outputs, dim=1).eq(label)
            acc_count += correct.sum().item()
            loss_list.append(loss.item())

        if rank == 0 or verbose:
            print('epoch', epoch, 'acc', acc_count / total_count,\
                'loss', '{:.03}'.format(sum(loss_list) / len(loss_list)))

    # output
    if rank == 0 or verbose:
        t = time.time() - t
        print('Cost Time:', t, 'avg time', t / max_epochs)

if __name__ == '__main__':
    argv = sys.argv
    rank = int(sys.argv[1])
    size = int(sys.argv[2])

    print(f'Running basic DDP example on rank {rank}.')
    setup(rank, size)

    demo_basic(rank, size, verbose=True)
