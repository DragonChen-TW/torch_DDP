import os, sys
import time
#
import torch
import torch.distributed as dist
import torch.nn as nn
#
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("gloo",
        init_method='tcp://127.0.0.1:8900',
        rank=rank, world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def demo_basic(rank, world_size, d_size, n_run=20):
    # create model and move it to GPU with id rank
    device = torch.device('cpu')
    model = ToyModel().to(device)
    ddp_model = DDP(model)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    t = time.time()

    for i in range(n_run):
        if i % 5 == 0:
            print('Round', i)
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(d_size, 10))
        labels = torch.randn(d_size, 5).to(device)
        loss_fn(outputs, labels).backward()
        optimizer.step()

    # output
    print('Cost Time:', time.time() - t)

if __name__ == '__main__':
    # try:
    argv = sys.argv
    rank = int(argv[1])
    d_size = int(argv[2])
    n_run = int(argv[3])
    size = 3


    print(f"Running basic DDP example on rank {rank}.")
    print('n_run =', n_run)
    setup(rank, size)

    demo_basic(rank, size, d_size, n_run)

    cleanup()
