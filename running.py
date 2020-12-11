import time
import torch
from torch import nn

from models import (
    ResNetOriginal, BottleNeck,
    cifar10_origin_pre,
)
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 10

class ModelParallelResNet50(ResNetOriginal):
# class ModelParallelResNet50(ResNet):
    def __init__(self):
        super().__init__(
#             Bottleneck, [3, 4, 6, 3],
            BottleNeck, [3, 4, 6, 3],
            pre=cifar10_origin_pre
        )

        self.shard1 = nn.Sequential(
#             self.conv1,
            self.pre,
            self.layer1,
            self.layer2
        ).to('cuda:1')

        self.shard2 = nn.Sequential(
            self.layer3,
            self.layer4,
#             self.avgpool,
            self.avg_pool,
        ).to('cuda:2')

        self.fc.to('cuda:2')

    def forward(self, x):
        out = self.shard1(x).to('cuda:2') # from cuda:0 to cuda:1
        out = self.shard2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc(out)
        return out

def basic_shard():
    print('Basic Shard')
    runs = [1, 10, 100, 500]
    
    for run in runs:
        print(run, 'runs')
        
        device = torch.device('cuda:1')
        x = torch.rand(256, 1, 28, 28).to(device)
        model = ModelParallelResNet50()

        t = time.time()
        for _ in range(run):
             model(x)
        print(time.time() - t)


        print('single GPU')
        device = torch.device('cuda:2')
        x = x.to(device)
        model = ModelParallelResNet50().to(device)

        t = time.time()
        for _ in range(run):
            model(x)
        print(time.time() - t)

class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=4, *args, **kwargs):
        super().__init__()
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.shard1(s_next).to('cuda:2')
        returns = []

        for s_next in splits:
            # A. s_prev runs on cuda:2
            s_prev = self.shard2(s_prev)
            returns.append(self.fc(s_prev.view(s_prev.shape[0], -1)))

            # B. s_next runs on cuda:1, which can run concurrently with A
            s_prev = self.shard1(s_next).to('cuda:2')

        s_prev = self.shard2(s_prev)
        returns.append(self.fc(s_prev.view(s_prev.shape[0], -1)))

        return torch.cat(returns)

def pipeline_model_parallel():
    print('Pipeline Model Parallel')
    runs = [1, 10, 100, 500]
    
    for run in runs:
        print(run, 'runs')
        
        device = torch.device('cuda:1')
        x = torch.rand(128, 3, 224, 224).to(device)
        model = PipelineParallelResNet50()

        t = time.time()
        for _ in range(run):
             model(x)
        print(time.time() - t)

if __name__ == '__main__':
    basic_shard()

#     pipeline_model_parallel()
