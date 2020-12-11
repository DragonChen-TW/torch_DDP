from random import Random

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class Partition:
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        data_idx = self.index[idx]
        return self.data[data_idx]

class DataPartitioner:
    def __init__(self, data, sizes=[1], seed=1340):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)

        data_len = len(data)
        indexes = list(range(data_len))
        rng.shuffle(indexes)

        for part in sizes:
            part_len = int(part * data_len)
            self.partitions.append(indexes[0: part_len])
            indexes = indexes[part_len:]

    def use(self, rank):
        return Partition(self.data, self.partitions[rank])

def get_mnist(data_dir, rank, size):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.1307, 0.3081),
    ])

    batch_size = 128
    num_workers = 8
    download = False

    dataset_train = datasets.MNIST(root=data_dir, train=False,
                            transform=trans,
                            download=download)

    batch_size_part = int(batch_size / size)
    partition_sizes = [1.0 / size for _ in range(size)]
    paritition = DataPartitioner(dataset_train, partition_sizes)
    paritition = paritition.use(rank)

    train_data = DataLoader(dataset=paritition,
                            batch_size=batch_size_part,
                            num_workers=num_workers,
                            shuffle=True)

    print('data shape', next(iter(train_data))[0].shape)

    return train_data

if __name__ == '__main__':
    data = get_mnist('~/data/', 0, 3)
