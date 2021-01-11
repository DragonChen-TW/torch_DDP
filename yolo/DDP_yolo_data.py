from random import Random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# 
import glob
from PIL import Image

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
        # rng.shuffle(indexes)

        for part in sizes:
            part_len = round(part * data_len)
            self.partitions.append(indexes[0: part_len])
            indexes = indexes[part_len:]

    def use(self, rank):
        return Partition(self.data, self.partitions[rank])

# =========== Data ===========
class ImageFolder(Dataset):
    def __init__(self, transform=None):
        self.img_list = sorted(glob.glob('./cv2test/*.jpg'))
        self.transform = transform
    
    def __getitem__(self, idx):
        file_name = self.img_list[idx]
        img = Image.open(file_name)
        if self.transform:
            img = self.transform(img)
        return img, idx

    def __len__(self):
        return len(self.img_list)

def get_tensor_data():
    '''Tensor 4-dim (batch, channel, height, width)'''
    transform = transforms.Compose([
        transforms.Resize(720),
        transforms.CenterCrop((720, 720)),
        transforms.ToTensor(),
    ])
    img_folder = ImageFolder(transform)

    def collate_list(batch):
        return batch

    # Split data with dataloader
    batch_size = 16
    data_loader = DataLoader(dataset=img_folder, # input Dataset
                            #  collate_fn=collate_list,
                            batch_size=batch_size,
                            shuffle=False)
    return data_loader

def get_PIL_data(batch_size, rank, size):
    img_folder = ImageFolder()
    # img_folder.img_list = img_folder.img_list[:30]

    batch_size_part = int(batch_size / size)
    partition_sizes = [1.0 / size for _ in range(size)]
    paritition = DataPartitioner(img_folder, partition_sizes)
    paritition = paritition.use(rank)

    print('len of rank', rank, 'is', len(paritition))

    imgs = [p for p in paritition]
    # print('imgs', imgs)
    return imgs

if __name__ == '__main__':
    data = get_PIL_data(8, 0, 1)