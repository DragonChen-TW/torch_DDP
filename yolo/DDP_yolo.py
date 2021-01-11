import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
# 
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# 
import glob
import time
# 
from DDP_yolo_data import get_PIL_data

# =========== Plot ===========
import random
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         t_size = 1
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# =========== Communication ===========
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group('nccl',
        init_method='tcp://127.0.0.1:8901',
        rank=rank, world_size=world_size
    )

# ========== Running ==========
def infrence(rank, size):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    gpu_rank = rank
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_rank))
        model.to(device)
        ddp_model = DDP(model, device_ids=[gpu_rank])
    else:
        device = torch.device('cpu')
        model.to(device)
        ddp_model = DDP(model)

    batch_size = 8
    # data = get_tensor_data()
    # imgs = next(iter(data))
    data = get_PIL_data(batch_size, rank, size)
    # print('total data', data)

    imgs = [d[0] for d in data]

    t = time.time()
    out = model(imgs)
    print('Cost', time.time() - t, 'secs')

    # =========== Draw Plot ===========
    plot_file = True
    if plot_file:
        print('Plotting into file')
        out_imgs = []

    for i, bboxs in enumerate(out.xyxy):
        if isinstance(imgs[i], torch.Tensor):
            out_img = imgs[i].detach().cpu().numpy()
        else:
            out_img = np.array(imgs[i])

        for pred in bboxs:
            threshold = 0
            if pred[4] > threshold:
                label = int(pred[5])
                c = colors[label]
                text = '{} {:.2f}'.format(
                    out.names[label],
                    pred[4]
                )
                # print(text)

            plot_one_box(pred, out_img, label=text, color=c, line_thickness=3)
        
        if plot_file:
            # plt.figure(figsize=(12, 12))
            plt.imshow(out_img)
            plt.axis('off')
            plt.savefig('out_frames/{:03}.png'.format(data[i][1]))
            plt.clf()
            out_imgs.append(Image.fromarray(out_img))
            # plt.show()
    
    if rank == 0:
        # output gif
        fp_out = 'out.gif'
        out_imgs[0].save(fp=fp_out, format='GIF', append_images=out_imgs,
         save_all=True, duration=200, loop=0)

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) > 2:
        rank = int(argv[1])
        size = int(argv[2])
    else:
        rank = 0
        size = 1

    print('Running DDP yolo on rank {} world {}'.format(rank, size))

    setup(rank, size)

    infrence(rank, size)