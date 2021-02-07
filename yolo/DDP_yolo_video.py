import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager
from queue import PriorityQueue

import time
import cv2
import matplotlib.pyplot as plt

from DDP_yolo import colors, plot_one_box  # plotting and name text

# ========== DDP ==========
def init_process(rank, size, data, f_size, output_queue, fps_quque, backend='nccl'):
    '''Initialize the distributed environment.'''
    try:
        print('rank', rank, 'is listening')
        dist.init_process_group(backend, init_method='tcp://127.0.0.1:8901',
                                rank=rank, world_size=size)
        print('rank', rank, 'is starting')

        # RUN the main function
        main(rank, size, data, f_size, output_queue, fps_quque)
    finally:
        cleanup()


def cleanup():
    dist.destroy_process_group()

# ========== main ==========
def set_saved_video(input_video, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    print('fps', fps)
    video = cv2.VideoWriter('out.mp4', fourcc, fps, size)
    return video

def draw(out, frames):
    out_frames = []
    for i, bboxs in enumerate(out.xyxy):
        if isinstance(frames[i], torch.Tensor):
            frame = frames[i].detach().cpu().numpy()
        else:
            frame = np.array(frames[i])

        for bbox in bboxs:
            threshold = 0.3
            if bbox[4] > threshold:
                label = int(bbox[5])
                c = colors[label]
                text = '{} {:.2f}'.format(
                    out.names[label],
                    bbox[4]
                )

                plot_one_box(bbox, frame, label=text,
                             color=c, line_thickness=3)
        out_frames.append(frame)
    return out_frames

def main(rank, size, data, f_size, output_queue, fps_quque):
    # ---------- load model ----------
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

    # ---------- DDP model ----------
    gpu_rank = rank
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(gpu_rank))
        model.to(device)
        ddp_model = DDP(model, device_ids=[gpu_rank])
    else:
        device = torch.device('cpu')
        model.to(device)
        ddp_model = DDP(model)

    # ---------- get data ----------
    width, height = f_size
    print('w', width, 'h', height)

    # ----------loop ----------
    frame_count = 0
    for i, frames in data:
        t = time.time()

        # ---------- inference ----------
        out = model(frames)
        frames = draw(out, frames)

        # ---------- output file ----------
        frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
#         [save_cap.write(f) for f in frames]
        output_queue.put((i, frames))

        fps = len(frames) / (time.time() - t)
        fps_queue.put(fps)

        if frame_count == 600:
            break

    cap.release()
    save_cap.release()

class MyManager(BaseManager):
    pass

MyManager.register('PriorityQueue', PriorityQueue)

if __name__ == "__main__":
    # load data
    input_path = 'office_parkour_720P.mp4'

    all_frames = []
    frame_count = 0
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_cap = set_saved_video(cap, (width, height))
    while cap.isOpened():
        ret, frame_raw = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height),
                           interpolation=cv2.INTER_LINEAR)
        all_frames.append(frame)

        if frame_count == 600:
            break

    # partition data
    size = 3
    cont_frames = 8
    frame_splits = [[] for _ in range(size)]
    for i in range(0, len(all_frames), cont_frames):
        idx = int((i / cont_frames) % size)
#         print('i', i, 'idx', idx)
        frame_splits[idx].append((i, all_frames[i:i + cont_frames]))

    print([len(f) for f in frame_splits])

    m = MyManager()
    m.start()
    output_queue = m.PriorityQueue()
    fps_queue = mp.Queue()

    processes = []

    start_t = time.time()
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, frame_splits[rank], (width, height),
                                                  output_queue, fps_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('size', size, 'frames', cont_frames)

    start_t = time.time() - start_t
    print('cost', start_t, 'secs')
    print('total fps', 600 / start_t)

    print('All Done.')

    # output video
    results = []
    while not output_queue.empty():
        results.append(output_queue.get())

    results.sort(key=lambda x: x[0])

    for i, fs in results:
        for f in fs:
            save_cap.write(f)
    save_cap.release()

    # calculate fps
    fps_list = []
    while not fps_queue.empty():
        fps_list.append(fps_queue.get())
    print('avg fps', sum(fps_list) / len(fps_list))
