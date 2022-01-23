import time

import subprocess
import os
from pathlib import Path
import cv2
import torch
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords)
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import time_sync


@torch.no_grad()
def detect(source, device, model):
    # 图片大小
    imgsz = [640, 640]
    # 是否使用半精度加速
    half = False
    # 保存地址
    save_dir = Path('web/static')
    # 加载一下我们的模型
    stride, names = model.stride, model.names
    # 检查图片大小
    imgsz = check_img_size(imgsz, s=stride)
    # 使用半精度，可以给我们的模型加速 only supported by PyTorch on CUDA
    half &= device.type != 'cpu'
    # 判断是否需要使用半精度
    model.model.half() if half else model.model.float()
    # 加载我们的视频
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)
    # 一次取多少张图片
    bs = 1
    # 保存的路径和write对象
    vid_path, vid_writer = [None] * bs, [None] * bs
    # 减少过拟合现象
    model.warmup(imgsz=(1, 3, *imgsz), half=half)
    # 加载视频数据
    for path, im, im0s, vid_cap, s in dataset:
        # 将数组array转换为张量Tensor
        im = torch.from_numpy(im).to(device)
        # 判断是否需要半精度
        im = im.half() if half else im.float()
        # 除于255，把图片变成0-1的范围内
        im /= 255
        if len(im.shape) == 3:
            # 展开图片
            im = im[None]
        # 使用模型来进行预测
        pred = model(im, augment=False, visualize=False)
        # 模型检测出了很多框，我应该留哪些
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        # 调用算法进行处理
        for i, det in enumerate(pred):
            im0 = im0s.copy()
            # 视频保存的路径
            save_path = str(save_dir / 'detect_row.mp4')
            # 获取一个绘制的对象
            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # 缩放图片
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # 获取当前绘制的信息
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # 绘制结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
            # 打印一下转换的进度
            LOGGER.info(f'{s}Done.)')
            im0 = annotator.result()
            # 保存视频
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                # 获取视频的宽
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                # 写入视频信息，这里使用mp4v编码
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)
