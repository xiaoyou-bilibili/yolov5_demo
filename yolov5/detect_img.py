import cv2
import torch
import numpy as np
from pathlib import Path
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import (LOGGER, check_img_size, colorstr, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box


@torch.no_grad()
def detect(img, device, model):
    # 设置必要的参数
    project = 'web/static'
    imgsz = [640, 640]
    half = False
    # 设置保存路径
    save_dir = Path(project)
    # 加载一下我们的模型
    stride, names = model.stride, model.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # 使用半精度，可以给我们的模型加速
    half &= device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    model.model.half() if half else model.model.float()
    # 减少过拟合现象
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    # 创建边框
    im = letterbox(img, imgsz, stride=stride, auto=True)[0]
    # 转换色彩 HWC to CHW, BGR to RGB
    im = im.transpose((2, 0, 1))[::-1]
    # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    im = np.ascontiguousarray(im)
    # 将数组array转换为张量Tensor
    im = torch.from_numpy(im).to(device)
    # 判断是否需要半精度
    im = im.half() if half else im.float()  # uint8 to fp16/32
    # 除于255，把图片变成0-1的范围内
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        # 展开图片
        im = im[None]  # expand for batch dim
    # 使用我们的模型去检测结果
    pred = model(im, augment=False, visualize=False)
    # 模型检测出了很多框，我应该留哪些
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    # 使用我们预测的结果
    det = pred[0]
    im0 = img.copy()
    # 保存地址
    save_path = str(save_dir / "detect_res.jpg")  # im.jpg
    # 获取一个绘制的对象
    annotator = Annotator(im0, line_width=3, example=str(names))
    # 这里包装一下最后的结果
    detect = []
    # 如果检测到有内容我们就可以准备绘制了
    if len(det):
        # 缩放图片
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
        # 这里就是绘制我们的最终结果了
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            detect.append({
                "point": "%.2f,%.2f,%.2f,%.2f" % (xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()),
                "label": names[c],
                "conf": "%.2f" % conf.item()
            })
            # 这里设置我们的标签
            label = f'{names[c]} {conf:.2f}'
            annotator.box_label(xyxy, label, color=colors(c, True))
    # 绘制结果
    im0 = annotator.result()
    # 保存图片
    cv2.imwrite(save_path, im0)
    return detect
