import torch.nn as nn
import torch
from yolov5.detect_img import detect
from yolov5.detect_video import detect as video_detect
import os
from yolov5.models.experimental import attempt_load

# 全局算法模型
model = None
device = None

# 加载模型
def load_model():
    global model, device
    # 选择设备与加载模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(device=device, dnn=False)


# 检测图片
def detect_img(img):
    res = detect(img, device, model)
    return {
        "detect": "/static/detect_res.jpg",
        "row": "/static/detect.jpg",
        "box": res
    }


# 检测视频
def detect_video():
    video_detect('web/static/detect.mp4', device, model)
    # 这里我们调用一下ffmpeg来转换一下视频格式
    path = os.getcwd()
    # 删除旧的转换好的视频
    os.remove("%s/web/static/detect_res.mp4" % path)
    cmd = "ffmpeg -nostdin -i %s/web/static/detect_row.mp4 -vcodec h264 %s/web/static/detect_res.mp4" % (path, path)
    print(cmd)
    os.system(cmd)
    print("检测完成")
    return {
        "detect": "/static/detect_res.mp4",
        "row": "/static/detect.mp4"
    }


# 加载模型
class DetectMultiBackend(nn.Module):
    def __init__(self, device=None, dnn=True):
        super().__init__()
        # 设置我们的模型
        w = './yolov5/weight/best.pt'
        # 设置步幅和分类的名字
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        # 加载我们的模型
        model = attempt_load(w, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        # 返回全部的局部变量
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        y = self.model(im, augment=augment, visualize=visualize)
        return y if val else y[0]

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
            im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
            self.forward(im)  # warmup
