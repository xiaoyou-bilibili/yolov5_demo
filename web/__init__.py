import cv2
from flask import Flask, request, Response, render_template
from flask_cors import CORS
import json
import numpy as np
from yolov5 import detect_img,detect_video

# 初始化flaskAPP
app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
# 允许跨域请求
CORS(app, resources=r'/*')


# 返回JSON字符串
def return_json(data):
    return Response(json.dumps(data, ensure_ascii=False), mimetype='application/json')


# 基于图片的目标检测
@app.route('/yolov5/detect_img', methods=['POST'])
def detect_image():
    # 获取请求的图片
    file = request.files['file']
    # 使用opencv读取图像
    data = cv2.imdecode(np.frombuffer(file.stream.read(), np.uint8), cv2.IMREAD_COLOR)
    # 这里顺便保存一下图片用于前端展示
    cv2.imwrite("./web/static/detect.jpg", data)
    # 运行检测模型
    res = detect_img(data)
    # 返回json类型字符串
    return return_json(res)


# 基于图片的目标检测
@app.route('/yolov5/detect_video', methods=['POST'])
def detect_vide():
    # 获取请求的视频
    file = request.files['file']
    # 保存一下这个视频
    file.save("./web/static/detect.mp4")
    res = detect_video()
    # 返回json类型字符串
    return return_json(res)


# 主页显示HTML
@app.route('/', methods=['GET'])
def index():
    return render_template('content.html')
