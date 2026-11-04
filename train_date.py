#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
from PIL import Image
import numpy as np
import urllib.request
from pathlib import Path

# 官方 GitHub 模型下载地址
_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

def get_cascade_path():
    """获取级联分类器路径：优先本地，不存在则自动下载"""
    # 放在脚本同目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_path = Path(script_dir) / "haarcascade_frontalface_default.xml"
    
    if cache_path.exists() and cache_path.stat().st_size > 10000:
        return str(cache_path)
    
    print(f"本地未找到分类器文件，正在从网络下载到: {cache_path}")
    try:
        urllib.request.urlretrieve(_CASCADE_URL, cache_path)
        print("下载成功！")
        return str(cache_path)
    except Exception as e:
        print(f"下载失败: {e}")
        print("请检查网络，或手动下载该文件放到 scripts 目录下。")
        return None

def getImageAndLables(path):
    faceSamples = []
    ids = []
    names = []
    
    if not os.path.exists(path):
        print(f"错误: 找不到文件夹 {path}")
        return faceSamples, ids, names
        
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f'共发现 {len(imagePaths)} 张图片。')

    # 加载分类器
    cascade_path = get_cascade_path()
    if cascade_path is None:
        raise RuntimeError("无法获取人脸分类器文件！")

    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise RuntimeError("人脸分类器加载失败！请检查XML文件是否完整。")

    for imagePath in imagePaths:
        try:
            # 按照 1.姓名.jpg 的格式提取
            filename = os.path.basename(imagePath)
            id = int(filename.split('.')[0])
            name = str(filename.split('.')[1])
        except Exception as e:
            print(f"文件名格式错误，跳过: {filename}")
            continue

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        
        # 优化：添加了常用的检测参数，提高识别率
        faces = face_detector.detectMultiScale(
            img_numpy,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            print(f"[警告] 未在 {filename} 中检测到人脸，已跳过。")
            continue

        for x, y, w, h in faces:
            ids.append(id)
            names.append(name)
            faceSamples.append(img_numpy[y:y+h, x:x+w])
        
        print(f'成功提取: {filename} (姓名: {name})')

    return faceSamples, ids, names

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "../faces")
    
    print(f"正在从 {path} 读取训练数据...")
    faces, ids, names = getImageAndLables(path)

    if len(faces) == 0:
        print("没有提取到任何人脸数据，训练终止。请检查图片和文件名格式。")
        exit()

    print(f"开始训练 LBPH 模型，共 {len(set(ids))} 个人，{len(faces)} 张有效人脸...")
    
    # 注意：如果这里报错，说明你安装的不是 opencv-contrib-python
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))

    # 确保 data 文件夹存在
    data_dir = os.path.join(script_dir, "../data")
    os.makedirs(data_dir, exist_ok=True)
    
    out_file = os.path.join(data_dir, "trainer.yml")
    recognizer.write(out_file)
    
    print(f"\n✅ 训练完成！模型已保存至: {out_file}")