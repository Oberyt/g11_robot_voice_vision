#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import config
import time
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import requests
import urllib3
import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

_CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)

def get_cascade_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_xml = Path(script_dir) / "haarcascade_frontalface_default.xml"
    if local_xml.exists() and local_xml.stat().st_size > 10000:
        return str(local_xml)
    sys_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if os.path.exists(sys_xml):
        return sys_xml
    try:
        urllib.request.urlretrieve(_CASCADE_URL, local_xml)
        return str(local_xml)
    except Exception:
        return None

class AssistantNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.unlocked = False

        # ---------- 加载人脸识别模型 ----------
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "../data/trainer.yml")
        
        cascade_path = get_cascade_path()
        if cascade_path is None:
            raise RuntimeError("找不到人脸分类器文件！")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(model_path)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.names = {1: "林倍铭", 2: "Dante", 3: "Vergil"}
        self.confidence_threshold = 80

        # ---------- 颜色 HSV 范围 ----------
        self.color_ranges = {
            "红色": [(np.array([0, 100, 100]), np.array([10, 255, 255])), (np.array([160, 100, 100]), np.array([180, 255, 255]))],
            "绿色": [(np.array([36, 25, 25]), np.array([70, 255, 255]))],
            "蓝色": [(np.array([110, 50, 50]), np.array([130, 255, 255]))],
            "黄色": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
            "橙色": [(np.array([11, 100, 100]), np.array([25, 255, 255]))],
        }
        self.min_color_area_ratio = 0.02

        # ---------- 和风天气 ----------
        self.qweather_key = config.QWEATHER_KEY
        self.qweather_city_id = "101280601"
        self.qweather_url = config.QWEATHER_HOST
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # ---------- ROS 通信 ----------
        self.tts_pub = rospy.Publisher("/voice/tts", String, queue_size=10)
        self.unlock_pub = rospy.Publisher("/voice/unlock_status", Bool, queue_size=10, latch=True)
        # 启动时默认未解锁，通知 ASR 开启拦截
        self.unlock_pub.publish(False)

        self.stt_sub = rospy.Subscriber("/voice/stt", String, self.stt_callback, queue_size=1)
        
        # 订阅彩色图
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1, buff_size=2**24)

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 全局发声锁，防抖
        self._last_speak_time = 0.0

        rospy.loginfo("🤖 AssistantNode 就绪，等待人脸解锁...")

    def image_callback(self, msg):
        # 【终极拦截】如果已经解锁过了，直接扔掉后续所有的图像，不再进行人脸检测
        if self.unlocked:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        with self.frame_lock:
            self.latest_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            try:
                label, confidence = self.recognizer.predict(roi)
            except Exception:
                continue

            # 如果置信度达标且 ID 匹配
            if confidence <= self.confidence_threshold and label in self.names:
                name = self.names[label]
                
                # 立即锁死状态
                self.unlocked = True
                self.unlock_pub.publish(True)
                rospy.loginfo("🔓 人脸解锁成功，语音已激活！")
                
                self.speak(f"你好，{name}先生。")
                return  # 直接 return，不再处理画面里的其他脸

    def stt_callback(self, msg):
        text = str(msg.data).strip()
        if not text or len(text) <= 1:
            return

        if "颜色" in text:
            self.handle_color()
        elif "天气" in text:
            self.handle_weather()

    def handle_color(self):
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()

        if frame is None:
            self.speak("还没有拿到摄像头画面。")
            return

        color_name = self.detect_dominant_color(frame)
        if color_name:
            self.speak(f"这是{color_name}。")
        else:
            self.speak("没有识别到明显的颜色。")

    def detect_dominant_color(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        total_area = h * w
        best_color = None
        best_area = 0

        for color_name, ranges in self.color_ranges.items():
            mask = None
            for lower, upper in ranges:
                m = cv2.inRange(hsv, lower, upper)
                mask = m if mask is None else cv2.bitwise_or(mask, m)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            area = cv2.countNonZero(mask)
            if area > best_area and area / total_area >= self.min_color_area_ratio:
                best_area = area
                best_color = color_name
        return best_color

    def handle_weather(self):
        try:
            params = {"location": self.qweather_city_id, "key": self.qweather_key}
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            
            resp = requests.get(self.qweather_url, params=params, headers=headers, timeout=10, verify=False)
            if resp.status_code != 200:
                self.speak("天气查询失败了。")
                return
                
            data = resp.json()
            if data.get("code") != "200":
                self.speak("天气查询失败了。")
                return

            now = data["now"]
            msg = f"当前{now.get('text', '未知')}，温度{now.get('temp')}摄氏度。"
            self.speak(msg)
        except Exception:
            self.speak("天气查询出现问题。")

    def speak(self, text):
        now = time.time()
        if now - self._last_speak_time < 2.0:
            return
        self._last_speak_time = now
        
        rospy.loginfo(f"🗣️ TTS: {text}")
        self.tts_pub.publish(String(data=text))

if __name__ == "__main__":
    rospy.init_node("assistant_node", anonymous=False)
    node = AssistantNode()
    rospy.spin()
