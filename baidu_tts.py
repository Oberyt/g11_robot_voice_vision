#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import config
import time
import queue
import threading
import requests
import warnings
warnings.filterwarnings("ignore")
import rospy
from std_msgs.msg import String
import numpy as np
import sounddevice as sd

APPID = config.BAIDU_TTS_APPID
APIKEY = config.BAIDU_TTS_APIKEY
SECRETSkEY = config.BAIDU_TTS_SECRETKEY

class Baidu_tts():
    def __init__(self):
        self.url = 'http://tsn.baidu.com/text2audio'
        self.__accessTokenUrl = 'https://aip.baidubce.com/oauth/2.0/token'
        self._apiKey = APIKEY.strip()
        self._secretKey = SECRETSkEY.strip()
        self._authObj = {}
        self.__connectTimeout = 60.0
        self.__socketTimeout = 60.0
        self._proxies = {}

        self.text_queue = queue.Queue()
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()

        # 文本防抖锁
        self.last_text = ""
        self.last_play_time = 0.0

        self.sub = rospy.Subscriber('/voice/tts', String, self.callback)

    def _auth(self, refresh=False):
        if not refresh:
            tm = self._authObj.get('time', 0) + int(self._authObj.get('expires_in', 0)) - 30
            if tm > int(time.time()):
                return self._authObj

        obj = requests.get(
            self.__accessTokenUrl,
            verify=False,
            params={'grant_type': 'client_credentials', 'client_id': self._apiKey, 'client_secret': self._secretKey},
            timeout=self.__connectTimeout,
            proxies=self._proxies,
        ).json()

        obj['time'] = int(time.time())
        self._authObj = obj
        return obj

    def callback(self, msg):
        text = str(msg.data)
        now = time.time()
        
        # 【核心防抖】如果文本和上一次一样，且间隔小于 3 秒，直接丢弃
        if text == self.last_text and (now - self.last_play_time < 3.0):
            rospy.logwarn(f"拦截重复文本: {text}")
            return
            
        self.last_text = text
        self.last_play_time = now
        rospy.loginfo("TTS: " + text)
        self.text_queue.put(text)

    def _play_loop(self):
        while not rospy.is_shutdown():
            try:
                text = self.text_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self.tts(text)
            except Exception as e:
                rospy.logerr(f"播放异常: {e}")

    def tts(self, text=None):
        if not text:
            return
        try:
            data = {
                'tex': text, 'lan': 'zh', 'ctp': 1, 'per': 5118, 'cuid': 'test', 'aue': 4,
                'tok': self._auth()['access_token']
            }
            headers = {'content-type': 'application/x-www-form-urlencoded'}
            response = requests.post(self.url, data=data, headers=headers, timeout=self.__socketTimeout)

            if 'application/json' in response.headers.get('Content-Type', ''):
                rospy.logerr("百度 API 报错: " + response.text)
                return

            if response:
                play_data = np.frombuffer(response.content, dtype=np.short)
                sd.play(play_data, 16000)
                sd.wait()
        except Exception as e:
            rospy.logerr(f"TTS 异常: {e}")

if __name__ == '__main__':
    try:
        rospy.init_node('baidu_tts', anonymous=False)
        rec = Baidu_tts()
        rospy.spin()
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
