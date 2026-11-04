#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import config
import sys
import json
import base64
import hmac
import hashlib
from urllib.parse import urlencode
import sounddevice as sd
import websocket
import rospy
from std_msgs.msg import String, Bool
from datetime import datetime

APPID = config.XFYUN_ASR_APPID
API_KEY = config.XFYUN_ASR_APIKEY
API_SECRET = config.XFYUN_ASR_APISECRET

class Xfyun_asr():
    def __init__(self):
        self.ws_url = self._build_auth_url()
        self.ws_xf = websocket.create_connection(self.ws_url)

        self.pub = rospy.Publisher('voice/stt', String, queue_size=10)

        # 恢复解锁状态订阅
        self.is_unlocked = False
        self.unlock_sub = rospy.Subscriber('/voice/unlock_status', Bool, self.unlock_callback, queue_size=1)

        self.input_dev_num = sd.query_hostapis()[0]['default_input_device']
        if self.input_dev_num == -1:
            raise ValueError('No input device found')

        self.samplerate = 16000
        self.asr()

    def unlock_callback(self, msg):
        self.is_unlocked = msg.data
        if self.is_unlocked:
            rospy.loginfo("🔓 收到解锁信号，开启语音识别发布")
        else:
            rospy.loginfo("🔒 收到锁定信号，语音功能已静音屏蔽")

    def _build_auth_url(self):
        host = "iat-api.xfyun.cn"
        uri = "/v2/iat"
        date = datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT")
        sign_origin = f"host: {host}\ndate: {date}\nGET {uri} HTTP/1.1"
        hmac_obj = hmac.new(API_SECRET.encode(), sign_origin.encode(), hashlib.sha256)
        sign = base64.b64encode(hmac_obj.digest()).decode()
        auth_raw = f'api_key="{API_KEY}",algorithm="hmac-sha256",headers="host date request-line",signature="{sign}"'
        auth = base64.b64encode(auth_raw.encode()).decode()
        params = {"authorization": auth, "date": date, "host": host}
        return f"wss://{host}{uri}?{urlencode(params)}"

    def send_start_params(self):
        req = {
            "common": {"app_id": APPID},
            "business": {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vad_eos": 2000},
            "data": {"status": 0, "format": "audio/L16;rate=16000", "audio": ""}
        }
        try:
            self.ws_xf.send(json.dumps(req), websocket.ABNF.OPCODE_TEXT)
        except Exception:
            pass

    def send_audio_frame(self, pcm_data):
        req = {
            "data": {"status": 1, "format": "audio/L16;rate=16000", "audio": base64.b64encode(pcm_data).decode()}
        }
        try:
            self.ws_xf.send(json.dumps(req), websocket.ABNF.OPCODE_TEXT)
        except Exception:
            pass

    def callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        try:
            # 未解锁时发送全零静音数据
            if not self.is_unlocked:
                silence = bytes(len(indata)) 
                self.send_audio_frame(silence)
            else:
                self.send_audio_frame(bytes(indata))
        except Exception:
            pass

    def asr(self):
        try:
            with sd.RawInputStream(samplerate=self.samplerate, blocksize=640,
                                   device=self.input_dev_num, dtype='int16',
                                   channels=1, callback=self.callback):
                self.send_start_params()
                rospy.loginfo("讯飞在线语音识别开始监听!")
                full_text = ""
                while not rospy.is_shutdown():
                    result = self.ws_xf.recv()
                    if not result:
                        continue
                    result_data = json.loads(result)
                    if result_data["code"] != 0:
                        continue
                    data = result_data["data"]
                    res = data["result"]
                    seg_text = ""
                    for ws_item in res["ws"]:
                        for cw in ws_item["cw"]:
                            seg_text += cw["w"]
                    
                    full_text += seg_text
                    
                    if res["ls"]:
                        if self.is_unlocked:
                            rospy.loginfo(f"【完整STT结果】{full_text}")
                            self.pub.publish(full_text)
                        else:
                            rospy.loginfo(f"【未解锁-已拦截】{full_text}")
                        full_text = ""
                        
                    if data["status"] == 2:
                        self.ws_xf.close()
                        self.ws_url = self._build_auth_url()
                        self.ws_xf = websocket.create_connection(self.ws_url)
                        self.send_start_params()
        except Exception as e:
            rospy.logerr(f"ASR异常: {type(e).__name__}: {str(e)}")
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    try:
        rospy.init_node('xfyun_asr', anonymous=False)
        rec = Xfyun_asr()
    except (KeyboardInterrupt, rospy.ROSInterruptException) as e:
        pass
