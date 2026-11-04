#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import config
import rospy
from std_msgs.msg import String
import websocket
import time
import json
import base64
import hmac
import hashlib
from urllib.parse import urlencode
from datetime import datetime
import threading

APPID = config.XFYUN_ASR_APPID
APIKEY = config.XFYUN_ASR_APIKEY
APISecret = XFYUN_ASR_APISECRET
SPARK_WSS_URL = "wss://spark-api.xf-yun.com/v4.0/chat"
DOMAIN = "4.0Ultra"
ENABLE_HISTORY = True
MAX_HISTORY_LEN = 6

LOCAL_KEYWORDS = ["天气", "跟我走", "跟我来", "跟着我", "跟着", "停下", "停止", "别跟了", "不用跟了"]

class SparkClient:
    def __init__(self):
        self.appid = APPID.strip()
        self.apikey = APIKEY.strip()
        self.apisecret = APISecret.strip()
        self._lock = threading.Lock()
        self.system_prompt = {
            "role": "system",
            "content": "你是一个家用智能机器人助手。你的名字叫小G。回答要简洁明了、口语化，适合语音播报。遇到不知道的事情不要瞎编。"
        }
        self.chat_history = []

    def _build_auth_url(self):
        gmt_format = '%a, %d %b %Y %H:%M:%S GMT'
        date = datetime.utcnow().strftime(gmt_format)
        host = "spark-api.xf-yun.com"
        path = "/v4.0/chat"
        signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        secret = self.apisecret.encode("utf-8")
        msg = signature_origin.encode("utf-8")
        hmac_obj = hmac.new(secret, msg, digestmod=hashlib.sha256)
        sign = base64.b64encode(hmac_obj.digest()).decode()
        auth_raw = f'api_key="{self.apikey}",algorithm="hmac-sha256",headers="host date request-line",signature="{sign}"'
        auth = base64.b64encode(auth_raw.encode()).decode()
        params = urlencode({"authorization": auth, "date": date, "host": host})
        return f"{SPARK_WSS_URL}?{params}"

    def chat_sync(self, question: str, timeout=30):
        if not self._lock.acquire(blocking=True, timeout=timeout):
            return None, "繁忙：上一轮请求尚未结束"
        try:
            url = self._build_auth_url()
            reply_buffer = []
            complete_flag = False
            err_msg = None
            send_payload = None
            conn_open_event = threading.Event()

            def on_open(ws):
                conn_open_event.set()
                ws.send(json.dumps(send_payload))

            def on_message(ws, message):
                nonlocal complete_flag
                data = json.loads(message)
                if data["header"]["code"] != 0:
                    nonlocal err_msg
                    err_msg = f"{data['header']['code']}:{data['header']['message']}"
                    complete_flag = True
                    ws.close()
                    return
                chunk = data["payload"]["choices"]["text"][0]["content"]
                reply_buffer.append(chunk)
                if data["payload"]["choices"]["status"] == 2:
                    complete_flag = True
                    ws.close()

            def on_error(ws, error):
                nonlocal err_msg
                err_msg = str(error)
                complete_flag = True

            if ENABLE_HISTORY:
                self.chat_history.append({"role": "user", "content": question})
                recent_msg = self.chat_history[-(MAX_HISTORY_LEN - 1):]
                payload_msg = [self.system_prompt] + recent_msg
            else:
                payload_msg = [self.system_prompt, {"role": "user", "content": question}]

            send_payload = {
                "header": {"app_id": self.appid},
                "parameter": {"chat": {"domain": DOMAIN, "temperature": 0.5, "max_tokens": 1024}},
                "payload": {"message": {"text": payload_msg}}
            }

            ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error)
            threading.Thread(target=ws.run_forever, daemon=True).start()

            if not conn_open_event.wait(timeout=3.0):
                ws.close()
                return None, "WebSocket 连接建立超时"

            start_time = time.time()
            while not complete_flag and (time.time() - start_time < timeout):
                time.sleep(0.05)

            full_answer = "".join(reply_buffer)
            if err_msg:
                return None, err_msg
            if not complete_flag:
                return None, "请求超时"

            if ENABLE_HISTORY:
                self.chat_history.append({"role": "assistant", "content": full_answer})
            return full_answer, None
        finally:
            self._lock.release()


class Xunfei_Llm_ROS:
    def __init__(self):
        self.client = SparkClient()
        self.pub = rospy.Publisher("/voice/tts", String, queue_size=1)
        rospy.Subscriber("/voice/stt", String, self.callback, queue_size=1)
        self._last_ask_time = 0.0  # 防抖锁
        rospy.loginfo("✅ 星火大模型对话节点已就绪")

    def callback(self, msg):
        text = str(msg.data).strip()
        if not text or len(text) <= 1:
            return

        if any(kw in text for kw in LOCAL_KEYWORDS):
            return

        # 2.5秒内只处理一次提问，防止麦克风把一句话拆成多句导致重复回复
        now = time.time()
        if now - self._last_ask_time < 2.5:
            return
        self._last_ask_time = now

        rospy.loginfo("【大模型收到提问】%s", text)
        answer, error = self.client.chat_sync(text)
        
        if error is not None:
            rospy.logerr("调用星火失败：%s", error)
            self.pub.publish("对不起，我网络好像出了点问题，请稍后再试。")
            return
            
        rospy.loginfo("【大模型回复】%s", answer)
        self.pub.publish(answer)

if __name__ == '__main__':
    rospy.init_node("xunfei_llm_node")
    Xunfei_Llm_ROS()
    rospy.spin()
