#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import signal
import subprocess
import threading
import rospy
from std_msgs.msg import String

class VoiceFollowTrigger:
    def __init__(self):
        self.start_keywords = ["跟我走", "跟我来", "跟着我", "跟着", "跟随我"]
        self.stop_keywords  = ["停下", "停止", "别跟了", "不用跟了", "停"]

        self.follow_proc = None
        self.proc_lock = threading.Lock()
        self.follow_node_names = ["nxfollower", "nxfollower_nodelet", "nxfollower_nodelet_manager"]

        self.tts_pub = rospy.Publisher("/voice/tts", String, queue_size=10)
        rospy.Subscriber("/voice/stt", String, self.stt_callback, queue_size=1)
        rospy.on_shutdown(self._cleanup)
        rospy.loginfo("语音跟随触发节点就绪...")

    def stt_callback(self, msg):
        text = str(msg.data).strip()
        if not text:
            return

        if any(kw in text for kw in self.stop_keywords):
            self.stop_follow()
            return

        if any(kw in text for kw in self.start_keywords):
            self.start_follow()

    def _node_names_bases(self):
        # 【修改】通过 rosnode list 获取节点，彻底避免 rospy API 报错
        try:
            output = subprocess.check_output(["rosnode", "list"], stderr=subprocess.DEVNULL).decode('utf-8')
            nodes = output.strip().split('\n')
            return {name.lstrip("/").split("/")[-1] for name in nodes if name}
        except Exception:
            return set()

    def is_follow_running(self):
        if self.follow_proc is not None and self.follow_proc.poll() is None:
            return True
        bases = self._node_names_bases()
        return any(n in bases for n in self.follow_node_names)

    def start_follow(self):
        with self.proc_lock:
            if self.is_follow_running():
                self._speak("跟随已经开启了。")
                return
            try:
                rospy.loginfo("正在启动 spark_follower...")
                self.follow_proc = subprocess.Popen(
                    ["roslaunch", "spark_follower", "nxfollower_bringup_nodelet.launch"],
                    start_new_session=True,
                )
                self._speak("好的，开始跟随。")
            except Exception as e:
                rospy.logerr("启动跟随失败：%s", e)
                self._speak("跟随启动失败了。")

    def stop_follow(self):
        with self.proc_lock:
            if self.follow_proc is not None and self.follow_proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.follow_proc.pid), signal.SIGINT)
                    try:
                        self.follow_proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(self.follow_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                finally:
                    self.follow_proc = None
                    self._speak("已停止跟随。")
                return

            bases = self._node_names_bases()
            targets = [n for n in self.follow_node_names if n in bases]
            if targets:
                for n in targets:
                    os.system("rosnode kill /{} >/dev/null 2>&1".format(n))
                self._speak("已停止跟随。")
            else:
                self._speak("跟随本来就没有开启。")

    def _speak(self, text):
        for _ in range(20):
            if self.tts_pub.get_num_connections() > 0:
                break
            rospy.sleep(0.05)
        self.tts_pub.publish(String(data=text))

    def _cleanup(self):
        with self.proc_lock:
            if self.follow_proc is not None and self.follow_proc.poll() is None:
                try:
                    os.killpg(os.getpgid(self.follow_proc.pid), signal.SIGINT)
                except Exception:
                    pass
                self.follow_proc = None

if __name__ == "__main__":
    rospy.init_node("voice_follow_trigger", anonymous=False)
    VoiceFollowTrigger()
    rospy.spin()
