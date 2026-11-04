# G11 智能语音交互与多模态跟随机器人

本项目基于 ROS Noetic 开发，集成了人脸识别、大模型对话、天气查询与语音跟随功能。

## 运行方法：
1. 进入 `scripts/` 目录，将 `config_example.py` 复制并重命名为 `config.py`。
2. 在 `config.py` 中填入相关的 API Key（和风天气、讯飞、百度）。
3. 编译并运行：
   `roslaunch g11_robot bringup.launch 2>&1 | grep -v "messenger-libusb"`
