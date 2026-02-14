## 基于 YOLOv13 + 卡尔曼滤波 + PID控制 的动态目标追踪算法

---

## 系统架构

```
视频输入 → 光照自适应预处理 → YOLOv13目标检测 → CNN特征提取
         → 卡尔曼滤波多目标跟踪 → 主目标选择 → 运动预测
         → PID云台控制计算 → 云台角度执行 → 闭环反馈
```

## 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 配置中心 | `config.py` | 全局参数配置（检测、跟踪、PID、相机） |
| 目标检测 | `modules/detector.py` | YOLOv13多尺度检测 + 光照自适应 + CNN特征提取 |
| 目标跟踪 | `modules/tracker.py` | 卡尔曼滤波 + CNN外观匹配 + 光流估计 + 遮挡处理 |
| 云台控制 | `modules/gimbal_controller.py` | 双轴PID控制 + 运动预测前馈 + 输出平滑 + 云台仿真 |
| 可视化 | `modules/visualizer.py` | 实时画面叠加 + 轨迹绘制 + 数据记录 + 报告生成 |
| 主流程 | `main.py` | 系统集成 + 仿真测试 + 基准测试 |

## 安装与运行

### 1. 环境准备

```bash
# 创建虚拟环境 (推荐)
conda create -n 项目名称 python=3.11
conda activate 项目名称
# 安装PyTorch (根据CUDA版本选择)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行方式

```bash
# 摄像头实时模式
python main.py

# 视频文件模式
python main.py --source video.mp4

# 仿真测试模式 (无需摄像头)
python main.py --simulate

# 性能基准测试
python main.py --benchmark

# 自定义参数
python main.py --source 0 --model yolov8n.pt --device cuda:0 --conf 0.45 \
               --kp-yaw 0.8 --ki-yaw 0.05 --kd-yaw 0.15
```

### 3. PyCharm配置

1. 打开项目根目录 
2. 设置Python解释器为已安装依赖的环境
3. 右键 `main.py` → Run
4. 或配置 Run Configuration，在 Parameters 中添加参数如 `--simulate`

## 快捷键

| 按键 | 功能 |
|------|------|
| `q` / `ESC` | 退出 |
| `空格` | 暂停/继续 |
| `r` | 重置跟踪器 |
| `s` | 保存截图 |
| `t` | 切换轨迹显示 |
| `i` | 切换信息面板 |
| `鼠标左键` | 选择主跟踪目标 |

## 算法原理

### 1. 目标检测 (YOLOv13)
- 基于 ultralytics 框架的YOLO系列模型
- 多尺度特征金字塔检测
- CLAHE光照自适应预处理
- CNN外观特征提取（128维向量）用于跟踪关联

### 2. 多目标跟踪 (卡尔曼滤波 + CNN)
- **状态向量**: `[x, y, w, h, vx, vy, vw, vh]` (位置+速度)
- **匀速运动模型**预测 + 观测更新
- **数据关联**: 匈牙利算法，融合 IOU距离 + 余弦外观距离
- **光流辅助**: Lucas-Kanade稀疏光流，前后向一致性验证
- **遮挡处理**: 纯预测模式（最大15帧），特征队列保持重识别能力

### 3. PID云台控制
- **双轴独立PID**: Yaw(偏航) + Pitch(俯仰) 分别控制
- **增量式PID**: 积分限幅 + 微分低通滤波 + 死区控制
- **运动预测前馈**: 二次运动模型（位置+速度+加速度）预测补偿
- **输出平滑**: 指数移动平均，避免云台抖动
- **闭环反馈**: 目标偏离画面中心 → 误差 → PID → 云台转动 → 目标回归中心

## 输出文件

运行后在 `results/` 目录生成：

| 文件 | 内容 |
|------|------|
| `tracking_data.json` | 逐帧跟踪数据（位置、速度、误差、控制量） |
| `performance_report.json` | 性能分析报告 |
| `analysis_plots.png` | 分析图表（误差曲线、云台响应、PID输出、目标速度） |
| `output.mp4` | 录制的结果视频（需 `--record` 参数） |

## 项目结构

```
gimbal_tracking_system/
├── main.py                      # 主程序入口
├── config.py                    # 全局配置
├── requirements.txt             # 依赖清单
├── README.md                    # 说明文档
├── modules/
│   ├── __init__.py
│   ├── detector.py              # 目标检测模块
│   ├── tracker.py               # 目标跟踪模块
│   ├── gimbal_controller.py     # 云台控制模块
│   └── visualizer.py            # 可视化与分析模块
├── results/                     # 输出结果
├── logs/                        # 运行日志
└── data/                        # 测试数据
```

# Z-2Mini 集成修改说明
```
项目/
├── config.py                     [修改] 新增 Z-2Mini 硬件配置字段
├── main.py                       [修改] 支持 RTSP 取流 + --z2mini 启动参数
└── modules/
    ├── gcu_protocol.py           [新增] GCU 私有协议通信层
    ├── gimbal_controller.py      [修改] 新增 GimbalHardwareZ2Mini 硬件驱动
    ├── detector.py               [未修改]
    ├── tracker.py                [未修改]
    └── visualizer.py             [未修改]
```



### 1. `gcu_protocol.py` 

完整实现 GCU 私有协议 V2.0.6:
- CRC16 校验、数据包构建/解析
- UDP/TCP 通信管理
- 高级命令封装 (跟踪、指点、变焦、拍照等)

---

## 使用方法

### 仿真模式 

```bash
python main.py --source 0
python main.py --source video.mp4
```

### Z-2Mini 硬件模式

**前置准备:**
```bash
# 1. 上位机网络配置 (与 GCU 同子网)
sudo ip addr add 192.168.144.10/24 dev eth0

# 2. 测试视频流
ffplay rtsp://192.168.144.108

# 3. 测试 UDP 连通
python -c "
from modules.gcu_protocol import GCUConnection, GCUCommander
conn = GCUConnection('192.168.144.108', 'udp')
conn.connect()
cmd = GCUCommander(conn)
s = cmd.heartbeat()
print(f'Mode={s.mode_name}, Pitch={s.abs_pitch}, Yaw={s.abs_yaw}')
conn.disconnect()
"
```

**启动跟踪:**
```bash
# 吊舱内置跟踪 (推荐，低延迟)
python main.py --z2mini --model yolov13n.pt --target-class 0

# 软件PID跟踪 (更灵活)
python main.py --z2mini --track-mode software_pid

# 指定 GCU IP (如果修改过)
python main.py --z2mini --gcu-ip 192.168.144.200

# TCP 模式
python main.py --z2mini --comm-mode tcp
```

### 代码中直接使用

```python
from config import get_z2mini_config

config = get_z2mini_config()
config.detector.model_name = "yolov13s.pt"   # 换更大模型
config.detector.target_classes = [0, 2]       # 人+车
config.gimbal.track_mode = "gimbal_builtin"

system = GimbalTrackingSystem(config)
system.run()
```

---

## 两种跟踪模式对比

| | `gimbal_builtin` (推荐) | `software_pid` |
|---|---|---|
| 原理 | YOLO检测框 → 0x17指令 → 吊舱接管跟踪 | YOLO持续检测 → PID计算角速度 → 0x11指令 |
| 延迟 | 低 (吊舱硬件处理) | 较高 (RTSP→推理→控制 全链路) |
| CPU占用 | 低 (跟踪由吊舱完成) | 高 (每帧都需YOLO推理) |
| 切换目标 | 需要停止再重新发送 | 可以随时切换 |
| 目标丢失 | YOLO重新检测后自动恢复 | YOLO重新检测后自动恢复 |
| 适用场景 | 单目标持续跟踪 | 多目标切换、自定义逻辑 |

---

## 系统架构

```
                    你的上位机
┌──────────────────────────────────────────┐
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ detector │→│ tracker  │→│ gimbal │ │
│  │ (YOLO13) │  │(Kalman+  │  │controller│
│  │          │  │ CNN)     │  │          │ │
│  └─────┬────┘  └──────────┘  └─┬──┬───┘ │
│        │                        │  │      │
│  RTSP取流                  simulate │ hardware
│        │                    mode │  │ mode │
│        │              ┌─────┘  └──────┐  │
│        │              │ Simulator  │ Z2Mini │
│        │              │ (原有)     │ Hardware│
│        │              └───────────┘└──┬───┘  │
└────────┼──────────────────────────────┼──────┘
         │ rtsp://                      │ UDP/TCP
         │                              │ GCU协议
    ┌────┴──────────────────────────────┴──────┐
    │            Z-2Mini 云台相机                │
    │  4K可见光 + 热成像 + 内置跟踪 + 3轴稳定   │
    └──────────────────────────────────────────┘
```
