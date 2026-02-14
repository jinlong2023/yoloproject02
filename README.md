# Z-2Mini 集成修改说明

## 文件变更一览

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

## 具体修改内容

### 1. `config.py` — 新增字段

`GimbalConfig` 新增了 4 个字段:

```python
simulate_mode: bool = True           # ← 改为 False 启用硬件
gcu_ip: str = "192.168.144.108"
comm_mode: str = "udp"               # "udp" 或 "tcp"
track_mode: str = "gimbal_builtin"   # "gimbal_builtin" 或 "software_pid"
```

新增 `get_z2mini_config()` 函数，一键获取硬件配置。

### 2. `gimbal_controller.py` — 核心修改

**保留了全部原有代码**（PIDController、MotionPredictor、GimbalSimulator、GimbalController），新增:

- `GimbalHardwareZ2Mini` 类: 实现与 `GimbalSimulator` **相同接口**
  - `apply_command(yaw_cmd, pitch_cmd, dt)` → 通过 0x11 指向锁定模式发送角速度
  - `get_state()` → 从 GCU 回传数据获取真实姿态
  - `reset()` → 发送回中指令
  - `start_tracking() / stop_tracking()` → 0x17 吊舱内置跟踪
- `pixel_to_gcu_coord()`: 像素坐标到 GCU [0,10000] 坐标转换
- `GimbalController` 修改:
  - 构造函数: `simulate_mode=False` 时自动创建 `GimbalHardwareZ2Mini`
  - `compute_control()`: 新增 `target_bbox` 参数，`gimbal_builtin` 模式分支

### 3. `main.py` — 主流程修改

- 新增 `RTSPCapture` 类: 线程化 RTSP 取流，消除缓冲延迟
- `_open_video()`: 支持 `rtsp://` 前缀自动使用 RTSPCapture
- `process_frame()`: 将主目标 bbox 传给 `compute_control(target_bbox=...)`
- 新增快捷键: `p`=拍照, `v`=录像, `h`=回中
- CLI: `--z2mini`, `--gcu-ip`, `--comm-mode`, `--track-mode`

### 4. `gcu_protocol.py` — 全新文件

完整实现 GCU 私有协议 V2.0.6:
- CRC16 校验、数据包构建/解析
- UDP/TCP 通信管理
- 高级命令封装 (跟踪、指点、变焦、拍照等)

---

## 使用方法

### 仿真模式 (与你原来完全一样)

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
