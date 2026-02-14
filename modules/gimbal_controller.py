"""
========================================================================
模块3: 云台PID控制模块 (运动预测 + 平滑补偿)
========================================================================
功能:
- 双轴PID闭环控制 (Yaw/Pitch)
- 运动预测前馈补偿
- 输出平滑化与死区处理
- 积分限幅与抗饱和
- 云台仿真器 (无硬件模式)
- Z-2Mini 硬件驱动 (GCU私有协议)
- 响应曲线与误差分析

修改记录:
- 新增 GimbalHardwareZ2Mini 类 (对接真实吊舱)
- GimbalController 在 simulate_mode=False 时自动使用硬件驱动
- 支持两种跟踪策略: software_pid / gimbal_builtin
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from collections import deque
from dataclasses import dataclass


@dataclass
class GimbalState:
    """云台状态"""
    yaw: float = 0.0          # 当前偏航角 (°)
    pitch: float = 0.0        # 当前俯仰角 (°)
    yaw_speed: float = 0.0    # 偏航角速度 (°/s)
    pitch_speed: float = 0.0  # 俯仰角速度 (°/s)
    timestamp: float = 0.0


@dataclass
class ControlOutput:
    """控制输出"""
    yaw_cmd: float = 0.0      # 偏航控制量
    pitch_cmd: float = 0.0    # 俯仰控制量
    yaw_error: float = 0.0    # 偏航误差 (像素)
    pitch_error: float = 0.0  # 俯仰误差 (像素)
    is_locked: bool = False   # 是否锁定目标
    timestamp: float = 0.0


# ====================================================================
# 增量式PID控制器  (原有代码，未修改)
# ====================================================================
class PIDController:
    """
    增量式PID控制器
    支持积分限幅、微分滤波、死区控制
    """

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float = 180.0,
                 integral_limit: float = 50.0,
                 dead_zone: float = 0.0,
                 dt: float = 0.02):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.dead_zone = dead_zone
        self.dt = dt

        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_derivative = 0.0
        self.derivative_filter_alpha = 0.2

        self.output_history = deque(maxlen=500)
        self.error_history = deque(maxlen=500)

    def compute(self, error: float, dt: Optional[float] = None) -> float:
        if dt is None:
            dt = self.dt

        if abs(error) < self.dead_zone:
            self.error_history.append(error)
            self.output_history.append(0.0)
            return 0.0

        p_term = self.kp * error

        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral

        if dt > 0:
            raw_derivative = (error - self.prev_error) / dt
        else:
            raw_derivative = 0.0

        alpha = self.derivative_filter_alpha
        filtered_derivative = alpha * raw_derivative + (1 - alpha) * self.prev_derivative
        d_term = self.kd * filtered_derivative

        output = p_term + i_term + d_term
        output = np.clip(output, -self.output_limit, self.output_limit)

        if abs(output) >= self.output_limit:
            self.integral -= error * dt

        self.prev_error = error
        self.prev_derivative = filtered_derivative

        self.error_history.append(error)
        self.output_history.append(output)

        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_derivative = 0.0

    def get_stats(self) -> Dict:
        if not self.error_history:
            return {'mean_error': 0, 'max_error': 0, 'std_error': 0}
        errors = np.array(list(self.error_history))
        return {
            'mean_error': float(np.mean(np.abs(errors))),
            'max_error': float(np.max(np.abs(errors))),
            'std_error': float(np.std(errors)),
            'integral': float(self.integral)
        }


# ====================================================================
# 运动预测器  (原有代码，未修改)
# ====================================================================
class MotionPredictor:
    """目标运动预测器 - 基于历史运动状态预测未来位置"""

    def __init__(self, horizon: int = 5, smooth_factor: float = 0.7):
        self.horizon = horizon
        self.smooth_factor = smooth_factor

        self.position_history = deque(maxlen=60)
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.smoothed_velocity = np.array([0.0, 0.0])
        self.prediction_errors = deque(maxlen=100)

    def update(self, position: Tuple[float, float], timestamp: float):
        self.position_history.append({'pos': np.array(position), 'time': timestamp})

        if len(self.position_history) >= 2:
            p1 = self.position_history[-1]
            p0 = self.position_history[-2]
            dt = p1['time'] - p0['time']
            if dt > 0:
                new_vel = (p1['pos'] - p0['pos']) / dt
                alpha = self.smooth_factor
                self.smoothed_velocity = alpha * self.smoothed_velocity + (1 - alpha) * new_vel
                self.velocity = new_vel

        if len(self.position_history) >= 3:
            p2 = self.position_history[-1]
            p1 = self.position_history[-2]
            p0 = self.position_history[-3]
            dt1 = p2['time'] - p1['time']
            dt0 = p1['time'] - p0['time']
            if dt1 > 0 and dt0 > 0:
                v1 = (p2['pos'] - p1['pos']) / dt1
                v0 = (p1['pos'] - p0['pos']) / dt0
                self.acceleration = (v1 - v0) / ((dt1 + dt0) / 2)

    def predict(self, steps: Optional[int] = None, dt: float = 1.0 / 30) -> np.ndarray:
        if steps is None:
            steps = self.horizon
        if not self.position_history:
            return np.array([0.0, 0.0])
        t = steps * dt
        return self.smoothed_velocity * t + 0.5 * self.acceleration * t * t

    def evaluate_prediction(self, actual_position, predicted_offset, base_position):
        predicted_position = base_position + predicted_offset
        error = np.linalg.norm(actual_position - predicted_position)
        self.prediction_errors.append(error)

    @property
    def mean_prediction_error(self) -> float:
        if not self.prediction_errors:
            return 0.0
        return float(np.mean(list(self.prediction_errors)))


# ====================================================================
# 云台仿真器  (原有代码，未修改)
# ====================================================================
class GimbalSimulator:
    """云台物理仿真器 (无硬件时使用)"""

    def __init__(self, config, damping: float = 0.3):
        self.config = config
        self.damping = damping
        self.state = GimbalState()
        self.state.timestamp = time.time()

    def apply_command(self, yaw_cmd: float, pitch_cmd: float, dt: float):
        cfg = self.config
        yaw_cmd = np.clip(yaw_cmd, -cfg.max_yaw_speed, cfg.max_yaw_speed)
        pitch_cmd = np.clip(pitch_cmd, -cfg.max_pitch_speed, cfg.max_pitch_speed)

        self.state.yaw_speed = (
            self.damping * self.state.yaw_speed +
            (1 - self.damping) * yaw_cmd
        )
        self.state.pitch_speed = (
            self.damping * self.state.pitch_speed +
            (1 - self.damping) * pitch_cmd
        )

        self.state.yaw += self.state.yaw_speed * dt
        self.state.pitch += self.state.pitch_speed * dt

        self.state.yaw = np.clip(self.state.yaw, -cfg.max_yaw_angle, cfg.max_yaw_angle)
        self.state.pitch = np.clip(self.state.pitch, cfg.min_pitch_angle, cfg.max_pitch_angle)
        self.state.timestamp = time.time()

    def get_state(self) -> GimbalState:
        return self.state

    def reset(self):
        self.state = GimbalState()
        self.state.timestamp = time.time()


# ====================================================================
# ★ Z-2Mini 硬件驱动  (新增)
# ====================================================================
class GimbalHardwareZ2Mini:
    """
    Z-2Mini 云台硬件驱动

    通过 GCU 私有协议与真实吊舱通信，实现与 GimbalSimulator 相同的接口。
    同时提供吊舱内置跟踪等高级功能。

    接口兼容:
        apply_command(yaw_cmd, pitch_cmd, dt)  # PID 角速度控制
        get_state() -> GimbalState
        reset()
    """

    def __init__(self, config, gcu_ip: str = "192.168.144.108",
                 comm_mode: str = "udp"):
        self.config = config
        self.gcu_ip = gcu_ip
        self.comm_mode = comm_mode

        self.state = GimbalState()
        self.state.timestamp = time.time()

        # GCU 通信 (延迟导入避免无硬件时出错)
        self._conn = None
        self._commander = None
        self._gcu_status = None       # 最新 GCU 回传状态
        self._connected = False
        self._tracking_active = False  # 吊舱内置跟踪是否激活

    def connect(self) -> bool:
        """连接 GCU"""
        try:
            from modules.gcu_protocol import GCUConnection, GCUCommander
            self._conn = GCUConnection(self.gcu_ip, self.comm_mode)
            if not self._conn.connect():
                print("[Z2Mini] ⚠ GCU 连接失败, 回退到心跳模式")
                return False
            self._commander = GCUCommander(self._conn)
            self._connected = True

            # 首次心跳获取状态
            status = self._commander.heartbeat()
            if status:
                self._update_state(status)
                print(f"[Z2Mini] ✓ 已连接! 模式={status.mode_name}, "
                      f"Pitch={status.abs_pitch:.1f}°, Yaw={status.abs_yaw:.1f}°, "
                      f"Zoom={status.cam1_zoom:.1f}x")
            return True
        except ImportError:
            print("[Z2Mini] ⚠ gcu_protocol 模块未找到")
            return False
        except Exception as e:
            print(f"[Z2Mini] 连接异常: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        if self._tracking_active:
            self.stop_tracking()
        if self._conn:
            self._conn.disconnect()
        self._connected = False

    def _update_state(self, gcu_status):
        """从 GCU 回传更新内部状态"""
        if gcu_status is None:
            return
        self._gcu_status = gcu_status
        self.state.yaw = gcu_status.abs_yaw
        self.state.pitch = gcu_status.abs_pitch
        self.state.yaw_speed = gcu_status.gyro_z
        self.state.pitch_speed = gcu_status.gyro_y
        self.state.timestamp = time.time()

    # ---- 与 GimbalSimulator 兼容的接口 ----

    def apply_command(self, yaw_cmd: float, pitch_cmd: float, dt: float):
        """
        PID 角速度控制 (software_pid 模式使用)

        将 PID 输出的 °/s 角速度转换为 GCU 协议的 0.1°/s 单位,
        通过指向锁定模式 (0x11) 发送给吊舱。
        """
        if not self._connected or not self._commander:
            # 未连接时回退到仿真
            return

        # 你的 PID 输出单位是 °/s, GCU 协议单位是 0.1°/s
        yaw_speed_raw = int(yaw_cmd * 10)
        pitch_speed_raw = int(pitch_cmd * 10)

        # 变焦补偿: GCU 实际角速度 = 设定值 ÷ 变焦倍率
        # 所以高变焦时需要放大控制量
        zoom = 1.0
        if self._gcu_status and self._gcu_status.cam1_zoom > 0:
            zoom = self._gcu_status.cam1_zoom

        yaw_speed_compensated = int(yaw_speed_raw * zoom)
        pitch_speed_compensated = int(pitch_speed_raw * zoom)

        # 限幅到 S16 范围
        yaw_speed_compensated = max(-32000, min(yaw_speed_compensated, 32000))
        pitch_speed_compensated = max(-32000, min(pitch_speed_compensated, 32000))

        status = self._commander.pointing_lock(
            pitch_speed=pitch_speed_compensated,
            yaw_speed=yaw_speed_compensated,
            ctrl_valid=True,
        )
        self._update_state(status)

    def get_state(self) -> GimbalState:
        return self.state

    def reset(self):
        """回中"""
        if self._connected and self._commander:
            self._commander.home()
            time.sleep(0.1)
            status = self._commander.heartbeat()
            self._update_state(status)
        self.state = GimbalState()
        self.state.timestamp = time.time()

    # ---- Z-2Mini 特有功能 (吊舱内置跟踪) ----

    def start_tracking(self, x0: int, y0: int, x1: int, y1: int) -> bool:
        """
        启动吊舱内置跟踪 (0x17)

        Args:
            x0, y0: 目标框左上角, GCU 坐标 [0, 10000]
            x1, y1: 目标框右下角, GCU 坐标 [0, 10000]
        """
        if not self._connected or not self._commander:
            return False

        status = self._commander.start_tracking(x0, y0, x1, y1)
        self._update_state(status)
        self._tracking_active = True
        return True

    def stop_tracking(self) -> bool:
        if not self._connected or not self._commander:
            return False
        status = self._commander.stop_tracking()
        self._update_state(status)
        self._tracking_active = False
        return True

    def heartbeat(self):
        """发送心跳并更新状态"""
        if self._connected and self._commander:
            status = self._commander.heartbeat()
            self._update_state(status)

    def point_move(self, x: int, y: int):
        """指点平移"""
        if self._connected and self._commander:
            status = self._commander.point_move(x, y)
            self._update_state(status)

    def take_photo(self):
        if self._connected and self._commander:
            self._commander.take_photo()

    def toggle_record(self):
        if self._connected and self._commander:
            self._commander.toggle_record()

    @property
    def gcu_status(self):
        """获取最新 GCU 原始状态"""
        return self._gcu_status

    @property
    def is_gimbal_tracking(self) -> bool:
        """吊舱内置跟踪是否正在进行"""
        if self._gcu_status:
            return (self._gcu_status.mode == 0x17 and
                    self._gcu_status.is_tracking_ok)
        return False

    @property
    def tracking_miss(self) -> Tuple[int, int]:
        """获取吊舱跟踪脱靶量"""
        if self._gcu_status:
            return (self._gcu_status.miss_h, self._gcu_status.miss_v)
        return (0, 0)


# ====================================================================
# 坐标转换工具  (新增)
# ====================================================================
def pixel_to_gcu_coord(px: int, py: int, frame_w: int, frame_h: int) -> Tuple[int, int]:
    """
    像素坐标 → GCU 协议坐标 [0, 10000]

    GCU 坐标系: 图像左上角为原点, x∈[0,10000], y∈[0,10000]
    """
    gx = int(px / frame_w * 10000)
    gy = int(py / frame_h * 10000)
    return (max(0, min(gx, 10000)), max(0, min(gy, 10000)))


# ====================================================================
# 云台控制器主类  (修改: 集成 Z-2Mini 硬件 + 双跟踪模式)
# ====================================================================
class GimbalController:
    """
    云台闭环控制器主类

    当 simulate_mode=True:  使用 GimbalSimulator (仿真)
    当 simulate_mode=False: 使用 GimbalHardwareZ2Mini (真实硬件)

    跟踪模式 (track_mode):
      - "software_pid":   上位机 PID 持续控制角速度 (原有逻辑)
      - "gimbal_builtin": YOLO 检测框 → 0x17 指令, 吊舱内置跟踪接管 (推荐)
    """

    def __init__(self, config, frame_size: Tuple[int, int] = (1280, 720)):
        self.config = config
        self.frame_width, self.frame_height = frame_size

        self.frame_center_x = self.frame_width / 2.0
        self.frame_center_y = self.frame_height / 2.0

        # 双轴PID控制器
        self.pid_yaw = PIDController(
            kp=config.kp_yaw, ki=config.ki_yaw, kd=config.kd_yaw,
            output_limit=config.max_yaw_speed,
            integral_limit=config.integral_limit,
            dead_zone=config.dead_zone,
            dt=1.0 / config.control_frequency
        )
        self.pid_pitch = PIDController(
            kp=config.kp_pitch, ki=config.ki_pitch, kd=config.kd_pitch,
            output_limit=config.max_pitch_speed,
            integral_limit=config.integral_limit,
            dead_zone=config.dead_zone,
            dt=1.0 / config.control_frequency
        )

        # 运动预测器
        self.motion_predictor = MotionPredictor(
            horizon=config.prediction_horizon,
            smooth_factor=config.compensation_factor
        )

        # ★ 云台后端选择
        if config.simulate_mode:
            self.gimbal = GimbalSimulator(config)
            self._is_hardware = False
            print("[Gimbal] 运行在仿真模式")
        else:
            hw = GimbalHardwareZ2Mini(
                config,
                gcu_ip=getattr(config, 'gcu_ip', '192.168.144.108'),
                comm_mode=getattr(config, 'comm_mode', 'udp'),
            )
            if hw.connect():
                self.gimbal = hw
                self._is_hardware = True
                print("[Gimbal] ✓ Z-2Mini 硬件模式")
            else:
                print("[Gimbal] ⚠ 硬件连接失败, 自动回退到仿真模式")
                self.gimbal = GimbalSimulator(config)
                self._is_hardware = False

        # 跟踪模式
        self._track_mode = getattr(config, 'track_mode', 'software_pid')
        if self._is_hardware:
            print(f"[Gimbal] 跟踪模式: {self._track_mode}")

        # 输出平滑
        self.smooth_factor = config.output_smooth_factor
        self.prev_yaw_cmd = 0.0
        self.prev_pitch_cmd = 0.0

        # 控制记录
        self.control_history = deque(maxlen=1000)
        self.response_curve = deque(maxlen=500)
        self.frame_count = 0

        # 锁定状态
        self.lock_threshold = 10.0
        self.is_locked = False
        self.lock_duration = 0

        # 吊舱内置跟踪状态
        self._builtin_tracking_active = False
        self._builtin_lost_time = 0.0
        self._builtin_lost_timeout = 3.0   # 秒

    def compute_control(self, target_position: Optional[Tuple[float, float]],
                        target_velocity: Optional[Tuple[float, float]] = None,
                        timestamp: Optional[float] = None,
                        target_bbox: Optional[Tuple[int, int, int, int]] = None,
                        ) -> ControlOutput:
        """
        计算云台控制输出

        Args:
            target_position: 目标在图像中的中心坐标 (x, y) 像素
            target_velocity: 目标速度 (vx, vy) 像素/帧
            timestamp: 时间戳
            target_bbox: 目标边界框 (x1,y1,x2,y2) 像素
                         gimbal_builtin 模式下必需

        Returns:
            ControlOutput
        """
        self.frame_count += 1
        if timestamp is None:
            timestamp = time.time()

        # ===== 分支: 吊舱内置跟踪模式 =====
        if (self._is_hardware and self._track_mode == "gimbal_builtin"):
            return self._compute_builtin_tracking(
                target_position, target_bbox, timestamp
            )

        # ===== 原有逻辑: 软件 PID 模式 =====
        return self._compute_pid_control(
            target_position, target_velocity, timestamp
        )

    def _compute_pid_control(self, target_position, target_velocity, timestamp
                             ) -> ControlOutput:
        """原有 PID 控制逻辑 (未修改)"""
        if target_position is None:
            self.pid_yaw.reset()
            self.pid_pitch.reset()
            self.is_locked = False
            self.lock_duration = 0
            return ControlOutput(timestamp=timestamp)

        tx, ty = target_position
        error_x = tx - self.frame_center_x
        error_y = ty - self.frame_center_y

        # 运动预测前馈补偿
        feedforward_x, feedforward_y = 0.0, 0.0
        if self.config.prediction_enabled and target_velocity is not None:
            self.motion_predictor.update((tx, ty), timestamp)
            pred_offset = self.motion_predictor.predict(
                steps=self.config.prediction_horizon,
                dt=1.0 / self.config.control_frequency
            )
            feedforward_x = pred_offset[0] * self.config.compensation_factor
            feedforward_y = pred_offset[1] * self.config.compensation_factor

        total_error_x = error_x + feedforward_x
        total_error_y = error_y + feedforward_y

        dt = 1.0 / self.config.control_frequency
        yaw_cmd = self.pid_yaw.compute(total_error_x, dt)
        pitch_cmd = self.pid_pitch.compute(total_error_y, dt)

        alpha = self.smooth_factor
        yaw_cmd = alpha * yaw_cmd + (1 - alpha) * self.prev_yaw_cmd
        pitch_cmd = alpha * pitch_cmd + (1 - alpha) * self.prev_pitch_cmd
        self.prev_yaw_cmd = yaw_cmd
        self.prev_pitch_cmd = pitch_cmd

        # 应用到云台
        self.gimbal.apply_command(yaw_cmd, pitch_cmd, dt)

        pixel_error = np.sqrt(error_x ** 2 + error_y ** 2)
        if pixel_error < self.lock_threshold:
            self.is_locked = True
            self.lock_duration += 1
        else:
            self.is_locked = False
            self.lock_duration = 0

        output = ControlOutput(
            yaw_cmd=yaw_cmd, pitch_cmd=pitch_cmd,
            yaw_error=error_x, pitch_error=error_y,
            is_locked=self.is_locked, timestamp=timestamp
        )

        self._record(error_x, error_y, yaw_cmd, pitch_cmd, pixel_error, timestamp)
        return output

    def _compute_builtin_tracking(self, target_position, target_bbox, timestamp
                                  ) -> ControlOutput:
        """
        ★ 吊舱内置跟踪模式

        流程:
        1. YOLO 有检测结果 → 发送 0x17 启动跟踪 (or 更新目标框)
        2. 吊舱内置跟踪接管 → 读取脱靶量反馈
        3. YOLO 无检测 + 吊舱跟踪仍然成功 → 维持心跳
        4. 都丢失 → 超时后停止跟踪
        """
        hw = self.gimbal  # type: GimbalHardwareZ2Mini

        if target_position is not None and target_bbox is not None:
            # --- YOLO 有检测到目标 ---
            self._builtin_lost_time = 0.0

            x1, y1, x2, y2 = target_bbox
            gx0, gy0 = pixel_to_gcu_coord(x1, y1, self.frame_width, self.frame_height)
            gx1, gy1 = pixel_to_gcu_coord(x2, y2, self.frame_width, self.frame_height)

            # 判断是否需要 (重新) 启动跟踪
            need_start = False
            gcu = hw.gcu_status
            if gcu is None:
                need_start = True
            elif gcu.mode != 0x17:
                need_start = True
            elif not gcu.is_tracking_ok:
                need_start = True
            elif not self._builtin_tracking_active:
                need_start = True

            if need_start:
                hw.start_tracking(gx0, gy0, gx1, gy1)
                self._builtin_tracking_active = True
                print(f"[Track] 发送跟踪: ({gx0},{gy0})-({gx1},{gy1})")
            else:
                hw.heartbeat()

            # 计算误差 (用于 UI 显示)
            tx, ty = target_position
            error_x = tx - self.frame_center_x
            error_y = ty - self.frame_center_y

        elif hw.is_gimbal_tracking:
            # --- YOLO 未检测到, 但吊舱仍在跟踪 ---
            hw.heartbeat()
            miss_h, miss_v = hw.tracking_miss
            # 脱靶量映射为像素误差 (近似)
            error_x = miss_h / 1000.0 * self.frame_center_x
            error_y = miss_v / 1000.0 * self.frame_center_y

        else:
            # --- 都丢失 ---
            if self._builtin_tracking_active:
                if self._builtin_lost_time == 0.0:
                    self._builtin_lost_time = timestamp
                elif timestamp - self._builtin_lost_time > self._builtin_lost_timeout:
                    hw.stop_tracking()
                    self._builtin_tracking_active = False
                    print("[Track] 超时停止跟踪")
            hw.heartbeat()
            error_x, error_y = 0.0, 0.0
            self.is_locked = False
            self.lock_duration = 0
            return ControlOutput(timestamp=timestamp)

        # 锁定判断
        pixel_error = np.sqrt(error_x ** 2 + error_y ** 2)
        if pixel_error < self.lock_threshold:
            self.is_locked = True
            self.lock_duration += 1
        else:
            self.is_locked = False
            self.lock_duration = 0

        output = ControlOutput(
            yaw_cmd=0, pitch_cmd=0,  # 吊舱自主控制, 无需PID输出
            yaw_error=error_x, pitch_error=error_y,
            is_locked=self.is_locked, timestamp=timestamp
        )
        self._record(error_x, error_y, 0, 0, pixel_error, timestamp)
        return output

    def _record(self, error_x, error_y, yaw_cmd, pitch_cmd, pixel_error, timestamp):
        """记录控制历史"""
        gimbal_state = self.gimbal.get_state()
        self.control_history.append({
            'frame': self.frame_count,
            'error_x': error_x, 'error_y': error_y,
            'yaw_cmd': yaw_cmd, 'pitch_cmd': pitch_cmd,
            'gimbal_yaw': gimbal_state.yaw, 'gimbal_pitch': gimbal_state.pitch,
            'pixel_error': pixel_error,
            'is_locked': self.is_locked,
            'timestamp': timestamp
        })
        self.response_curve.append({
            'time': timestamp, 'error': pixel_error,
            'yaw_angle': gimbal_state.yaw, 'pitch_angle': gimbal_state.pitch
        })

    def get_gimbal_state(self) -> GimbalState:
        return self.gimbal.get_state()

    def reset(self):
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.gimbal.reset()
        self.prev_yaw_cmd = 0.0
        self.prev_pitch_cmd = 0.0
        self.is_locked = False
        self.lock_duration = 0
        self._builtin_tracking_active = False

    def get_stats(self) -> Dict:
        gimbal_state = self.gimbal.get_state()
        stats = {
            'frame_count': self.frame_count,
            'gimbal_yaw': round(gimbal_state.yaw, 2),
            'gimbal_pitch': round(gimbal_state.pitch, 2),
            'is_locked': self.is_locked,
            'lock_duration': self.lock_duration,
            'pid_yaw_stats': self.pid_yaw.get_stats(),
            'pid_pitch_stats': self.pid_pitch.get_stats(),
            'prediction_error': round(self.motion_predictor.mean_prediction_error, 2),
            'is_hardware': self._is_hardware,
            'track_mode': self._track_mode,
        }
        # 硬件模式附加信息
        if self._is_hardware and hasattr(self.gimbal, 'gcu_status'):
            gcu = self.gimbal.gcu_status
            if gcu:
                stats['gcu_mode'] = gcu.mode_name
                stats['gcu_zoom'] = gcu.cam1_zoom
                stats['gcu_tracking'] = gcu.is_tracking_ok
        return stats

    def get_response_data(self) -> Dict:
        if not self.control_history:
            return {}
        history = list(self.control_history)
        return {
            'errors_x': [h['error_x'] for h in history],
            'errors_y': [h['error_y'] for h in history],
            'pixel_errors': [h['pixel_error'] for h in history],
            'yaw_cmds': [h['yaw_cmd'] for h in history],
            'pitch_cmds': [h['pitch_cmd'] for h in history],
            'gimbal_yaws': [h['gimbal_yaw'] for h in history],
            'gimbal_pitches': [h['gimbal_pitch'] for h in history],
            'locked': [h['is_locked'] for h in history],
            'frames': [h['frame'] for h in history]
        }

    # ---- 快捷操作 (硬件模式) ----

    def take_photo(self):
        if self._is_hardware and hasattr(self.gimbal, 'take_photo'):
            self.gimbal.take_photo()

    def toggle_record(self):
        if self._is_hardware and hasattr(self.gimbal, 'toggle_record'):
            self.gimbal.toggle_record()
