"""
GCU 通信协议模块 (基于 GCU 私有协议 V2.0.6)
已适配 SystemConfig / GimbalConfig
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional
from config import GimbalConfig

# ── 协议常量 ──────────────────────────────────────────────────
PROTOCOL_HEADER_SEND = bytes([0xA8, 0xE5])
PROTOCOL_HEADER_RECV = bytes([0x8A, 0x5E])
PROTOCOL_VERSION     = 0x01
GCU_SEND_PORT        = 2337   # 发送到设备的端口
GCU_RECV_PORT        = 2338   # 本地绑定接收端口

# ── 状态数据类 ─────────────────────────────────────────────────
@dataclass
class GimbalStatus:
    mode:           int
    pitch:          float
    yaw:            float
    roll:           float
    cam1_zoom:      float
    cam2_zoom:      float
    is_tracking_ok: bool
    target_x:       int
    target_y:       int

# ── CRC16 ─────────────────────────────────────────────────────
def _crc16(data: bytes) -> int:
    crc_ta = [
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
        0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef
    ]
    crc = 0
    for byte in data:
        da  = (crc >> 12) & 0x0F
        crc = (crc << 4) & 0xFFFF
        crc ^= crc_ta[da ^ ((byte >> 4) & 0x0F)]
        da  = (crc >> 12) & 0x0F
        crc = (crc << 4) & 0xFFFF
        crc ^= crc_ta[da ^ (byte & 0x0F)]
    return crc

# ── GCU 连接 ──────────────────────────────────────────────────
class GCUConnection:
    """GCU UDP 通信连接（兼容旧接口）"""

    MODE_POINTING_LOCK   = 0x11
    MODE_POINTING_FOLLOW = 0x12
    MODE_TRACKING        = 0x17
    FLAG_CONTROL_VALID   = 0x04
    FLAG_IMU_VALID       = 0x01

    def __init__(self, cfg: GimbalConfig):
        self.host = cfg.gcu_ip
        self.mode = cfg.comm_mode.lower()
        self.sock: Optional[socket.socket] = None
        self.running = False
        self.recv_thread: Optional[threading.Thread] = None

        self.latest_status: Optional[GimbalStatus] = None
        self.status_lock   = threading.Lock()
        self._recv_buffer  = bytearray()

        # 控制量
        self._roll    = 0
        self._pitch   = 0
        self._yaw     = 0
        self._ctrl_ok = True

        # 统计
        self._total_recv      = 0
        self._crc_errors      = 0
        self._last_recv_time  = 0.0
        self._last_err_report = 0.0

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("0.0.0.0", GCU_RECV_PORT))
            self.sock.settimeout(1.0)
            self.running = True
            self.recv_thread = threading.Thread(
                target=self._recv_loop, daemon=True)
            self.recv_thread.start()
            print(f"[GCU] 连接成功 (UDP {self.host}:{GCU_SEND_PORT}, 本地:{GCU_RECV_PORT})")
            return True
        except Exception as e:
            print(f"[GCU] 连接失败: {e}")
            return False

    def disconnect(self):
        self.running = False
        if self.recv_thread:
            self.recv_thread.join(timeout=2.0)
        if self.sock:
            self.sock.close()
        print("[GCU] 连接已断开")

    def shutdown(self):
        self.running = False
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self.sock.close()

    # ── 发送 ──────────────────────────────────────────────────
    def send(self, cmd_id: int = 0x00, payload: bytes = b"") -> bool:
        """兼容旧 GCUCommander 接口，cmd_id 映射到 GCU 指令字节"""
        CMD_MAP = {
            0x0001: 0x00,   # HEARTBEAT       → 空命令
            0x0002: 0x00,   # GIMBAL_CONTROL  → 控制量在 _pitch/_yaw
            0x0003: 0x00,   # START_TRACKING
            0x0004: 0x00,   # STOP_TRACKING
            0x0005: 0x11,   # POINTING_LOCK
            0x0006: 0x03,   # RESET_GIMBAL
            0x0007: 0x25,   # ZOOM_CONTROL    → CMD_ZOOM_SET
        }
        gcu_cmd = CMD_MAP.get(cmd_id, cmd_id & 0xFF)
        return self._send_packet(gcu_cmd)

    def _send_packet(self, command: int = 0x00) -> bool:
        if not self.sock:
            return False
        try:
            pkt = self._build_packet(command)
            self.sock.sendto(pkt, (self.host, GCU_SEND_PORT))
            return True
        except Exception as e:
            print(f"[GCU] 发送失败: {e}")
            return False

    def _build_packet(self, command: int = 0x00) -> bytes:
        pkt = bytearray()
        pkt.extend(PROTOCOL_HEADER_SEND)            # 2字节 帧头
        pkt.extend(struct.pack("<H", 72))            # 2字节 包长度
        pkt.append(PROTOCOL_VERSION)                 # 1字节 版本号
        pkt.extend(struct.pack("<h", self._roll))    # 2字节 滚转控制量
        pkt.extend(struct.pack("<h", self._pitch))   # 2字节 俯仰控制量
        pkt.extend(struct.pack("<h", self._yaw))     # 2字节 偏航控制量
        flag = 0
        if self._ctrl_ok:
            flag |= self.FLAG_CONTROL_VALID
        flag |= self.FLAG_IMU_VALID
        pkt.append(flag)                             # 1字节 状态标志
        pkt.extend(bytes(18))                        # 22字节 载机数据（全0）
        pkt.append(0x01)                             # 1字节 请求副帧
        pkt.extend(bytes(6))                         # 6字节 预留
        pkt.append(0x01)                             # 1字节 副帧帧头
        pkt.extend(bytes(31))                        # 31字节 GNSS数据（全0）
        pkt.append(command)                          # 1字节 指令
        crc = _crc16(bytes(pkt))
        pkt.extend(struct.pack(">H", crc))           # 2字节 CRC（大端）
        return bytes(pkt)

    def set_control(self, pitch: int = 0, yaw: int = 0, valid: bool = True):
        """设置控制量（单位: 0.1°/s）"""
        self._pitch   = pitch
        self._yaw     = yaw
        self._ctrl_ok = valid

    # ── 接收 ──────────────────────────────────────────────────
    def _recv_loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(2048)
                self._recv_buffer.extend(data)
                self._parse_buffer()
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"[GCU] 接收错误: {e}")
                break

    def _parse_buffer(self):
        while len(self._recv_buffer) >= 72:
            idx = self._recv_buffer.find(PROTOCOL_HEADER_RECV)
            if idx == -1:
                self._recv_buffer.clear()
                return
            if idx > 0:
                self._recv_buffer = self._recv_buffer[idx:]
            if len(self._recv_buffer) < 4:
                return
            pkt_len = struct.unpack("<H", self._recv_buffer[2:4])[0]
            if len(self._recv_buffer) < pkt_len:
                return
            pkt = bytes(self._recv_buffer[:pkt_len])
            self._recv_buffer = self._recv_buffer[pkt_len:]
            self._parse_packet(pkt)

    def _parse_packet(self, pkt: bytes):
        if len(pkt) < 72:
            return
        self._total_recv += 1
        self._last_recv_time = time.time()

        crc_recv = struct.unpack(">H", pkt[-2:])[0]
        crc_calc = _crc16(pkt[:-2])
        if crc_recv != crc_calc:
            self._crc_errors += 1
            now = time.time()
            if now - self._last_err_report > 5.0:
                rate = self._crc_errors / max(self._total_recv, 1)
                print(f"[GCU] ⚠ CRC错误 累计{self._crc_errors}/{self._total_recv} ({rate:.1%})")
                self._last_err_report = now
            return

        try:
            work_mode = pkt[5]
            pitch     = struct.unpack("<h", pkt[20:22])[0] / 100.0
            yaw       = struct.unpack("<H", pkt[22:24])[0] / 100.0
            roll      = struct.unpack("<h", pkt[18:20])[0] / 100.0
            cam1_zoom = 1.0
            cam2_zoom = 1.0
            if len(pkt) >= 63:
                cam1_zoom = struct.unpack("<H", pkt[59:61])[0] / 10.0
                cam2_zoom = struct.unpack("<H", pkt[61:63])[0] / 10.0
            off_h = struct.unpack("<h", pkt[8:10])[0] / 10.0
            off_v = struct.unpack("<h", pkt[10:12])[0] / 10.0

            status = GimbalStatus(
                mode=work_mode,
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                cam1_zoom=cam1_zoom,
                cam2_zoom=cam2_zoom,
                is_tracking_ok=(work_mode == 0x17),
                target_x=int(off_h),
                target_y=int(off_v)
            )
            with self.status_lock:
                self.latest_status = status
        except Exception:
            pass

    # ── 状态查询 ──────────────────────────────────────────────
    def get_status(self) -> Optional[GimbalStatus]:
        with self.status_lock:
            return self.latest_status

    def is_healthy(self, timeout: float = 2.0) -> bool:
        if not self._last_recv_time:
            return False
        return (time.time() - self._last_recv_time) < timeout

    def get_stats(self) -> dict:
        return {
            "total_recv":    self._total_recv,
            "crc_errors":    self._crc_errors,
            "error_rate":    self._crc_errors / max(self._total_recv, 1),
            "last_recv_ago": time.time() - self._last_recv_time if self._last_recv_time else float("inf"),
            "is_healthy":    self.is_healthy()
        }

# ── GCU 命令接口（兼容旧 GCUCommander 接口）─────────────────
class GCUCommander:
    """保持与 gimbal_controller.py 的接口兼容"""

    def __init__(self, connection: GCUConnection):
        self.conn = connection

    def heartbeat(self) -> Optional[GimbalStatus]:
        self.conn._send_packet(0x00)
        time.sleep(0.05)
        return self.conn.get_status()

    def gimbal_control(self, pitch_speed: int, yaw_speed: int) -> bool:
        self.conn.set_control(pitch=pitch_speed, yaw=yaw_speed, valid=True)
        return self.conn._send_packet(0x00)

    def start_tracking(self, x0: int, y0: int, x1: int, y1: int) -> Optional[bool]:
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            print(f"[GCU] ⚠ 坐标非法: ({x0},{y0})-({x1},{y1})")
            return None
        return self.conn._send_packet(0x17)

    def stop_tracking(self) -> bool:
        self.conn.set_control(0, 0, False)
        return self.conn._send_packet(0x11)

    def pointing_lock(self, pitch_speed: int, yaw_speed: int,
                      auto_zoom_compensate: bool = True) -> bool:
        if auto_zoom_compensate:
            st = self.conn.get_status()
            if st and st.cam1_zoom > 1.0:
                pitch_speed = int(pitch_speed * st.cam1_zoom)
                yaw_speed   = int(yaw_speed   * st.cam1_zoom)
        self.conn.set_control(pitch=pitch_speed, yaw=yaw_speed, valid=True)
        return self.conn._send_packet(0x00)

    def reset_gimbal(self) -> bool:
        self.conn.set_control(0, 0, False)
        return self.conn._send_packet(0x03)

    def zoom_control(self, camera_id: int, zoom_speed: int) -> bool:
        return self.conn._send_packet(0x25)
