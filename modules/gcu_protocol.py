"""
========================================================================
Z-2Mini GCU 私有通信协议 Python 实现
========================================================================
协议版本: V0.2 (文档版本 V2.0.6)
支持 UDP / TCP 通信方式

基于《GCU私有通信协议-XF_A5_V2_0_6》实现
此模块作为底层通信层，由 gimbal_controller.py 中的 GimbalHardwareZ2Mini 调用。
"""

import struct
import socket
import threading
import time
from typing import Optional, Tuple
from dataclasses import dataclass

# ============================================================
# 常量
# ============================================================
HEADER_SEND = bytes([0xA8, 0xE5])
HEADER_RECV = bytes([0x8A, 0x5E])
PROTOCOL_VERSION = 0x02

DEFAULT_GCU_IP = "192.168.144.108"
UDP_SRC_PORT = 2337   # GCU 监听端口 (上位机发往此端口)
UDP_DST_PORT = 2338   # 上位机监听端口 (GCU 回传到此端口)
TCP_PORT = 2332


# ============================================================
# CRC16 (CCITT-FALSE, 与文档 C 代码一致)
# ============================================================
_CRC_TABLE = [
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
    0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
]

def _crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        da = (crc >> 12) & 0x0F
        crc = ((crc << 4) & 0xFFFF) ^ _CRC_TABLE[da ^ (byte >> 4)]
        da = (crc >> 12) & 0x0F
        crc = ((crc << 4) & 0xFFFF) ^ _CRC_TABLE[da ^ (byte & 0x0F)]
    return crc


# ============================================================
# 指令码
# ============================================================
CMD_EMPTY           = 0x00
CMD_CALIBRATE       = 0x01
CMD_HOME            = 0x03
CMD_ANGLE_CTRL      = 0x10
CMD_POINTING_LOCK   = 0x11
CMD_POINTING_FOLLOW = 0x12
CMD_NADIR           = 0x13
CMD_EULER_CTRL      = 0x14
CMD_STARE_COORD     = 0x15
CMD_STARE_TARGET    = 0x16
CMD_TRACK           = 0x17
CMD_POINT_MOVE      = 0x1A
CMD_FPV             = 0x1C
CMD_PHOTO           = 0x20
CMD_RECORD          = 0x21
CMD_ZOOM_IN         = 0x22
CMD_ZOOM_OUT        = 0x23
CMD_ZOOM_STOP       = 0x24
CMD_ZOOM_SET        = 0x25
CMD_TARGET_DETECT   = 0x75
CMD_PIP             = 0x74


# ============================================================
# GCU 返回状态解析结果
# ============================================================
@dataclass
class GCUStatus:
    """GCU 回传数据解析结果"""
    # 主帧
    mode: int = 0                   # 工作模式
    status_flags: int = 0           # 状态标志
    miss_h: int = 0                 # 水平脱靶量 [-1000, 1000]
    miss_v: int = 0                 # 垂直脱靶量 [-1000, 1000]
    rel_roll: float = 0.0           # 相机相对滚转 (deg, 编码器)
    rel_pitch: float = 0.0          # 相机相对俯仰 (deg, 编码器)
    rel_yaw: float = 0.0            # 相机相对偏航 (deg, 编码器)
    abs_roll: float = 0.0           # 绝对滚转 (deg)
    abs_pitch: float = 0.0          # 绝对俯仰 (deg)
    abs_yaw: float = 0.0            # 绝对偏航 (deg)
    gyro_x: float = 0.0             # 角速度 X (deg/s)
    gyro_y: float = 0.0             # 角速度 Y (deg/s)
    gyro_z: float = 0.0             # 角速度 Z (deg/s)

    # 副帧
    target_dist: float = 0.0        # 目标距离 (m)
    target_lon: float = 0.0
    target_lat: float = 0.0
    target_alt: float = 0.0
    cam1_zoom: float = 1.0
    cam2_zoom: float = 1.0
    cam_status: int = 0
    ir_status: int = 0

    # 命令反馈
    cmd_fb: int = 0
    cmd_result: int = -1            # 0=成功, 1=失败, 2=执行中

    @property
    def is_tracking_ok(self) -> bool:
        """吊舱跟踪是否成功 (B0)"""
        return (self.status_flags & 0x01) == 1

    @property
    def is_inverted(self) -> bool:
        """倒装模式 (B12)"""
        return (self.status_flags >> 12 & 0x01) == 1

    @property
    def is_recording(self) -> bool:
        return (self.cam_status >> 4 & 0x01) == 1

    @property
    def mode_name(self) -> str:
        names = {
            0x10: "ANGLE_CTRL", 0x11: "POINT_LOCK", 0x12: "POINT_FOLLOW",
            0x13: "NADIR", 0x14: "EULER_CTRL", 0x16: "STARE",
            0x17: "TRACKING", 0x1C: "FPV",
        }
        return names.get(self.mode, f"0x{self.mode:02X}")


# ============================================================
# 数据包构建
# ============================================================
def _build_main_frame(
    roll_ctrl: int = 0, pitch_ctrl: int = 0, yaw_ctrl: int = 0,
    ctrl_valid: bool = False, imu_valid: bool = False,
    v_roll: float = 0, v_pitch: float = 0, v_yaw: float = 0,
    sub_req: int = 0x01,
) -> bytes:
    """构建上位机→GCU 主帧 (32 字节)"""
    status = 0
    if ctrl_valid: status |= 0x04
    if imu_valid:  status |= 0x01

    data = struct.pack('<hhh', roll_ctrl, pitch_ctrl, yaw_ctrl)
    data += struct.pack('<B', status)
    data += struct.pack('<hh', int(v_roll * 100), int(v_pitch * 100))
    data += struct.pack('<H', int(v_yaw * 100) & 0xFFFF)
    data += b'\x00' * 12   # 加速度 + 速度 (未接飞控时全0)
    data += struct.pack('<B', sub_req)
    data += b'\x00' * 6
    return data


def _build_sub_frame(lon=0.0, lat=0.0, alt=0.0, sat=0, rel_alt=0.0) -> bytes:
    """构建上位机→GCU 副帧 (32 字节)"""
    data = struct.pack('<B', 0x01)
    data += struct.pack('<i', int(lon * 1e7))
    data += struct.pack('<i', int(lat * 1e7))
    data += struct.pack('<i', int(alt * 1000))
    data += struct.pack('<B', sat)
    data += struct.pack('<I', 0)   # GNSS time ms
    data += struct.pack('<h', 0)   # GNSS week
    data += struct.pack('<i', int(rel_alt * 1000))
    data += b'\x00' * 8
    return data


def _build_packet(main: bytes, sub: bytes, cmd: int, params: bytes = b'') -> bytes:
    """组装完整数据包"""
    pkt_len = 72 + len(params)
    data = HEADER_SEND
    data += struct.pack('<H', pkt_len)
    data += struct.pack('<B', PROTOCOL_VERSION)
    data += main    # 5~36
    data += sub     # 37~68
    data += struct.pack('<B', cmd)
    data += params
    crc = _crc16(data)
    data += struct.pack('>H', crc)   # CRC 大端序
    return data


def _parse_response(data: bytes) -> Optional[GCUStatus]:
    """解析 GCU→上位机 数据包"""
    if len(data) < 72:
        return None
    if data[0:2] != HEADER_RECV:
        return None

    pkt_len = struct.unpack('<H', data[2:4])[0]
    if len(data) < pkt_len:
        return None

    # CRC 校验
    crc_recv = struct.unpack('>H', data[pkt_len-2:pkt_len])[0]
    crc_calc = _crc16(data[:pkt_len-2])
    if crc_recv != crc_calc:
        return None

    s = GCUStatus()

    # 主帧 (5~36)
    s.mode = data[5]
    s.status_flags = struct.unpack('<H', data[6:8])[0]
    s.miss_h = struct.unpack('<h', data[8:10])[0]
    s.miss_v = struct.unpack('<h', data[10:12])[0]
    s.rel_roll  = struct.unpack('<h', data[12:14])[0] / 100.0
    s.rel_pitch = struct.unpack('<h', data[14:16])[0] / 100.0
    s.rel_yaw   = struct.unpack('<h', data[16:18])[0] / 100.0
    s.abs_roll  = struct.unpack('<h', data[18:20])[0] / 100.0
    s.abs_pitch = struct.unpack('<h', data[20:22])[0] / 100.0
    s.abs_yaw   = struct.unpack('<H', data[22:24])[0] / 100.0
    s.gyro_x    = struct.unpack('<h', data[24:26])[0] / 10.0
    s.gyro_y    = struct.unpack('<h', data[26:28])[0] / 10.0
    s.gyro_z    = struct.unpack('<h', data[28:30])[0] / 10.0

    # 副帧 (37~68)
    if data[37] == 0x01:
        s.target_dist = struct.unpack('<i', data[43:47])[0] / 10.0
        s.target_lon  = struct.unpack('<i', data[47:51])[0] / 1e7
        s.target_lat  = struct.unpack('<i', data[51:55])[0] / 1e7
        s.target_alt  = struct.unpack('<i', data[55:59])[0] / 1000.0
        s.cam1_zoom   = struct.unpack('<H', data[59:61])[0] / 10.0
        s.cam2_zoom   = struct.unpack('<H', data[61:63])[0] / 10.0
        s.ir_status   = data[63]
        s.cam_status  = struct.unpack('<H', data[64:66])[0]

    # 命令反馈 (69+)
    s.cmd_fb = data[69]
    if pkt_len > 72:
        s.cmd_result = data[70]

    return s


# ============================================================
# GCU 通信管理器
# ============================================================
class GCUConnection:
    """
    GCU 底层通信连接

    支持 UDP / TCP 两种方式。
    发送数据包并接收回传，线程安全。
    """

    def __init__(self, gcu_ip: str = DEFAULT_GCU_IP, mode: str = "udp"):
        self.gcu_ip = gcu_ip
        self.mode = mode
        self.sock = None
        self._lock = threading.Lock()
        self._connected = False

    def connect(self) -> bool:
        try:
            if self.mode == "udp":
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind(('', UDP_DST_PORT))
                self.sock.settimeout(0.5)
                print(f"[GCU] UDP 就绪 (GCU={self.gcu_ip}, 监听端口={UDP_DST_PORT})")
            elif self.mode == "tcp":
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(5.0)
                self.sock.connect((self.gcu_ip, TCP_PORT))
                self.sock.settimeout(0.5)
                print(f"[GCU] TCP 已连接 {self.gcu_ip}:{TCP_PORT}")
            self._connected = True
            return True
        except Exception as e:
            print(f"[GCU] 连接失败: {e}")
            return False

    def disconnect(self):
        self._connected = False
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_and_recv(self, packet: bytes) -> Optional[GCUStatus]:
        """发送数据包并接收解析回传"""
        if not self._connected or not self.sock:
            return None
        with self._lock:
            try:
                if self.mode == "udp":
                    self.sock.sendto(packet, (self.gcu_ip, UDP_SRC_PORT))
                    data, _ = self.sock.recvfrom(1024)
                else:
                    self.sock.sendall(packet)
                    data = self.sock.recv(1024)
                return _parse_response(data)
            except socket.timeout:
                return None
            except Exception as e:
                print(f"[GCU] 通信错误: {e}")
                return None

    @property
    def connected(self) -> bool:
        return self._connected


# ============================================================
# 高级命令接口 (供 GimbalHardwareZ2Mini 调用)
# ============================================================
class GCUCommander:
    """
    GCU 指令发送器

    封装常用命令的参数构建, 返回 GCUStatus。
    """

    def __init__(self, conn: GCUConnection):
        self.conn = conn

    def _send(self, cmd: int, params: bytes = b'',
              pitch_ctrl: int = 0, yaw_ctrl: int = 0, roll_ctrl: int = 0,
              ctrl_valid: bool = False) -> Optional[GCUStatus]:
        main = _build_main_frame(
            roll_ctrl=roll_ctrl, pitch_ctrl=pitch_ctrl, yaw_ctrl=yaw_ctrl,
            ctrl_valid=ctrl_valid, sub_req=0x01,
        )
        sub = _build_sub_frame()
        pkt = _build_packet(main, sub, cmd, params)
        return self.conn.send_and_recv(pkt)

    def heartbeat(self) -> Optional[GCUStatus]:
        """空命令心跳"""
        return self._send(CMD_EMPTY)

    def home(self) -> Optional[GCUStatus]:
        """回中"""
        return self._send(CMD_HOME)

    # ---- 跟踪 (0x17) ----

    def start_tracking(self, x0: int, y0: int, x1: int, y1: int) -> Optional[GCUStatus]:
        """
        启动吊舱内置跟踪

        坐标系: 图像左上角原点, [0, 10000]
        """
        params = struct.pack('<B', 0x01)          # TT=0x01 进入跟踪
        params += struct.pack('<HHHH', x0, y0, x1, y1)
        return self._send(CMD_TRACK, params)

    def stop_tracking(self) -> Optional[GCUStatus]:
        """停止跟踪"""
        params = struct.pack('<B', 0x00)
        params += struct.pack('<HHHH', 0, 0, 0, 0)
        return self._send(CMD_TRACK, params)

    # ---- 指向锁定 + 角速度控制 (0x11) ----

    def pointing_lock(self, pitch_speed: int = 0, yaw_speed: int = 0,
                      ctrl_valid: bool = True) -> Optional[GCUStatus]:
        """
        指向锁定模式 - 发送角速度控制

        speed 单位: 0.1°/s (除以当前变焦倍率)
        """
        return self._send(CMD_POINTING_LOCK,
                          pitch_ctrl=pitch_speed, yaw_ctrl=yaw_speed,
                          ctrl_valid=ctrl_valid)

    # ---- 指点平移 (0x1A) ----

    def point_move(self, x: int, y: int) -> Optional[GCUStatus]:
        """指点平移, 坐标 [0, 10000]"""
        params = struct.pack('<B', 0x01)
        params += struct.pack('<HH', x, y)
        return self._send(CMD_POINT_MOVE, params)

    # ---- 相机功能 ----

    def take_photo(self) -> Optional[GCUStatus]:
        return self._send(CMD_PHOTO, bytes([0x01]))

    def toggle_record(self) -> Optional[GCUStatus]:
        return self._send(CMD_RECORD, bytes([0x01]))

    def zoom_set(self, cam_id: int = 0x01, value: int = 1) -> Optional[GCUStatus]:
        params = struct.pack('<Bh', cam_id, value)
        return self._send(CMD_ZOOM_SET, params)

    def zoom_in(self, cam_id: int = 0x01) -> Optional[GCUStatus]:
        return self._send(CMD_ZOOM_IN, bytes([cam_id]))

    def zoom_out(self, cam_id: int = 0x01) -> Optional[GCUStatus]:
        return self._send(CMD_ZOOM_OUT, bytes([cam_id]))

    def zoom_stop(self, cam_id: int = 0x01) -> Optional[GCUStatus]:
        return self._send(CMD_ZOOM_STOP, bytes([cam_id]))

    def set_pip(self, mode: int = 0) -> Optional[GCUStatus]:
        return self._send(CMD_PIP, bytes([mode]))
