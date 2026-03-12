"""
多目标跟踪模块 (优化版 v2.0)
适配 TrackerConfig
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
from typing import Optional, List, Tuple, Dict
from collections import deque
from scipy.optimize import linear_sum_assignment

from config import TrackerConfig

# ── 卡尔曼跟踪器 ──────────────────────────────────────────────
class KalmanTracker:
    """
    状态向量: [cx, cy, w, h, vx, vy, vw, vh, ax, ay]
    新增: 面积稳定性检测 / 遮挡标记 / 匀加速多步预测
    """
    _id_counter = 0

    def __init__(self, bbox: np.ndarray, cfg: Optional[TrackerConfig] = None):
        KalmanTracker._id_counter += 1
        self.id  = KalmanTracker._id_counter
        self.cfg = cfg or TrackerConfig()

        self.state_dim = 10
        self.obs_dim   = 4
        self._init_kalman()

        cx, cy, w, h = self._bbox_to_xywh(bbox)
        self.x = np.zeros((self.state_dim, 1))
        self.x[0], self.x[1], self.x[2], self.x[3] = cx, cy, w, h

        self.age               = 0
        self.time_since_update = 0
        self.hit_streak        = 0
        self.hits              = 0

        # 遮挡检测 (新增)
        self.is_occluded    = False
        self.occlusion_count = 0
        self._area_history  = deque(maxlen=10)
        self._area_history.append(w * h)

        self.feature: Optional[np.ndarray] = None
        self.trajectory = deque(maxlen=50)
        self.trajectory.append((cx, cy))
        self.last_update_time = time.time()

    def _init_kalman(self):
        n, m, dt = self.state_dim, self.obs_dim, self.cfg.dt

        self.F = np.eye(n)
        self.F[0, 4] = dt;  self.F[1, 5] = dt
        self.F[2, 6] = dt;  self.F[3, 7] = dt
        self.F[0, 8] = 0.5 * dt * dt
        self.F[1, 9] = 0.5 * dt * dt
        self.F[4, 8] = dt;  self.F[5, 9] = dt

        self.H = np.zeros((m, n))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1

        q = self.cfg.process_noise_std ** 2
        r = self.cfg.measurement_noise_std ** 2
        self.Q = np.eye(n) * q
        self.Q[4:8, 4:8] *= 10
        self.Q[8:10, 8:10] *= 5
        self.R = np.eye(m) * r
        self.P = np.eye(n) * 10.0
        self.P[4:, 4:] *= 100

    @staticmethod
    def _bbox_to_xywh(bbox: np.ndarray) -> Tuple[float, float, float, float]:
        x0, y0, x1, y1 = bbox
        return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0

    def _state_to_bbox(self) -> np.ndarray:
        cx, cy = self.x[0, 0], self.x[1, 0]
        w,  h  = max(self.x[2, 0], 1), max(self.x[3, 0], 1)
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox()

    def update(self, bbox: np.ndarray, feature: Optional[np.ndarray] = None):
        cx, cy, w, h = self._bbox_to_xywh(bbox)
        z = np.array([[cx], [cy], [w], [h]])

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

        self._area_history.append(w * h)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_update_time = time.time()

        if feature is not None:
            self.feature = feature if self.feature is None \
                else 0.9 * self.feature + 0.1 * feature

        self.trajectory.append((self.x[0, 0], self.x[1, 0]))

        if self.is_occluded:
            self.is_occluded    = False
            self.occlusion_count = 0

    def mark_occluded(self):
        self.is_occluded = True
        self.occlusion_count += 1
        self.hit_streak = 0

    def is_area_stable(self, threshold: float = 0.5) -> bool:
        if len(self._area_history) < 3:
            return True
        areas      = list(self._area_history)
        mean_area  = np.mean(areas[:-1])
        if mean_area < 1:
            return True
        return abs(areas[-1] - mean_area) / mean_area < threshold

    def predict_future(self, steps: int = 5) -> List[Tuple[float, float]]:
        cx = self.x[0, 0]; cy = self.x[1, 0]
        vx = self.x[4, 0]; vy = self.x[5, 0]
        ax = self.x[8, 0]; ay = self.x[9, 0]
        return [
            (cx + vx * t + 0.5 * ax * t * t,
             cy + vy * t + 0.5 * ay * t * t)
            for t in range(1, steps + 1)
        ]

    @property
    def current_bbox(self)   -> np.ndarray:          return self._state_to_bbox()
    @property
    def current_center(self) -> Tuple[float, float]: return (self.x[0,0], self.x[1,0])
    @property
    def current_velocity(self)-> Tuple[float, float]:return (self.x[4,0], self.x[5,0])
    @property
    def current_area(self)   -> float:
        return max(self.x[2,0], 1) * max(self.x[3,0], 1)
    @property
    def current_speed(self) -> float:
        """计算标量速度 (像素/帧)"""
        vx, vy = self.current_velocity
        return float(np.sqrt(vx ** 2 + vy ** 2))

# ── 多目标跟踪管理器 ──────────────────────────────────────────
class MultiObjectTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.trackers: List[KalmanTracker] = []
        self.primary_target_id: Optional[int] = None

        # 主目标超时 (新增)
        self._primary_missing_frames  = 0
        self._primary_timeout         = getattr(cfg, 'prediction_only_frames', 15) * 6

        self._frame_count      = 0
        self._total_detections = 0

    def update(self, detections: list) -> List[KalmanTracker]:
        self._frame_count      += 1
        self._total_detections += len(detections)

        for t in self.trackers:
            t.predict()

        matched, unmatched_dets, unmatched_trks = self._associate(detections)

        for d_idx, t_idx in matched:
            det = detections[d_idx]
            self.trackers[t_idx].update(
                det.bbox,
                feature=getattr(det, 'feature', None)
            )

        # 遮挡标记 (新增)
        for t_idx in unmatched_trks:
            if not self.trackers[t_idx].is_area_stable():
                self.trackers[t_idx].mark_occluded()

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            tk  = KalmanTracker(det.bbox, self.cfg)
            if getattr(det, 'feature', None) is not None:
                tk.feature = det.feature
            self.trackers.append(tk)

        self.trackers = [t for t in self.trackers
                         if t.time_since_update <= self.cfg.max_age]

        self._check_primary_timeout()

        return [t for t in self.trackers
                if t.hit_streak >= self.cfg.min_hits or t.hits >= self.cfg.min_hits]

    def _check_primary_timeout(self):
        if self.primary_target_id is None:
            self._primary_missing_frames = 0
            return

        primary = next((t for t in self.trackers
                        if t.id == self.primary_target_id), None)

        if primary is None or primary.time_since_update > 0:
            self._primary_missing_frames += 1
        else:
            self._primary_missing_frames = 0

        if self._primary_missing_frames >= self._primary_timeout:
            print(f"[Tracker] 主目标 {self.primary_target_id} "
                  f"超时 ({self._primary_timeout}帧)，已清除")
            self.primary_target_id       = None
            self._primary_missing_frames = 0

    def _associate(self, detections: list) -> Tuple[List, List, List]:
        if not self.trackers:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.trackers)))

        iou_mat = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_mat[d, t] = self._iou(det.bbox, trk.current_bbox)

        if self.cfg.feature_weight > 0:
            feat_mat = np.zeros_like(iou_mat)
            for d, det in enumerate(detections):
                for t, trk in enumerate(self.trackers):
                    df = getattr(det, 'feature', None)
                    if df is not None and trk.feature is not None:
                        sim = np.dot(df, trk.feature) / (
                            np.linalg.norm(df) * np.linalg.norm(trk.feature) + 1e-6)
                        feat_mat[d, t] = max(0.0, sim)
            cost = (1 - self.cfg.feature_weight) * iou_mat + \
                        self.cfg.feature_weight  * feat_mat
        else:
            cost = iou_mat

        row_ind, col_ind = linear_sum_assignment(-cost)

        matched, unmatched_dets, unmatched_trks = \
            [], list(range(len(detections))), list(range(len(self.trackers)))

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= self.cfg.iou_threshold:
                matched.append((r, c))
                unmatched_dets.remove(r)
                unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks

    @staticmethod
    def _iou(b1: np.ndarray, b2: np.ndarray) -> float:
        x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
        x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
        inter = max(0, x1-x0) * max(0, y1-y0)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / (union + 1e-6)

    def set_primary_target(self, target_id: int):
        self.primary_target_id       = target_id
        self._primary_missing_frames = 0
        print(f"[Tracker] 设置主目标 ID: {target_id}")

    def get_primary_target(self) -> Optional[KalmanTracker]:
        if self.primary_target_id is None:
            return None
        return next((t for t in self.trackers
                     if t.id == self.primary_target_id), None)

    def get_active_trackers(self) -> List[KalmanTracker]:
        """获取当前活跃的跟踪目标列表"""
        return [t for t in self.trackers
                if t.hit_streak >= self.cfg.min_hits or t.hits >= self.cfg.min_hits]

    def auto_select_primary(self, frame_center: Tuple[float, float]) \
            -> Optional[KalmanTracker]:
        active = [t for t in self.trackers if t.hit_streak >= self.cfg.min_hits]
        if not active:
            return None
        cx, cy = frame_center
        best = min(active, key=lambda t:
                   (t.current_center[0]-cx)**2 + (t.current_center[1]-cy)**2)
        self.set_primary_target(best.id)
        return best

    def get_stats(self) -> Dict:
        return {
            'frame_count':          self._frame_count,
            'active_tracks':        len([t for t in self.trackers
                                         if t.hit_streak >= self.cfg.min_hits]),
            'total_tracks':         len(self.trackers),
            'primary_target_id':    self.primary_target_id,
            'primary_missing_frames': self._primary_missing_frames,
            'avg_detections':       self._total_detections / max(self._frame_count, 1)
        }
