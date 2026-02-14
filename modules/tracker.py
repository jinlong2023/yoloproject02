"""
========================================================================
模块2: 目标跟踪模块 (卡尔曼滤波 + CNN特征融合)
Module 2: Target Tracking (Kalman Filter + CNN Feature Fusion)
========================================================================
功能:
- 卡尔曼滤波状态估计与预测
- CNN外观特征匹配 (抗遮挡、重识别)
- 光流辅助运动估计
- 多目标关联与管理
- 遮挡检测与预测补偿
- 运动轨迹记录与分析
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from scipy.optimize import linear_sum_assignment
from collections import deque
import time


# ====================================================================
# 卡尔曼滤波器
# ====================================================================
class KalmanTracker:
    """
    目标状态卡尔曼滤波器
    状态向量: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): 边界框中心坐标
    - (w, h): 边界框宽度和高度
    - (vx, vy, vw, vh): 对应速度分量
    """

    _count = 0  # 全局ID计数器

    def __init__(self, bbox: np.ndarray, feature: Optional[np.ndarray] = None,
                 process_noise: float = 1.0, measurement_noise: float = 0.5):
        """
        Args:
            bbox: [x1, y1, x2, y2] 初始边界框
            feature: CNN外观特征向量
            process_noise: 过程噪声标准差
            measurement_noise: 测量噪声标准差
        """
        # 分配唯一ID
        KalmanTracker._count += 1
        self.id = KalmanTracker._count

        # 状态维度
        self.state_dim = 8    # [x, y, w, h, vx, vy, vw, vh]
        self.measure_dim = 4  # [x, y, w, h]

        # 初始化卡尔曼滤波器
        self.kf = cv2.KalmanFilter(self.state_dim, self.measure_dim, 0)

        # 状态转移矩阵 F (匀速运动模型)
        self.kf.transitionMatrix = np.eye(self.state_dim, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0  # x += vx * dt

        # 观测矩阵 H
        self.kf.measurementMatrix = np.zeros(
            (self.measure_dim, self.state_dim), dtype=np.float32
        )
        for i in range(self.measure_dim):
            self.kf.measurementMatrix[i, i] = 1.0

        # 过程噪声协方差 Q
        q = process_noise ** 2
        self.kf.processNoiseCov = np.eye(self.state_dim, dtype=np.float32) * q
        self.kf.processNoiseCov[4:, 4:] *= 2.0  # 速度项噪声更大

        # 测量噪声协方差 R
        r = measurement_noise ** 2
        self.kf.measurementNoiseCov = np.eye(self.measure_dim, dtype=np.float32) * r

        # 后验误差协方差 P
        self.kf.errorCovPost = np.eye(self.state_dim, dtype=np.float32) * 10.0
        self.kf.errorCovPost[4:, 4:] *= 100.0  # 初始速度不确定性大

        # 初始状态
        z = self._bbox_to_measurement(bbox)
        self.kf.statePost[:4, 0] = z[:, 0]
        self.kf.statePost[4:, 0] = 0.0  # 初始速度为0

        # 跟踪管理
        self.hits = 1           # 总命中次数
        self.hit_streak = 1     # 连续命中次数
        self.age = 0            # 生存帧数
        self.time_since_update = 0  # 上次更新后的帧数
        self.is_confirmed = False   # 是否已确认

        # 外观特征队列 (保存最近N帧特征用于匹配)
        self.feature_queue = deque(maxlen=30)
        if feature is not None:
            self.feature_queue.append(feature)

        # 运动轨迹记录
        self.trajectory = deque(maxlen=300)
        self.velocity_history = deque(maxlen=60)
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        self.trajectory.append((cx, cy))

        # 状态标记
        self.is_occluded = False
        self.occlusion_count = 0

    def _bbox_to_measurement(self, bbox: np.ndarray) -> np.ndarray:
        """bbox [x1,y1,x2,y2] -> measurement [cx, cy, w, h]"""
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return np.array([[cx], [cy], [w], [h]], dtype=np.float32)

    def _state_to_bbox(self, state: np.ndarray) -> np.ndarray:
        """state [cx, cy, w, h, ...] -> bbox [x1, y1, x2, y2]"""
        cx, cy, w, h = state[0], state[1], max(state[2], 1), max(state[3], 1)
        return np.array([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2
        ], dtype=np.float32)

    def predict(self) -> np.ndarray:
        """
        卡尔曼预测步骤
        Returns:
            预测的边界框 [x1, y1, x2, y2]
        """
        # 防止宽高为负
        if self.kf.statePost[6, 0] + self.kf.statePost[2, 0] <= 0:
            self.kf.statePost[6, 0] = 0.0
        if self.kf.statePost[7, 0] + self.kf.statePost[3, 0] <= 0:
            self.kf.statePost[7, 0] = 0.0

        predicted_state = self.kf.predict()
        self.age += 1
        self.time_since_update += 1

        if self.time_since_update > 1:
            self.hit_streak = 0

        return self._state_to_bbox(predicted_state[:, 0])

    def update(self, bbox: np.ndarray, feature: Optional[np.ndarray] = None):
        """
        卡尔曼更新步骤

        Args:
            bbox: 观测边界框 [x1, y1, x2, y2]
            feature: CNN外观特征
        """
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0
        self.is_occluded = False
        self.occlusion_count = 0

        # 确认跟踪
        if self.hit_streak >= 3:
            self.is_confirmed = True

        # 卡尔曼更新
        z = self._bbox_to_measurement(bbox)
        self.kf.correct(z)

        # 更新外观特征
        if feature is not None:
            self.feature_queue.append(feature)

        # 记录轨迹
        state = self.kf.statePost[:, 0]
        self.trajectory.append((state[0], state[1]))

        # 记录速度
        vx, vy = state[4], state[5]
        speed = np.sqrt(vx ** 2 + vy ** 2)
        self.velocity_history.append({
            'vx': float(vx), 'vy': float(vy), 'speed': float(speed)
        })

    def mark_occluded(self):
        """标记为遮挡状态"""
        self.is_occluded = True
        self.occlusion_count += 1

    @property
    def current_bbox(self) -> np.ndarray:
        """当前估计的边界框"""
        return self._state_to_bbox(self.kf.statePost[:, 0])

    @property
    def current_center(self) -> Tuple[float, float]:
        """当前中心坐标"""
        state = self.kf.statePost[:, 0]
        return (float(state[0]), float(state[1]))

    @property
    def current_velocity(self) -> Tuple[float, float]:
        """当前速度 (vx, vy)"""
        state = self.kf.statePost[:, 0]
        return (float(state[4]), float(state[5]))

    @property
    def current_speed(self) -> float:
        """当前速率"""
        vx, vy = self.current_velocity
        return np.sqrt(vx ** 2 + vy ** 2)

    @property
    def mean_feature(self) -> Optional[np.ndarray]:
        """外观特征均值"""
        if not self.feature_queue:
            return None
        features = np.array(list(self.feature_queue))
        return np.mean(features, axis=0)

    def predict_future(self, steps: int = 5) -> List[Tuple[float, float]]:
        """
        预测未来位置序列

        Args:
            steps: 预测步数

        Returns:
            未来位置列表 [(x, y), ...]
        """
        state = self.kf.statePost[:, 0].copy()
        F = self.kf.transitionMatrix.copy()
        positions = []

        for _ in range(steps):
            state = F @ state
            positions.append((float(state[0]), float(state[1])))

        return positions


# ====================================================================
# 光流运动估计器
# ====================================================================
class OpticalFlowEstimator:
    """
    稀疏光流运动估计
    用于辅助目标状态估计和遮挡检测
    """

    def __init__(self, win_size: int = 15, max_level: int = 3):
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7
        )
        self.prev_gray = None
        self.prev_points = None

    def compute(self, frame: np.ndarray, roi: Optional[np.ndarray] = None
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        计算光流

        Args:
            frame: 当前帧 (BGR)
            roi: 感兴趣区域 [x1, y1, x2, y2]

        Returns:
            (good_new, good_old) 匹配的点对, 或 None
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            mask = np.zeros_like(gray)
            if roi is not None:
                x1, y1, x2, y2 = roi.astype(int)
                mask[y1:y2, x1:x2] = 255
            else:
                mask[:] = 255
            self.prev_points = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            return None

        if self.prev_points is None or len(self.prev_points) < 3:
            self.prev_gray = gray
            self.prev_points = cv2.goodFeaturesToTrack(gray, **self.feature_params)
            return None

        # 前向光流
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None, **self.lk_params
        )

        if next_pts is None:
            self.prev_gray = gray
            self.prev_points = None
            return None

        # 后向光流验证 (前后向一致性检查)
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, next_pts, None, **self.lk_params
        )

        # 一致性检查
        if back_pts is not None:
            dist = np.linalg.norm(self.prev_points - back_pts, axis=2).reshape(-1)
            consistent = dist < 1.0
            status = status.reshape(-1) & consistent.astype(np.uint8)
        else:
            status = status.reshape(-1)

        good_new = next_pts[status == 1].reshape(-1, 2)
        good_old = self.prev_points[status == 1].reshape(-1, 2)

        self.prev_gray = gray
        self.prev_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

        if len(good_new) < 3:
            return None

        return good_new, good_old

    def estimate_motion(self, good_new: np.ndarray, good_old: np.ndarray
                        ) -> Tuple[float, float]:
        """估计全局运动 (dx, dy)"""
        if len(good_new) < 3:
            return 0.0, 0.0
        # 确保点数组为 (N, 2) 形状
        good_new = good_new.reshape(-1, 2)
        good_old = good_old.reshape(-1, 2)
        displacement = good_new - good_old
        dx = float(np.median(displacement[:, 0]))
        dy = float(np.median(displacement[:, 1]))
        return dx, dy

    def reset(self):
        self.prev_gray = None
        self.prev_points = None


# ====================================================================
# IOU计算工具
# ====================================================================
def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """计算两个bbox的IOU"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_iou_matrix(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """计算IOU矩阵"""
    n = len(bboxes1)
    m = len(bboxes2)
    iou_matrix = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(bboxes1[i], bboxes2[j])

    return iou_matrix


def compute_cosine_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
    """计算余弦距离"""
    if feat1 is None or feat2 is None:
        return 1.0
    sim = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6)
    return 1.0 - sim


# ====================================================================
# 多目标跟踪器主类
# ====================================================================
class MultiObjectTracker:
    """
    多目标跟踪器
    融合卡尔曼滤波 + CNN外观特征 + 光流运动估计
    """

    def __init__(self, config):
        """
        Args:
            config: TrackerConfig 配置对象
        """
        self.config = config

        # 活跃跟踪器列表
        self.trackers: List[KalmanTracker] = []

        # 光流估计器
        self.flow_estimator = OpticalFlowEstimator(
            win_size=config.optical_flow_winsize,
            max_level=config.optical_flow_maxlevel
        )

        # 主目标ID (单目标锁定模式)
        self.primary_target_id: Optional[int] = None

        # 统计信息
        self.frame_count = 0
        self.track_history = []

    def update(self, detections, frame: Optional[np.ndarray] = None
               ) -> List[KalmanTracker]:
        """
        跟踪器主更新函数

        Args:
            detections: 检测结果列表 (Detection对象)
            frame: 当前帧 (用于光流计算)

        Returns:
            活跃的跟踪器列表
        """
        self.frame_count += 1

        # Step 1: 光流辅助运动估计
        global_motion = (0.0, 0.0)
        if frame is not None:
            flow_result = self.flow_estimator.compute(frame)
            if flow_result is not None:
                global_motion = self.flow_estimator.estimate_motion(*flow_result)

        # Step 2: 所有现有跟踪器执行预测
        predicted_bboxes = []
        for tracker in self.trackers:
            pred_bbox = tracker.predict()
            predicted_bboxes.append(pred_bbox)

        # Step 3: 数据关联 (匈牙利算法)
        if len(detections) > 0 and len(self.trackers) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate(
                detections, predicted_bboxes
            )
        elif len(detections) > 0:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = []
        else:
            matched = []
            unmatched_dets = []
            unmatched_trks = list(range(len(self.trackers)))

        # Step 4: 更新匹配的跟踪器
        for det_idx, trk_idx in matched:
            det = detections[det_idx]
            self.trackers[trk_idx].update(det.bbox, det.feature)

        # Step 5: 处理未匹配的跟踪器 (可能遮挡)
        for trk_idx in unmatched_trks:
            self.trackers[trk_idx].mark_occluded()

        # Step 6: 为未匹配的检测创建新跟踪器
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_tracker = KalmanTracker(
                bbox=det.bbox,
                feature=det.feature,
                process_noise=self.config.process_noise_std,
                measurement_noise=self.config.measurement_noise_std
            )
            self.trackers.append(new_tracker)

        # Step 7: 删除过期跟踪器
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.config.max_age
        ]

        # 记录跟踪历史
        active = self.get_active_trackers()
        self.track_history.append({
            'frame_id': self.frame_count,
            'num_active': len(active),
            'num_total': len(self.trackers),
            'matched': len(matched),
            'new_tracks': len(unmatched_dets),
            'lost_tracks': len(unmatched_trks)
        })

        return active

    def _associate(self, detections, predicted_bboxes
                   ) -> Tuple[List, List, List]:
        """
        数据关联 (融合IOU + 外观特征)

        Returns:
            (matched_pairs, unmatched_dets, unmatched_trks)
        """
        num_dets = len(detections)
        num_trks = len(self.trackers)

        # 计算IOU代价矩阵
        det_bboxes = np.array([d.bbox for d in detections])
        pred_bboxes = np.array(predicted_bboxes)
        iou_matrix = compute_iou_matrix(det_bboxes, pred_bboxes)

        # 计算外观特征代价矩阵
        feature_cost = np.ones((num_dets, num_trks))
        for i, det in enumerate(detections):
            for j, trk in enumerate(self.trackers):
                if det.feature is not None and trk.mean_feature is not None:
                    feature_cost[i, j] = compute_cosine_distance(
                        det.feature, trk.mean_feature
                    )

        # 融合代价 (IOU越大越好 -> 1-IOU作为代价; 特征距离越小越好)
        fw = self.config.feature_weight
        cost_matrix = fw * feature_cost + (1 - fw) * (1.0 - iou_matrix)

        # 门限过滤
        cost_matrix[iou_matrix < self.config.iou_threshold] = 1e5

        # 匈牙利算法
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = np.array([]), np.array([])

        matched = []
        unmatched_dets = list(range(num_dets))
        unmatched_trks = list(range(num_trks))

        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < 1e4:
                matched.append((r, c))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks

    def get_active_trackers(self) -> List[KalmanTracker]:
        """获取已确认且活跃的跟踪器"""
        return [
            t for t in self.trackers
            if t.is_confirmed and t.time_since_update <= 1
        ]

    def get_primary_target(self) -> Optional[KalmanTracker]:
        """获取主目标 (用于云台跟踪)"""
        active = self.get_active_trackers()
        if not active:
            # 尝试从预测中恢复主目标
            if self.primary_target_id is not None:
                for t in self.trackers:
                    if (t.id == self.primary_target_id and
                            t.time_since_update <= self.config.prediction_only_frames):
                        return t
            return None

        # 如果有指定主目标
        if self.primary_target_id is not None:
            for t in active:
                if t.id == self.primary_target_id:
                    return t

        # 否则选择最大/最近/最高置信度的目标
        return max(active, key=lambda t: t.hits)

    def set_primary_target(self, target_id: int):
        """设置主跟踪目标"""
        self.primary_target_id = target_id

    def auto_select_primary(self, frame_center: Tuple[float, float]):
        """自动选择距画面中心最近的目标为主目标"""
        active = self.get_active_trackers()
        if not active:
            return

        cx, cy = frame_center
        closest = min(active, key=lambda t: (
            (t.current_center[0] - cx) ** 2 + (t.current_center[1] - cy) ** 2
        ))
        self.primary_target_id = closest.id

    def get_stats(self) -> Dict:
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.get_active_trackers()),
            'total_tracks': len(self.trackers),
            'primary_target_id': self.primary_target_id,
        }
