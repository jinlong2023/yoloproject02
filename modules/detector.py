"""
========================================================================
模块1: 目标检测模块 (YOLOv13)
========================================================================
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# ============================================================
# 检查 YOLOv13 是否正确安装
# ============================================================
YOLO_AVAILABLE = False
YOLO_CLASS = None

try:
    from ultralytics import YOLO
    YOLO_CLASS = YOLO
    YOLO_AVAILABLE = True
    print("[Detector] ultralytics 导入成功")
except ImportError:
    print("=" * 65)
    print("[致命错误] ultralytics 未安装!")
    print("")
    print("  YOLOv13 安装步骤:")
    print("    git clone https://github.com/iMoonLab/yolov13.git")
    print("    cd yolov13")
    print("    pip install -r requirements.txt")
    print("    pip install -e .")
    print("")
    print("  然后下载权重文件 yolov13n.pt 放到项目根目录")
    print("=" * 65)


@dataclass
class Detection:
    """检测结果"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    feature: Optional[np.ndarray] = None
    center: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        if self.center is None:
            self.center = (
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2
            )

    @property
    def width(self): return self.bbox[2] - self.bbox[0]
    @property
    def height(self): return self.bbox[3] - self.bbox[1]
    @property
    def area(self): return self.width * self.height


class FeatureExtractorCNN(nn.Module):
    """轻量级CNN特征提取器 (128维外观特征, 用于跟踪关联)"""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(256, feature_dim), nn.BatchNorm1d(feature_dim))

    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        f = self.fc(f)
        return F.normalize(f, p=2, dim=1)


class IlluminationAdapter:
    """光照自适应 CLAHE 增强"""
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.brightness_history = []

    def process(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        self.brightness_history.append(float(np.mean(l)))
        if len(self.brightness_history) > 30:
            self.brightness_history.pop(0)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @property
    def current_brightness(self):
        return self.brightness_history[-1] if self.brightness_history else 128.0

    @property
    def brightness_stable(self):
        if len(self.brightness_history) < 5: return True
        return np.std(self.brightness_history[-5:]) < 20.0


class TargetDetector:
    """
    目标检测器 - 基于 YOLOv13
    需要先从 https://github.com/iMoonLab/yolov13 安装
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() and "cuda" in config.device else "cpu"
        )

        # ---- YOLO模型 ----
        self.yolo_model = None
        self.use_yolo = False
        self.class_names = {}
        self._init_yolo()

        # ---- CNN特征提取 ----
        self.feature_extractor = FeatureExtractorCNN(128).to(self.device)
        self.feature_extractor.eval()

        # ---- 光照自适应 ----
        self.illumination = IlluminationAdapter()

        # ---- 统计 ----
        self.frame_count = 0
        self.total_time = 0.0
        self.detection_history = []

    def _init_yolo(self):
        """初始化 YOLOv13 模型"""
        if not YOLO_AVAILABLE:
            print("[Detector] YOLO 不可用!")
            return

        model_path = self.config.model_name
        print(f"[Detector] 正在加载模型: {model_path}")
        print(f"[Detector] 设备: {self.device}")

        try:
            self.yolo_model = YOLO_CLASS(model_path)

            # 测试推理 - 确认模型真正能工作
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy, "TEST", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
            test_results = self.yolo_model(dummy, verbose=False)

            # 获取类别名称
            if test_results and hasattr(test_results[0], 'names'):
                self.class_names = test_results[0].names
                print(f"[Detector] 模型支持 {len(self.class_names)} 种类别")
                sample = [f"{v}" for k, v in list(self.class_names.items())[:10]]
                print(f"[Detector]   前10种: {', '.join(sample)}")

            self.use_yolo = True
            print(f"[Detector] ✓ YOLOv13 模型加载成功!")

        except Exception as e:
            print(f"[Detector] ✗ 模型加载失败: {e}")
            print(f"[Detector]")
            print(f"[Detector] 排查步骤:")
            print(f"[Detector]   1. 确认已安装 YOLOv13:")
            print(f"[Detector]      git clone https://github.com/iMoonLab/yolov13.git")
            print(f"[Detector]      cd yolov13 && pip install -e .")
            print(f"[Detector]   2. 确认权重文件存在: {model_path}")
            print(f"[Detector]      从 GitHub releases 下载对应的 .pt 文件")
            print(f"[Detector]   3. 验证: python -c \"from ultralytics import YOLO; YOLO('{model_path}')\"")

    def detect(self, frame: np.ndarray, enhance=True) -> List[Detection]:
        """执行目标检测"""
        t0 = time.time()
        self.frame_count += 1

        if not self.use_yolo:
            return []

        # 光照增强
        inp = self.illumination.process(frame) if enhance else frame

        # ===== YOLOv13 推理 =====
        try:
            results = self.yolo_model(
                inp,
                conf=self.config.confidence_threshold,
                iou=self.config.nms_threshold,
                imgsz=self.config.input_size[0],
                max_det=self.config.max_det,
                verbose=False
            )
        except Exception as e:
            if self.frame_count <= 3:
                print(f"[Detector] 推理出错: {e}")
            return []

        # ===== 解析检测结果 =====
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                # 类别过滤: 空列表=全部, 非空=只要指定类
                if self.config.target_classes and cls_id not in self.config.target_classes:
                    continue
                bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                conf = float(boxes.conf[i].item())
                name = self.class_names.get(cls_id, f"cls_{cls_id}")
                detections.append(Detection(bbox=bbox, confidence=conf,
                                            class_id=cls_id, class_name=name))

        # CNN特征提取
        if detections:
            self._extract_features(frame, detections)

        # 统计
        dt = time.time() - t0
        self.total_time += dt
        self.detection_history.append({
            'frame_id': self.frame_count, 'num': len(detections),
            'ms': dt * 1000, 'brightness': self.illumination.current_brightness
        })

        # 诊断日志
        if self.frame_count == 1:
            if detections:
                names = [f"{d.class_name}({d.confidence:.2f})" for d in detections[:5]]
                print(f"[Detector] 第1帧: 检测到 {len(detections)} 个目标: {', '.join(names)}")
            else:
                print(f"[Detector] 第1帧: 未检测到目标 (确认摄像头画面中有物体)")

        return detections

    def _extract_features(self, frame, detections):
        """提取CNN外观特征"""
        crops, valid = [], []
        h, w = frame.shape[:2]
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = np.clip(det.bbox.astype(int), 0, [w, h, w, h])
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crops.append(cv2.resize(frame[y1:y2, x1:x2], (64, 64)))
            valid.append(idx)
        if not crops:
            return
        batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
        with torch.no_grad():
            feats = self.feature_extractor(batch).cpu().numpy()
        for i, idx in enumerate(valid):
            detections[idx].feature = feats[i]

    @property
    def avg_detect_time(self):
        return (self.total_time / self.frame_count * 1000) if self.frame_count else 0

    @property
    def fps(self):
        return (self.frame_count / self.total_time) if self.total_time > 0 else 0

    def get_stats(self):
        return {
            'total_frames': self.frame_count,
            'avg_detect_time_ms': round(self.avg_detect_time, 2),
            'detector_fps': round(self.fps, 1),
            'device': str(self.device),
            'use_yolo': self.use_yolo,
            'brightness_stable': self.illumination.brightness_stable
        }
