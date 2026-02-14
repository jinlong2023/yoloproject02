"""
========================================================================
YOLOv13 COCO 数据集训练脚本
========================================================================

!!!! 训练前必须完成以下准备 !!!!

第1步: 下载 COCO 2017 数据集, 放到项目目录下:

  E:\yolov13project\datasets\coco\
  ├── images\
  │   ├── train2017\     ← 下载 http://images.cocodataset.org/zips/train2017.zip (~18GB)
  │   └── val2017\       ← 下载 http://images.cocodataset.org/zips/val2017.zip (~1GB)
  └── labels\
      ├── train2017\     ← 下载 https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip
      └── val2017\

第2步: 修改 coco_local.yaml 中的 path 为你的实际路径

第3步: 运行训练:
  python train_coco.py --mode finetune --model yolov13s.pt --data coco_local.yaml --epochs 100 --batch 8
========================================================================
"""

import argparse
import os
import sys
import torch


def check_environment():
    """检查运行环境"""
    print("=" * 60)
    print("  YOLOv13 训练环境检查")
    print("=" * 60)

    try:
        from ultralytics import YOLO
        print(f"  ✓ ultralytics 已安装")
    except ImportError:
        print(f"  ✗ ultralytics 未安装!")
        print(f"    git clone https://github.com/iMoonLab/yolov13.git")
        print(f"    cd yolov13 && pip install -e .")
        sys.exit(1)

    print(f"  ✓ PyTorch {torch.__version__}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory/ 1024**3
        print(f"  ✓ CUDA: {gpu_name} ({gpu_mem:.1f} GB)")
        if gpu_mem < 4:
            print(f"  ⚠ 显存较小, 建议 --batch 4 --imgsz 480")
        elif gpu_mem < 8:
            print(f"  建议 --batch 8 --imgsz 640")
        else:
            print(f"  建议 --batch 16 --imgsz 640")
    else:
        print(f"  ⚠ CUDA 不可用, 将使用 CPU (非常慢!)")

    print("=" * 60)


def check_dataset(data_yaml):
    """检查数据集配置和文件是否存在"""
    print(f"\n[检查数据集] 配置文件: {data_yaml}")

    if not os.path.exists(data_yaml):
        print(f"  ✗ 配置文件不存在: {data_yaml}")
        print(f"  请确认 coco_local.yaml 在项目根目录")
        return False

    # 读取yaml检查path
    try:
        import yaml
        with open(data_yaml, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        data_path = cfg.get('path', '')
        train_sub = cfg.get('train', '')
        val_sub = cfg.get('val', '')

        train_dir = os.path.join(data_path, train_sub)
        val_dir = os.path.join(data_path, val_sub)
        train_label_dir = os.path.join(data_path, 'labels', 'train2017')
        val_label_dir = os.path.join(data_path, 'labels', 'val2017')

        print(f"  数据根目录: {data_path}")
        print(f"  训练图片: {train_dir}  → {'✓ 存在' if os.path.isdir(train_dir) else '✗ 不存在!'}")
        print(f"  验证图片: {val_dir}  → {'✓ 存在' if os.path.isdir(val_dir) else '✗ 不存在!'}")
        print(f"  训练标签: {train_label_dir}  → {'✓ 存在' if os.path.isdir(train_label_dir) else '✗ 不存在!'}")
        print(f"  验证标签: {val_label_dir}  → {'✓ 存在' if os.path.isdir(val_label_dir) else '✗ 不存在!'}")

        # 检查图片数量
        if os.path.isdir(train_dir):
            n = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
            print(f"  训练图片数量: {n}")
            if n == 0:
                print(f"  ✗ 训练图片为0! 请下载 COCO train2017.zip 并解压到 {train_dir}")
                return False

        if os.path.isdir(train_label_dir):
            n = len([f for f in os.listdir(train_label_dir) if f.endswith('.txt')])
            print(f"  训练标签数量: {n}")
            if n == 0:
                print(f"  ✗ 标签为0! 请下载 coco2017labels.zip 并解压到 {data_path}/labels/")
                return False

        all_exist = (os.path.isdir(train_dir) and os.path.isdir(val_dir) and
                     os.path.isdir(train_label_dir) and os.path.isdir(val_label_dir))

        if not all_exist:
            print(f"\n  ✗ 数据集不完整! 请按以下步骤下载:")
            print(f"    1. 图片: http://images.cocodataset.org/zips/train2017.zip")
            print(f"       解压到: {data_path}/images/train2017/")
            print(f"    2. 图片: http://images.cocodataset.org/zips/val2017.zip")
            print(f"       解压到: {data_path}/images/val2017/")
            print(f"    3. 标签: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip")
            print(f"       解压到: {data_path}/labels/")
            return False

        print(f"  ✓ 数据集检查通过!\n")
        return True

    except ImportError:
        print(f"  (跳过详细检查, pyyaml未安装)")
        return True
    except Exception as e:
        print(f"  检查异常: {e}")
        return True


def train_finetune(args):
    """基于预训练权重微调"""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  模式: 预训练微调")
    print(f"  模型: {args.model}")
    print(f"  数据: {args.data}")
    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch}  |  ImgSz: {args.imgsz}")
    print(f"  设备: {args.device}")
    print(f"{'='*60}\n")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,              # 使用自定义数据集配置
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project="runs/detect",
        name=args.name,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        mosaic=1.0,
        mixup=0.05,
        copy_paste=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        warmup_epochs=3,
        warmup_momentum=0.8,
        amp=True,
        close_mosaic=10,
        patience=50,

        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\n  ✓ 训练完成! 最佳模型: runs/detect/{args.name}/weights/best.pt\n")
    return results


def train_scratch(args):
    """从零开始训练"""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  模式: 从零训练")
    print(f"  配置: {args.model}  |  数据: {args.data}")
    print(f"  Epochs: {args.epochs} (建议 300+)")
    print(f"{'='*60}\n")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project="runs/detect",
        name=args.name,

        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.15,
        scale=0.9,
        fliplr=0.5,

        warmup_epochs=5,
        amp=True,
        close_mosaic=15,
        patience=0,

        save=True,
        save_period=20,
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\n  ✓ 训练完成! 最佳模型: runs/detect/{args.name}/weights/best.pt\n")
    return results


def validate(args):
    """验证模型"""
    from ultralytics import YOLO

    print(f"\n  验证: {args.model} on {args.data}\n")
    model = YOLO(args.model)

    metrics = model.val(
        data=args.data,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        split="val",
        save_json=True,
        plots=True,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"  mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"  Precision     : {metrics.box.mp:.4f}")
    print(f"  Recall        : {metrics.box.mr:.4f}")
    print(f"{'='*60}\n")
    return metrics


def export_model(args):
    """导出模型"""
    from ultralytics import YOLO
    print(f"\n  导出: {args.model} → {args.export_format}\n")
    model = YOLO(args.model)
    model.export(format=args.export_format, imgsz=args.imgsz, half=args.half, dynamic=True)
    print(f"\n  ✓ 导出完成!\n")


def predict_test(args):
    """推理测试"""
    from ultralytics import YOLO
    print(f"\n  推理: {args.model} on {args.source}\n")
    model = YOLO(args.model)
    results = model.predict(
        source=args.source, conf=0.35, iou=0.5, imgsz=args.imgsz,
        device=args.device, save=True, save_txt=True,
        project="runs/predict", name=args.name, show=False,
    )
    print(f"\n  ✓ 结果: runs/predict/{args.name}/\n")
    return results


def resume_training(args):
    """恢复训练"""
    from ultralytics import YOLO
    print(f"\n  恢复训练: {args.model}\n")
    model = YOLO(args.model)
    return model.train(resume=True)


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv13 COCO Training")

    p.add_argument('--mode', type=str, default='finetune',
                   choices=['finetune', 'scratch', 'val', 'export', 'predict', 'resume'],
                   help='finetune/scratch/val/export/predict/resume')

    p.add_argument('--model', type=str, default='yolov13n.pt',
                   help='yolov13n.pt / yolov13s.pt / yolov13n.yaml / best.pt')

    # ===== 关键: 数据集配置文件 =====
    p.add_argument('--data', type=str, default='coco_local.yaml',
                   help='数据集配置文件 (默认 coco_local.yaml, 需要修改其中的path)')

    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', type=str, default='0', help='0 / 0,1 / cpu')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--name', type=str, default='train')

    p.add_argument('--export-format', type=str, default='onnx',
                   choices=['onnx', 'torchscript', 'engine', 'openvino'])
    p.add_argument('--half', action='store_true')
    p.add_argument('--source', type=str, default='ultralytics/assets')

    return p.parse_args()


def main():
    args = parse_args()
    check_environment()

    # 训练/验证模式下检查数据集
    if args.mode in ('finetune', 'scratch', 'val'):
        if not check_dataset(args.data):
            print("\n[错误] 数据集检查未通过, 请按提示下载和配置数据集后重试!")
            sys.exit(1)

    if args.mode == 'finetune':
        train_finetune(args)
    elif args.mode == 'scratch':
        if not args.model.endswith('.yaml'):
            args.model = 'yolov13n.yaml'
        train_scratch(args)
    elif args.mode == 'val':
        validate(args)
    elif args.mode == 'export':
        export_model(args)
    elif args.mode == 'predict':
        predict_test(args)
    elif args.mode == 'resume':
        resume_training(args)


if __name__ == "__main__":
    main()
