#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '4'

import argparse
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path
import json
import time
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from model import DualBranchFlowerModel
from utils import val_transform, Config

MODEL_PATH = os.path.join(parent_dir, "model", "best_model.pth")
CONFIG_PATH = os.path.join(parent_dir, "model", "config.json")


class OptimizedRemoteSensingPredictor:
    def __init__(self, model_path=MODEL_PATH, config_path=CONFIG_PATH, batch_size=16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_path = model_path
        self.config_path = config_path

        self.class_mapping = self._load_class_mapping()
        self.model = None
        self.transform = val_transform

        self._setup_model()

    def _load_class_mapping(self):
        """从配置文件加载类别映射"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if 'class_mapping' in config:
                        mapping = config['class_mapping']
                        # 反转映射：索引到类别名
                        return {v: k for k, v in mapping.items()}
            return None
        except Exception as e:
            print(f"加载类别映射失败: {e}")
            return None

    def _setup_model(self):
        """设置和优化模型"""
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，使用CPU进行推理")

        print("🚀 开始加载模型...")

        # 加载配置
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                num_classes = config.get('num_classes', 30)
        else:
            num_classes = 30
            print("⚠️ 配置文件不存在，使用默认类别数30")

        # 加载模型
        self.model = DualBranchFlowerModel(
            num_classes=num_classes,
            fusion_method='attention'
        )

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"✅ 模型加载成功: {self.model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.model = self.model.to(self.device).eval()

        # 应用混合精度优化[11](@ref)
        if torch.cuda.is_available():
            self.model = self.model.half()  # 半精度
            print("✅ 已启用半精度推理")

        print("✅ 模型初始化完成")

    def _preprocess_batch(self, image_batch):
        """批量预处理图像"""
        batch_tensors = []
        for img_path in image_batch:
            try:
                image = Image.open(img_path).convert("RGB")
                tensor = self.transform(image)
                if torch.cuda.is_available():
                    tensor = tensor.half()  # 半精度
                tensor = tensor.to(self.device)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"⚠️ 图像预处理失败 {img_path}: {e}")
                # 创建备用张量
                fallback_tensor = torch.zeros(3, 224, 224, device=self.device)
                if torch.cuda.is_available():
                    fallback_tensor = fallback_tensor.half()
                batch_tensors.append(fallback_tensor)

        if batch_tensors:
            return torch.stack(batch_tensors)
        else:
            return torch.zeros(0, 3, 224, 224, device=self.device)

    def predict_batch(self, image_paths):
        """批量预测[1](@ref)"""
        if not image_paths:
            return []

        all_predictions = []

        for batch_idx in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[batch_idx:batch_idx + self.batch_size]

            try:
                # 预处理批次
                batch_tensor = self._preprocess_batch(batch_paths)

                if batch_tensor.size(0) == 0:
                    continue

                # 推理[11](@ref)
                start_time = time.time()

                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            logits, _ = self.model(batch_tensor)
                    else:
                        logits, _ = self.model(batch_tensor)

                inference_time = (time.time() - start_time) * 1000

                # 计算概率
                probabilities = torch.softmax(logits, dim=1)
                batch_confidences, batch_preds = torch.max(probabilities, dim=1)

                # 收集结果
                for i, (img_path, pred_idx, confidence) in enumerate(zip(
                        batch_paths, batch_preds.cpu().numpy(), batch_confidences.cpu().numpy())):

                    # 获取类别名称
                    if self.class_mapping:
                        class_name = self.class_mapping.get(int(pred_idx), f"class_{pred_idx}")
                    else:
                        class_name = f"class_{pred_idx}"

                    all_predictions.append({
                        'filename': os.path.basename(img_path),
                        'filepath': img_path,
                        'predicted_class': class_name,
                        'class_id': int(pred_idx),
                        'confidence': round(float(confidence), 4)
                    })

                # 进度显示
                if (batch_idx // self.batch_size) % 10 == 0:
                    progress = min(batch_idx + self.batch_size, len(image_paths))
                    print(f"📊 处理进度: {progress}/{len(image_paths)} "
                          f"({progress / len(image_paths) * 100:.1f}%)")

            except Exception as e:
                print(f"❌ 批次推理失败: {e}")
                # 为失败的批次添加默认结果
                for img_path in batch_paths:
                    all_predictions.append({
                        'filename': os.path.basename(img_path),
                        'filepath': img_path,
                        'predicted_class': 'unknown',
                        'class_id': -1,
                        'confidence': 0.0
                    })

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_predictions


def get_image_files(img_dir, limit=None):
    """获取目录中的所有图片文件"""
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    img_dir_path = Path(img_dir)
    image_files = []

    for ext in img_extensions:
        image_files.extend(img_dir_path.glob(f'**/*{ext}'))
        image_files.extend(img_dir_path.glob(f'**/*{ext.upper()}'))

    image_files = sorted([str(f) for f in image_files])

    if limit and len(image_files) > limit:
        print(f"⚠️  图片数量超过限制，只处理前 {limit} 张")
        image_files = image_files[:limit]

    return image_files


def main():
    parser = argparse.ArgumentParser(description='遥感图像分类预测')
    parser.add_argument('test_img_dir', type=str, help='测试图片目录路径')
    parser.add_argument('output_path', type=str, help='预测结果CSV输出路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批处理大小，默认16')
    parser.add_argument('--limit', type=int, default=5000, help='最大处理图片数量，默认5000')
    args = parser.parse_args()

    if not os.path.exists(args.test_img_dir):
        print(f"错误：目录不存在 - {args.test_img_dir}")
        return 1

    image_files = get_image_files(args.test_img_dir, args.limit)
    if not image_files:
        print(f"警告：目录中没有找到图片文件 - {args.test_img_dir}")
        return 1

    print(f"📁 找到 {len(image_files)} 张图片")
    print(f"⚙️  批处理大小: {args.batch_size}")

    try:
        predictor = OptimizedRemoteSensingPredictor(batch_size=args.batch_size)

        start_time = time.time()
        predictions = predictor.predict_batch(image_files)
        total_time = time.time() - start_time

        if not predictions:
            print("❌ 没有生成任何预测结果")
            return 1

        # 保存结果
        results_df = pd.DataFrame(predictions)

        output_dir = os.path.dirname(args.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        results_df.to_csv(args.output_path, index=False, encoding='utf-8')
        print(f"💾 结果已保存到: {args.output_path}")
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        print(f"📈 吞吐量: {len(image_files) / total_time:.2f} 张/秒")

        # 打印统计信息
        print("\n📊 预测统计:")
        print(f"总处理图片数: {len(predictions)}")
        confidence_stats = results_df['confidence'].describe()
        print(f"平均置信度: {confidence_stats['mean']:.4f}")
        print(f"最低置信度: {confidence_stats['min']:.4f}")
        print(f"最高置信度: {confidence_stats['max']:.4f}")

        print("✅ 预测完成")

        return 0

    except Exception as e:
        print(f"❌ 预测过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())