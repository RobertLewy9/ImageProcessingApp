# utils.py
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
from typing import Optional, Callable
import pandas as pd
from PIL import Image
from torchvision import transforms
import random
import numpy as np
import json
import time
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import ImageFolder


# ==================== 配置部分 ====================
class Config:
    # 数据配置 - 修改为遥感图像路径
    data_path = "../data/"  # 数据根目录，包含train和val文件夹
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_size = (224, 224)  # 遥感图像尺寸

    # 训练配置
    batch_size = 32  # 遥感图像可能较大，适当减小batch size
    num_workers = 4
    lr = 1e-4  # 学习率
    epochs = 100  # 训练轮次
    model_dir = "../model/"  # 模型保存路径
    log_dir = "../logs/" + time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())  # 日志路径

    # OHEM配置
    dynamic_ohem = True
    min_ohem_ratio = 0.15
    max_ohem_ratio = 0.85


# 自定义JSON编码器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        else:
            return super().default(obj)


def save_config(dataset):
    """保存配置到model/config.json"""
    config_dict = {
        "data_path": Config.data_path,
        "device": str(Config.device),
        "image_size": Config.image_size,
        "batch_size": Config.batch_size,
        "num_workers": Config.num_workers,
        "lr": Config.lr,
        "epochs": Config.epochs,
        "class_mapping": dataset.class_to_idx,  # 使用ImageFolder的class_to_idx
        "num_classes": len(dataset.classes),
        "dynamic_ohem": Config.dynamic_ohem,
        "min_ohem_ratio": Config.min_ohem_ratio,
        "max_ohem_ratio": Config.max_ohem_ratio
    }

    os.makedirs(Config.model_dir, exist_ok=True)
    config_path = os.path.join(Config.model_dir, "config.json")

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    return config_path


# ==================== 工具函数部分 ====================
def set_seed(seed=42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """为每个数据加载工作进程设置随机种子"""
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)


def custom_collate_fn(batch):
    """简化版的collate_fn"""
    images, labels = zip(*batch)
    images_tensor = torch.stack(images)
    labels_tensor = torch.tensor(labels)
    return images_tensor, labels_tensor


# 数据增强配置 - 针对遥感图像调整
train_transform = transforms.Compose([
    transforms.Resize(Config.image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),  # 增加旋转角度
    transforms.ColorJitter(
        brightness=0.2,  # 调整亮度对比度
        contrast=0.2,
        saturation=0.2,
        hue=0.05),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
    ], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(Config.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ==================== 数据加载部分 ====================
def create_data_loaders():
    """创建训练和验证数据加载器 - 使用ImageFolder加载文件夹结构数据[2,4](@ref)"""

    # 检查数据路径
    train_dir = os.path.join(Config.data_path, 'train')
    val_dir = os.path.join(Config.data_path, 'val')

    if not os.path.exists(train_dir):
        # 如果不存在train/val子目录，使用整个数据目录并自动分割
        print("⚠️ 未找到train/val子目录，将自动分割数据集...")
        return create_data_loaders_with_split()

    print(f"📁 训练数据路径: {train_dir}")
    print(f"📁 验证数据路径: {val_dir}")

    # 创建数据集
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform)

    print(f"✅ 数据集加载成功")
    print(f"训练集数量: {len(train_dataset)}")
    print(f"验证集数量: {len(val_dataset)}")
    print(f"类别数量: {len(train_dataset.classes)}")
    print(f"类别映射: {train_dataset.class_to_idx}")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker,
        pin_memory=True
    )

    print("✅ 数据加载器创建完成！")
    return train_loader, val_loader, train_dataset


def create_data_loaders_with_split():
    """当没有train/val分割时，自动分割数据集[2](@ref)"""

    # 创建完整数据集
    full_dataset = ImageFolder(root=Config.data_path, transform=None)

    print(f"📁 数据路径: {Config.data_path}")
    print(f"总样本数: {len(full_dataset)}")
    print(f"类别数量: {len(full_dataset.classes)}")
    print(f"类别映射: {full_dataset.class_to_idx}")

    # 获取所有标签
    labels = [label for _, label in full_dataset.samples]

    # 分层分割
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_indices, val_indices = next(sss.split(np.zeros(len(labels)), labels))

    # 创建应用变换的子集
    class TransformSubset(Dataset):
        def __init__(self, subset, indices, transform=None):
            self.subset = subset
            self.indices = indices
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[self.indices[index]]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.indices)

    train_subset = TransformSubset(full_dataset, train_indices, transform=train_transform)
    val_subset = TransformSubset(full_dataset, val_indices, transform=val_transform)

    print(f"训练集数量: {len(train_subset)}")
    print(f"验证集数量: {len(val_subset)}")

    # 创建DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=custom_collate_fn,
        worker_init_fn=seed_worker,
        pin_memory=True
    )

    print("✅ 数据加载器创建完成（自动分割）！")
    return train_loader, val_loader, full_dataset
if __name__ == '__main__':
    create_data_loaders()