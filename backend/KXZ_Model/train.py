# train.py
# -*- coding: utf-8 -*-
import time
import torch
import math
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model import DualBranchFlowerModel, FlowerModel, CombinedLoss
from torch import optim
import torch.nn.functional as F
import os
import numpy as np
from utils import Config, save_config, set_seed, create_data_loaders
from torch.cuda.amp import autocast, GradScaler

# 设置随机种子确保可重复性
set_seed(42)

# 设备配置
device = Config.device

# 在模型类中添加解冻方法
def unfreeze_stage(self, stage_prefix):
    """解冻指定前缀的层"""
    for name, param in self.named_parameters():
        if stage_prefix in name:
            param.requires_grad = True
            print(f"解冻层: {name}")

# 动态添加到模型类
FlowerModel.unfreeze_stage = unfreeze_stage
DualBranchFlowerModel.unfreeze_stage = unfreeze_stage

def create_optimizer_with_layerwise_lr(model, base_lr=Config.lr, current_epoch=0):
    """
    针对ConvNeXt V2的分层学习率优化器
    """
    # 调整学习率倍数，考虑GRN层的特性
    lr_multipliers = {
        'convnextv2_branch.convnextv2.encoder.stages.0': 0.05,  # 浅层小学习率
        'convnextv2_branch.convnextv2.encoder.stages.1': 0.1,
        'convnextv2_branch.convnextv2.encoder.stages.2': 0.2,
        'convnextv2_branch.convnextv2.encoder.stages.3': 0.5,    # 深层大学习率
        'swin_branch.layers.0': 0.05,
        'swin_branch.layers.1': 0.1,
        'swin_branch.layers.2': 0.2,
        'swin_branch.layers.3': 0.5,
        'convnextv2_proj': 0.3,  # 投影层
        'swin_proj': 0.3,
        'classifier': 1.0,       # 分类器大学习率
        'attention': 0.8,
        'non_local': 0.5
    }

    params_group = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr_multiplier = 0.1
        for layer_prefix, multiplier in lr_multipliers.items():
            if name.startswith(layer_prefix):
                lr_multiplier = multiplier
                break

        # 渐进式学习率调整
        if current_epoch < 10:
            lr_multiplier *= 0.5
        elif current_epoch > 50:
            lr_multiplier *= 1.2

        weight_decay = 0.01 if 'bias' not in name else 0.0
        params_group.append({
            'params': param,
            'lr': base_lr * lr_multiplier,
            'weight_decay': weight_decay
        })

    return torch.optim.AdamW(params_group, lr=base_lr, weight_decay=0.01)

# 自适应学习率管理器
class AdaptiveLRManager:
    """自适应学习率管理器"""

    def __init__(self, optimizer, base_lr=Config.lr):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.patience_counter = 0
        self.best_val_acc = 0.0
        self.lr_reduction_count = 0

    def update_lr(self, current_val_acc, epoch):
        """根据验证准确率动态调整学习率"""
        if current_val_acc > self.best_val_acc:
            self.best_val_acc = current_val_acc
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # 如果验证准确率连续5个epoch没有提升，降低学习率
        if self.patience_counter >= 5:
            self._reduce_lr()
            self.patience_counter = 0
            return True
        return False

    def _reduce_lr(self):
        """降低学习率"""
        if self.lr_reduction_count < 3:  # 最多降低3次
            self.current_lr *= 0.5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
            self.lr_reduction_count += 1
            print(f"📉 学习率降低到: {self.current_lr}")

    def warmup_lr(self, epoch, warmup_epochs=5):
        """学习率预热"""
        if epoch <= warmup_epochs:
            warmup_lr = self.base_lr * (epoch / warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
            return warmup_lr
        return None

# 混合精度训练
scaler = torch.cuda.amp.GradScaler()

def train(model, optimizer, epoch, trainLoader, device, criterion):
    """训练函数 - 添加混合精度训练和OHEM策略"""
    model.train()
    print(f'\n遥感图像分类训练 Epoch:{epoch} 开始==========')
    epoch_loss = 0.0
    correct_num = 0
    total_samples = 0

    for batch_idx, (data, labels) in enumerate(trainLoader):
        data, labels = data.to(device), labels.to(device)

        with torch.cuda.amp.autocast():
            outputs, features = model(data)
            loss = criterion(outputs, labels, features)

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度裁剪防止梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # 统计信息
        epoch_loss += loss.item() * data.size(0)
        preds = outputs.argmax(dim=1)
        correct_num += preds.eq(labels).sum().item()
        total_samples += data.size(0)

        if batch_idx % 10 == 0:
            batch_acc = preds.eq(labels).sum().item() / data.size(0)
            print(f"Epoch:{epoch}\t Batch:{batch_idx}\t Batch准确率:{batch_acc:.4f}\t 损失:{loss.item():.4f}")

    epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_num / total_samples if total_samples > 0 else 0
    print(f"Epoch:{epoch}\t 平均损失:{epoch_loss:.4f}\t 平均准确率:{epoch_acc:.4f}")
    return epoch_loss, epoch_acc

def validate(model, epoch, valLoader, device, criterion):
    """简化验证函数 - 不使用OHEM策略"""
    model.eval()
    print(f'\n遥感图像分类验证 Epoch:{epoch} 开始==========')
    epoch_loss = 0.0
    correct_num = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(valLoader):
            data, labels = data.to(device), labels.to(device)

            outputs, _ = model(data)
            loss = criterion(outputs, labels, None)

            epoch_loss += loss.item() * data.size(0)
            preds = outputs.argmax(dim=1)
            correct_num += preds.eq(labels).sum().item()
            total_samples += data.size(0)

            if batch_idx % 10 == 0:
                batch_acc = preds.eq(labels).sum().item() / data.size(0)
                print(f"Epoch:{epoch}\t Batch:{batch_idx}\t 验证批次准确率:{batch_acc:.4f}")

    epoch_loss = epoch_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_num / total_samples if total_samples > 0 else 0
    print(f"Epoch:{epoch}\t 验证损失:{epoch_loss:.4f}\t 验证准确率:{epoch_acc:.4f}")
    return epoch_loss, epoch_acc

class ImprovedEarlyStopping:
    def __init__(self, patience=25, min_delta=0.0005, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_acc = 0.0
        self.best_epoch = 0
        self.counter = 0
        self.best_state = None

    def __call__(self, current_acc, epoch, model, optimizer, scheduler):
        improvement = current_acc - self.best_acc

        if current_acc > self.best_acc:
            self.best_acc = current_acc
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self.best_state = {
                    'model_state': model.state_dict().copy(),
                    'optimizer_state': optimizer.state_dict().copy(),
                    'scheduler_state': scheduler.state_dict().copy() if scheduler else None
                }
            return False, improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best and self.best_state:
                    model.load_state_dict(self.best_state['model_state'])
                    optimizer.load_state_dict(self.best_state['optimizer_state'])
                    if self.best_state['scheduler_state'] and scheduler:
                        scheduler.load_state_dict(self.best_state['scheduler_state'])
                return True, improvement
            return False, improvement

def get_dynamic_ohem_schedule(epoch, total_epochs, current_accuracy=None):
    """动态OHEM调度函数 - 基于训练进度和性能自适应调整"""

    # 基础余弦退火策略
    progress = epoch / total_epochs
    base_ratio = 0.2 + 0.6 * (0.85 - 0.2) * (1 + math.cos(math.pi * progress))

    # 基于准确率的调整
    if current_accuracy is not None:
        if current_accuracy > 0.97:
            accuracy_factor = 0.15
        elif current_accuracy > 0.95:
            accuracy_factor = 0.3
        elif current_accuracy > 0.9:
            accuracy_factor = 0.5
        else:
            accuracy_factor = 0.7
    else:
        accuracy_factor = base_ratio

    # 结合策略权重
    final_ratio = 0.7 * base_ratio + 0.3 * accuracy_factor

    # 确保在合理范围内
    final_ratio = max(0.15, min(0.85, final_ratio))

    return {
        'use_ohem': True,
        'ohem_ratio': final_ratio,
        'strategy': 'dynamic'
    }

def main():
    """主训练流程 - ConvNeXt V2增强版"""
    # 保存配置
    trainLoader, valLoader, full_dataset = create_data_loaders()
    config_path = save_config(full_dataset)
    print(f"✅ 配置文件已保存: {config_path}")

    num_classes = len(full_dataset.classes)
    Config.num_classes = num_classes
    print(f"✅ 从数据集中获取类别数量: {num_classes}")
    print(f"✅ 类别列表: {full_dataset.classes}")

    # 创建使用ConvNeXt V2的双分支模型
    model = DualBranchFlowerModel(
        num_classes=num_classes,
        fusion_method='attention'
    )
    model = model.to(device)
    print(f"✅ ConvNeXt V2双分支模型创建完成")

    # 优化器配置 - 针对ConvNeXt V2调整
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=0.01)

    # 解冻计划调整 - 针对ConvNeXt V2的GRN特性优化
    unfreeze_schedule = {
        5: ['classifier'],
        10: ['convnextv2_proj', 'swin_proj'],
        15: ['attention'],
        20: ['convnextv2_branch.convnextv2.encoder.stages.3', 'swin_branch.layers.3'],
        30: ['convnextv2_branch.convnextv2.encoder.stages.2', 'swin_branch.layers.2'],
        40: ['convnextv2_branch.convnextv2.encoder.stages.1', 'swin_branch.layers.1'],
        50: ['convnextv2_branch.convnextv2.encoder.stages.0', 'swin_branch.layers.0'],
        60: ['non_local']
    }

    # 损失函数
    criterion = CombinedLoss(
        alpha=0.7,
        smoothing=0.1,
        num_classes=num_classes,
        feat_dim=1024,
        center_weight=0.05,
        use_ohem=True,
        ohem_ratio=0.7,
        dynamic_ohem=True,
        min_ohem_ratio=0.15,
        max_ohem_ratio=0.85
    )

    # 创建tensorboard写入器
    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=Config.log_dir, flush_secs=500)

    # 初始化改进的早停参数
    early_stopping = ImprovedEarlyStopping(patience=40, min_delta=0.0005)

    # 初始化自适应学习率管理器
    lr_manager = AdaptiveLRManager(optimizer, Config.lr)
    val_acc_history = []
    train_loss_history = []

    print("\n开始遥感图像分类训练")
    print(f"设备: {device}")
    print(f"总训练轮次: {Config.epochs}")
    print(f"类别数量: {num_classes}")
    print("=" * 60)

    # 核心训练循环
    for epoch in range(1, Config.epochs + 1):
        # 学习率预热
        warmup_lr = lr_manager.warmup_lr(epoch)
        if warmup_lr:
            print(f"🔥 学习率预热: {warmup_lr}")

        # 检查是否需要解冻新的层
        if epoch in unfreeze_schedule:
            for stage_prefix in unfreeze_schedule[epoch]:
                model.unfreeze_stage(stage_prefix)
                print(f"🔓 Epoch {epoch}: 解冻 {stage_prefix}")

            optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=0.01)
            print(f"🔄 重新创建优化器以适应新解冻的层")

        # 动态调整OHEM参数
        previous_val_acc = early_stopping.best_acc if epoch > 1 else 0.0
        ohem_params = get_dynamic_ohem_schedule(epoch, Config.epochs, previous_val_acc)

        # 更新损失函数的OHEM参数
        criterion.use_ohem = ohem_params['use_ohem']
        criterion.ohem_ratio = ohem_params['ohem_ratio']

        print(f"🔧 Epoch {epoch}: OHEM比例={criterion.ohem_ratio:.3f}")

        start_time = time.time()

        # 单轮训练+验证
        train_loss, train_acc = train(model, optimizer, epoch, trainLoader, device, criterion)
        val_loss, val_acc = validate(model, epoch, valLoader, device, criterion)

        # 记录历史
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)

        # 动态调整学习率
        lr_reduced = lr_manager.update_lr(val_acc, epoch)
        if lr_reduced:
            print(f"📉 自适应学习率调整，当前学习率: {lr_manager.current_lr}")

        # 记录到tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('OHEM_Ratio', criterion.ohem_ratio, epoch)

        # 改进的早停判断
        stop_training, improvement = early_stopping(val_acc, epoch, model, optimizer, None)

        if improvement > early_stopping.min_delta:
            print(f"✅ 验证准确率有效提升 {improvement:.4f}，最佳准确率: {early_stopping.best_acc:.4f}")
            best_model_path = os.path.join(Config.model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"📥 保存最佳模型权重到: {best_model_path} (准确率: {val_acc:.4f})")
        elif improvement > 0:
            print(f"⚠️  验证准确率微提升 {improvement:.4f}，等待计数 {early_stopping.counter}/{early_stopping.patience}")
        else:
            print(f"🔴 验证准确率无提升/下降，等待计数 {early_stopping.counter}/{early_stopping.patience}")

        # 检查是否触发早停
        if stop_training:
            print(f"\n⏹️  早停于第 {epoch} 轮（最优轮次：第{early_stopping.best_epoch}轮，最优准确率：{early_stopping.best_acc:.4f}）")
            break

        # 打印单轮耗时
        epoch_time = time.time() - start_time
        print(f"🔔 Epoch {epoch} 耗时: {epoch_time:.2f}秒")
        print("=" * 60)

        # 周期性保存模型
        if epoch % 20 == 0:
            checkpoint_path = os.path.join(Config.model_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"📥 保存周期模型到: {checkpoint_path}")

    # 训练结束：保存最终模型
    print(f"\n✅ 训练完成!")
    print(f"🏆 最佳验证准确率: {early_stopping.best_acc:.4f} (第{early_stopping.best_epoch}轮)")

    final_model_path = os.path.join(Config.model_dir, f"remote_sensing-final-epoch{epoch}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'best_acc': early_stopping.best_acc
    }, final_model_path)
    print(f"📥 保存完整最终模型到: {final_model_path}")

    writer.close()
    print("\n✅ 训练完成！")

if __name__ == '__main__':
    main()