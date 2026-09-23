# model.py
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, AutoModelForImageClassification
import os
import sys
import math

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import Config


def get_convnext_model():
    """安全地获取ConvNeXt V2模型 - 返回特征提取器"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model", "convnext")

    print(f"✅ 尝试从本地加载ConvNeXt V2模型: {model_path}")

    if os.path.exists(model_path):
        try:
            # 首先尝试使用timm加载
            try:
                model = timm.create_model('convnextv2_base', pretrained=False)
                # 加载本地权重
                state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'),
                                        map_location='cpu')
                model.load_state_dict(state_dict)
                print("✅ ConvNeXt V2模型通过timm加载成功")
                return model
            except:
                # 尝试使用transformers加载
                model = AutoModel.from_pretrained(model_path)
                print("✅ ConvNeXt V2模型通过transformers加载成功")
                return model
        except Exception as e:
            print(f"❌ ConvNeXt V2模型加载失败: {e}")
            raise RuntimeError(f"无法加载本地ConvNeXt V2模型: {e}")
    else:
        raise FileNotFoundError(f"ConvNeXt V2模型目录不存在: {model_path}")


def get_swin_base_model():
    """安全地获取Swin Transformer模型 - 返回特征提取器"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model", "swin")

    print(f"✅ 尝试从本地加载Swin模型: {model_path}")

    if os.path.exists(model_path):
        try:
            # 优先使用timm加载Swin模型
            try:
                model = timm.create_model('swin_base_patch4_window7_224',
                                          pretrained=False, num_classes=0)
                # 尝试加载本地权重
                state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'),
                                        map_location='cpu')
                model.load_state_dict(state_dict)
                print("✅ Swin模型通过timm加载成功（特征提取模式）")
                return model
            except Exception as e:
                print(f"❌ timm加载失败: {e}，尝试transformers")

            # transformers加载
            model = AutoModel.from_pretrained(model_path)
            print("✅ Swin模型通过transformers加载成功（特征提取模式）")
            return model

        except Exception as e:
            print(f"❌ Swin模型加载失败: {e}")
            try:
                model = AutoModelForImageClassification.from_pretrained(model_path)
                print("✅ Swin模型加载成功（分类模式，将提取特征）")
                return model
            except Exception as e2:
                print(f"❌ Swin模型完全加载失败: {e2}")
                raise RuntimeError("无法加载本地Swin模型")
    else:
        raise FileNotFoundError(f"Swin模型目录不存在: {model_path}")





class GRNLayer(nn.Module):
    """全局响应归一化层"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class NonLocalFusion(nn.Module):
    """非局部特征融合模块[7](@ref)"""

    def __init__(self, in_channels):
        super(NonLocalFusion, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.size(0)
        g = self.conv1(x).view(batch_size, -1, x.size(2))
        theta = self.conv2(x).view(batch_size, -1, x.size(2))
        phi = self.conv3(x).view(batch_size, -1, x.size(2))

        f = torch.einsum('bnc,bmc->bnm', theta, phi)
        f = F.softmax(f, dim=-1)

        y = torch.einsum('bnm,bmc->bnc', f, g)
        y = y.view(batch_size, -1, x.size(2))
        return self.gamma * y + x


class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CenterLoss(nn.Module):
    """中心损失增强特征判别性"""

    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features, labels):
        centers = self.centers.to(labels.device)
        centers_batch = centers[labels]
        dist = (features - centers_batch).pow(2).sum(dim=1)
        loss = dist.mean()
        return self.alpha * loss


class CombinedLoss(nn.Module):
    """组合损失函数"""

    def __init__(self, alpha=0.7, smoothing=0.1, num_classes=30, feat_dim=1024,
                 center_weight=0.1, use_ohem=False, ohem_ratio=0.7,
                 dynamic_ohem=True, min_ohem_ratio=0.2, max_ohem_ratio=0.8):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=1, gamma=2, reduction='none')
        self.center_loss = CenterLoss(num_classes, feat_dim)
        self.alpha = alpha
        self.center_weight = center_weight
        self.use_ohem = use_ohem
        self.ohem_ratio = ohem_ratio
        self.dynamic_ohem = dynamic_ohem
        self.min_ohem_ratio = min_ohem_ratio
        self.max_ohem_ratio = max_ohem_ratio

        # 标签平滑的交叉熵损失
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, targets, features=None, reduction='mean'):
        # 基础分类损失
        if self.use_ohem:
            # 正确的OHEM实现 - 需要计算每个样本的损失[7,8](@ref)
            with torch.no_grad():
                # 计算每个样本的损失，而不是平均损失
                losses = F.cross_entropy(logits, targets, reduction='none')  # 返回形状为 [batch_size] 的张量
                batch_size = losses.size(0)
                k = max(1, int(self.ohem_ratio * batch_size))  # 确保至少选择1个样本[3](@ref)
                _, indices = torch.topk(losses, k, largest=True)  # 选择损失最大的样本[1,4](@ref)

            classification_loss = F.cross_entropy(logits[indices], targets[indices])
        else:
            classification_loss = F.cross_entropy(logits, targets)

        # 中心损失
        if features is not None:
            center_loss_val = self.center_loss(features, targets)
            total_loss = classification_loss + self.center_weight * center_loss_val
        else:
            total_loss = classification_loss

        return total_loss


class ConvNeXtBranch(nn.Module):
    """ConvNeXt分支"""

    def __init__(self, pretrained=True):
        super(ConvNeXtBranch, self).__init__()
        self.convnext = get_convnext_model()
        self._freeze_initial_layers()

    def _freeze_initial_layers(self):
        """冻结初始层"""
        for name, param in self.convnext.named_parameters():
            param.requires_grad = False
        print("✅ ConvNeXt初始层已冻结")

    def unfreeze_stage(self, stage_prefix):
        """解冻指定阶段"""
        unfrozen_count = 0
        for name, param in self.convnext.named_parameters():
            if stage_prefix in name:
                param.requires_grad = True
                unfrozen_count += 1
        print(f"✅ 解冻ConvNeXt包含 '{stage_prefix}' 的 {unfrozen_count} 个参数")

    def forward(self, x):
        """前向传播"""
        outputs = self.convnext(x)

        # 特征提取
        if hasattr(outputs, 'last_hidden_state'):
            features = torch.mean(outputs.last_hidden_state, dim=[2, 3])
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        else:
            features = outputs[0] if isinstance(outputs, tuple) else outputs

        return features

    def get_feature_dim(self):
        """获取特征维度"""
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_output = self.forward(test_input)
            return test_output.shape[-1]


class SwinBaseBranch(nn.Module):
    """Swin-Base分支（替换原来的Swin-Tiny）[4](@ref)"""

    def __init__(self, pretrained=True):
        super(SwinBaseBranch, self).__init__()
        self.swin = get_swin_base_model()
        self._freeze_initial_layers()

    def _freeze_initial_layers(self):
        """冻结初始层"""
        for name, param in self.swin.named_parameters():
            param.requires_grad = False
        print("✅ Swin-Base初始层已冻结")

    def unfreeze_stage(self, stage_prefix):
        """解冻指定阶段"""
        unfrozen_count = 0
        for name, param in self.swin.named_parameters():
            if stage_prefix in name:
                param.requires_grad = True
                unfrozen_count += 1
        print(f"✅ 解冻Swin-Base包含 '{stage_prefix}' 的 {unfrozen_count} 个参数")

    def forward(self, x):
        """前向传播"""
        outputs = self.swin(x)

        # 特征提取
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
            if len(last_hidden.shape) == 3:
                features = torch.mean(last_hidden, dim=1)
            else:
                features = last_hidden
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        else:
            features = outputs[0] if isinstance(outputs, tuple) else outputs

        return features

    def get_feature_dim(self):
        """获取特征维度"""
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_output = self.forward(test_input)
            return test_output.shape[-1]


class DualBranchFlowerModel(nn.Module):
    """双分支模型：ConvNeXt + Swin-Base[7,8](@ref)"""

    def __init__(self, num_classes=30, fusion_dim=1024, fusion_method='attention'):
        super(DualBranchFlowerModel, self).__init__()

        # 使用双分支
        self.convnext_branch = ConvNeXtBranch()
        self.swin_branch = SwinBaseBranch()

        self.fusion_method = fusion_method
        self.num_classes = num_classes

        # 获取特征维度
        self.convnext_feat_dim = self.convnext_branch.get_feature_dim()
        self.swin_feat_dim = self.swin_branch.get_feature_dim()

        print(f"✅ ConvNeXt特征维度: {self.convnext_feat_dim}")
        print(f"✅ Swin-Base特征维度: {self.swin_feat_dim}")

        # 特征投影层
        self.convnext_proj = nn.Sequential(
            nn.Linear(self.convnext_feat_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.swin_proj = nn.Sequential(
            nn.Linear(self.swin_feat_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 特征融合模块
        if fusion_method == 'concat':
            fusion_input_dim = fusion_dim * 2
        elif fusion_method == 'sum' or fusion_method == 'attention':
            fusion_input_dim = fusion_dim
            if fusion_method == 'attention':
                self.attention = nn.Sequential(
                    nn.Linear(fusion_dim * 2, fusion_dim),
                    nn.ReLU(),
                    nn.Linear(fusion_dim, fusion_dim // 2),
                    nn.ReLU(),
                    nn.Linear(fusion_dim // 2, 2),
                    nn.Softmax(dim=1)
                )
                self.non_local = NonLocalFusion(fusion_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Dropout(0.3),
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.GELU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim // 2, num_classes)
        )

        self._init_weights()
        print(f"✅ 双分支模型创建完成（ConvNeXt + Swin-Base），融合方法: {fusion_method}")

    def _init_weights(self):
        """初始化权重"""
        for m in [self.convnext_proj, self.swin_proj]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)

        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def unfreeze_stage(self, stage_prefix):
        """解冻指定阶段"""
        if 'convnext' in stage_prefix:
            self.convnext_branch.unfreeze_stage(stage_prefix)
        else:
            self.swin_branch.unfreeze_stage(stage_prefix)

    def forward(self, x):
        # 提取双分支特征
        convnext_features = self.convnext_branch(x)
        swin_features = self.swin_branch(x)

        # 特征投影
        convnext_proj = self.convnext_proj(convnext_features)
        swin_proj = self.swin_proj(swin_features)

        # 特征融合[6](@ref)
        if self.fusion_method == 'concat':
            fused_features = torch.cat([convnext_proj, swin_proj], dim=1)
        elif self.fusion_method == 'sum':
            fused_features = convnext_proj + swin_proj
        elif self.fusion_method == 'attention':
            combined = torch.cat([convnext_proj, swin_proj], dim=1)
            attention_weights = self.attention(combined)
            conv_weight = attention_weights[:, 0].unsqueeze(1)
            swin_weight = attention_weights[:, 1].unsqueeze(1)
            fused_features = conv_weight * convnext_proj + swin_weight * swin_proj
            fused_features = self.non_local(fused_features.unsqueeze(2)).squeeze(2)

        # 分类
        logits = self.classifier(fused_features)

        return logits, fused_features


# 保持兼容性
class FlowerModel(nn.Module):
    def __init__(self, model, num_classes):
        super(FlowerModel, self).__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self._get_in_features()
        self.model.classifier = nn.Linear(in_features, num_classes)
        print(f"✅ 替换分类器: {in_features} -> {num_classes}")

    def _get_in_features(self):
        if hasattr(self.model, 'classifier'):
            classifier = self.model.classifier
            if isinstance(classifier, nn.Linear):
                return classifier.in_features
            elif isinstance(classifier, nn.Sequential):
                for module in reversed(classifier):
                    if isinstance(module, nn.Linear):
                        return module.in_features
        try:
            if hasattr(self.model, 'config'):
                config = self.model.config
                if hasattr(config, 'hidden_sizes'):
                    return config.hidden_sizes[-1]
                elif hasattr(config, 'num_features'):
                    return config.num_features
        except:
            pass
        raise ValueError("无法自动推断模型分类器的输入特征数。")

    def unfreeze_stage(self, stage_name):
        for name, param in self.model.named_parameters():
            if stage_name in name:
                param.requires_grad = True
                print(f"解冻层: {name}")

    def forward(self, x):
        output = self.model(x)
        return output.logits if hasattr(output, 'logits') else output