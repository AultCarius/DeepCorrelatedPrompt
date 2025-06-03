# clip/modules/reasonable_quality_estimator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TaskRelevanceQualityEstimator(nn.Module):
    """基于任务相关性的质量评估器 - 无信息泄露版本"""

    def __init__(self, hidden_size=512, num_classes=23):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # 【修复】信息论质量评估的编码器
        self.info_encoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 64)  # 用于离散化计算熵
        )

        # 模态重要性预测器 - 预测每个模态对任务的重要性
        self.modality_importance_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 重要性分数 [0,1]
        )

        # 特征判别器 - 预测特征是原始的还是生成的
        self.feature_discriminator = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 原始特征置信度
        )

        # 跨模态一致性预测器
        self.consistency_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 任务难度预测器 - 预测样本的任务难度
        self.task_difficulty_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 难度分数 [0,1]，越高越难
        )

    def compute_geometric_quality(self, features):
        """几何质量评估：基于特征空间的几何性质"""
        batch_size = features.size(0)
        geometric_scores = []

        for i in range(batch_size):
            feat = features[i]  # [hidden_size]

            # 转换为float32以避免精度问题
            feat = feat.float()

            # 1. 特征范数 (强度指标)
            feature_norm = torch.norm(feat, p=2)
            norm_quality = torch.sigmoid((feature_norm - 1.0) * 2.0)

            # 2. 特征稀疏性 (有效维度)
            sparsity = (torch.abs(feat) > 0.01).float().mean()
            sparsity_quality = sparsity  # 稀疏度适中最好

            # 3. 特征分布质量
            feat_std = torch.std(feat)
            std_quality = torch.sigmoid(feat_std * 10.0 - 1.0)

            geometric_quality = (norm_quality + sparsity_quality + std_quality) / 3.0
            geometric_scores.append(geometric_quality.to(features.device))

        return torch.stack(geometric_scores)

    def compute_information_quality(self, features):
        """信息论质量评估：基于信息熵和压缩性"""
        batch_size = features.size(0)
        info_scores = []

        for i in range(batch_size):
            feat = features[i]  # [hidden_size]

            # 转换为float32以支持histc等操作
            feat = feat.float()

            # 1. 特征熵 (信息丰富度)
            # 通过编码器离散化特征
            encoded = self.info_encoder(feat.unsqueeze(0))  # [1, 64]
            encoded_discrete = torch.round(torch.sigmoid(encoded) * 10).int()  # 离散化到0-10

            # 计算经验熵
            unique_vals = torch.unique(encoded_discrete)
            entropy = 0.0
            for val in unique_vals:
                prob = (encoded_discrete == val).float().mean()
                if prob > 0:
                    entropy -= prob * torch.log(prob + 1e-8)

            entropy_quality = torch.sigmoid(entropy - 2.0)  # 适度熵表示好质量

            # 2. 压缩比 (复杂度指标)
            # 使用特征的有效维度
            try:
                singular_values = torch.svd(feat.reshape(1, -1))[1]
                effective_rank = (singular_values > 0.01).sum().float()
                compression_quality = effective_rank / feat.size(-1)  # 有效维度比例
            except:
                # SVD失败时使用备用方法
                compression_quality = torch.tensor(0.5).to(feat.device)

            # 3. 信息密度
            feat_normalized = F.normalize(feat.unsqueeze(0), dim=-1)
            info_density = torch.sum(feat_normalized ** 2)  # L2范数的平方
            density_quality = torch.sigmoid(info_density - 0.5)

            # 综合信息论质量
            info_quality = (entropy_quality + compression_quality + density_quality) / 3.0
            info_scores.append(info_quality.to(features.device))

        return torch.stack(info_scores)

    def compute_consistency_quality(self, image_feat, text_feat):
        """跨模态一致性质量"""
        combined = torch.cat([image_feat, text_feat], dim=-1)
        consistency_scores = self.consistency_predictor(combined).squeeze(-1)
        return consistency_scores

    def predict_modality_importance(self, image_feat, text_feat):
        """预测模态重要性 - 这是我们要学习的关键指标"""
        image_importance = self.modality_importance_predictor(image_feat).squeeze(-1)
        text_importance = self.modality_importance_predictor(text_feat).squeeze(-1)
        return image_importance, text_importance

    def predict_feature_authenticity(self, original_feat, enhanced_feat):
        """预测特征的真实性 (原始 vs 生成)"""
        original_score = self.feature_discriminator(original_feat).squeeze(-1)
        enhanced_score = self.feature_discriminator(enhanced_feat).squeeze(-1)
        return original_score, enhanced_score

    def predict_task_difficulty(self, image_feat, text_feat):
        """预测任务难度"""
        combined = torch.cat([image_feat, text_feat], dim=-1)
        difficulty_score = self.task_difficulty_predictor(combined).squeeze(-1)
        return difficulty_score

    def forward(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat, missing_type):
        """
        完全无监督的质量评估 - 不需要真实标签

        Returns:
            quality_scores: List[Dict] 每个样本的质量分数
            predictions: Dict 用于训练的预测结果
        """
        batch_size = image_feat.size(0)

        # 1. 几何和信息论质量 (完全无监督)
        geometric_img = self.compute_geometric_quality(image_feat)
        geometric_text = self.compute_geometric_quality(text_feat)
        geometric_enhanced_img = self.compute_geometric_quality(enhanced_image_feat)
        geometric_enhanced_text = self.compute_geometric_quality(enhanced_text_feat)

        info_img = self.compute_information_quality(image_feat)
        info_text = self.compute_information_quality(text_feat)

        # 2. 跨模态一致性
        consistency_original = self.compute_consistency_quality(image_feat, text_feat)
        consistency_enhanced = self.compute_consistency_quality(enhanced_image_feat, enhanced_text_feat)

        # 3. 模态重要性预测 (关键 - 要与真实重要性对比学习)
        img_importance, text_importance = self.predict_modality_importance(image_feat, text_feat)
        enhanced_img_importance, enhanced_text_importance = self.predict_modality_importance(
            enhanced_image_feat, enhanced_text_feat
        )

        # 4. 特征真实性预测
        img_authenticity_orig, img_authenticity_enh = self.predict_feature_authenticity(
            image_feat, enhanced_image_feat
        )
        text_authenticity_orig, text_authenticity_enh = self.predict_feature_authenticity(
            text_feat, enhanced_text_feat
        )

        # 5. 任务难度预测
        task_difficulty = self.predict_task_difficulty(enhanced_image_feat, enhanced_text_feat)

        # 组合质量分数
        quality_scores = []
        for i in range(batch_size):
            sample_quality = {
                # 基础质量指标
                'geometric_image': geometric_img[i],
                'geometric_text': geometric_text[i],
                'information_image': info_img[i],
                'information_text': info_text[i],
                'consistency_original': consistency_original[i],
                'consistency_enhanced': consistency_enhanced[i],

                # 关键预测指标
                'predicted_image_importance': img_importance[i],
                'predicted_text_importance': text_importance[i],
                'predicted_task_difficulty': task_difficulty[i],

                # 特征真实性
                'image_authenticity': img_authenticity_orig[i],
                'text_authenticity': text_authenticity_orig[i],

                # 综合质量分数
                'overall_quality': (
                                           geometric_img[i] + geometric_text[i] +
                                           info_img[i] + info_text[i] +
                                           consistency_enhanced[i] +
                                           (img_importance[i] + text_importance[i]) / 2.0
                                   ) / 6.0
            }
            quality_scores.append(sample_quality)

        # 用于训练的预测结果
        predictions = {
            'image_importance': img_importance,
            'text_importance': text_importance,
            'enhanced_image_importance': enhanced_img_importance,
            'enhanced_text_importance': enhanced_text_importance,
            'task_difficulty': task_difficulty,
            'image_authenticity_original': img_authenticity_orig,
            'text_authenticity_original': text_authenticity_orig,
            'image_authenticity_enhanced': img_authenticity_enh,
            'text_authenticity_enhanced': text_authenticity_enh
        }

        return quality_scores, predictions


class QualityAwareObjective(nn.Module):
    """质量感知的目标函数 - 通过任务性能反向训练质量评估"""

    def __init__(self, importance_weight=0.1, authenticity_weight=0.05, difficulty_weight=0.05):
        super().__init__()
        self.importance_weight = importance_weight
        self.authenticity_weight = authenticity_weight
        self.difficulty_weight = difficulty_weight

    def compute_true_modality_importance(self, image_feat, text_feat, missing_type, task_predictions, task_targets):
        """
        通过消融实验计算真实的模态重要性

        核心思想: 如果去掉某个模态后任务性能下降很多，说明这个模态很重要
        """
        batch_size = image_feat.size(0)
        device = image_feat.device

        # 计算完整模态的任务损失
        full_task_loss = F.binary_cross_entropy_with_logits(
            task_predictions, task_targets, reduction='none'
        ).mean(dim=-1)  # [batch]

        true_img_importance = []
        true_text_importance = []

        for i in range(batch_size):
            if missing_type[i] == 0:  # 完整模态 - 可以计算真实重要性
                # 模拟去掉图像模态 (用零向量替代)
                simulated_img_feat = torch.zeros_like(image_feat[i:i + 1])
                simulated_combined = torch.cat([simulated_img_feat, text_feat[i:i + 1]], dim=-1)

                # 需要一个简单的任务预测器来评估性能
                # 这里用特征的范数作为性能的简单近似
                no_img_performance = torch.norm(simulated_combined, dim=-1)
                full_performance = torch.norm(torch.cat([image_feat[i:i + 1], text_feat[i:i + 1]], dim=-1), dim=-1)

                # 图像重要性 = 去掉图像后的性能下降
                img_importance = torch.clamp((full_performance - no_img_performance) / (full_performance + 1e-8), 0, 1)

                # 类似地计算文本重要性
                simulated_text_feat = torch.zeros_like(text_feat[i:i + 1])
                simulated_combined = torch.cat([image_feat[i:i + 1], simulated_text_feat], dim=-1)
                no_text_performance = torch.norm(simulated_combined, dim=-1)

                text_importance = torch.clamp((full_performance - no_text_performance) / (full_performance + 1e-8), 0,
                                              1)

            elif missing_type[i] == 1:  # 缺失文本 - 文本重要性很高
                img_importance = torch.tensor(0.3).to(device)  # 图像重要性中等
                text_importance = torch.tensor(0.9).to(device)  # 文本很重要(因为缺失)

            elif missing_type[i] == 2:  # 缺失图像 - 图像重要性很高
                img_importance = torch.tensor(0.9).to(device)  # 图像很重要(因为缺失)
                text_importance = torch.tensor(0.3).to(device)  # 文本重要性中等

            true_img_importance.append(img_importance)
            true_text_importance.append(text_importance)

        return torch.stack(true_img_importance), torch.stack(true_text_importance)

    def compute_true_task_difficulty(self, task_predictions, task_targets):
        """
        计算真实的任务难度

        思想: 预测概率低、标签稀有的样本更难
        """
        # 1. 预测不确定性 (熵)
        pred_probs = torch.sigmoid(task_predictions)
        pred_entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8) +
                                  (1 - pred_probs) * torch.log(1 - pred_probs + 1e-8), dim=-1)

        # 2. 标签稀有度 (罕见标签更难)
        label_frequency = task_targets.mean(dim=0)  # 每个类别的频率
        sample_rarity = (task_targets * (1 - label_frequency)).sum(dim=-1)

        # 3. 综合难度
        true_difficulty = 0.7 * pred_entropy / pred_entropy.max() + 0.3 * sample_rarity
        true_difficulty = torch.clamp(true_difficulty, 0, 1)

        return true_difficulty

    def compute_true_authenticity(self, missing_type):
        """
        计算特征的真实真实性

        思想: 完整模态的原始特征真实性高，缺失模态的生成特征真实性低
        """
        batch_size = len(missing_type)
        device = missing_type.device

        true_img_auth = []
        true_text_auth = []

        for i in range(batch_size):
            if missing_type[i] == 0:  # 完整模态
                img_auth = torch.tensor(0.9).to(device)  # 原始特征很真实
                text_auth = torch.tensor(0.9).to(device)
            elif missing_type[i] == 1:  # 缺失文本
                img_auth = torch.tensor(0.9).to(device)  # 图像是真实的
                text_auth = torch.tensor(0.3).to(device)  # 文本是生成的
            elif missing_type[i] == 2:  # 缺失图像
                img_auth = torch.tensor(0.3).to(device)  # 图像是生成的
                text_auth = torch.tensor(0.9).to(device)  # 文本是真实的

            true_img_auth.append(img_auth)
            true_text_auth.append(text_auth)

        return torch.stack(true_img_auth), torch.stack(true_text_auth)

    def forward(self, predicted_qualities, image_feat, text_feat, missing_type, task_predictions, task_targets):
        """
        计算质量感知的监督损失

        Args:
            predicted_qualities: 质量评估器的预测结果
            image_feat, text_feat: 特征
            missing_type: 缺失类型
            task_predictions: 任务预测结果 [batch, num_classes]
            task_targets: 任务真实标签 [batch, num_classes]
        """
        # 计算真实的质量指标
        true_img_importance, true_text_importance = self.compute_true_modality_importance(
            image_feat, text_feat, missing_type, task_predictions, task_targets
        )

        true_task_difficulty = self.compute_true_task_difficulty(task_predictions, task_targets)

        true_img_authenticity, true_text_authenticity = self.compute_true_authenticity(missing_type)

        # 计算预测损失
        importance_loss = (
                                  F.mse_loss(predicted_qualities['image_importance'], true_img_importance) +
                                  F.mse_loss(predicted_qualities['text_importance'], true_text_importance)
                          ) / 2.0

        difficulty_loss = F.mse_loss(predicted_qualities['task_difficulty'], true_task_difficulty)

        authenticity_loss = (
                                    F.mse_loss(predicted_qualities['image_authenticity_original'],
                                               true_img_authenticity) +
                                    F.mse_loss(predicted_qualities['text_authenticity_original'],
                                               true_text_authenticity)
                            ) / 2.0

        # 总质量监督损失
        total_quality_loss = (
                self.importance_weight * importance_loss +
                self.difficulty_weight * difficulty_loss +
                self.authenticity_weight * authenticity_loss
        )

        return {
            'quality_supervision_loss': total_quality_loss,
            'importance_loss': importance_loss,
            'difficulty_loss': difficulty_loss,
            'authenticity_loss': authenticity_loss
        }