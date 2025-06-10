# clip/modules/simplified_quality_estimator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SimplifiedQualityEstimator(nn.Module):
    """精简的质量评估器 - 保留数学指标和简化梯度指标"""

    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size

        # 任务相关性预测网络（替代复杂梯度计算）
        self.task_relevance_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 跨模态一致性网络
        self.consistency_evaluator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # 生成质量评估网络
        self.generation_quality_evaluator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def compute_mathematical_quality(self, features):
        """
        计算基于数学的特征质量指标
        Args:
            features: [batch_size, hidden_size] 或 [hidden_size]
        Returns:
            质量分数字典
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        batch_size, feature_dim = features.shape

        # 1. 特征范数稳定性
        feature_norms = torch.norm(features, dim=-1)
        ideal_norm = math.sqrt(feature_dim)
        norm_stability = torch.exp(-torch.abs(feature_norms - ideal_norm) / ideal_norm)

        # 2. 特征信息熵（分布均匀性）
        feature_probs = F.softmax(torch.abs(features), dim=-1)
        entropy = -torch.sum(feature_probs * torch.log(feature_probs + 1e-8), dim=-1)
        max_entropy = math.log(feature_dim)
        normalized_entropy = entropy / max_entropy

        # 3. 特征稀疏性质量
        sparsity_ratio = torch.mean((torch.abs(features) < 0.01).float(), dim=-1)
        optimal_sparsity = 0.1
        sparsity_quality = torch.exp(-torch.abs(sparsity_ratio - optimal_sparsity) / optimal_sparsity)

        # 4. 特征方差（信息丰富度）
        feature_variance = torch.var(features, dim=-1)
        variance_quality = torch.sigmoid(feature_variance - 0.1)  # 期望方差大于0.1

        return {
            'norm_stability': norm_stability,
            'information_entropy': normalized_entropy,
            'sparsity_quality': sparsity_quality,
            'variance_quality': variance_quality
        }

    def compute_cross_modal_consistency(self, img_feat, text_feat):
        """
        计算跨模态一致性
        Args:
            img_feat: [batch_size, 512]
            text_feat: [batch_size, 512]
        """
        if img_feat.dim() == 1:
            img_feat = img_feat.unsqueeze(0)
            text_feat = text_feat.unsqueeze(0)

        # 1. 余弦相似度
        cosine_similarity = F.cosine_similarity(img_feat, text_feat, dim=-1)
        semantic_alignment = (cosine_similarity + 1) / 2  # 归一化到[0,1]

        # 2. 特征距离质量
        feature_distance = torch.norm(img_feat - text_feat, dim=-1)
        max_distance = torch.norm(img_feat, dim=-1) + torch.norm(text_feat, dim=-1)
        normalized_distance = feature_distance / (max_distance + 1e-8)
        distance_quality = 1.0 - normalized_distance

        # 3. 使用网络评估一致性
        combined_feat = torch.cat([img_feat, text_feat], dim=-1)
        network_consistency = self.consistency_evaluator(combined_feat).squeeze(-1)

        return {
            'semantic_alignment': semantic_alignment,
            'distance_quality': distance_quality,
            'network_consistency': network_consistency
        }

    def compute_task_relevance(self, img_feat, text_feat):
        """
        计算任务相关性（简化版，避免复杂梯度计算）
        """
        # 使用网络预测任务相关性
        img_relevance = self.task_relevance_predictor(img_feat).squeeze(-1)
        text_relevance = self.task_relevance_predictor(text_feat).squeeze(-1)

        # 计算跨模态协同性
        cross_modal_synergy = F.cosine_similarity(img_feat, text_feat, dim=-1)
        cross_modal_synergy = (cross_modal_synergy + 1) / 2  # 归一化

        return {
            'img_task_relevance': img_relevance,
            'text_task_relevance': text_relevance,
            'cross_modal_synergy': cross_modal_synergy
        }

    def compute_generation_confidence(self, original_feat, generated_feat):
        """
        计算生成特征的可信度
        """
        if original_feat.dim() == 1:
            original_feat = original_feat.unsqueeze(0)
            generated_feat = generated_feat.unsqueeze(0)

        # 1. 特征相似度
        cosine_sim = F.cosine_similarity(original_feat, generated_feat, dim=-1)
        cosine_sim = (cosine_sim + 1) / 2

        # 2. 使用网络评估生成质量
        combined = torch.cat([original_feat, generated_feat], dim=-1)
        network_quality = self.generation_quality_evaluator(combined).squeeze(-1)

        # 3. 特征距离评估
        l2_distance = torch.norm(original_feat - generated_feat, dim=-1)
        distance_quality = torch.sigmoid(-l2_distance + 1.0)

        # 综合评估
        generation_confidence = 0.4 * cosine_sim + 0.4 * network_quality + 0.2 * distance_quality

        return torch.clamp(generation_confidence, 0.1, 0.9)

    def forward(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat, missing_type):
        """
        主要的质量评估接口

        Returns:
            quality_scores: List[Dict] 每个样本的质量分数
            格式: {
                'image_quality': {
                    'mathematical': Dict,     # 数学质量指标
                    'task_relevance': float,  # 任务相关性
                    'generation_confidence': float  # 生成置信度
                },
                'text_quality': {
                    'mathematical': Dict,
                    'task_relevance': float,
                    'generation_confidence': float
                },
                'cross_modal_consistency': Dict,  # 跨模态一致性
                'overall_confidence': float       # 整体置信度
            }
        """
        batch_size = image_feat.size(0)
        quality_scores = []

        # 计算任务相关性（一次计算，所有样本）
        task_relevance = self.compute_task_relevance(image_feat, text_feat)

        for i in range(batch_size):
            sample_quality = {}

            # === 图像质量评估 ===
            if missing_type[i] == 2:  # 缺失图像，使用生成的图像
                current_img_feat = enhanced_image_feat[i]
                img_generation_confidence = self.compute_generation_confidence(
                    image_feat[i], enhanced_image_feat[i]
                )
            else:  # 完整图像或缺失文本
                current_img_feat = image_feat[i]
                img_generation_confidence = torch.tensor(0.9).to(image_feat.device)

            img_math_quality = self.compute_mathematical_quality(current_img_feat)

            sample_quality['image_quality'] = {
                'mathematical': img_math_quality,
                'task_relevance': task_relevance['img_task_relevance'][i],
                'generation_confidence': img_generation_confidence
            }

            # === 文本质量评估 ===
            if missing_type[i] == 1:  # 缺失文本，使用生成的文本
                current_text_feat = enhanced_text_feat[i]
                text_generation_confidence = self.compute_generation_confidence(
                    text_feat[i], enhanced_text_feat[i]
                )
            else:  # 完整文本或缺失图像
                current_text_feat = text_feat[i]
                text_generation_confidence = torch.tensor(0.9).to(text_feat.device)

            text_math_quality = self.compute_mathematical_quality(current_text_feat)

            sample_quality['text_quality'] = {
                'mathematical': text_math_quality,
                'task_relevance': task_relevance['text_task_relevance'][i],
                'generation_confidence': text_generation_confidence
            }

            # === 跨模态一致性 ===
            cross_modal_consistency = self.compute_cross_modal_consistency(
                current_img_feat.unsqueeze(0), current_text_feat.unsqueeze(0)
            )
            sample_quality['cross_modal_consistency'] = cross_modal_consistency

            # === 整体置信度 ===
            img_overall = (
                    img_math_quality['norm_stability'][0] * 0.3 +
                    img_math_quality['information_entropy'][0] * 0.3 +
                    sample_quality['image_quality']['task_relevance'] * 0.4
            )

            text_overall = (
                    text_math_quality['norm_stability'][0] * 0.3 +
                    text_math_quality['information_entropy'][0] * 0.3 +
                    sample_quality['text_quality']['task_relevance'] * 0.4
            )

            cross_modal_overall = cross_modal_consistency['network_consistency'][0]

            overall_confidence = (img_overall + text_overall + cross_modal_overall) / 3
            sample_quality['overall_confidence'] = overall_confidence

            quality_scores.append(sample_quality)

        return quality_scores

    def compute_quality_loss(self, quality_scores, task_performance=None):
        """
        计算质量评估的训练损失
        """
        device = next(self.parameters()).device

        # 初始化为张量而不是标量
        quality_loss = torch.tensor(0.0, device=device, requires_grad=True)
        count = 0

        for quality in quality_scores:
            # 1. 任务相关性应该与实际任务性能相关
            if task_performance is not None:
                img_relevance = quality['image_quality']['task_relevance']
                text_relevance = quality['text_quality']['task_relevance']

                # 安全提取标量值并转换为张量
                if torch.is_tensor(img_relevance):
                    img_rel_value = img_relevance.item() if img_relevance.dim() == 0 else img_relevance.flatten()[
                        0].item()
                else:
                    img_rel_value = float(img_relevance)

                if torch.is_tensor(text_relevance):
                    text_rel_value = text_relevance.item() if text_relevance.dim() == 0 else text_relevance.flatten()[
                        0].item()
                else:
                    text_rel_value = float(text_relevance)

                # 转换为张量并确保在正确设备上
                avg_relevance = torch.tensor((img_rel_value + text_rel_value) / 2, device=device, requires_grad=True)

                # 确保task_performance在正确设备上并提取标量
                if torch.is_tensor(task_performance[count]):
                    task_perf_value = task_performance[count].item() if task_performance[count].dim() == 0 else \
                    task_performance[count].flatten()[0].item()
                else:
                    task_perf_value = float(task_performance[count])

                task_perf_tensor = torch.tensor(task_perf_value, device=device, requires_grad=True)

                # 计算MSE损失
                relevance_performance_loss = F.mse_loss(avg_relevance.unsqueeze(0), task_perf_tensor.unsqueeze(0))
                quality_loss = quality_loss + relevance_performance_loss

            # 2. 生成置信度应该与特征相似度一致
            img_confidence = quality['image_quality']['generation_confidence']
            text_confidence = quality['text_quality']['generation_confidence']

            # 安全提取置信度值
            if torch.is_tensor(img_confidence):
                img_conf_value = img_confidence.item() if img_confidence.dim() == 0 else img_confidence.flatten()[
                    0].item()
            else:
                img_conf_value = float(img_confidence)

            if torch.is_tensor(text_confidence):
                text_conf_value = text_confidence.item() if text_confidence.dim() == 0 else text_confidence.flatten()[
                    0].item()
            else:
                text_conf_value = float(text_confidence)

            # 转换为张量
            avg_confidence = torch.tensor((img_conf_value + text_conf_value) / 2, device=device, requires_grad=True)

            # 高置信度应该对应高质量特征
            confidence_consistency_loss = F.relu(torch.tensor(0.8, device=device) - avg_confidence)
            quality_loss = quality_loss + confidence_consistency_loss

            count += 1

        # 确保返回正确的张量
        if count > 0:
            return quality_loss / count
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)