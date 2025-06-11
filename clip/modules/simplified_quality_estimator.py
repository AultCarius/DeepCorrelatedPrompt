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
        self.original_input_quality_evaluator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def compute_mathematical_quality(self, features):
        """=== 修改：添加数值稳定性保护 ==="""
        if features.dim() == 1:
            features = features.unsqueeze(0)

        batch_size, feature_dim = features.shape

        # 1. 特征范数稳定性 - 添加稳定性保护
        feature_norms = torch.norm(features, dim=-1)
        ideal_norm = math.sqrt(feature_dim)
        norm_diff = torch.abs(feature_norms - ideal_norm) / (ideal_norm + 1e-8)
        norm_stability = torch.exp(-torch.clamp(norm_diff, 0, 10))  # 限制指数范围

        # 2. 特征信息熵 - 添加稳定性保护
        feature_probs = F.softmax(torch.abs(features) + 1e-8, dim=-1)  # 防止零概率
        entropy = -torch.sum(feature_probs * torch.log(feature_probs + 1e-8), dim=-1)
        max_entropy = math.log(feature_dim)
        normalized_entropy = entropy / max_entropy

        # 3. 特征稀疏性质量
        sparsity_ratio = torch.mean((torch.abs(features) < 0.01).float(), dim=-1)
        optimal_sparsity = 0.1
        sparsity_quality = torch.exp(-torch.abs(sparsity_ratio - optimal_sparsity) / (optimal_sparsity + 1e-8))

        # 4. 特征方差
        feature_variance = torch.var(features, dim=-1)
        variance_quality = torch.sigmoid(feature_variance - 0.1)

        # === 添加：确保所有输出都是有限的 ===
        norm_stability = torch.clamp(norm_stability, 1e-6, 1.0)
        normalized_entropy = torch.clamp(normalized_entropy, 1e-6, 1.0)
        sparsity_quality = torch.clamp(sparsity_quality, 1e-6, 1.0)
        variance_quality = torch.clamp(variance_quality, 1e-6, 1.0)

        return {
            'norm_stability': norm_stability,
            'information_entropy': normalized_entropy,
            'sparsity_quality': sparsity_quality,
            'variance_quality': variance_quality
        }

    def compute_cross_modal_consistency(self, img_feat, text_feat):
        """=== 保留：计算跨模态一致性 ==="""
        if img_feat.dim() == 1:
            img_feat = img_feat.unsqueeze(0)
            text_feat = text_feat.unsqueeze(0)

        # 1. 余弦相似度
        cosine_similarity = F.cosine_similarity(img_feat, text_feat, dim=-1)
        semantic_alignment = (cosine_similarity + 1) / 2

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
        """=== 修改：移除torch.no_grad()，让网络可以学习 ==="""
        # 使用网络预测任务相关性（移除no_grad）
        img_relevance = self.task_relevance_predictor(img_feat).squeeze(-1)
        text_relevance = self.task_relevance_predictor(text_feat).squeeze(-1)

        # 计算跨模态协同性
        cross_modal_synergy = F.cosine_similarity(img_feat, text_feat, dim=-1)
        cross_modal_synergy = (cross_modal_synergy + 1) / 2

        return {
            'img_task_relevance': img_relevance,
            'text_task_relevance': text_relevance,
            'cross_modal_synergy': cross_modal_synergy
        }

    def compute_generation_confidence(self, original_feat, generated_feat):
        """=== 保留：计算生成特征的可信度 ==="""
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

    def evaluate_input_quality(self, input_features, is_generated=False):
        """
        === 新增：评估输入质量（区分原始和生成）===

        Args:
            input_features: 输入特征
            is_generated: 是否为生成的输入

        Returns:
            质量分数
        """
        if is_generated:
            # 对生成的输入，质量稍微打折
            base_quality = self.original_input_quality_evaluator(input_features).squeeze(-1)
            return base_quality * 0.8  # 生成质量打8折
        else:
            # 原始输入的质量评估
            return self.original_input_quality_evaluator(input_features).squeeze(-1)

    def forward(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat, missing_type,
                generation_info=None):
        """
        === 修改：主要质量评估接口 ===

        新逻辑：
        1. 对于完整模态，评估原始质量
        2. 对于生成模态，评估生成质量
        3. 基于完整特征进行跨模态评估
        """
        batch_size = image_feat.size(0)
        quality_scores = []

        # 计算任务相关性（基于所有特征，无论是否生成）
        task_relevance = self.compute_task_relevance(image_feat, text_feat)

        for i in range(batch_size):
            sample_quality = {}

            # === 图像质量评估 ===
            if missing_type[i] == 2:  # 缺失图像，使用生成的图像
                current_img_feat = enhanced_image_feat[i] if enhanced_image_feat is not None else image_feat[i]

                # 如果有生成信息，计算生成置信度
                if generation_info is not None and generation_info['generated_mask'][i, 0] > 0:
                    img_generation_confidence = self.compute_generation_confidence(
                        image_feat[i], current_img_feat
                    )
                else:
                    img_generation_confidence = torch.tensor(0.9).to(image_feat.device)
            else:  # 完整图像或缺失文本
                current_img_feat = image_feat[i]
                img_generation_confidence = torch.tensor(0.9, device=image_feat.device, dtype=image_feat.dtype)

            img_math_quality = self.compute_mathematical_quality(current_img_feat)

            sample_quality['image_quality'] = {
                'mathematical': img_math_quality,
                'task_relevance': task_relevance['img_task_relevance'][i],
                'generation_confidence': img_generation_confidence
            }

            # === 文本质量评估 ===
            if missing_type[i] == 1:  # 缺失文本，使用生成的文本
                current_text_feat = enhanced_text_feat[i] if enhanced_text_feat is not None else text_feat[i]

                # 如果有生成信息，计算生成置信度
                if generation_info is not None and generation_info['generated_mask'][i, 1] > 0:
                    text_generation_confidence = self.compute_generation_confidence(
                        text_feat[i], current_text_feat
                    )
                else:
                    text_generation_confidence = torch.tensor(0.9).to(text_feat.device)
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
        """=== 修改：确保返回可求导的张量 ==="""
        device = next(self.parameters()).device

        # 初始化为可求导的张量
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        count = 0

        for i, quality in enumerate(quality_scores):
            # 1. 任务相关性应该与实际任务性能相关
            if task_performance is not None and i < len(task_performance):
                img_relevance = quality['image_quality']['task_relevance']
                text_relevance = quality['text_quality']['task_relevance']

                # 确保所有值都是张量且在正确设备上
                if not torch.is_tensor(img_relevance):
                    img_relevance = torch.tensor(float(img_relevance), device=device, requires_grad=True)
                if not torch.is_tensor(text_relevance):
                    text_relevance = torch.tensor(float(text_relevance), device=device, requires_grad=True)
                if not torch.is_tensor(task_performance[i]):
                    task_perf = torch.tensor(float(task_performance[i]), device=device, requires_grad=True)
                else:
                    task_perf = task_performance[i].to(device)

                # 计算平均相关性
                avg_relevance = (img_relevance + text_relevance) / 2

                # MSE损失
                relevance_performance_loss = (avg_relevance - task_perf) ** 2
                total_loss = total_loss + relevance_performance_loss

            # 2. 生成置信度应该合理
            img_confidence = quality['image_quality']['generation_confidence']
            text_confidence = quality['text_quality']['generation_confidence']

            # 确保置信度是张量
            if not torch.is_tensor(img_confidence):
                img_confidence = torch.tensor(float(img_confidence), device=device, requires_grad=True)
            if not torch.is_tensor(text_confidence):
                text_confidence = torch.tensor(float(text_confidence), device=device, requires_grad=True)

            # 高置信度的正则化损失
            confidence_reg_loss = F.relu(torch.tensor(0.8, device=device) - img_confidence) + \
                                  F.relu(torch.tensor(0.8, device=device) - text_confidence)
            total_loss = total_loss + confidence_reg_loss

            count += 1

        # 确保返回正确的张量
        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)