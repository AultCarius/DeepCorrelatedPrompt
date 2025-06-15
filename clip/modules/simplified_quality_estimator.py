# clip/modules/simplified_quality_estimator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SimplifiedQualityEstimator(nn.Module):
    """
    简化的质量评估器 - 基于embedding层特征进行质量评估
    核心功能：
    1. 数学质量指标计算（范数稳定性、信息熵等）
    2. 任务相关性预测
    3. 不确定性评估
    4. 生成质量评估
    """

    def __init__(self, image_embed_dim=768, text_embed_dim=512):
        super().__init__()
        self.image_embed_dim = image_embed_dim
        self.text_embed_dim = text_embed_dim

        # === 任务相关性预测网络 ===
        self.img_task_relevance_net = nn.Sequential(
            nn.Linear(image_embed_dim, image_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(image_embed_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.text_task_relevance_net = nn.Sequential(
            nn.Linear(text_embed_dim, text_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(text_embed_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # === 不确定性评估网络 ===
        self.img_uncertainty_net = nn.Sequential(
            nn.Linear(image_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.text_uncertainty_net = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # === 跨模态一致性评估网络 ===
        self.consistency_net = nn.Sequential(
            nn.Linear(image_embed_dim + text_embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # === 生成质量评估网络 - 分别为图像和文本 ===
        self.img_generation_quality_net = nn.Sequential(
            nn.Linear(image_embed_dim * 2, 512),  # 原始+生成 = 768*2
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.text_generation_quality_net = nn.Sequential(
            nn.Linear(text_embed_dim * 2, 512),  # 原始+生成 = 512*2
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def compute_mathematical_quality(self, embeddings):
        """
        计算基于embedding的数学质量指标

        Args:
            embeddings: [batch, embed_dim] 或 [batch, seq_len, embed_dim]

        Returns:
            Dict 包含各种数学质量指标
        """
        if embeddings.dim() == 3:  # [batch, seq_len, embed_dim]
            # 对序列维度进行平均池化
            embeddings = embeddings.mean(dim=1)  # [batch, embed_dim]

        batch_size, embed_dim = embeddings.shape

        # 1. 特征范数稳定性
        feature_norms = torch.norm(embeddings, dim=-1)  # [batch]
        ideal_norm = math.sqrt(embed_dim)
        norm_diff = torch.abs(feature_norms - ideal_norm) / (ideal_norm + 1e-8)
        norm_stability = torch.exp(-torch.clamp(norm_diff, 0, 10))

        # 2. 信息熵
        # 将embedding值归一化为概率分布
        abs_embeddings = torch.abs(embeddings) + 1e-8
        feature_probs = F.softmax(abs_embeddings, dim=-1)  # [batch, embed_dim]
        entropy = -torch.sum(feature_probs * torch.log(feature_probs + 1e-8), dim=-1)  # [batch]
        max_entropy = math.log(embed_dim)
        normalized_entropy = entropy / max_entropy

        # 3. 特征稀疏性
        sparsity_ratio = torch.mean((torch.abs(embeddings) < 0.01).float(), dim=-1)  # [batch]
        optimal_sparsity = 0.1
        sparsity_quality = torch.exp(-torch.abs(sparsity_ratio - optimal_sparsity) / (optimal_sparsity + 1e-8))

        # 4. 特征方差（质量）
        feature_variance = torch.var(embeddings, dim=-1)  # [batch]
        variance_quality = torch.sigmoid(feature_variance - 0.1)

        # 5. 特征集中度
        max_values = torch.max(torch.abs(embeddings), dim=-1)[0]  # [batch]
        mean_values = torch.mean(torch.abs(embeddings), dim=-1)  # [batch]
        concentration = max_values / (mean_values + 1e-8)
        concentration_quality = torch.sigmoid(5.0 - concentration)  # 集中度过高不好

        # 确保所有输出都是有限的
        norm_stability = torch.clamp(norm_stability, 1e-6, 1.0)
        normalized_entropy = torch.clamp(normalized_entropy, 1e-6, 1.0)
        sparsity_quality = torch.clamp(sparsity_quality, 1e-6, 1.0)
        variance_quality = torch.clamp(variance_quality, 1e-6, 1.0)
        concentration_quality = torch.clamp(concentration_quality, 1e-6, 1.0)

        return {
            'norm_stability': norm_stability,  # [batch]
            'information_entropy': normalized_entropy,  # [batch]
            'sparsity_quality': sparsity_quality,  # [batch]
            'variance_quality': variance_quality,  # [batch]
            'concentration_quality': concentration_quality  # [batch]
        }

    def compute_task_relevance(self, img_embeddings, text_embeddings):
        """
        计算任务相关性

        Args:
            img_embeddings: [batch, img_embed_dim]
            text_embeddings: [batch, text_embed_dim]

        Returns:
            Dict 包含任务相关性评估
        """
        # 处理输入维度
        if img_embeddings.dim() == 3:
            img_embeddings = img_embeddings.mean(dim=1)
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.mean(dim=1)

        # 使用网络预测任务相关性
        img_task_relevance = self.img_task_relevance_net(img_embeddings).squeeze(-1)  # [batch]
        text_task_relevance = self.text_task_relevance_net(text_embeddings).squeeze(-1)  # [batch]

        # 计算跨模态协同性 - 需要维度对齐
        img_norm = F.normalize(img_embeddings, p=2, dim=-1)  # [batch, 768]
        text_norm = F.normalize(text_embeddings, p=2, dim=-1)  # [batch, 512]

        # 添加维度对齐的投影层
        if not hasattr(self, 'cross_modal_projection'):
            self.cross_modal_projection = nn.Linear(self.text_embed_dim, self.image_embed_dim).to(img_embeddings.device)

        # 将文本特征投影到图像特征空间
        text_projected = self.cross_modal_projection(text_norm)  # [batch, 768]
        text_projected_norm = F.normalize(text_projected, p=2, dim=-1)  # [batch, 768]

        # 现在可以计算余弦相似度
        cross_modal_synergy = torch.sum(img_norm * text_projected_norm, dim=-1)  # [batch]
        cross_modal_synergy = (cross_modal_synergy + 1) / 2  # 归一化到[0,1]

        return {
            'img_task_relevance': img_task_relevance,  # [batch]
            'text_task_relevance': text_task_relevance,  # [batch]
            'cross_modal_synergy': cross_modal_synergy  # [batch]
        }

    def compute_uncertainty(self, img_embeddings, text_embeddings):
        """
        计算不确定性评估

        Args:
            img_embeddings: [batch, img_embed_dim]
            text_embeddings: [batch, text_embed_dim]

        Returns:
            Dict 包含不确定性评估
        """
        # 处理输入维度
        if img_embeddings.dim() == 3:
            img_embeddings = img_embeddings.mean(dim=1)
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.mean(dim=1)

        # 网络预测不确定性
        img_uncertainty = self.img_uncertainty_net(img_embeddings).squeeze(-1)  # [batch]
        text_uncertainty = self.text_uncertainty_net(text_embeddings).squeeze(-1)  # [batch]

        # 基于方差的不确定性
        img_variance_uncertainty = 1.0 - torch.sigmoid(torch.var(img_embeddings, dim=-1))
        text_variance_uncertainty = 1.0 - torch.sigmoid(torch.var(text_embeddings, dim=-1))

        # 综合不确定性
        img_overall_uncertainty = (img_uncertainty + img_variance_uncertainty) / 2
        text_overall_uncertainty = (text_uncertainty + text_variance_uncertainty) / 2

        return {
            'img_uncertainty': img_overall_uncertainty,  # [batch]
            'text_uncertainty': text_overall_uncertainty,  # [batch]
            'prediction_uncertainty': img_uncertainty,  # [batch] 网络预测的
            'variance_uncertainty': img_variance_uncertainty  # [batch] 基于方差的
        }

    def compute_cross_modal_consistency(self, img_embeddings, text_embeddings):
        """
        计算跨模态一致性

        Args:
            img_embeddings: [batch, img_embed_dim]
            text_embeddings: [batch, text_embed_dim]

        Returns:
            Dict 包含跨模态一致性评估
        """
        # 处理输入维度
        if img_embeddings.dim() == 3:
            img_embeddings = img_embeddings.mean(dim=1)
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.mean(dim=1)

        # 1. 余弦相似度 - 需要维度对齐
        img_norm = F.normalize(img_embeddings, p=2, dim=-1)  # [batch, 768]
        text_norm = F.normalize(text_embeddings, p=2, dim=-1)  # [batch, 512]

        # 使用之前创建的投影层或创建新的
        if not hasattr(self, 'consistency_projection'):
            self.consistency_projection = nn.Linear(self.text_embed_dim, self.image_embed_dim).to(img_embeddings.device)

        text_projected = self.consistency_projection(text_norm)  # [batch, 768]
        text_projected_norm = F.normalize(text_projected, p=2, dim=-1)  # [batch, 768]

        cosine_similarity = torch.sum(img_norm * text_projected_norm, dim=-1)  # [batch]
        semantic_alignment = (cosine_similarity + 1) / 2  # 归一化到[0,1]

        # 2. 特征距离质量 - 在对齐后的空间计算
        feature_distance = torch.norm(img_embeddings - text_projected, dim=-1)  # [batch]
        max_distance = torch.norm(img_embeddings, dim=-1) + torch.norm(text_projected, dim=-1)
        normalized_distance = feature_distance / (max_distance + 1e-8)
        distance_quality = 1.0 - normalized_distance

        # 3. 网络评估一致性
        combined_embeddings = torch.cat([img_embeddings, text_embeddings], dim=-1)  # [batch, 768+512]
        network_consistency = self.consistency_net(combined_embeddings).squeeze(-1)  # [batch]

        # 4. 综合一致性
        overall_consistency = (semantic_alignment + distance_quality + network_consistency) / 3

        return {
            'semantic_alignment': semantic_alignment,  # [batch]
            'distance_quality': distance_quality,  # [batch]
            'network_consistency': network_consistency,  # [batch]
            'overall_consistency': overall_consistency  # [batch]
        }

    def compute_generation_confidence(self, original_embeddings, generated_embeddings, modality):
        """
        计算生成质量置信度

        Args:
            original_embeddings: [batch, embed_dim] 原始embedding
            generated_embeddings: [batch, embed_dim] 生成的embedding
            modality: str 'image' 或 'text'

        Returns:
            Tensor [batch] 生成置信度
        """
        # 处理输入维度
        if original_embeddings.dim() == 3:
            original_embeddings = original_embeddings.mean(dim=1)
        if generated_embeddings.dim() == 3:
            generated_embeddings = generated_embeddings.mean(dim=1)

        # 确保两个embedding维度一致
        assert original_embeddings.shape == generated_embeddings.shape, \
            f"Original shape {original_embeddings.shape} != Generated shape {generated_embeddings.shape}"

        # 1. 余弦相似度
        orig_norm = F.normalize(original_embeddings, p=2, dim=-1)
        gen_norm = F.normalize(generated_embeddings, p=2, dim=-1)
        cosine_sim = torch.sum(orig_norm * gen_norm, dim=-1)  # [batch]
        cosine_sim = (cosine_sim + 1) / 2  # 归一化到[0,1]

        # 2. L2距离质量
        l2_distance = torch.norm(original_embeddings - generated_embeddings, dim=-1)  # [batch]
        max_norm = torch.norm(original_embeddings, dim=-1) + torch.norm(generated_embeddings, dim=-1)
        normalized_distance = l2_distance / (max_norm + 1e-8)
        distance_quality = 1.0 - normalized_distance

        # 3. 网络评估生成质量 - 根据模态选择对应的网络
        combined = torch.cat([original_embeddings, generated_embeddings], dim=-1)  # [batch, embed_dim*2]

        if modality == 'image':
            # 图像模态：[batch, 768*2] -> [batch, 1]
            network_quality = self.img_generation_quality_net(combined).squeeze(-1)
        elif modality == 'text':
            # 文本模态：[batch, 512*2] -> [batch, 1]
            network_quality = self.text_generation_quality_net(combined).squeeze(-1)
        else:
            raise ValueError(f"Unknown modality: {modality}")

        # 4. 综合生成置信度
        generation_confidence = (0.4 * cosine_sim + 0.3 * distance_quality + 0.3 * network_quality)
        generation_confidence = torch.clamp(generation_confidence, 0.1, 0.95)

        return generation_confidence

    def forward(self, img_embeddings, text_embeddings, enhanced_img_embeddings, enhanced_text_embeddings,
                missing_type, generation_info=None):
        """
        主要质量评估接口

        Args:
            img_embeddings: [batch, img_embed_dim] 图像embedding（来自conv1+pos_emb）
            text_embeddings: [batch, text_embed_dim] 文本embedding（来自token_emb+pos_emb）
            enhanced_img_embeddings: [batch, img_embed_dim] 增强后的图像embedding
            enhanced_text_embeddings: [batch, text_embed_dim] 增强后的文本embedding
            missing_type: [batch] 缺失类型
            generation_info: Dict 生成信息

        Returns:
            List[Dict] 每个样本的质量评估结果
        """
        batch_size = img_embeddings.size(0)
        quality_scores = []

        # 处理输入维度统一
        if img_embeddings.dim() == 3:
            img_embeddings = img_embeddings.mean(dim=1)
        if text_embeddings.dim() == 3:
            text_embeddings = text_embeddings.mean(dim=1)
        if enhanced_img_embeddings.dim() == 3:
            enhanced_img_embeddings = enhanced_img_embeddings.mean(dim=1)
        if enhanced_text_embeddings.dim() == 3:
            enhanced_text_embeddings = enhanced_text_embeddings.mean(dim=1)

        # 计算全局质量指标
        task_relevance = self.compute_task_relevance(img_embeddings, text_embeddings)
        uncertainty = self.compute_uncertainty(img_embeddings, text_embeddings)
        cross_modal_consistency = self.compute_cross_modal_consistency(img_embeddings, text_embeddings)

        for i in range(batch_size):
            sample_quality = {}

            # === 图像质量评估 ===
            current_img_embed = enhanced_img_embeddings[i:i + 1] if missing_type[i] == 2 else img_embeddings[i:i + 1]
            img_math_quality = self.compute_mathematical_quality(current_img_embed)

            # 生成置信度
            if missing_type[i] == 2:  # 缺失图像，使用生成的
                img_generation_confidence = self.compute_generation_confidence(
                    img_embeddings[i:i + 1], enhanced_img_embeddings[i:i + 1], 'image'
                )
            else:
                img_generation_confidence = torch.tensor(0.95, device=img_embeddings.device)

            sample_quality['image_quality'] = {
                'mathematical': {
                    'norm_stability': img_math_quality['norm_stability'][0],
                    'information_entropy': img_math_quality['information_entropy'][0],
                    'sparsity_quality': img_math_quality['sparsity_quality'][0],
                    'variance_quality': img_math_quality['variance_quality'][0],
                    'concentration_quality': img_math_quality['concentration_quality'][0]
                },
                'task_relevance': task_relevance['img_task_relevance'][i],
                'uncertainty': uncertainty['img_uncertainty'][i],
                'generation_confidence': img_generation_confidence[
                    0] if img_generation_confidence.dim() > 0 else img_generation_confidence
            }

            # === 文本质量评估 ===
            current_text_embed = enhanced_text_embeddings[i:i + 1] if missing_type[i] == 1 else text_embeddings[i:i + 1]
            text_math_quality = self.compute_mathematical_quality(current_text_embed)

            # 生成置信度
            if missing_type[i] == 1:  # 缺失文本，使用生成的
                text_generation_confidence = self.compute_generation_confidence(
                    text_embeddings[i:i + 1], enhanced_text_embeddings[i:i + 1], 'text'
                )
            else:
                text_generation_confidence = torch.tensor(0.95, device=text_embeddings.device)

            sample_quality['text_quality'] = {
                'mathematical': {
                    'norm_stability': text_math_quality['norm_stability'][0],
                    'information_entropy': text_math_quality['information_entropy'][0],
                    'sparsity_quality': text_math_quality['sparsity_quality'][0],
                    'variance_quality': text_math_quality['variance_quality'][0],
                    'concentration_quality': text_math_quality['concentration_quality'][0]
                },
                'task_relevance': task_relevance['text_task_relevance'][i],
                'uncertainty': uncertainty['text_uncertainty'][i],
                'generation_confidence': text_generation_confidence[
                    0] if text_generation_confidence.dim() > 0 else text_generation_confidence
            }

            # === 跨模态质量评估 ===
            sample_quality['cross_modal_consistency'] = {
                'semantic_alignment': cross_modal_consistency['semantic_alignment'][i],
                'distance_quality': cross_modal_consistency['distance_quality'][i],
                'network_consistency': cross_modal_consistency['network_consistency'][i],
                'overall_consistency': cross_modal_consistency['overall_consistency'][i]
            }

            # === 整体置信度计算 ===
            img_overall = (
                    sample_quality['image_quality']['mathematical']['norm_stability'] * 0.2 +
                    sample_quality['image_quality']['mathematical']['information_entropy'] * 0.2 +
                    sample_quality['image_quality']['task_relevance'] * 0.3 +
                    (1.0 - sample_quality['image_quality']['uncertainty']) * 0.3
            )

            text_overall = (
                    sample_quality['text_quality']['mathematical']['norm_stability'] * 0.2 +
                    sample_quality['text_quality']['mathematical']['information_entropy'] * 0.2 +
                    sample_quality['text_quality']['task_relevance'] * 0.3 +
                    (1.0 - sample_quality['text_quality']['uncertainty']) * 0.3
            )

            cross_modal_overall = sample_quality['cross_modal_consistency']['overall_consistency']

            overall_confidence = (img_overall + text_overall + cross_modal_overall) / 3
            sample_quality['overall_confidence'] = overall_confidence

            quality_scores.append(sample_quality)

        return quality_scores

    def compute_quality_loss(self, quality_scores, task_performance=None):
        """
        计算质量评估的训练损失

        Args:
            quality_scores: List[Dict] 质量分数
            task_performance: Tensor [batch] 实际任务性能（可选）

        Returns:
            Tensor 质量损失
        """
        if not quality_scores:
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)

        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        count = 0

        for i, quality in enumerate(quality_scores):
            # 1. 任务相关性损失：与实际性能的一致性
            if task_performance is not None and i < len(task_performance):
                img_relevance = quality['image_quality']['task_relevance']
                text_relevance = quality['text_quality']['task_relevance']

                # 确保是tensor
                if not torch.is_tensor(img_relevance):
                    img_relevance = torch.tensor(float(img_relevance), device=device, requires_grad=True)
                if not torch.is_tensor(text_relevance):
                    text_relevance = torch.tensor(float(text_relevance), device=device, requires_grad=True)
                if not torch.is_tensor(task_performance[i]):
                    task_perf = torch.tensor(float(task_performance[i]), device=device)
                else:
                    task_perf = task_performance[i].to(device)

                # 平均相关性应该与任务性能相关
                avg_relevance = (img_relevance + text_relevance) / 2
                relevance_loss = F.mse_loss(avg_relevance, task_perf)
                total_loss = total_loss + relevance_loss

            # 2. 不确定性损失：不确定性不应该过高
            img_uncertainty = quality['image_quality']['uncertainty']
            text_uncertainty = quality['text_quality']['uncertainty']

            if not torch.is_tensor(img_uncertainty):
                img_uncertainty = torch.tensor(float(img_uncertainty), device=device, requires_grad=True)
            if not torch.is_tensor(text_uncertainty):
                text_uncertainty = torch.tensor(float(text_uncertainty), device=device, requires_grad=True)

            # 不确定性不应该太高
            uncertainty_penalty = F.relu(img_uncertainty - 0.7) + F.relu(text_uncertainty - 0.7)
            total_loss = total_loss + uncertainty_penalty

            # 3. 一致性损失：跨模态应该保持一致
            consistency = quality['cross_modal_consistency']['overall_consistency']
            if not torch.is_tensor(consistency):
                consistency = torch.tensor(float(consistency), device=device, requires_grad=True)

            # 一致性应该高
            consistency_loss = F.mse_loss(consistency, torch.ones_like(consistency))
            total_loss = total_loss + consistency_loss

            count += 1

        if count > 0:
            return total_loss / count
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)