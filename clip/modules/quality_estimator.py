import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QualityEstimator(nn.Module):
    """多维度质量评估器"""

    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size

        # 单模态质量评估网络
        self.image_quality_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 4),  # 4个质量维度
            nn.Sigmoid()
        )

        self.text_quality_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 4),
            nn.Sigmoid()
        )

        # 跨模态一致性评估
        self.consistency_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 生成质量评估
        self.generation_confidence_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def evaluate_intrinsic_quality(self, features, modality='image'):
        """评估单模态内在质量"""
        if modality == 'image':
            return self.image_quality_net(features)
        else:
            return self.text_quality_net(features)

    def evaluate_cross_modal_consistency(self, image_feat, text_feat):
        """评估跨模态一致性"""
        combined = torch.cat([image_feat, text_feat], dim=-1)
        return self.consistency_net(combined)

    def evaluate_generation_confidence(self, original_feat, generated_feat):
        """评估生成特征的可信度"""
        combined = torch.cat([original_feat, generated_feat], dim=-1)
        return self.generation_confidence_net(combined)

    def forward(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat, missing_type):
        """
        Args:
            image_feat: [batch, 512] 原始图像特征
            text_feat: [batch, 512] 原始文本特征
            enhanced_image_feat: [batch, 512] 增强图像特征
            enhanced_text_feat: [batch, 512] 增强文本特征
            missing_type: [batch] 缺失类型

        Returns:
            List[Dict]: 每个样本的质量分数字典
        """
        batch_size = image_feat.size(0)
        quality_scores = []

        for i in range(batch_size):
            sample_quality = {}

            if missing_type[i] == 0:  # 完整模态
                # 原始特征质量
                sample_quality['image_intrinsic'] = self.evaluate_intrinsic_quality(
                    image_feat[i:i + 1], 'image')
                sample_quality['text_intrinsic'] = self.evaluate_intrinsic_quality(
                    text_feat[i:i + 1], 'text')

                # 跨模态一致性
                sample_quality['cross_modal_consistency'] = self.evaluate_cross_modal_consistency(
                    image_feat[i:i + 1], text_feat[i:i + 1])

                # 完整模态，生成置信度设为1
                sample_quality['generation_confidence'] = torch.ones(1, 1).to(image_feat.device)

            elif missing_type[i] == 1:  # 缺失文本
                # 图像质量保持
                sample_quality['image_intrinsic'] = self.evaluate_intrinsic_quality(
                    image_feat[i:i + 1], 'image')

                # 生成文本的置信度
                sample_quality['generation_confidence'] = self.evaluate_generation_confidence(
                    text_feat[i:i + 1], enhanced_text_feat[i:i + 1])

                # 生成文本质量 = 置信度 * 基准质量
                base_text_quality = self.evaluate_intrinsic_quality(enhanced_text_feat[i:i + 1], 'text')
                sample_quality['text_intrinsic'] = base_text_quality * sample_quality['generation_confidence'] * 0.8

                # 跨模态一致性 (原始图像 vs 增强文本)
                sample_quality['cross_modal_consistency'] = self.evaluate_cross_modal_consistency(
                    image_feat[i:i + 1], enhanced_text_feat[i:i + 1]) * 0.9

            elif missing_type[i] == 2:  # 缺失图像
                # 文本质量保持
                sample_quality['text_intrinsic'] = self.evaluate_intrinsic_quality(
                    text_feat[i:i + 1], 'text')

                # 生成图像的置信度
                sample_quality['generation_confidence'] = self.evaluate_generation_confidence(
                    image_feat[i:i + 1], enhanced_image_feat[i:i + 1])

                # 生成图像质量
                base_image_quality = self.evaluate_intrinsic_quality(enhanced_image_feat[i:i + 1], 'image')
                sample_quality['image_intrinsic'] = base_image_quality * sample_quality['generation_confidence'] * 0.8

                # 跨模态一致性 (增强图像 vs 原始文本)
                sample_quality['cross_modal_consistency'] = self.evaluate_cross_modal_consistency(
                    enhanced_image_feat[i:i + 1], text_feat[i:i + 1]) * 0.9

            quality_scores.append(sample_quality)

        return quality_scores


class ObjectiveQualityAssessor(nn.Module):
    """
    客观质量评估器 - 第一步实现
    核心思想：用6个客观指标替换原有的MLP+人工标签方式
    """

    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入维度投影器 - 统一处理不同维度输入
        self.single_modal_proj = nn.Linear(hidden_size, hidden_size)  # 512 -> 512
        self.dual_modal_proj = nn.Linear(hidden_size * 2, hidden_size)  # 1024 -> 512

        # 1. InfoNCE对比质量网络 (单模态输入)
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )

        # 2. 预测不确定性网络 (双模态输入)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # 3. 表示质量网络 (单模态输入)
        self.representation_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # 4. CKA一致性计算网络 (处理两个单模态特征)
        self.consistency_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        # 温度参数用于InfoNCE计算
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def compute_infonce_quality(self, anchor, positive, negative_pool):
        """
        计算InfoNCE质量分数
        anchor: 当前特征
        positive: 正样本特征
        negative_pool: 负样本池
        """
        # 投影到对比空间
        anchor_proj = F.normalize(self.contrastive_head(anchor), dim=-1)
        positive_proj = F.normalize(self.contrastive_head(positive), dim=-1)

        if negative_pool.size(0) > 0:
            negative_proj = F.normalize(self.contrastive_head(negative_pool), dim=-1)

            # 计算相似度
            pos_sim = torch.sum(anchor_proj * positive_proj, dim=-1) / self.temperature
            neg_sim = torch.matmul(anchor_proj, negative_proj.transpose(-2, -1)) / self.temperature

            # InfoNCE loss作为质量指标（取负值，越小越好转换为越大越好）
            infonce_loss = -torch.log(
                torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim), dim=-1))
            )

            # 转换为质量分数（0-1之间，越大越好）
            quality_score = torch.sigmoid(-infonce_loss)
        else:
            # 如果没有负样本，直接用正样本相似度
            quality_score = torch.sigmoid(torch.sum(anchor_proj * positive_proj, dim=-1))

        return quality_score.clamp(0.1, 0.9)  # 避免极值

    def compute_cka_consistency(self, features_a, features_b):
        """
        计算CKA (Centered Kernel Alignment) 一致性
        输入: 两个单模态特征 [512, 512]
        """
        # 确保输入是[512]维度
        features_a = self.single_modal_proj(features_a)
        features_b = self.single_modal_proj(features_b)

        # 中心化
        features_a_centered = features_a - features_a.mean(dim=-1, keepdim=True)
        features_b_centered = features_b - features_b.mean(dim=-1, keepdim=True)

        # 计算CKA
        numerator = torch.sum(features_a_centered * features_b_centered) ** 2
        denominator = torch.sum(features_a_centered ** 2) * torch.sum(features_b_centered ** 2)

        cka_score = numerator / (denominator + 1e-8)
        return torch.clamp(cka_score, 0.0, 1.0)

    def compute_prediction_uncertainty(self, combined_features):
        """
        计算预测不确定性（基于熵）
        输入: 拼接特征 [1024] -> 投影到 [512]
        """
        # 投影到统一维度
        if combined_features.dim() == 1:
            combined_features = combined_features.unsqueeze(0)

        if combined_features.size(-1) == self.hidden_size * 2:
            features = self.dual_modal_proj(combined_features)
        else:
            features = self.single_modal_proj(combined_features)

        # 通过网络预测不确定性
        uncertainty = self.uncertainty_head(features)

        # 计算特征熵作为补充指标
        probs = F.softmax(features, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        normalized_entropy = entropy / np.log(features.size(-1))  # 归一化到[0,1]

        # 结合网络预测和特征熵
        combined_uncertainty = 0.7 * uncertainty.squeeze() + 0.3 * normalized_entropy

        # 不确定性越低，质量越高
        return 1.0 - combined_uncertainty

    def approximate_shapley_value(self, image_feat, text_feat, task_performance_fn=None):
        """
        近似计算Shapley值（模态贡献度）
        输入: 两个单模态特征 [512, 512]
        """
        # 确保输入维度正确
        image_feat = self.single_modal_proj(image_feat)
        text_feat = self.single_modal_proj(text_feat)

        with torch.no_grad():
            # 单模态性能
            img_only_score = self.evaluate_single_modal_quality(image_feat)
            text_only_score = self.evaluate_single_modal_quality(text_feat)

            # 双模态性能 - 先拼接再投影
            combined_feat = torch.cat([image_feat, text_feat], dim=-1)
            combined_feat_proj = self.dual_modal_proj(combined_feat)
            combined_score = self.representation_head(combined_feat_proj).squeeze()

            # 近似Shapley值：边际贡献
            img_contribution = combined_score - text_only_score
            text_contribution = combined_score - img_only_score

            # 归一化
            total_contribution = img_contribution + text_contribution
            if total_contribution > 0:
                img_shapley = img_contribution / total_contribution
            else:
                img_shapley = torch.tensor(0.5).to(image_feat.device)

        return torch.clamp(img_shapley, 0.1, 0.9)

    def evaluate_single_modal_quality(self, features):
        """
        评估单模态表示质量
        输入: 单模态特征 [512]
        """
        # 确保输入维度正确
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = self.single_modal_proj(features)
        return self.representation_head(features).squeeze()

    def evaluate_representation_quality(self, features):
        """
        评估特征表示质量
        输入: 单模态特征 [512]
        """
        # 确保输入维度正确
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = self.single_modal_proj(features)

        # 特征范数作为表示质量的指标之一
        feat_norm = torch.norm(features, dim=-1)
        norm_score = torch.sigmoid(feat_norm - 1.0)  # 假设理想范数约为1

        # 网络预测的表示质量
        network_score = self.representation_head(features).squeeze()

        # 结合两个指标
        return 0.6 * network_score + 0.4 * norm_score

    def evaluate_generation_confidence(self, original_feat, generated_feat):
        """
        评估生成特征的可信度
        输入: 两个单模态特征 [512, 512]
        """
        # 确保输入维度正确
        original_feat = self.single_modal_proj(original_feat)
        generated_feat = self.single_modal_proj(generated_feat)

        # 计算原始特征和生成特征的相似度
        cosine_sim = F.cosine_similarity(original_feat, generated_feat, dim=-1)

        # 计算特征距离
        l2_distance = torch.norm(original_feat - generated_feat, dim=-1)
        distance_score = torch.sigmoid(-l2_distance + 1.0)  # 距离越小，分数越高

        # 结合相似度和距离
        confidence = 0.7 * cosine_sim + 0.3 * distance_score
        return torch.clamp(confidence, 0.1, 0.9)

    def forward(self, image_features, text_features, enhanced_image_features,
                enhanced_text_features, missing_type):
        """
        主要的质量评估接口 - 重新设计为分模态质量评估

        Returns:
            quality_scores: List[Dict] 每个样本的质量分数
            格式: {
                'image_quality': {
                    'intrinsic_quality': float,      # 图像内在质量
                    'representation_quality': float,  # 图像表示质量
                    'generation_confidence': float,   # 图像生成置信度
                    'task_contribution': float        # 图像任务贡献度
                },
                'text_quality': {
                    'intrinsic_quality': float,       # 文本内在质量
                    'representation_quality': float,  # 文本表示质量
                    'generation_confidence': float,   # 文本生成置信度
                    'task_contribution': float        # 文本任务贡献度
                },
                'cross_modal_consistency': float,     # 跨模态一致性
                'overall_uncertainty': float          # 整体不确定性
            }
        """
        batch_size = image_features.size(0)
        quality_scores = []

        for i in range(batch_size):
            sample_quality = {}

            # === 图像质量评估 ===
            if missing_type[i] == 2:  # 缺失图像，使用生成的图像
                img_feat = enhanced_image_features[i]
                img_generation_confidence = self.evaluate_generation_confidence(
                    image_features[i], enhanced_image_features[i]
                )
            else:  # 完整图像或缺失文本
                img_feat = image_features[i]
                img_generation_confidence = torch.tensor(0.9).to(image_features.device)

            img_intrinsic = self.compute_infonce_quality(
                img_feat,
                text_features[i] if missing_type[i] != 1 else enhanced_text_features[i],
                image_features[torch.randperm(batch_size)[:min(4, batch_size - 1)]]
            )
            img_representation = self.evaluate_representation_quality(img_feat)

            # === 文本质量评估 ===
            if missing_type[i] == 1:  # 缺失文本，使用生成的文本
                text_feat = enhanced_text_features[i]
                text_generation_confidence = self.evaluate_generation_confidence(
                    text_features[i], enhanced_text_features[i]
                )
            else:  # 完整文本或缺失图像
                text_feat = text_features[i]
                text_generation_confidence = torch.tensor(0.9).to(text_features.device)

            text_intrinsic = self.compute_infonce_quality(
                text_feat,
                img_feat,
                text_features[torch.randperm(batch_size)[:min(4, batch_size - 1)]]
            )
            text_representation = self.evaluate_representation_quality(text_feat)

            # === 任务贡献度评估 (Shapley值) ===
            img_task_contrib = self.approximate_shapley_value(img_feat, text_feat)
            text_task_contrib = 1.0 - img_task_contrib  # 互补关系

            # === 跨模态指标 ===
            cross_modal_consistency = self.compute_cka_consistency(img_feat, text_feat)

            overall_uncertainty = self.compute_prediction_uncertainty(
                torch.cat([img_feat, text_feat], dim=-1)
            )

            # 根据缺失类型调整质量分数
            if missing_type[i] == 1:  # 缺失文本
                # 文本质量需要打折
                text_intrinsic *= 0.7
                text_representation *= 0.7
                cross_modal_consistency *= 0.8
            elif missing_type[i] == 2:  # 缺失图像
                # 图像质量需要打折
                img_intrinsic *= 0.7
                img_representation *= 0.7
                cross_modal_consistency *= 0.8

            sample_quality = {
                'image_quality': {
                    'intrinsic_quality': img_intrinsic,
                    'representation_quality': img_representation,
                    'generation_confidence': img_generation_confidence,
                    'task_contribution': img_task_contrib
                },
                'text_quality': {
                    'intrinsic_quality': text_intrinsic,
                    'representation_quality': text_representation,
                    'generation_confidence': text_generation_confidence,
                    'task_contribution': text_task_contrib
                },
                'cross_modal_consistency': cross_modal_consistency,
                'overall_uncertainty': overall_uncertainty
            }

            quality_scores.append(sample_quality)

        return quality_scores