
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpdatedQualityGuidedFusion(nn.Module):
    """
    更新的质量引导融合器 - 适配简化的质量评估
    """

    def __init__(self, hidden_size=512, fusion_strategy='adaptive_attention'):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_strategy = fusion_strategy

        # 质量权重计算网络
        self.quality_weight_calculator = nn.Sequential(
            nn.Linear(6, 32),  # 6个主要质量指标
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 输出图像和文本权重
            nn.Softmax(dim=-1)
        )

        # 任务相关性权重网络
        self.task_relevance_weight_net = nn.Sequential(
            nn.Linear(3, 16),  # img_relevance, text_relevance, synergy
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.Softmax(dim=-1)
        )

        # 特征增强网络（基于质量分数）
        self.image_enhancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

        self.text_enhancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

        # 跨模态注意力机制
        if fusion_strategy == 'adaptive_attention':
            self.cross_modal_attention = nn.MultiheadAttention(
                hidden_size, num_heads=8, dropout=0.1
            )

        # 自适应门控网络
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, hidden_size),  # +1 for confidence score
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # 最终融合网络
        self.final_fusion_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size * 2)
        )

    def extract_quality_features(self, quality_score):
        """
        从简化的质量分数中提取特征向量
        """

        # 安全提取函数
        def safe_extract_value(value):
            if torch.is_tensor(value):
                if value.dim() == 0:  # 0维tensor
                    return value.item()
                elif value.numel() == 1:  # 只有一个元素
                    return value.item()
                elif value.numel() > 0:  # 多个元素，取第一个
                    return value.flatten()[0].item()
                else:
                    return 0.5  # 默认值
            else:
                return float(value)

        # 提取图像质量特征
        img_math = quality_score['image_quality']['mathematical']
        img_norm_stability = safe_extract_value(img_math['norm_stability'])
        img_entropy = safe_extract_value(img_math['information_entropy'])
        img_task_relevance = safe_extract_value(quality_score['image_quality']['task_relevance'])

        # 提取文本质量特征
        text_math = quality_score['text_quality']['mathematical']
        text_norm_stability = safe_extract_value(text_math['norm_stability'])
        text_entropy = safe_extract_value(text_math['information_entropy'])
        text_task_relevance = safe_extract_value(quality_score['text_quality']['task_relevance'])

        # 构建质量特征向量
        quality_vector = torch.tensor([
            img_norm_stability,
            img_entropy,
            img_task_relevance,
            text_norm_stability,
            text_entropy,
            text_task_relevance
        ], device=next(self.parameters()).device, dtype=torch.float32)

        return quality_vector
    def compute_modality_weights(self, quality_scores):
        """
        计算模态权重
        """

        def safe_extract_value(value):
            if torch.is_tensor(value):
                if value.dim() == 0:  # 0维tensor
                    return value.item()
                elif value.numel() == 1:  # 只有一个元素
                    return value.item()
                elif value.numel() > 0:  # 多个元素，取第一个
                    return value.flatten()[0].item()
                else:
                    return 0.5  # 默认值
            else:
                return float(value)

        batch_size = len(quality_scores)
        all_weights = []

        for quality_score in quality_scores:
            # 提取质量特征
            quality_vector = self.extract_quality_features(quality_score)

            # 计算质量权重
            quality_weights = self.quality_weight_calculator(quality_vector.unsqueeze(0)).squeeze(0)

            # 提取任务相关性特征
            img_relevance = quality_score['image_quality']['task_relevance']
            text_relevance = quality_score['text_quality']['task_relevance']

            # 安全提取跨模态协同性
            cross_modal_consistency = quality_score['cross_modal_consistency']
            if 'network_consistency' in cross_modal_consistency:
                synergy = safe_extract_value(cross_modal_consistency['network_consistency'])
            else:
                synergy = torch.tensor(0.5).to(quality_weights.device)

            if torch.is_tensor(img_relevance) and img_relevance.dim() > 0:
                img_relevance = img_relevance.item()
            if torch.is_tensor(text_relevance) and text_relevance.dim() > 0:
                text_relevance = text_relevance.item()
            if torch.is_tensor(synergy) and synergy.dim() > 0:
                synergy = synergy.item()

            relevance_vector = torch.tensor([
                float(img_relevance),
                float(text_relevance),
                float(synergy)
            ]).to(quality_weights.device)

            # 计算任务相关性权重
            task_weights = self.task_relevance_weight_net(relevance_vector.unsqueeze(0)).squeeze(0)

            # 结合两种权重
            combined_weights = 0.6 * quality_weights + 0.4 * task_weights
            all_weights.append(combined_weights)

        return torch.stack(all_weights)  # [batch_size, 2]

    def quality_guided_enhancement(self, img_feat, text_feat, quality_scores):
        """
        基于质量分数进行特征增强
        """
        batch_size = img_feat.size(0)
        enhanced_img_list = []
        enhanced_text_list = []

        for i in range(batch_size):
            # 获取整体置信度
            overall_confidence = quality_scores[i]['overall_confidence']
            if torch.is_tensor(overall_confidence) and overall_confidence.dim() > 0:
                confidence_value = overall_confidence.item()
            else:
                confidence_value = float(overall_confidence)

            # 低置信度需要更多增强
            enhancement_strength = max(0.1, 1.0 - confidence_value)

            enhanced_img = img_feat[i:i + 1]
            enhanced_text = text_feat[i:i + 1]

            if enhancement_strength > 0.3:  # 需要增强
                img_enhancement = self.image_enhancer(enhanced_img)
                text_enhancement = self.text_enhancer(enhanced_text)

                enhanced_img = enhanced_img + enhancement_strength * img_enhancement
                enhanced_text = enhanced_text + enhancement_strength * text_enhancement

            enhanced_img_list.append(enhanced_img)
            enhanced_text_list.append(enhanced_text)

        return torch.cat(enhanced_img_list, dim=0), torch.cat(enhanced_text_list, dim=0)

    def adaptive_cross_modal_attention(self, img_feat, text_feat, quality_scores):
        """
        自适应跨模态注意力
        """
        if self.fusion_strategy != 'adaptive_attention':
            return img_feat, text_feat

        batch_size = img_feat.size(0)
        enhanced_img_list = []
        enhanced_text_list = []

        for i in range(batch_size):
            # 判断是否需要跨模态注意力
            cross_modal_consistency = quality_scores[i]['cross_modal_consistency']
            if 'network_consistency' in cross_modal_consistency:
                consistency_score = cross_modal_consistency['network_consistency']
            else:
                consistency_score = torch.tensor(0.5)

            if torch.is_tensor(consistency_score) and consistency_score.dim() > 0:
                consistency_value = consistency_score.item()
            else:
                consistency_value = float(consistency_score)

            if consistency_value > 0.5:  # 高一致性时进行注意力增强
                img_seq = img_feat[i:i + 1].unsqueeze(0)  # [1, 1, 512]
                text_seq = text_feat[i:i + 1].unsqueeze(0)  # [1, 1, 512]

                # 图像attend到文本
                img_attended, _ = self.cross_modal_attention(
                    query=img_seq, key=text_seq, value=text_seq
                )

                # 文本attend到图像
                text_attended, _ = self.cross_modal_attention(
                    query=text_seq, key=img_seq, value=img_seq
                )

                # 基于一致性分数混合
                img_result = consistency_value * img_attended.squeeze(0) + (1 - consistency_value) * img_feat[i:i + 1]
                text_result = consistency_value * text_attended.squeeze(0) + (1 - consistency_value) * text_feat[
                                                                                                       i:i + 1]

                enhanced_img_list.append(img_result)
                enhanced_text_list.append(text_result)
            else:
                enhanced_img_list.append(img_feat[i:i + 1])
                enhanced_text_list.append(text_feat[i:i + 1])

        return torch.cat(enhanced_img_list, dim=0), torch.cat(enhanced_text_list, dim=0)

    def adaptive_feature_gating(self, img_feat, text_feat, quality_scores):
        """
        自适应特征门控
        """
        batch_size = img_feat.size(0)
        gated_features = []

        for i in range(batch_size):
            # 获取整体置信度
            overall_confidence = quality_scores[i]['overall_confidence']
            if torch.is_tensor(overall_confidence) and overall_confidence.dim() > 0:
                confidence_value = overall_confidence.item()
            else:
                confidence_value = float(overall_confidence)

            # 构建门控输入
            combined_feat = torch.cat([
                img_feat[i], text_feat[i], torch.tensor([confidence_value]).to(img_feat.device)
            ])

            # 计算门控权重
            gate = self.adaptive_gate(combined_feat.unsqueeze(0)).squeeze(0)  # [512]

            # 应用门控
            gated_img = gate * img_feat[i]
            gated_text = (1 - gate) * text_feat[i]

            # 拼接特征
            gated_feature = torch.cat([gated_img, gated_text])  # [1024]
            gated_features.append(gated_feature.unsqueeze(0))

        return torch.cat(gated_features, dim=0)

    def forward(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat,
                quality_scores, missing_type):
        """
        主要融合接口

        Args:
            image_feat: [batch, 512] 原始图像特征
            text_feat: [batch, 512] 原始文本特征
            enhanced_image_feat: [batch, 512] 增强图像特征
            enhanced_text_feat: [batch, 512] 增强文本特征
            quality_scores: List[Dict] 质量分数
            missing_type: [batch] 缺失类型

        Returns:
            [batch, 1024] 融合后的特征
        """
        batch_size = image_feat.size(0)

        # 1. 根据缺失类型选择特征
        final_img_feat = image_feat.clone()
        final_text_feat = text_feat.clone()

        for i, miss_type in enumerate(missing_type):
            if miss_type == 1:  # 缺失文本，使用增强文本
                # 基于生成置信度决定使用程度
                gen_confidence = quality_scores[i]['text_quality']['generation_confidence']
                if torch.is_tensor(gen_confidence) and gen_confidence.dim() > 0:
                    confidence_value = gen_confidence.item()
                else:
                    confidence_value = float(gen_confidence)

                if confidence_value > 0.6:
                    final_text_feat[i] = enhanced_text_feat[i]
                else:
                    # 低置信度时混合使用
                    final_text_feat[i] = confidence_value * enhanced_text_feat[i] + (1 - confidence_value) * text_feat[
                        i]

            elif miss_type == 2:  # 缺失图像，使用增强图像
                gen_confidence = quality_scores[i]['image_quality']['generation_confidence']
                if torch.is_tensor(gen_confidence) and gen_confidence.dim() > 0:
                    confidence_value = gen_confidence.item()
                else:
                    confidence_value = float(gen_confidence)

                if confidence_value > 0.6:
                    final_img_feat[i] = enhanced_image_feat[i]
                else:
                    final_img_feat[i] = confidence_value * enhanced_image_feat[i] + (1 - confidence_value) * image_feat[
                        i]

        # 2. 计算模态权重
        modality_weights = self.compute_modality_weights(quality_scores)  # [batch, 2]

        # 3. 质量引导的特征增强
        enhanced_img, enhanced_text = self.quality_guided_enhancement(
            final_img_feat, final_text_feat, quality_scores
        )

        # 4. 自适应跨模态注意力
        attended_img, attended_text = self.adaptive_cross_modal_attention(
            enhanced_img, enhanced_text, quality_scores
        )

        # 5. 应用模态权重
        img_weights = modality_weights[:, 0:1]  # [batch, 1]
        text_weights = modality_weights[:, 1:2]  # [batch, 1]

        weighted_img = img_weights * attended_img
        weighted_text = text_weights * attended_text

        # 6. 自适应门控融合
        final_features = self.adaptive_feature_gating(
            weighted_img, weighted_text, quality_scores
        )

        # 7. 最终融合网络
        refined_features = self.final_fusion_net(final_features)

        return refined_features

    def get_fusion_weights_info(self, quality_scores):
        """
        获取融合权重信息用于调试
        """
        modality_weights = self.compute_modality_weights(quality_scores)

        weights_info = []
        for i, quality_score in enumerate(quality_scores):
            info = {
                'img_weight': modality_weights[i, 0].item(),
                'text_weight': modality_weights[i, 1].item(),
                'overall_confidence': float(quality_score['overall_confidence']),
                'img_task_relevance': float(quality_score['image_quality']['task_relevance']),
                'text_task_relevance': float(quality_score['text_quality']['task_relevance'])
            }
            weights_info.append(info)

        return weights_info