# clip/modules/quality_aware_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityAwareFusion(nn.Module):
    """基于新质量评估的质量感知融合器"""

    def __init__(self, hidden_size=512, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 质量权重计算网络
        self.quality_weight_net = nn.Sequential(
            nn.Linear(4, 16),  # 4个综合质量指标
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),  # 输出图像和文本的权重
            nn.Softmax(dim=-1)
        )

        # 任务相关性权重网络
        self.task_relevance_weight_net = nn.Sequential(
            nn.Linear(3, 8),  # img_relevance, text_relevance, synergy
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 2),
            nn.Softmax(dim=-1)
        )

        # 特征增强网络
        self.img_enhancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        self.text_enhancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        # 跨模态交互网络
        self.cross_modal_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=dropout
        )

        # 自适应门控网络
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def compute_quality_weights(self, quality_scores):
        """基于质量分数计算融合权重 - 修复张量访问"""

        # 安全地提取质量分数
        def safe_extract_value(tensor_or_value):
            if isinstance(tensor_or_value, torch.Tensor):
                if tensor_or_value.dim() == 0:
                    return tensor_or_value.item()
                elif tensor_or_value.numel() > 0:
                    return tensor_or_value[0].item() if tensor_or_value.dim() > 0 else tensor_or_value.item()
                else:
                    return 0.5  # 默认值
            else:
                return float(tensor_or_value)

        quality_vector = torch.tensor([
            safe_extract_value(quality_scores['overall_img_quality']),
            safe_extract_value(quality_scores['overall_text_quality']),
            safe_extract_value(quality_scores['overall_cross_modal_quality']),
            safe_extract_value(quality_scores['overall_confidence'])
        ]).to(next(self.parameters()).device)

        quality_weights = self.quality_weight_net(quality_vector.unsqueeze(0))
        return quality_weights  # [1, 2] (img_weight, text_weight)

    def compute_task_relevance_weights(self, task_relevance):
        """基于任务相关性计算权重 - 修复张量访问"""

        def safe_extract_value(tensor_or_value):
            if isinstance(tensor_or_value, torch.Tensor):
                if tensor_or_value.dim() == 0:
                    return tensor_or_value.item()
                elif tensor_or_value.numel() > 0:
                    return tensor_or_value[0].item() if tensor_or_value.dim() > 0 else tensor_or_value.item()
                else:
                    return 0.5  # 默认值
            else:
                return float(tensor_or_value)

        if 'img_task_relevance' in task_relevance:
            relevance_vector = torch.tensor([
                safe_extract_value(task_relevance['img_task_relevance']),
                safe_extract_value(task_relevance['text_task_relevance']),
                safe_extract_value(task_relevance['cross_modal_synergy'])
            ]).to(next(self.parameters()).device)
        else:
            # 兼容梯度模式
            relevance_vector = torch.tensor([
                safe_extract_value(task_relevance['img_gradient_magnitude']),
                safe_extract_value(task_relevance['text_gradient_magnitude']),
                safe_extract_value(task_relevance['cross_modal_synergy'])
            ]).to(next(self.parameters()).device)

        task_weights = self.task_relevance_weight_net(relevance_vector.unsqueeze(0))
        return task_weights  # [1, 2]

    def quality_guided_enhancement(self, img_feat, text_feat, quality_scores):
        """基于质量分数的特征增强"""
        batch_size = img_feat.size(0)
        enhanced_img_list = []
        enhanced_text_list = []

        for i in range(batch_size):
            # 获取单样本质量分数 - 修复索引错误
            if isinstance(quality_scores['overall_img_quality'], torch.Tensor):
                if quality_scores['overall_img_quality'].dim() == 0:
                    # 0维张量，直接取值
                    img_quality = quality_scores['overall_img_quality'].item()
                    text_quality = quality_scores['overall_text_quality'].item()
                else:
                    # 多维张量，使用索引
                    img_quality = quality_scores['overall_img_quality'][i].item()
                    text_quality = quality_scores['overall_text_quality'][i].item()
            else:
                # 如果是标量值
                img_quality = float(quality_scores['overall_img_quality'])
                text_quality = float(quality_scores['overall_text_quality'])

            # 低质量特征需要更多增强
            img_enhancement_strength = max(0.1, 1.0 - img_quality)
            text_enhancement_strength = max(0.1, 1.0 - text_quality)

            # 特征增强
            enhanced_img = img_feat[i:i + 1]
            enhanced_text = text_feat[i:i + 1]

            if img_enhancement_strength > 0.3:  # 需要增强
                img_enhancement = self.img_enhancer(enhanced_img)
                enhanced_img = enhanced_img + img_enhancement_strength * img_enhancement

            if text_enhancement_strength > 0.3:  # 需要增强
                text_enhancement = self.text_enhancer(enhanced_text)
                enhanced_text = enhanced_text + text_enhancement_strength * text_enhancement

            enhanced_img_list.append(enhanced_img)
            enhanced_text_list.append(enhanced_text)

        enhanced_img_feat = torch.cat(enhanced_img_list, dim=0)
        enhanced_text_feat = torch.cat(enhanced_text_list, dim=0)

        return enhanced_img_feat, enhanced_text_feat

    def cross_modal_interaction(self, img_feat, text_feat, interaction_strength):
        """跨模态交互增强"""
        batch_size = img_feat.size(0)

        # 只对交互强度高的样本进行跨模态注意力
        high_interaction_mask = interaction_strength > 0.5

        if high_interaction_mask.any():
            # 准备注意力输入 [seq_len, batch, hidden_size]
            img_seq = img_feat.unsqueeze(0)  # [1, batch, 512]
            text_seq = text_feat.unsqueeze(0)  # [1, batch, 512]

            # 图像attend到文本
            img_attended, _ = self.cross_modal_attention(
                query=img_seq, key=text_seq, value=text_seq
            )

            # 文本attend到图像
            text_attended, _ = self.cross_modal_attention(
                query=text_seq, key=img_seq, value=img_seq
            )

            img_attended = img_attended.squeeze(0)  # [batch, 512]
            text_attended = text_attended.squeeze(0)  # [batch, 512]

            # 基于交互强度混合
            interaction_strength = interaction_strength.unsqueeze(-1)  # [batch, 1]

            enhanced_img = (
                    interaction_strength * img_attended +
                    (1 - interaction_strength) * img_feat
            )
            enhanced_text = (
                    interaction_strength * text_attended +
                    (1 - interaction_strength) * text_feat
            )
        else:
            enhanced_img = img_feat
            enhanced_text = text_feat

        return enhanced_img, enhanced_text

    def adaptive_fusion(self, img_feat, text_feat, quality_weights, task_weights):
        """自适应融合"""
        # 结合质量权重和任务权重
        combined_weights = 0.6 * quality_weights + 0.4 * task_weights
        img_weight = combined_weights[:, 0:1]  # [batch, 1]
        text_weight = combined_weights[:, 1:2]  # [batch, 1]

        # 加权特征
        weighted_img = img_weight * img_feat
        weighted_text = text_weight * text_feat

        # 自适应门控
        combined_feat = torch.cat([weighted_img, weighted_text], dim=-1)
        gate = self.adaptive_gate(combined_feat)

        # 门控融合
        gated_img = gate * weighted_img
        gated_text = (1 - gate) * weighted_text

        # 最终拼接
        fused_features = torch.cat([gated_img, gated_text], dim=-1)  # [batch, 1024]

        return fused_features

    def forward(self, img_feat, text_feat, enhanced_img_feat, enhanced_text_feat,
                quality_results, missing_type):
        """
        主要融合接口 - 修复质量分数访问错误
        Args:
            img_feat: [batch, 512] 原始图像特征
            text_feat: [batch, 512] 原始文本特征
            enhanced_img_feat: [batch, 512] 生成增强的图像特征
            enhanced_text_feat: [batch, 512] 生成增强的文本特征
            quality_results: List[Dict] 质量评估结果
            missing_type: [batch] 缺失类型
        """
        batch_size = img_feat.size(0)
        final_features = []

        for i in range(batch_size):
            sample_quality = quality_results[i]

            # 根据缺失类型选择特征
            if missing_type[i] == 0:  # 完整模态
                current_img = img_feat[i:i + 1]
                current_text = text_feat[i:i + 1]

            elif missing_type[i] == 1:  # 缺失文本
                current_img = img_feat[i:i + 1]
                # 使用生成的文本特征，但根据质量调整
                generation_confidence = sample_quality['task_relevance'].get(
                    'text_task_relevance', torch.tensor(0.5)
                )

                # 修复张量访问
                if isinstance(generation_confidence, torch.Tensor):
                    if generation_confidence.dim() > 0:
                        confidence_value = generation_confidence[0].item() if generation_confidence.numel() > 0 else 0.5
                    else:
                        confidence_value = generation_confidence.item()
                else:
                    confidence_value = float(generation_confidence)

                if confidence_value > 0.7:
                    current_text = enhanced_text_feat[i:i + 1]
                else:
                    # 低置信度时混合原始和生成特征
                    current_text = confidence_value * enhanced_text_feat[i:i + 1] + (1 - confidence_value) * text_feat[
                                                                                                             i:i + 1]

            elif missing_type[i] == 2:  # 缺失图像
                current_text = text_feat[i:i + 1]
                # 类似处理图像特征
                generation_confidence = sample_quality['task_relevance'].get(
                    'img_task_relevance', torch.tensor(0.5)
                )

                # 修复张量访问
                if isinstance(generation_confidence, torch.Tensor):
                    if generation_confidence.dim() > 0:
                        confidence_value = generation_confidence[0].item() if generation_confidence.numel() > 0 else 0.5
                    else:
                        confidence_value = generation_confidence.item()
                else:
                    confidence_value = float(generation_confidence)

                if confidence_value > 0.7:
                    current_img = enhanced_img_feat[i:i + 1]
                else:
                    current_img = confidence_value * enhanced_img_feat[i:i + 1] + (1 - confidence_value) * img_feat[
                                                                                                           i:i + 1]

            # 计算质量权重 - 传入单样本质量结果
            quality_weights = self.compute_quality_weights(sample_quality)

            # 计算任务相关性权重
            task_weights = self.compute_task_relevance_weights(sample_quality['task_relevance'])

            # 质量引导的特征增强
            enhanced_img, enhanced_text = self.quality_guided_enhancement(
                current_img, current_text, sample_quality
            )

            # 跨模态交互
            interaction_strength = sample_quality['task_relevance']['cross_modal_synergy']
            if isinstance(interaction_strength, dict):
                interaction_strength = interaction_strength.get('cross_modal_synergy', torch.tensor(0.5))

            # 修复交互强度访问
            if isinstance(interaction_strength, torch.Tensor):
                if interaction_strength.dim() > 0:
                    strength_value = interaction_strength[0] if interaction_strength.numel() > 0 else torch.tensor(0.5)
                else:
                    strength_value = interaction_strength
            else:
                strength_value = torch.tensor(float(interaction_strength))

            enhanced_img, enhanced_text = self.cross_modal_interaction(
                enhanced_img, enhanced_text, strength_value.unsqueeze(0)
            )

            # 自适应融合
            sample_fused = self.adaptive_fusion(
                enhanced_img, enhanced_text, quality_weights, task_weights
            )

            final_features.append(sample_fused)

        return torch.cat(final_features, dim=0)  # [batch, 1024]

    def get_fusion_info(self, quality_results):
        """获取融合信息用于调试和可视化"""
        fusion_info = []

        for sample_quality in quality_results:
            quality_weights = self.compute_quality_weights(sample_quality)
            task_weights = self.compute_task_relevance_weights(sample_quality['task_relevance'])

            info = {
                'quality_weights': quality_weights.detach().cpu().numpy(),
                'task_weights': task_weights.detach().cpu().numpy(),
                'img_quality': sample_quality['overall_img_quality'].item(),
                'text_quality': sample_quality['overall_text_quality'].item(),
                'cross_modal_quality': sample_quality['overall_cross_modal_quality'].item(),
                'confidence': sample_quality['overall_confidence'].item()
            }
            fusion_info.append(info)

        return fusion_info