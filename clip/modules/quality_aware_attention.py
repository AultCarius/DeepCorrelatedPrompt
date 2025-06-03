# clip/modules/attention_reweighting_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionReweightingFusion(nn.Module):
    """基于新质量指标的注意力重加权融合器"""

    def __init__(self, hidden_size=512, fusion_strategy='quality_attention'):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_strategy = fusion_strategy

        # 【更新】基于新质量指标的注意力网络
        # 输入：几何质量(2) + 信息质量(2) + 一致性(2) + 重要性(2) + 真实性(2) + 难度(1) = 11维
        self.quality_attention_net = nn.Sequential(
            nn.Linear(11, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 总体质量注意力权重 [0,1]
        )

        # 【新增】模态重要性权重计算器
        self.modality_importance_net = nn.Sequential(
            nn.Linear(4, 32),  # 图像重要性 + 文本重要性 + 图像真实性 + 文本真实性
            nn.GELU(),
            nn.Linear(32, 2),  # 输出图像权重和文本权重
            nn.Softmax(dim=-1)
        )

        # 【新增】任务难度感知的补偿网络
        self.difficulty_compensation_net = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),  # 特征 + 难度分数
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()  # 补偿强度
        )

        # 【更新】跨模态注意力 - 考虑重要性权重
        self.cross_modal_attention = nn.MultiheadAttention(
            hidden_size, num_heads=8, dropout=0.1
        )

        # 【新增】缺失模态补偿强度控制
        self.missing_compensation_strength = nn.Parameter(torch.tensor(0.5))

    def extract_quality_features(self, quality_scores):
        """从新质量分数中提取特征向量"""
        quality_vector = torch.tensor([
            # 几何质量 (2维)
            quality_scores['geometric_image'].item(),
            quality_scores['geometric_text'].item(),

            # 信息质量 (2维)
            quality_scores['information_image'].item(),
            quality_scores['information_text'].item(),

            # 一致性质量 (2维)
            quality_scores['consistency_original'].item(),
            quality_scores['consistency_enhanced'].item(),

            # 【新】重要性预测 (2维)
            quality_scores['predicted_image_importance'].item(),
            quality_scores['predicted_text_importance'].item(),

            # 【新】真实性 (2维)
            quality_scores['image_authenticity'].item(),
            quality_scores['text_authenticity'].item(),

            # 【新】任务难度 (1维)
            quality_scores['predicted_task_difficulty'].item()
        ]).to(quality_scores['geometric_image'].device)

        return quality_vector

    def compute_modality_weights(self, quality_scores):
        """基于重要性和真实性计算模态权重"""
        # 提取关键指标
        importance_authenticity_vector = torch.tensor([
            quality_scores['predicted_image_importance'].item(),
            quality_scores['predicted_text_importance'].item(),
            quality_scores['image_authenticity'].item(),
            quality_scores['text_authenticity'].item()
        ]).to(quality_scores['predicted_image_importance'].device)

        # 计算模态权重
        modality_weights = self.modality_importance_net(importance_authenticity_vector.unsqueeze(0)).squeeze(0)

        return modality_weights[0], modality_weights[1]  # image_weight, text_weight

    def apply_difficulty_compensation(self, features, task_difficulty):
        """基于任务难度进行特征补偿"""
        # 将难度分数与特征拼接
        difficulty_expanded = task_difficulty.unsqueeze(-1).expand(-1, self.hidden_size)
        feat_with_difficulty = torch.cat([features, task_difficulty.unsqueeze(-1)], dim=-1)

        # 生成补偿向量
        compensation_strength = self.difficulty_compensation_net(feat_with_difficulty)

        # 难度越高，补偿越强
        compensation_factor = task_difficulty.unsqueeze(-1)  # [batch, 1]
        compensated_features = features + compensation_factor * compensation_strength * features

        return compensated_features

    def handle_missing_modality(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat,
                                quality_scores, missing_type, batch_idx):
        """处理缺失模态的特征选择和融合"""
        img_weight, text_weight = self.compute_modality_weights(quality_scores)

        if missing_type[batch_idx] == 0:  # 完整模态
            # 主要使用原始特征，根据重要性调整权重
            final_img = image_feat[batch_idx:batch_idx + 1]
            final_text = text_feat[batch_idx:batch_idx + 1]

            # 根据重要性微调
            importance_ratio = quality_scores['predicted_image_importance'] / (
                    quality_scores['predicted_image_importance'] + quality_scores['predicted_text_importance'] + 1e-8
            )

            if importance_ratio > 0.6:  # 图像更重要
                final_img = final_img * 1.1
                final_text = final_text * 0.9
            elif importance_ratio < 0.4:  # 文本更重要
                final_img = final_img * 0.9
                final_text = final_text * 1.1

        elif missing_type[batch_idx] == 1:  # 缺失文本
            # 图像用原始，文本需要融合
            final_img = image_feat[batch_idx:batch_idx + 1]

            # 根据真实性和重要性融合文本特征
            text_authenticity = quality_scores['text_authenticity']
            text_importance = quality_scores['predicted_text_importance']

            # 融合权重：真实性低时更多依赖生成特征
            original_weight = text_authenticity * 0.3  # 原始特征（prompt信息）
            enhanced_weight = (1 - text_authenticity) * text_importance  # 生成特征

            final_text = (original_weight * text_feat[batch_idx:batch_idx + 1] +
                          enhanced_weight * enhanced_text_feat[batch_idx:batch_idx + 1])

        elif missing_type[batch_idx] == 2:  # 缺失图像
            # 文本用原始，图像需要融合
            final_text = text_feat[batch_idx:batch_idx + 1]

            # 根据真实性和重要性融合图像特征
            img_authenticity = quality_scores['image_authenticity']
            img_importance = quality_scores['predicted_image_importance']

            original_weight = img_authenticity * 0.3
            enhanced_weight = (1 - img_authenticity) * img_importance

            final_img = (original_weight * image_feat[batch_idx:batch_idx + 1] +
                         enhanced_weight * enhanced_image_feat[batch_idx:batch_idx + 1])

        return final_img, final_text, img_weight, text_weight

    def forward(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat, quality_scores, missing_type):
        """
        基于新质量指标进行注意力重加权融合

        Args:
            image_feat: [batch, 512] 原始图像特征
            text_feat: [batch, 512] 原始文本特征
            enhanced_image_feat: [batch, 512] 增强图像特征
            enhanced_text_feat: [batch, 512] 增强文本特征
            quality_scores: List[Dict] 新的质量分数
            missing_type: [batch] 缺失类型

        Returns:
            [batch, 1024] 融合后的特征
        """
        batch_size = image_feat.size(0)
        fused_features = []

        for i in range(batch_size):
            sample_quality = quality_scores[i]

            # 1. 提取质量特征并计算总体质量注意力
            quality_features = self.extract_quality_features(sample_quality)
            overall_quality_attention = self.quality_attention_net(quality_features.unsqueeze(0)).squeeze(0)

            # 2. 处理缺失模态，选择合适的特征
            final_img, final_text, img_weight, text_weight = self.handle_missing_modality(
                image_feat, text_feat, enhanced_image_feat, enhanced_text_feat,
                sample_quality, missing_type, i
            )

            # 3. 基于任务难度进行补偿
            task_difficulty = sample_quality['predicted_task_difficulty'].unsqueeze(0)
            compensated_img = self.apply_difficulty_compensation(final_img, task_difficulty)
            compensated_text = self.apply_difficulty_compensation(final_text, task_difficulty)

            # 4. 跨模态注意力融合 (如果启用)
            if self.fusion_strategy == 'quality_attention':
                # 构建注意力输入
                img_seq = compensated_img.unsqueeze(0)  # [1, 1, 512]
                text_seq = compensated_text.unsqueeze(0)  # [1, 1, 512]

                # 图像attend to文本
                img_attended, _ = self.cross_modal_attention(
                    img_seq, text_seq, text_seq
                )

                # 文本attend to图像
                text_attended, _ = self.cross_modal_attention(
                    text_seq, img_seq, img_seq
                )

                final_img_feat = img_attended.squeeze(0)  # [1, 512]
                final_text_feat = text_attended.squeeze(0)  # [1, 512]
            else:
                final_img_feat = compensated_img
                final_text_feat = compensated_text

            # 5. 应用模态权重和质量注意力
            weighted_img = final_img_feat * img_weight * overall_quality_attention
            weighted_text = final_text_feat * text_weight * overall_quality_attention

            # 6. 最终拼接
            sample_fused = torch.cat([weighted_img, weighted_text], dim=-1)  # [1, 1024]
            fused_features.append(sample_fused)

        return torch.cat(fused_features, dim=0)  # [batch, 1024]


class QualityAwareTaskLoss(nn.Module):
    """基于新质量指标的任务损失"""

    def __init__(self, importance_weight=0.3, difficulty_weight=0.4, authenticity_weight=0.3):
        super().__init__()
        self.importance_weight = importance_weight
        self.difficulty_weight = difficulty_weight
        self.authenticity_weight = authenticity_weight

    def forward(self, logits, targets, quality_scores):
        """
        基于新质量分数的加权损失

        Args:
            logits: [batch, num_classes] 模型预测
            targets: [batch, num_classes] 真实标签 (multi-label)
            quality_scores: List[Dict] 新的质量分数
        """
        # 基础任务损失
        base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # [batch, num_classes]
        base_loss = base_loss.mean(dim=-1)  # [batch] 每个样本的平均损失

        # 计算质量权重
        quality_weights = []
        for quality in quality_scores:
            # 1. 重要性权重：重要性高的样本权重大
            avg_importance = (quality['predicted_image_importance'] + quality['predicted_text_importance']) / 2.0

            # 2. 难度权重：难样本权重大（更需要学习）
            difficulty = quality['predicted_task_difficulty']

            # 3. 真实性权重：真实性高的样本权重大
            avg_authenticity = (quality['image_authenticity'] + quality['text_authenticity']) / 2.0

            # 4. 综合权重
            combined_weight = (
                    self.importance_weight * avg_importance +
                    self.difficulty_weight * difficulty +
                    self.authenticity_weight * avg_authenticity
            )

            # 确保权重在合理范围内
            combined_weight = torch.clamp(combined_weight, 0.1, 2.0)
            quality_weights.append(combined_weight)

        quality_weights = torch.stack(quality_weights)  # [batch]

        # 加权损失
        weighted_loss = base_loss * quality_weights

        return weighted_loss.mean()