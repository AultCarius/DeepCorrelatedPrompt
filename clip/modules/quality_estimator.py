import torch
import torch.nn as nn
import torch.nn.functional as F


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