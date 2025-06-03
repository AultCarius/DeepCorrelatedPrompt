import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityGuidedFusion(nn.Module):
    """质量引导的特征融合器"""

    def __init__(self, hidden_size=512, fusion_strategy='weighted'):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_strategy = fusion_strategy

        # 权重计算网络 - 从质量分数到融合权重
        self.weight_net = nn.Sequential(
            nn.Linear(6, 32),  # 6个质量指标输入
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 3),  # 3个权重：原始图像权重、原始文本权重、生成特征权重
            nn.Softmax(dim=-1)
        )

        # 质量补偿网络
        self.compensation_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

        # 如果使用注意力融合
        if fusion_strategy == 'attention':
            self.self_attention = nn.MultiheadAttention(hidden_size, 8, dropout=0.1)

    def extract_quality_vector(self, quality_scores):
        """将质量字典转换为向量"""
        # 提取主要质量指标
        img_quality = quality_scores['image_intrinsic'].mean()  # 平均质量
        text_quality = quality_scores['text_intrinsic'].mean()
        consistency = quality_scores['cross_modal_consistency'].squeeze()
        confidence = quality_scores['generation_confidence'].squeeze()

        # 计算综合指标
        overall_img = img_quality
        overall_text = text_quality

        # 6维质量向量
        quality_vector = torch.tensor([
            overall_img.item(), overall_text.item(),
            consistency.item(), confidence.item(),
            (overall_img * consistency).item(),  # 图像与一致性的交互
            (overall_text * consistency).item()  # 文本与一致性的交互
        ]).to(quality_scores['image_intrinsic'].device)

        return quality_vector

    def forward(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat, quality_scores, missing_type):
        """
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
        fused_features = []

        for i in range(batch_size):
            # 获取当前样本的质量分数
            sample_quality = quality_scores[i]
            quality_vector = self.extract_quality_vector(sample_quality)

            # 计算融合权重
            weights = self.weight_net(quality_vector.unsqueeze(0)).squeeze(0)  # [3]
            w_img_orig, w_text_orig, w_generated = weights[0], weights[1], weights[2]

            if missing_type[i] == 0:  # 完整模态
                # 主要使用原始特征，权重集中在原始特征上
                final_img = image_feat[i:i + 1]
                final_text = text_feat[i:i + 1]

            elif missing_type[i] == 1:  # 缺失文本
                # 图像使用原始特征
                final_img = image_feat[i:i + 1]

                # 文本特征需要融合：原始(prompt信息) + 生成(内容信息)
                confidence = sample_quality['generation_confidence']
                compensated_text = self.compensation_net(enhanced_text_feat[i:i + 1])

                # 加权融合文本特征
                final_text = (w_text_orig * text_feat[i:i + 1] +
                              w_generated * compensated_text * confidence)

            elif missing_type[i] == 2:  # 缺失图像
                # 文本使用原始特征
                final_text = text_feat[i:i + 1]

                # 图像特征融合
                confidence = sample_quality['generation_confidence']
                compensated_img = self.compensation_net(enhanced_image_feat[i:i + 1])

                final_img = (w_img_orig * image_feat[i:i + 1] +
                             w_generated * compensated_img * confidence)

            # 可选的注意力融合
            if self.fusion_strategy == 'attention':
                # 构建序列进行自注意力
                seq = torch.cat([final_img, final_text], dim=0).unsqueeze(1)  # [2, 1, 512]
                seq = seq.transpose(0, 1)  # [1, 2, 512] 适配旧版PyTorch
                attended, _ = self.self_attention(seq, seq, seq)
                attended = attended.transpose(0, 1)  # [2, 1, 512]
                final_img, final_text = attended[0:1], attended[1:2]

            # 最终拼接，保持与原架构一致的输出格式
            fused_feat = torch.cat([final_img, final_text], dim=-1)  # [1, 1024]
            fused_features.append(fused_feat)

        return torch.cat(fused_features, dim=0)  # [batch, 1024]