import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityGuidedFusion(nn.Module):
    """
    基于新质量分数结构的质量引导融合器

    新质量分数格式:
    {
        'image_quality': {
            'intrinsic_quality': float,
            'representation_quality': float,
            'generation_confidence': float,
            'task_contribution': float
        },
        'text_quality': {
            'intrinsic_quality': float,
            'representation_quality': float,
            'generation_confidence': float,
            'task_contribution': float
        },
        'cross_modal_consistency': float,
        'overall_uncertainty': float
    }
    """

    def __init__(self, hidden_size=512, fusion_strategy='adaptive_attention'):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_strategy = fusion_strategy

        # 模态质量综合评估网络
        self.modal_quality_aggregator = nn.Sequential(
            nn.Linear(4, 16),  # 4个质量指标 -> 16维
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # 跨模态交互强度控制网络
        self.interaction_controller = nn.Sequential(
            nn.Linear(2, 8),  # [consistency, uncertainty] -> 8维
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # 生成特征补偿网络
        self.image_compensator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

        self.text_compensator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )

        # 任务贡献度权重调整网络
        self.contribution_adjuster = nn.Sequential(
            nn.Linear(2, 8),  # [img_contrib, text_contrib] -> 8维
            nn.GELU(),
            nn.Linear(8, 2),
            nn.Softmax(dim=-1)
        )

        # 自适应注意力机制 (可选)
        if fusion_strategy == 'adaptive_attention':
            self.cross_attention = nn.MultiheadAttention(
                hidden_size, num_heads=8, dropout=0.1
            )

    def compute_modal_quality_score(self, modal_quality_dict):
        """
        计算单个模态的综合质量分数

        Args:
            modal_quality_dict: 包含4个质量指标的字典

        Returns:
            综合质量分数 [0, 1]
        """
        # todo:先禁用一下这里
        quality_vector = torch.tensor([
            modal_quality_dict['intrinsic_quality'].item(),
            modal_quality_dict['representation_quality'].item(),
            modal_quality_dict['generation_confidence'].item(),
            modal_quality_dict['task_contribution'].item()
        ]).to(next(self.parameters()).device)



        return self.modal_quality_aggregator(quality_vector.unsqueeze(0)).squeeze(0)

    def compute_interaction_strength(self, quality_scores):
        """
        计算跨模态交互强度

        Args:
            quality_scores: 质量分数字典

        Returns:
            交互强度 [0, 1]
        """
        consistency = quality_scores['cross_modal_consistency']
        uncertainty = quality_scores['overall_uncertainty']

        interaction_input = torch.tensor([
            consistency.item(),
            1.0 - uncertainty.item()  # 不确定性越低，交互越强
        ]).to(next(self.parameters()).device)

        return self.interaction_controller(interaction_input.unsqueeze(0)).squeeze(0)

    def adaptive_feature_mixing(self, original_feat, generated_feat, generation_confidence):
        """
        基于生成置信度的自适应特征混合

        Args:
            original_feat: 原始特征 [hidden_size]
            generated_feat: 生成特征 [hidden_size]
            generation_confidence: 生成置信度 [0, 1]

        Returns:
            混合后的特征 [hidden_size]
        """
        # 非线性混合，而非简单线性插值
        confidence = generation_confidence.item()

        if confidence > 0.8:  # 高置信度：主要使用生成特征
            alpha = 0.8
        elif confidence > 0.5:  # 中等置信度：平衡混合
            alpha = confidence
        else:  # 低置信度：主要使用原始特征
            alpha = 0.2

        return alpha * generated_feat + (1 - alpha) * original_feat

    def quality_aware_compensation(self, features, modality, quality_score):
        """
        基于质量分数的特征补偿

        Args:
            features: 输入特征 [hidden_size]
            modality: 'image' 或 'text'
            quality_score: 综合质量分数

        Returns:
            补偿后的特征 [hidden_size]
        """
        if quality_score < 0.5:  # 低质量需要补偿
            compensator = self.image_compensator if modality == 'image' else self.text_compensator
            compensation = compensator(features.unsqueeze(0)).squeeze(0)

            # 补偿强度与质量成反比
            compensation_strength = (0.5 - quality_score) * 2.0  # [0, 1]
            return features + compensation_strength * compensation
        else:
            return features

    def forward(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat,
                quality_scores, missing_type, debug=False, sample_idx_to_debug=0):
        """
        主要融合接口 - 增加调试信息和模块开关

        Args:
            debug: 是否打印调试信息
            sample_idx_to_debug: 调试哪个样本的索引
        """

        if random.random() < 0.001: debug = True

        # ==================== 模块开关配置 ====================
        USE_ADAPTIVE_MIXING = True  # 是否使用自适应特征混合
        USE_QUALITY_COMPENSATION = True  # 是否使用质量补偿
        USE_TASK_CONTRIBUTION_WEIGHT = True  # 是否使用任务贡献度权重
        USE_CROSS_MODAL_INTERACTION = True  # 是否使用跨模态交互
        USE_QUALITY_ADJUSTED_WEIGHT = True  # 是否使用质量调整权重

        batch_size = image_feat.size(0)
        fused_features = []

        for i in range(batch_size):
            sample_quality = quality_scores[i]

            if debug and i == sample_idx_to_debug:
                print(f"\n=== 调试样本 {i}, 缺失类型: {missing_type[i]} ===")
                print(f"原始图像特征范数: {torch.norm(image_feat[i]):.4f}")
                print(f"原始文本特征范数: {torch.norm(text_feat[i]):.4f}")
                print(f"增强图像特征范数: {torch.norm(enhanced_image_feat[i]):.4f}")
                print(f"增强文本特征范数: {torch.norm(enhanced_text_feat[i]):.4f}")

                print(f"\n质量分数:")
                print(f"  图像质量: {sample_quality['image_quality']}")
                print(f"  文本质量: {sample_quality['text_quality']}")
                print(f"  跨模态一致性: {sample_quality['cross_modal_consistency'].item():.4f}")
                print(f"  整体不确定性: {sample_quality['overall_uncertainty'].item():.4f}")

            # 1. 计算模态综合质量分数
            img_quality_score = self.compute_modal_quality_score(sample_quality['image_quality'])
            text_quality_score = self.compute_modal_quality_score(sample_quality['text_quality'])

            if debug and i == sample_idx_to_debug:
                print(f"\n综合质量分数:")
                print(f"  图像综合质量: {img_quality_score.item():.4f}")
                print(f"  文本综合质量: {text_quality_score.item():.4f}")

            # 2. 计算跨模态交互强度
            interaction_strength = self.compute_interaction_strength(sample_quality)

            if debug and i == sample_idx_to_debug:
                print(f"  跨模态交互强度: {interaction_strength.item():.4f}")

            # 3. 根据缺失类型进行特征选择和混合
            if missing_type[i] == 0:  # 完整模态
                final_img = image_feat[i]
                final_text = text_feat[i]

                if debug and i == sample_idx_to_debug:
                    print(f"\n完整模态 - 直接使用原始特征")

            elif missing_type[i] == 1:  # 缺失文本
                final_img = image_feat[i]

                if USE_ADAPTIVE_MIXING:
                    # 基于生成置信度混合文本特征
                    # 基于generation_confidence
                    original_text = text_feat[i]
                    final_text = self.adaptive_feature_mixing(
                        text_feat[i],
                        enhanced_text_feat[i],
                        sample_quality['text_quality']['generation_confidence']
                    )

                    if debug and i == sample_idx_to_debug:
                        print(f"\n缺失文本 - 自适应混合:")
                        print(f"  生成置信度: {sample_quality['text_quality']['generation_confidence']:.4f}")
                        print(f"  原始文本范数: {torch.norm(original_text):.4f}")
                        print(f"  混合后文本范数: {torch.norm(final_text):.4f}")
                        print(f"  混合前后差异: {torch.norm(final_text - original_text):.4f}")
                else:
                    final_text = text_feat[i]
                    if debug and i == sample_idx_to_debug:
                        print(f"\n缺失文本 - 跳过自适应混合，使用原始特征")

            elif missing_type[i] == 2:  # 缺失图像
                final_text = text_feat[i]

                if USE_ADAPTIVE_MIXING:
                    original_img = image_feat[i]
                    final_img = self.adaptive_feature_mixing(
                        image_feat[i],
                        enhanced_image_feat[i],
                        sample_quality['image_quality']['generation_confidence']
                    )

                    if debug and i == sample_idx_to_debug:
                        print(f"\n缺失图像 - 自适应混合:")
                        print(f"  生成置信度: {sample_quality['image_quality']['generation_confidence']:.4f}")
                        print(f"  原始图像范数: {torch.norm(original_img):.4f}")
                        print(f"  混合后图像范数: {torch.norm(final_img):.4f}")
                        print(f"  混合前后差异: {torch.norm(final_img - original_img):.4f}")
                else:
                    final_img = image_feat[i]
                    if debug and i == sample_idx_to_debug:
                        print(f"\n缺失图像 - 跳过自适应混合，使用原始特征")

            # 4. 质量感知特征补偿
            #这里的质量分数其实是基于几个不可靠的质量给出来的。我在评估那里姑且先都给1
            if USE_QUALITY_COMPENSATION:
                original_img_norm = torch.norm(final_img)
                original_text_norm = torch.norm(final_text)

                final_img = self.quality_aware_compensation(
                    final_img, 'image', img_quality_score
                )
                final_text = self.quality_aware_compensation(
                    final_text, 'text', text_quality_score
                )

                if debug and i == sample_idx_to_debug:
                    print(f"\n质量补偿:")
                    print(f"  图像: {original_img_norm:.4f} -> {torch.norm(final_img):.4f}")
                    print(f"  文本: {original_text_norm:.4f} -> {torch.norm(final_text):.4f}")
            else:
                if debug and i == sample_idx_to_debug:
                    print(f"\n跳过质量补偿")

            # 5. 基于任务贡献度调整权重
            if USE_TASK_CONTRIBUTION_WEIGHT:
                img_contribution = sample_quality['image_quality']['task_contribution']
                text_contribution = sample_quality['text_quality']['task_contribution']

                contribution_input = torch.tensor([
                    img_contribution.item(),
                    text_contribution.item()
                ]).to(image_feat.device)

                adjusted_weights = self.contribution_adjuster(contribution_input.unsqueeze(0)).squeeze(0)
                img_weight, text_weight = adjusted_weights[0], adjusted_weights[1]

                if debug and i == sample_idx_to_debug:
                    print(f"\n任务贡献度权重:")
                    print(f"  原始贡献度 - 图像: {img_contribution:.4f}, 文本: {text_contribution:.4f}")
                    print(f"  调整后权重 - 图像: {img_weight:.4f}, 文本: {text_weight:.4f}")
            else:
                img_weight = text_weight = 0.5
                if debug and i == sample_idx_to_debug:
                    print(f"\n跳过任务贡献度权重，使用默认0.5")

            # 6. 跨模态交互调制
            if USE_CROSS_MODAL_INTERACTION and self.fusion_strategy == 'adaptive_attention' and interaction_strength > 0.3:
                original_img_norm = torch.norm(final_img)
                original_text_norm = torch.norm(final_text)

                # 使用注意力机制增强跨模态交互
                img_seq = final_img.unsqueeze(0).unsqueeze(0)  # [1, 1, 512]
                text_seq = final_text.unsqueeze(0).unsqueeze(0)  # [1, 1, 512]

                # 交叉注意力：图像attend到文本
                img_attended, _ = self.cross_attention(img_seq, text_seq, text_seq)
                text_attended, _ = self.cross_attention(text_seq, img_seq, img_seq)

                # 基于交互强度混合
                final_img = (interaction_strength * img_attended.squeeze() +
                             (1 - interaction_strength) * final_img)
                final_text = (interaction_strength * text_attended.squeeze() +
                              (1 - interaction_strength) * final_text)

                if debug and i == sample_idx_to_debug:
                    print(f"\n跨模态交互:")
                    print(f"  交互强度: {interaction_strength.item():.4f}")
                    print(f"  图像: {original_img_norm:.4f} -> {torch.norm(final_img):.4f}")
                    print(f"  文本: {original_text_norm:.4f} -> {torch.norm(final_text):.4f}")
            else:
                if debug and i == sample_idx_to_debug:
                    if not USE_CROSS_MODAL_INTERACTION:
                        print(f"\n跳过跨模态交互")
                    else:
                        print(f"\n跨模态交互强度过低({interaction_strength.item():.4f})，跳过")

            # 7. 最终加权融合
            if USE_QUALITY_ADJUSTED_WEIGHT:
                # 考虑质量分数对权重的影响
                quality_adjusted_img_weight = img_weight * img_quality_score
                quality_adjusted_text_weight = text_weight * text_quality_score

                # 重新归一化
                total_weight = quality_adjusted_img_weight + quality_adjusted_text_weight
                if total_weight > 0:
                    quality_adjusted_img_weight = quality_adjusted_img_weight / total_weight
                    quality_adjusted_text_weight = quality_adjusted_text_weight / total_weight
                else:
                    quality_adjusted_img_weight = quality_adjusted_text_weight = 0.5

                if debug and i == sample_idx_to_debug:
                    print(f"\n质量调整权重:")
                    print(f"  调整前 - 图像: {img_weight:.4f}, 文本: {text_weight:.4f}")
                    print(
                        f"  调整后 - 图像: {quality_adjusted_img_weight.item():.4f}, 文本: {quality_adjusted_text_weight.item():.4f}")
            else:
                quality_adjusted_img_weight = img_weight
                quality_adjusted_text_weight = text_weight
                if debug and i == sample_idx_to_debug:
                    print(f"\n跳过质量调整权重")

            # 最终特征拼接
            weighted_img = quality_adjusted_img_weight * final_img
            weighted_text = quality_adjusted_text_weight * final_text

            fused_feat = torch.cat([weighted_img, weighted_text], dim=-1)  # [1024]

            if debug and i == sample_idx_to_debug:
                print(f"\n最终融合:")
                print(f"  加权图像范数: {torch.norm(weighted_img):.4f}")
                print(f"  加权文本范数: {torch.norm(weighted_text):.4f}")
                print(f"  融合特征范数: {torch.norm(fused_feat):.4f}")

                # 对比简单拼接的效果
                simple_concat = torch.cat([final_img, final_text], dim=-1)
                print(f"  简单拼接范数: {torch.norm(simple_concat):.4f}")
                print(f"  融合vs简单拼接差异: {torch.norm(fused_feat - simple_concat):.4f}")
                print("=" * 60)

            fused_features.append(fused_feat.unsqueeze(0))

        return torch.cat(fused_features, dim=0)  # [batch, 1024]

    # def forward(self, image_feat, text_feat, enhanced_image_feat, enhanced_text_feat,
    #             quality_scores, missing_type):
    #     """
    #     主要融合接口
    #
    #     Args:
    #         image_feat: [batch, 512] 原始图像特征
    #         text_feat: [batch, 512] 原始文本特征
    #         enhanced_image_feat: [batch, 512] 增强图像特征
    #         enhanced_text_feat: [batch, 512] 增强文本特征
    #         quality_scores: List[Dict] 每个样本的质量分数
    #         missing_type: [batch] 缺失类型
    #
    #     Returns:
    #         [batch, 1024] 融合后的特征
    #     """
    #     batch_size = image_feat.size(0)
    #     fused_features = []
    #
    #     for i in range(batch_size):
    #         sample_quality = quality_scores[i]
    #
    #         # 1. 计算模态综合质量分数
    #         img_quality_score = self.compute_modal_quality_score(sample_quality['image_quality'])
    #         text_quality_score = self.compute_modal_quality_score(sample_quality['text_quality'])
    #
    #         # 2. 计算跨模态交互强度
    #         interaction_strength = self.compute_interaction_strength(sample_quality)
    #
    #         # 3. 根据缺失类型进行特征选择和混合
    #         if missing_type[i] == 0:  # 完整模态
    #             final_img = image_feat[i]
    #             final_text = text_feat[i]
    #
    #         elif missing_type[i] == 1:  # 缺失文本
    #             final_img = image_feat[i]
    #
    #             # 基于生成置信度混合文本特征
    #             final_text = self.adaptive_feature_mixing(
    #                 text_feat[i],
    #                 enhanced_text_feat[i],
    #                 sample_quality['text_quality']['generation_confidence']
    #             )
    #
    #         elif missing_type[i] == 2:  # 缺失图像
    #             final_text = text_feat[i]
    #
    #             # 基于生成置信度混合图像特征
    #             final_img = self.adaptive_feature_mixing(
    #                 image_feat[i],
    #                 enhanced_image_feat[i],
    #                 sample_quality['image_quality']['generation_confidence']
    #             )
    #
    #         # 4. 质量感知特征补偿
    #         final_img = self.quality_aware_compensation(
    #             final_img, 'image', img_quality_score
    #         )
    #         final_text = self.quality_aware_compensation(
    #             final_text, 'text', text_quality_score
    #         )
    #
    #         # 5. 基于任务贡献度调整权重
    #         img_contribution = sample_quality['image_quality']['task_contribution']
    #         text_contribution = sample_quality['text_quality']['task_contribution']
    #
    #         contribution_input = torch.tensor([
    #             img_contribution.item(),
    #             text_contribution.item()
    #         ]).to(image_feat.device)
    #
    #         adjusted_weights = self.contribution_adjuster(contribution_input.unsqueeze(0)).squeeze(0)
    #         img_weight, text_weight = adjusted_weights[0], adjusted_weights[1]
    #
    #         # 6. 跨模态交互调制
    #         if self.fusion_strategy == 'adaptive_attention' and interaction_strength > 0.3:
    #             # 使用注意力机制增强跨模态交互
    #             img_seq = final_img.unsqueeze(0).unsqueeze(0)  # [1, 1, 512]
    #             text_seq = final_text.unsqueeze(0).unsqueeze(0)  # [1, 1, 512]
    #
    #             # 交叉注意力：图像attend到文本
    #             img_attended, _ = self.cross_attention(img_seq, text_seq, text_seq)
    #             text_attended, _ = self.cross_attention(text_seq, img_seq, img_seq)
    #
    #             # 基于交互强度混合
    #             final_img = (interaction_strength * img_attended.squeeze() +
    #                          (1 - interaction_strength) * final_img)
    #             final_text = (interaction_strength * text_attended.squeeze() +
    #                           (1 - interaction_strength) * final_text)
    #
    #         # 7. 最终加权融合
    #         # 考虑质量分数对权重的影响
    #         quality_adjusted_img_weight = img_weight * img_quality_score
    #         quality_adjusted_text_weight = text_weight * text_quality_score
    #
    #         # 重新归一化
    #         total_weight = quality_adjusted_img_weight + quality_adjusted_text_weight
    #         if total_weight > 0:
    #             quality_adjusted_img_weight = quality_adjusted_img_weight / total_weight
    #             quality_adjusted_text_weight = quality_adjusted_text_weight / total_weight
    #         else:
    #             quality_adjusted_img_weight = quality_adjusted_text_weight = 0.5
    #
    #         # 最终特征拼接
    #         weighted_img = quality_adjusted_img_weight * final_img
    #         weighted_text = quality_adjusted_text_weight * final_text
    #
    #         fused_feat = torch.cat([weighted_img, weighted_text], dim=-1)  # [1024]
    #         fused_features.append(fused_feat.unsqueeze(0))
    #
    #     return torch.cat(fused_features, dim=0)  # [batch, 1024]

    def get_fusion_weights(self, quality_scores):
        """
        获取融合权重信息，用于可视化和分析

        Returns:
            Dict: 包含各种权重信息
        """
        weights_info = []

        for sample_quality in quality_scores:
            img_quality = self.compute_modal_quality_score(sample_quality['image_quality'])
            text_quality = self.compute_modal_quality_score(sample_quality['text_quality'])
            interaction_strength = self.compute_interaction_strength(sample_quality)

            weights_info.append({
                'image_quality': img_quality.item(),
                'text_quality': text_quality.item(),
                'interaction_strength': interaction_strength.item(),
                'cross_modal_consistency': sample_quality['cross_modal_consistency'].item(),
                'overall_uncertainty': sample_quality['overall_uncertainty'].item()
            })

        return weights_info

