# """
# Enhanced Modal Generator with Feature Augmentation
# 增强型模态生成器 - 特征增强方案
#
# 核心思想：
# 1. 保留原有特征（包含prompt学习信息）
# 2. 生成补充特征来增强缺失部分
# 3. 使用Transformer进行跨模态生成
# 4. 循环一致性确保生成质量
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from typing import Optional, Tuple, Dict
#
#
# class MultiHeadAttention(nn.Module):
#     """多头注意力机制"""
#
#     def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
#         super().__init__()
#         assert d_model % num_heads == 0
#
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.d_k = d_model // num_heads
#
#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)
#         self.w_o = nn.Linear(d_model, d_model)
#
#         self.dropout = nn.Dropout(dropout)
#         self.scale = math.sqrt(self.d_k)
#
#     def forward(self, query, key, value, mask=None):
#         batch_size = query.size(0)
#
#         # Linear projections
#         Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#         V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
#
#         # Attention
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)
#
#         context = torch.matmul(attn_weights, V)
#         context = context.transpose(1, 2).contiguous().view(
#             batch_size, -1, self.d_model
#         )
#
#         output = self.w_o(context)
#         return output, attn_weights
#
#
# class TransformerBlock(nn.Module):
#     """Transformer编码块"""
#
#     def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
#         super().__init__()
#         self.attention = MultiHeadAttention(d_model, num_heads, dropout)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x, mask=None):
#         # Self-attention with residual connection
#         attn_output, attn_weights = self.attention(x, x, x, mask)
#         x = self.norm1(x + attn_output)
#
#         # Feed-forward with residual connection
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + ff_output)
#
#         return x, attn_weights
#
#
# class CrossModalGenerator(nn.Module):
#     """跨模态生成器 - 使用Transformer架构"""
#
#     def __init__(
#             self,
#             input_dim: int = 512,
#             output_dim: int = 512,
#             num_layers: int = 3,
#             num_heads: int = 8,
#             d_ff: int = 2048,
#             dropout: float = 0.1
#     ):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#
#         # Input projection
#         self.input_projection = nn.Linear(input_dim, input_dim)
#
#         # Positional encoding (learnable)
#         self.pos_embedding = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
#
#         # Transformer layers
#         self.transformer_layers = nn.ModuleList([
#             TransformerBlock(input_dim, num_heads, d_ff, dropout)
#             for _ in range(num_layers)
#         ])
#
#         # Output projection
#         self.output_projection = nn.Sequential(
#             nn.Linear(input_dim, d_ff),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, output_dim),
#             nn.Tanh()  # 输出范围约束
#         )
#
#         # Initialize weights
#         self.apply(self._init_weights)
#
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.normal_(module.weight, std=0.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif isinstance(module, nn.LayerNorm):
#             nn.init.zeros_(module.bias)
#             nn.init.ones_(module.weight)
#
#     def forward(self, source_features):
#         """
#         Args:
#             source_features: [batch_size, feature_dim]
#         Returns:
#             generated_features: [batch_size, output_dim]
#         """
#         batch_size = source_features.size(0)
#
#         # Add sequence dimension and project
#         x = source_features.unsqueeze(1)  # [batch, 1, dim]
#         x = self.input_projection(x)
#
#         # Add positional encoding
#         x = x + self.pos_embedding
#
#         # Pass through transformer layers
#         attention_weights = []
#         for layer in self.transformer_layers:
#             x, attn_weights = layer(x)
#             attention_weights.append(attn_weights)
#
#         # Output projection
#         x = x.squeeze(1)  # [batch, dim]
#         generated_features = self.output_projection(x)
#
#         return generated_features
#
#
# class FeatureEnhancer(nn.Module):
#     """特征增强器 - 融合原始特征和生成特征"""
#
#     def __init__(self, feature_dim: int = 512, dropout: float = 0.1):
#         super().__init__()
#         self.feature_dim = feature_dim
#
#         # Gate mechanism for adaptive fusion
#         self.gate_network = nn.Sequential(
#             nn.Linear(feature_dim * 2, feature_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(feature_dim, feature_dim),
#             nn.Sigmoid()
#         )
#
#         # Feature refinement
#         self.refinement_network = nn.Sequential(
#             nn.Linear(feature_dim * 2, feature_dim * 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(feature_dim * 2, feature_dim),
#             nn.LayerNorm(feature_dim)
#         )
#
#     def forward(self, original_features, generated_features, missing_type):
#         """
#         Args:
#             original_features: 原始特征 (可能包含零值)
#             generated_features: 生成特征
#             missing_type: 缺失类型 (0: 完整, 1: 缺失文本, 2: 缺失图像)
#         """
#         if missing_type == 0:  # 模态完整，返回原始特征
#             return original_features
#
#         # 计算融合门控
#         combined = torch.cat([original_features, generated_features], dim=-1)
#         gate = self.gate_network(combined)
#
#         # 自适应融合
#         enhanced_features = gate * original_features + (1 - gate) * generated_features
#
#         # 特征精化
#         refined_input = torch.cat([enhanced_features, generated_features], dim=-1)
#         enhanced_features = self.refinement_network(refined_input)
#
#         return enhanced_features
#
#
# class CycleConsistencyModule(nn.Module):
#     """循环一致性模块"""
#
#     def __init__(self, feature_dim: int = 512):
#         super().__init__()
#         self.img_to_text_generator = CrossModalGenerator(feature_dim, feature_dim)
#         self.text_to_img_generator = CrossModalGenerator(feature_dim, feature_dim)
#
#     def forward(self, image_features, text_features):
#         """
#         计算循环一致性损失
#         """
#         # Image -> Text -> Image
#         fake_text = self.img_to_text_generator(image_features)
#         reconstructed_image = self.text_to_img_generator(fake_text)
#
#         # Text -> Image -> Text
#         fake_image = self.text_to_img_generator(text_features)
#         reconstructed_text = self.img_to_text_generator(fake_image)
#
#         return {
#             'fake_text': fake_text,
#             'fake_image': fake_image,
#             'reconstructed_image': reconstructed_image,
#             'reconstructed_text': reconstructed_text
#         }
#
#     def compute_cycle_loss(self, image_features, text_features):
#         """计算循环一致性损失"""
#         cycle_outputs = self.forward(image_features, text_features)
#
#         # L1 reconstruction loss
#         img_cycle_loss = F.l1_loss(
#             cycle_outputs['reconstructed_image'],
#             image_features
#         )
#         text_cycle_loss = F.l1_loss(
#             cycle_outputs['reconstructed_text'],
#             text_features
#         )
#
#         # Cosine similarity loss
#         img_cos_loss = 1 - F.cosine_similarity(
#             cycle_outputs['reconstructed_image'],
#             image_features,
#             dim=-1
#         ).mean()
#         text_cos_loss = 1 - F.cosine_similarity(
#             cycle_outputs['reconstructed_text'],
#             text_features,
#             dim=-1
#         ).mean()
#
#         total_cycle_loss = (img_cycle_loss + text_cycle_loss +
#                             img_cos_loss + text_cos_loss) / 4
#
#         return total_cycle_loss, cycle_outputs
#
#
# class EnhancedModalGenerator(nn.Module):
#     """增强型模态生成器 - 主要模块"""
#
#     def __init__(
#             self,
#             feature_dim: int = 512,
#             num_layers: int = 3,
#             num_heads: int = 8,
#             d_ff: int = 2048,
#             dropout: float = 0.1,
#             use_cycle_consistency: bool = True
#     ):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.use_cycle_consistency = use_cycle_consistency
#
#         # 跨模态生成器
#         self.img_to_text_generator = CrossModalGenerator(
#             feature_dim, feature_dim, num_layers, num_heads, d_ff, dropout
#         )
#         self.text_to_img_generator = CrossModalGenerator(
#             feature_dim, feature_dim, num_layers, num_heads, d_ff, dropout
#         )
#
#         # 特征增强器
#         self.image_enhancer = FeatureEnhancer(feature_dim, dropout)
#         self.text_enhancer = FeatureEnhancer(feature_dim, dropout)
#
#         # 循环一致性模块
#         if use_cycle_consistency:
#             self.cycle_module = CycleConsistencyModule(feature_dim)
#
#         # 质量预测器（用于生成特征的质量评估）
#         self.quality_predictor = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim // 2),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(feature_dim // 2, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, image_features, text_features, missing_type):
#         """
#         Args:
#             image_features: [batch_size, 512] - 图像特征
#             text_features: [batch_size, 512] - 文本特征
#             missing_type: List[int] - 每个样本的缺失类型
#         Returns:
#             enhanced_features: Dict - 增强后的特征
#             generation_info: Dict - 生成相关信息
#         """
#         batch_size = image_features.size(0)
#         device = image_features.device
#
#         enhanced_image_features = image_features.clone()
#         enhanced_text_features = text_features.clone()
#
#         generated_image_features = torch.zeros_like(image_features)
#         generated_text_features = torch.zeros_like(text_features)
#
#         quality_scores = torch.ones(batch_size, 2).to(device)  # [img_quality, text_quality]
#
#         # 处理每种缺失类型
#         for i, miss_type in enumerate(missing_type):
#             if miss_type == 1:  # 缺失文本
#                 # 从图像生成文本特征
#                 gen_text = self.img_to_text_generator(image_features[i:i + 1])
#                 generated_text_features[i:i + 1] = gen_text
#
#                 # 增强文本特征（融合原始和生成）
#                 enhanced_text_features[i:i + 1] = self.text_enhancer(
#                     text_features[i:i + 1], gen_text, miss_type
#                 )
#
#                 # 预测生成质量
#                 quality_scores[i, 1] = self.quality_predictor(gen_text).squeeze()
#
#             elif miss_type == 2:  # 缺失图像
#                 # 从文本生成图像特征
#                 gen_image = self.text_to_img_generator(text_features[i:i + 1])
#                 generated_image_features[i:i + 1] = gen_image
#
#                 # 增强图像特征
#                 enhanced_image_features[i:i + 1] = self.image_enhancer(
#                     image_features[i:i + 1], gen_image, miss_type
#                 )
#
#                 # 预测生成质量
#                 quality_scores[i, 0] = self.quality_predictor(gen_image).squeeze()
#
#         enhanced_features = {
#             'image': enhanced_image_features,
#             'text': enhanced_text_features,
#             'original_image': image_features,
#             'original_text': text_features,
#             'generated_image': generated_image_features,
#             'generated_text': generated_text_features
#         }
#
#         generation_info = {
#             'quality_scores': quality_scores,
#             'missing_type': missing_type
#         }
#
#         return enhanced_features, generation_info
#
#     def compute_generation_losses(self, enhanced_features, generation_info, targets=None):
#         """计算生成相关损失"""
#         losses = {}
#
#         # 1. 循环一致性损失
#         if self.use_cycle_consistency:
#             # 只对完整样本计算循环一致性
#             complete_mask = torch.tensor([mt == 0 for mt in generation_info['missing_type']])
#             if complete_mask.any():
#                 complete_img = enhanced_features['original_image'][complete_mask]
#                 complete_text = enhanced_features['original_text'][complete_mask]
#
#                 cycle_loss, cycle_outputs = self.cycle_module.compute_cycle_loss(
#                     complete_img, complete_text
#                 )
#                 losses['cycle_consistency'] = cycle_loss
#             else:
#                 losses['cycle_consistency'] = torch.tensor(0.0, device=enhanced_features['image'].device)
#
#         # 2. 质量预测损失（自监督）
#         quality_scores = generation_info['quality_scores']
#
#         # 使用特征相似度作为质量标签
#         img_similarity = F.cosine_similarity(
#             enhanced_features['original_image'],
#             enhanced_features['generated_image'],
#             dim=-1
#         )
#         text_similarity = F.cosine_similarity(
#             enhanced_features['original_text'],
#             enhanced_features['generated_text'],
#             dim=-1
#         )
#
#         quality_target = torch.stack([img_similarity, text_similarity], dim=1)
#         quality_target = torch.clamp(quality_target, 0, 1)
#
#         losses['quality_prediction'] = F.mse_loss(quality_scores, quality_target)
#
#         # 3. 特征平滑损失（防止生成特征过度偏离）
#         img_smooth_loss = F.mse_loss(
#             enhanced_features['generated_image'],
#             enhanced_features['original_image']
#         )
#         text_smooth_loss = F.mse_loss(
#             enhanced_features['generated_text'],
#             enhanced_features['original_text']
#         )
#         losses['smoothness'] = (img_smooth_loss + text_smooth_loss) / 2
#
#         return losses
#
#
# def test_enhanced_modal_generator():
#     """测试模态生成器"""
#     print("Testing Enhanced Modal Generator...")
#
#     # 创建模型
#     generator = EnhancedModalGenerator(
#         feature_dim=512,
#         num_layers=3,
#         num_heads=8,
#         use_cycle_consistency=True
#     )
#
#     # 创建测试数据
#     batch_size = 4
#     image_features = torch.randn(batch_size, 512)
#     text_features = torch.randn(batch_size, 512)
#     missing_type = [0, 1, 2, 1]  # 完整, 缺失文本, 缺失图像, 缺失文本
#
#     # 模拟缺失（置零）
#     for i, miss_type in enumerate(missing_type):
#         if miss_type == 1:  # 缺失文本
#             text_features[i] = 0
#         elif miss_type == 2:  # 缺失图像
#             image_features[i] = 0
#
#     # 前向传播
#     enhanced_features, generation_info = generator(image_features, text_features, missing_type)
#
#     # 计算损失
#     losses = generator.compute_generation_losses(enhanced_features, generation_info)
#
#     print(f"Enhanced image features shape: {enhanced_features['image'].shape}")
#     print(f"Enhanced text features shape: {enhanced_features['text'].shape}")
#     print(f"Quality scores shape: {generation_info['quality_scores'].shape}")
#     print(f"Losses: {list(losses.keys())}")
#     print("✅ Enhanced Modal Generator test passed!")
#
#
# if __name__ == "__main__":
#     test_enhanced_modal_generator()

# clip/modules/modal_generator.py
# clip/modules/modal_generator.py
# clip/modules/modal_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerBlock(nn.Module):
    """标准Transformer块用于跨模态生成"""

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # 转换为 [seq_len, batch, hidden_size] 格式
        x = x.transpose(0, 1)  # [batch, seq, hidden] -> [seq, batch, hidden]
        if context is not None:
            context = context.transpose(0, 1)

        # Self-attention或Cross-attention
        if context is None:
            attn_out, _ = self.attention(x, x, x)
        else:
            attn_out, _ = self.attention(x, context, context)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # 转回 [batch, seq, hidden] 格式
        return x.transpose(0, 1)


class CrossModalGenerator(nn.Module):
    """跨模态生成器：利用现有特征生成增强特征"""

    def __init__(self, hidden_size=512, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入投影层
        self.input_proj = nn.Linear(hidden_size, hidden_size)

        # Transformer生成器
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 输出投影层
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, source_feat, target_feat=None):
        """
        source_feat: 源模态特征 [batch, hidden_size]
        target_feat: 目标模态特征(可选，用于增强) [batch, hidden_size]
        """
        batch_size = source_feat.size(0)

        # 投影到序列维度 [batch, 1, hidden_size]
        x = self.input_proj(source_feat).unsqueeze(1)

        # 如果有目标特征，用作上下文
        context = target_feat.unsqueeze(1) if target_feat is not None else None

        # Transformer生成
        for layer in self.transformer_layers:
            x = layer(x, context)

        # 输出投影 [batch, hidden_size]
        generated = self.output_proj(x.squeeze(1))

        # 如果有目标特征，进行残差连接增强
        if target_feat is not None:
            enhanced = self.residual_weight * target_feat + (1 - self.residual_weight) * generated
            return enhanced

        return generated


class ModalGenerator(nn.Module):
    """模态生成器主模块"""

    def __init__(self, hidden_size=512, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()

        # 图像→文本生成器
        self.img_to_text = CrossModalGenerator(hidden_size, num_layers, num_heads, dropout)

        # 文本→图像生成器
        self.text_to_img = CrossModalGenerator(hidden_size, num_layers, num_heads, dropout)

        # 先验生成器（当两个模态都缺失时）
        self.prior_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * 2),  # 生成图像+文本特征
        )

        # 可学习的先验向量
        self.prior_embedding = nn.Parameter(torch.randn(1, hidden_size))

    def forward(self, image_features, text_features, missing_type):
        """
        image_features: [batch, 512] 图像特征
        text_features: [batch, 512] 文本特征
        missing_type: [batch] 缺失类型 0=完整, 1=缺失文本, 2=缺失图像

        返回: enhanced_image_features, enhanced_text_features
        """
        batch_size = image_features.size(0)
        device = image_features.device

        enhanced_image_features = image_features.clone()
        enhanced_text_features = text_features.clone()

        for i in range(batch_size):
            if missing_type[i] == 0:  # 模态完整，不需要生成
                continue

            elif missing_type[i] == 1:  # 缺失文本，从图像生成文本特征
                generated_text = self.img_to_text(
                    image_features[i:i + 1],
                    text_features[i:i + 1]  # 利用原有文本特征（可能是prompt信息）
                )
                enhanced_text_features[i:i + 1] = generated_text

            elif missing_type[i] == 2:  # 缺失图像，从文本生成图像特征
                generated_image = self.text_to_img(
                    text_features[i:i + 1],
                    image_features[i:i + 1]  # 利用原有图像特征（可能是prompt信息）
                )
                enhanced_image_features[i:i + 1] = generated_image

            elif missing_type[i] == 3:  # 两个模态都缺失，使用先验生成
                prior_input = self.prior_embedding.expand(1, -1)
                generated_both = self.prior_generator(prior_input)

                enhanced_image_features[i:i + 1] = generated_both[:, :512]
                enhanced_text_features[i:i + 1] = generated_both[:, 512:]

        return enhanced_image_features, enhanced_text_features

    def compute_cycle_consistency_loss(self, image_features, text_features, missing_type):
        """计算循环一致性损失 - 简化版本避免编码错误"""
        if image_features is None or text_features is None:
            return torch.tensor(0.0, requires_grad=True)

        cycle_loss = 0.0
        count = 0

        for i in range(image_features.size(0)):
            if missing_type[i] == 0:  # 只对完整样本计算循环损失
                # 图像→文本→图像
                img_to_text = self.img_to_text(image_features[i:i + 1])
                text_to_img = self.text_to_img(img_to_text)
                cycle_loss += F.mse_loss(text_to_img, image_features[i:i + 1])

                # 文本→图像→文本
                text_to_img = self.text_to_img(text_features[i:i + 1])
                img_to_text = self.img_to_text(text_to_img)
                cycle_loss += F.mse_loss(img_to_text, text_features[i:i + 1])

                count += 2

        if count == 0:
            return torch.tensor(0.0, requires_grad=True)

        return cycle_loss / count