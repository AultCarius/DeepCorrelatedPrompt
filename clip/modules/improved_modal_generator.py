# clip/modules/improved_modal_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImprovedModalGenerator(nn.Module):
    """改进的模态生成器 - 更好的训练策略和特征替代机制"""

    def __init__(self, hidden_size=512, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()

        # 图像→文本生成器
        self.img_to_text = CrossModalGenerator(hidden_size, num_layers, num_heads, dropout)

        # 文本→图像生成器
        self.text_to_img = CrossModalGenerator(hidden_size, num_layers, num_heads, dropout)

        # 对比学习头（用于训练）
        self.contrastive_head_img = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )

        self.contrastive_head_text = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )

        # 质量预测器（预测生成特征的质量）
        self.generation_quality_predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, image_features, text_features, missing_type):
        """
        主要前向传播，同时进行生成和特征替代

        Args:
            image_features: [batch, 512] 图像特征（缺失时为提示学习输出）
            text_features: [batch, 512] 文本特征（缺失时为提示学习输出）
            missing_type: [batch] 缺失类型

        Returns:
            enhanced_features: Dict 包含所有特征的字典
        """
        batch_size = image_features.size(0)
        device = image_features.device

        # 初始化增强特征
        enhanced_image_features = image_features.clone()
        enhanced_text_features = text_features.clone()

        # 记录生成信息
        generation_info = {
            'generated_mask': torch.zeros(batch_size, 2).to(device),  # [img_generated, text_generated]
            'generation_quality': torch.ones(batch_size, 2).to(device),  # 生成质量分数
            'original_features': {
                'image': image_features.clone(),
                'text': text_features.clone()
            }
        }

        # 处理每种缺失类型
        for i, miss_type in enumerate(missing_type):
            if miss_type == 1:  # 缺失文本
                # 从图像生成文本特征
                generated_text = self.img_to_text(
                    image_features[i:i + 1],
                    text_features[i:i + 1]  # 利用提示学习的输出作为先验
                )

                # 预测生成质量
                quality_input = torch.cat([image_features[i:i + 1], generated_text], dim=-1)
                gen_quality = self.generation_quality_predictor(quality_input)

                # 替换特征
                enhanced_text_features[i:i + 1] = generated_text
                generation_info['generated_mask'][i, 1] = 1.0
                generation_info['generation_quality'][i, 1] = gen_quality.squeeze()

            elif miss_type == 2:  # 缺失图像
                # 从文本生成图像特征
                generated_image = self.text_to_img(
                    text_features[i:i + 1],
                    image_features[i:i + 1]  # 利用提示学习的输出作为先验
                )

                # 预测生成质量
                quality_input = torch.cat([text_features[i:i + 1], generated_image], dim=-1)
                gen_quality = self.generation_quality_predictor(quality_input)

                # 替换特征
                enhanced_image_features[i:i + 1] = generated_image
                generation_info['generated_mask'][i, 0] = 1.0
                generation_info['generation_quality'][i, 0] = gen_quality.squeeze()

        return {
            'enhanced_image_features': enhanced_image_features,
            'enhanced_text_features': enhanced_text_features,
            'generation_info': generation_info
        }

    def compute_contrastive_loss(self, image_features, text_features, missing_type, temperature=0.07):
        """
        利用完整样本计算对比学习损失，训练生成器

        Args:
            image_features: [batch, 512] 图像特征
            text_features: [batch, 512] 文本特征
            missing_type: [batch] 缺失类型

        Returns:
            contrastive_loss: 对比学习损失
        """
        # 只使用完整样本进行对比学习
        complete_mask = torch.tensor([mt == 0 for mt in missing_type]).to(image_features.device)

        if not complete_mask.any():
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        # 提取完整样本
        complete_img = image_features[complete_mask]  # [n_complete, 512]
        complete_text = text_features[complete_mask]  # [n_complete, 512]

        if complete_img.size(0) < 2:  # 需要至少2个完整样本
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        # 投影到对比空间
        img_proj = F.normalize(self.contrastive_head_img(complete_img), dim=-1)  # [n_complete, 128]
        text_proj = F.normalize(self.contrastive_head_text(complete_text), dim=-1)  # [n_complete, 128]

        # 计算相似度矩阵
        sim_matrix = torch.matmul(img_proj, text_proj.T) / temperature  # [n_complete, n_complete]

        # 对比学习损失（InfoNCE）
        labels = torch.arange(complete_img.size(0)).to(image_features.device)

        # 图像到文本
        img_to_text_loss = F.cross_entropy(sim_matrix, labels)
        # 文本到图像
        text_to_img_loss = F.cross_entropy(sim_matrix.T, labels)

        contrastive_loss = (img_to_text_loss + text_to_img_loss) / 2

        return contrastive_loss

    def compute_generation_consistency_loss(self, image_features, text_features, missing_type):
        """
        计算生成一致性损失：生成的特征应该与真实模态相似
        利用完整样本作为监督信号
        """
        consistency_loss = 0.0
        count = 0

        # 利用完整样本训练生成器
        complete_indices = [i for i, mt in enumerate(missing_type) if mt == 0]

        if len(complete_indices) < 2:
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        for i in complete_indices:
            # 模拟缺失情况，测试生成质量
            img_feat = image_features[i:i + 1]
            text_feat = text_features[i:i + 1]

            # 从图像生成文本，与真实文本比较
            generated_text = self.img_to_text(img_feat, text_feat)
            text_consistency_loss = F.mse_loss(generated_text, text_feat)

            # 从文本生成图像，与真实图像比较
            generated_image = self.text_to_img(text_feat, img_feat)
            img_consistency_loss = F.mse_loss(generated_image, img_feat)

            consistency_loss += (text_consistency_loss + img_consistency_loss)
            count += 2

        return consistency_loss / max(count, 1)

    def compute_cycle_consistency_loss(self, image_features, text_features, missing_type):
        """
        计算循环一致性损失：图像→文本→图像 应该回到原图像
        """
        cycle_loss = 0.0
        count = 0

        # 只对完整样本计算循环损失
        complete_indices = [i for i, mt in enumerate(missing_type) if mt == 0]

        if len(complete_indices) == 0:
            return torch.tensor(0.0, device=image_features.device, requires_grad=True)

        for i in complete_indices:
            img_feat = image_features[i:i + 1]
            text_feat = text_features[i:i + 1]

            # 图像→文本→图像
            fake_text = self.img_to_text(img_feat, text_feat)
            reconstructed_img = self.text_to_img(fake_text, img_feat)
            img_cycle_loss = F.mse_loss(reconstructed_img, img_feat)

            # 文本→图像→文本
            fake_img = self.text_to_img(text_feat, img_feat)
            reconstructed_text = self.img_to_text(fake_img, text_feat)
            text_cycle_loss = F.mse_loss(reconstructed_text, text_feat)

            cycle_loss += (img_cycle_loss + text_cycle_loss)
            count += 2

        return cycle_loss / max(count, 1)

    def compute_all_generation_losses(self, image_features, text_features, missing_type):
        """
        计算所有生成相关的损失

        Returns:
            dict: 包含各种损失的字典
        """
        losses = {}

        # 1. 对比学习损失
        losses['contrastive_loss'] = self.compute_contrastive_loss(
            image_features, text_features, missing_type
        )

        # 2. 生成一致性损失
        losses['generation_consistency_loss'] = self.compute_generation_consistency_loss(
            image_features, text_features, missing_type
        )

        # 3. 循环一致性损失
        losses['cycle_consistency_loss'] = self.compute_cycle_consistency_loss(
            image_features, text_features, missing_type
        )

        # 4. 生成质量损失（自监督）
        generation_results = self.forward(image_features, text_features, missing_type)
        generation_quality = generation_results['generation_info']['generation_quality']

        # 期望高质量生成
        quality_target = torch.ones_like(generation_quality) * 0.8  # 目标质量
        losses['generation_quality_loss'] = F.mse_loss(generation_quality, quality_target)

        return losses


class CrossModalGenerator(nn.Module):
    """跨模态生成器的改进版本"""

    def __init__(self, hidden_size=512, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 输入投影
        self.input_proj = nn.Linear(hidden_size, hidden_size)

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh()
        )

        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, source_feat, target_feat_prior=None):
        """
        Args:
            source_feat: [batch, hidden_size] 源模态特征
            target_feat_prior: [batch, hidden_size] 目标模态的先验（来自提示学习）
        """
        # 投影到序列维度
        x = self.input_proj(source_feat).unsqueeze(1)  # [batch, 1, hidden_size]

        # 如果有先验，用作上下文
        if target_feat_prior is not None:
            context = target_feat_prior.unsqueeze(1)  # [batch, 1, hidden_size]

            # 将先验作为额外的上下文
            x = torch.cat([x, context], dim=1)  # [batch, 2, hidden_size]

        # Transformer生成
        for layer in self.transformer_layers:
            x = layer(x)

        # 提取生成的特征（第一个位置）
        generated = self.output_proj(x[:, 0, :])  # [batch, hidden_size]

        # 如果有先验，进行残差连接
        if target_feat_prior is not None:
            generated = self.residual_weight * target_feat_prior + (1 - self.residual_weight) * generated

        return generated


class TransformerBlock(nn.Module):
    """Transformer块"""

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

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        x = x.transpose(0, 1)  # [seq_len, batch, hidden_size]

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x.transpose(0, 1)  # [batch, seq_len, hidden_size]