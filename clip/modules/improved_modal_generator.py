# clip/modules/improved_modal_generator.py - 完全重写
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class ImprovedModalGenerator(nn.Module):
    """
    重写版：在原始数据空间进行模态生成
    核心思路：
    1. 完整样本训练生成器
    2. 缺失样本用生成器生成原始数据替换零填充
    3. 生成的是可以直接输入CLIP编码器的原始数据
    """

    def __init__(self, hidden_size=512, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()

        # === 原始数据空间的生成器 ===
        # 图像特征 -> 文本tokens (CLIP tokenizer 兼容)
        self.img_to_text_generator = nn.Sequential(
            nn.Linear(768, 512),  # 图像编码维度到中间维度
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 77),  # 输出77个token位置的logits
            nn.Tanh()  # 限制输出范围
        )

        # 文本特征 -> 图像pixels
        self.text_to_img_generator = nn.Sequential(
            nn.Linear(512, 1024),  # 文本编码维度到中间维度
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 3 * 224 * 224),  # RGB图像像素
            nn.Tanh()  # 输出范围[-1, 1]
        )

        # === 获取正确的vocab_size ===
        from . import clip
        _tokenizer = clip._tokenizer
        self.vocab_size = len(_tokenizer.encoder)  # 动态获取真实vocab_size
        self.start_token = _tokenizer.encoder["<|startoftext|>"]
        self.end_token = _tokenizer.encoder["<|endoftext|>"]

        # === 用于训练的临时编码器（轻量级） ===
        self.temp_img_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((14, 14)),  # 简化的图像编码
            nn.Flatten(),
            nn.Linear(3 * 14 * 14, 768),
            nn.GELU(),
            nn.Linear(768, 768)
        )

        self.temp_text_encoder = nn.Sequential(
            nn.Embedding(self.vocab_size, 512),  # 使用真实vocab_size
            nn.GELU(),
            nn.Linear(512, 512)
        )

        # === 对比学习头（用于训练） ===
        self.img_contrastive_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.text_contrastive_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def preprocess_missing_modalities(self, image_input, text_input, missing_type):
        """
        === 核心功能：在输入编码器前处理缺失模态 ===

        Args:
            image_input: [batch, 3, 224, 224] 原始图像（缺失时为零）
            text_input: [batch, 77] 原始文本token（缺失时为零）
            missing_type: [batch] 缺失类型

        Returns:
            processed_images: [batch, 3, 224, 224] 处理后的图像
            processed_texts: [batch, 77] 处理后的文本token
            generation_info: Dict 生成信息
        """
        batch_size = len(missing_type)
        device = image_input.device

        processed_images = image_input.clone()
        processed_texts = text_input.clone()

        generation_info = {
            'generated_mask': torch.zeros(batch_size, 2).to(device),  # [img_generated, text_generated]
            'complete_indices': []  # 完整样本的索引
        }

        # 记录完整样本索引（用于训练）
        for i, miss_type in enumerate(missing_type):
            if miss_type == 0:
                generation_info['complete_indices'].append(i)

        # 处理缺失模态
        for i, miss_type in enumerate(missing_type):
            if miss_type == 1:  # 缺失文本
                # 从图像生成文本token
                with torch.no_grad():  # 推理时不需要梯度
                    # 使用临时编码器获取图像特征
                    img_feat = self.temp_img_encoder(image_input[i:i + 1])  # [1, 768]

                    # 生成文本token的logits
                    text_logits = self.img_to_text_generator(img_feat)  # [1, 77]

                    # 转换为token indices
                    # 将tanh输出[-1,1]映射到token范围[0, vocab_size-1]
                    text_tokens = ((text_logits + 1) / 2 * (self.vocab_size - 1)).long()  # [1, 77]
                    text_tokens = torch.clamp(text_tokens, 0, self.vocab_size - 1)

                    # 确保起始和结束token正确
                    text_tokens[0, 0] = self.start_token  # start token
                    text_tokens[0, -1] = self.end_token  # end token

                    processed_texts[i] = text_tokens.squeeze(0)  # [77]

                generation_info['generated_mask'][i, 1] = 1.0

            elif miss_type == 2:  # 缺失图像
                # 从文本生成图像
                with torch.no_grad():
                    # 使用临时编码器获取文本特征
                    # 注意：需要先对token进行embedding
                    text_embedded = self.temp_text_encoder[0](text_input[i:i + 1])  # [1, 77, 512]
                    text_feat = text_embedded.mean(dim=1)  # [1, 512] 简单平均池化

                    # 生成图像像素
                    img_pixels = self.text_to_img_generator(text_feat)  # [1, 3*224*224]
                    img_pixels = img_pixels.view(1, 3, 224, 224)  # [1, 3, 224, 224]

                    # 确保像素值在合理范围内
                    img_pixels = torch.clamp(img_pixels, -2.5, 2.5)  # CLIP的输入范围

                    processed_images[i] = img_pixels.squeeze(0)  # [3, 224, 224]

                generation_info['generated_mask'][i, 0] = 1.0

        return processed_images, processed_texts, generation_info

    def compute_generation_losses(self, image_input, text_input, missing_type):
        """
        === 训练时计算生成损失：只使用完整样本 ===

        核心思路：
        1. 找到完整样本（两个模态都有）
        2. 模拟缺失：从一个模态生成另一个模态
        3. 与真实模态计算损失
        """
        device = image_input.device
        total_losses = {
            'generation_consistency_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'contrastive_loss': torch.tensor(0.0, device=device, requires_grad=True)
        }

        # 找到完整样本
        complete_indices = [i for i, mt in enumerate(missing_type) if mt == 0]

        if len(complete_indices) < 2:  # 需要至少2个完整样本
            return total_losses

        # === 1. 生成一致性损失 ===
        consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for i in complete_indices:
            # 从图像生成文本，与真实文本比较
            img_feat = self.temp_img_encoder(image_input[i:i + 1])
            generated_text_logits = self.img_to_text_generator(img_feat)  # [1, 77]

            # 真实文本的one-hot编码
            real_text_onehot = F.one_hot(text_input[i:i + 1].long(),
                                         num_classes=self.vocab_size).float()  # [1, 77, vocab]

            # 将generated logits转换为概率分布
            generated_text_probs = torch.softmax(generated_text_logits.unsqueeze(-1).expand(-1, -1, self.vocab_size),
                                                 dim=-1)

            # KL散度损失
            text_gen_loss = F.kl_div(
                F.log_softmax(generated_text_probs.view(-1, self.vocab_size), dim=-1),
                real_text_onehot.view(-1, self.vocab_size),
                reduction='batchmean'
            )

            # 从文本生成图像，与真实图像比较
            text_embedded = self.temp_text_encoder[0](text_input[i:i + 1])
            text_feat = text_embedded.mean(dim=1)
            generated_img_pixels = self.text_to_img_generator(text_feat)  # [1, 3*224*224]
            generated_img = generated_img_pixels.view(1, 3, 224, 224)

            # 真实图像
            real_img = image_input[i:i + 1]

            # L2损失
            img_gen_loss = F.mse_loss(generated_img, real_img)

            consistency_loss = consistency_loss + text_gen_loss + img_gen_loss

        consistency_loss = consistency_loss / len(complete_indices)
        total_losses['generation_consistency_loss'] = consistency_loss

        # === 2. 对比学习损失 ===
        if len(complete_indices) >= 2:
            # 提取完整样本的特征
            complete_imgs = image_input[complete_indices]
            complete_texts = text_input[complete_indices]

            # 编码
            img_features = self.temp_img_encoder(complete_imgs)  # [n, 768]
            text_embedded = self.temp_text_encoder[0](complete_texts)  # [n, 77, 512]
            text_features = text_embedded.mean(dim=1)  # [n, 512]

            # 对比学习投影
            img_proj = F.normalize(self.img_contrastive_head(img_features), dim=-1)  # [n, 128]
            text_proj = F.normalize(self.text_contrastive_head(text_features), dim=-1)  # [n, 128]

            # 对比损失
            temperature = 0.07
            sim_matrix = torch.matmul(img_proj, text_proj.T) / temperature  # [n, n]
            labels = torch.arange(len(complete_indices)).to(device)

            img_to_text_loss = F.cross_entropy(sim_matrix, labels)
            text_to_img_loss = F.cross_entropy(sim_matrix.T, labels)

            contrastive_loss = (img_to_text_loss + text_to_img_loss) / 2
            total_losses['contrastive_loss'] = contrastive_loss

        return total_losses

    def forward(self, image_features, text_features, missing_type):
        """
        === 保留兼容性接口（主要用于旧的特征空间处理） ===
        现在这个方法主要用于保持接口兼容性
        """
        # 这个方法现在主要返回输入特征，不做实际生成
        generation_info = {
            'generated_mask': torch.zeros(len(missing_type), 2).to(image_features.device),
            'generation_quality': torch.ones(len(missing_type), 2).to(image_features.device),
        }

        return {
            'enhanced_image_features': image_features,
            'enhanced_text_features': text_features,
            'generation_info': generation_info
        }

# === 冗余部分：可以删除的类 ===
# class CrossModalGenerator(nn.Module):
#     """这个类现在是冗余的，可以删除"""
#     pass

# class TransformerBlock(nn.Module):
#     """这个类现在是冗余的，可以删除"""
#     pass