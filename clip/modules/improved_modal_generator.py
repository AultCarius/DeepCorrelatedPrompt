# clip/modules/improved_modal_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ImprovedModalGenerator(nn.Module):
    """
    改进的模态生成器 - 在原始数据空间生成缺失模态
    核心功能：
    1. 从完整模态生成缺失模态的原始数据
    2. 替换零填充的缺失数据
    3. 提供生成损失用于训练
    """

    def __init__(self, image_size=224, text_length=77, hidden_dim=512):
        super().__init__()

        self.image_size = image_size
        self.text_length = text_length
        self.hidden_dim = hidden_dim

        # 获取CLIP的实际vocab_size
        try:
            import clip
            _tokenizer = clip._tokenizer
            self.vocab_size = len(_tokenizer.encoder)
            self.start_token = _tokenizer.encoder["<|startoftext|>"]
            self.end_token = _tokenizer.encoder["<|endoftext|>"]
        except:
            # 备用设置
            self.vocab_size = 49408
            self.start_token = 49406
            self.end_token = 49407

        print(f"Modal Generator initialized with vocab_size: {self.vocab_size}")

        # === 图像到文本生成器 ===
        # 输入：RGB图像 [3, 224, 224] -> 输出：文本token indices [77]
        self.img_to_text_generator = nn.Sequential(
            # 图像编码器
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),

            # 特征变换
            nn.Linear(128 * 8 * 8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),

            # 文本解码器 - 修复输出维度
            nn.Linear(hidden_dim // 2, text_length * self.vocab_size),
        )

        # === 文本到图像生成器 ===
        # 输入：文本token indices [77] -> 输出：RGB图像 [3, 224, 224]

        # 文本编码器部分
        self.text_embedding = nn.Embedding(self.vocab_size, 128)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_length * 128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 256 * 7 * 7),
            nn.ReLU(),
        )

        # 图像上采样网络
        self.img_upsampler = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.Tanh(),
        )

        # 最终调整到224x224
        self.final_resize = nn.AdaptiveAvgPool2d((image_size, image_size))

        # === 生成质量评估网络 ===
        # 图像质量网络：接收8192维特征
        self.img_generation_quality_net = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),  # 8192 -> 256
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # 文本质量网络：接收128维特征
        self.text_generation_quality_net = nn.Sequential(
            nn.Linear(128, 64),  # 128 -> 64
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # === 对比学习头（用于训练） ===
        # 图像特征：128*8*8=8192 -> 投影到128维
        self.img_projection = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),  # 8192 -> 256
            nn.ReLU(),
            nn.Linear(256, 128)  # 256 -> 128
        )

        # 文本特征：128维 -> 投影到128维
        self.text_projection = nn.Sequential(
            nn.Linear(128, 256),  # 128 -> 256
            nn.ReLU(),
            nn.Linear(256, 128)  # 256 -> 128
        )

    def preprocess_missing_modalities(self, image_input, text_input, missing_type):
        """
        核心功能：在原始输入空间处理缺失模态

        Args:
            image_input: [batch, 3, 224, 224] 原始图像（缺失时为零填充）
            text_input: [batch, 77] 原始文本token（缺失时为零填充）
            missing_type: [batch] 缺失类型 (0:完整, 1:缺失文本, 2:缺失图像)

        Returns:
            processed_images: [batch, 3, 224, 224] 处理后的图像
            processed_texts: [batch, 77] 处理后的文本
            generation_info: Dict 生成信息
        """
        batch_size = len(missing_type)
        device = image_input.device

        processed_images = image_input.clone()
        processed_texts = text_input.clone()

        generation_info = {
            'generated_mask': torch.zeros(batch_size, 2, device=device),  # [img_generated, text_generated]
            'generation_quality': torch.ones(batch_size, 2, device=device) * 0.8,
            'complete_indices': []
        }

        # 记录完整样本索引
        complete_indices = [i for i, mt in enumerate(missing_type) if mt == 0]
        generation_info['complete_indices'] = complete_indices

        # 处理缺失模态
        for i, miss_type in enumerate(missing_type):
            if miss_type == 1:  # 缺失文本，从图像生成文本
                with torch.no_grad():  # 推理时不需要梯度
                    generated_text_logits = self.img_to_text_generator(image_input[i:i + 1])  # [1, 77*vocab_size]
                    generated_text_logits = generated_text_logits.view(1, self.text_length,
                                                                       self.vocab_size)  # [1, 77, vocab_size]

                    # 转换为token indices
                    generated_tokens = torch.argmax(generated_text_logits, dim=-1)  # [1, 77]

                    # 确保起始和结束token
                    generated_tokens[0, 0] = self.start_token  # start token
                    generated_tokens[0, -1] = self.end_token  # end token

                    # 确保token在有效范围内
                    generated_tokens = torch.clamp(generated_tokens, 0, self.vocab_size - 1)

                    processed_texts[i] = generated_tokens.squeeze(0)

                generation_info['generated_mask'][i, 1] = 1.0

            elif miss_type == 2:  # 缺失图像，从文本生成图像
                with torch.no_grad():
                    # 安全处理文本输入 - 确保token在有效范围内
                    safe_text_input = torch.clamp(text_input[i:i + 1], 0, self.vocab_size - 1)

                    # 处理文本输入
                    text_embedded = self.text_embedding(safe_text_input)  # [1, 77, 128]
                    text_flattened = text_embedded.view(1, -1)  # [1, 77*128]

                    # 生成图像特征
                    img_features = self.text_encoder(text_flattened)  # [1, 256*7*7]
                    img_features = img_features.view(1, 256, 7, 7)  # [1, 256, 7, 7]

                    # 上采样到图像
                    generated_img = self.img_upsampler(img_features)  # [1, 3, 112, 112]
                    generated_img = self.final_resize(generated_img)  # [1, 3, 224, 224]

                    # 限制像素值范围（适应CLIP预处理）
                    generated_img = torch.clamp(generated_img, -2.5, 2.5)

                    processed_images[i] = generated_img.squeeze(0)

                generation_info['generated_mask'][i, 0] = 1.0

        return processed_images, processed_texts, generation_info

    def compute_generation_losses(self, image_input, text_input, missing_type):
        """
        计算生成器的训练损失：只使用完整样本训练

        Args:
            image_input: [batch, 3, 224, 224] 原始图像
            text_input: [batch, 77] 原始文本token
            missing_type: [batch] 缺失类型

        Returns:
            Dict 包含各种训练损失
        """
        device = image_input.device
        total_losses = {
            'generation_consistency_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'contrastive_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'reconstruction_loss': torch.tensor(0.0, device=device, requires_grad=True)
        }

        # 找到完整样本
        complete_indices = [i for i, mt in enumerate(missing_type) if mt == 0]

        if len(complete_indices) < 1:  # 需要至少1个完整样本
            return total_losses

        try:
            # === 1. 简化的重构损失 ===
            reconstruction_loss = torch.tensor(0.0, device=device, requires_grad=True)

            for i in complete_indices[:2]:  # 限制处理数量
                # 确保文本token在有效范围内
                safe_text_input = torch.clamp(text_input[i:i + 1], 0, self.vocab_size - 1)

                # 图像->文本生成
                img_to_text_logits = self.img_to_text_generator(image_input[i:i + 1])  # [1, 77*vocab_size]
                img_to_text_logits = img_to_text_logits.view(1, self.text_length, self.vocab_size)

                # 文本重构损失 - 使用ignore_index忽略padding token
                text_recon_loss = F.cross_entropy(
                    img_to_text_logits.view(-1, self.vocab_size),
                    safe_text_input.view(-1).long(),
                    ignore_index=0,  # 忽略padding token
                    reduction='mean'
                )

                # 文本->图像生成（简化版本）
                text_embedded = self.text_embedding(safe_text_input)  # [1, 77, 128]
                text_flattened = text_embedded.view(1, -1)  # [1, 77*128] = [1, 9856]

                # print(f"Debug: text_flattened shape: {text_flattened.shape}")
                # print(f"Debug: text_encoder expects input dim: {self.text_encoder[0].in_features}")

                # 检查维度匹配
                expected_dim = self.text_length * 128  # 77 * 128 = 9856
                actual_dim = text_flattened.shape[1]

                if actual_dim != expected_dim:
                    print(f"Warning: Dimension mismatch - expected {expected_dim}, got {actual_dim}")
                    # 如果维度不匹配，跳过这个样本
                    continue

                img_features = self.text_encoder(text_flattened)  # [1, 256*7*7]
                img_features = img_features.view(1, 256, 7, 7)  # [1, 256, 7, 7]
                generated_img = self.img_upsampler(img_features)  # [1, 3, H, W]
                generated_img = self.final_resize(generated_img)  # [1, 3, 224, 224]

                # 图像重构损失
                img_recon_loss = F.mse_loss(generated_img, image_input[i:i + 1])

                reconstruction_loss = reconstruction_loss + text_recon_loss + img_recon_loss

            if len(complete_indices) > 0:
                reconstruction_loss = reconstruction_loss / min(2, len(complete_indices))
                total_losses['reconstruction_loss'] = reconstruction_loss

            # === 2. 对比损失（简化版本） ===
            if len(complete_indices) >= 2:
                # 提取特征用于对比学习
                img_features = []
                text_features = []

                for i in complete_indices[:min(2, len(complete_indices))]:  # 限制数量避免内存问题
                    # 图像特征提取 - 到flatten层
                    # img_to_text_generator 结构：Conv->ReLU->MaxPool->Conv->ReLU->AdaptiveAvgPool->Flatten->Linear...
                    img_conv_out = image_input[i:i + 1]  # [1, 3, 224, 224]
                    img_conv_out = self.img_to_text_generator[0](img_conv_out)  # Conv2d [1, 64, 112, 112]
                    img_conv_out = self.img_to_text_generator[1](img_conv_out)  # ReLU
                    img_conv_out = self.img_to_text_generator[2](img_conv_out)  # MaxPool2d [1, 64, 56, 56]
                    img_conv_out = self.img_to_text_generator[3](img_conv_out)  # Conv2d [1, 128, 28, 28]
                    img_conv_out = self.img_to_text_generator[4](img_conv_out)  # ReLU
                    img_conv_out = self.img_to_text_generator[5](img_conv_out)  # AdaptiveAvgPool2d [1, 128, 8, 8]
                    img_feat = self.img_to_text_generator[6](img_conv_out)  # Flatten [1, 128*8*8] = [1, 8192]

                    # print(f"Debug: img_feat shape: {img_feat.shape}")
                    # print(f"Debug: img_projection expects: {self.img_projection[0].in_features}")

                    # 安全的文本处理
                    safe_text = torch.clamp(text_input[i:i + 1], 0, self.vocab_size - 1)
                    text_embedded = self.text_embedding(safe_text)  # [1, 77, 128]
                    text_feat = text_embedded.mean(dim=1)  # 简单平均池化 [1, 128]

                    # print(f"Debug: text_feat shape: {text_feat.shape}")
                    # print(f"Debug: text_projection expects: {self.text_projection[0].in_features}")

                    # 现在维度应该匹配了
                    img_features.append(img_feat)  # [1, 8192]
                    text_features.append(text_feat)  # [1, 128]

                if len(img_features) >= 2:
                    img_features = torch.cat(img_features, dim=0)  # [n, 8192]
                    text_features = torch.cat(text_features, dim=0)  # [n, 128]

                    # 投影到对比空间
                    img_proj = F.normalize(self.img_projection(img_features), dim=-1)  # [n, 128]
                    text_proj = F.normalize(self.text_projection(text_features), dim=-1)  # [n, 128]

                    # 对比损失
                    temperature = 0.1  # 降低温度避免数值不稳定
                    sim_matrix = torch.matmul(img_proj, text_proj.T) / temperature
                    labels = torch.arange(len(img_features), device=device)

                    img_to_text_loss = F.cross_entropy(sim_matrix, labels)
                    text_to_img_loss = F.cross_entropy(sim_matrix.T, labels)

                    contrastive_loss = (img_to_text_loss + text_to_img_loss) / 2
                    total_losses['contrastive_loss'] = contrastive_loss

            # === 3. 生成一致性损失（大幅简化） ===
            consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # 直接使用简单的特征而不是通过复杂网络
            for i in complete_indices[:1]:  # 只处理1个样本
                # 使用简单的特征
                img_simple_feat = torch.mean(image_input[i:i + 1].view(1, -1), dim=1, keepdim=True)  # [1, 1]
                text_simple_feat = torch.mean(text_input[i:i + 1].float(), dim=1, keepdim=True)  # [1, 1]

                # 简单的质量损失：特征应该有合理的范围
                img_quality_loss = F.mse_loss(torch.sigmoid(img_simple_feat), torch.tensor(0.8, device=device))
                text_quality_loss = F.mse_loss(torch.sigmoid(text_simple_feat), torch.tensor(0.8, device=device))

                consistency_loss = consistency_loss + img_quality_loss + text_quality_loss

            total_losses['generation_consistency_loss'] = consistency_loss

        except Exception as e:
            print(f"Warning: Generation loss computation failed: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            # 返回零损失而不是崩溃
            pass

        return total_losses


def forward(self, image_features, text_features, missing_type):
    """
    兼容性接口：保持与原有代码的兼容性
    现在主要功能已转移到preprocess_missing_modalities
    """
    generation_info = {
        'generated_mask': torch.zeros(len(missing_type), 2, device=image_features.device),
        'generation_quality': torch.ones(len(missing_type), 2, device=image_features.device) * 0.8,
    }

    return {
        'enhanced_image_features': image_features,
        'enhanced_text_features': text_features,
        'generation_info': generation_info
    }


def compute_all_generation_losses(self, image_features, text_features, missing_type):
    """
    计算所有生成相关损失的简化接口

    Args:
        image_features: [batch, dim] 图像特征
        text_features: [batch, dim] 文本特征
        missing_type: [batch] 缺失类型

    Returns:
        Dict 包含生成损失
    """
    device = image_features.device

    # 简化的损失计算 - 避免复杂的重构过程
    losses = {
        'contrastive_loss': torch.tensor(0.0, device=device, requires_grad=True),
        'generation_consistency_loss': torch.tensor(0.0, device=device, requires_grad=True),
        'generation_quality_loss': torch.tensor(0.0, device=device, requires_grad=True)
    }

    # 找到完整样本
    complete_indices = [i for i, mt in enumerate(missing_type) if mt == 0]

    if len(complete_indices) >= 2:
        try:
            # 简化的对比损失
            complete_img_feats = image_features[complete_indices]  # [n, dim]
            complete_text_feats = text_features[complete_indices]  # [n, dim]

            # 假设特征维度一致，如果不一致需要投影
            if complete_img_feats.shape[1] != complete_text_feats.shape[1]:
                # 创建临时投影层使维度一致
                if not hasattr(self, 'temp_img_proj'):
                    self.temp_img_proj = nn.Linear(complete_img_feats.shape[1], 128).to(device)
                if not hasattr(self, 'temp_text_proj'):
                    self.temp_text_proj = nn.Linear(complete_text_feats.shape[1], 128).to(device)

                img_proj = F.normalize(self.temp_img_proj(complete_img_feats), dim=-1)
                text_proj = F.normalize(self.temp_text_proj(complete_text_feats), dim=-1)
            else:
                img_proj = F.normalize(complete_img_feats, dim=-1)
                text_proj = F.normalize(complete_text_feats, dim=-1)

            # 对比损失
            temperature = 0.1
            sim_matrix = torch.matmul(img_proj, text_proj.T) / temperature
            labels = torch.arange(len(complete_indices), device=device)

            contrastive_loss = (F.cross_entropy(sim_matrix, labels) +
                                F.cross_entropy(sim_matrix.T, labels)) / 2
            losses['contrastive_loss'] = contrastive_loss

            # 简化的生成质量损失
            quality_loss = torch.tensor(0.0, device=device, requires_grad=True)
            target = torch.tensor(0.8, device=device)

            for i in complete_indices[:2]:  # 限制数量
                # 简单的质量评估：特征范数应该稳定
                img_norm = torch.norm(image_features[i], p=2)
                text_norm = torch.norm(text_features[i], p=2)

                # 范数应该在合理范围内
                img_quality_loss = F.mse_loss(torch.sigmoid(img_norm / 10), target)
                text_quality_loss = F.mse_loss(torch.sigmoid(text_norm / 10), target)

                quality_loss = quality_loss + img_quality_loss + text_quality_loss

            losses['generation_quality_loss'] = quality_loss / min(2, len(complete_indices))

        except Exception as e:
            print(f"Warning: Simplified generation loss failed: {e}")
            pass

    return losses