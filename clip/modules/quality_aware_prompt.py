# clip/modules/quality_aware_prompt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class QualityAwarePromptLearner(nn.Module):
    """质量感知的Prompt学习器"""

    def __init__(self, original_prompt_learner, prompt_length, prompt_depth):
        super().__init__()
        self.base_prompt_learner = original_prompt_learner
        self.prompt_length = prompt_length
        self.prompt_depth = prompt_depth

        # 从MultiModalPromptLearner分析得出的prompt长度
        self.prompt_length_half = prompt_length // 3  # 12
        self.first_layer_length = self.prompt_length_half * 2  # 24 (static + dynamic)
        self.other_layer_length = self.prompt_length_half  # 12 (compound only)

        # 质量信息提取器
        self.quality_summarizer = nn.Sequential(
            nn.Linear(8, 32),  # 假设8维质量向量
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU()
        )

        # Prompt调制器 - 为不同层生成不同长度的调制因子
        self.first_layer_modulator = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, self.first_layer_length),  # 24
            nn.Tanh()
        )

        self.other_layer_modulator = nn.Sequential(
            nn.Linear(16, 64),
            nn.GELU(),
            nn.Linear(64, self.other_layer_length),  # 12
            nn.Tanh()
        )

        # 自适应权重 - 决定使用多少质量调制
        self.adaptation_weight = nn.Parameter(torch.tensor(0.1))

    def extract_quality_vector(self, quality_scores):
        """从质量字典提取特征向量"""
        if quality_scores is None:
            return None

        batch_size = len(quality_scores)
        quality_vectors = []

        for i in range(batch_size):
            quality = quality_scores[i]

            # 提取关键质量指标
            img_quality = quality['image_intrinsic'].mean().item()
            text_quality = quality['text_intrinsic'].mean().item()
            consistency = quality['cross_modal_consistency'].item()
            confidence = quality['generation_confidence'].item()

            # 计算衍生指标
            overall_quality = (img_quality + text_quality) / 2
            quality_variance = abs(img_quality - text_quality)
            weighted_consistency = consistency * overall_quality
            reliability_score = confidence * consistency

            quality_vec = torch.tensor([
                img_quality, text_quality, consistency, confidence,
                overall_quality, quality_variance, weighted_consistency, reliability_score
            ]).to(quality['image_intrinsic'].device)

            quality_vectors.append(quality_vec)

        return torch.stack(quality_vectors)  # [batch, 8]

    def forward(self, missing_type, quality_scores=None):
        """
        Args:
            missing_type: [batch] 缺失类型
            quality_scores: List[Dict] 质量分数 (可选)
        """
        # 获取基础prompts
        base_prompts_image, base_prompts_text = self.base_prompt_learner(missing_type)

        # 如果没有质量信息，直接返回基础prompts
        if quality_scores is None:
            return base_prompts_image, base_prompts_text

        # 提取质量向量
        quality_vectors = self.extract_quality_vector(quality_scores)  # [batch, 8]
        quality_features = self.quality_summarizer(quality_vectors)  # [batch, 16]

        # 为不同层生成不同长度的调制因子
        first_layer_modulation = self.first_layer_modulator(quality_features)  # [batch, 24]
        other_layer_modulation = self.other_layer_modulator(quality_features)  # [batch, 12]

        # 应用调制到prompts
        enhanced_prompts_image = []
        enhanced_prompts_text = []

        for depth in range(self.prompt_depth):
            # 根据层数选择合适的调制因子
            if depth == 0:
                # 第0层使用24长度的调制
                img_modulation = first_layer_modulation.unsqueeze(-1)  # [batch, 24, 1]
                text_modulation = first_layer_modulation.unsqueeze(-1)  # [batch, 24, 1]
            else:
                # 其他层使用12长度的调制
                img_modulation = other_layer_modulation.unsqueeze(-1)  # [batch, 12, 1]
                text_modulation = other_layer_modulation.unsqueeze(-1)  # [batch, 12, 1]

            # 图像prompts调制
            if depth < len(base_prompts_image):
                original_img_prompt = base_prompts_image[depth]  # [batch, actual_length, 768]

                # 确保调制因子长度匹配
                actual_length = original_img_prompt.size(1)
                if depth == 0:
                    assert actual_length == self.first_layer_length, f"第0层期望长度{self.first_layer_length}，实际{actual_length}"
                    modulation = img_modulation
                else:
                    assert actual_length == self.other_layer_length, f"第{depth}层期望长度{self.other_layer_length}，实际{actual_length}"
                    modulation = img_modulation

                # 自适应调制
                enhanced_img_prompt = original_img_prompt * (1 + self.adaptation_weight * modulation)
                enhanced_prompts_image.append(enhanced_img_prompt)
            else:
                enhanced_prompts_image.append(base_prompts_image[depth])

            # 文本prompts调制 (类似逻辑)
            if depth < len(base_prompts_text):
                original_text_prompt = base_prompts_text[depth]  # [batch, actual_length, 512]

                actual_length = original_text_prompt.size(1)
                if depth == 0:
                    modulation = text_modulation
                else:
                    modulation = text_modulation

                enhanced_text_prompt = original_text_prompt * (1 + self.adaptation_weight * modulation)
                enhanced_prompts_text.append(enhanced_text_prompt)
            else:
                enhanced_prompts_text.append(base_prompts_text[depth])

        return enhanced_prompts_image, enhanced_prompts_text


class IterativeQualityOptimization(nn.Module):
    """迭代质量优化模块"""

    def __init__(self, clip_model_components):
        super().__init__()
        self.image_encoder = clip_model_components['image_encoder']
        self.text_encoder = clip_model_components['text_encoder']
        self.modal_generator = clip_model_components['modal_generator']
        self.quality_estimator = clip_model_components['quality_estimator']
        self.quality_fusion = clip_model_components['quality_fusion']
        self.quality_prompt_learner = clip_model_components['quality_prompt_learner']

        # 迭代控制参数
        self.max_iterations = 2
        self.early_stop_threshold = 0.01  # 质量改进小于此值时提前停止

    def forward(self, image, text, missing_type, tokenized_texts):
        """迭代优化过程"""

        best_quality_score = 0
        best_features = None
        quality_history = []

        # 第一次: 使用基础prompts
        current_prompts_img, current_prompts_text = self.quality_prompt_learner(missing_type, None)

        for iteration in range(self.max_iterations):
            # 编码
            image_features = self.image_encoder(image, current_prompts_img, missing_type)
            text_features = self.text_encoder(tokenized_texts, current_prompts_text, missing_type)

            # 生成增强特征
            enhanced_image, enhanced_text = self.modal_generator(
                image_features, text_features, missing_type
            )

            # 质量评估
            quality_scores = self.quality_estimator(
                image_features, text_features, enhanced_image, enhanced_text, missing_type
            )

            # 融合特征
            fused_features = self.quality_fusion(
                image_features, text_features, enhanced_image, enhanced_text,
                quality_scores, missing_type
            )

            # 计算平均质量分数
            avg_quality = self.compute_average_quality(quality_scores)
            quality_history.append(avg_quality)

            # 检查是否是最佳结果
            if avg_quality > best_quality_score:
                best_quality_score = avg_quality
                best_features = fused_features
                best_image_features = image_features
                best_text_features = text_features

            # 早停检查
            if iteration > 0 and abs(quality_history[-1] - quality_history[-2]) < self.early_stop_threshold:
                break

            # 为下一次迭代准备质量感知prompts
            if iteration < self.max_iterations - 1:
                current_prompts_img, current_prompts_text = self.quality_prompt_learner(
                    missing_type, quality_scores
                )

        # 返回最佳结果
        return best_features, best_image_features, best_text_features

    def compute_average_quality(self, quality_scores):
        """计算批次的平均质量分数"""
        total_quality = 0
        for quality in quality_scores:
            sample_quality = (
                                     quality['image_intrinsic'].mean() +
                                     quality['text_intrinsic'].mean() +
                                     quality['cross_modal_consistency'] +
                                     quality['generation_confidence']
                             ) / 4
            total_quality += sample_quality.item()

        return total_quality / len(quality_scores)