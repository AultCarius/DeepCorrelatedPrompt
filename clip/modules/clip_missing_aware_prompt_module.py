import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip.modules.vision_transformer_prompts as vit
import math
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from clip.modules import clip_utils, heads, objectives, clip
import copy

# ==========================================
from .modal_generator import ModalGenerator
import torch.nn.functional as F
from .quality_estimator import QualityEstimator
from .quality_guide_fusion import QualityGuidedFusion
# ====质量提示
from .quality_aware_prompt import QualityAwarePromptLearner, IterativeQualityOptimization
# ====新增新质量与新融合
from  .enhanced_quality_estimator import EnhancedQualityEstimator
from  .enhanced_quality_guide_fusion import QualityAwareFusion


def load_clip_to_cpu(backbone_name, prompt_length, prompt_depth):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")  # .eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = vit.build_model(state_dict or model.state_dict(), prompt_length, prompt_depth)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.prompt_length = clip_model.prompt_length

    def forward(self, tokenized_texts, all_prompts_text, missing_type):
        x = self.token_embedding(tokenized_texts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, all_prompts_text, 0, missing_type]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        # x = outputs[0][self.prompt_length:]  # extract the x back from here

        x = outputs[0]
        # 【修改】动态提取文本特征
        # 根据实际提示长度动态调整
        if len(all_prompts_text) > 0 and all_prompts_text[0] is not None:
            actual_prompt_length = all_prompts_text[0].shape[1]
            x = x[actual_prompt_length:, :, :]  # 移除提示部分
        else:
            x = x[self.prompt_length:, :, :]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection

        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class QualityInformedPromptGenerator(nn.Module):
    """质量信息注入的提示生成器"""

    def __init__(self, prompt_length_half, prompt_depth, dtype):
        super().__init__()
        self.prompt_length_half = prompt_length_half  # 12
        self.prompt_depth = prompt_depth
        self.dtype = dtype

        # 质量维度定义：6个主要质量指标
        # [img_norm_stability, img_entropy, img_task_relevance,
        #  text_norm_stability, text_entropy, text_task_relevance]
        self.quality_dim = 6

        # 基础质量提示模板 - 为图像和文本分别设计
        self.base_quality_prompt_image = nn.Parameter(
            nn.init.normal_(torch.empty(self.prompt_length_half, 768, dtype=dtype), std=0.02)
        )
        self.base_quality_prompt_text = nn.Parameter(
            nn.init.normal_(torch.empty(self.prompt_length_half, 512, dtype=dtype), std=0.02)
        )

        # 质量信息注入网络 - 图像分支
        self.quality_injector_image = nn.Sequential(
            nn.Linear(self.quality_dim, 256),
            nn.GELU(),
            nn.Linear(256, 768),  # 输出到图像维度
            nn.Tanh()
        )

        # 质量信息注入网络 - 文本分支
        self.quality_injector_text = nn.Sequential(
            nn.Linear(self.quality_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),  # 输出到文本维度
            nn.Tanh()
        )

        # 自适应混合权重网络
        self.mixing_weight_calculator = nn.Sequential(
            nn.Linear(self.quality_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 深度相关的质量投影网络（用于不同层）
        embed_dim_text = 512
        embed_dim_image = 768
        embed_dim = embed_dim_text + embed_dim_image
        r = 16

        # 为每一层创建质量投影网络
        single_layer_image = nn.Sequential(
            nn.Linear(embed_dim + self.quality_dim, embed_dim // r),
            nn.GELU(),
            nn.Linear(embed_dim // r, embed_dim_image),
        )
        self.quality_projections_image = _get_clones(single_layer_image, self.prompt_depth)

        single_layer_text = nn.Sequential(
            nn.Linear(embed_dim + self.quality_dim, embed_dim // r),
            nn.GELU(),
            nn.Linear(embed_dim // r, embed_dim_text),
        )
        self.quality_projections_text = _get_clones(single_layer_text, self.prompt_depth)

        # LayerNorm for quality-enhanced features
        self.layernorm_quality_image = nn.ModuleList([
            torch.nn.LayerNorm(embed_dim + self.quality_dim) for _ in range(self.prompt_depth)
        ])
        self.layernorm_quality_text = nn.ModuleList([
            torch.nn.LayerNorm(embed_dim + self.quality_dim) for _ in range(self.prompt_depth)
        ])

    def extract_quality_vector(self, quality_scores):
        """
        从SimplifiedQualityEstimator的输出中提取质量向量

        Args:
            quality_scores: List[Dict] 来自SimplifiedQualityEstimator的质量分数

        Returns:
            quality_vectors: [batch_size, quality_dim] 质量向量
        """
        batch_size = len(quality_scores)
        quality_vectors = []

        for quality in quality_scores:
            # 提取图像质量特征
            img_math = quality['image_quality']['mathematical']
            img_norm_stability = self._safe_extract_value(img_math['norm_stability'])
            img_entropy = self._safe_extract_value(img_math['information_entropy'])
            img_task_relevance = self._safe_extract_value(quality['image_quality']['task_relevance'])

            # 提取文本质量特征
            text_math = quality['text_quality']['mathematical']
            text_norm_stability = self._safe_extract_value(text_math['norm_stability'])
            text_entropy = self._safe_extract_value(text_math['information_entropy'])
            text_task_relevance = self._safe_extract_value(quality['text_quality']['task_relevance'])

            # 构建质量向量
            quality_vector = torch.tensor([
                img_norm_stability,
                img_entropy,
                img_task_relevance,
                text_norm_stability,
                text_entropy,
                text_task_relevance
            ]).to(next(self.parameters()).device)

            quality_vectors.append(quality_vector)

        return torch.stack(quality_vectors)  # [batch_size, 6]

    def _safe_extract_value(self, tensor_or_value):
        """安全提取标量值"""
        if torch.is_tensor(tensor_or_value):
            if tensor_or_value.dim() == 0:
                return tensor_or_value.item()
            elif tensor_or_value.numel() > 0:
                return tensor_or_value.flatten()[0].item()
            else:
                return 0.5  # 默认值
        else:
            return float(tensor_or_value)

    def generate_quality_prompts_for_layer_0(self, quality_vectors):
        """
        为第0层生成质量提示（直接注入基础提示）

        Args:
            quality_vectors: [batch_size, quality_dim]

        Returns:
            quality_prompts_image: [batch_size, prompt_length_half, 768]
            quality_prompts_text: [batch_size, prompt_length_half, 512]
        """
        batch_size = quality_vectors.size(0)

        # 基础提示重复到batch
        base_prompts_image = self.base_quality_prompt_image.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 12, 768]
        base_prompts_text = self.base_quality_prompt_text.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 12, 512]

        # 质量信息注入
        quality_injection_image = self.quality_injector_image(quality_vectors)  # [batch, 768]
        quality_injection_text = self.quality_injector_text(quality_vectors)  # [batch, 512]

        # 计算自适应混合权重
        mixing_weights = self.mixing_weight_calculator(quality_vectors)  # [batch, 1]

        # 应用质量注入到每个提示位置
        quality_injection_image_expanded = quality_injection_image.unsqueeze(1).expand(-1, self.prompt_length_half,
                                                                                       -1)  # [batch, 12, 768]
        quality_injection_text_expanded = quality_injection_text.unsqueeze(1).expand(-1, self.prompt_length_half,
                                                                                     -1)  # [batch, 12, 512]

        # 自适应混合
        mixing_weights_image = mixing_weights.unsqueeze(-1).expand(-1, self.prompt_length_half, 768)  # [batch, 12, 768]
        mixing_weights_text = mixing_weights.unsqueeze(-1).expand(-1, self.prompt_length_half, 512)  # [batch, 12, 512]

        quality_prompts_image = base_prompts_image + mixing_weights_image * quality_injection_image_expanded
        quality_prompts_text = base_prompts_text + mixing_weights_text * quality_injection_text_expanded

        return quality_prompts_image, quality_prompts_text

    def generate_quality_enhanced_compound_prompts(self, original_prompts_image, original_prompts_text,
                                                   quality_vectors):
        """
        为深度提示生成质量增强的复合提示

        Args:
            original_prompts_image: List[[batch, length, 768]] 原始图像提示
            original_prompts_text: List[[batch, length, 512]] 原始文本提示
            quality_vectors: [batch_size, quality_dim]

        Returns:
            quality_enhanced_prompts_image: List[[batch, length, 768]]
            quality_enhanced_prompts_text: List[[batch, length, 512]]
        """
        quality_enhanced_prompts_image = []
        quality_enhanced_prompts_text = []

        for depth in range(self.prompt_depth):
            if depth < len(original_prompts_image) and depth < len(original_prompts_text):
                # 获取当前层的原始提示
                current_img_prompts = original_prompts_image[depth]  # [batch, length, 768]
                current_text_prompts = original_prompts_text[depth]  # [batch, length, 512]

                # 拼接图像和文本提示
                combined_prompts = torch.cat([current_img_prompts, current_text_prompts],
                                             dim=-1)  # [batch, length, 1280]

                # 为每个样本添加质量信息
                batch_size, prompt_length, _ = combined_prompts.shape
                quality_expanded = quality_vectors.unsqueeze(1).expand(-1, prompt_length, -1)  # [batch, length, 6]

                # 拼接质量信息
                combined_with_quality = torch.cat([combined_prompts, quality_expanded], dim=-1)  # [batch, length, 1286]

                # 通过质量感知投影网络
                enhanced_img_prompts = self.quality_projections_image[depth](
                    self.layernorm_quality_image[depth](combined_with_quality)
                )  # [batch, length, 768]

                enhanced_text_prompts = self.quality_projections_text[depth](
                    self.layernorm_quality_text[depth](combined_with_quality)
                )  # [batch, length, 512]

                quality_enhanced_prompts_image.append(enhanced_img_prompts)
                quality_enhanced_prompts_text.append(enhanced_text_prompts)
            else:
                # 如果超出原始提示范围，保持原样
                if depth < len(original_prompts_image):
                    quality_enhanced_prompts_image.append(original_prompts_image[depth])
                if depth < len(original_prompts_text):
                    quality_enhanced_prompts_text.append(original_prompts_text[depth])

        return quality_enhanced_prompts_image, quality_enhanced_prompts_text

    def forward(self, quality_scores):
        """
        主要接口：基于质量分数生成质量提示

        Args:
            quality_scores: List[Dict] 来自SimplifiedQualityEstimator的质量分数

        Returns:
            layer_0_prompts: Tuple[质量提示图像, 质量提示文本] 用于第0层
            quality_vectors: [batch_size, quality_dim] 质量向量，用于深度提示增强
        """
        if quality_scores is None:
            return None, None

        # 提取质量向量
        quality_vectors = self.extract_quality_vector(quality_scores)

        # 生成第0层的质量提示
        layer_0_prompts = self.generate_quality_prompts_for_layer_0(quality_vectors)

        return layer_0_prompts, quality_vectors


class EnhancedMultiModalPromptLearner(nn.Module):
    """增强的多模态提示学习器 - 集成质量感知提示"""

    def __init__(self, prompt_length, prompt_depth, clip_model):
        super().__init__()
        dtype = clip_model.dtype
        prompt_length_half = prompt_length // 3  # 12 - 保持原有设计

        self.prompt_depth = prompt_depth
        self.prompt_length_half = prompt_length_half
        self.dtype = dtype

        # ===================== 原有的提示参数 =====================
        # 静态提示
        self.visual_prompt_complete = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.visual_prompt_missing = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.text_prompt_complete = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.text_prompt_missing = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))

        # 通用提示
        self.common_prompt_complete = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_image = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_text = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))

        # 原有的投影网络
        embed_dim_text = 512
        embed_dim_image = 768
        embed_dim = embed_dim_text + embed_dim_image
        r = 16

        single_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // r),
            nn.GELU(),
            nn.Linear(embed_dim // r, embed_dim_text),
        )
        self.compound_prompt_projections_text = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_text = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])

        single_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // r),
            nn.GELU(),
            nn.Linear(embed_dim // r, embed_dim_image),
        )
        self.compound_prompt_projections_image = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_image = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])

        # 通用提示的投影网络
        self.common_prompt_projection_image = nn.Sequential(
            nn.Linear(embed_dim_text, embed_dim_text // r),
            nn.GELU(),
            nn.Linear(embed_dim_text // r, embed_dim_image),
        )
        self.common_prompt_projection_text = nn.Sequential(
            nn.Linear(embed_dim_text, embed_dim_text // r),
            nn.GELU(),
            nn.Linear(embed_dim_text // r, embed_dim_text),
        )

        # ===================== 新增：质量感知提示生成器 =====================
        self.quality_prompt_generator = QualityInformedPromptGenerator(
            prompt_length_half, prompt_depth, dtype
        )

        # 质量提示开关（可以通过配置控制）
        self.enable_quality_prompts = True

    def forward(self, missing_type, quality_scores=None):
        """
        增强的前向传播，支持质量感知提示

        Args:
            missing_type: List[int] 缺失类型
            quality_scores: List[Dict] 质量分数（可选）

        Returns:
            all_prompts_image: List[[batch, length, 768]] 图像提示
            all_prompts_text: List[[batch, length, 512]] 文本提示
        """
        batch_size = len(missing_type)

        # ===================== 第一步：生成基础提示（原有逻辑） =====================
        all_prompts_image = [[] for _ in range(self.prompt_depth)]
        all_prompts_text = [[] for _ in range(self.prompt_depth)]

        for i in range(batch_size):
            # 根据缺失类型选择初始提示
            if missing_type[i] == 0:  # 模态完整
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_complete
            elif missing_type[i] == 1:  # 缺失文本
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_missing
                common_prompt = self.common_prompt_image
            elif missing_type[i] == 2:  # 缺失图像
                initial_prompt_image = self.visual_prompt_missing
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_text

            # 生成第0层的复合提示
            all_prompts_image[0].append(self.compound_prompt_projections_image[0](
                self.layernorm_image[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            all_prompts_text[0].append(self.compound_prompt_projections_text[0](
                self.layernorm_text[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))

            # 生成后续层的提示
            for index in range(1, self.prompt_depth):
                all_prompts_image[index].append(
                    self.compound_prompt_projections_image[index](self.layernorm_image[index](
                        torch.cat([all_prompts_image[index - 1][-1], all_prompts_text[index - 1][-1]], -1))))
                all_prompts_text[index].append(
                    self.compound_prompt_projections_text[index](self.layernorm_text[index](
                        torch.cat([all_prompts_image[index - 1][-1], all_prompts_text[index - 1][-1]], -1))))

            # 添加通用提示到第0层
            all_prompts_image[0][i] = torch.cat([
                all_prompts_image[0][i],
                self.common_prompt_projection_image(common_prompt)
            ], 0)
            all_prompts_text[0][i] = torch.cat([
                all_prompts_text[0][i],
                self.common_prompt_projection_text(common_prompt)
            ], 0)

        # 转换为张量
        base_prompts_image = [torch.stack(prompts) for prompts in all_prompts_image]
        base_prompts_text = [torch.stack(prompts) for prompts in all_prompts_text]

        # ===================== 第二步：质量感知提示增强 =====================
        if self.enable_quality_prompts and quality_scores is not None:
            return self._apply_quality_enhancement(
                base_prompts_image, base_prompts_text, quality_scores, missing_type
            )
        else:
            # 如果没有质量信息，返回原有提示
            return base_prompts_image, base_prompts_text

    def _apply_quality_enhancement(self, base_prompts_image, base_prompts_text, quality_scores, missing_type):
        """
        应用质量感知提示增强

        Args:
            base_prompts_image: List[[batch, length, 768]] 基础图像提示
            base_prompts_text: List[[batch, length, 512]] 基础文本提示
            quality_scores: List[Dict] 质量分数
            missing_type: List[int] 缺失类型

        Returns:
            enhanced_prompts_image: List[[batch, enhanced_length, 768]]
            enhanced_prompts_text: List[[batch, enhanced_length, 512]]
        """
        # 生成质量提示
        layer_0_quality_prompts, quality_vectors = self.quality_prompt_generator(quality_scores)

        if layer_0_quality_prompts is None:
            return base_prompts_image, base_prompts_text

        quality_prompts_image, quality_prompts_text = layer_0_quality_prompts

        # 增强后的提示列表
        enhanced_prompts_image = []
        enhanced_prompts_text = []

        # ===================== 第0层：添加质量提示 =====================
        # 原有第0层提示形状：[batch, 24, dim] (12 compound + 12 common)
        # 新增质量提示形状：[batch, 12, dim]
        # 最终第0层形状：[batch, 36, dim] (12 quality + 12 compound + 12 common)

        enhanced_layer_0_image = torch.cat([
            quality_prompts_image,  # [batch, 12, 768] 质量提示
            base_prompts_image[0]  # [batch, 24, 768] 原有提示
        ], dim=1)  # [batch, 36, 768]

        enhanced_layer_0_text = torch.cat([
            quality_prompts_text,  # [batch, 12, 512] 质量提示
            base_prompts_text[0]  # [batch, 24, 512] 原有提示
        ], dim=1)  # [batch, 36, 512]

        enhanced_prompts_image.append(enhanced_layer_0_image)
        enhanced_prompts_text.append(enhanced_layer_0_text)

        # ===================== 后续层：质量增强的复合提示 =====================
        if self.prompt_depth > 1:
            # 使用质量信息增强后续层的提示
            quality_enhanced_compound = self.quality_prompt_generator.generate_quality_enhanced_compound_prompts(
                base_prompts_image[1:], base_prompts_text[1:], quality_vectors
            )

            enhanced_compound_image, enhanced_compound_text = quality_enhanced_compound
            enhanced_prompts_image.extend(enhanced_compound_image)
            enhanced_prompts_text.extend(enhanced_compound_text)

        return enhanced_prompts_image, enhanced_prompts_text

    def get_prompt_info(self):
        """
        【新增】获取提示信息，用于调试和可视化
        """
        return {
            'original_prompt_length_per_layer': {
                'layer_0': self.prompt_length_half * 2,  # 24 (compound + common)
                'other_layers': self.prompt_length_half  # 12 (compound only)
            },
            'quality_enhanced_prompt_length_per_layer': {
                'layer_0': self.prompt_length_half * 3,  # 36 (quality + compound + common)
                'other_layers': self.prompt_length_half  # 12 (quality-enhanced compound)
            },
            'prompt_depth': self.prompt_depth,
            'quality_prompts_enabled': self.enable_quality_prompts
        }

    def set_quality_prompts_enabled(self, enabled: bool):
        """
        【新增】动态控制质量提示的启用状态
        """
        self.enable_quality_prompts = enabled

class MultiModalPromptLearner(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model):
        super().__init__()
        dtype = clip_model.dtype
        prompt_length_half = prompt_length // 3  # use half length for generating static prompts, and the other for generating dynamic prompts
        # Default is 1, which is compound shallow prompting
        self.prompt_depth = prompt_depth  # max=12, but will create 11 such shared prompts
        self.visual_prompt_complete = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.visual_prompt_missing = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.text_prompt_complete = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.text_prompt_missing = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_complete = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_image = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_text = nn.Parameter(
            nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        # Also make corresponding projection layers, for each prompt
        embed_dim_text = 512
        embed_dim_image = 768
        embed_dim = embed_dim_text + embed_dim_image
        r = 16
        single_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // r),
            nn.GELU(),
            nn.Linear(embed_dim // r, embed_dim_text),
        )
        self.compound_prompt_projections_text = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_text = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])

        single_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // r),
            nn.GELU(),
            nn.Linear(embed_dim // r, embed_dim_image),
        )
        self.compound_prompt_projections_image = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_image = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        self.common_prompt_projection_image = nn.Sequential(
            nn.Linear(embed_dim_text, embed_dim_text // r),
            nn.GELU(),
            nn.Linear(embed_dim_text // r, embed_dim_image),
        )
        self.common_prompt_projection_text = nn.Sequential(
            nn.Linear(embed_dim_text, embed_dim_text // r),
            nn.GELU(),
            nn.Linear(embed_dim_text // r, embed_dim_text),
        )

    def forward(self, missing_type):

        # Before returning, need to transform
        # prompts to 768 for the visual side
        all_prompts_image = [[] for _ in range(self.prompt_depth)]  # Prompts of prompt_depth layers
        all_prompts_text = [[] for _ in range(self.prompt_depth)]  # Prompts of prompt_depth layers
        for i in range(len(missing_type)):
            # set initial prompts for each modality
            if missing_type[i] == 0:  # modality complete
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_complete
            elif missing_type[i] == 1:  # missing text
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_missing
                common_prompt = self.common_prompt_image
            elif missing_type[i] == 2:  # missing image
                initial_prompt_image = self.visual_prompt_missing
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_text
            # generate the prompts of the first layer
            all_prompts_image[0].append(self.compound_prompt_projections_image[0](
                self.layernorm_image[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            all_prompts_text[0].append(self.compound_prompt_projections_text[0](
                self.layernorm_text[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            # generate the prompts of the rest layers
            for index in range(1, self.prompt_depth):
                all_prompts_image[index].append(
                    self.compound_prompt_projections_image[index](self.layernorm_image[index](
                        torch.cat([all_prompts_image[index - 1][-1], all_prompts_text[index - 1][-1]], -1))))
                all_prompts_text[index].append(
                    self.compound_prompt_projections_text[index](self.layernorm_text[index](
                        torch.cat([all_prompts_image[index - 1][-1], all_prompts_text[index - 1][-1]], -1))))
            all_prompts_image[0][i] = torch.cat([
                all_prompts_image[0][i],
                self.common_prompt_projection_image(common_prompt)]
                , 0)
            all_prompts_text[0][i] = torch.cat([
                all_prompts_text[0][i],
                self.common_prompt_projection_text(common_prompt)]
                , 0)
        # generate the prompts in each layer as a tensor [B, L, C]

        all_prompts_image = [torch.stack(prompts) for prompts in all_prompts_image]
        all_prompts_text = [torch.stack(prompts) for prompts in all_prompts_text]
        # print(all_prompts_image)
        return all_prompts_image, all_prompts_text


class CustomCLIP(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model):
        super().__init__()
        self.prompt_learner = EnhancedMultiModalPromptLearner(prompt_length, prompt_depth, clip_model)
        # self.prompt_learner = MultiModalPromptLearner(prompt_length, prompt_depth, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # 【修改】使用改进的模态生成器
        from .improved_modal_generator import ImprovedModalGenerator
        self.modal_generator = ImprovedModalGenerator(
            hidden_size=512,
            num_layers=3,
            num_heads=8,
            dropout=0.1
        )

        # 【修改】质量评估器
        # self.quality_estimator = QualityEstimator(hidden_size=512)
        from .simplified_quality_estimator import SimplifiedQualityEstimator
        self.quality_estimator = SimplifiedQualityEstimator(hidden_size=512)

        # 【修改】质量引导融合器
        # self.quality_guided_fusion = QualityGuidedFusion(hidden_size=512)
        from .improved_quality_guide_fusion import UpdatedQualityGuidedFusion
        self.quality_guided_fusion = UpdatedQualityGuidedFusion(
            hidden_size=512,
            fusion_strategy='adaptive_attention'
        )

        self.cached_features = {}
        self.cached_generation_info = None
        self.cached_quality_scores = None

        # 【新增】质量提示控制
        self.quality_prompts_enabled = True
        self.quality_prompt_warmup_epochs = 3  # 前3个epoch不使用质量提示


    def set_quality_prompts_enabled(self, enabled: bool):
        """动态控制质量提示的启用"""
        self.quality_prompts_enabled = enabled
        self.prompt_learner.set_quality_prompts_enabled(enabled)

    def forward(self, image, text, missing_type, current_epoch=0):

        tokenized_texts = torch.stack([clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0).to(
            image.get_device()).squeeze(1)

        # 1. 【修改】两阶段提示生成：先基础提示，再质量增强

        # 第一阶段：生成基础提示（不使用质量信息）
        base_prompts_image, base_prompts_text = self.prompt_learner(missing_type, quality_scores=None)

        # 基础编码
        image_features = self.image_encoder(image.type(self.dtype), base_prompts_image, missing_type)
        text_features = self.text_encoder(tokenized_texts, base_prompts_text, missing_type)

        if self.training:
            image_features = image_features.requires_grad_(True)
            text_features = text_features.requires_grad_(True)

        # 2. 模态生成和特征替代
        generation_results = self.modal_generator(image_features, text_features, missing_type)

        enhanced_image_features = generation_results['enhanced_image_features']
        enhanced_text_features = generation_results['enhanced_text_features']
        generation_info = generation_results['generation_info']

        self.cached_generation_info = generation_info

        # 3. 质量评估
        quality_scores = self.quality_estimator(
            image_features, text_features,
            enhanced_image_features, enhanced_text_features,
            missing_type
        )
        self.cached_quality_scores = quality_scores

        # 4. 【新增】第二阶段：质量感知提示重新编码（可选）
        if (self.quality_prompts_enabled and
                current_epoch >= self.quality_prompt_warmup_epochs and
                self.training):
            # 使用质量信息重新生成提示
            quality_enhanced_prompts_image, quality_enhanced_prompts_text = self.prompt_learner(
                missing_type, quality_scores
            )

            # 重新编码（使用质量增强的提示）
            image_features_enhanced = self.image_encoder(
                image.type(self.dtype), quality_enhanced_prompts_image, missing_type
            )
            text_features_enhanced = self.text_encoder(
                tokenized_texts, quality_enhanced_prompts_text, missing_type
            )

            # 更新特征（用质量增强的特征）
            image_features = image_features_enhanced
            text_features = text_features_enhanced

            # 重新生成增强特征
            generation_results_enhanced = self.modal_generator(image_features, text_features, missing_type)
            enhanced_image_features = generation_results_enhanced['enhanced_image_features']
            enhanced_text_features = generation_results_enhanced['enhanced_text_features']

        # 5. 质量引导融合
        fused_features = self.quality_guided_fusion(
            image_features, text_features,
            enhanced_image_features, enhanced_text_features,
            quality_scores, missing_type
        )

        # 缓存特征用于损失计算
        self.cached_features = {
            'original_image_features': image_features,
            'original_text_features': text_features,
            'enhanced_image_features': enhanced_image_features,
            'enhanced_text_features': enhanced_text_features,
            'fused_features': fused_features
        }


        return fused_features


    def get_cached_data_for_loss_computation(self):
        """
        【新增】为外部损失计算提供缓存数据
        """
        return {
            'features': self.cached_features,
            'generation_info': self.cached_generation_info,
            'quality_scores': self.cached_quality_scores
        }


class CLIPransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        clip_model = load_clip_to_cpu(config['vit'], config['prompt_length'], config['prompt_depth'])

        print("Building custom CLIP")
        hidden_size = 512 * 2
        print(config['prompt_length'])
        self.model = CustomCLIP(config['prompt_length'], config['prompt_depth'], clip_model)

        # 【新增】循环一致性损失权重
        self.cycle_loss_weight = config.get('cycle_loss_weight', 0.02)
        self.quality_loss_weight = config.get('quality_loss_weight', 0.01)  # 【新增】
        self.predictor_loss_weight = config.get('predictor_loss_weight', 0.005)  # 预测器损失权重
        # 【新增】训练策略控制
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.quality_aware_epochs = config.get('quality_aware_epochs', 10)

        # 【新增】生成器相关损失权重
        self.contrastive_loss_weight = config.get('contrastive_loss_weight', 0.1)
        self.generation_consistency_weight = config.get('generation_consistency_weight', 0.05)
        self.cycle_loss_weight = config.get('cycle_loss_weight', 0.02)
        self.generation_quality_weight = config.get('generation_quality_weight', 0.01)


        # 【新增】质量相关损失权重
        self.contrastive_loss_weight = config.get('contrastive_loss_weight', 0.1)
        self.generation_consistency_weight = config.get('generation_consistency_weight', 0.05)
        self.cycle_loss_weight = config.get('cycle_loss_weight', 0.02)
        self.generation_quality_weight = config.get('generation_quality_weight', 0.01)
        self.quality_prediction_weight = config.get('quality_prediction_weight', 0.02)  # 【新增】

        # 自适应损失权重调度器
        self.loss_scheduler = self.create_loss_scheduler()

        # ===================== Downstream ===================== #
        if (
                self.hparams.config["load_path"] != ""
                and not self.hparams.config["test_only"]
                and not self.hparams.config["finetune_first"]
        ):
            #
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)

        if self.hparams.config["loss_names"]["hatememes"] > 0:
            cls_num = self.hparams.config["hatememes_class_num"]
            self.hatememes_classifier = nn.Linear(hidden_size, cls_num)
            self.hatememes_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["food101"] > 0:
            cls_num = self.hparams.config["food101_class_num"]
            self.food101_classifier = nn.Linear(hidden_size, cls_num)
            self.food101_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Linear(hidden_size, cls_num)
            self.mmimdb_classifier.apply(objectives.init_weights)

        if self.hparams.config["load_path"] != "" and self.hparams.config["finetune_first"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)
            print("use pre-finetune model")

        if not self.hparams.config["test_only"]:
            for name, param in self.model.named_parameters():
                if "prompt_learner" not in name and "prompt" not in name and 'ln_final' not in name and 'ln_post' not in name and \
                        name.split('.')[-1] != 'proj':
                    param.requires_grad_(False)

        clip_utils.set_metrics(self)
        self.current_tasks = list()

        # ===================== load downstream (test_only) ======================

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=True)
        self.records = {}



    def infer(
            self,
            batch,
    ):
        text = batch["text"]
        img = batch["image"][0]  # extract the first view (total 1)
        if self.hparams.config["test_only"]:
            self.model.eval()
            if self.hparams.config["loss_names"]["hatememes"] > 0:
                self.hatememes_classifier.eval()

            if self.hparams.config["loss_names"]["food101"] > 0:
                self.food101_classifier.eval()

            if self.hparams.config["loss_names"]["mmimdb"] > 0:
                self.mmimdb_classifier.eval()

        # 获取增强后的特征
        both_feats = self.model(img, text, batch["missing_type"])

        ret = {
            "cls_feats": both_feats,
        }

        return ret

    def forward(self, batch):

        # 【新增】缓存missing_type用于损失计算
        self._current_missing_type = batch["missing_type"]

        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Masked Patch Prediction
        if "mpp" in self.current_tasks:
            ret.update(objectives.compute_mpp(self, batch))

        # Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_wpa(self, batch))

        # Binary classification for Hateful Memes
        if "hatememes" in self.current_tasks:
            ret.update(objectives.compute_hatememes(self, batch))

        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:

            ret.update(objectives.compute_mmimdb(self, batch))

            # ret.update(objectives.compute_enhanced_mmimdb(self, batch))
            # ret.update(objectives.compute_enhanced_mmimdb_v2(self, batch))


        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        # 【新增】提取任务性能指标
        self._extract_task_performance_from_batch(batch)

        output = self(batch)


        # 【新增】提取任务性能指标
        self._extract_task_performance_from_batch(batch)

        # 主任务损失
        main_loss = sum([v for k, v in output.items() if "loss" in k and "generation" not in k])

        # 【新增】生成器相关损失
        generation_losses = self._compute_generation_losses()
        # 【新增】质量预测损失
        quality_prediction_loss = self._compute_quality_prediction_loss()
        total_loss = main_loss

        # 添加生成器损失
        if generation_losses['contrastive_loss'] > 0:
            total_loss += self.contrastive_loss_weight * generation_losses['contrastive_loss']
            self.log("train/contrastive_loss", generation_losses['contrastive_loss'])

        if generation_losses['generation_consistency_loss'] > 0:
            total_loss += self.generation_consistency_weight * generation_losses['generation_consistency_loss']
            self.log("train/generation_consistency_loss", generation_losses['generation_consistency_loss'])

        if generation_losses['cycle_consistency_loss'] > 0:
            total_loss += self.cycle_loss_weight * generation_losses['cycle_consistency_loss']
            self.log("train/cycle_consistency_loss", generation_losses['cycle_consistency_loss'])

        if generation_losses['generation_quality_loss'] > 0:
            total_loss += self.generation_quality_weight * generation_losses['generation_quality_loss']
            self.log("train/generation_quality_loss", generation_losses['generation_quality_loss'])

        # 【新增】质量预测损失
        if quality_prediction_loss > 0:
            total_loss += self.quality_prediction_weight * quality_prediction_loss
            self.log("train/quality_prediction_loss", quality_prediction_loss)

        # 记录损失
        self.log("train/main_loss", main_loss)
        self.log("train/total_loss", total_loss)
        self.log("train/epoch", float(self.current_epoch))

        return total_loss


    def training_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

    #         print('missing_img:', self.missing_img_prompt[0,0:3,0:8])
    #         print('missing_text:', self.missing_text_prompt[0,0:3,0:8])
    #         print('complete:', self.complete_prompt[0,0:3,0:8])

    def test_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        clip_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return clip_utils.set_schedule(self)


    def create_loss_scheduler(self):
        """创建自适应损失调度器"""
        return {
            'warmup': {'cycle': 0.05, 'quality': 0.001},  # 前5轮: 重点训练生成器
            'quality_aware': {'cycle': 0.02, 'quality': 0.01},  # 中期: 平衡训练
            'full_optimization': {'cycle': 0.01, 'quality': 0.02}  # 后期: 重点质量
        }

    def get_current_loss_weights(self):
        """根据当前epoch获取损失权重"""
        current_epoch = self.current_epoch

        if current_epoch < self.warmup_epochs:
            return self.loss_scheduler['warmup']
        elif current_epoch < self.warmup_epochs + self.quality_aware_epochs:
            return self.loss_scheduler['quality_aware']
        else:
            return self.loss_scheduler['full_optimization']

    def configure_model_for_epoch(self):
        """根据训练阶段配置模型"""
        current_epoch = self.current_epoch

        if current_epoch < self.warmup_epochs:
            # 阶段1: 只训练基础组件
            self.model.use_iterative_optimization = False
            self.model.use_quality_aware_prompts = False

            # 冻结质量相关组件
            for param in self.model.quality_estimator.parameters():
                param.requires_grad = False
            for param in self.model.quality_prompt_learner.parameters():
                param.requires_grad = False

        elif current_epoch < self.warmup_epochs + self.quality_aware_epochs:
            # 阶段2: 启用质量感知但不迭代
            self.model.use_iterative_optimization = False
            self.model.use_quality_aware_prompts = True

            # 解冻质量组件
            for param in self.model.quality_estimator.parameters():
                param.requires_grad = True
            for param in self.model.quality_prompt_learner.parameters():
                param.requires_grad = True

        else:
            # 阶段3: 完全启用所有功能
            self.model.use_iterative_optimization = True
            self.model.use_quality_aware_prompts = True


    def _extract_task_performance_from_batch(self, batch):
        """
        【新增】从当前batch提取任务性能指标，用于质量监督
        """
        if "label" in batch:
            labels = batch["label"]
            if isinstance(labels, list):
                labels = torch.tensor(labels).float().to(self.device)
            elif not torch.is_tensor(labels):
                labels = torch.tensor(labels).float().to(self.device)
            else:
                labels = labels.float().to(self.device)

            # 计算标签复杂度作为任务性能的代理
            if labels.dim() > 1:  # 多标签情况 (如 MMIMDb)
                # 使用标签数量的倒数作为性能指标（标签越多越复杂，性能期望越低）
                label_count = labels.sum(dim=-1)
                max_labels = labels.size(-1)
                performance = 1.0 - (label_count / max_labels) * 0.5  # 映射到[0.5, 1.0]
            else:  # 单标签情况 (如 Food101, HateMemes)
                # 使用固定的中等性能
                performance = torch.ones(labels.size(0)) * 0.7

            self._current_task_performance = performance.to(self.device)
        else:
            # 如果没有标签，使用默认性能
            batch_size = len(batch["missing_type"])
            self._current_task_performance = torch.ones(batch_size).to(self.device) * 0.6


    def _compute_generation_losses(self):
        """
        【新增】计算生成器相关的损失
        """
        # 获取缓存数据
        cached_data = self.model.get_cached_data_for_loss_computation()

        if not cached_data['features']:
            return {
                'contrastive_loss': torch.tensor(0.0, requires_grad=True),
                'generation_consistency_loss': torch.tensor(0.0, requires_grad=True),
                'cycle_consistency_loss': torch.tensor(0.0, requires_grad=True),
                'generation_quality_loss': torch.tensor(0.0, requires_grad=True)
            }

        cached_features = cached_data['features']
        missing_type = self._current_missing_type if self._current_missing_type is not None else [0] * cached_features[
            'original_image_features'].size(0)

        # 使用模态生成器计算所有损失
        generation_losses = self.model.modal_generator.compute_all_generation_losses(
            cached_features['original_image_features'],
            cached_features['original_text_features'],
            missing_type
        )

        return generation_losses

    def _compute_quality_prediction_loss(self):
        """
        【新增】计算质量预测的训练损失
        """
        # 获取缓存的质量分数
        cached_data = self.model.get_cached_data_for_loss_computation()
        quality_scores = cached_data['quality_scores']

        if quality_scores is None:
            return torch.tensor(0.0, requires_grad=True)


        # 使用质量评估器计算损失
        quality_loss = self.model.quality_estimator.compute_quality_loss(
            quality_scores, self._current_task_performance
        )

        return quality_loss

    def log_quality_statistics(self):
        """
        【新增】记录质量统计信息
        """
        if hasattr(self.model, 'cached_quality_scores') and self.model.cached_quality_scores:
            quality_scores = self.model.cached_quality_scores

            # 统计各种质量指标
            img_relevances = []
            text_relevances = []
            overall_confidences = []

            for quality in quality_scores:
                img_rel = quality['image_quality']['task_relevance']
                text_rel = quality['text_quality']['task_relevance']
                overall_conf = quality['overall_confidence']

                # 安全提取数值
                if torch.is_tensor(img_rel):
                    img_rel = img_rel.item() if img_rel.dim() == 0 else img_rel[0].item()
                if torch.is_tensor(text_rel):
                    text_rel = text_rel.item() if text_rel.dim() == 0 else text_rel[0].item()
                if torch.is_tensor(overall_conf):
                    overall_conf = overall_conf.item() if overall_conf.dim() == 0 else overall_conf[0].item()

                img_relevances.append(float(img_rel))
                text_relevances.append(float(text_rel))
                overall_confidences.append(float(overall_conf))

            # 记录平均值
            self.log("quality/avg_img_relevance", sum(img_relevances) / len(img_relevances))
            self.log("quality/avg_text_relevance", sum(text_relevances) / len(text_relevances))
            self.log("quality/avg_overall_confidence", sum(overall_confidences) / len(overall_confidences))


