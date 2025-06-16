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
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # === 核心组件：原始空间模态生成器 ===
        from .improved_modal_generator import ImprovedModalGenerator
        self.modal_generator = ImprovedModalGenerator(
            image_size=224,
            text_length=77,
            hidden_dim=512
        )

        # === 核心组件：基于embedding的质量评估器 ===
        from .simplified_quality_estimator import SimplifiedQualityEstimator
        self.quality_estimator = SimplifiedQualityEstimator(
            image_embed_dim=768,  # CLIP图像embedding维度
            text_embed_dim=512  # CLIP文本embedding维度
        )

        # === 核心组件：质量引导融合器（可选） ===
        from .improved_quality_guide_fusion import UpdatedQualityGuidedFusion
        self.quality_guided_fusion = UpdatedQualityGuidedFusion(
            hidden_size=512,
            fusion_strategy='adaptive_attention'
        )

        self.alignment_proj = nn.Linear(512, 768)

        # === 控制参数 ===
        self.use_quality_fusion = True  # 是否使用质量引导融合
        self.quality_prompts_enabled = True
        self.training_epoch = 0

        # === 缓存变量（用于损失计算） ===
        self.cached_quality_results = None
        self.cached_generation_info = None
        self.cached_embeddings = None
        self.cached_training_losses = None

        self.cache_step_count = 0  # 记录缓存步数
        self.max_cache_steps = 100  # 最大缓存步数，超过就清理

    def clear_cache(self):
        """【新增】清理所有缓存，释放内存"""
        self.cached_quality_results = None
        self.cached_generation_info = None
        self.cached_embeddings = None
        self.cache_step_count = 0

        # 清理modal_generator的内部缓存（如果有）
        if hasattr(self.modal_generator, 'clear_cache'):
            self.modal_generator.clear_cache()

        # 清理quality_estimator的内部缓存（如果有）
        if hasattr(self.quality_estimator, 'clear_cache'):
            self.quality_estimator.clear_cache()

        # 强制垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def get_embedding_features(self, processed_images, processed_texts, require_grad=True):
        """
        获取embedding层特征用于质量评估
        不经过完整的transformer，只到embedding层

        Args:
            processed_images: [batch, 3, 224, 224]
            processed_texts: [batch, 77]

        Returns:
            img_embeddings: [batch, 197, 768] 图像patch embedding + cls token + pos embedding
            text_embeddings: [batch, 77, 512] 文本token embedding + pos embedding
        """
        if self.training:
            # === 图像embedding（保留梯度） ===
            x = self.image_encoder.conv1(processed_images.type(self.dtype))  # [batch, 768, 14, 14]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, 768, 196]
            x = x.permute(0, 2, 1)  # [batch, 196, 768]

            # 添加class token
            cls_token = self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
            x = torch.cat([cls_token, x], dim=1)  # [batch, 197, 768]

            # 添加位置编码
            img_embeddings = x + self.image_encoder.positional_embedding.to(x.dtype)

            # === 文本embedding（保留梯度） ===
            text_embeddings = self.text_encoder.token_embedding(processed_texts).type(self.dtype)  # [batch, 77, 512]
            text_embeddings = text_embeddings + self.text_encoder.positional_embedding.type(self.dtype)
        else:
            # 推理时关闭梯度
            with torch.no_grad():
                x = self.image_encoder.conv1(processed_images.type(self.dtype))
                x = x.reshape(x.shape[0], x.shape[1], -1)
                x = x.permute(0, 2, 1)

                cls_token = self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                x = torch.cat([cls_token, x], dim=1)
                img_embeddings = x + self.image_encoder.positional_embedding.to(x.dtype)

                text_embeddings = self.text_encoder.token_embedding(processed_texts).type(self.dtype)
                text_embeddings = text_embeddings + self.text_encoder.positional_embedding.type(self.dtype)


        return img_embeddings, text_embeddings

    def forward(self, image, text, missing_type, current_epoch=0):
        """
        主要前向传播逻辑

        核心流程：
        1. 预处理：在原始空间生成缺失模态
        2. 质量评估：基于embedding特征评估质量
        3. 提示生成：基于质量信息生成质量感知提示
        4. 单次编码：使用提示进行编码
        5. 融合：质量引导的特征融合
        """
        self.training_epoch = current_epoch
        self.cache_step_count += 1

        # 【修复】定期清理缓存防止累积
        if self.cache_step_count % self.max_cache_steps == 0:
            self.clear_cache()
        # print(type(image),type(text))
        # print(text)
        # print("start dtype",image.dtype,text.dtype)

        # === 第1步：文本预处理 ===
        tokenized_texts = torch.stack([
            clip.tokenize(tx, context_length=77, truncate=True)
            for tx in text[0]
        ], 0).to(image.device).squeeze(1)

        # print("token dtype",image.dtype,tokenized_texts.dtype)
        # === 第2步：原始空间模态生成 ===
        processed_images, processed_texts, generation_info = self.modal_generator.preprocess_missing_modalities(
            image, tokenized_texts, missing_type
        )
        # print("processed dtype",processed_images.dtype,processed_texts.dtype)

        # 缓存生成信息用于损失计算
        self.cached_generation_info = generation_info

        # === 第3步：获取embedding特征进行质量评估 ===
        img_embeddings, text_embeddings = self.get_embedding_features(processed_images, processed_texts)
        # print("embeding dtype",img_embeddings.dtype,text_embeddings.dtype)
        # 对于生成的模态，也获取其embedding
        enhanced_img_embeddings = img_embeddings.clone()
        enhanced_text_embeddings = text_embeddings.clone()

        # 对于生成的模态，我们需要重新计算embedding以获得正确的梯度
        for i, miss_type in enumerate(missing_type):
            if miss_type == 2:  # 缺失图像，重新计算生成图像的embedding
                gen_img_emb, _ = self.get_embedding_features(
                    processed_images[i:i + 1], processed_texts[i:i + 1]
                )
                enhanced_img_embeddings[i] = gen_img_emb[0]
            elif miss_type == 1:  # 缺失文本，重新计算生成文本的embedding
                _, gen_text_emb = self.get_embedding_features(
                    processed_images[i:i + 1], processed_texts[i:i + 1]
                )
                enhanced_text_embeddings[i] = gen_text_emb[0]

        # 缓存embedding用于损失计算
        self.cached_embeddings = {
            'original_img': img_embeddings,
            'original_text': text_embeddings,
            'enhanced_img': enhanced_img_embeddings,
            'enhanced_text': enhanced_text_embeddings
        }

        # === 第4步：质量评估 ===
        quality_scores = self.quality_estimator(
            img_embeddings, text_embeddings,
            enhanced_img_embeddings, enhanced_text_embeddings,
            missing_type, generation_info
        )

        # 缓存质量结果用于损失计算
        self.cached_quality_results = quality_scores

        # === 第5步：质量感知提示生成 ===
        if self.quality_prompts_enabled and current_epoch >= 0:  # 可以设置warmup
            quality_prompts_image, quality_prompts_text = self.prompt_learner(
                missing_type, quality_scores
            )
        else:
            # 使用基础提示
            quality_prompts_image, quality_prompts_text = self.prompt_learner(
                missing_type, quality_scores=None
            )

        # === 第6步：单次编码（带提示学习） ===
        # 使用处理后的完整输入和质量感知提示进行编码
        image_features = self.image_encoder(
            processed_images.type(self.dtype),
            quality_prompts_image,
            missing_type
        )
        text_features = self.text_encoder(
            processed_texts,
            quality_prompts_text,
            missing_type
        )

        # 确保梯度传播（训练时）
        if self.training:
            image_features = image_features.requires_grad_(True)
            text_features = text_features.requires_grad_(True)

        # === 第7步：质量引导融合 ===
        if self.use_quality_fusion:
            # 使用质量引导融合
            fused_features = self.quality_guided_fusion(
                image_features, text_features,
                image_features, text_features,  # 这里可以传入增强特征
                quality_scores, missing_type
            )
        else:
            # 简单拼接
            fused_features = torch.cat([image_features, text_features], dim=-1)

        return fused_features

    def compute_training_losses(self, image_input, text_input, missing_type, task_performance=None):
        """
        【修复】计算所有训练损失，确保梯度正确传播

        Args:
            image_input: 原始图像输入
            text_input: 原始文本token
            missing_type: 缺失类型
            task_performance: 任务性能（用于质量损失）

        Returns:
            Dict 包含所有训练损失
        """
        device = image_input.device

        total_losses = {
            'generation_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'quality_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'consistency_loss': torch.tensor(0.0, device=device, requires_grad=True)
        }

        # === 1. 生成器损失（使用原始输入） ===
        if self.training:
            generation_losses = self.modal_generator.compute_generation_losses(
                image_input, text_input, missing_type
            )

            gen_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for loss_name, loss_value in generation_losses.items():
                if isinstance(loss_value, torch.Tensor) and loss_value.requires_grad:
                    gen_loss = gen_loss + loss_value

            total_losses['generation_loss'] = gen_loss

        # === 2. 质量评估损失 ===
        if self.cached_quality_results is not None and self.training:
            quality_loss = self.quality_estimator.compute_quality_loss(
                self.cached_quality_results,
                task_performance=task_performance
            )
            total_losses['quality_loss'] = quality_loss


        # === 3. 一致性损失（基于embedding） ===
        if self.cached_embeddings is not None and self.training:
            consistency_loss = self._compute_embedding_consistency_loss(missing_type)
            total_losses['consistency_loss'] = consistency_loss

        # 缓存损失用于外部访问
        self.cached_training_losses = total_losses

        return total_losses

    def _compute_embedding_consistency_loss(self, missing_type):
        """
        【修复】计算embedding一致性损失
        """
        device = next(self.parameters()).device
        consistency_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if self.cached_embeddings is None:
            return consistency_loss

        orig_img = self.cached_embeddings['original_img']  # [batch, 197, 768]
        orig_text = self.cached_embeddings['original_text']  # [batch, 77, 512]

        complete_count = 0

        for i, miss_type in enumerate(missing_type):
            if miss_type == 0:  # 完整样本
                # 平均池化到向量
                # print(type(orig_img))
                # print(orig_img.shape,orig_img.dtype,orig_img.device)
                # print(orig_text.shape,orig_text.dtype,orig_text.device)

                img_feat = orig_img[i].mean(dim=0)  # [768]
                text_feat = orig_text[i].mean(dim=0)  # [512]
                # print(img_feat.shape, img_feat.dtype, img_feat.device)
                # print(text_feat.shape, text_feat.dtype ,text_feat.device)

                # 投影到相同维度进行对齐

                text_aligned = self.alignment_proj(text_feat)  # [768]

                # 余弦相似度损失
                img_norm = F.normalize(img_feat, p=2, dim=-1)
                text_norm = F.normalize(text_aligned, p=2, dim=-1)
                similarity = torch.dot(img_norm, text_norm)

                # 相似度应该高
                similarity_loss = 1.0 - similarity
                consistency_loss = consistency_loss + similarity_loss
                complete_count += 1

        if complete_count > 0:
            consistency_loss = consistency_loss / complete_count

        return consistency_loss

    def get_cached_losses(self):
        """获取缓存的损失用于外部访问"""
        return self.cached_training_losses

    def set_quality_prompts_enabled(self, enabled: bool):
        """动态控制质量提示的启用"""
        self.quality_prompts_enabled = enabled
        if hasattr(self.prompt_learner, 'set_quality_prompts_enabled'):
            self.prompt_learner.set_quality_prompts_enabled(enabled)

    def set_quality_fusion_enabled(self, enabled: bool):
        """动态控制质量融合的启用"""
        self.use_quality_fusion = enabled

    def get_quality_info(self):
        """获取质量信息用于调试"""
        return {
            'quality_results': self.cached_quality_results,
            'generation_info': self.cached_generation_info,
            'embeddings_cached': self.cached_embeddings is not None,
            'training_losses': self.cached_training_losses
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

        # === 损失权重配置 ===
        self.generation_loss_weight = config.get('generation_loss_weight', 0.1)
        self.quality_loss_weight = config.get('quality_loss_weight', 0.05)
        self.consistency_loss_weight = config.get('consistency_loss_weight', 0.02)

        # === 训练策略配置 ===
        self.warmup_epochs = config.get('warmup_epochs', 3)
        self.quality_warmup_epochs = config.get('quality_warmup_epochs', 1)

        # === 添加学习率衰减控制 ===
        self.lr_decay_factor = config.get('lr_decay_factor', 0.95)
        self.lr_decay_patience = config.get('lr_decay_patience', 2)
        # === 损失历史记录 ===
        self.loss_history = {
            'main_loss': [],
            'generation_loss': [],
            'quality_loss': [],
            'consistency_loss': []
        }

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
            # for name, param in self.model.named_parameters():
            #     if "prompt_learner" not in name and "prompt" not in name and 'ln_final' not in name and 'ln_post' not in name and \
            #             name.split('.')[-1] != 'proj':
            #         param.requires_grad_(False)

            print("============start----freeze===========")
            for name, param in self.model.named_parameters():

                # 只训练以下组件
                trainable_components = [
                    'prompt_learner', 'prompt', 'ln_final', 'ln_post',
                    'modal_generator', 'quality_estimator', 'quality_guided_fusion',
                    'alignment_proj'  # 添加对齐投影层
                ]

                is_trainable = any(comp in name for comp in trainable_components) or name.split('.')[-1] == 'proj'

                if not is_trainable:
                    param.requires_grad_(False)
                    print(f"Not Trainable parameter: {name}")
                else:
                    param.requires_grad_(True)
                    print(f"Trainable parameter: {name}")

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
        both_feats = self.model(img, text, batch["missing_type"], current_epoch=self.current_epoch)

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

        # === 分阶段训练策略 ===
        current_epoch = self.current_epoch

        if current_epoch < self.quality_warmup_epochs:
            # 阶段1：只训练基础组件
            self.model.set_quality_prompts_enabled(False)
            self.model.set_quality_fusion_enabled(False)
            effective_gen_weight = 0.0
            effective_quality_weight = 0.0
            effective_consistency_weight = 0.0
        elif current_epoch < self.warmup_epochs:
            # 阶段2：启用质量感知
            self.model.set_quality_prompts_enabled(True)
            self.model.set_quality_fusion_enabled(False)
            effective_gen_weight = self.generation_loss_weight * 0.5  # 渐进式增加
            effective_quality_weight = self.quality_loss_weight * 0.5
            effective_consistency_weight = 0.0
        else:
            # 阶段3：完全启用
            self.model.set_quality_prompts_enabled(True)
            self.model.set_quality_fusion_enabled(True)
            effective_gen_weight = self.generation_loss_weight
            effective_quality_weight = self.quality_loss_weight
            effective_consistency_weight = self.consistency_loss_weight

        output = self(batch)
        main_loss = sum([v for k, v in output.items() if "loss" in k])

        # === 【修复】提取任务性能用于质量损失 ===
        task_performance = self._extract_task_performance(batch, output)

        # === 【修复】获取原始输入并计算训练损失 ===
        image_input = batch["image"][0]
        tokenized_texts = torch.stack([
            clip.tokenize(tx, context_length=77, truncate=True)
            for tx in batch["text"][0]
        ], 0).to(image_input.device).squeeze(1)
        missing_type = batch["missing_type"]

        # 计算所有训练损失
        training_losses = self.model.compute_training_losses(
            image_input, tokenized_texts, missing_type, task_performance
        )

        # === 【修复】确保损失正确加到总损失中 ===
        total_loss = main_loss

        generation_loss = training_losses['generation_loss']
        quality_loss = training_losses['quality_loss']
        consistency_loss = training_losses['consistency_loss']

        # 添加生成器损失
        if effective_gen_weight > 0 and generation_loss.item() > 0:
            weighted_gen_loss = effective_gen_weight * generation_loss
            total_loss = total_loss + weighted_gen_loss
            self.log("train/generation_loss", generation_loss, prog_bar=True)
            self.log("train/weighted_generation_loss", weighted_gen_loss)

        # 添加质量损失
        if effective_quality_weight > 0 and quality_loss.item() > 0:
            weighted_quality_loss = effective_quality_weight * quality_loss
            total_loss = total_loss + weighted_quality_loss
            self.log("train/quality_loss", quality_loss, prog_bar=True)
            self.log("train/weighted_quality_loss", weighted_quality_loss)

        # 添加一致性损失
        if effective_consistency_weight > 0 and consistency_loss.item() > 0:
            weighted_consistency_loss = effective_consistency_weight * consistency_loss
            total_loss = total_loss + weighted_consistency_loss
            self.log("train/consistency_loss", consistency_loss, prog_bar=True)
            self.log("train/weighted_consistency_loss", weighted_consistency_loss)

        # === 【修复】记录损失差异以验证 ===
        loss_difference = total_loss - main_loss
        self.log("train/loss_difference", loss_difference, prog_bar=True)

        # === 记录各种损失 ===
        self.log("train/main_loss", main_loss, prog_bar=True)
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/epoch", float(self.current_epoch))

        # 记录权重
        self.log("train/gen_weight", effective_gen_weight)
        self.log("train/quality_weight", effective_quality_weight)
        self.log("train/consistency_weight", effective_consistency_weight)

        # === 【修复】记录质量统计信息 ===
        if current_epoch >= self.quality_warmup_epochs:
            self._log_quality_statistics()

        # === 记录损失历史 ===
        self.loss_history['main_loss'].append(main_loss.item())
        self.loss_history['generation_loss'].append(generation_loss.item())
        self.loss_history['quality_loss'].append(quality_loss.item())
        self.loss_history['consistency_loss'].append(consistency_loss.item())

        # === 【修复】梯度裁剪防止梯度爆炸 ===
        if self.training:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        # print(total_loss)

        return total_loss

    def _log_quality_statistics(self):
        """
        【修复】记录质量统计信息，添加更多调试信息
        """
        quality_info = self.model.get_quality_info()
        quality_results = quality_info.get('quality_results')

        if quality_results:
            # 统计各种质量指标
            stats = {
                'img_relevances': [],
                'text_relevances': [],
                'overall_confidences': [],
                'img_uncertainties': [],
                'text_uncertainties': [],
                'cross_modal_consistencies': [],
                'generation_confidences': []
            }

            for quality in quality_results:
                stats['img_relevances'].append(
                    self._safe_extract_value(quality['image_quality']['task_relevance'])
                )
                stats['text_relevances'].append(
                    self._safe_extract_value(quality['text_quality']['task_relevance'])
                )
                stats['overall_confidences'].append(
                    self._safe_extract_value(quality['overall_confidence'])
                )
                stats['img_uncertainties'].append(
                    self._safe_extract_value(quality['image_quality']['uncertainty'])
                )
                stats['text_uncertainties'].append(
                    self._safe_extract_value(quality['text_quality']['uncertainty'])
                )
                stats['cross_modal_consistencies'].append(
                    self._safe_extract_value(quality['cross_modal_consistency']['overall_consistency'])
                )
                stats['generation_confidences'].append(
                    self._safe_extract_value(quality['image_quality']['generation_confidence'])
                )

            # 记录平均值和标准差
            for key, values in stats.items():
                if values:
                    mean_val = sum(values) / len(values)
                    std_val = (sum([(v - mean_val) ** 2 for v in values]) / len(values)) ** 0.5

                    self.log(f"quality/avg_{key}", mean_val)
                    self.log(f"quality/std_{key}", std_val)

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


    def training_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)
        # === 打印训练阶段信息 ===
        current_epoch = self.current_epoch
        print(f"=== Epoch {self.current_epoch} ended, cleaning up memory ===")

        # 1. 清理模型内部缓存
        self.model.clear_cache()

        # 2. 清理PyTorch缓存
        torch.cuda.empty_cache()

        # 3. 强制垃圾回收
        import gc
        gc.collect()

        # 4. 记录内存使用情况
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"Memory after cleanup - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # 5. 如果内存使用过高，额外清理
        if allocated > 10.0:  # 如果分配内存超过10GB
            print("Memory usage high, performing additional cleanup...")

            # 清理所有模块的缓存
            for module in self.modules():
                if hasattr(module, 'clear_cache'):
                    module.clear_cache()

            # 再次清理
            torch.cuda.empty_cache()
            gc.collect()

            # 重新记录
            allocated_after = torch.cuda.memory_allocated() / 1024 ** 3
            print(f"Memory after additional cleanup: {allocated_after:.2f}GB")


    def on_train_epoch_start(self):
        """训练epoch开始时的配置"""
        current_epoch = self.current_epoch

        # 动态调整损失权重
        if current_epoch < self.quality_warmup_epochs:
            # 阶段1：只训练基础组件
            self.generation_loss_weight = 0.0
            self.quality_loss_weight = 0.0
            self.consistency_loss_weight = 0.0
        elif current_epoch < self.warmup_epochs:
            # 阶段2：逐渐增加辅助损失
            progress = (current_epoch - self.quality_warmup_epochs) / max(1,
                                                                          self.warmup_epochs - self.quality_warmup_epochs)
            self.generation_loss_weight = 0.05 * progress
            self.quality_loss_weight = 0.03 * progress
            self.consistency_loss_weight = 0.01 * progress
        else:
            # 阶段3：完全权重
            self.generation_loss_weight = 0.1
            self.quality_loss_weight = 0.05
            self.consistency_loss_weight = 0.02


    def on_validation_epoch_start(self):
        """验证epoch开始时确保模型状态"""
        # 验证时总是使用完整功能
        self.model.set_quality_prompts_enabled(True)
        self.model.set_quality_fusion_enabled(True)

    def validation_step(self, batch, batch_idx):
        # clip_utils.set_task(self)
        # output = self(batch)
        """【修复】验证步骤，确保模型状态一致"""
        # 确保验证时使用相同的模型配置
        self.model.set_quality_prompts_enabled(True)
        self.model.set_quality_fusion_enabled(True)

        clip_utils.set_task(self)
        output = self(batch)

        return output


    def validation_epoch_end(self, outs):
        clip_utils.epoch_wrapup(self)

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

    def _extract_task_performance(self, batch, output):
        """
        【新增】从当前batch和输出中提取任务性能
        """
        if "mmimdb" in self.current_tasks and "mmimdb_logits" in output:
            # MMIMDb任务：计算预测置信度
            logits = output["mmimdb_logits"]
            labels = torch.tensor(batch["label"]).float().to(logits.device)

            with torch.no_grad():
                # 计算预测置信度
                probs = torch.sigmoid(logits)
                # 使用预测置信度的最大值作为性能指标
                max_probs = torch.max(probs, dim=-1)[0]
                return max_probs

        elif "food101" in self.current_tasks and "food101_logits" in output:
            # Food101任务：计算预测置信度
            logits = output["food101_logits"]
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                max_probs = torch.max(probs, dim=-1)[0]
                return max_probs

        elif "hatememes" in self.current_tasks and "hatememes_logits" in output:
            # HateMemes任务：计算预测置信度
            logits = output["hatememes_logits"]
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)
                max_probs = torch.max(probs, dim=-1)[0]
                return max_probs

        else:
            # 默认返回中等性能
            batch_size = len(batch["missing_type"])
            return torch.ones(batch_size, device=self.device) * 0.6








