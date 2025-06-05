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
        x = outputs[0][self.prompt_length:]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection

        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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
        # print("========")
        # print(len(all_prompts_image))
        # # 遍历每个样本的 prompts 并获取长度
        # for i, prompts in enumerate(all_prompts_image):
        #     num_prompts = len(prompts)  # 列表长度即 prompts 数量
        #     print(f"样本 {i + 1} 的 prompts 数量：{num_prompts}")
        #     print(prompts[0].shape)
        all_prompts_image = [torch.stack(prompts) for prompts in all_prompts_image]
        all_prompts_text = [torch.stack(prompts) for prompts in all_prompts_text]
        # print(all_prompts_image)
        return all_prompts_image, all_prompts_text


class CustomCLIP(nn.Module):
    def __init__(self, prompt_length, prompt_depth, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(prompt_length, prompt_depth, clip_model)
        # print("self.prompt_learner",self.prompt_learner)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        # 【新增】模态生成器
        self.modal_generator = ModalGenerator(
            hidden_size=512,  # CLIP特征维度
            num_layers=3,
            num_heads=8,
            dropout=0.1
        )
        # 【新增】质量评估器
        self.quality_estimator = QualityEstimator(hidden_size=512)

        # 【新增】质量引导融合器
        self.quality_fusion = QualityGuidedFusion(
            hidden_size=512,
            fusion_strategy='adaptive_attention'  # 可配置
        )

        # 【新增】质量感知Prompt学习器
        self.quality_prompt_learner = QualityAwarePromptLearner(
            self.prompt_learner, prompt_length, prompt_depth
        )

        # 【新增】迭代优化器
        self.iterative_optimizer = IterativeQualityOptimization({
            'image_encoder': self.image_encoder,
            'text_encoder': self.text_encoder,
            'modal_generator': self.modal_generator,
            'quality_estimator': self.quality_estimator,
            'quality_fusion': self.quality_fusion,
            'quality_prompt_learner': self.quality_prompt_learner
        })

        # 训练策略控制
        self.use_iterative_optimization = True
        self.use_quality_aware_prompts = True
        #========================================
        # 【第一步新增】客观质量评估器
        from .quality_estimator import ObjectiveQualityAssessor
        self.objective_quality_assessor = ObjectiveQualityAssessor(hidden_size=512)



    def forward(self, image, text, missing_type):
        tokenized_texts = torch.stack([clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0).to(
            image.get_device()).squeeze(1)  # extract texts from the first key  # [b, 77]
        # 1. 基础编码
        all_prompts_image, all_prompts_text = self.prompt_learner(missing_type)
        text_features = self.text_encoder(tokenized_texts, all_prompts_text, missing_type)
        image_features = self.image_encoder(image.type(self.dtype), all_prompts_image, missing_type)

        # 2. 模态生成
        enhanced_image_features, enhanced_text_features = self.modal_generator(
            image_features, text_features, missing_type
        )

        # 3. 【第一步新增】客观质量评估
        quality_scores = self.objective_quality_assessor(
            image_features, text_features,
            enhanced_image_features, enhanced_text_features,
            missing_type
        )

        # 保存质量分数供损失计算使用
        # 保存所有必要的特征
        self.last_image_features = image_features
        self.last_text_features = text_features
        self.last_enhanced_image_features = enhanced_image_features  # 新增
        self.last_enhanced_text_features = enhanced_text_features  # 新增
        self.last_quality_scores = quality_scores

        # 4. 特征融合（暂时保持原有逻辑，后续步骤会改进）
        fused_features = self.quality_fusion(
            image_features, text_features,
            enhanced_image_features, enhanced_text_features,
            quality_scores, missing_type
        )

        return fused_features


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
        # 【新增】训练策略控制
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.quality_aware_epochs = config.get('quality_aware_epochs', 10)

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

            # # Double check
            # enabled = set()
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         enabled.add(name)
            # print(f"Parameters to be updated: {enabled}")

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

        # 处理缺失模态的掩码（保持原有逻辑）
        feature_dim = both_feats.shape[1] // 2
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                pass
            elif batch["missing_type"][idx] == 1:  # missing text
                both_feats[idx, feature_dim:].zero_()
            elif batch["missing_type"][idx] == 2:  # missing image
                both_feats[idx, :feature_dim].zero_()

        ret = {
            "cls_feats": both_feats,
        }

        return ret

    def forward(self, batch):
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
            # ret.update(objectives.compute_mmimdb(self, batch))
            ret.update(objectives.compute_enhanced_mmimdb(self, batch))

        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        # 主任务损失
        main_loss = sum([v for k, v in output.items() if "loss" in k and "quality" not in k])

        # 质量损失（已经在各个任务中计算）
        quality_loss = sum([v for k, v in output.items() if "quality_loss" in k])

        current_weights = self.get_current_loss_weights()
        cycle_loss = 0.0
        total_loss = main_loss + self.quality_loss_weight * quality_loss
        # 循环一致性损失
        if current_weights['cycle'] > 0 and hasattr(self.model, 'last_image_features'):
            cycle_loss = self.model.modal_generator.compute_cycle_consistency_loss(
                self.model.last_image_features,
                self.model.last_text_features,
                batch["missing_type"]
            )
            if cycle_loss is not None and cycle_loss > 0:
                total_loss = total_loss + current_weights['cycle'] * cycle_loss
                self.log("train/cycle_loss", cycle_loss)
                self.log("train/cycle_weight", current_weights['cycle'])


        if not torch.isfinite(total_loss):
            print("[NaN Detect] total_loss is NaN")
            print("main_loss =", main_loss)
            print("quality_loss =", quality_loss)
            print("cycle_loss =", cycle_loss)

        # 记录训练阶段
        self.log("train/training_stage", float(self.current_epoch))
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

    def compute_quality_loss_in_training(self, model_output, batch):
        """
        在训练中计算真正客观的质量损失
        """
        if not hasattr(self.model, 'last_quality_scores'):
            return torch.tensor(0.0)

        # 获取当前任务性能
        current_task = self.current_tasks[0] if self.current_tasks else None
        task_performance = extract_current_task_performance(
            model_output, batch, current_task
        )

        # 计算客观质量损失
        quality_loss = compute_objective_quality_loss(
            self.model.last_quality_scores,
            self.model.last_image_features,
            self.model.last_text_features,
            getattr(self.model, 'last_enhanced_image_features', None),
            getattr(self.model, 'last_enhanced_text_features', None),
            batch["missing_type"],
            task_performance
        )

        return quality_loss

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


def compute_objective_quality_loss(quality_scores, image_features, text_features,
                                   enhanced_image_features, enhanced_text_features,
                                   missing_type, current_task_performance):
    """
    完全基于客观指标的质量损失，不使用任何人工标签

    核心思想：
    1. 质量指标与任务性能应该正相关
    2. 不同质量指标之间应该有合理的内在一致性
    3. 利用数据本身的内在结构作为监督信号
    """

    # 1. 【任务相关性损失】质量分数应该与实际任务性能正相关
    task_correlation_loss = 0.0
    if current_task_performance is not None:
        for i, quality in enumerate(quality_scores):
            # 计算综合质量分数
            overall_quality = (
                                      quality['contrastive_quality'] +
                                      quality['cross_modal_consistency'] +
                                      quality['representation_quality']
                              ) / 3

            # 期望：质量高的样本任务性能也高
            # 使用ranking loss而非绝对值匹配
            task_correlation_loss += F.relu(0.1 - overall_quality * current_task_performance[i])

    # 2. 【内在一致性损失】不同质量指标间应该有合理关系
    consistency_loss = 0.0
    for quality in quality_scores:
        contrastive_q = quality['contrastive_quality']
        consistency_q = quality['cross_modal_consistency']
        confidence_q = quality['generation_confidence']

        # 对比质量高 → 一致性也应该高
        consistency_loss += F.relu(contrastive_q - consistency_q - 0.2)

        # 一致性高 → 生成置信度也应该高
        consistency_loss += F.relu(consistency_q - confidence_q - 0.1)

    # 3. 【物理约束损失】利用数据内在结构
    physics_loss = 0.0
    batch_size = len(quality_scores)

    if batch_size > 1:
        # 完整模态样本的质量应该普遍高于缺失模态样本
        complete_indices = [i for i, mt in enumerate(missing_type) if mt == 0]
        missing_indices = [i for i, mt in enumerate(missing_type) if mt != 0]

        if len(complete_indices) > 0 and len(missing_indices) > 0:
            complete_qualities = torch.stack([
                (quality_scores[i]['contrastive_quality'] +
                 quality_scores[i]['cross_modal_consistency']) / 2
                for i in complete_indices
            ])
            missing_qualities = torch.stack([
                (quality_scores[i]['contrastive_quality'] +
                 quality_scores[i]['cross_modal_consistency']) / 2
                for i in missing_indices
            ])

            # margin ranking loss: 完整模态质量应该高于缺失模态
            complete_mean = complete_qualities.mean()
            missing_mean = missing_qualities.mean()
            physics_loss += F.relu(missing_mean - complete_mean + 0.1)

    # 4. 【特征距离一致性】生成特征的距离应该反映质量
    distance_consistency_loss = 0.0
    for i, quality in enumerate(quality_scores):
        if missing_type[i] == 1:  # 缺失文本
            # 原始文本与生成文本的距离
            text_distance = F.mse_loss(text_features[i], enhanced_text_features[i])
            generation_confidence = quality['generation_confidence']

            # 期望：距离小 ↔ 置信度高
            distance_consistency_loss += F.mse_loss(
                1.0 - text_distance,  # 距离小 → 值大
                generation_confidence
            )

        elif missing_type[i] == 2:  # 缺失图像
            img_distance = F.mse_loss(image_features[i], enhanced_image_features[i])
            generation_confidence = quality['generation_confidence']

            distance_consistency_loss += F.mse_loss(
                1.0 - img_distance,
                generation_confidence
            )

    # 5. 【信息论约束】利用熵和互信息
    info_theory_loss = 0.0
    for quality in quality_scores:
        uncertainty = quality['prediction_uncertainty']
        consistency = quality['cross_modal_consistency']

        # 高不确定性 + 高一致性 = 矛盾，应该惩罚
        contradiction_penalty = uncertainty * consistency
        info_theory_loss += contradiction_penalty

    # 总的客观质量损失
    total_quality_loss = (
                                 0.3 * task_correlation_loss +
                                 0.25 * consistency_loss +
                                 0.2 * physics_loss +
                                 0.15 * distance_consistency_loss +
                                 0.1 * info_theory_loss
                         ) / max(len(quality_scores), 1)


# 任务性能提取函数
def extract_current_task_performance(model_output, batch, task_name):
    """
    从当前任务输出中提取性能指标作为质量监督信号
    """
    if task_name == "mmimdb":
        # 对于多标签分类，用预测置信度作为性能指标
        logits = model_output.get('mmimdb_logits')
        if logits is not None:
            # 计算预测置信度
            probs = torch.sigmoid(logits)
            confidence = torch.max(probs, dim=-1)[0]  # 最高类别置信度
            return confidence.detach()

    elif task_name == "hatememes":
        # 对于二分类，用预测确定性
        logits = model_output.get('hatememes_logits')
        if logits is not None:
            probs = F.softmax(logits, dim=-1)
            # 预测越确定（接近0或1），性能指标越高
            certainty = torch.max(probs, dim=-1)[0]
            return certainty.detach()

    return None
