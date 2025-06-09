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
        self.enhanced_quality_estimator = EnhancedQualityEstimator(hidden_size=512)

        # 【新增】质量引导融合器
        self.quality_aware_fusion = QualityAwareFusion(hidden_size=512)

        # 【新增】质量感知Prompt学习器
        self.quality_prompt_learner = QualityAwarePromptLearner(
            self.prompt_learner, prompt_length, prompt_depth
        )


        # 训练策略控制
        self.use_iterative_optimization = True
        self.use_quality_aware_prompts = True


        # 训练控制参数
        self.use_enhanced_quality = True  # 是否使用新的质量评估
        self.use_quality_aware_prompts = False  # 是否使用质量感知prompt

        # 缓存变量，用于损失计算
        self.cached_features = {}
        self.cached_quality_results = None

        #========================================
        # # 【第一步新增】客观质量评估器
        # from .quality_estimator import ObjectiveQualityAssessor
        # self.objective_quality_assessor = ObjectiveQualityAssessor(hidden_size=512)

    def create_task_forward_function(self, tokenized_texts, all_prompts_text, all_prompts_image, missing_type):
        """创建任务前向函数，用于质量评估中的扰动分析"""
        def task_forward_fn(img_feat, text_feat):
            # 简化的前向，只做特征编码，不涉及prompt
            return torch.cat([img_feat, text_feat], dim=-1)
        return task_forward_fn
      
    def forward(self, image, text, missing_type):
        tokenized_texts = torch.stack([clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0).to(
            image.get_device()).squeeze(1)  # extract texts from the first key  # [b, 77]

        # 1. 基础编码
        all_prompts_image, all_prompts_text = self.prompt_learner(missing_type)

        # 启用梯度计算，为质量评估做准备
        image_features = self.image_encoder(image.type(self.dtype), all_prompts_image, missing_type)
        text_features = self.text_encoder(tokenized_texts, all_prompts_text, missing_type)

        # 确保特征有梯度（用于梯度分析）
        if self.training:
            image_features = image_features.requires_grad_(True)
            text_features = text_features.requires_grad_(True)

        # 2. 模态生成
        enhanced_image_features, enhanced_text_features = self.modal_generator(
            image_features, text_features, missing_type
        )

        # 3. 【新增】增强质量评估
        if self.use_enhanced_quality:
            # 创建任务前向函数用于扰动分析
            task_forward_fn = self.create_task_forward_function(
                tokenized_texts, all_prompts_text, all_prompts_image, missing_type
            )

            # 计算临时任务损失用于梯度分析（在训练模式下）
            task_loss = None
            if self.training:
                # 创建临时的分类头用于计算损失
                temp_classifier = nn.Linear(1024, 2).to(image_features.device)  # 假设2分类任务
                temp_features = torch.cat([image_features, text_features], dim=-1)
                temp_logits = temp_classifier(temp_features)
                temp_labels = torch.zeros(temp_logits.size(0), dtype=torch.long).to(image_features.device)
                task_loss = F.cross_entropy(temp_logits, temp_labels)

            # 质量评估
            quality_results = self.enhanced_quality_estimator(
                image_features, text_features, task_loss, task_forward_fn
            )

            # 缓存质量结果用于损失计算
            self.cached_quality_results = quality_results

            # 4. 【新增】质量感知融合
            fused_features = self.quality_aware_fusion(
                image_features, text_features,
                enhanced_image_features, enhanced_text_features,
                quality_results, missing_type
            )
        else:
            # 使用原有融合方式
            fused_features = torch.cat([image_features, text_features], dim=-1)

        # 缓存特征用于损失计算
        self.cached_features = {
            'image_features': image_features,
            'text_features': text_features,
            'enhanced_image_features': enhanced_image_features,
            'enhanced_text_features': enhanced_text_features,
            'fused_features': fused_features
        }


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
        self.predictor_loss_weight = config.get('predictor_loss_weight', 0.005)  # 预测器损失权重
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
            # ret.update(objectives.compute_enhanced_mmimdb(self, batch))
            ret.update(objectives.compute_enhanced_mmimdb_v2(self, batch))


        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)

        # 主任务损失
        main_loss = sum([v for k, v in output.items() if "loss" in k and "quality" not in k and "predictor" not in k])

        # 质量相关损失
        quality_losses = self.compute_quality_losses()

        total_loss = main_loss

        # 添加质量损失
        if quality_losses['predictor_loss'] > 0:
            total_loss += self.predictor_loss_weight * quality_losses['predictor_loss']
            self.log("train/predictor_loss", quality_losses['predictor_loss'])

        if quality_losses['cycle_loss'] > 0:
            total_loss += self.cycle_loss_weight * quality_losses['cycle_loss']
            self.log("train/cycle_loss", quality_losses['cycle_loss'])

        # 记录各个损失
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


    def compute_quality_losses(self):
        """计算质量相关的损失"""
        total_quality_loss = 0.0
        total_predictor_loss = 0.0

        if self.model.cached_quality_results is not None:
            for quality_result in self.model.cached_quality_results:
                # 预测器训练损失
                predictor_loss = quality_result.get('predictor_loss', 0)
                if isinstance(predictor_loss, torch.Tensor):
                    total_predictor_loss += predictor_loss


        # 循环一致性损失
        cycle_loss = 0.0
        if hasattr(self.model, 'cached_features') and self.model.cached_features:
            cycle_loss = self.model.modal_generator.compute_cycle_consistency_loss(
                self.model.cached_features['image_features'],
                self.model.cached_features['text_features'],
                [0] * self.model.cached_features['image_features'].size(0)  # 假设都是完整模态
            )

        return {
            'predictor_loss': total_predictor_loss,
            'cycle_loss': cycle_loss,
            'total_quality_loss': total_predictor_loss + cycle_loss
        }


