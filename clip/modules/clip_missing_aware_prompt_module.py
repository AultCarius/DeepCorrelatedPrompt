import torch
import torch.nn as nn
import pytorch_lightning as pl
import clip.modules.vision_transformer_prompts as vit
import math
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from clip.modules import clip_utils, heads, objectives, clip
import copy

#==========================================
from .modal_generator import ModalGenerator
import torch.nn.functional as F
from .quality_estimator import QualityEstimator
from .quality_guide_fusion import QualityGuidedFusion
#====质量提示
from .quality_aware_prompt import QualityAwarePromptLearner, IterativeQualityOptimization
from .enhanced_quality_estimator import TaskRelevanceQualityEstimator,QualityAwareObjective
from .quality_aware_attention import AttentionReweightingFusion, QualityAwareTaskLoss

def load_clip_to_cpu(backbone_name, prompt_length, prompt_depth):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu")#.eval()
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
        prompt_length_half = prompt_length//3 # use half length for generating static prompts, and the other for generating dynamic prompts
        # Default is 1, which is compound shallow prompting
        self.prompt_depth = prompt_depth  # max=12, but will create 11 such shared prompts
        self.visual_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.visual_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 768, dtype=dtype), std=0.02))
        self.text_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.text_prompt_missing = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_complete = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_image = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        self.common_prompt_text = nn.Parameter(nn.init.normal_(torch.empty(prompt_length_half, 512, dtype=dtype), std=0.02))
        # Also make corresponding projection layers, for each prompt
        embed_dim_text = 512
        embed_dim_image = 768
        embed_dim = embed_dim_text + embed_dim_image
        r = 16
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_text),
                )
        self.compound_prompt_projections_text = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_text = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        
        single_layer = nn.Sequential(
                nn.Linear(embed_dim, embed_dim//r),
                nn.GELU(),
                nn.Linear(embed_dim//r, embed_dim_image),
                )
        self.compound_prompt_projections_image = _get_clones(single_layer, self.prompt_depth)
        self.layernorm_image = nn.ModuleList([torch.nn.LayerNorm(embed_dim) for _ in range(self.prompt_depth)])
        self.common_prompt_projection_image = nn.Sequential(
                nn.Linear(embed_dim_text, embed_dim_text//r),
                nn.GELU(),
                nn.Linear(embed_dim_text//r, embed_dim_image),
                )
        self.common_prompt_projection_text = nn.Sequential(
                nn.Linear(embed_dim_text, embed_dim_text//r),
                nn.GELU(),
                nn.Linear(embed_dim_text//r, embed_dim_text),
                )

    def forward(self, missing_type):

        # Before returning, need to transform
        # prompts to 768 for the visual side
        all_prompts_image = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        all_prompts_text = [ [] for _ in range(self.prompt_depth)]   # Prompts of prompt_depth layers
        for i in range(len(missing_type)):
            # set initial prompts for each modality
            if missing_type[i]==0:  # modality complete
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_complete
            elif missing_type[i]==1:  # missing text 
                initial_prompt_image = self.visual_prompt_complete
                initial_prompt_text = self.text_prompt_missing
                common_prompt = self.common_prompt_image
            elif missing_type[i]==2:  # missing image 
                initial_prompt_image = self.visual_prompt_missing
                initial_prompt_text = self.text_prompt_complete
                common_prompt = self.common_prompt_text
            # generate the prompts of the first layer
            all_prompts_image[0].append(self.compound_prompt_projections_image[0](self.layernorm_image[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            all_prompts_text[0].append(self.compound_prompt_projections_text[0](self.layernorm_text[0](torch.cat([initial_prompt_image, initial_prompt_text], -1))))
            # generate the prompts of the rest layers
            for index in range(1, self.prompt_depth):
                all_prompts_image[index].append(
                    self.compound_prompt_projections_image[index](self.layernorm_image[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_text[index-1][-1]], -1))))
                all_prompts_text[index].append(
                    self.compound_prompt_projections_text[index](self.layernorm_text[index](torch.cat([all_prompts_image[index-1][-1], all_prompts_text[index-1][-1]], -1))))
            all_prompts_image[0][i] = torch.cat([
                    all_prompts_image[0][i], 
                    self.common_prompt_projection_image(common_prompt)]
                    ,0)
            all_prompts_text[0][i] = torch.cat([
                    all_prompts_text[0][i], 
                    self.common_prompt_projection_text(common_prompt)]
                    ,0)
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
        # # 【新增】质量评估器
        # self.quality_estimator = QualityEstimator(hidden_size=512)
        #
        # # 【新增】质量引导融合器
        # self.quality_fusion = QualityGuidedFusion(
        #     hidden_size=512,
        #     fusion_strategy='adaptive_attention'  # 可配置
        # )
        #
        # # 【新增】质量感知Prompt学习器
        # self.quality_prompt_learner = QualityAwarePromptLearner(
        #     self.prompt_learner, prompt_length, prompt_depth
        # )
        #
        # # 【新增】迭代优化器
        # self.iterative_optimizer = IterativeQualityOptimization({
        #     'image_encoder': self.image_encoder,
        #     'text_encoder': self.text_encoder,
        #     'modal_generator': self.modal_generator,
        #     'quality_estimator': self.quality_estimator,
        #     'quality_fusion': self.quality_fusion,
        #     'quality_prompt_learner': self.quality_prompt_learner
        # })
        #
        # # 训练策略控制
        # self.use_iterative_optimization = True
        # self.use_quality_aware_prompts = True
        # 【新版】多维度质量评估器
        self.quality_estimator = TaskRelevanceQualityEstimator(
            hidden_size=512,
            num_classes=23  # mmimdb类别数
        )

        # 【新版】注意力重加权融合器
        self.quality_fusion = AttentionReweightingFusion(
            hidden_size=512,
            fusion_strategy='quality_attention'
        )

        # 质量感知的训练损失
        # self.quality_training_loss = QualityAwareTaskLoss()


    def forward(self, image, text, missing_type,true_labels=None):
        tokenized_texts = torch.stack([clip.tokenize(tx, context_length=77, truncate=True) for tx in text[0]], 0).to(image.get_device()).squeeze(1)  # extract texts from the first key  # [b, 77]
        # #logit_scale = self.logit_scale.exp()
        #
        # all_prompts_image, all_prompts_text = self.prompt_learner(missing_type)
        # text_features = self.text_encoder(tokenized_texts, all_prompts_text, missing_type)
        # image_features = self.image_encoder(image.type(self.dtype), all_prompts_image, missing_type)
        #
        # # 【新增】模态生成和特征增强
        # enhanced_image_features, enhanced_text_features = self.modal_generator(
        #     image_features, text_features, missing_type
        # )
        #
        # # 【新增】保存原始特征用于循环损失
        # self.last_image_features = image_features
        # self.last_text_features = text_features
        # # 3. 【新增】质量评估
        # quality_scores = self.quality_estimator(
        #     image_features, text_features,
        #     enhanced_image_features, enhanced_text_features,
        #     missing_type
        # )
        #
        # # 4. 【新增】质量引导融合
        # fused_features = self.quality_fusion(
        #     image_features, text_features,
        #     enhanced_image_features, enhanced_text_features,
        #     quality_scores, missing_type
        # )
        # # 在质量评估后保存质量分数
        # self.last_quality_scores = quality_scores
        #
        # # return torch.cat([image_features, text_features], -1)
        # # 使用增强后的特征进行拼接
        # # return torch.cat([enhanced_image_features, enhanced_text_features], -1)
        # return fused_features

        # if self.use_iterative_optimization and self.training:
        #     # 【新方法】迭代质量优化
        #     fused_features, image_features, text_features = self.iterative_optimizer(
        #         image, tokenized_texts, missing_type, tokenized_texts
        #     )
        #
        #     # 保存特征用于损失计算
        #     self.last_image_features = image_features
        #     self.last_text_features = text_features
        #
        # else:
        #     # 【原方法】单次前向传播 (推理时使用)
        #     if self.use_quality_aware_prompts:
        #         # 两阶段方法: 先获取质量，再生成质量感知prompts
        #
        #         # 第一阶段: 基础编码获取初始质量
        #         base_prompts_img, base_prompts_text = self.quality_prompt_learner(missing_type, None)
        #
        #         initial_image_features = self.image_encoder(image.type(self.dtype), base_prompts_img, missing_type)
        #         initial_text_features = self.text_encoder(tokenized_texts, base_prompts_text, missing_type)
        #
        #         initial_enhanced_img, initial_enhanced_text = self.modal_generator(
        #             initial_image_features, initial_text_features, missing_type
        #         )
        #
        #         initial_quality_scores = self.quality_estimator(
        #             initial_image_features, initial_text_features,
        #             initial_enhanced_img, initial_enhanced_text, missing_type
        #         )
        #
        #         # 第二阶段: 使用质量感知prompts重新编码
        #         quality_prompts_img, quality_prompts_text = self.quality_prompt_learner(
        #             missing_type, initial_quality_scores
        #         )
        #
        #         image_features = self.image_encoder(image.type(self.dtype), quality_prompts_img, missing_type)
        #         text_features = self.text_encoder(tokenized_texts, quality_prompts_text, missing_type)
        #
        #     else:
        #         # 原始方法
        #         all_prompts_image, all_prompts_text = self.prompt_learner(missing_type)
        #         image_features = self.image_encoder(image.type(self.dtype), all_prompts_image, missing_type)
        #         text_features = self.text_encoder(tokenized_texts, all_prompts_text, missing_type)
        #
        #     # 保存特征
        #     self.last_image_features = image_features
        #     self.last_text_features = text_features
        #
        #     # 生成和融合
        #     enhanced_image, enhanced_text = self.modal_generator(image_features, text_features, missing_type)
        #     quality_scores = self.quality_estimator(image_features, text_features, enhanced_image, enhanced_text,
        #                                             missing_type)
        #     fused_features = self.quality_fusion(image_features, text_features, enhanced_image, enhanced_text,
        #                                          quality_scores, missing_type)
        # 1. 原有编码流程
        all_prompts_image, all_prompts_text = self.prompt_learner(missing_type)
        text_features = self.text_encoder(tokenized_texts, all_prompts_text, missing_type)
        image_features = self.image_encoder(image.type(self.dtype), all_prompts_image, missing_type)

        # 保存原始特征用于循环损失
        self.last_image_features = image_features
        self.last_text_features = text_features

        # 2. 模态生成 (保持不变)
        enhanced_image_features, enhanced_text_features = self.modal_generator(
            image_features, text_features, missing_type
        )

        # 3. 【新版】多维度质量评估
        quality_scores, task_logits = self.quality_estimator(
            image_features, text_features,
            enhanced_image_features, enhanced_text_features,
            missing_type
        )

        # 保存质量分数和任务预测用于损失计算
        self.last_quality_scores = quality_scores
        self.last_task_logits = task_logits

        # 4. 【新版】注意力重加权融合
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
        hidden_size = 512*2
        print(config['prompt_length'])
        self.model = CustomCLIP(config['prompt_length'], config['prompt_depth'], clip_model)

        # 损失权重
        self.cycle_loss_weight = config.get('cycle_loss_weight', 0.02)
        self.quality_loss_weight = config.get('quality_loss_weight', 0.01)
        self.task_auxiliary_weight = config.get('task_auxiliary_weight', 0.1)  # 【新增】任务辅助损失权重

        # 质量感知的任务损失
        # 质量感知的任务损失
        self.quality_task_loss = QualityAwareTaskLoss(
            importance_weight=0.3,
            difficulty_weight=0.4,
            authenticity_weight=0.3
        )

        # 下游任务分类器修改为使用质量感知损失
        if self.hparams.config["loss_names"]["mmimdb"] > 0:
            cls_num = self.hparams.config["mmimdb_class_num"]
            self.mmimdb_classifier = nn.Linear(hidden_size, cls_num)
            self.mmimdb_classifier.apply(objectives.init_weights)

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
                if "prompt_learner" not in name and "prompt" not in name and 'ln_final' not in name and 'ln_post' not in name and name.split('.')[-1]!='proj':
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
        both_feats = self.model(img, text, batch["missing_type"],None)

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
            # ret.update(objectives.compute_hatememes(self, batch))
            ret.update(objectives.compute_enhanced_mmimdb(self, batch))

        # Multi-label classification for MM-IMDb
        if "mmimdb" in self.current_tasks:
            ret.update(objectives.compute_mmimdb(self, batch))
            
        # Classification for Food101
        if "food101" in self.current_tasks:
            ret.update(objectives.compute_food101(self, batch))              

        return ret

    def training_step(self, batch, batch_idx):
        clip_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k])
        # current_weights = self.get_current_loss_weights()
        # 【新增】循环一致性损失
        # if self.cycle_loss_weight > 0 and self.training:
        #     # 从模型中获取刚刚保存的原始特征
        #     if hasattr(self.model, 'last_image_features') and hasattr(self.model, 'last_text_features'):
        #         cycle_loss = self.model.modal_generator.compute_cycle_consistency_loss(
        #             self.model.last_image_features,
        #             self.model.last_text_features,
        #             batch["missing_type"]
        #         )
        #         if cycle_loss is not None and cycle_loss > 0:
        #             total_loss = total_loss + self.cycle_loss_weight * cycle_loss
        #             self.log("train/cycle_loss", cycle_loss)
        # # 【新增】质量监督损失
        # if self.quality_loss_weight > 0 and self.training:
        #     if hasattr(self.model, 'last_quality_scores'):
        #         quality_loss = self.compute_quality_supervision_loss(
        #             self.model.last_quality_scores, batch["missing_type"]
        #         )
        #         total_loss = total_loss + self.quality_loss_weight * quality_loss
        #         self.log("train/quality_loss", quality_loss)
        # 循环一致性损失 (保持不变)
        if self.cycle_loss_weight > 0 and self.training:
            if hasattr(self.model, 'last_image_features') and hasattr(self.model, 'last_text_features'):
                cycle_loss = self.model.modal_generator.compute_cycle_consistency_loss(
                    self.model.last_image_features,
                    self.model.last_text_features,
                    batch["missing_type"]
                )
                if cycle_loss is not None and cycle_loss > 0:
                    total_loss = total_loss + self.cycle_loss_weight * cycle_loss
                    self.log("train/cycle_loss", cycle_loss)

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

    def compute_quality_supervision_loss(self, quality_scores, missing_type):
        """计算质量监督损失"""
        if quality_scores is None:
            return torch.tensor(0.0, requires_grad=True).to(self.device)

        quality_loss = 0.0
        count = 0

        for i, mt in enumerate(missing_type):
            sample_quality = quality_scores[i]

            # 根据缺失类型设置目标质量
            if mt == 0:  # 完整样本
                target_confidence = 0.9
                target_consistency = 0.8
            else:  # 缺失样本
                target_confidence = 0.6
                target_consistency = 0.6

            # 监督生成置信度
            predicted_confidence = sample_quality['generation_confidence']
            quality_loss += F.mse_loss(
                predicted_confidence,
                torch.tensor(target_confidence).to(predicted_confidence.device).expand_as(predicted_confidence)
            )

            # 监督跨模态一致性
            predicted_consistency = sample_quality['cross_modal_consistency']
            quality_loss += F.mse_loss(
                predicted_consistency,
                torch.tensor(target_consistency).to(predicted_consistency.device).expand_as(predicted_consistency)
            )

            count += 2

        return quality_loss / max(count, 1)

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

