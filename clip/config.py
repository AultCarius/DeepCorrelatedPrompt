from sacred import Experiment

ex = Experiment("CLIP")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "mppd": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mmimdb": 0,
        "hatememes": 0,
        "food101": 0,
        "modal_generation": 0,
        "cycle_consistency": 0,
        "quality_estimation": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "exp"
    seed = 42
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # eval config (for bash execution)
    test_ratio = None
    test_type = None
    test_exp_name = None
    
    # fix backbone model (CLIP) weights
    fix_model = True
    
    # missing modality config
    missing_ratio = {'train': 0.7, 'val': 0.7, 'test': 0.7}
    missing_type = {'train': 'both', 'val': 'both', 'test': 'both'} # ['text', 'image', 'both'] in VL taskss
    both_ratio = 0.5   # missing both ratio
    missing_table_root = './datasets/missing_tables/'
    simulate_missing = False
    
    # missing_aware_prompts config
    prompt_length = 36
    prompt_depth = 6
        
    # Image setting
    train_transform_keys = ["CLIP_transform"]
    val_transform_keys = ["CLIP_transform"]
    image_size = 224
    draw_false_image = 1
    image_only = False

    # Text Setting
    max_text_len = 40
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    vit = "ViT-B/16"

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False
    mmimdb_class_num = 23
    hatememes_class_num = 2
    food101_class_num = 101    

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    finetune_first = False

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    # 【新增】模态生成器配置 - 必须在主config中定义
    modal_generator_config = {
        'hidden_size': 512,
        'num_layers': 3,
        'num_heads': 8,
        'dropout': 0.1
    }

    # 【新增】循环一致性损失权重 - 必须在主config中定义
    # cycle_loss_weight = 0.02
    # 【新增】质量评估配置
    quality_estimator_config = {
        'hidden_size': 512,
        'fusion_strategy': 'weighted'  # 'weighted' 或 'attention'
    }

    # 【更新】损失权重配置
    cycle_loss_weight = 0.02  # 循环一致性损失权重
    quality_loss_weight = 0.01  # 质量一致性损失权重
    task_auxiliary_weight = 0.1  # 【新增】任务辅助损失权重

    # 【新增】多维度质量评估配置
    enhanced_quality_config = {
        'use_geometric_quality': True,  # 几何质量评估
        'use_information_quality': True,  # 信息论质量评估
        'use_task_quality': True,  # 任务相关质量评估
        'use_contrastive_quality': True,  # 对比学习质量评估
        'use_mathematical_quality': True,  # 数学质量评估
        'fusion_strategy': 'quality_attention'  # 融合策略
    }

    # 【新增】合理质量评估配置
    reasonable_quality_config = {
        'importance_weight': 0.1,  # 模态重要性损失权重
        'authenticity_weight': 0.05,  # 特征真实性损失权重
        'difficulty_weight': 0.05,  # 任务难度损失权重
        'use_quality_fusion': True,  # 是否使用质量引导融合
    }

    # 【新增】质量监督损失权重
    quality_loss_weight = 0.01

    # 【新增】分阶段训练配置
    warmup_epochs = 5  # 前5轮只训练基础组件
    quality_aware_epochs = 10  # 接下来10轮启用质量感知

    # 【新增】质量感知prompt配置
    quality_aware_prompt_config = {
        'adaptation_weight_init': 0.1,  # 初始自适应权重
        'max_iterations': 2,  # 最大迭代次数
        'early_stop_threshold': 0.01  # 早停阈值
    }

    # 【新增】增强质量评估配置
    enhanced_quality_config = {
        'use_mathematical_quality': True,  # 是否使用数学特征质量
        'use_gradient_task_relevance': True,  # 是否使用基于梯度的任务相关性
        'use_perturbation_analysis': True,  # 是否使用扰动敏感性分析
        'use_task_relevance_predictor': True,  # 是否使用任务相关性预测器
        'predictor_hidden_size': 512,  # 预测器隐藏层大小
        'quality_aggregator_layers': [8, 16, 4]  # 质量聚合器层配置
    }

    # 【新增】质量感知融合配置
    quality_fusion_config = {
        'use_quality_guided_enhancement': True,  # 是否使用质量引导增强
        'use_cross_modal_interaction': True,  # 是否使用跨模态交互
        'use_adaptive_fusion': True,  # 是否使用自适应融合
        'interaction_threshold': 0.5,  # 跨模态交互阈值
        'enhancement_threshold': 0.3,  # 特征增强阈值
        'fusion_dropout': 0.1  # 融合层dropout
    }

    # 【更新】损失权重配置
    loss_weights = {
        'main_task_weight': 1.0,  # 主任务损失权重
        'cycle_loss_weight': 0.02,  # 循环一致性损失权重
        'predictor_loss_weight': 0.005,  # 预测器训练损失权重
        'quality_consistency_weight': 0.01,  # 质量一致性损失权重
        'gradient_alignment_weight': 0.002  # 梯度对齐损失权重（可选）
    }

    # 【新增】训练策略配置
    training_strategy = {
        'warmup_epochs': 5,  # 预热轮数
        'quality_aware_epochs': 10,  # 质量感知训练轮数
        'use_enhanced_quality': True,  # 是否使用增强质量评估
        'use_quality_aware_prompts': False,  # 是否使用质量感知prompt
        'enable_gradient_quality': True,  # 是否启用梯度质量分析
        'quality_loss_schedule': 'linear'  # 质量损失调度策略
    }

    # 【新增】调试和监控配置
    debug_config = {
        'log_quality_scores': True,  # 是否记录质量分数
        'save_quality_analysis': False,  # 是否保存质量分析结果
        'visualize_fusion_weights': False,  # 是否可视化融合权重
        'quality_log_interval': 100  # 质量分析日志间隔
    }




# Named configs for "environment" which define gpus and nodes, and paths
@ex.named_config
def env_dandelin():
    data_root = "/data2/dsets/dataset"
    log_dir = "/data2/clip/result"
    num_gpus = 8
    num_nodes = 1


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_randaug():
    exp_name = "mlm_itm_randaug"
    datasets = ["coco", "vg", "sbu", "gcc"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_mlm_itm_mpp():
    exp_name = "mlm_itm_mpp"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mpp": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200


@ex.named_config
def task_finetune_nlvr2():
    exp_name = "finetune_nlvr2"
    datasets = ["nlvr2"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_nlvr2_randaug():
    exp_name = "finetune_nlvr2_randaug"
    datasets = ["nlvr2"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"nlvr2": 1})
    batch_size = 128
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4


@ex.named_config
def task_finetune_vqa():
    exp_name = "finetune_vqa"
    datasets = ["vqa"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10


@ex.named_config
def task_finetune_vqa_randaug():
    exp_name = "finetune_vqa_randaug"
    datasets = ["vqa"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"vqa": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-4
    val_check_interval = 0.1
    lr_mult = 10

@ex.named_config
def task_finetune_hatememes():
    exp_name = "finetune_hatememes"
    datasets = ["Hatefull_Memes"]
    loss_names = _loss_names({"hatememes": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2 
    val_check_interval = 0.11
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 128 
    
@ex.named_config
def task_finetune_food101():
    exp_name = "finetune_food101"
    datasets = ["Food101"]
    loss_names = _loss_names({"food101": 1})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 512     
    
@ex.named_config
def task_finetune_mmimdb():
    exp_name = "finetune_mmimdb"
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
#     loss_names = _loss_names({"mmimdb": 1, "prompt": -0.5})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 1024
    # 【新增】模态生成器配置
    modal_generator_config = {
        'hidden_size': 512,
        'num_layers': 3,
        'num_heads': 8,
        'dropout': 0.1
    }
    # 【新增】循环一致性损失权重
    cycle_loss_weight = 0.02

@ex.named_config
def task_finetune_irtr_coco():
    exp_name = "finetune_irtr_coco"
    datasets = ["coco"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_coco_randaug():
    exp_name = "finetune_irtr_coco_randaug"
    datasets = ["coco"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k():
    exp_name = "finetune_irtr_f30k"
    datasets = ["f30k"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


@ex.named_config
def task_finetune_irtr_f30k_randaug():
    exp_name = "finetune_irtr_f30k_randaug"
    datasets = ["f30k"]
    train_transform_keys = ["pixelbert_randaug"]
    loss_names = _loss_names({"itm": 0.5, "irtr": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = None
    warmup_steps = 0.1
    get_recall_metric = True
    draw_false_text = 15
    learning_rate = 1e-4


# Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000


@ex.named_config
def vit32_base():
    vit = "vit_base_patch32_384"
    patch_size = 32
    hidden_size = 768
    num_heads = 12
    num_layers = 12
