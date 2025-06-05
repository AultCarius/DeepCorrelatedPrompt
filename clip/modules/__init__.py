from .clip_missing_aware_prompt_module import CLIPransformerSS
# 添加新的导入
from .modal_generator import ModalGenerator, CrossModalGenerator
from .quality_estimator import QualityEstimator
from .quality_guide_fusion import QualityGuidedFusion
from .quality_aware_prompt import QualityAwarePromptLearner, IterativeQualityOptimization  # 【新增】
from .enhanced_quality_estimator import TaskRelevanceQualityEstimator,QualityAwareObjective # 【新增】
from .quality_aware_attention import AttentionReweightingFusion, QualityAwareTaskLoss          # 【新增】
from .quality_estimator import ObjectiveQualityAssessor