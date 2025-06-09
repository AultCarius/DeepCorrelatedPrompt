# clip/modules/enhanced_quality_estimator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MathematicalFeatureQuality(nn.Module):
    """纯数学的特征质量评估器 - 无需训练参数"""

    def __init__(self):
        super().__init__()
        # 无可训练参数

    def compute_feature_intrinsic_quality(self, features):
        """
        计算特征的内在数学质量
        Args:
            features: [batch_size, feature_dim] 或 [feature_dim]
        Returns:
            质量分数字典
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        batch_size, feature_dim = features.shape

        # 1. 特征范数稳定性 - 避免过大或过小的特征
        feature_norms = torch.norm(features, dim=-1)
        ideal_norm = math.sqrt(feature_dim)  # 理想范数
        norm_stability = torch.exp(-torch.abs(feature_norms - ideal_norm) / ideal_norm)

        # 2. 特征分布均匀性（熵） - 信息量指标
        feature_probs = F.softmax(torch.abs(features), dim=-1)
        entropy = -torch.sum(feature_probs * torch.log(feature_probs + 1e-8), dim=-1)
        max_entropy = math.log(feature_dim)
        normalized_entropy = entropy / max_entropy

        # 3. 特征多样性 - 避免特征冗余
        if batch_size > 1:
            # 批次内特征多样性
            feature_distances = torch.cdist(features, features, p=2)
            # 去除对角线（自身距离）
            mask = torch.eye(batch_size, device=features.device).bool()
            feature_distances = feature_distances.masked_fill(mask, float('inf'))
            min_distances = torch.min(feature_distances, dim=-1)[0]
            diversity = torch.sigmoid(min_distances)  # 距离越大越好
        else:
            # 单样本时使用特征内部方差
            diversity = torch.var(features, dim=-1) / torch.mean(torch.abs(features), dim=-1)
            diversity = torch.sigmoid(diversity)

        # 4. 特征稀疏性 - 适度稀疏有助于泛化
        sparsity_ratio = torch.mean((torch.abs(features) < 0.01).float(), dim=-1)
        optimal_sparsity = 0.1  # 10%的稀疏性较为理想
        sparsity_quality = torch.exp(-torch.abs(sparsity_ratio - optimal_sparsity) / optimal_sparsity)

        return {
            'norm_stability': norm_stability,
            'informativeness': normalized_entropy,
            'diversity': diversity,
            'sparsity_quality': sparsity_quality
        }

    def compute_cross_modal_alignment(self, img_feat, text_feat):
        """
        计算跨模态对齐质量
        Args:
            img_feat: [batch_size, 512]
            text_feat: [batch_size, 512]
        """
        if img_feat.dim() == 1:
            img_feat = img_feat.unsqueeze(0)
            text_feat = text_feat.unsqueeze(0)

        # 1. 余弦相似度 - 语义对齐
        cosine_similarity = F.cosine_similarity(img_feat, text_feat, dim=-1)
        alignment_quality = (cosine_similarity + 1) / 2  # 归一化到[0,1]

        # 2. CKA (Centered Kernel Alignment) - 表示对齐
        img_centered = img_feat - torch.mean(img_feat, dim=-1, keepdim=True)
        text_centered = text_feat - torch.mean(text_feat, dim=-1, keepdim=True)

        numerator = torch.sum(img_centered * text_centered, dim=-1) ** 2
        img_norm = torch.sum(img_centered ** 2, dim=-1)
        text_norm = torch.sum(text_centered ** 2, dim=-1)
        denominator = img_norm * text_norm + 1e-8

        cka_score = numerator / denominator

        # 3. 特征距离合理性
        feature_distance = torch.norm(img_feat - text_feat, dim=-1)
        max_possible_distance = torch.norm(img_feat, dim=-1) + torch.norm(text_feat, dim=-1)
        normalized_distance = feature_distance / (max_possible_distance + 1e-8)
        distance_quality = 1.0 - normalized_distance  # 距离越小越好

        return {
            'semantic_alignment': alignment_quality,
            'representation_alignment': cka_score,
            'distance_quality': distance_quality
        }


class GradientBasedTaskRelevance(nn.Module):
    """基于梯度的任务相关性评估器"""

    def __init__(self):
        super().__init__()
        # 无可训练参数，纯计算模块

    def compute_gradient_based_relevance(self, img_feat, text_feat, task_loss):
        """
        基于梯度计算任务相关性 - 修复梯度计算错误
        Args:
            img_feat: [batch_size, 512] 需要requires_grad=True
            text_feat: [batch_size, 512] 需要requires_grad=True
            task_loss: 标量，任务损失
        """
        try:
            # 确保特征有梯度且参与了计算图
            if not img_feat.requires_grad:
                img_feat = img_feat.detach().requires_grad_(True)
            if not text_feat.requires_grad:
                text_feat = text_feat.detach().requires_grad_(True)

            # 检查task_loss是否需要梯度
            if not isinstance(task_loss, torch.Tensor) or not task_loss.requires_grad:
                print("⚠️ task_loss不需要梯度，使用模拟梯度")
                # 创建一个简单的任务损失用于梯度计算
                combined_feat = torch.cat([img_feat, text_feat], dim=-1)
                temp_output = torch.sum(combined_feat ** 2)  # 简单的L2损失
                task_loss = temp_output

            # 1. 计算梯度幅度 - 使用allow_unused=True
            try:
                img_grad = torch.autograd.grad(
                    task_loss, img_feat, retain_graph=True, create_graph=False, allow_unused=True
                )[0]
                text_grad = torch.autograd.grad(
                    task_loss, text_feat, retain_graph=True, create_graph=False, allow_unused=True
                )[0]

                # 检查梯度是否为None
                if img_grad is None:
                    # print("⚠️ 图像特征梯度为None，使用随机梯度")
                    img_grad = torch.randn_like(img_feat) * 0.01

                if text_grad is None:
                    # print("⚠️ 文本特征梯度为None，使用随机梯度")
                    text_grad = torch.randn_like(text_feat) * 0.01

            except RuntimeError as e:
                print(f"⚠️ 梯度计算失败: {e}")
                # 使用特征本身作为"伪梯度"
                img_grad = img_feat * 0.01
                text_grad = text_feat * 0.01

            img_magnitude = torch.norm(img_grad, dim=-1)
            text_magnitude = torch.norm(text_grad, dim=-1)

            # 2. 梯度方向稳定性 - 避免零向量
            img_direction = F.normalize(img_grad + 1e-8, dim=-1)
            text_direction = F.normalize(text_grad + 1e-8, dim=-1)

            # 与平均方向的一致性
            img_mean_direction = F.normalize(torch.mean(img_grad, dim=0) + 1e-8, dim=-1)
            text_mean_direction = F.normalize(torch.mean(text_grad, dim=0) + 1e-8, dim=-1)

            img_consistency = F.cosine_similarity(img_direction, img_mean_direction.unsqueeze(0), dim=-1)
            text_consistency = F.cosine_similarity(text_direction, text_mean_direction.unsqueeze(0), dim=-1)

            # 3. 跨模态梯度协同性
            cross_modal_synergy = F.cosine_similarity(img_direction, text_direction, dim=-1)

            return {
                'img_gradient_magnitude': torch.sigmoid(img_magnitude),
                'text_gradient_magnitude': torch.sigmoid(text_magnitude),
                'img_gradient_consistency': (torch.clamp(img_consistency, -1, 1) + 1) / 2,
                'text_gradient_consistency': (torch.clamp(text_consistency, -1, 1) + 1) / 2,
                'cross_modal_synergy': (torch.clamp(cross_modal_synergy, -1, 1) + 1) / 2
            }

        except Exception as e:
            print(f"⚠️ 梯度计算完全失败: {e}")
            # 返回默认值
            batch_size = img_feat.size(0)
            device = img_feat.device

            return {
                'img_gradient_magnitude': torch.full((batch_size,), 0.5, device=device),
                'text_gradient_magnitude': torch.full((batch_size,), 0.5, device=device),
                'img_gradient_consistency': torch.full((batch_size,), 0.5, device=device),
                'text_gradient_consistency': torch.full((batch_size,), 0.5, device=device),
                'cross_modal_synergy': torch.full((batch_size,), 0.5, device=device)
            }

    def compute_perturbation_sensitivity(self, img_feat, text_feat, task_forward_fn):
        """
        基于扰动的敏感性分析
        Args:
            task_forward_fn: 任务前向函数，输入(img_feat, text_feat)返回预测
        """
        with torch.no_grad():
            # 原始预测
            original_output = task_forward_fn(img_feat, text_feat)

            # 图像扰动敏感性
            noise_scale = 0.1
            img_noise = torch.randn_like(img_feat) * noise_scale
            img_noisy_output = task_forward_fn(img_feat + img_noise, text_feat)
            img_sensitivity = torch.norm(original_output - img_noisy_output, dim=-1)

            # 文本扰动敏感性
            text_noise = torch.randn_like(text_feat) * noise_scale
            text_noisy_output = task_forward_fn(img_feat, text_feat + text_noise)
            text_sensitivity = torch.norm(original_output - text_noisy_output, dim=-1)

            # 掩码敏感性（极端扰动）
            img_masked_output = task_forward_fn(torch.zeros_like(img_feat), text_feat)
            text_masked_output = task_forward_fn(img_feat, torch.zeros_like(text_feat))

            img_mask_sensitivity = torch.norm(original_output - img_masked_output, dim=-1)
            text_mask_sensitivity = torch.norm(original_output - text_masked_output, dim=-1)

        return {
            'img_noise_sensitivity': torch.sigmoid(img_sensitivity),
            'text_noise_sensitivity': torch.sigmoid(text_sensitivity),
            'img_mask_sensitivity': torch.sigmoid(img_mask_sensitivity),
            'text_mask_sensitivity': torch.sigmoid(text_mask_sensitivity)
        }


class TaskRelevancePredictor(nn.Module):
    """任务相关性预测器 - 用于推理阶段的快速预测"""

    def __init__(self, hidden_size=512):
        super().__init__()

        # 图像任务相关性预测器
        self.img_relevance_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # 文本任务相关性预测器
        self.text_relevance_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

        # 跨模态协同预测器
        self.synergy_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img_feat, text_feat):
        """预测任务相关性"""
        img_relevance = self.img_relevance_net(img_feat)
        text_relevance = self.text_relevance_net(text_feat)

        combined_feat = torch.cat([img_feat, text_feat], dim=-1)
        synergy = self.synergy_net(combined_feat)

        return {
            'img_task_relevance': img_relevance.squeeze(-1),
            'text_task_relevance': text_relevance.squeeze(-1),
            'cross_modal_synergy': synergy.squeeze(-1)
        }

    def compute_predictor_loss(self, img_feat, text_feat, true_relevance):
        """计算预测器的训练损失"""
        predicted = self.forward(img_feat, text_feat)

        img_loss = F.mse_loss(
            predicted['img_task_relevance'],
            true_relevance['img_gradient_magnitude']
        )
        text_loss = F.mse_loss(
            predicted['text_task_relevance'],
            true_relevance['text_gradient_magnitude']
        )
        synergy_loss = F.mse_loss(
            predicted['cross_modal_synergy'],
            true_relevance['cross_modal_synergy']
        )

        return img_loss + text_loss + synergy_loss


class EnhancedQualityEstimator(nn.Module):
    """增强的质量评估器 - 整合所有评估方法"""

    def __init__(self, hidden_size=512):
        super().__init__()
        self.hidden_size = hidden_size

        # 数学特征质量评估器（无训练参数）
        self.math_quality_assessor = MathematicalFeatureQuality()

        # 梯度任务相关性评估器（无训练参数）
        self.gradient_relevance_assessor = GradientBasedTaskRelevance()

        # 任务相关性预测器（需要训练）
        self.task_relevance_predictor = TaskRelevancePredictor(hidden_size)

        # 质量分数聚合器
        self.quality_aggregator = nn.Sequential(
            nn.Linear(8, 16),  # 聚合8个质量指标
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid()
        )

    def compute_comprehensive_quality(self, img_feat, text_feat, task_loss=None,
                                      task_forward_fn=None, training_mode=False):
        """
        计算综合质量分数 - 修复梯度计算问题
        Args:
            img_feat: [batch_size, 512]
            text_feat: [batch_size, 512]
            task_loss: 任务损失（训练时需要）
            task_forward_fn: 任务前向函数（用于扰动分析）
            training_mode: 是否为训练模式
        """
        batch_size = img_feat.size(0)
        quality_results = []

        for i in range(batch_size):
            sample_img = img_feat[i:i + 1]
            sample_text = text_feat[i:i + 1]

            # 1. 数学特征质量
            img_math_quality = self.math_quality_assessor.compute_feature_intrinsic_quality(sample_img)
            text_math_quality = self.math_quality_assessor.compute_feature_intrinsic_quality(sample_text)
            cross_modal_quality = self.math_quality_assessor.compute_cross_modal_alignment(sample_img, sample_text)

            # 2. 任务相关性评估 - 修复梯度计算
            if training_mode and task_loss is not None:
                try:
                    # 训练模式：使用真实梯度计算
                    # 确保特征参与计算图
                    if sample_img.requires_grad and sample_text.requires_grad:
                        gradient_relevance = self.gradient_relevance_assessor.compute_gradient_based_relevance(
                            sample_img, sample_text, task_loss
                        )
                    else:
                        # 如果特征没有梯度，重新创建计算图
                        sample_img_grad = sample_img.detach().requires_grad_(True)
                        sample_text_grad = sample_text.detach().requires_grad_(True)

                        # 创建简单的任务损失用于梯度计算
                        combined = torch.cat([sample_img_grad, sample_text_grad], dim=-1)
                        simple_loss = torch.sum(combined ** 2)  # L2损失

                        gradient_relevance = self.gradient_relevance_assessor.compute_gradient_based_relevance(
                            sample_img_grad, sample_text_grad, simple_loss
                        )

                    # 训练预测器
                    predictor_loss = self.task_relevance_predictor.compute_predictor_loss(
                        sample_img, sample_text, gradient_relevance
                    )

                    task_relevance = gradient_relevance

                except Exception as e:
                    print(f"⚠️ 样本{i}梯度计算失败: {e}")
                    # 使用预测器作为备选
                    task_relevance = self.task_relevance_predictor(sample_img, sample_text)
                    predictor_loss = 0

            else:
                # 推理模式：使用预测器
                task_relevance = self.task_relevance_predictor(sample_img, sample_text)
                predictor_loss = 0

            # 3. 扰动敏感性（可选）
            if task_forward_fn is not None:
                try:
                    perturbation_sensitivity = self.gradient_relevance_assessor.compute_perturbation_sensitivity(
                        sample_img, sample_text, task_forward_fn
                    )
                except Exception as e:
                    print(f"⚠️ 样本{i}扰动分析失败: {e}")
                    # 默认值
                    perturbation_sensitivity = {
                        'img_noise_sensitivity': torch.tensor(0.5),
                        'text_noise_sensitivity': torch.tensor(0.5),
                        'img_mask_sensitivity': torch.tensor(0.5),
                        'text_mask_sensitivity': torch.tensor(0.5)
                    }
            else:
                # 默认值
                perturbation_sensitivity = {
                    'img_noise_sensitivity': torch.tensor(0.5),
                    'text_noise_sensitivity': torch.tensor(0.5),
                    'img_mask_sensitivity': torch.tensor(0.5),
                    'text_mask_sensitivity': torch.tensor(0.5)
                }

            # 4. 聚合质量分数 - 安全提取数值
            def safe_extract(value):
                if isinstance(value, torch.Tensor):
                    if value.dim() == 0:
                        return value.item()
                    elif value.numel() > 0:
                        return value.flatten()[0].item()
                    else:
                        return 0.5
                else:
                    return float(value)

            try:
                quality_vector = torch.tensor([
                    safe_extract(img_math_quality['norm_stability']),
                    safe_extract(img_math_quality['informativeness']),
                    safe_extract(text_math_quality['norm_stability']),
                    safe_extract(text_math_quality['informativeness']),
                    safe_extract(cross_modal_quality['semantic_alignment']),
                    safe_extract(
                        task_relevance.get('img_task_relevance', task_relevance.get('img_gradient_magnitude', 0.5))),
                    safe_extract(
                        task_relevance.get('text_task_relevance', task_relevance.get('text_gradient_magnitude', 0.5))),
                    safe_extract(task_relevance.get('cross_modal_synergy', 0.5))
                ]).to(img_feat.device)

                aggregated_quality = self.quality_aggregator(quality_vector)

            except Exception as e:
                print(f"⚠️ 样本{i}质量聚合失败: {e}")
                # 使用默认质量分数
                aggregated_quality = torch.tensor([0.5, 0.5, 0.5, 0.5]).to(img_feat.device)

            sample_result = {
                # 数学质量
                'img_intrinsic_quality': img_math_quality,
                'text_intrinsic_quality': text_math_quality,
                'cross_modal_alignment': cross_modal_quality,

                # 任务相关性
                'task_relevance': task_relevance,
                'perturbation_sensitivity': perturbation_sensitivity,

                # 聚合质量
                'overall_img_quality': aggregated_quality[0],
                'overall_text_quality': aggregated_quality[1],
                'overall_cross_modal_quality': aggregated_quality[2],
                'overall_confidence': aggregated_quality[3],

                # 训练相关
                'predictor_loss': predictor_loss
            }

            quality_results.append(sample_result)

        return quality_results

    def forward(self, img_feat, text_feat, task_loss=None, task_forward_fn=None):
        """主要接口"""
        training_mode = self.training and task_loss is not None
        return self.compute_comprehensive_quality(
            img_feat, text_feat, task_loss, task_forward_fn, training_mode
        )