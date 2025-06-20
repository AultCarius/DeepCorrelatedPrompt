import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import random

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from clip.modules.dist_utils import all_gather


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance


def compute_mlm(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    mlm_loss = 0.1*F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mlm_loss")(ret["mlm_loss"])
    acc = getattr(pl_module, f"{phase}_mlm_accuracy")(
        ret["mlm_logits"], ret["mlm_labels"]
    )
    pl_module.log(f"mlm/{phase}/loss", loss)
    pl_module.log(f"mlm/{phase}/accuracy", acc)

    return ret


def compute_mpp(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpp_logits = pl_module.mpp_score(infer["image_feats"])
    mpp_logits = torch.stack(
        [
            mpp_logits[:, :, 0:256],
            mpp_logits[:, :, 256:512],
            mpp_logits[:, :, 512:768],
        ],
        dim=2,
    )
    mpp_labels = infer["image_labels"]

    mpp_loss = 0.1*F.cross_entropy(
        mpp_logits.view(-1, 256),
        mpp_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mpp_loss": mpp_loss,
        "mpp_logits": mpp_logits,
        "mpp_labels": mpp_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpp_loss")(ret["mpp_loss"])
    acc = getattr(pl_module, f"{phase}_mpp_accuracy")(
        ret["mpp_logits"], ret["mpp_labels"]
    )
    pl_module.log(f"mpp/{phase}/loss", loss)
    pl_module.log(f"mpp/{phase}/accuracy", acc)

    return ret


def compute_mppd(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mppd_logits = pl_module.mppd_score(infer["image_feats"])
    mppd_labels = infer["image_labels_mppd"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mppd_labels[filter_to_train]
    logits = mppd_logits[filter_to_train]
    mppd_loss = 0.1*F.mse_loss(logits, labels)

    ret = {
        "mppd_loss": mppd_loss,
        "mppd_logits": mppd_logits,
        "mppd_labels": mppd_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mppd_loss")(ret["mppd_loss"])
    pl_module.log(f"mppd/{phase}/loss", loss)

    return ret


def compute_mpfr(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=True)
    mpfr_logits = pl_module.mpfr_score(infer["image_feats"])
    mpfr_labels = infer["image_labels_mpfr"]
    filter_to_train = infer["image_labels"].float().mean(dim=-1) != -100

    labels = mpfr_labels[filter_to_train]
    logits = mpfr_logits[filter_to_train]
    mpfr_loss = F.mse_loss(logits, labels)

    ret = {
        "mpfr_loss": mpfr_loss,
        "mpfr_logits": mpfr_logits,
        "mpfr_labels": mpfr_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_mpfr_loss")(ret["mpfr_loss"])
    pl_module.log(f"mpfr/{phase}/loss", loss)

    return ret


def compute_itm_wpa(pl_module, batch):
    pos_len = len(batch["text"]) // 2
    neg_len = len(batch["text"]) - pos_len
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )
    itm_labels = itm_labels[torch.randperm(itm_labels.size(0))]

    itm_images = [
        torch.stack(
            [
                ti if itm_labels[i] == 1 else fi
                for i, (ti, fi) in enumerate(zip(bti, bfi))
            ]
        )
        for bti, bfi in zip(batch["image"], batch["false_image_0"])
    ]

    batch = {k: v for k, v in batch.items()}
    batch["image"] = itm_images

    infer = pl_module.infer(batch, mask_text=False, mask_image=False)

    with torch.cuda.amp.autocast(enabled=False):
        txt_emb, img_emb = infer["text_feats"], infer["image_feats"]
        txt_mask, img_mask = infer["text_masks"].bool(), infer["image_masks"].bool()
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        if "deit" in pl_module.hparams.config["vit"]:
            img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        )
        distance = trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))

    itm_logits = pl_module.itm_score(infer["cls_feats"])
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_wpa_loss": 0.1 * ot_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_itm_loss")(ret["itm_loss"])
    wpa_loss = getattr(pl_module, f"{phase}_itm_wpa_loss")(ret["itm_wpa_loss"])
    acc = getattr(pl_module, f"{phase}_itm_accuracy")(
        ret["itm_logits"], ret["itm_labels"]
    )
    pl_module.log(f"itm/{phase}/loss", loss)
    pl_module.log(f"itm/{phase}/wpa_loss", wpa_loss)
    pl_module.log(f"itm/{phase}/accuracy", acc)

    return ret

def compute_mmimdb(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    imgcls_logits = pl_module.mmimdb_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).float()
    imgcls_loss = F.binary_cross_entropy_with_logits(imgcls_logits, imgcls_labels)

    ret = {
        "mmimdb_loss": imgcls_loss,
        "mmimdb_logits": imgcls_logits,
        "mmimdb_labels": imgcls_labels,
    }

    loss = getattr(pl_module, f"{phase}_mmimdb_loss")(ret["mmimdb_loss"])
    
    f1_scores = getattr(pl_module, f"{phase}_mmimdb_F1_scores")(
        ret["mmimdb_logits"], ret["mmimdb_labels"]
    )
    pl_module.log(f"mmimdb/{phase}/loss", loss)

    return ret

def compute_hatememes(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    imgcls_logits = pl_module.hatememes_classifier(infer["cls_feats"])

    imgcls_labels = batch["label"]
#     imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).float().view(-1,1)
#     imgcls_loss = F.binary_cross_entropy_with_logits(imgcls_logits, imgcls_labels)
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "hatememes_loss": imgcls_loss,
        "hatememes_logits": imgcls_logits,
        "hatememes_labels": imgcls_labels,
    }

    loss = getattr(pl_module, f"{phase}_hatememes_loss")(ret["hatememes_loss"])
    acc = getattr(pl_module, f"{phase}_hatememes_accuracy")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )
    auroc = getattr(pl_module, f"{phase}_hatememes_AUROC")(
        ret["hatememes_logits"], ret["hatememes_labels"]
    )    
    pl_module.log(f"hatememes/{phase}/loss", loss)

    return ret


def compute_objective_quality_loss(quality_scores, current_task_performance,
                                   image_features=None, text_features=None,
                                   enhanced_image_features=None, enhanced_text_features=None,
                                   missing_type=None):
    """
    【修复版】质量感知损失函数
    """
    if not quality_scores or current_task_performance is None:
        return torch.tensor(0.0, requires_grad=True)

    device = current_task_performance.device
    batch_size = len(quality_scores)

    try:
        # 验证输入
        for i, q in enumerate(quality_scores):
            if not isinstance(q, dict):
                continue

            for modality in ['image_quality', 'text_quality']:
                if modality in q:
                    for metric_name, metric_value in q[modality].items():
                        if isinstance(metric_value, torch.Tensor) and not torch.isfinite(metric_value):
                            print(f"[NaN Detect] {modality}.{metric_name} = {metric_value} at batch {i}")
                            q[modality][metric_name] = torch.tensor(0.5, device=device)

        # 1. 质量预测与任务性能一致性损失
        predicted_quality = []

        for i, quality_score in enumerate(quality_scores):
            if isinstance(quality_score, dict):
                img_task_contrib = quality_score.get('image_quality', {}).get('task_contribution', 0.5)
                text_task_contrib = quality_score.get('text_quality', {}).get('task_contribution', 0.5)

                # 转换为标量
                if isinstance(img_task_contrib, torch.Tensor):
                    img_task_contrib = img_task_contrib.item()
                if isinstance(text_task_contrib, torch.Tensor):
                    text_task_contrib = text_task_contrib.item()

                # 根据缺失类型计算整体质量
                miss_type = missing_type[i] if missing_type and i < len(missing_type) else 0
                if miss_type == 1:  # 缺失文本
                    overall_quality = img_task_contrib
                elif miss_type == 2:  # 缺失图像
                    overall_quality = text_task_contrib
                else:  # 完整样本
                    overall_quality = (img_task_contrib + text_task_contrib) / 2

                predicted_quality.append(overall_quality)
            else:
                predicted_quality.append(0.5)

        predicted_quality = torch.tensor(predicted_quality, device=device, requires_grad=True)

        # 确保维度匹配
        if predicted_quality.size(0) != current_task_performance.size(0):
            min_size = min(predicted_quality.size(0), current_task_performance.size(0))
            predicted_quality = predicted_quality[:min_size]
            current_task_performance = current_task_performance[:min_size]

        # 计算一致性损失
        consistency_loss = F.mse_loss(predicted_quality, current_task_performance)

        # 2. 质量稳定性损失 - 防止质量预测过于极端
        stability_loss = torch.mean(torch.abs(predicted_quality - 0.5)) * 0.1

        total_loss = consistency_loss + stability_loss

        return total_loss

    except Exception as e:
        print(f"Error in objective quality loss computation: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)



# def extract_task_performance_mmimdb(mmimdb_logits, mmimdb_labels):
#     """
#     从MMIMDb任务输出中提取性能指标
#     """
#     if mmimdb_logits is None:
#         return None
#
#     with torch.no_grad():
#         # 计算预测置信度作为性能指标
#         probs = torch.sigmoid(mmimdb_logits)
#
#         # 方法1：最高预测概率的平均值
#         max_probs = torch.max(probs, dim=-1)[0]
#
#         # 方法2：预测分布的熵（低熵=高确定性=高性能）
#         eps = 1e-8
#         entropy = -torch.sum(probs * torch.log(probs + eps) +
#                              (1 - probs) * torch.log(1 - probs + eps), dim=-1)
#         normalized_entropy = entropy / (23 * np.log(2))  # 23个类别
#         certainty = 1.0 - normalized_entropy
#
#         # 结合两个指标
#         performance_score = 0.6 * max_probs + 0.4 * certainty
#
#     if not torch.isfinite(performance_score).all():
#         print("[NaN Detect] performance_score contains NaN:", performance_score)
#
#     return performance_score.clamp(0.1, 0.9)
def extract_task_performance_mmimdb(logits, labels):
    """
    提取MMIMDb任务性能指标
    """
    with torch.no_grad():
        predictions = torch.sigmoid(logits) > 0.5

        # 计算F1分数
        tp = (predictions * labels).sum(dim=1).float()
        fp = (predictions * (1 - labels)).sum(dim=1).float()
        fn = ((1 - predictions) * labels).sum(dim=1).float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return f1

def compute_enhanced_mmimdb(pl_module, batch):
    """

    """
    phase = "train" if pl_module.training else "val"

    # 1. 标准的MMIMDb前向传播
    infer = pl_module.infer(batch)
    imgcls_logits = pl_module.mmimdb_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).float()

    # 2. 主要的分类损失
    mmimdb_loss = F.binary_cross_entropy_with_logits(imgcls_logits, imgcls_labels)

    # 3. 【修复】质量感知损失 - 只在训练时且有质量结果时计算
    quality_losses = {}

    if phase == "train":
        # 3.1 增强质量一致性损失
        enhanced_quality_loss = torch.tensor(0.0, device=pl_module.device, requires_grad=True)
        if (hasattr(pl_module.model, 'cached_quality_results') and
            pl_module.model.cached_quality_results):
            try:
                enhanced_quality_loss = compute_enhanced_quality_consistency_loss(
                    pl_module.model.cached_quality_results,
                    imgcls_logits,
                    imgcls_labels,
                    batch["missing_type"]
                )
                quality_losses['mmimdb_enhanced_quality_loss'] = enhanced_quality_loss

                if enhanced_quality_loss.item() > 0:
                    pl_module.log(f"mmimdb/{phase}/enhanced_quality_loss", enhanced_quality_loss)
            except Exception as e:
                print(f"Warning: Enhanced quality loss computation failed: {e}")
                enhanced_quality_loss = torch.tensor(0.0, device=pl_module.device, requires_grad=True)

        # 3.2 预测器训练损失
        predictor_training_loss = torch.tensor(0.0, device=pl_module.device, requires_grad=True)
        if (hasattr(pl_module.model, 'cached_quality_results') and
            pl_module.model.cached_quality_results):
            try:
                total_predictor_loss = 0.0
                count = 0
                for quality_result in pl_module.model.cached_quality_results:
                    predictor_loss = quality_result.get('predictor_loss', 0)
                    if isinstance(predictor_loss, torch.Tensor) and predictor_loss.requires_grad:
                        total_predictor_loss += predictor_loss
                        count += 1

                if count > 0:
                    predictor_training_loss = total_predictor_loss / count
                    quality_losses['mmimdb_predictor_loss'] = predictor_training_loss

                    if predictor_training_loss.item() > 0:
                        pl_module.log(f"mmimdb/{phase}/predictor_loss", predictor_training_loss)
            except Exception as e:
                print(f"Warning: Predictor loss computation failed: {e}")
                predictor_training_loss = torch.tensor(0.0, device=pl_module.device, requires_grad=True)

    # 4. 【修复】构建返回结果 - 确保所有损失都被正确处理
    ret = {
        "mmimdb_loss": mmimdb_loss,
        "mmimdb_logits": imgcls_logits,
        "mmimdb_labels": imgcls_labels,
    }

    # 添加质量损失到返回结果
    ret.update(quality_losses)

    # 5. 记录指标
    loss = getattr(pl_module, f"{phase}_mmimdb_loss")(ret["mmimdb_loss"])

    f1_scores = getattr(pl_module, f"{phase}_mmimdb_F1_scores")(
        ret["mmimdb_logits"], ret["mmimdb_labels"]
    )
    pl_module.log(f"mmimdb/{phase}/loss", loss)

    return ret


def compute_enhanced_mmimdb_v2(pl_module, batch):
    """
    增强版MMIMDb计算 - 集成新的质量感知损失
    基于EnhancedQualityEstimator的版本
    """
    phase = "train" if pl_module.training else "val"

    # 1. 标准的MMIMDb前向传播
    infer = pl_module.infer(batch)
    imgcls_logits = pl_module.mmimdb_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).float()

    # 2. 主要的分类损失
    mmimdb_loss = F.binary_cross_entropy_with_logits(imgcls_logits, imgcls_labels)

    # 3. 【新增】基于增强质量评估的损失
    enhanced_quality_loss = torch.tensor(0.0, device=pl_module.device, requires_grad=True)
    predictor_training_loss = torch.tensor(0.0, device=pl_module.device, requires_grad=True)

    if (hasattr(pl_module.model, 'cached_quality_results') and
            pl_module.model.cached_quality_results and
            phase == "train"):  # 只在训练时计算质量损失

        quality_results = pl_module.model.cached_quality_results

        # 3.1 预测器训练损失
        total_predictor_loss = 0.0
        for quality_result in quality_results:
            predictor_loss = quality_result.get('predictor_loss', 0)
            if isinstance(predictor_loss, torch.Tensor) and predictor_loss.requires_grad:
                total_predictor_loss += predictor_loss

        predictor_training_loss = total_predictor_loss / max(len(quality_results), 1)

        # 3.2 质量一致性损失 - 基于任务性能的质量验证
        enhanced_quality_loss = compute_enhanced_quality_consistency_loss(
            quality_results, imgcls_logits, imgcls_labels, batch["missing_type"]
        )

    # 4. 返回结果
    ret = {
        "mmimdb_loss": mmimdb_loss,
        "mmimdb_enhanced_quality_loss": enhanced_quality_loss,  # 新的质量损失
        "mmimdb_predictor_loss": predictor_training_loss,  # 预测器训练损失
        "mmimdb_logits": imgcls_logits,
        "mmimdb_labels": imgcls_labels,
    }

    # 5. 记录指标
    loss = getattr(pl_module, f"{phase}_mmimdb_loss")(ret["mmimdb_loss"])

    # 记录质量损失
    if enhanced_quality_loss.item() > 0:
        pl_module.log(f"mmimdb/{phase}/enhanced_quality_loss", enhanced_quality_loss)
    if predictor_training_loss.item() > 0:
        pl_module.log(f"mmimdb/{phase}/predictor_loss", predictor_training_loss)

    f1_scores = getattr(pl_module, f"{phase}_mmimdb_F1_scores")(
        ret["mmimdb_logits"], ret["mmimdb_labels"]
    )
    pl_module.log(f"mmimdb/{phase}/loss", loss)

    return ret


def compute_enhanced_quality_consistency_loss(quality_results, task_logits, task_labels, missing_type):
    """
    【修复版】计算增强的质量一致性损失
    """
    if not quality_results:
        return torch.tensor(0.0, requires_grad=True)

    device = task_logits.device
    batch_size = len(quality_results)

    # 防止batch_size不匹配
    if batch_size != task_logits.size(0):
        print(f"Warning: Quality results batch size ({batch_size}) != logits batch size ({task_logits.size(0)})")
        return torch.tensor(0.0, device=device, requires_grad=True)

    try:
        # 1. 计算任务性能（F1分数作为性能指标）
        with torch.no_grad():
            task_predictions = torch.sigmoid(task_logits) > 0.5
            task_performance = []

            for i in range(batch_size):
                # 计算每个样本的F1分数
                pred = task_predictions[i].float()
                label = task_labels[i].float()

                # 避免除零错误
                tp = (pred * label).sum()
                fp = (pred * (1 - label)).sum()
                fn = ((1 - pred) * label).sum()

                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)

                task_performance.append(f1.item())

            task_performance = torch.tensor(task_performance, device=device)

        # 2. 提取质量预测
        predicted_quality = []
        for i, quality_result in enumerate(quality_results):
            if 'image_quality' in quality_result and 'text_quality' in quality_result:
                img_qual = quality_result['image_quality'].get('task_contribution', 0.5)
                text_qual = quality_result['text_quality'].get('task_contribution', 0.5)

                # 转换为tensor
                if isinstance(img_qual, torch.Tensor):
                    img_qual = img_qual.item()
                if isinstance(text_qual, torch.Tensor):
                    text_qual = text_qual.item()

                # 根据缺失类型选择质量
                miss_type = missing_type[i] if i < len(missing_type) else 0
                if miss_type == 1:  # 缺失文本，使用图像质量
                    overall_quality = img_qual
                elif miss_type == 2:  # 缺失图像，使用文本质量
                    overall_quality = text_qual
                else:  # 完整样本，使用平均质量
                    overall_quality = (img_qual + text_qual) / 2

                predicted_quality.append(overall_quality)
            else:
                predicted_quality.append(0.5)  # 默认中等质量

        predicted_quality = torch.tensor(predicted_quality, device=device, requires_grad=True)

        # 3. 计算质量一致性损失
        quality_consistency_loss = F.mse_loss(predicted_quality, task_performance)

        # 4. 添加正则化项防止质量预测过于极端
        quality_regularization = torch.mean(torch.abs(predicted_quality - 0.5)) * 0.1

        total_loss = quality_consistency_loss + quality_regularization

        return total_loss

    except Exception as e:
        print(f"Error in quality consistency loss computation: {e}")
        return torch.tensor(0.0, device=device, requires_grad=True)


def compute_gradient_quality_alignment_loss(quality_results, model_features):
    """
    计算梯度与质量的对齐损失（可选的额外损失）
    """
    if not quality_results or not hasattr(model_features, 'cached_features'):
        return torch.tensor(0.0, requires_grad=True)

    alignment_loss = 0.0

    for i, quality_result in enumerate(quality_results):
        # 如果有梯度信息
        if 'task_relevance' in quality_result:
            task_relevance = quality_result['task_relevance']

            if 'img_gradient_magnitude' in task_relevance:
                # 梯度幅度与预测的任务相关性应该一致
                if 'img_task_relevance' in task_relevance:
                    img_gradient_pred_gap = torch.abs(
                        task_relevance['img_gradient_magnitude'] -
                        task_relevance['img_task_relevance']
                    )
                    text_gradient_pred_gap = torch.abs(
                        task_relevance['text_gradient_magnitude'] -
                        task_relevance['text_task_relevance']
                    )

                    alignment_loss += (img_gradient_pred_gap + text_gradient_pred_gap) / 2

    if len(quality_results) > 0:
        alignment_loss = alignment_loss / len(quality_results)

    return alignment_loss




# 同样的方式可以为其他任务实现
def compute_enhanced_hatememes(pl_module, batch):
    """类似的增强版HateMemes计算"""
    # ... 类似实现
    pass


def compute_enhanced_food101(pl_module, batch):
    """类似的增强版Food101计算"""
    # ... 类似实现
    pass


def compute_food101(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    if phase == "train":
        infer = pl_module.infer(batch)
    else:
        infer = pl_module.infer(batch)

    imgcls_logits = pl_module.food101_classifier(infer["cls_feats"])

    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "food101_loss": imgcls_loss,
        "food101_logits": imgcls_logits,
        "food101_labels": imgcls_labels,
    }

    loss = getattr(pl_module, f"{phase}_food101_loss")(ret["food101_loss"])
    acc = getattr(pl_module, f"{phase}_food101_accuracy")(
        ret["food101_logits"], ret["food101_labels"]
    )
    pl_module.log(f"food101/{phase}/loss", loss)

    return ret


def compute_imgcls(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    imgcls_logits = pl_module.img_classifier(infer["cls_feats"])
    imgcls_labels = batch["label"]
    imgcls_labels = torch.tensor(imgcls_labels).to(pl_module.device).long()
    imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

    ret = {
        "imgcls_loss": imgcls_loss,
        "imgcls_logits": imgcls_logits,
        "imgcls_labels": imgcls_labels,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_imgcls_loss")(ret["imgcls_loss"])
    acc = getattr(pl_module, f"{phase}_imgcls_accuracy")(
        ret["imgcls_logits"], ret["imgcls_labels"]
    )
    pl_module.log(f"imgcls/{phase}/loss", loss)
    pl_module.log(f"imgcls/{phase}/accuracy", acc)

    return ret

def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    loss = getattr(pl_module, f"{phase}_vqa_loss")(ret["vqa_loss"])
    score = getattr(pl_module, f"{phase}_vqa_score")(
        ret["vqa_logits"], ret["vqa_targets"]
    )
    pl_module.log(f"vqa/{phase}/loss", loss)
    pl_module.log(f"vqa/{phase}/score", score)

    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        loss = getattr(pl_module, f"{phase}_nlvr2_loss")(ret["nlvr2_loss"])
        acc = getattr(pl_module, f"{phase}_nlvr2_accuracy")(
            ret["nlvr2_logits"], ret["nlvr2_labels"]
        )
        pl_module.log(f"nlvr2/{phase}/loss", loss)
        pl_module.log(f"nlvr2/{phase}/accuracy", acc)
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        if dev_batches:
            dev_loss = getattr(pl_module, f"dev_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
                )
            )
            dev_acc = getattr(pl_module, f"dev_nlvr2_accuracy")(
                ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches]
            )
            pl_module.log(f"nlvr2/dev/loss", dev_loss)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc)
        if test_batches:
            test_loss = getattr(pl_module, f"test_nlvr2_loss")(
                F.cross_entropy(
                    ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
                )
            )
            test_acc = getattr(pl_module, f"test_nlvr2_accuracy")(
                ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches]
            )
            pl_module.log(f"nlvr2/test/loss", test_loss)
            pl_module.log(f"nlvr2/test/accuracy", test_acc)

    return ret


def compute_irtr(pl_module, batch):
    is_training_phase = pl_module.training

    _bs, _c, _h, _w = batch["image"][0].shape
    false_len = pl_module.hparams.config["draw_false_text"]
    text_ids = torch.stack(
        [batch[f"false_text_{i}_ids"] for i in range(false_len)], dim=1
    )
    text_masks = torch.stack(
        [batch[f"false_text_{i}_masks"] for i in range(false_len)], dim=1
    )
    text_labels = torch.stack(
        [batch[f"false_text_{i}_labels"] for i in range(false_len)], dim=1
    )

    text_ids = torch.cat([batch["text_ids"].unsqueeze(1), text_ids], dim=1)
    text_masks = torch.cat([batch["text_masks"].unsqueeze(1), text_masks], dim=1)
    text_labels = torch.cat([batch["text_labels"].unsqueeze(1), text_labels], dim=1)
    images = batch["image"][0].unsqueeze(1).expand(_bs, false_len + 1, _c, _h, _w)

    infer = pl_module.infer(
        {
            "image": [rearrange(images, "bs fs c h w -> (bs fs) c h w")],
            "text_ids": rearrange(text_ids, "bs fs tl -> (bs fs) tl"),
            "text_masks": rearrange(text_masks, "bs fs tl -> (bs fs) tl"),
            "text_labels": rearrange(text_labels, "bs fs tl -> (bs fs) tl"),
        }
    )
    score = pl_module.rank_output(infer["cls_feats"])[:, 0]
    score = rearrange(score, "(bs fs) -> bs fs", bs=_bs, fs=false_len + 1)
    answer = torch.zeros(_bs).to(score).long()
    irtr_loss = F.cross_entropy(score, answer)

    ret = {
        "irtr_loss": irtr_loss,
    }

    phase = "train" if pl_module.training else "val"
    irtr_loss = getattr(pl_module, f"{phase}_irtr_loss")(ret["irtr_loss"])

    pl_module.log(f"irtr/{phase}/irtr_loss", irtr_loss)

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module):
    text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=64,
        num_workers=pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
        image_only=True
    )
    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    dist_sampler = DistributedSampler(image_dset, shuffle=False)
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=1,
        num_workers=pl_module.hparams.config["num_workers"],
        sampler=dist_sampler,
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        (ie, im, _, _) = pl_module.transformer.visual_embed(
            _b["image"][0].to(pl_module.device),
            max_image_len=pl_module.hparams.config["max_image_len"],
            mask_it=False,
        )
        image_preload.append((ie, im, _b["img_index"][0]))

    rank_scores = list()
    rank_iids = list()

    for img_batch in tqdm.tqdm(image_preload, desc="rank loop"):
        _ie, _im, _iid = img_batch
        _, l, c = _ie.shape

        img_batch_score = list()
        for txt_batch in text_preload:
            fblen = len(txt_batch["text_ids"])
            ie = _ie.expand(fblen, l, c)
            im = _im.expand(fblen, l)

            with torch.cuda.amp.autocast():
                score = pl_module.rank_output(
                    pl_module.infer(
                        {
                            "text_ids": txt_batch["text_ids"],
                            "text_masks": txt_batch["text_masks"],
                            "text_labels": txt_batch["text_labels"],
                        },
                        image_embeds=ie,
                        image_masks=im,
                    )["cls_feats"]
                )[:, 0]

            img_batch_score.append(score)

        img_batch_score = torch.cat(img_batch_score)
        rank_scores.append(img_batch_score.cpu().tolist())
        rank_iids.append(_iid)

    torch.distributed.barrier()
    gather_rank_scores = all_gather(rank_scores)
    gather_rank_iids = all_gather(rank_iids)

    iids = torch.tensor(gather_rank_iids)
    iids = iids.view(-1)
    scores = torch.tensor(gather_rank_scores)
    scores = scores.view(len(iids), -1)

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def arc_test_step(pl_module, batch, output):
    return output


def vqa_test_wrapup(outs, model_name):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result", exist_ok=True)
        with open(f"result/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")


def arc_test_wrapup(outs, caplen, model_name):
    rank = torch.distributed.get_rank()
    iids, captions = list(), list()
    for out in outs:
        iids += out["iid"]
        captions += out["captions"]

    rets = list()
    for iid, caption in zip(iids, captions):
        rets.append({"image_id": iid, "caption": caption})
    with open(f"coco_cap_len{caplen}_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob(f"coco_cap_len{caplen}_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        os.makedirs("result/arc", exist_ok=True)
        jsons = sorted(jsons, key=lambda x: x["image_id"])
        with open(f"result/arc/coco_cap_{model_name}_len{caplen}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"coco_cap_len{caplen}_{rank}.json")
