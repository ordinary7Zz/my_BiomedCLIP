"""
BiomedCLIP 评估脚本
===================
对带标签的测试集进行分类性能评估，输出 accuracy、precision、recall、f1、
混淆矩阵、AUC 等指标，支持二分类和多分类。

用法:
    # 二分类评估
    python evaluate.py --ckpt output/BM/best_model.pth --data_dir dataset/test

    # 多分类评估
    python evaluate.py \
        --config config_multiclass \
        --ckpt output/TIRADS/best_model.pth \
        --data_dir /path/to/test \
        --num_classes 5 \
        --class_names 1 2 3 4 5
"""

import os
import argparse
import importlib
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from tqdm import tqdm

from data_utils import get_transforms
from model import BiomedCLIPClassifier

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="BiomedCLIP 分类评估")
    parser.add_argument("--config", type=str, default="config",
                        help="配置文件模块名 (不含 .py), 如 config 或 config_multiclass")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="模型权重文件路径 (.pth)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="带标签数据目录 (ImageFolder 格式: data_dir/class_A/img.png ...)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="类别数 (默认从配置文件读取)")
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="类别名列表, 如: --class_names benign malignant")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批大小 (默认从配置文件读取)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备: cuda 或 cpu")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, num_classes, class_names):
    """
    评估模型性能。

    返回:
        metrics: dict  各指标
        cm:      ndarray 混淆矩阵
        y_pred:  ndarray 预测标签
        y_true:  ndarray 真实标签
        y_prob:  ndarray 预测概率 (N, num_classes)
    """
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = logits.softmax(dim=1)
        preds = logits.argmax(dim=1)

        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---- 通用指标 ----
    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
    }

    # ---- 二分类额外指标 ----
    if num_classes == 2:
        metrics["precision_pos"] = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        metrics["recall_pos"] = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        metrics["f1_pos"] = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        try:
            metrics["auc"] = roc_auc_score(all_labels, all_probs[:, 1])
        except ValueError:
            metrics["auc"] = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    return metrics, cm, all_preds, all_labels, all_probs


def print_results(metrics, cm, class_names, num_samples):
    """格式化打印评估结果"""
    print(f"\n{'='*60}")
    print(f"  评估结果 (共 {num_samples} 张图片)")
    print(f"{'='*60}")

    # ---- 主要指标 ----
    print(f"\n  📊 主要指标:")
    print(f"  {'-'*42}")
    key_metrics = [
        ("accuracy",            "准确率"),
        ("precision_macro",     "精确率 (macro)"),
        ("recall_macro",        "召回率 (macro)"),
        ("f1_macro",            "F1 (macro)"),
        ("precision_weighted",  "精确率 (weighted)"),
        ("recall_weighted",     "召回率 (weighted)"),
        ("f1_weighted",         "F1 (weighted)"),
    ]
    for k, label in key_metrics:
        if k in metrics:
            print(f"  {label:<20s}: {metrics[k]:.4f}")

    if "auc" in metrics:
        print(f"  {'AUC':<20s}: {metrics['auc']:.4f}")

    if "loss" in metrics:
        print(f"  {'Loss':<20s}: {metrics['loss']:.4f}")

    # ---- 混淆矩阵 ----
    n = len(cm)
    display_names = class_names if class_names else [str(i) for i in range(n)]
    print(f"\n  📋 混淆矩阵 (行=真实标签, 列=预测标签):")
    header = f"  {'':>12s}" + "".join(f"{name:>8s}" for name in display_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {display_names[i]:>10s}  " + "".join(f"{v:>8d}" for v in row))

    # ---- 每类准确率 ----
    print(f"\n  📈 每类准确率:")
    for i in range(n):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        name = display_names[i]
        print(f"  {name:<10s}: {class_correct}/{class_total} = {class_correct / class_total:.4f}")

    # ---- 二分类详细 ----
    if n == 2 and "precision_pos" in metrics:
        pos_name = display_names[1]
        print(f"\n  🔬 正类 ({pos_name}) 详细指标:")
        print(f"  {'精确率':<20s}: {metrics['precision_pos']:.4f}")
        print(f"  {'召回率':<20s}: {metrics['recall_pos']:.4f}")
        print(f"  {'F1':<20s}: {metrics['f1_pos']:.4f}")


def main():
    args = parse_args()

    # 加载配置文件
    config_module = importlib.import_module(args.config)
    cfg = config_module.Config()

    # HuggingFace 镜像
    if cfg.hf_endpoint:
        os.environ["HF_ENDPOINT"] = cfg.hf_endpoint

    # 参数优先级: 命令行 > 配置文件
    num_classes = args.num_classes if args.num_classes is not None else cfg.num_classes
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  BiomedCLIP 模型评估")
    print(f"{'='*60}")
    print(f"  权重文件:  {args.ckpt}")
    print(f"  数据目录:  {args.data_dir}")
    print(f"  类别数:    {num_classes}")
    print(f"  设备:      {device}")
    print(f"  批大小:    {batch_size}")
    print(f"{'='*60}")

    # ---- 检查数据目录 ----
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")

    # ---- 加载有标签数据 ----
    eval_transform = get_transforms(cfg, is_train=False)
    dataset = datasets.ImageFolder(args.data_dir, transform=eval_transform)
    class_names = dataset.classes

    # 类别名覆写
    if args.class_names:
        if len(args.class_names) != num_classes:
            print(f"\n  ⚠ class_names 长度 ({len(args.class_names)}) 与 num_classes ({num_classes})"
                  f" 不一致, 使用数据集自动检测的类别名")
        else:
            class_names = args.class_names

    print(f"\n  数据统计:")
    print(f"  总样本数:  {len(dataset)}")
    for i, name in enumerate(class_names):
        count = sum(1 for _, label in dataset.samples if label == i)
        print(f"    {name}: {count}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # ---- 加载模型 ----
    print(f"\n  加载模型...")
    model = BiomedCLIPClassifier(
        model_name=cfg.model_name,
        num_classes=num_classes,
        strategy="full_finetune",  # 策略不影响推理
    )
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # ---- 评估 ----
    criterion = nn.CrossEntropyLoss()
    metrics, cm, y_pred, y_true, y_prob = evaluate_model(
        model, loader, criterion, device, num_classes, class_names,
    )

    # ---- 输出结果 ----
    print_results(metrics, cm, class_names, len(dataset))

    # 多分类: sklearn 分类报告
    if num_classes > 2:
        print(f"\n  📝 分类报告 (per-class):")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print(f"\n{'='*60}")
    print(f"  评估完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
