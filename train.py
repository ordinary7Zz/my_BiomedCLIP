"""
BiomedCLIP 微调训练脚本
=========================
支持 Linear Probe 和 Full Fine-tune 两种策略,
自动处理类别不平衡、混合精度训练、TensorBoard 日志。

用法:
    # 编辑 config.py 后直接运行:
    python train.py

    # 或在命令行覆盖参数:
    python train.py --strategy full_finetune --lr 1e-5 --epochs 30 --batch_size 16
"""

import os
import sys
import random
import argparse
import importlib
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_utils import create_dataloaders, create_kfold_dataloaders

warnings.filterwarnings("ignore")


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="微调 BiomedCLIP 做医学图像分类")
    parser.add_argument("--config", type=str, default="config",
                        help="配置文件模块名 (不含 .py), 如 config 或 config_multiclass")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["linear_probe", "full_finetune"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--use_kfold", action="store_true", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练模型权重路径 (如用二分类权重初始化多分类 ViT)")
    return parser.parse_args()


# ============================================================================
# 训练一个 Epoch
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=cfg.use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_value = loss.item() * cfg.gradient_accumulation_steps
        total_loss += loss_value
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({"loss": f"{loss_value:.4f}", "acc": f"{acc:.4f}"})

    avg_loss = total_loss / len(loader)
    train_acc = accuracy_score(all_labels, all_preds)
    return avg_loss, train_acc


# ============================================================================
# 验证/测试
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, cfg, class_names=None):
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="Eval"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=cfg.use_amp):
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

    num_classes = cfg.num_classes

    # ---- 多分类指标 ----
    metrics = {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
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

    # ---- 混淆矩阵 ----
    cm = confusion_matrix(all_labels, all_preds)

    return metrics, cm, all_preds, all_labels, all_probs


def print_metrics(metrics, cm, class_names, phase="Val"):
    print(f"\n{'='*55}")
    print(f"  {phase} Results")
    print(f"{'='*55}")
    for k, v in metrics.items():
        print(f"  {k:<18s}: {v:.4f}")
    print(f"\n  Confusion Matrix:")
    header = "         " + "".join(f"{n:>8s}" for n in (class_names or [str(i) for i in range(len(cm))]))
    print(header)
    for i, row in enumerate(cm):
        name = (class_names[i] if class_names else str(i))
        print(f"  {name:<7s} {str(row):>20s}")


# ============================================================================
# 主训练函数
# ============================================================================

def run_training(cfg, fold_idx: int = -1, pretrained: str = None):
    # 延迟导入: 确保 HF_ENDPOINT 已在 main() 中设置
    from model import create_classifier

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ---- 数据 ----
    if cfg.use_kfold:
        folds, class_names, class_weights = create_kfold_dataloaders(cfg)
        train_loader, val_loader, _, _ = folds[fold_idx]
    else:
        train_loader, val_loader, test_loader, class_names, class_weights = create_dataloaders(cfg)

    # 自动检测实际类别数，与配置对比
    actual_num_classes = len(class_names)
    if actual_num_classes != cfg.num_classes:
        print(f"\n  ⚠ 警告: 配置 num_classes={cfg.num_classes}，但数据集有 {actual_num_classes} 个类别")
        print(f"      自动使用数据集实际类别数: {actual_num_classes}")
        cfg.num_classes = actual_num_classes

    # 覆写 class_names (如果 config 中指定了)
    if cfg.class_names is not None:
        class_names = cfg.class_names

    # ---- 模型 ----
    model = create_classifier(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        strategy=cfg.strategy,
        dropout=cfg.dropout,
        pretrained=pretrained,
    ).to(device)

    print(f"\n模型: {cfg.model_name}")
    print(f"策略: {cfg.strategy}")
    print(f"可训练参数: {model.trainable_param_count:,} / {model.total_param_count:,}")
    print(f"类别: {class_names}")

    # ---- 损失函数 ----
    if cfg.use_class_weights and class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"类别权重: {class_weights.tolist()}")
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=cfg.label_smoothing,
    )

    # ---- 优化器 & 调度器 ----
    if cfg.strategy == "linear_probe":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
    else:
        # Full fine-tune: 图像编码器用更小的 LR
        visual_params = list(model.visual.parameters())
        head_params = list(model.classifier.parameters())
        optimizer = torch.optim.AdamW([
            {"params": visual_params, "lr": cfg.lr * 0.1},
            {"params": head_params, "lr": cfg.lr},
        ], weight_decay=cfg.weight_decay)

    # Cosine 退火
    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    scaler = GradScaler(enabled=cfg.use_amp)

    # ---- Logger ----
    fold_tag = f"_fold{fold_idx}" if fold_idx >= 0 else ""
    run_name = f"{cfg.strategy}{fold_tag}_{datetime.now().strftime('%m%d_%H%M')}"
    log_dir = os.path.join(cfg.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # ---- 训练循环 ----
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch,
        )

        if scheduler is not None:
            scheduler.step()

        val_metrics, val_cm, _, _, _ = evaluate(
            model, val_loader, criterion, device, cfg, class_names,
        )

        # 用 F1 作为早停指标
        current_metric = val_metrics["f1_macro"]

        print(f"Epoch {epoch:3d} | Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.4f} "
              f"F1={val_metrics['f1_macro']:.4f}")

        # TensorBoard
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Acc", train_acc, epoch)
        writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
        writer.add_scalar("Val/Acc", val_metrics["accuracy"], epoch)
        writer.add_scalar("Val/F1", val_metrics["f1_macro"], epoch)

        # ---- 保存最佳模型 ----
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(cfg.output_dir, f"best_model{fold_tag}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ 保存最佳模型 (F1={best_metric:.4f}) -> {ckpt_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= cfg.early_stopping_patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={cfg.early_stopping_patience})")
            break

        # 定期保存
        if epoch % cfg.save_every_epoch == 0:
            ckpt = os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch}{fold_tag}.pth")
            torch.save(model.state_dict(), ckpt)

    writer.close()

    # ---- 加载最佳模型, 在测试集上评估 ----
    print(f"\n{'='*55}")
    print(f"  加载最佳模型 (Epoch {best_epoch}, F1={best_metric:.4f})")
    print(f"{'='*55}")

    ckpt = torch.load(os.path.join(cfg.output_dir, f"best_model{fold_tag}.pth"), map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)

    if not cfg.use_kfold:
        test_metrics, test_cm, y_pred, y_true, y_prob = evaluate(
            model, test_loader, criterion, device, cfg, class_names,
        )
        print_metrics(test_metrics, test_cm, class_names, phase="Test")

        if cfg.num_classes > 2:
            print("\n  分类报告 (per-class):")
            print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        val_metrics, val_cm, _, _, _ = evaluate(
            model, val_loader, criterion, device, cfg, class_names,
        )
        print_metrics(val_metrics, val_cm, class_names, phase="Val (best model)")

    return best_metric


# ============================================================================
# 入口
# ============================================================================

def main():
    args = parse_args()
    # 动态加载指定的配置文件
    config_module = importlib.import_module(args.config)
    cfg = config_module.Config()

    # 命令行覆盖配置
    if args.data_dir: cfg.data_dir = args.data_dir
    if args.num_classes: cfg.num_classes = args.num_classes
    if args.strategy: cfg.strategy = args.strategy
    if args.batch_size: cfg.batch_size = args.batch_size

    # 设置 HuggingFace 镜像 (国内加速, 需在加载模型前设置)
    if cfg.hf_endpoint:
        os.environ["HF_ENDPOINT"] = cfg.hf_endpoint
        print(f"  HF Mirror: {cfg.hf_endpoint}")
    if args.epochs: cfg.epochs = args.epochs
    if args.lr: cfg.lr = args.lr
    if args.dropout is not None: cfg.dropout = args.dropout
    if args.seed is not None: cfg.seed = args.seed
    if args.use_kfold: cfg.use_kfold = args.use_kfold

    set_seed(cfg.seed)

    print("\n" + "=" * 55)
    print("  BiomedCLIP Fine-tuning for Medical Image Classification")
    print("=" * 55)
    print(f"  Model:       {cfg.model_name}")
    print(f"  Strategy:    {cfg.strategy}")
    print(f"  Num Classes: {cfg.num_classes}")
    print(f"  Batch Size:  {cfg.batch_size}")
    print(f"  Epochs:      {cfg.epochs}")
    print(f"  LR:          {cfg.lr}")
    print(f"  Data Dir:    {cfg.data_dir}")
    print(f"  K-Fold:      {cfg.use_kfold}")
    print("=" * 55)

    if cfg.use_kfold:
        scores = []
        for fold in range(cfg.kfold_n_splits):
            print(f"\n{'#'*55}")
            print(f"  Fold {fold + 1}/{cfg.kfold_n_splits}")
            print(f"{'#'*55}")
            score = run_training(cfg, fold_idx=fold, pretrained=args.pretrained)
            scores.append(score)
        print(f"\n{'='*55}")
        print(f"  K-Fold Average F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"{'='*55}")
    else:
        run_training(cfg, pretrained=args.pretrained)


if __name__ == "__main__":
    main()
