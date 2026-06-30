"""
BiomedCLIP 微调训练脚本
=========================
支持 Linear Probe 和 Full Fine-tune 两种策略,
自动处理类别不平衡、混合精度训练、日志保存为文本文件。

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
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from tqdm import tqdm

from data_utils import create_dataloaders, create_kfold_dataloaders

warnings.filterwarnings("ignore")


# ============================================================================
# Focal Loss: 专门处理极端类别不平衡
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    
    公式: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    相比 CrossEntropyLoss + class_weights 的优势:
      - 自动降低已分类正确样本的 loss, 聚焦难分类样本
      - 不会因为少数类权重过大而忽略多数类
      - gamma 越大, 对易分类样本的压制越强
    
    Args:
        alpha: 各类别权重 (可选), shape (num_classes,)
        gamma: 聚焦参数, 默认 2.0. 越大越关注难样本
        reduction: "mean" | "sum"
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha          # (num_classes,) tensor or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) long labels
        Returns:
            scalar loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # (N,)
        pt = torch.exp(-ce_loss)                                       # p_t
        focal_weight = (1 - pt) ** self.gamma                         # (1-p_t)^gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]            # 每个样本对应类别的 alpha
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# 文本日志工具
# ============================================================================

class TrainingLogger:
    """将训练指标写入文本日志文件, 同时打印到终端."""
    def __init__(self, log_file: str):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
        self.fh = open(log_file, "w", encoding="utf-8")

    def log(self, msg: str):
        """写入日志并打印到终端."""
        print(msg)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float, val_f1: float, lr: float = None):
        """记录一个 epoch 的训练/验证指标."""
        lr_str = f" LR={lr:.2e}" if lr is not None else ""
        line = (f"Epoch {epoch:3d} | "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} "
                f"F1={val_f1:.4f}{lr_str}")
        self.log(line)

    def log_section(self, title: str):
        """记录分隔标题."""
        self.log(f"\n{'='*60}")
        self.log(f"  {title}")
        self.log(f"{'='*60}")

    def log_info(self, msg: str):
        self.log(f"  {msg}")

    def close(self):
        self.fh.close()
        print(f"✅ 日志已保存至: {self.log_file}")


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
    use_focal = getattr(cfg, "use_focal_loss", False)

    if use_focal:
        # Focal Loss: 适合极端类别不平衡
        focal_gamma = getattr(cfg, "focal_gamma", 2.0)
        focal_alpha = getattr(cfg, "focal_alpha", None)
        alpha_tensor = None
        if focal_alpha is not None:
            # 从训练集统计类别分布, 计算温和的 alpha (逆频率的 sqrt)
            from collections import Counter
            if cfg.use_kfold:
                # K-fold 下从 train_loader 获取
                label_counter = Counter()
                for _, labels in train_loader:
                    label_counter.update(labels.tolist())
            else:
                train_labels = [s[1] for s in train_loader.dataset.samples]
                label_counter = Counter(train_labels)
            total = sum(label_counter.values())
            n = cfg.num_classes
            raw_weights = torch.tensor(
                [total / (n * label_counter[i]) for i in range(n)],
                dtype=torch.float32,
            )
            # sqrt 温和缩放, 避免极端权重
            alpha_tensor = raw_weights ** focal_alpha
            alpha_tensor = alpha_tensor / alpha_tensor.sum() * n  # 归一化到均值为1
            alpha_tensor = alpha_tensor.to(device)
            print(f"Focal alpha (sqrt-scaled): {[f'{v:.3f}' for v in alpha_tensor.tolist()]}")
        criterion = FocalLoss(alpha=alpha_tensor, gamma=focal_gamma, reduction="mean")
        print(f"损失函数: Focal Loss (gamma={focal_gamma}, alpha_pow={focal_alpha})")
    else:
        if cfg.use_class_weights and class_weights is not None:
            class_weights = class_weights.to(device)
            print(f"类别权重: {class_weights.tolist()}")
        else:
            class_weights = None
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=cfg.label_smoothing,
        )
        print(f"损失函数: CrossEntropyLoss")

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
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = os.path.join(cfg.log_dir, f"{run_name}.log")
    logger = TrainingLogger(log_file)

    # 记录训练配置
    logger.log_section("Training Config")
    logger.log_info(f"Model:      {cfg.model_name}")
    logger.log_info(f"Strategy:   {cfg.strategy}")
    logger.log_info(f"Num Classes:{cfg.num_classes}")
    logger.log_info(f"Batch Size: {cfg.batch_size}")
    logger.log_info(f"Epochs:     {cfg.epochs}")
    logger.log_info(f"LR:         {cfg.lr}")
    logger.log_info(f"Seed:       {cfg.seed}")
    logger.log_info(f"Log File:   {log_file}")

    # ---- 训练循环 ----
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0

    logger.log_section("Training Progress")
    logger.log(f"{'Epoch':>6s} | {'Train Loss':>10s} {'Train Acc':>10s} | "
               f"{'Val Loss':>10s} {'Val Acc':>10s} {'Val F1':>10s} {'LR':>10s}")

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch,
        )

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        val_metrics, val_cm, _, _, _ = evaluate(
            model, val_loader, criterion, device, cfg, class_names,
        )

        # 用 F1 作为早停指标
        current_metric = val_metrics["f1_macro"]

        # 记录到日志文件
        logger.log_epoch(epoch, train_loss, train_acc,
                         val_metrics["loss"], val_metrics["accuracy"],
                         val_metrics["f1_macro"], current_lr)

        # ---- 保存最佳模型 ----
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = os.path.join(cfg.output_dir, f"best_model{fold_tag}.pth")
            torch.save(model.state_dict(), ckpt_path)
            logger.log(f"  ✓ 保存最佳模型 (F1={best_metric:.4f}) -> {ckpt_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= cfg.early_stopping_patience:
            logger.log(f"\n  Early stopping at epoch {epoch} (patience={cfg.early_stopping_patience})")
            break

        # 定期保存
        if epoch % cfg.save_every_epoch == 0:
            ckpt = os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch}{fold_tag}.pth")
            torch.save(model.state_dict(), ckpt)
            logger.log(f"  ✓ 定期保存 checkpoint -> {ckpt}")

    # ---- 加载最佳模型, 在测试集上评估 ----
    logger.log_section(f"Best Model: Epoch {best_epoch}, F1={best_metric:.4f}")

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

    logger.close()

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
