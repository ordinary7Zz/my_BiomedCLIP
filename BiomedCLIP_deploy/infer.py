"""
BiomedCLIP 分类推理脚本 (独立版)
==================================
对图片文件夹进行批量推理，输出分类结果 CSV。
若提供标签 JSON 文件，额外计算分类性能指标并保存到 txt 文件。

用法:
    # 只推理，输出 CSV
    python infer.py \\
        --ckpt /path/to/best_model.pth \\
        --folder /path/to/images/ \\
        --num_classes 2 \\
        --class_names benign malignant \\
        --output results.csv

    # 推理 + 评估（二分类）
    python infer.py \\
        --ckpt /path/to/best_model.pth \\
        --folder /path/to/images/ \\
        --num_classes 2 \\
        --class_names benign malignant \\
        --label_json /path/to/labels.json \\
        --label_field malignancy \\
        --output results.csv \\
        --eval_output eval_result.txt

    # 推理 + 评估（TIRADS 五分类）
    python infer.py \\
        --ckpt /path/to/best_model.pth \\
        --folder /path/to/images/ \\
        --num_classes 5 \\
        --class_names 1 2 3 4 5 \\
        --label_json /path/to/labels.json \\
        --label_field tirads \\
        --output results.csv \\
        --eval_output eval_result.txt

标签 JSON 格式示例:
    [
        {"filename": "a.jpg", "malignancy": 0, "tirads": 2},
        {"filename": "b.jpg", "malignancy": 1, "tirads": 4}
    ]

注意:
    - 标签值应为整数索引，与 --class_names 的顺序对应
      例如 --class_names benign malignant，则 malignancy=0 → benign，malignancy=1 → malignant
    - 模型路径通过 --model_dir 指定本地预训练模型目录（默认 ./pretrained_models/biomedclip）
"""

import os
import sys
import csv
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ---- 可选：scikit-learn 用于评估指标 ----
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")


# ============================================================================
# 模型定义（内联自 model.py，无需外部依赖）
# ============================================================================

class BiomedCLIPClassifier(nn.Module):
    """
    在 BiomedCLIP 图像编码器之上添加分类头。
    策略: full_finetune（推理时策略不影响结果）
    """

    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        from open_clip import create_model_from_pretrained
        model, _ = create_model_from_pretrained(model_name)

        self.visual = model.visual
        self.logit_scale = model.logit_scale
        self.embed_dim = self._get_embed_dim(model.visual)

        # 分类头（与训练时结构完全一致）
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    @staticmethod
    def _get_embed_dim(visual) -> int:
        try:
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = visual(dummy)
            return out.shape[-1]
        except Exception:
            pass
        if hasattr(visual, 'output_dim'):
            return visual.output_dim
        if hasattr(visual, 'trunk'):
            trunk = visual.trunk
            for attr in ('num_features', 'embed_dim'):
                if hasattr(trunk, attr):
                    return getattr(trunk, attr)
        return 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.visual(x)
        logits = self.classifier(features)
        return logits


# ============================================================================
# 工具函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="BiomedCLIP 分类推理（支持可选标签文件进行性能评估）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 必选参数
    parser.add_argument("--ckpt", type=str, required=True,
                        help="训练好的模型权重路径 (.pth)")
    parser.add_argument("--folder", type=str, required=True,
                        help="待推理的图片文件夹路径")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="类别数，二分类填 2，TIRADS 五分类填 5")
    parser.add_argument("--class_names", type=str, nargs="+", required=True,
                        help="类别名称列表，顺序与训练时一致，例如: benign malignant 或 1 2 3 4 5")

    # 模型相关
    parser.add_argument("--model_dir", type=str,
                        default="./pretrained_models/biomedclip",
                        help="本地 BiomedCLIP 预训练模型目录（默认 ./pretrained_models/biomedclip）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备: cuda 或 cpu（默认 cuda，无 GPU 自动回退 cpu）")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批推理大小（默认 32）")

    # 输出
    parser.add_argument("--output", type=str, default="results.csv",
                        help="分类结果 CSV 输出路径（默认 results.csv）")

    # 可选：标签文件与评估
    parser.add_argument("--label_json", type=str, default=None,
                        help="标签 JSON 文件路径（可选）；提供后将额外输出分类性能指标")
    parser.add_argument("--label_field", type=str, default=None,
                        help="JSON 中用于评估的标签字段名，例如 malignancy 或 tirads（提供 --label_json 时必填）")
    parser.add_argument("--eval_output", type=str, default=None,
                        help="评估结果保存路径 (.txt/.log)；未指定时自动在 --output 同级目录生成")

    return parser.parse_args()


def get_preprocess(image_size: int = 224):
    """与训练一致的推理预处理"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def load_model(ckpt_path: str, model_name: str, num_classes: int, device: torch.device) -> BiomedCLIPClassifier:
    """加载训练好的分类模型"""
    print(f"  加载预训练骨干: {model_name}")
    model = BiomedCLIPClassifier(
        model_name=model_name,
        num_classes=num_classes,
    )
    print(f"  加载分类权重:   {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量:     {total_params:,}")
    return model


def collect_images(folder: str):
    """收集文件夹中所有图片文件（按文件名排序）"""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted([
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    ])
    return files


def load_label_json(json_path: str, label_field: str):
    """
    加载标签 JSON 文件。
    返回: dict {filename: label_int}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    label_map = {}
    missing = []
    for rec in records:
        fname = rec.get("filename")
        if fname is None:
            continue
        if label_field not in rec:
            missing.append(fname)
            continue
        label_map[fname] = int(rec[label_field])

    if missing:
        print(f"  ⚠ 以下 {len(missing)} 条记录缺少字段 '{label_field}'，将跳过评估: {missing[:5]}{'...' if len(missing)>5 else ''}")

    return label_map


@torch.no_grad()
def batch_infer(model: BiomedCLIPClassifier, img_paths: list,
                preprocess, device: torch.device, batch_size: int):
    """
    批量推理，返回每张图片的概率数组。
    返回: np.ndarray, shape (N, num_classes)
    """
    all_probs = []
    for i in tqdm(range(0, len(img_paths), batch_size), desc="推理中"):
        batch_paths = img_paths[i: i + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
            except Exception as e:
                print(f"\n  ⚠ 读取图片失败: {p} ({e})，使用零张量替代")
                tensors.append(torch.zeros(3, 224, 224))

        batch = torch.stack(tensors).to(device)
        logits = model(batch)
        probs = logits.softmax(dim=1).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)  # (N, num_classes)


def save_csv(output_path: str, filenames: list, all_probs: np.ndarray, class_names: list):
    """保存分类结果到 CSV"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fieldnames = ["filename", "predict_label", "predict_confidence"]
    for cname in class_names:
        fieldnames.append(f"prob_{cname}")

    rows = []
    for fname, probs in zip(filenames, all_probs):
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]
        pred_conf = float(probs[pred_idx])

        row = {
            "filename": fname,
            "predict_label": pred_name,
            "predict_confidence": round(pred_conf, 6),
        }
        for i, cname in enumerate(class_names):
            row[f"prob_{cname}"] = round(float(probs[i]), 6)
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


# ============================================================================
# 评估相关
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray, num_classes: int):
    """计算分类性能指标"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if num_classes == 2:
        metrics["precision_pos"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics["recall_pos"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        metrics["f1_pos"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        except ValueError:
            metrics["auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    return metrics, cm


def format_eval_report(metrics: dict, cm: np.ndarray, class_names: list,
                       num_samples: int, num_classes: int,
                       y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """格式化评估报告为字符串"""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  BiomedCLIP 分类评估结果")
    lines.append(f"  评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  评估样本数: {num_samples}")
    lines.append("=" * 60)

    lines.append("\n  主要指标:")
    lines.append("  " + "-" * 42)
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
            lines.append(f"  {label:<22s}: {metrics[k]:.4f}")
    if "auc" in metrics:
        lines.append(f"  {'AUC':<22s}: {metrics['auc']:.4f}")

    # 混淆矩阵
    display_names = class_names if class_names else [str(i) for i in range(len(cm))]
    lines.append(f"\n  混淆矩阵 (行=真实标签, 列=预测标签):")
    header = f"  {'':>12s}" + "".join(f"{name:>8s}" for name in display_names)
    lines.append(header)
    for i, row in enumerate(cm):
        lines.append(f"  {display_names[i]:>10s}  " + "".join(f"{v:>8d}" for v in row))

    # 每类准确率
    lines.append(f"\n  每类准确率:")
    for i in range(len(cm)):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        name = display_names[i]
        acc = class_correct / class_total if class_total > 0 else float("nan")
        lines.append(f"  {name:<12s}: {class_correct}/{class_total} = {acc:.4f}")

    # 二分类正类详细
    if num_classes == 2 and "precision_pos" in metrics:
        pos_name = display_names[1]
        lines.append(f"\n  正类 ({pos_name}) 详细指标:")
        lines.append(f"  {'精确率':<22s}: {metrics['precision_pos']:.4f}")
        lines.append(f"  {'召回率':<22s}: {metrics['recall_pos']:.4f}")
        lines.append(f"  {'F1':<22s}: {metrics['f1_pos']:.4f}")

    # 多分类：per-class 分类报告
    if num_classes > 2:
        lines.append(f"\n  分类报告 (per-class):")
        report = classification_report(y_true, y_pred, target_names=display_names, zero_division=0)
        for line in report.splitlines():
            lines.append("  " + line)

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def run_evaluation(filenames: list, all_probs: np.ndarray, label_map: dict,
                   class_names: list, num_classes: int, eval_output: str):
    """
    对有标签的样本进行评估，保存结果到 txt 文件。
    """
    if not SKLEARN_AVAILABLE:
        print("  ⚠ scikit-learn 未安装，无法进行性能评估。请执行: pip install scikit-learn")
        return

    # 匹配有标签的样本
    y_true_list, y_pred_list, y_prob_list = [], [], []
    skipped = []

    for fname, probs in zip(filenames, all_probs):
        if fname not in label_map:
            skipped.append(fname)
            continue
        true_label = label_map[fname]
        pred_idx = int(np.argmax(probs))
        y_true_list.append(true_label)
        y_pred_list.append(pred_idx)
        y_prob_list.append(probs)

    if not y_true_list:
        print("  ⚠ 没有找到任何匹配的标签记录，无法评估。请检查 JSON 中的 filename 是否与图片文件名一致。")
        return

    if skipped:
        print(f"  ⚠ {len(skipped)} 张图片在标签文件中未找到对应记录，已跳过评估。")

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    y_prob = np.array(y_prob_list)

    num_eval = len(y_true)
    print(f"\n  参与评估样本数: {num_eval}")

    metrics, cm = compute_metrics(y_true, y_pred, y_prob, num_classes)
    report_str = format_eval_report(metrics, cm, class_names, num_eval, num_classes, y_true, y_pred)

    # 打印到终端
    print(report_str)

    # 保存到文件
    os.makedirs(os.path.dirname(os.path.abspath(eval_output)), exist_ok=True)
    with open(eval_output, "w", encoding="utf-8") as f:
        f.write(report_str)
        f.write("\n")

    print(f"\n  评估结果已保存至: {eval_output}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()

    # 参数校验
    if len(args.class_names) != args.num_classes:
        print(f"错误: --class_names 长度 ({len(args.class_names)}) 与 --num_classes ({args.num_classes}) 不一致")
        sys.exit(1)

    if args.label_json is not None and args.label_field is None:
        print("错误: 提供了 --label_json 时必须同时指定 --label_field")
        sys.exit(1)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  BiomedCLIP 分类推理")
    print(f"{'='*60}")
    print(f"  图片文件夹: {args.folder}")
    print(f"  模型权重:   {args.ckpt}")
    print(f"  类别数:     {args.num_classes}")
    print(f"  类别名称:   {args.class_names}")
    print(f"  设备:       {device}")
    print(f"  批大小:     {args.batch_size}")
    print(f"  输出 CSV:   {args.output}")
    if args.label_json:
        print(f"  标签文件:   {args.label_json}")
        print(f"  标签字段:   {args.label_field}")
    print(f"{'='*60}")

    # 检查文件夹
    if not os.path.isdir(args.folder):
        print(f"错误: 图片文件夹不存在: {args.folder}")
        sys.exit(1)

    # 确定模型名称
    model_dir = args.model_dir
    if os.path.isdir(model_dir):
        model_name = f"local-dir:{model_dir}"
    else:
        print(f"  ⚠ 本地模型目录不存在: {model_dir}，将从 HuggingFace Hub 加载")
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    # 加载模型
    print(f"\n  加载模型...")
    model = load_model(args.ckpt, model_name, args.num_classes, device)
    preprocess = get_preprocess(image_size=224)

    # 收集图片
    filenames = collect_images(args.folder)
    if not filenames:
        print(f"错误: 文件夹中未找到图片文件: {args.folder}")
        sys.exit(1)
    print(f"\n  找到 {len(filenames)} 张图片")

    img_paths = [os.path.join(args.folder, f) for f in filenames]

    # 批量推理
    print()
    all_probs = batch_infer(model, img_paths, preprocess, device, args.batch_size)

    # 保存 CSV
    save_csv(args.output, filenames, all_probs, args.class_names)
    print(f"\n  分类结果已保存至: {args.output}  (共 {len(filenames)} 条记录)")

    # 可选：性能评估
    if args.label_json:
        print(f"\n{'='*60}")
        print(f"  开始性能评估")
        print(f"{'='*60}")

        label_map = load_label_json(args.label_json, args.label_field)
        print(f"  标签文件共 {len(label_map)} 条有效记录")

        # 确定评估结果保存路径
        if args.eval_output:
            eval_output = args.eval_output
        else:
            out_dir = os.path.dirname(os.path.abspath(args.output))
            timestamp = datetime.now().strftime("%m%d_%H%M%S")
            eval_output = os.path.join(out_dir, f"eval_result_{timestamp}.txt")

        run_evaluation(filenames, all_probs, label_map, args.class_names,
                       args.num_classes, eval_output)

    print(f"\n  完成")


if __name__ == "__main__":
    main()
