"""
单张图像 / 批量推理脚本
===========================
加载训练好的 BiomedCLIP 分类模型进行预测。

用法:
    # 单张图片推理
    python inference.py --ckpt output/best_model.pth --image path/to/img.png

    # 文件夹批量推理
    python inference.py --ckpt output/best_model.pth --folder path/to/images/
"""

import os
import argparse
import warnings

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import Config
from model import BiomedCLIPClassifier

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="BiomedCLIP 分类推理")
    parser.add_argument("--ckpt", type=str, required=True, help="训练好的模型权重路径")
    parser.add_argument("--image", type=str, default=None, help="单张图片路径")
    parser.add_argument("--folder", type=str, default=None, help="图片文件夹路径")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="类别名称, 例如: --class_names benign malignant")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--topk", type=int, default=0, help="输出 top-k 类别, 0 表示全部")
    return parser.parse_args()


def load_model(ckpt_path: str, num_classes: int, device: str):
    """加载训练好的分类模型"""
    model = BiomedCLIPClassifier(
        model_name=Config.model_name,
        num_classes=num_classes,
        strategy="full_finetune",  # 策略不影响推理
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def get_preprocess(image_size: int = 224):
    """与训练一致的预处理"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


@torch.no_grad()
def predict_image(model, image_path: str, preprocess, device: str,
                  class_names: list, topk: int = 0):
    """预测单张图片"""
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = logits.softmax(dim=1).squeeze(0).cpu().numpy()

    if topk > 0 and topk < len(probs):
        top_indices = np.argsort(probs)[::-1][:topk]
    else:
        top_indices = np.argsort(probs)[::-1]

    results = []
    for idx in top_indices:
        name = class_names[idx] if class_names else f"class_{idx}"
        results.append((name, idx, float(probs[idx])))

    return results


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 类别名称
    if args.class_names:
        class_names = args.class_names
    else:
        class_names = [f"class_{i}" for i in range(args.num_classes)]

    # 加载模型
    print(f"Loading model: {args.ckpt}")
    model = load_model(args.ckpt, args.num_classes, device)
    preprocess = get_preprocess()

    # ---- 单张推理 ----
    if args.image:
        results = predict_image(model, args.image, preprocess, device, class_names, args.topk)
        print(f"\n图像: {args.image}")
        for name, idx, prob in results:
            print(f"  {name:<15s}  {prob:.4f}  ({prob*100:.2f}%)")
        pred_name = results[0][0]
        pred_prob = results[0][2]
        print(f"\n  => 预测: {pred_name} (置信度: {pred_prob:.4f})")

    # ---- 批量推理 ----
    elif args.folder:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        img_files = sorted([
            f for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        print(f"\n找到 {len(img_files)} 张图片")
        for fname in tqdm(img_files):
            fpath = os.path.join(args.folder, fname)
            results = predict_image(model, fpath, preprocess, device, class_names, topk=1)
            pred_name, _, prob = results[0]
            print(f"  {fname:<40s} => {pred_name:<15s} ({prob:.4f})")

    else:
        print("请指定 --image 或 --folder 参数")


if __name__ == "__main__":
    main()
