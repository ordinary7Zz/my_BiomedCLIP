"""
数据准备与数据增强
===================
根据 config.py 中的参数自动组装 dataloader。
支持 ImageFolder 格式数据 和 K-Fold Cross Validation。
"""

import os
import random
import numpy as np
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold

from config import Config


def get_transforms(cfg: Config, is_train: bool = True):
    """
    获取图像预处理 pipeline。
    BiomedCLIP 使用 CLIP 标准归一化参数。
    """
    if is_train:
        aug_list = []
        if cfg.use_random_resized_crop:
            aug_list.append(transforms.RandomResizedCrop(cfg.image_size, scale=cfg.crop_scale))
        else:
            aug_list.append(transforms.Resize((cfg.image_size, cfg.image_size)))

        if cfg.use_random_hflip:
            aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if cfg.use_random_rotation:
            aug_list.append(transforms.RandomRotation(cfg.rotation_degrees))
        aug_list.append(transforms.ToTensor())
        aug_list.append(
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
        )
        return transforms.Compose(aug_list)
    else:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """根据训练集标签分布自动计算类别权重 (用于 CrossEntropyLoss)"""
    labels = [label for _, label in dataset.samples]
    counter = Counter(labels)
    total = len(labels)
    num_classes = len(counter)
    weights = [total / (num_classes * counter[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(cfg: Config):
    """
    创建 train / val / test dataloader。

    返回:
        train_loader, val_loader, test_loader, class_names, class_weights
    """
    train_transform = get_transforms(cfg, is_train=True)
    eval_transform = get_transforms(cfg, is_train=False)

    train_dir = os.path.join(cfg.data_dir, "train")
    val_dir = os.path.join(cfg.data_dir, "val")
    test_dir = os.path.join(cfg.data_dir, "test")

    # 检查目录
    for d, name in [(train_dir, "train"), (val_dir, "val"), (test_dir, "test")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"找不到 {name} 数据目录: {d}")

    # 创建数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    class_names = train_dataset.classes

    # 验证类别一致
    assert train_dataset.classes == val_dataset.classes == test_dataset.classes, \
        "train/val/test 的类别顺序不一致!"

    # 类别权重
    class_weights = compute_class_weights(train_dataset) if cfg.use_class_weights else None

    # --- 类别不平衡采样 (可选) ---
    train_sampler = None
    if cfg.use_weighted_sampler:
        labels = [s[1] for s in train_dataset.samples]
        class_counts = Counter(labels)
        sample_weights = [1.0 / class_counts[l] for l in labels]
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(f"✓ Train: {len(train_dataset)} ({dict(Counter([s[1] for s in train_dataset.samples]))})")
    print(f"✓ Val:   {len(val_dataset)}")
    print(f"✓ Test:  {len(test_dataset)}")
    print(f"✓ Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names, class_weights


def create_kfold_dataloaders(cfg: Config):
    """
    创建 K-Fold Cross Validation 数据加载器。

    数据需要放在 data_dir/ 下 (不分 train/val/test, 全部数据按 ImageFolder 组织),
    自动按 K 折划分。

    返回:
        folds: List of (train_loader, val_loader) pairs, class_names, class_weights
    """
    train_transform = get_transforms(cfg, is_train=True)
    eval_transform = get_transforms(cfg, is_train=False)

    full_dataset = datasets.ImageFolder(cfg.data_dir, transform=None)
    class_names = full_dataset.classes
    labels = np.array([s[1] for s in full_dataset.samples])

    skf = StratifiedKFold(n_splits=cfg.kfold_n_splits, shuffle=True, random_state=cfg.seed)

    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # K-Fold 中有两个 transform 版本需要分别处理
        train_ds = SubsetWrapper(full_dataset, train_indices, train_transform)
        val_ds = SubsetWrapper(full_dataset, val_indices, eval_transform)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                                  num_workers=cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=cfg.num_workers, pin_memory=True)
        folds.append((train_loader, val_loader, train_indices, val_indices))
        print(f"Fold {fold_idx + 1}: train={len(train_indices)}, val={len(val_indices)}")

    # 类别权重 (基于全量数据)
    class_weights = compute_class_weights(full_dataset) if cfg.use_class_weights else None

    return folds, class_names, class_weights


class SubsetWrapper(torch.utils.data.Dataset):
    """对 Subset 包装, 允许使用不同 transform"""
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label
