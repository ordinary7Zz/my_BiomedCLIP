"""
BiomedCLIP 微调配置文件 — 多分类版本
======================================
修改此文件中的参数以适配多分类任务。
使用方式:
    python train.py --config config_multiclass
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Config:
    # ==================== 模型 ====================
    model_name: str = "local-dir:./pretrained_models/biomedclip"
    local_model_path: Optional[str] = None
    hf_endpoint: str = ""

    # ==================== 数据集 ====================
    # 结构:
    #   data_dir/
    #   ├── train/class_A/img001.png  ...
    #   ├── val/  class_A/img001.png  ...
    #   └── test/ class_A/img001.png  ...
    data_dir: str = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/Cine-Clip_by_TIRADS/images"

    # ==================== 分类任务 ====================
    num_classes: int = 5  # 修改为你的实际类别数
    class_names: Optional[List[str]] = None
    # class_names = ["class_A", "class_B", "class_C"]

    # ==================== 微调策略 ====================
    strategy: str = "full_finetune"    # linear_probe → full_finetune: 解冻 ViT 学细粒度特征

    # ==================== 训练超参 ====================
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-5                     # full_finetune 用更小的学习率
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    lr_scheduler: str = "cosine"

    # ==================== 数据增强 ====================
    image_size: int = 224
    use_random_hflip: bool = True
    use_random_rotation: bool = True
    rotation_degrees: float = 10.0
    use_random_resized_crop: bool = True
    crop_scale: tuple = (0.85, 1.0)

    # ==================== 类别不平衡处理 ====================
    use_class_weights: bool = False      # 关掉: 极端不平衡下权重反而导致多数类被忽略
    use_weighted_sampler: bool = True    # 启用: 按样本权重上采样, 保证每批各类均衡
    use_focal_loss: bool = True          # Focal Loss: 自动降低易分类样本权重, 聚焦难样本
    focal_alpha: float = 0.25            # Focal Loss alpha 参数
    focal_gamma: float = 2.0             # Focal Loss gamma 参数

    # ==================== 正则化 ====================
    dropout: float = 0.3
    label_smoothing: float = 0.0         # Focal Loss 自带正则效果, 关掉 label smoothing

    # ==================== 硬件 ====================
    device: str = "cuda"
    num_workers: int = 4
    use_amp: bool = True
    gradient_accumulation_steps: int = 1

    # ==================== 日志与保存 ====================
    output_dir: str = "./output/TIRADS"
    log_dir: str = "./output/logs/TIRADS"
    save_every_epoch: int = 10
    early_stopping_patience: int = 10
    seed: int = 42

    # ==================== 验证策略 ====================
    use_kfold: bool = False
    kfold_n_splits: int = 5
