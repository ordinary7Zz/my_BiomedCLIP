"""
BiomedCLIP 微调配置文件
========================
修改此文件中的参数以适配你的任务。
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Config:
    # ==================== 模型 ====================
    # BiomedCLIP 模型名称 (从 HuggingFace Hub 加载)
    model_name: str = "./pretrained_models/biomedclip"
    # 本地模型路径 (如果已下载, 设置为实际路径即可跳过在线下载)
    local_model_path: Optional[str] = None
    # HuggingFace 镜像 (国内用户推荐 https://hf-mirror.com)
    hf_endpoint: str = ""  # 留空用官方, 设为 "https://hf-mirror.com" 用镜像

    # ==================== 数据集 ====================
    # 数据根目录, 结构应为:
    #   data_dir/
    #   ├── train/class_0/img001.png ...
    #   ├── val/  class_0/img001.png ...
    #   └── test/ class_0/img001.png ...
    data_dir: str = "/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3_cls"

    # ==================== 分类任务 ====================
    # 类别数量: 2 为二分类, >2 为多分类
    num_classes: int = 2

    # 类别名称列表 (用于日志和可视化, 二分类示例见下)
    class_names: Optional[List[str]] = None
    # class_names = ["benign", "malignant"]   # 二分类示例
    # class_names = ["class_A", "class_B", "class_C"]  # 三分类示例

    # ==================== 微调策略 ====================
    # 可选: "linear_probe" | "full_finetune"
    strategy: str = "linear_probe"

    # ==================== 训练超参 ====================
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3            # linear_probe 推荐 1e-3, full_finetune 推荐 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 3
    lr_scheduler: str = "cosine"  # "cosine" | "step" | "none"

    # ==================== 数据增强 ====================
    image_size: int = 224        # BiomedCLIP 使用 224x224
    # 训练集增强 (医学图像建议保守):
    use_random_hflip: bool = True
    use_random_rotation: bool = True
    rotation_degrees: float = 10.0
    use_random_resized_crop: bool = True
    crop_scale: tuple = (0.85, 1.0)   # 医学图像建议别裁剪太多

    # ==================== 类别不平衡处理 ====================
    use_class_weights: bool = True     # 自动按训练集频率计算类别权重
    use_weighted_sampler: bool = False # 也可以用 WeightedRandomSampler

    # ==================== 正则化 ====================
    dropout: float = 0.3
    label_smoothing: float = 0.0       # 0.0 表示不启用

    # ==================== 硬件 ====================
    device: str = "cuda"
    num_workers: int = 4
    use_amp: bool = True               # 混合精度训练 (节省显存)
    gradient_accumulation_steps: int = 1  # 显存不足时可增大, 等效于增大 batch_size

    # ==================== 日志与保存 ====================
    output_dir: str = "./output/BM"
    log_dir: str = "./output/logs/BM"
    save_every_epoch: int = 10         # 每隔多少 epoch 保存一次 checkpoint
    early_stopping_patience: int = 10   # 验证集不再提升时的等待轮数
    seed: int = 42

    # ==================== 验证策略 ====================
    # 仅当数据量很少时用 k-fold, 否则用固定 train/val/test 划分
    use_kfold: bool = False
    kfold_n_splits: int = 5
