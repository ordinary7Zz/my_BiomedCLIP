# BiomedCLIP Fine-tuning for Medical Image Classification

基于 BiomedCLIP 的 2D 医学图像二分类/多分类微调框架。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

按以下结构组织你的 2D 医学图像：

```
dataset/
├── train/
│   ├── benign/          # 类别 0
│   │   ├── img001.png
│   │   └── ...
│   └── malignant/       # 类别 1
│       ├── img001.png
│       └── ...
├── val/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

> **关于灰度图**: 如果你的医学图像是单通道灰度图（如 DICOM 导出的 .png），BiomedCLIP 期望 RGB 3 通道输入，代码中的 Resize+ToTensor 会自动复制通道，无需额外处理。

### 3. 修改配置

编辑 `config.py` 中的参数：

```python
# 必须修改的:
data_dir: str = "./dataset"     # 你的数据集路径
num_classes: int = 2            # 2=二分类, >2=多分类
strategy: str = "linear_probe"  # "linear_probe" 或 "full_finetune"

# Linear Probe 推荐参数:
lr: float = 1e-3
epochs: int = 50

# Full Fine-tune 推荐参数 (在 config.py 中手动改):
lr: float = 1e-5
epochs: int = 30
dropout: float = 0.3
```

### 4. 开始训练

```bash
# 使用 config.py 中的默认配置
python train.py

# 或命令行覆盖参数 (推荐先试 Linear Probe):
python train.py --strategy linear_probe --lr 1e-3 --epochs 50 --batch_size 32

# Full Fine-tune (需要更多数据, LR 要小):
python train.py --strategy full_finetune --lr 1e-5 --epochs 30 --batch_size 16

# K-Fold 交叉验证 (小数据集推荐):
python train.py --use_kfold --strategy linear_probe
```

### 5. 推理

```bash
# 单张图片推理
python inference.py --ckpt output/best_model.pth --image test.png --num_classes 2

# 带类别名
python inference.py --ckpt output/best_model.pth --image test.png \
    --num_classes 2 --class_names benign malignant

# 批量推理
python inference.py --ckpt output/best_model.pth --folder ./test_images/ --num_classes 2
```

### 6. 查看训练曲线

```bash
tensorboard --logdir output/logs
```

然后打开 http://localhost:6006

---

## 项目结构

```
.
├── config.py          # 所有可调参数 (修改这个文件即可)
├── model.py           # BiomedCLIP 分类模型定义
├── data_utils.py      # 数据加载、增强、K-Fold 划分
├── train.py           # 训练主脚本
├── inference.py       # 推理脚本
├── requirements.txt   # 依赖包
├── dataset/           # 你的数据放这里
└── output/            # 模型权重 + TensorBoard 日志
    ├── best_model.pth
    ├── logs/
    └── checkpoints/
```

---

## 两种微调策略对比

| 策略 | 适用场景 | 推荐 LR | 过拟合风险 |
|------|---------|---------|-----------|
| `linear_probe` | 数据少 (<1k/类)，快速验证 | 1e-3 | 低 |
| `full_finetune` | 数据充足 (>2k/类)，追求最优 | 1e-5 | 中-高 |

---

## 针对医学图像的特别说明

### 灰度图

如果你的原图是单通道医学图像（如 X 光、CT 切片），转换为 RGB 即可：

```python
img = Image.open("xray.png").convert("RGB")
```

### 大尺寸图像（病理全切片 WSI）

如果你的图像分辨率远大于 224×224（如病理全切片），需要先切分成小 tile，每张 tile 独立预测后投票或取平均。

### 类别不平衡

代码已自动支持：
- `use_class_weights=True`：自动按训练集频率计算 CrossEntropyLoss 权重
- `use_weighted_sampler=True`：使用 WeightedRandomSampler 均衡采样

### 数据增强

医学图像应谨慎使用强数据增强。`config.py` 中默认为保守增强：
- 随机水平翻转
- ±10° 小角度旋转
- 轻微缩放裁剪 (scale 0.85-1.0)

---

## 模型信息

- **论文**: [BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs](https://arxiv.org/abs/2303.00915)
- **权重**: [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- **图像编码器**: ViT-B/16 (224×224 → 768-dim features)
- **文本编码器**: PubMedBERT (256 token context)
