# BiomedCLIP 分类推理（独立部署版）

本目录是最小可运行版本，**不依赖项目中的任何其他文件**，可直接复制到任意位置运行。

---

## 目录结构

```
BiomedCLIP_deploy/
├── infer.py                    # 推理主脚本（含模型定义）
├── requirements.txt            # 依赖包
├── README.md                   # 使用说明
└── pretrained_models/          # 本地预训练模型
    └── biomedclip/
        ├── open_clip_config.json
        ├── open_clip_model.safetensors
        └── ...
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 用法

### 仅推理（输出 CSV）

二分类：

```bash
python infer.py \
    --ckpt pretrained_models/best_model.pth \
    --folder /path/to/images/ \
    --num_classes 2 \
    --class_names benign malignant \
    --output results.csv
```

TIRADS 五分类：

```bash
python infer.py \
    --ckpt pretrained_models/tirads_model.pth \
    --folder /path/to/images/ \
    --num_classes 5 \
    --class_names 1 2 3 4 5 \
    --output results.csv
```

### 推理 + 性能评估（提供标签文件）

二分类：

```bash
python infer.py \
    --ckpt pretrained_models/best_model.pth \
    --folder /path/to/images/ \
    --num_classes 2 \
    --class_names benign malignant \
    --label_json labels.json \
    --label_field malignancy \
    --output results.csv \
    --eval_output eval_result.txt
```

TIRADS 五分类：

```bash
python infer.py \
    --ckpt pretrained_models/tirads_model.pth \
    --folder /path/to/images/ \
    --num_classes 5 \
    --class_names 1 2 3 4 5 \
    --label_json labels.json \
    --label_field tirads \
    --output results.csv \
    --eval_output eval_result.txt
```

---

## 参数说明

| 参数 | 必选 | 说明 |
|------|------|------|
| `--ckpt` | ✅ | 训练好的分类模型权重路径 (.pth) |
| `--folder` | ✅ | 待推理的图片文件夹路径 |
| `--num_classes` | ✅ | 类别数（二分类填 2，TIRADS 五分类填 5） |
| `--class_names` | ✅ | 类别名称列表，顺序与训练时一致 |
| `--model_dir` | | 本地预训练模型目录（默认 `./pretrained_models/biomedclip`） |
| `--device` | | 推理设备，`cuda` 或 `cpu`（默认 `cuda`，无 GPU 自动回退） |
| `--batch_size` | | 批推理大小（默认 32） |
| `--output` | | CSV 结果输出路径（默认 `results.csv`） |
| `--label_json` | | 标签 JSON 文件路径（可选，提供后进行性能评估） |
| `--label_field` | | JSON 中的标签字段名（提供 `--label_json` 时必填） |
| `--eval_output` | | 评估结果保存路径（未指定时自动生成） |

---

## 标签 JSON 格式

```json
[
    {"filename": "a.jpg", "malignancy": 0, "tirads": 2},
    {"filename": "b.jpg", "malignancy": 1, "tirads": 4}
]
```

- `filename`：与图片文件夹中的文件名对应（仅文件名，不含路径）
- 标签值为整数索引，与 `--class_names` 的位置对应
  - 例如 `--class_names benign malignant`，则 `malignancy=0` → benign，`malignancy=1` → malignant

---

## 输出文件说明

### results.csv

| 列名 | 说明 |
|------|------|
| `filename` | 图片文件名 |
| `predict_label` | 预测类别名 |
| `predict_confidence` | 预测置信度 |
| `prob_<class>` | 每个类别的 softmax 概率 |

### eval_result.txt（仅提供标签文件时生成）

包含以下评估指标：
- 准确率、精确率、召回率、F1（macro / weighted）
- 二分类额外输出：AUC、正类详细指标
- 混淆矩阵
- 每类准确率
- 多分类额外输出：per-class 分类报告
