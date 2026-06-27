"""
BiomedCLIP 分类模型
===================
支持 linear_probe 和 full_finetune 两种策略。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model_from_pretrained


class BiomedCLIPClassifier(nn.Module):
    """
    在 BiomedCLIP 图像编码器之上添加分类头。

    两种策略:
      - linear_probe:   冻结图像编码器, 只训练分类头
      - full_finetune:  同时训练图像编码器和分类头
    """

    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        num_classes: int = 2,
        strategy: str = "linear_probe",
        dropout: float = 0.3,
        pretrained: str = None,       # 本地权重路径 (可选)
    ):
        super().__init__()

        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        # ---- 加载 BiomedCLIP ----
        if pretrained is not None:
            # 从本地加载
            import json
            model, _ = create_model_from_pretrained(model_name)
            state = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(state, strict=False)
        else:
            model, _ = create_model_from_pretrained(model_name)

        self.visual = model.visual          # ViT-B/16 图像编码器
        self.logit_scale = model.logit_scale  # 跨模态 temperature (微调分类时基本不用)

        # 获取 ViT 输出维度 (多种 fallback 方式)
        self.embed_dim = self._get_embed_dim(model.visual)

        # ---- 策略: 冻结/解冻 ----
        self.strategy = strategy
        if strategy == "linear_probe":
            for param in self.visual.parameters():
                param.requires_grad = False
        elif strategy == "full_finetune":
            pass  # 全部可训练
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'linear_probe' or 'full_finetune'.")

        # ---- 分类头 ----
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

        self._init_classifier()

    @staticmethod
    def _get_embed_dim(visual) -> int:
        """获取视觉编码器实际输出维度，兼容不同 open_clip 版本和模型配置"""
        # 方式 1: 用 dummy input 直接推断 (最可靠)
        try:
            import torch
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                out = visual(dummy)
            return out.shape[-1]
        except Exception:
            pass

        # 方式 2: 直接属性 output_dim (标准 CLIP 模型)
        if hasattr(visual, 'output_dim'):
            return visual.output_dim

        # 方式 3: ViT trunk 内部属性
        if hasattr(visual, 'trunk'):
            trunk = visual.trunk
            for attr in ('num_features', 'embed_dim'):
                if hasattr(trunk, attr):
                    return getattr(trunk, attr)

        # 方式 4: 硬编码默认值
        return 768

    def _init_classifier(self):
        """用 Xavier 初始化分类头参数"""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) 图像 batch
        Returns:
            logits: (B, num_classes)
        """
        # 提取图像特征
        features = self.visual(x)          # (B, 768)
        # 分类头
        logits = self.classifier(features)  # (B, num_classes)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """仅提取图像特征, 不经过分类头"""
        with torch.no_grad():
            return self.visual(x)

    @property
    def trainable_param_count(self) -> int:
        """可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_param_count(self) -> int:
        """总参数量"""
        return sum(p.numel() for p in self.parameters())


def create_classifier(
    model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    num_classes: int = 2,
    strategy: str = "linear_probe",
    dropout: float = 0.3,
    pretrained: str = None,
) -> BiomedCLIPClassifier:
    """工厂函数: 创建 BiomedCLIP 分类器"""
    return BiomedCLIPClassifier(
        model_name=model_name,
        num_classes=num_classes,
        strategy=strategy,
        dropout=dropout,
        pretrained=pretrained,
    )
