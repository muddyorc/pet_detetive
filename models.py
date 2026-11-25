"""As duas arquiteturas do projeto.

PetCNN      -- rede convolucional propria, treinada do zero.
build_mobilenet -- MobileNetV3 pre-treinada, adaptada por transferencia.

As duas expõem `gradcam_target_layer()`, que devolve a ultima camada
convolucional. E dela que o Grad-CAM tira os gradientes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# --------------------------------------------------------------------------
# 1. CNN propria, treinada do zero
# --------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU, duas vezes, seguido de pooling.

    Batch normalization estabiliza o treino e dispensa o bias da convolucao.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.body(x))


class PetCNN(nn.Module):
    """CNN propria: 4 blocos convolucionais, pooling global e classificador.

    224 -> 112 -> 56 -> 28 -> 14, com 32/64/128/256 canais.
    O dropout antes da camada final e a principal defesa contra sobreajuste,
    que e o risco real com ~170 imagens de treino por raca.
    """

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))

    def gradcam_target_layer(self) -> nn.Module:
        # Ultima ReLU do ultimo bloco, antes do pooling: mapas 14x14.
        return self.features[-1].body[-1]


# --------------------------------------------------------------------------
# 2. MobileNetV3 com transferencia de aprendizado
# --------------------------------------------------------------------------

class MobileNetV3Transfer(nn.Module):
    """MobileNetV3-Large pre-treinada no ImageNet, com classificador novo.

    freeze_backbone=True congela o extrator e treina so o classificador --
    mais rapido e menos propenso a sobreajuste nas primeiras epocas.
    O treino padrao descongela tudo com taxa de aprendizagem menor no corpo.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = net.classifier[0].in_features  # 960
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))

    def gradcam_target_layer(self) -> nn.Module:
        # Ultimo bloco convolucional do extrator: mapas 7x7.
        return self.features[-1]

    def param_groups(self, head_lr: float, backbone_lr: float) -> list[dict]:
        """Taxa de aprendizagem menor no corpo pre-treinado que no classificador novo."""
        return [
            {"params": [p for p in self.features.parameters() if p.requires_grad],
             "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
        ]


# --------------------------------------------------------------------------

MODEL_NAMES = ("cnn", "mobilenet")


def build_model(name: str, num_classes: int, **kwargs) -> nn.Module:
    if name == "cnn":
        return PetCNN(num_classes, dropout=kwargs.get("dropout", 0.4))
    if name == "mobilenet":
        return MobileNetV3Transfer(
            num_classes,
            dropout=kwargs.get("dropout", 0.3),
            pretrained=kwargs.get("pretrained", True),
            freeze_backbone=kwargs.get("freeze_backbone", False),
        )
    raise ValueError(f"modelo desconhecido: {name!r}. Use um de {MODEL_NAMES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    for name in MODEL_NAMES:
        m = build_model(name, num_classes=37, pretrained=False)
        y = m(x)
        print(f"{name:10s} saida {tuple(y.shape)}  parametros {count_parameters(m):,}")
