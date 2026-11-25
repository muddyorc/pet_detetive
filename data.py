"""Pipeline de dados do Pet Detetive.

Baixa o Oxford-IIIT Pet (37 racas de caes e gatos), faz o split estratificado
treino/validacao e monta os dataloaders. O MESMO pipeline e usado pelas duas
arquiteturas -- essa e a condicao para a comparacao entre elas ser honesta.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

# Estatisticas do ImageNet. Obrigatorias para a MobileNetV3 pre-treinada e
# usadas tambem na CNN propria, para manter o pipeline identico.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_SIZE = 224


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def build_transforms(train: bool) -> transforms.Compose:
    """Aumento de dados no treino, apenas redimensionamento na avaliacao.

    O espelhamento horizontal e seguro aqui: um animal espelhado continua
    sendo o mesmo animal. (Em problemas com maos isso nao valeria.)
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _stratified_split(labels: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Separa indices por classe, garantindo a mesma proporcao em treino e validacao."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class[label].append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for label in sorted(by_class):
        idxs = by_class[label]
        perm = torch.randperm(len(idxs), generator=generator).tolist()
        n_val = max(1, int(round(len(idxs) * val_fraction)))
        for position, p in enumerate(perm):
            (val_idx if position < n_val else train_idx).append(idxs[p])
    return sorted(train_idx), sorted(val_idx)


def _keep_classes(dataset, labels: list[int], keep: set[int]) -> tuple[list[int], dict[int, int]]:
    """Filtra o dataset para um subconjunto de classes e remapeia os rotulos.

    Usado pelo plano de contorno: se 37 racas nao convergirem a tempo, treine
    com 10 ou 12 bem distintas entre si.
    """
    remap = {old: new for new, old in enumerate(sorted(keep))}
    indices = [i for i, label in enumerate(labels) if label in keep]
    return indices, remap


class RemappedSubset(torch.utils.data.Dataset):
    """Subset que reescreve os rotulos para 0..n-1 apos o filtro de classes."""

    def __init__(self, base, indices: list[int], remap: dict[int, int]):
        self.base = base
        self.indices = indices
        self.remap = remap

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, label = self.base[self.indices[i]]
        return image, self.remap[int(label)]


def load_data(
    root: str | Path = "./data",
    batch_size: int = 32,
    val_fraction: float = 0.15,
    seed: int = 42,
    num_workers: int = 2,
    num_classes: int | None = None,
    download: bool = True,
) -> DataBundle:
    """Monta os tres dataloaders.

    num_classes: se informado, mantem apenas as N primeiras racas em ordem
    alfabetica. Util para reduzir o escopo sem mexer em mais nada.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    train_tf = build_transforms(train=True)
    eval_tf = build_transforms(train=False)

    # O Oxford-IIIT Pet ja vem com os splits "trainval" e "test" definidos
    # pelos autores. Respeitamos o split de teste e dividimos o trainval.
    trainval_train_view = OxfordIIITPet(root, split="trainval", target_types="category",
                                        transform=train_tf, download=download)
    trainval_eval_view = OxfordIIITPet(root, split="trainval", target_types="category",
                                       transform=eval_tf, download=False)
    test_set = OxfordIIITPet(root, split="test", target_types="category",
                             transform=eval_tf, download=False)

    all_class_names = list(trainval_train_view.classes)
    trainval_labels = [int(l) for l in trainval_train_view._labels]
    test_labels = [int(l) for l in test_set._labels]

    if num_classes is not None and num_classes < len(all_class_names):
        keep = set(range(num_classes))
        class_names = [all_class_names[i] for i in sorted(keep)]
        tv_idx, remap = _keep_classes(trainval_train_view, trainval_labels, keep)
        te_idx, _ = _keep_classes(test_set, test_labels, keep)
        kept_labels = [remap[trainval_labels[i]] for i in tv_idx]
        train_pos, val_pos = _stratified_split(kept_labels, val_fraction, seed)
        train_ds = RemappedSubset(trainval_train_view, [tv_idx[i] for i in train_pos], remap)
        val_ds = RemappedSubset(trainval_eval_view, [tv_idx[i] for i in val_pos], remap)
        test_ds = RemappedSubset(test_set, te_idx, remap)
    else:
        class_names = all_class_names
        train_pos, val_pos = _stratified_split(trainval_labels, val_fraction, seed)
        train_ds = Subset(trainval_train_view, train_pos)
        val_ds = Subset(trainval_eval_view, val_pos)
        test_ds = test_set

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return DataBundle(
        train_loader=DataLoader(train_ds, shuffle=True, drop_last=False, **common),
        val_loader=DataLoader(val_ds, shuffle=False, **common),
        test_loader=DataLoader(test_ds, shuffle=False, **common),
        class_names=[c.replace("_", " ") for c in class_names],
    )


def save_class_names(class_names: list[str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    bundle = load_data(batch_size=8, num_workers=0)
    images, labels = next(iter(bundle.train_loader))
    print(f"{bundle.num_classes} classes")
    print(f"lote: {tuple(images.shape)}  rotulos: {labels.tolist()}")
    print(f"treino {len(bundle.train_loader.dataset)} | "
          f"val {len(bundle.val_loader.dataset)} | "
          f"teste {len(bundle.test_loader.dataset)}")
