"""Grad-CAM: o mapa de atencao que vai para a tela da bancada.

A ideia: pegue os mapas de ativacao A^k da ultima camada convolucional e os
gradientes da classe predita em relacao a eles. A media espacial de cada
gradiente da o peso daquele mapa; a soma ponderada, passada por ReLU, mostra
quais regioes empurraram a decisao para aquela classe.

Referencia: SELVARAJU, R. R. et al. Grad-CAM. ICCV, 2017.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, build_transforms


class GradCAM:
    """Calcula o mapa de atencao de um modelo para uma imagem.

    Use como context manager para garantir que os hooks sejam removidos:
        with GradCAM(model, model.gradcam_target_layer()) as cam:
            heat = cam(tensor)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __call__(self, image_tensor: torch.Tensor,
                 class_index: int | None = None) -> tuple[np.ndarray, int, torch.Tensor]:
        """Devolve (mapa 0..1 no tamanho da entrada, indice da classe, probabilidades)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.clone().requires_grad_(True)

        logits = self.model(image_tensor)
        probs = F.softmax(logits, dim=1).detach()[0]
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        logits[0, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("hooks nao capturaram ativacoes; verifique a camada alvo")

        # Peso de cada mapa = media espacial do gradiente (alpha^c_k do artigo).
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(image_tensor.shape[2], image_tensor.shape[3]),
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        lo, hi = float(cam.min()), float(cam.max())
        cam = (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)
        return cam, class_index, probs.cpu()


def _jet(values: np.ndarray) -> np.ndarray:
    """Mapa de cores tipo 'jet' sem depender do matplotlib (azul -> vermelho)."""
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Desfaz a normalizacao do ImageNet e devolve HxWx3 em 0..1."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay_heatmap(base_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45,
                    floor: float = 0.25) -> Image.Image:
    """Sobrepoe o mapa de calor na foto original.

    `floor` deixa as regioes frias quase transparentes, para a foto do animal
    continuar reconhecivel -- e isso que faz o visitante entender a figura.
    """
    heat = _jet(cam)
    weight = (np.clip((cam - floor) / max(1e-6, 1 - floor), 0, 1) * alpha)[..., None]
    blended = base_rgb * (1 - weight) + heat * weight
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def explain_image(model: torch.nn.Module, pil_image: Image.Image,
                  class_index: int | None = None) -> tuple[Image.Image, int, torch.Tensor]:
    """Caminho completo: PIL -> tensor -> Grad-CAM -> PIL com sobreposicao."""
    tf = build_transforms(train=False)
    tensor = tf(pil_image.convert("RGB"))
    with GradCAM(model, model.gradcam_target_layer()) as cam_fn:
        cam, idx, probs = cam_fn(tensor, class_index)
    overlay = overlay_heatmap(denormalize(tensor), cam)
    return overlay, idx, probs


def to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    import argparse

    from predict import load_checkpoint

    ap = argparse.ArgumentParser(description="Gera o mapa de atencao de uma foto")
    ap.add_argument("image")
    ap.add_argument("--model", default="mobilenet")
    ap.add_argument("--checkpoints", default="./checkpoints")
    ap.add_argument("--out", default="gradcam.png")
    args = ap.parse_args()

    model, class_names = load_checkpoint(args.model, args.checkpoints)
    overlay, idx, probs = explain_image(model, Image.open(args.image))
    overlay.save(args.out)
    order = probs.argsort(descending=True)[:3]
    print(f"mapa salvo em {args.out}")
    for rank, i in enumerate(order, 1):
        print(f"  {rank}. {class_names[int(i)]:28s} {probs[int(i)] * 100:5.1f}%")
