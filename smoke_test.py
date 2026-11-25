"""Teste de fumaca: roda sem o dataset, com imagens sinteticas.

Serve para verificar que o codigo funciona antes de voce gastar tempo
baixando os dados e treinando. Deve passar em menos de um minuto na CPU.

    python smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from data import IMAGE_SIZE, _stratified_split, build_transforms
from gradcam import explain_image
from models import MODEL_NAMES, build_model, count_parameters
from predict import ler
from train import evaluate_epoch, train_one_epoch

OK = "  [ok] "
N_CLASSES = 6


def imagem_sintetica(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 190, size=(320, 320, 3), dtype=np.uint8)
    base[80:240, 80:240] = rng.integers(150, 255, size=(160, 160, 3), dtype=np.uint8)
    return Image.fromarray(base)


def test_transforms() -> None:
    t_train = build_transforms(train=True)(imagem_sintetica())
    t_eval = build_transforms(train=False)(imagem_sintetica())
    assert t_train.shape == (3, IMAGE_SIZE, IMAGE_SIZE), t_train.shape
    assert t_eval.shape == (3, IMAGE_SIZE, IMAGE_SIZE), t_eval.shape
    print(OK + f"transformacoes devolvem {tuple(t_eval.shape)}")


def test_split_estratificado() -> None:
    labels = [i % 5 for i in range(100)]
    train_idx, val_idx = _stratified_split(labels, val_fraction=0.2, seed=7)
    assert set(train_idx) & set(val_idx) == set(), "treino e validacao se sobrepoem"
    assert len(train_idx) + len(val_idx) == len(labels), "indices perdidos no split"
    por_classe_val = {c: sum(1 for i in val_idx if labels[i] == c) for c in range(5)}
    assert all(v == 4 for v in por_classe_val.values()), por_classe_val
    print(OK + f"split estratificado: {len(train_idx)} treino / {len(val_idx)} val, "
               "4 por classe na validacao")


def test_modelos() -> None:
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    for nome in MODEL_NAMES:
        model = build_model(nome, N_CLASSES, pretrained=False)
        y = model(x)
        assert y.shape == (2, N_CLASSES), y.shape
        camada = model.gradcam_target_layer()
        assert isinstance(camada, nn.Module)
        print(OK + f"{nome}: saida {tuple(y.shape)}, "
                   f"{count_parameters(model):,} parametros, camada alvo {type(camada).__name__}")


def test_loop_de_treino() -> None:
    """Duas epocas em dados aleatorios: verifica que o loop roda e a perda muda."""
    x = torch.randn(12, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.randint(0, N_CLASSES, (12,))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=4)

    model = build_model("cnn", N_CLASSES)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    perda_1, _ = train_one_epoch(model, loader, torch.device("cpu"), criterion, optimizer)
    perda_2, _ = train_one_epoch(model, loader, torch.device("cpu"), criterion, optimizer)
    val_loss, val_acc, val_f1 = evaluate_epoch(model, loader, torch.device("cpu"), criterion)

    assert np.isfinite(perda_1) and np.isfinite(perda_2), "perda virou NaN"
    assert 0.0 <= val_acc <= 1.0 and 0.0 <= val_f1 <= 1.0
    print(OK + f"treino roda: perda {perda_1:.3f} -> {perda_2:.3f}, "
               f"avaliacao devolve acc {val_acc:.2f} / f1 {val_f1:.2f}")


def test_gradcam() -> None:
    model = build_model("mobilenet", N_CLASSES, pretrained=False)
    overlay, idx, probs = explain_image(model, imagem_sintetica(1))
    assert overlay.size == (IMAGE_SIZE, IMAGE_SIZE), overlay.size
    assert 0 <= idx < N_CLASSES
    assert abs(float(probs.sum()) - 1.0) < 1e-4, float(probs.sum())
    arr = np.asarray(overlay)
    assert arr.std() > 1.0, "sobreposicao saiu uniforme; hooks provavelmente falharam"
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cam.png"
        overlay.save(p)
        assert p.stat().st_size > 1000
    print(OK + f"Grad-CAM gera sobreposicao {overlay.size}, classe {idx}, "
               f"probabilidades somam 1")


def test_regra_de_decisao() -> None:
    """A regra de confianca precisa cobrir os tres caminhos do agente."""
    class Fixo(nn.Module):
        def __init__(self, probs):
            super().__init__()
            self.logits = torch.log(torch.tensor([probs]))

        def forward(self, _x):
            return self.logits

    nomes = [f"raca {i}" for i in range(4)]
    casos = {
        "afirmar":   [0.80, 0.10, 0.06, 0.04],
        "comparar":  [0.46, 0.40, 0.10, 0.04],
        "perguntar": [0.28, 0.26, 0.24, 0.22],
    }
    for esperado, probs in casos.items():
        leitura = ler(Fixo(probs), nomes, imagem_sintetica(2))
        assert leitura.decisao == esperado, f"{probs} -> {leitura.decisao}, esperado {esperado}"
        print(OK + f"topo {leitura.topo:.0%} margem {leitura.margem:.0%} -> {esperado}")


def main() -> int:
    torch.manual_seed(0)
    testes = [
        ("transformacoes", test_transforms),
        ("split estratificado", test_split_estratificado),
        ("modelos", test_modelos),
        ("loop de treino", test_loop_de_treino),
        ("Grad-CAM", test_gradcam),
        ("regra de decisao", test_regra_de_decisao),
    ]
    falhas = 0
    for nome, fn in testes:
        print(f"\n{nome}")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            falhas += 1
            print(f"  [FALHOU] {type(exc).__name__}: {exc}")
    print("\n" + "=" * 60)
    if falhas:
        print(f"{falhas} teste(s) falharam")
        return 1
    print("tudo passou. Proximo passo:  python train.py --model mobilenet --epochs 20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
