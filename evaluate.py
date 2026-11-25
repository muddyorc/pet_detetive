"""Avaliacao no conjunto de teste: numeros e figuras que vao para o banner.

Gera:
  outputs/<modelo>_metricas.json      acuracia, F1 macro, latencia, parametros
  outputs/<modelo>_confusao.png       matriz de confusao
  outputs/<modelo>_por_classe.csv     precisao/revocacao/F1 por raca
  outputs/comparacao.md               tabela pronta para colar no banner

Uso:
    python evaluate.py --model cnn
    python evaluate.py --model mobilenet
    python evaluate.py --compare
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from data import load_data
from models import count_parameters
from predict import load_checkpoint
from train import pick_device


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_images, elapsed = 0, 0.0
    for images, labels in loader:
        images = images.to(device)
        t0 = time.perf_counter()
        logits = model(images)
        elapsed += time.perf_counter() - t0
        total_images += images.size(0)
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())
    latency_ms = (elapsed / max(total_images, 1)) * 1000
    return np.array(y_true), np.array(y_pred), latency_ms


def plot_confusion(cm: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    n = len(class_names)
    size = max(6.0, min(18.0, n * 0.42))
    fig, ax = plt.subplots(figsize=(size, size * 0.92), dpi=150)
    # Normaliza por linha: mostra para onde cada raca verdadeira foi parar.
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = cm / cm.sum(axis=1, keepdims=True)
    norm = np.nan_to_num(norm)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, pad=14)
    ax.set_xlabel("predito")
    ax.set_ylabel("verdadeiro")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fontsize = 7 if n > 20 else 9
    ax.set_xticklabels(class_names, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(class_names, fontsize=fontsize)
    if n <= 15:
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.8, label="proporcao da linha")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def piores_confusoes(cm: np.ndarray, class_names: list[str], n: int = 5) -> list[tuple[str, str, int]]:
    """As n maiores confusoes fora da diagonal -- o conteudo mais interessante do banner."""
    pares = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pares.append((class_names[i], class_names[j], int(cm[i, j])))
    return sorted(pares, key=lambda t: -t[2])[:n]


def avaliar(model_name: str, args) -> dict:
    device = pick_device(args.device)
    model, class_names = load_checkpoint(model_name, args.checkpoints, device)
    bundle = load_data(root=args.data_root, batch_size=args.batch_size,
                       num_workers=args.workers, num_classes=len(class_names),
                       download=False)

    y_true, y_pred, latency_ms = collect_predictions(model, bundle.test_loader, device)

    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plot_confusion(cm, class_names, out / f"{model_name}_confusao.png",
                   f"Matriz de confusao — {model_name}  (acc {acc:.1%}, F1 macro {macro:.3f})")

    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    linhas = ["raca,precisao,revocacao,f1,suporte"]
    for nome in class_names:
        r = report[nome]
        linhas.append(f"{nome},{r['precision']:.4f},{r['recall']:.4f},"
                      f"{r['f1-score']:.4f},{int(r['support'])}")
    (out / f"{model_name}_por_classe.csv").write_text("\n".join(linhas), encoding="utf-8")

    metrics = {
        "modelo": model_name,
        "classes": len(class_names),
        "imagens_teste": int(len(y_true)),
        "acuracia": acc,
        "f1_macro": macro,
        "latencia_ms_por_imagem": latency_ms,
        "parametros": count_parameters(model),
        "dispositivo": str(device),
        "piores_confusoes": piores_confusoes(cm, class_names),
    }
    (out / f"{model_name}_metricas.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {model_name} ===")
    print(f"acuracia           {acc:.4f}")
    print(f"F1 macro           {macro:.4f}")
    print(f"latencia/imagem    {latency_ms:.2f} ms  ({device})")
    print(f"parametros         {metrics['parametros']:,}")
    print("maiores confusoes:")
    for verdadeira, predita, qtd in metrics["piores_confusoes"]:
        print(f"   {verdadeira}  ->  {predita}   ({qtd}x)")
    return metrics


def escrever_comparacao(todas: list[dict], out: Path) -> None:
    linhas = [
        "| Modelo | Acuracia | F1 macro | Latencia | Parametros |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m in todas:
        linhas.append(f"| {m['modelo']} | {m['acuracia']:.1%} | {m['f1_macro']:.3f} | "
                      # .2f, nao .0f: na GPU a inferencia leva fracao de
                      # milissegundo e ".0f" imprimia "0 ms" na tabela do banner.
                      f"{m['latencia_ms_por_imagem']:.2f} ms | {m['parametros']:,} |")
    texto = "\n".join(linhas) + "\n"
    (out / "comparacao.md").write_text(texto, encoding="utf-8")
    print("\n" + texto)
    print(f"tabela salva em {out / 'comparacao.md'} — pronta para o banner")


def main() -> None:
    ap = argparse.ArgumentParser(description="Avalia as redes no conjunto de teste")
    ap.add_argument("--model", default=None, help="cnn ou mobilenet")
    ap.add_argument("--compare", action="store_true", help="avalia as duas e compara")
    ap.add_argument("--checkpoints", default="./checkpoints")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="./outputs")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not args.compare and not args.model:
        ap.error("informe --model cnn|mobilenet ou use --compare")

    nomes = ["cnn", "mobilenet"] if args.compare else [args.model]
    todas = [avaliar(nome, args) for nome in nomes]
    if len(todas) > 1:
        escrever_comparacao(todas, Path(args.out))


if __name__ == "__main__":
    main()
