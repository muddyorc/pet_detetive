"""Treino das duas arquiteturas.

Uso:
    python train.py --model cnn        --epochs 40
    python train.py --model mobilenet  --epochs 20

As duas chamadas usam o mesmo split (semente fixa), o mesmo aumento de dados
e a mesma rotina de avaliacao. So o modelo muda -- e e isso que torna a
comparacao do banner defensavel.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from data import load_data, save_class_names
from models import MODEL_NAMES, build_model, count_parameters


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_epoch(model: nn.Module, loader, device, criterion) -> tuple[float, float, float]:
    """Devolve (perda, acuracia, f1 macro) sobre um loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds: list[int] = []
    all_true: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_true.extend(labels.cpu().tolist())
    if total == 0:
        return 0.0, 0.0, 0.0
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    return total_loss / total, correct / total, float(macro_f1)


def train_one_epoch(model, loader, device, criterion, optimizer) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Treina uma das duas redes do Pet Detetive")
    ap.add_argument("--model", choices=MODEL_NAMES, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="taxa do classificador")
    ap.add_argument("--backbone-lr", type=float, default=1e-4,
                    help="taxa do corpo pre-treinado (so para mobilenet)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=6, help="epocas sem ganho antes de parar")
    ap.add_argument("--num-classes", type=int, default=None,
                    help="usa apenas as N primeiras racas (plano de contorno)")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", default="./checkpoints")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"dispositivo: {device}")
    bundle = load_data(
        root=args.data_root,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.workers,
        num_classes=args.num_classes,
    )
    save_class_names(bundle.class_names, out_dir / "class_names.json")
    print(f"{bundle.num_classes} classes | treino {len(bundle.train_loader.dataset)} | "
          f"val {len(bundle.val_loader.dataset)} | teste {len(bundle.test_loader.dataset)}")

    model = build_model(args.model, bundle.num_classes).to(device)
    print(f"modelo {args.model}: {count_parameters(model):,} parametros treinaveis")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if hasattr(model, "param_groups"):
        optimizer = torch.optim.AdamW(
            model.param_groups(head_lr=args.lr, backbone_lr=args.backbone_lr),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict] = []
    best_f1, best_epoch, epochs_without_gain = -1.0, -1, 0
    ckpt_path = out_dir / f"{args.model}.pt"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, bundle.train_loader, device,
                                                criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, bundle.val_loader, device, criterion)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1})

        marker = ""
        # Parada antecipada guiada por F1 macro, nao por acuracia: com classes
        # de tamanhos parecidos mas confundiveis, o F1 macro e mais informativo.
        if val_f1 > best_f1:
            best_f1, best_epoch, epochs_without_gain = val_f1, epoch, 0
            torch.save({"model": args.model,
                        "state_dict": model.state_dict(),
                        "class_names": bundle.class_names,
                        "val_macro_f1": val_f1,
                        "epoch": epoch}, ckpt_path)
            marker = "  <- melhor, salvo"
        else:
            epochs_without_gain += 1

        print(f"epoca {epoch:3d}/{args.epochs}  "
              f"treino perda {train_loss:.3f} acc {train_acc:.3f}  |  "
              f"val perda {val_loss:.3f} acc {val_acc:.3f} f1 {val_f1:.3f}  "
              f"({time.time() - t0:.0f}s){marker}")

        if epochs_without_gain >= args.patience:
            print(f"parada antecipada: {args.patience} epocas sem ganho de F1 macro")
            break

    (out_dir / f"{args.model}_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nmelhor epoca: {best_epoch}  F1 macro de validacao: {best_f1:.4f}")
    print(f"tempo total: {(time.time() - started) / 60:.1f} min")
    print(f"checkpoint: {ckpt_path}")
    print(f"\nproximo passo:  python evaluate.py --model {args.model}")


if __name__ == "__main__":
    main()
