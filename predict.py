"""Inferencia: carrega um checkpoint e classifica uma foto.

Tambem define a leitura de confianca que o agente usa para decidir entre
afirmar, comparar duas candidatas ou perguntar ao visitante.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from PIL import Image

from data import build_transforms
from models import build_model

# Limiares da camada de decisao. Ponto de partida -- calibre em FOTOS DE
# CELULAR, nao no conjunto de teste.
#
# O teste e Oxford-IIIT Pet: 37 racas puras, bem enquadradas, animal centralizado.
# A bancada e o oposto -- a maioria dos pets e SRD e a foto vem torta e com pouca
# luz. Sao duas distribuicoes diferentes. Limiares calibrados no teste ficam
# permissivos demais na bancada: a rede fica confiante em bobagem num mestico,
# o topo passa de CONFIANCA_ALTA e o sistema "afirma" uma raca exatamente na
# hora em que deveria "comparar" ou "perguntar".
#
# Use o conjunto de teste apenas para as metricas do banner.
CONFIANCA_ALTA = 0.55       # acima disso, afirma
MARGEM_AMBIGUA = 0.15       # diferenca entre 1o e 2o menor que isso: ambiguo
CONFIANCA_BAIXA = 0.30      # topo abaixo disso: pergunta ao visitante


@dataclass
class Palpite:
    raca: str
    probabilidade: float


@dataclass
class Leitura:
    """O que a camada de percepcao entrega ao agente."""
    palpites: list[Palpite]
    decisao: str            # "afirmar" | "comparar" | "perguntar"
    topo: float
    margem: float

    def to_dict(self) -> dict:
        return {"palpites": [asdict(p) for p in self.palpites],
                "decisao": self.decisao,
                "topo": self.topo,
                "margem": self.margem}


def load_checkpoint(model_name: str = "mobilenet",
                    checkpoints: str | Path = "./checkpoints",
                    device: torch.device | None = None) -> tuple[torch.nn.Module, list[str]]:
    ckpt_dir = Path(checkpoints)
    path = ckpt_dir / f"{model_name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint nao encontrado: {path}\n"
            f"Treine primeiro:  python train.py --model {model_name}")
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    class_names = payload.get("class_names")
    if class_names is None:
        class_names = json.loads((ckpt_dir / "class_names.json").read_text(encoding="utf-8"))
    model = build_model(model_name, len(class_names), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, class_names


@torch.no_grad()
def probabilidades(model: torch.nn.Module, pil_image: Image.Image) -> torch.Tensor:
    tensor = build_transforms(train=False)(pil_image.convert("RGB")).unsqueeze(0)
    return torch.softmax(model(tensor), dim=1)[0]


def ler(model: torch.nn.Module, class_names: list[str], pil_image: Image.Image,
        top_k: int = 3) -> Leitura:
    """Classifica e ja aplica a regra de decisao sobre a confianca."""
    probs = probabilidades(model, pil_image)
    ordem = probs.argsort(descending=True)
    palpites = [Palpite(class_names[int(i)], float(probs[int(i)])) for i in ordem[:top_k]]

    topo = palpites[0].probabilidade
    margem = topo - (palpites[1].probabilidade if len(palpites) > 1 else 0.0)

    if topo < CONFIANCA_BAIXA:
        decisao = "perguntar"
    elif margem < MARGEM_AMBIGUA:
        decisao = "comparar"
    elif topo >= CONFIANCA_ALTA:
        decisao = "afirmar"
    else:
        decisao = "comparar"

    return Leitura(palpites=palpites, decisao=decisao, topo=topo, margem=margem)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Classifica uma foto de pet")
    ap.add_argument("image")
    ap.add_argument("--model", default="mobilenet")
    ap.add_argument("--checkpoints", default="./checkpoints")
    args = ap.parse_args()

    model, class_names = load_checkpoint(args.model, args.checkpoints)
    leitura = ler(model, class_names, Image.open(args.image))
    print(f"decisao do agente: {leitura.decisao}  "
          f"(topo {leitura.topo:.0%}, margem {leitura.margem:.0%})")
    for rank, p in enumerate(leitura.palpites, 1):
        print(f"  {rank}. {p.raca:28s} {p.probabilidade * 100:5.1f}%")
