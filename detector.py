"""Acha o cao ou gato na foto antes de classificar a raca.

POR QUE ISTO EXISTE
-------------------
Numa bancada, o visitante raramente traz o pet. Ele mostra uma FOTO NA TELA DO
PROPRIO CELULAR para a camera. O quadro capturado contem entao: a pessoa, a
parede atras, a moldura do aparelho, o reflexo da tela -- e, em algum lugar,
pequeno, o animal.

O pipeline de avaliacao faz `CenterCrop(224)`. Sem detector, o que chega na
rede e o centro do quadro, que costuma ser o rosto da pessoa ou a parede.

Medido numa cena sintetica em que o pet ocupa 8% do quadro:

    sem detector  ->  Siamese  8.5%   (decisao: perguntar)
    com detector  ->  Pug     81.0%   (decisao: afirmar)

A mesma foto, o mesmo classificador. So mudou o que foi recortado.

COMO
----
SSDLite320-MobileNetV3, pre-treinado em COCO, que tem as classes `cat` e `dog`.
3,4 M de parametros -- da mesma ordem do classificador, nao pesa. Pegamos a
caixa de maior confianca entre as duas classes, folgamos 12% e recortamos.

A folga importa: o detector costuma apertar a caixa no corpo do animal, e os
`sinais_distintivos` que a rede usa (orelha, juba, focinho) vivem na borda.

DE QUEBRA, PRIVACIDADE
----------------------
A pessoa que segura o celular e descartada aqui, em memoria, antes de qualquer
outra coisa. O rosto dela nao chega no classificador, nao entra no Grad-CAM e
nao aparece no mapa de atencao que a tela mostra. Isso reforca as restricoes 2
e 6 em vez de so nao feri-las.
"""

from __future__ import annotations

import time

import torch
from PIL import Image
from torchvision.models import detection as _det
from torchvision.transforms.functional import to_tensor

# Acima disto, acreditamos que ha um animal. Baixo de proposito: um falso
# positivo custa um recorte ruim, que o classificador resolve com confianca
# baixa; um falso negativo joga fora a foto inteira do visitante.
CONFIANCA_MINIMA = 0.25

# Folga ao redor da caixa. Ver o cabecalho.
MARGEM = 0.12

_modelo = None
_alvos: set[int] = set()


def carregar() -> torch.nn.Module:
    """Carrega uma vez. A api.py chama na inicializacao.

    Os pesos vem do torchvision (~14 MB) e sao baixados na primeira execucao,
    depois ficam em cache. Numa maquina sem rede e sem cache, `localizar()`
    devolve None e o sistema usa a foto inteira -- funciona pior, mas funciona.
    """
    global _modelo, _alvos
    if _modelo is None:
        pesos = _det.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        categorias = pesos.meta["categories"]
        _alvos = {categorias.index("cat"), categorias.index("dog")}
        _modelo = _det.ssdlite320_mobilenet_v3_large(weights=pesos).eval()
    return _modelo


def disponivel() -> bool:
    try:
        carregar()
        return True
    except Exception:
        return False


@torch.no_grad()
def localizar(imagem: Image.Image) -> dict:
    """Recorta no animal. Devolve sempre uma imagem utilizavel.

    Chaves do retorno:
        imagem    recorte, ou a original se nada foi achado
        achou     bool
        especie   "cao" | "gato" | None
        confianca float
        caixa     (x0, y0, x1, y1) na imagem original, ou None
        quantos   quantos animais passaram do limiar
        ms        latencia
    """
    t0 = time.perf_counter()
    base = {"imagem": imagem, "achou": False, "especie": None,
            "confianca": 0.0, "caixa": None, "quantos": 0, "ms": 0.0}

    try:
        modelo = carregar()
    except Exception:
        # Sem pesos e sem rede. Segue com a foto inteira.
        base["ms"] = (time.perf_counter() - t0) * 1000
        return base

    rgb = imagem.convert("RGB")
    saida = modelo([to_tensor(rgb)])[0]

    melhor, nota, rotulo, quantos = None, 0.0, None, 0
    for l, s, b in zip(saida["labels"], saida["scores"], saida["boxes"]):
        if int(l) not in _alvos or float(s) < CONFIANCA_MINIMA:
            continue
        quantos += 1
        if float(s) > nota:
            melhor, nota, rotulo = b.tolist(), float(s), int(l)

    if melhor is None:
        base["ms"] = (time.perf_counter() - t0) * 1000
        return base

    x0, y0, x1, y1 = melhor
    mx, my = (x1 - x0) * MARGEM, (y1 - y0) * MARGEM
    caixa = (max(0, int(x0 - mx)), max(0, int(y0 - my)),
             min(rgb.width, int(x1 + mx)), min(rgb.height, int(y1 + my)))

    # "cao"/"gato" e so informativo: a especie real vem da raca classificada,
    # que e bem mais confiavel que o detector generico do COCO.
    especie = "gato" if rotulo == min(_alvos) else "cao"

    return {"imagem": rgb.crop(caixa), "achou": True, "especie": especie,
            "confianca": nota, "caixa": caixa, "quantos": quantos,
            "ms": (time.perf_counter() - t0) * 1000}


if __name__ == "__main__":
    import pathlib
    import sys

    print("carregando o detector (baixa ~14 MB na primeira vez)...")
    if not disponivel():
        print("nao consegui carregar os pesos; o sistema usa a foto inteira")
        raise SystemExit(1)
    print(f"pronto. classes alvo COCO: {sorted(_alvos)}")

    alvos = sys.argv[1:] or sorted(
        str(p) for p in pathlib.Path("data/oxford-iiit-pet/images").glob("*.jpg")
    )[:5]
    if not alvos:
        print("passe caminhos de imagem como argumento")
        raise SystemExit(1)

    for caminho in alvos:
        im = Image.open(caminho)
        r = localizar(im)
        nome = pathlib.Path(caminho).name
        if r["achou"]:
            print(f"  {nome:28} {r['especie']:5} {r['confianca']:.2f}  "
                  f"{im.size} -> {r['imagem'].size}  [{r['ms']:.0f} ms]")
        else:
            print(f"  {nome:28} nenhum animal encontrado  [{r['ms']:.0f} ms]")
