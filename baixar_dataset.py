"""Monta o Oxford-IIIT Pet localmente, rapido.

POR QUE ESTE ARQUIVO EXISTE
---------------------------
O `data.py` usa `OxfordIIITPet(download=True)`, que baixa as imagens de
`www.robots.ox.ac.uk`. Aquele servidor limita a banda por conexao: medimos
57 KB/s numa linha de 47 MB/s, o que da mais de tres horas para os 755 MiB.
Ele tambem ignora requisicoes de faixa (`Range`), entao baixar em paralelo
nao ajuda.

As mesmas imagens estao no CDN do Hugging Face, em dois arquivos parquet, a
dezenas de MB/s. As anotacoes (`trainval.txt`, `test.txt`) nao estao lá, mas
sao só 19 MB -- essas valem buscar em Oxford mesmo.

Este script junta as duas fontes e escreve a arvore de diretorios que o
torchvision espera:

    data/oxford-iiit-pet/
        annotations/{trainval.txt, test.txt, list.txt, ...}
        images/<image_id>.jpg

Com a arvore no lugar, `OxfordIIITPet` considera o dataset ja baixado e nao
toca na rede. Nada no `data.py` precisa mudar.

Uso:
    python baixar_dataset.py
    python baixar_dataset.py --manter-parquet   # nao apaga os 790 MB baixados
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import requests

# Espelho "thor" de Oxford: mesma origem, mas tres vezes mais rapido que o
# www.robots.ox.ac.uk que o torchvision usa.
URL_ANOTACOES = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"
MD5_ANOTACOES = "95a8c909bbe2e81eed6a22bccdf3f68f"  # o mesmo que o torchvision confere

REPO_HF = "https://huggingface.co/datasets/timm/oxford-iiit-pet/resolve/main/data"
PARQUETS = ("train-00000-of-00001.parquet", "test-00000-of-00001.parquet")

# Numeros oficiais do dataset. Servem de verificacao: se o resultado nao
# bater, algo veio incompleto e e melhor falhar aqui do que no meio do treino.
TOTAL_IMAGENS = 7349
TOTAL_TRAINVAL = 3680
TOTAL_TESTE = 3669


def _baixar(url: str, destino: Path, rotulo: str) -> Path:
    """Baixa com streaming e barra de progresso numa linha."""
    with requests.get(url, stream=True, timeout=(10, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        baixado = 0
        with destino.open("wb") as fh:
            for bloco in r.iter_content(chunk_size=1 << 20):
                fh.write(bloco)
                baixado += len(bloco)
                if total:
                    pct = 100 * baixado / total
                    print(f"\r  {rotulo}: {pct:5.1f}%  "
                          f"({baixado / 1e6:.0f}/{total / 1e6:.0f} MB)", end="", flush=True)
        print(f"\r  {rotulo}: pronto ({baixado / 1e6:.0f} MB)" + " " * 20)
    return destino


def _md5(caminho: Path) -> str:
    h = hashlib.md5()
    with caminho.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _escrever_imagens(parquets: list[Path], pasta_imagens: Path) -> int:
    """Grava image.bytes de cada linha como <image_id>.jpg.

    O parquet do Hugging Face guarda o JPEG original em `image.bytes` e o nome
    em `image_id` (ex.: "Maine_Coon_204"). O torchvision monta o caminho de
    cada foto como images/<image_id>.jpg, entao basta gravar os bytes crus --
    nao e preciso decodificar nem reencodar a imagem.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("\nFalta o pyarrow para ler os parquet. Instale com:\n"
              "    pip install pyarrow\n"
              "Ou, se preferir o caminho lento (mais de 3 horas), use:\n"
              "    python data.py", file=sys.stderr)
        raise SystemExit(1)

    pasta_imagens.mkdir(parents=True, exist_ok=True)
    escritas = 0
    for src in parquets:
        arquivo = pq.ParquetFile(src)
        for lote in arquivo.iter_batches(batch_size=256, columns=["image", "image_id"]):
            d = lote.to_pydict()
            for imagem, iid in zip(d["image"], d["image_id"]):
                (pasta_imagens / f"{iid}.jpg").write_bytes(imagem["bytes"])
                escritas += 1
            print(f"\r  imagens escritas: {escritas}", end="", flush=True)
    print(f"\r  imagens escritas: {escritas}" + " " * 20)
    return escritas


def montar(raiz: str = "./data", manter_parquet: bool = False) -> Path:
    destino = Path(raiz) / "oxford-iiit-pet"
    pasta_imagens = destino / "images"
    pasta_anotacoes = destino / "annotations"

    # Mesma condicao que o torchvision usa para decidir se ja baixou.
    if pasta_imagens.is_dir() and pasta_anotacoes.is_dir():
        qtd = len(list(pasta_imagens.glob("*.jpg")))
        if qtd == TOTAL_IMAGENS:
            print(f"dataset ja esta em {destino} ({qtd} imagens) -- nada a fazer")
            return destino
        print(f"aviso: {destino} tem {qtd} imagens, esperado {TOTAL_IMAGENS}. Remontando.")

    destino.mkdir(parents=True, exist_ok=True)
    temporaria = Path(tempfile.mkdtemp(prefix="petdetetive_"))

    try:
        print("1/3 anotacoes (19 MB, servidor de Oxford)")
        tar_anotacoes = _baixar(URL_ANOTACOES, temporaria / "annotations.tar.gz", "anotacoes")
        obtido = _md5(tar_anotacoes)
        if obtido != MD5_ANOTACOES:
            raise RuntimeError(f"MD5 das anotacoes nao confere: {obtido} != {MD5_ANOTACOES}")
        # filter="data" recusa membros com caminho absoluto ou ".." -- o arquivo
        # vem de fora, nao custa nada extrair com cuidado.
        with tarfile.open(tar_anotacoes) as tf:
            tf.extractall(destino, filter="data")

        print("2/3 imagens (790 MB, CDN do Hugging Face)")
        baixados = [
            _baixar(f"{REPO_HF}/{nome}", temporaria / nome, nome.split("-")[0])
            for nome in PARQUETS
        ]

        print("3/3 gravando as imagens")
        escritas = _escrever_imagens(baixados, pasta_imagens)
        if escritas != TOTAL_IMAGENS:
            raise RuntimeError(f"escritas {escritas} imagens, esperado {TOTAL_IMAGENS}")

        for nome, esperado in (("trainval.txt", TOTAL_TRAINVAL), ("test.txt", TOTAL_TESTE)):
            linhas = (pasta_anotacoes / nome).read_text(encoding="utf-8").strip().splitlines()
            if len(linhas) != esperado:
                raise RuntimeError(f"{nome} tem {len(linhas)} linhas, esperado {esperado}")

        if manter_parquet:
            guardados = destino / "_parquet"
            guardados.mkdir(exist_ok=True)
            for p in baixados:
                shutil.move(str(p), guardados / p.name)
            print(f"parquet guardados em {guardados}")

        print(f"\npronto: {destino}")
        print(f"  {escritas} imagens | trainval {TOTAL_TRAINVAL} | teste {TOTAL_TESTE}")
        print("\nproximo passo:  python data.py")
        return destino
    finally:
        shutil.rmtree(temporaria, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--manter-parquet", action="store_true",
                    help="guarda os parquet baixados em vez de apagar (790 MB)")
    args = ap.parse_args()
    montar(args.data_root, args.manter_parquet)
