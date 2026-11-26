"""API do Pet Detetive.

    uvicorn api:app --reload

O modelo e carregado UMA vez na inicializacao, junto com o indice do RAG, e o
SLM local e aquecido no mesmo momento. Sem esse aquecimento o primeiro
visitante do evento espera uns 75 segundos pela primeira frase, porque e nessa
hora que o Ollama carrega o modelo na memoria.

A FOTO NAO TOCA O DISCO
-----------------------
`POST /classificar` le o upload para `BytesIO` e abre com `PIL.Image`. Nao ha
`arquivo.save(...)`, nem pasta de temporarios, nem cache. Depois da resposta a
imagem e coletada pelo garbage collector e deixa de existir no processo -- a
sessao guarda o vetor de probabilidades, nao a foto (restricoes 2 e 6 do
CLAUDE.md). Ao fim de 30/11 o notebook nao tem uma pasta com fotos dos pets de
dezenas de desconhecidos.
"""

from __future__ import annotations

import io
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

import detector
import llm
import slm
from agente import Agente
from roteador import Roteador

WEB = Path(__file__).parent / "web"

# 12 MB: uma foto de celular moderna passa de 5 MB, e e melhor recusar cedo
# com mensagem clara do que estourar memoria no meio do evento.
TAMANHO_MAXIMO = 12 * 1024 * 1024

estado: dict = {}


@asynccontextmanager
async def ciclo(app: FastAPI):
    t0 = time.perf_counter()
    print("carregando modelo e indice...")
    roteador = Roteador()
    estado["agente"] = Agente(roteador=roteador)
    estado["roteador"] = roteador
    print(f"  modelo mobilenet, {len(estado['agente'].class_names)} classes")
    print(f"  RAG por {estado['agente'].indice.rota}")

    # Carrega o detector aqui, e nao na primeira foto: os pesos sao baixados
    # do torchvision na primeira vez (~14 MB) e nao e na frente do visitante
    # que isso deve acontecer.
    baralho = _baralho()
    estado["baralho"] = baralho
    estado["baralho_do_dataset"] = bool(baralho) and "oxford" in str(baralho[0])
    estado["pilha"] = []
    fonte = "dataset" if estado["baralho_do_dataset"] else "web/exemplos"
    print(f"  sorteio: {len(baralho)} fotos ({fonte})")

    estado["detector"] = detector.disponivel()
    print(f"  detector de pet: {'pronto' if estado['detector'] else 'INDISPONIVEL (usa a foto inteira)'}")

    # Aquece o SLM. Bloqueia a inicializacao de proposito: e melhor a API
    # demorar para subir do que o primeiro visitante esperar.
    print("aquecendo o SLM local...")
    estado["slm_quente"] = slm.aquecer()
    print(f"  {'pronto' if estado['slm_quente'] else 'Ollama fora do ar (modo degradado)'}")
    print(f"  nuvem: {'disponivel' if llm.disponivel() else 'sem chave (perfil cai no SLM)'}")
    print(f"pronto em {time.perf_counter() - t0:.1f}s")
    yield
    estado.clear()


app = FastAPI(title="Pet Detetive", lifespan=ciclo)


class Resposta(BaseModel):
    sessao: str
    texto: str


class PedidoPerfil(BaseModel):
    sessao: str
    raca: str


@app.get("/")
def raiz():
    indice = WEB / "index.html"
    if not indice.exists():
        return JSONResponse({"erro": "web/index.html nao encontrado"}, status_code=404)
    return FileResponse(indice)


@app.get("/saude")
def saude():
    a = estado.get("agente")
    if a is None:
        raise HTTPException(503, "ainda inicializando")
    return {
        "modelo": a.nome_modelo,
        "dispositivo": str(next(a.modelo.parameters()).device),
        "classes": len(a.class_names),
        "detector_disponivel": estado.get("detector", False),
        "slm_disponivel": slm.disponivel(),
        "slm_quente": estado.get("slm_quente", False),
        "nuvem_disponivel": llm.disponivel(),
        "rag": a.indice.rota,
    }


@app.post("/classificar")
async def classificar(foto: UploadFile = File(...), sessao: str | None = None):
    bruto = await foto.read()
    if not bruto:
        raise HTTPException(400, "arquivo vazio")
    if len(bruto) > TAMANHO_MAXIMO:
        raise HTTPException(413, f"foto maior que {TAMANHO_MAXIMO // (1024 * 1024)} MB")

    try:
        # BytesIO, nunca disco. Veja o cabecalho do modulo.
        imagem = Image.open(io.BytesIO(bruto))
        imagem.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "nao consegui ler essa imagem")

    a = estado.get("agente")
    if a is None:
        raise HTTPException(503, "ainda inicializando")
    return a.ver_foto(imagem, sessao)


@app.post("/responder")
def responder(corpo: Resposta):
    a = estado.get("agente")
    if a is None:
        raise HTTPException(503, "ainda inicializando")
    try:
        return a.responder(corpo.sessao, corpo.texto)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/perfil")
def perfil(corpo: PedidoPerfil):
    a = estado.get("agente")
    if a is None:
        raise HTTPException(503, "ainda inicializando")
    if corpo.raca not in a.class_names:
        raise HTTPException(404, f"raca desconhecida: {corpo.raca}")
    return a.perfil(corpo.sessao, corpo.raca)


@app.get("/metricas")
def metricas():
    r = estado.get("roteador")
    if r is None:
        raise HTTPException(503, "ainda inicializando")
    return r.metricas.to_dict()


EXTENSOES = {".jpg", ".jpeg", ".png", ".webp"}


def _baralho() -> list[Path]:
    """Imagens sorteaveis, listadas uma vez na inicializacao.

    Prefere o dataset inteiro (7349 fotos, 37 racas) e cai em web/exemplos/
    quando ele nao foi baixado. Listar 7349 arquivos a cada clique seria
    desperdicio numa bancada movimentada.
    """
    dataset = Path("data/oxford-iiit-pet/images")
    if dataset.is_dir():
        fotos = [p for p in dataset.iterdir() if p.suffix.lower() in EXTENSOES]
        if fotos:
            return fotos
    pasta = WEB / "exemplos"
    if pasta.is_dir():
        return [p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES]
    return []


@app.get("/exemplos")
def exemplos():
    """Quantas fotos ha para sortear, e de onde vem.

    A interface usa isso so para decidir se mostra o botao de sorteio: sem
    foto nenhuma ele some, em vez de dar erro na frente do visitante.
    """
    fotos = estado.get("baralho", [])
    return {"quantidade": len(fotos),
            "fonte": "dataset" if estado.get("baralho_do_dataset") else "pasta"}


@app.get("/exemplos/aleatorio")
def exemplo_aleatorio():
    """Uma foto qualquer do banco.

    Embaralha e consome, em vez de sortear a cada chamada: assim o visitante
    nao ve a mesma foto duas vezes seguidas, que na bancada pareceria defeito.
    """
    fotos = estado.get("baralho", [])
    if not fotos:
        raise HTTPException(404, "nenhuma foto de exemplo disponivel")

    pilha = estado.get("pilha") or []
    if not pilha:
        pilha = list(range(len(fotos)))
        random.shuffle(pilha)
    indice = pilha.pop()
    estado["pilha"] = pilha

    caminho = fotos[indice]

    # A raca verdadeira esta no nome do arquivo do Oxford-IIIT Pet
    # ("Maine_Coon_12.jpg", "american_pit_bull_terrier_100.jpg"). O arquivo
    # nao preserva a caixa, entao casamos com o nome canonico das classes --
    # senao a tela mostraria "era um american pit bull terrier".
    bruto = caminho.stem.rsplit("_", 1)[0].replace("_", " ")
    canonico = ""
    a = estado.get("agente")
    if a is not None:
        canonico = next((n for n in a.class_names if n.lower() == bruto.lower()), "")

    cabecalhos = {"Cache-Control": "no-store"}
    if canonico or bruto:
        cabecalhos["X-Raca-Verdadeira"] = canonico or bruto
    return FileResponse(caminho, headers=cabecalhos)


@app.get("/exemplos/{nome}")
def exemplo(nome: str):
    # Path(nome).name descarta qualquer "../" -- a rota serve um diretorio
    # publico, mas nao precisa servir o resto do disco junto.
    caminho = WEB / "exemplos" / Path(nome).name
    if not caminho.is_file():
        raise HTTPException(404, "exemplo nao encontrado")
    return FileResponse(caminho)


if __name__ == "__main__":
    import argparse
    import socket

    import uvicorn

    ap = argparse.ArgumentParser(description="Sobe a API do Pet Detetive")
    ap.add_argument("--porta", type=int, default=8000)
    ap.add_argument("--rede", action="store_true",
                    help="aceita celulares da mesma rede (escuta em 0.0.0.0)")
    args = ap.parse_args()

    if args.rede:
        # Descobre o IP da maquina na rede so para imprimir na tela.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except OSError:
            ip = "<ip-da-maquina>"
        finally:
            s.close()
        print(f"\nno celular, abra:  http://{ip}:{args.porta}")
        # A camera ao vivo NAO funciona por IP: getUserMedia exige contexto
        # seguro (HTTPS ou localhost). Pelo celular o caminho e "Escolher uma
        # foto", cujo seletor ja oferece a camera no menu do sistema.
        print("   a camera ao vivo so funciona em localhost; pelo celular use")
        print("   'Escolher uma foto', que abre a camera pelo menu do sistema")
        print("   a foto trafega pela rede local, e nunca sai dela\n")

    uvicorn.run(app, host="0.0.0.0" if args.rede else "127.0.0.1", port=args.porta)
