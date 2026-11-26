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


@app.get("/exemplos")
def exemplos():
    """Fotos de exemplo, para visitantes sem pet.

    Le o que existir em web/exemplos/. A galeria some sozinha se a pasta
    estiver vazia, em vez de mostrar imagens quebradas na bancada.
    """
    pasta = WEB / "exemplos"
    if not pasta.is_dir():
        return {"exemplos": []}
    nomes = sorted(p.name for p in pasta.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    return {"exemplos": [f"/exemplos/{n}" for n in nomes]}


@app.get("/exemplos/{nome}")
def exemplo(nome: str):
    # Path(nome).name descarta qualquer "../" -- a rota serve um diretorio
    # publico, mas nao precisa servir o resto do disco junto.
    caminho = WEB / "exemplos" / Path(nome).name
    if not caminho.is_file():
        raise HTTPException(404, "exemplo nao encontrado")
    return FileResponse(caminho)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
