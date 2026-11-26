"""Cliente do SLM local (Ollama).

Duas funcoes, como manda o CLAUDE.md:

    gerar(prompt)            texto curto
    extrair(texto, esquema)  saida estruturada em JSON

A segunda e o caso de uso canonico de SLM e a entrega mais concreta de ESW489:
converte "ele e meio peludo, sabe" em {"pelagem": "media"}.

POR QUE JSON SCHEMA, E NAO "responda apenas JSON"
-------------------------------------------------
O Ollama aceita um JSON Schema no campo `format` da API e restringe a
amostragem aos tokens que mantem a saida valida. Isso e diferente de pedir JSON
no prompt e torcer: aqui a saida *nao pode* sair do esquema. Sem isso seria
preciso um pos-processamento frageis com regex, que quebra na bancada.

LATENCIA
--------
A primeira chamada carrega o modelo e leva ~75 s nesta maquina; as seguintes,
menos de 1 s. Por isso existe `aquecer()`, que a api.py chama na inicializacao
-- senao o primeiro visitante do evento paga essa espera.
"""

from __future__ import annotations

import json
import os
import time

import requests

# Dois modelos pequenos, escolhidos para caber junto com a CNN numa GPU de 6 GB.
# O Llama 3.2 tem portugues entre os idiomas oficialmente suportados, o que
# importa porque todo texto da interface e em portugues.
MODELO = os.environ.get("PET_SLM_MODELO", "llama3.2:3b")
MODELO_EMBED = os.environ.get("PET_SLM_EMBED", "nomic-embed-text")
BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# Timeouts: (conexao, leitura). A leitura e generosa porque a primeira chamada
# carrega o modelo; quem nao quiser esperar usa `disponivel()` antes.
TIMEOUT_CURTO = (3, 20)
TIMEOUT_CARGA = (3, 180)

# Guardrail que acompanha todo prompt de linguagem do projeto. Esta aqui, e nao
# so na api.py, para que qualquer chamada ao SLM herde a restricao 1 do
# CLAUDE.md mesmo que alguem esqueca de repeti-la.
# Instrucao em forma POSITIVA de proposito. A versao anterior listava os
# termos proibidos ("nunca diga ascendencia genetica") e o modelo de 3B os
# repetia ao negar -- apareceu na tela "nao conseguiu identificar a
# ascendencia genetica". Citar a palavra proibida a torna disponivel.
SISTEMA_PADRAO = (
    "Voce ajuda numa feira de ciencias a falar sobre fotos de pets. "
    "A rede neural compara a foto com 37 racas puras e mede o quanto o animal "
    "SE PARECE com cada uma. "
    "Fale sempre em termos de aparencia e semelhanca visual: 'se parece com', "
    "'lembra', 'tem tracos de'. "
    "Responda em portugues do Brasil, em no maximo duas frases curtas."
)

# Rede de seguranca, porque prompt nao e garantia. A restricao 1 do CLAUDE.md
# e a mais importante do projeto, e um modelo pequeno escorrega. Se o texto
# gerado contiver qualquer destes, quem chamou usa o texto de reserva.
#
# Sem acento: o SLM as vezes responde sem acentuacao, e comparamos
# normalizado.
TERMOS_PROIBIDOS = (
    "ascendencia", "ancestralidade", "ancestral", "genetic", "dna",
    "composicao", "linhagem", "pedigree", "puro sangue", "cruzamento",
    "heranca", "sangue de",
)


def _sem_acento(texto: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


def violou_guardrail(texto: str) -> str | None:
    """Devolve o termo proibido encontrado, ou None.

    Exposta porque a api.py e os testes checam o mesmo criterio -- a regra
    mora num lugar so.
    """
    limpo = _sem_acento(texto)
    for termo in TERMOS_PROIBIDOS:
        if termo in limpo:
            return termo
    return None


class SLMIndisponivel(RuntimeError):
    """Ollama fora do ar ou modelo ausente. Quem chama decide o fallback."""


def disponivel(timeout: float = 2.0) -> bool:
    """True se o Ollama responde E o modelo de geracao esta baixado."""
    try:
        r = requests.get(f"{BASE}/api/tags", timeout=timeout)
        r.raise_for_status()
        nomes = [m["name"] for m in r.json().get("models", [])]
    except (requests.RequestException, ValueError, KeyError):
        return False
    # O Ollama responde "llama3.2:3b" ou "llama3.2:3b-instruct-q4_K_M" conforme
    # a tag; comparar pelo prefixo evita falso negativo.
    base = MODELO.split(":")[0]
    return any(n == MODELO or n.startswith(base) for n in nomes)


def _post(rota: str, carga: dict, timeout) -> dict:
    try:
        r = requests.post(f"{BASE}{rota}", json=carga, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise SLMIndisponivel(f"Ollama nao respondeu em {rota}: {e}") from e


def gerar(prompt: str, sistema: str | None = None, temperatura: float = 0.3,
          max_tokens: int = 80) -> tuple[str, dict]:
    """Texto curto. Devolve (texto, telemetria).

    A telemetria vem do proprio Ollama (`eval_count`), entao a contagem de
    tokens do GET /metricas e medida, nao estimada.
    """
    t0 = time.perf_counter()
    d = _post("/api/generate", {
        "model": MODELO,
        "prompt": prompt,
        "system": sistema or SISTEMA_PADRAO,
        "stream": False,
        "options": {"temperature": temperatura, "num_predict": max_tokens},
    }, TIMEOUT_CARGA)
    return d.get("response", "").strip(), {
        "latencia_ms": (time.perf_counter() - t0) * 1000,
        "tokens_entrada": d.get("prompt_eval_count", 0),
        "tokens_saida": d.get("eval_count", 0),
        "modelo": MODELO,
    }


def extrair(texto: str, esquema: dict, instrucao: str | None = None) -> tuple[dict, dict]:
    """Converte texto livre do visitante em JSON que obedece `esquema`.

    `esquema` e um JSON Schema. Exemplo:

        {"type": "object",
         "properties": {"pelagem": {"type": "string",
                                    "enum": ["curta", "media", "longa"]}},
         "required": ["pelagem"]}
    """
    props = esquema.get("properties", {})
    campos = ", ".join(props.keys()) or "os campos pedidos"

    # O `format` garante que a saida CABE no esquema, mas nao que a escolha
    # seja a certa: um modelo de 3B que nao ve os valores possiveis no prompt
    # chuta. Listar os enums aqui corrigiu "juba enorme" -> media para longa.
    opcoes = []
    for nome, definicao in props.items():
        if "enum" in definicao:
            opcoes.append(f"- {nome}: escolha entre {' | '.join(definicao['enum'])}")
        elif definicao.get("description"):
            opcoes.append(f"- {nome}: {definicao['description']}")
    bloco_opcoes = ("\nValores possiveis:\n" + "\n".join(opcoes)) if opcoes else ""

    prompt = (
        f"{instrucao or f'Extraia {campos} do texto abaixo.'}"
        f"{bloco_opcoes}\n\n"
        f"Texto do visitante: \"{texto}\"\n\n"
        "Escolha o valor que melhor descreve o que o visitante disse. "
        "Se o texto nao disser nada sobre um campo, escolha o valor mais comum."
    )
    t0 = time.perf_counter()
    d = _post("/api/generate", {
        "model": MODELO,
        "prompt": prompt,
        # temperatura 0: extracao e tarefa fechada, nao ha nada a inventar.
        "options": {"temperature": 0, "num_predict": 120},
        "format": esquema,
        "stream": False,
    }, TIMEOUT_CARGA)
    bruto = d.get("response", "").strip()
    try:
        dados = json.loads(bruto)
    except json.JSONDecodeError as e:
        # Nao deveria acontecer com `format` preenchido, mas se o Ollama for
        # trocado por outro servidor a falha tem que ser explicita.
        raise SLMIndisponivel(f"resposta nao e JSON valido: {bruto[:120]}") from e
    return dados, {
        "latencia_ms": (time.perf_counter() - t0) * 1000,
        "tokens_entrada": d.get("prompt_eval_count", 0),
        "tokens_saida": d.get("eval_count", 0),
        "modelo": MODELO,
    }


def embeddings(textos: list[str]) -> list[list[float]]:
    """Vetores do `nomic-embed-text`, para o rag.py."""
    d = _post("/api/embed", {"model": MODELO_EMBED, "input": textos}, TIMEOUT_CARGA)
    vetores = d.get("embeddings")
    if not vetores:
        raise SLMIndisponivel("resposta de embeddings vazia")
    return vetores


def aquecer() -> bool:
    """Carrega o modelo na memoria. Chamado pela api.py na inicializacao.

    Sem isso o primeiro visitante do evento espera ~75 s pela resposta.
    """
    try:
        _post("/api/generate", {"model": MODELO, "prompt": "ok",
                                "stream": False, "options": {"num_predict": 1}},
              TIMEOUT_CARGA)
        return True
    except SLMIndisponivel:
        return False


if __name__ == "__main__":
    print(f"servidor: {BASE}")
    print(f"modelo:   {MODELO}")
    if not disponivel():
        print("\nOllama fora do ar ou modelo ausente. Instale e baixe com:")
        print("    winget install Ollama.Ollama")
        print(f"    ollama pull {MODELO}")
        print(f"    ollama pull {MODELO_EMBED}")
        raise SystemExit(1)

    print("\n1. aquecendo (a primeira chamada carrega o modelo)")
    t0 = time.perf_counter()
    aquecer()
    print(f"   pronto em {time.perf_counter() - t0:.1f} s")

    print("\n2. geracao de texto curto")
    texto, tel = gerar("A rede achou que a foto se parece muito com um Pug. "
                       "Comente em uma frase.")
    print(f"   {texto}")
    print(f"   [{tel['tokens_saida']} tokens, {tel['latencia_ms']:.0f} ms]")

    print("\n3. extracao estruturada (o caso canonico de ESW489)")
    esquema = {"type": "object",
               "properties": {"pelagem": {"type": "string",
                                          "enum": ["curta", "media", "longa"]}},
               "required": ["pelagem"]}
    for frase in ["ele e meio peludo, sabe, nao e curtinho mas nao arrasta no chao",
                  "pelo bem curtinho, quase raspado",
                  "tem uma juba enorme, solta pelo o ano todo"]:
        dados, tel = extrair(frase, esquema)
        print(f"   \"{frase[:44]}...\"\n     -> {dados}  [{tel['latencia_ms']:.0f} ms]")

    print("\n4. embeddings")
    v = embeddings(["cao de porte grande", "gato de pelo curto"])
    print(f"   {len(v)} vetores de dimensao {len(v[0])}")
