"""Cliente do LLM em nuvem. Perfil longo e perguntas abertas.

E a UNICA parte do sistema que fala com a internet, e por isso a mais
cuidadosa. Tres regras, nessa ordem de importancia:

1. NUNCA RECEBE IMAGEM. O que sobe e o nome da raca, a pergunta digitada e os
   trechos da base de conhecimento local -- texto que ja estava no repositorio
   antes do visitante chegar. A foto, o tensor e o mapa de atencao nao saem da
   maquina. Esse e o argumento central de privacidade (restricoes 2 e 6 do
   CLAUDE.md). Toda funcao publica daqui recebe `str`, nunca `bytes`, e nao
   existe caminho neste modulo que aceite PIL.Image.

2. TIMEOUT CURTO E FALLBACK. Sem chave, sem rede ou com a API lenta, o texto e
   gerado pelo SLM local. O visitante nunca ve um erro, so um texto mais
   simples.

3. UMA CHAMADA POR VISITANTE NO CAMINHO PADRAO. Legenda, pergunta de
   refinamento e extracao sao sempre locais (restricao 4).

VARIOS PROVEDORES
-----------------
A chave e identificada pelo prefixo e cada familia tem seu jeito de falar:

    sk-ant-...   Anthropic    SDK oficial `anthropic`
    AIza...      Gemini       REST propria do Google
    sk-or-...    OpenRouter   |
    gsk_...      Groq         |  todas falam o mesmo dialeto
    sk-proj-...  OpenAI       |  /chat/completions, entao um cliente
    sk-...       OpenAI       |  HTTP cobre as quatro

Para Anthropic usamos o SDK oficial. Para as compativeis com OpenAI usamos
`requests` direto, em vez de instalar mais um SDK: o endpoint e estavel,
documentado, e o projeto ja depende de `requests` para falar com o Ollama.
"""

from __future__ import annotations

import os
import time

import requests

# O .env e opcional: sem ele o modulo cai no SLM local em vez de quebrar.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Timeout curto de proposito: numa bancada, dez segundos de silencio ja e
# tempo demais. Estourou, cai no SLM local, que responde em poucos segundos.
TIMEOUT_S = float(os.environ.get("PET_LLM_TIMEOUT", "20"))

# A ordem importa: "sk-ant-", "sk-or-" e "sk-proj-" precisam ser testados
# antes do "sk-" solto, senao tudo vira OpenAI.
PROVEDORES: list[dict] = [
    {"nome": "anthropic", "rotulo": "Anthropic (Claude)", "prefixos": ("sk-ant-",),
     "tipo": "anthropic", "modelo": "claude-opus-5-5"},
    {"nome": "openrouter", "rotulo": "OpenRouter", "prefixos": ("sk-or-",),
     "tipo": "openai", "base": "https://openrouter.ai/api/v1",
     "modelo": "anthropic/claude-sonnet-4.5"},
    {"nome": "groq", "rotulo": "Groq", "prefixos": ("gsk_",),
     "tipo": "openai", "base": "https://api.groq.com/openai/v1",
     "modelo": "llama-3.3-70b-versatile"},
    # "AQ." e o formato novo do AI Studio; "AIza" e o antigo. Os dois valem.
    {"nome": "gemini", "rotulo": "Google Gemini", "prefixos": ("AQ.", "AIza"),
     "tipo": "gemini", "base": "https://generativelanguage.googleapis.com/v1beta",
     "modelo": "gemini-3.5-flash-lite"},
    {"nome": "openai", "rotulo": "OpenAI", "prefixos": ("sk-proj-", "sk-"),
     "tipo": "openai", "base": "https://api.openai.com/v1",
     "modelo": "gpt-4o-mini"},
]

SISTEMA = (
    "Voce escreve perfis curtos de racas de caes e gatos para uma feira de "
    "ciencias, a partir de uma base de conhecimento fornecida.\n\n"
    "REGRAS:\n"
    "- Use SOMENTE os dados fornecidos. Nao acrescente fatos que nao estejam "
    "neles: a base existe justamente para impedir invencao.\n"
    "- A rede neural compara a foto com 37 racas puras e mede SEMELHANCA "
    "VISUAL. Escreva sempre 'se parece com' ou 'lembra'.\n"
    "- Portugues do Brasil, tom sobrio e direto, sem linguagem de catalogo de "
    "canil ('companheiro leal e amoroso' e o que nao queremos).\n"
    "- Tres paragrafos curtos, no maximo 140 palavras no total."
)

SISTEMA_PERGUNTA = (
    "Voce responde perguntas de visitantes de uma feira de ciencias sobre "
    "racas de caes e gatos, a partir de uma base de conhecimento fornecida.\n\n"
    "REGRAS:\n"
    "- Responda a pergunta DIRETAMENTE, na primeira frase.\n"
    "- Use os dados fornecidos como base. Se a pergunta exigir algo que nao "
    "esta neles, diga o que a base sustenta e seja explicito sobre o que e "
    "inferencia sua.\n"
    "- Se a pergunta nao for sobre o animal, diga isso em uma frase.\n"
    "- A rede mede SEMELHANCA VISUAL com 37 racas puras. Fale em termos de "
    "aparencia: 'se parece com', 'lembra'.\n"
    "- Portugues do Brasil, tom sobrio. No maximo 4 frases."
)


class SemNuvem(RuntimeError):
    """Sem chave, sem rede ou a API recusou. Quem chama usa o SLM local."""


# ----------------------------------------------------------------------
# Identificacao da chave
# ----------------------------------------------------------------------

def _chave() -> str | None:
    return (limpar(os.environ.get("PET_LLM_CHAVE"))
            or limpar(os.environ.get("ANTHROPIC_API_KEY"))
            or None)


def limpar(chave: str | None) -> str:
    """Tira espacos e aspas que vem junto ao colar de um painel web."""
    if not chave:
        return ""
    return chave.strip().strip('"').strip("'").strip()


# Erros de credencial que valem um aviso especifico, porque o erro cru do
# provedor ("API key not valid") nao diz o que fazer.
PISTAS = (
    ("ya29.", "isso e um token OAuth do Google, nao uma chave de API. "
              "Tokens OAuth expiram em cerca de uma hora e usam outro "
              "cabecalho. Pegue uma chave em aistudio.google.com/apikey -- "
              "ela comeca com AIzaSy."),
    ("Bearer ", "tire o 'Bearer ' da frente: cole so a credencial."),
    ("sk_live_", "isso e uma chave do Stripe, nao de um provedor de LLM."),
    ("ghp_", "isso e um token do GitHub, nao de um provedor de LLM."),
    ("hf_", "isso e um token do Hugging Face. O projeto nao usa a Inference "
            "API deles; use Anthropic, OpenAI, Groq, OpenRouter ou Gemini."),
)


def pista(chave: str | None) -> str:
    """Aviso util quando a credencial colada e do tipo errado."""
    limpa = limpar(chave)
    for prefixo, texto in PISTAS:
        if limpa.startswith(prefixo):
            return texto
    return ""


def por_nome(nome: str | None) -> dict | None:
    if not nome:
        return None
    return next((p for p in PROVEDORES if p["nome"] == nome), None)


def identificar(chave: str | None, provedor: str | None = None) -> dict | None:
    """Qual provedor essa chave e.

    `provedor` explicito vence o prefixo. Isso existe porque prefixo nao e
    contrato: o Google mudou o formato das chaves entre produtos, e um usuario
    com chave legitima ficaria travado. Quando a deteccao falha, a interface
    oferece a escolha manual em vez de recusar.
    """
    escolhido = por_nome(provedor)
    if escolhido:
        return escolhido
    limpa = limpar(chave)
    if not limpa:
        return None
    for p in PROVEDORES:
        if any(limpa.startswith(pre) for pre in p["prefixos"]):
            return p
    return None


def provedor_atual() -> dict | None:
    """Provedor em uso, respeitando um override manual de PET_LLM_PROVEDOR."""
    chave = _chave()
    if not chave:
        return None
    return identificar(chave, os.environ.get("PET_LLM_PROVEDOR"))


def modelo_atual() -> str:
    p = provedor_atual()
    return os.environ.get("PET_LLM_MODELO") or (p["modelo"] if p else "")


def mascarar(chave: str | None) -> str:
    """Nunca devolvemos a chave inteira para a interface."""
    if not chave:
        return ""
    limpa = chave.strip()
    return f"{limpa[:7]}…{limpa[-4:]}" if len(limpa) > 14 else "…"


def disponivel() -> bool:
    """True se ha chave reconhecida e o cliente dela existe.

    Nao faz chamada de rede: o GET /saude consulta isso a cada carga de
    pagina, e nao vale gastar uma requisicao por visita.
    """
    p = provedor_atual()
    if p is None:
        return False
    if p["tipo"] == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
    return True


# ----------------------------------------------------------------------
# Um transporte por familia de API
# ----------------------------------------------------------------------

def _via_anthropic(p: dict, chave: str, sistema: str,
                   mensagens: list[dict], max_tokens: int) -> tuple[str, dict]:
    import anthropic

    cliente = anthropic.Anthropic(api_key=chave, timeout=TIMEOUT_S, max_retries=1)
    try:
        r = cliente.beta.messages.create(
            model=modelo_atual() or p["modelo"],
            max_tokens=max_tokens,
            # Tarefa curta a partir de dados prontos: nao pede raciocinio
            # profundo, e esforco baixo corta a latencia.
            output_config={"effort": "low"},
            # Se o classificador de seguranca recusar, a propria API refaz a
            # chamada noutro modelo em vez de devolver resposta vazia.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=sistema,
            messages=mensagens,
        )
    except anthropic.APIError as e:
        raise SemNuvem(f"{p['rotulo']}: {e}") from e
    except Exception as e:
        raise SemNuvem(f"{p['rotulo']}: falha de rede: {e}") from e

    if r.stop_reason == "refusal":
        raise SemNuvem(f"{p['rotulo']}: a API recusou a solicitacao")
    texto = "".join(b.text for b in r.content if b.type == "text").strip()
    return texto, {"tokens_entrada": r.usage.input_tokens,
                   "tokens_saida": r.usage.output_tokens}


def _via_openai(p: dict, chave: str, sistema: str,
                mensagens: list[dict], max_tokens: int) -> tuple[str, dict]:
    """Dialeto /chat/completions: OpenAI, Groq, OpenRouter e compativeis."""
    base = os.environ.get("PET_LLM_BASE") or p["base"]
    corpo = {
        "model": modelo_atual() or p["modelo"],
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": sistema}] + mensagens,
    }
    try:
        r = requests.post(f"{base}/chat/completions", json=corpo,
                          headers={"Authorization": f"Bearer {chave}"},
                          timeout=TIMEOUT_S)
        if r.status_code >= 400:
            raise SemNuvem(f"{p['rotulo']}: HTTP {r.status_code} — {r.text[:160]}")
        d = r.json()
    except requests.RequestException as e:
        raise SemNuvem(f"{p['rotulo']}: falha de rede: {e}") from e

    escolhas = d.get("choices") or []
    texto = (escolhas[0].get("message", {}).get("content") or "").strip() if escolhas else ""
    uso = d.get("usage") or {}
    return texto, {"tokens_entrada": uso.get("prompt_tokens", 0),
                   "tokens_saida": uso.get("completion_tokens", 0)}


def listar_modelos(chave: str, provedor: str = "", base: str = "") -> dict:
    """Modelos que ESTA chave pode usar, perguntados ao proprio provedor.

    Existe porque o catalogo envelhece mais rapido que o codigo: o padrao do
    Gemini aqui ja ficou obsoleto uma vez. Em vez de alguem adivinhar o nome
    do modelo, a tela pergunta e mostra a lista.
    """
    p = identificar(chave, provedor)
    if p is None:
        return {"ok": False, "erro": "escolha o provedor na lista"}

    alvo = base or p.get("base", "")
    limpa = limpar(chave)
    try:
        if p["tipo"] == "gemini":
            nomes = _modelos_gemini(alvo, limpa)
        elif p["tipo"] == "openai":
            r = requests.get(f"{alvo}/models",
                             headers={"Authorization": f"Bearer {limpa}"}, timeout=15)
            if r.status_code >= 400:
                return {"ok": False, "erro": f"HTTP {r.status_code} — {r.text[:140]}"}
            nomes = sorted(m.get("id", "") for m in r.json().get("data", []))
        else:
            import anthropic
            cliente = anthropic.Anthropic(api_key=limpa, timeout=15, max_retries=1)
            nomes = [m.id for m in cliente.models.list(limit=30)]
    except Exception as e:
        return {"ok": False, "erro": str(e)[:200]}

    if not nomes:
        return {"ok": False, "erro": "o provedor nao devolveu nenhum modelo"}
    return {"ok": True, "provedor": p["nome"], "rotulo": p["rotulo"],
            "modelos": nomes[:40]}


def _modelos_gemini(base: str, chave: str) -> list[str]:
    """Modelos que essa chave pode usar. So para diagnostico."""
    try:
        r = requests.get(f"{base}/models", headers={"x-goog-api-key": chave},
                         timeout=10)
        if r.status_code >= 400:
            return []
        nomes = []
        for m in r.json().get("models", []):
            if "generateContent" in (m.get("supportedGenerationMethods") or []):
                nomes.append(m.get("name", "").removeprefix("models/"))
        return nomes
    except requests.RequestException:
        return []


def _via_gemini(p: dict, chave: str, sistema: str,
                mensagens: list[dict], max_tokens: int) -> tuple[str, dict]:
    base = os.environ.get("PET_LLM_BASE") or p["base"]
    modelo = modelo_atual() or p["modelo"]
    # O Gemini chama o papel do modelo de "model", nao "assistant".
    conteudo = [{"role": "model" if m["role"] == "assistant" else "user",
                 "parts": [{"text": m["content"]}]} for m in mensagens]
    corpo = {"contents": conteudo,
             "systemInstruction": {"parts": [{"text": sistema}]},
             "generationConfig": {"maxOutputTokens": max_tokens}}
    try:
        r = requests.post(f"{base}/models/{modelo}:generateContent",
                          json=corpo, headers={"x-goog-api-key": chave},
                          timeout=TIMEOUT_S)
        if r.status_code >= 400:
            # Modelo invalido e o erro mais comum depois da chave: o catalogo
            # do Gemini muda e o padrao daqui envelhece. Em vez de repassar
            # um 404 cru, perguntamos a lista e dizemos o que existe.
            if r.status_code == 404 or "model" in r.text.lower():
                disponiveis = _modelos_gemini(base, chave)
                if disponiveis:
                    raise SemNuvem(
                        f"{p['rotulo']}: o modelo '{modelo}' nao existe nessa "
                        f"conta. Disponiveis: {', '.join(disponiveis[:6])}")
            raise SemNuvem(f"{p['rotulo']}: HTTP {r.status_code} — {r.text[:160]}")
        d = r.json()
    except requests.RequestException as e:
        raise SemNuvem(f"{p['rotulo']}: falha de rede: {e}") from e

    candidatos = d.get("candidates") or []
    partes = candidatos[0].get("content", {}).get("parts", []) if candidatos else []
    texto = "".join(x.get("text", "") for x in partes).strip()
    uso = d.get("usageMetadata") or {}
    return texto, {"tokens_entrada": uso.get("promptTokenCount", 0),
                   "tokens_saida": uso.get("candidatesTokenCount", 0)}


_TRANSPORTES = {"anthropic": _via_anthropic, "openai": _via_openai,
                "gemini": _via_gemini}


def _chamar(sistema: str, mensagens: list[dict],
            max_tokens: int = 1000) -> tuple[str, dict]:
    """Despacha para o provedor da chave. Levanta SemNuvem em qualquer falha."""
    p = provedor_atual()
    chave = _chave()
    if p is None or not chave:
        raise SemNuvem("sem chave de API configurada")

    t0 = time.perf_counter()
    texto, uso = _TRANSPORTES[p["tipo"]](p, chave, sistema, mensagens, max_tokens)
    if not texto:
        raise SemNuvem(f"{p['rotulo']}: resposta vazia")
    return texto, {"rota": "nuvem", "provedor": p["nome"],
                   "modelo": modelo_atual() or p["modelo"],
                   "latencia_ms": (time.perf_counter() - t0) * 1000, **uso}


def testar(chave: str, modelo: str = "", provedor: str = "",
           base: str = "") -> dict:
    """Chamada minima, para a tela de configuracao dizer se a chave serve."""
    dica = pista(chave)
    p = identificar(chave, provedor)
    if p is None:
        return {"ok": False,
                "erro": dica or "nao reconheci o prefixo — escolha o provedor na lista"}

    antes = (os.environ.get("PET_LLM_CHAVE"), os.environ.get("PET_LLM_MODELO"),
             os.environ.get("PET_LLM_PROVEDOR"), os.environ.get("PET_LLM_BASE"))
    os.environ["PET_LLM_CHAVE"] = limpar(chave)
    os.environ["PET_LLM_PROVEDOR"] = p["nome"]
    if base:
        os.environ["PET_LLM_BASE"] = base
    if modelo:
        os.environ["PET_LLM_MODELO"] = modelo
    try:
        texto, tel = _chamar("Responda com uma palavra.",
                             [{"role": "user", "content": "Diga: ok"}], max_tokens=16)
        return {"ok": True, "provedor": p["nome"], "rotulo": p["rotulo"],
                "modelo": tel["modelo"], "latencia_ms": round(tel["latencia_ms"]),
                "resposta": texto[:60]}
    except SemNuvem as e:
        # A dica vem primeiro: "API key not valid" nao diz o que fazer, e
        # "isso e um token OAuth, pegue uma chave em ..." diz.
        return {"ok": False, "provedor": p["nome"],
                "erro": f"{dica} ({e})" if dica else str(e)}
    finally:
        for nome, valor in zip(("PET_LLM_CHAVE", "PET_LLM_MODELO",
                                "PET_LLM_PROVEDOR", "PET_LLM_BASE"), antes):
            if valor is None:
                os.environ.pop(nome, None)
            else:
                os.environ[nome] = valor


# ----------------------------------------------------------------------
# As duas tarefas do projeto
# ----------------------------------------------------------------------

def _montar_contexto(raca: str, trechos: list[str]) -> str:
    """Monta o texto que sobe para a nuvem.

    Recebe `str`, nunca bytes nem PIL.Image -- a assinatura e parte da
    garantia de privacidade, nao so uma conveniencia.
    """
    corpo = "\n".join(f"- {t}" for t in trechos)
    return (f"Raca com a qual a foto mais se parece: {raca}\n\n"
            f"Base de conhecimento sobre essa raca:\n{corpo}\n\n"
            f"Escreva o perfil.")


def responder(pergunta: str, trechos: list[str],
              historico: list[dict] | None = None) -> tuple[str, dict]:
    """Pergunta aberta do visitante, respondida pela nuvem.

    O historico e so texto da propria conversa, para o modelo nao repetir o
    que ja disse. Nunca contem imagem.
    """
    corpo = "\n".join(f"- {t}" for t in trechos)
    mensagens = [{"role": t["papel"], "content": t["texto"]}
                 for t in (historico or [])[-6:]]      # 3 trocas de contexto
    mensagens.append({"role": "user",
                      "content": f"Base de conhecimento:\n{corpo}\n\n"
                                 f"Pergunta do visitante: {pergunta}"})
    return _chamar(SISTEMA_PERGUNTA, mensagens)


def _pelo_slm(raca: str, trechos: list[str]) -> tuple[str, dict]:
    """Fallback local. Texto mais simples, mesma garantia de nao inventar."""
    import slm

    texto, tel = slm.gerar(_montar_contexto(raca, trechos), sistema=SISTEMA,
                           temperatura=0.4, max_tokens=220)
    tel["rota"] = "local"
    return texto, tel


def perfil(raca: str, trechos: list[str]) -> tuple[str, dict]:
    """Perfil longo da raca. Tenta a nuvem, cai no SLM local.

    Nao levanta excecao em operacao normal: devolve o melhor texto possivel e
    diz em `telemetria["rota"]` de onde ele veio. A interface mostra a rota, o
    que transforma o modo degradado em informacao, nao em erro.
    """
    motivo = ""
    if disponivel():
        try:
            return _chamar(SISTEMA,
                           [{"role": "user",
                             "content": _montar_contexto(raca, trechos)}])
        except SemNuvem as e:
            motivo = str(e)
    else:
        motivo = "sem chave de API configurada"

    try:
        texto, tel = _pelo_slm(raca, trechos)
        tel["motivo_fallback"] = motivo
        return texto, tel
    except Exception as e:
        # Nem nuvem nem SLM. Entrega a propria base, que e texto verificado.
        # Feio, mas honesto, e melhor que uma tela de erro na bancada.
        return ("\n".join(trechos),
                {"rota": "base", "latencia_ms": 0.0, "tokens_entrada": 0,
                 "tokens_saida": 0, "motivo_fallback": f"{motivo}; SLM: {e}"})


if __name__ == "__main__":
    import rag

    p = provedor_atual()
    print(f"chave:    {mascarar(_chave()) or 'AUSENTE'}")
    print(f"provedor: {p['rotulo'] if p else 'nenhum'}")
    print(f"modelo:   {modelo_atual() or '—'}")
    print(f"nuvem:    {'disponivel' if disponivel() else 'indisponivel (usa SLM local)'}")

    print("\nprefixos reconhecidos:")
    for prov in PROVEDORES:
        print(f"  {'  '.join(prov['prefixos']):22} {prov['rotulo']}")

    trechos = [str(t) for t in rag.indice().perfil_bruto("Pug")]
    print(f"\ncontexto que subiria ({len(trechos)} trechos, so texto da base):")
    for t in trechos[:2]:
        print(f"  {t[:74]}...")

    print("\ngerando perfil...")
    texto, tel = perfil("Pug", trechos)
    print(f"\n--- rota: {tel['rota']} ({tel['latencia_ms']:.0f} ms) ---")
    print(texto[:280])
    if tel.get("motivo_fallback"):
        print(f"\n[fallback porque: {tel['motivo_fallback']}]")
