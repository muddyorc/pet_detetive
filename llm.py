"""Cliente do LLM em nuvem. Monta o perfil longo da raca.

E a UNICA parte do sistema que fala com a internet, e por isso a mais
cuidadosa. Tres regras, nessa ordem de importancia:

1. NUNCA RECEBE IMAGEM. O que sobe e o nome da raca e os trechos da base de
   conhecimento local -- texto que ja estava no repositorio antes do visitante
   chegar. A foto, o tensor e o mapa de atencao nao saem da maquina. Esse e o
   argumento central de privacidade do projeto (restricoes 2 e 6 do CLAUDE.md);
   se a imagem subir, a tese cai. Veja `_montar_contexto`: ela recebe str, nao
   bytes, e nao existe caminho no modulo que aceite PIL.Image.

2. TIMEOUT CURTO E FALLBACK. Sem chave, sem rede ou com a API lenta, o perfil
   e gerado pelo SLM local. O visitante nunca ve um erro, so um texto um pouco
   mais simples. `perfil()` nao levanta excecao em operacao normal.

3. UMA CHAMADA POR VISITANTE. Perfil longo e a unica tarefa que vai para a
   nuvem; legenda, pergunta e extracao sao do SLM local (restricao 4).
"""

from __future__ import annotations

import os
import time

# O .env e opcional: sem ele o modulo cai no SLM local em vez de quebrar.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODELO = os.environ.get("PET_LLM_MODELO", "claude-opus-5-5")

# Timeout curto de proposito: numa bancada, dez segundos de silencio ja e
# tempo demais. Estourou, cai no SLM local, que responde em menos de um.
TIMEOUT_S = float(os.environ.get("PET_LLM_TIMEOUT", "12"))

SISTEMA = (
    "Voce escreve perfis curtos de racas de caes e gatos para uma feira de "
    "ciencias, a partir de uma base de conhecimento fornecida.\n\n"
    "REGRAS:\n"
    "- Use SOMENTE os dados fornecidos. Nao acrescente fatos que nao estejam "
    "neles: a base existe justamente para impedir invencao.\n"
    "- A rede neural compara a foto com 37 racas puras e mede SEMELHANCA "
    "VISUAL. Nunca escreva 'composicao', 'porcentagem de raca', 'DNA', "
    "'ancestralidade' ou 'o animal e da raca X'. Escreva 'se parece com'.\n"
    "- Portugues do Brasil, tom sobrio e direto, sem linguagem de catalogo de "
    "canil ('companheiro leal e amoroso' e exatamente o que nao queremos).\n"
    "- Tres paragrafos curtos, no maximo 140 palavras no total."
)


class SemNuvem(RuntimeError):
    """Sem chave, sem rede ou a API recusou. Quem chama usa o SLM local."""


def _chave() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None


def disponivel() -> bool:
    """True se ha chave configurada E o SDK esta instalado.

    Nao faz chamada de rede: isso e consultado pelo GET /saude a cada
    carregamento da pagina, e nao vale gastar uma requisicao por visita.
    """
    if not _chave():
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _montar_contexto(raca: str, trechos: list[str]) -> str:
    """Monta o texto que sobe para a nuvem.

    Recebe `str`, nunca bytes nem PIL.Image -- a assinatura e parte da
    garantia de privacidade, nao so uma conveniencia.
    """
    corpo = "\n".join(f"- {t}" for t in trechos)
    return (f"Raca com a qual a foto mais se parece: {raca}\n\n"
            f"Base de conhecimento sobre essa raca:\n{corpo}\n\n"
            f"Escreva o perfil.")


def _pela_nuvem(raca: str, trechos: list[str]) -> tuple[str, dict]:
    import anthropic

    cliente = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=1)
    t0 = time.perf_counter()
    try:
        r = cliente.beta.messages.create(
            model=MODELO,
            max_tokens=1000,
            # Perfil curto a partir de dados prontos: nao e tarefa que peca
            # raciocinio profundo, e esforco baixo corta a latencia.
            output_config={"effort": "low"},
            # Se o classificador de seguranca recusar, a propria API refaz a
            # chamada noutro modelo em vez de devolver resposta vazia.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SISTEMA,
            messages=[{"role": "user", "content": _montar_contexto(raca, trechos)}],
        )
    except anthropic.APIError as e:
        raise SemNuvem(f"API respondeu com erro: {e}") from e
    except Exception as e:  # timeout, DNS, rede caida
        raise SemNuvem(f"falha de rede: {e}") from e

    if r.stop_reason == "refusal":
        raise SemNuvem("a API recusou a solicitacao")

    texto = "".join(b.text for b in r.content if b.type == "text").strip()
    if not texto:
        raise SemNuvem("resposta vazia")

    return texto, {
        "rota": "nuvem",
        "modelo": MODELO,
        "latencia_ms": (time.perf_counter() - t0) * 1000,
        "tokens_entrada": r.usage.input_tokens,
        "tokens_saida": r.usage.output_tokens,
    }


def _pelo_slm(raca: str, trechos: list[str]) -> tuple[str, dict]:
    """Fallback local. Texto mais simples, mesma garantia de nao inventar."""
    import slm

    texto, tel = slm.gerar(
        _montar_contexto(raca, trechos),
        sistema=SISTEMA,
        temperatura=0.4,
        max_tokens=220,
    )
    tel["rota"] = "local"
    return texto, tel


def perfil(raca: str, trechos: list[str]) -> tuple[str, dict]:
    """Perfil longo da raca. Tenta a nuvem, cai no SLM local.

    Nao levanta excecao em operacao normal: devolve o melhor texto possivel e
    diz em `telemetria["rota"]` de onde ele veio. A interface mostra a rota, o
    que transforma o modo degradado em informacao, nao em erro.
    """
    if disponivel():
        try:
            return _pela_nuvem(raca, trechos)
        except SemNuvem as e:
            motivo = str(e)
        else:
            motivo = ""
    else:
        motivo = "sem ANTHROPIC_API_KEY configurada"

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

    print(f"modelo:    {MODELO}")
    print(f"chave:     {'configurada' if _chave() else 'AUSENTE'}")
    print(f"nuvem:     {'disponivel' if disponivel() else 'indisponivel (usa SLM local)'}")

    trechos = [str(t) for t in rag.indice().perfil_bruto("Pug")]
    print(f"\ncontexto que subiria ({len(trechos)} trechos, so texto da base):")
    for t in trechos[:3]:
        print(f"  {t[:76]}...")

    print("\ngerando perfil...")
    texto, tel = perfil("Pug", trechos)
    print(f"\n--- rota: {tel['rota']} ({tel['latencia_ms']:.0f} ms) ---")
    print(texto)
    if tel.get("motivo_fallback"):
        print(f"\n[fallback porque: {tel['motivo_fallback']}]")
