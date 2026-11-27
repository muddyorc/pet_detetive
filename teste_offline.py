"""Teste de aceitacao da etapa 5: o sistema sem rede.

    python teste_offline.py

O CLAUDE.md define este como o teste que vale mais que os outros:

    "Desligue a rede e use o sistema inteiro. Classificacao, mapa, decisao e
     respostas curtas continuam. So o perfil longo mostra um aviso. Se
     qualquer outra coisa quebrar, o roteamento esta errado."

Em vez de pedir que alguem desligue o wi-fi, o teste aponta o cliente do SLM
para uma porta morta e limpa a chave da nuvem. O efeito e o mesmo e roda em
qualquer maquina, inclusive em CI.

O que ele prova, nessa ordem:

    1. a CNN classifica                      (nao depende de rede)
    2. o Grad-CAM gera o mapa                (nao depende de rede)
    3. a regra de decisao decide             (nao depende de rede)
    4. as frases curtas caem no texto de reserva, escrito a mao
    5. o perfil longo degrada para a base de conhecimento, com aviso
    6. os contadores continuam registrando

Se o passo 1, 2, 3 ou 6 falhar, ha dependencia de rede onde nao deveria.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    # Antes de importar slm: a URL base e lida na importacao do modulo.
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:59999"

    # Vazio, nao removido. O llm.py chama load_dotenv() na importacao, e o
    # .env agora pode ter a chave; load_dotenv nao sobrescreve variavel que ja
    # existe, entao definir como "" bloqueia a leitura do arquivo. Remover com
    # pop() deixaria o .env repor a chave e o teste nao simularia queda
    # nenhuma -- foi o que aconteceu assim que uma chave real foi configurada.
    for nome in ("ANTHROPIC_API_KEY", "PET_LLM_CHAVE", "PET_LLM_PROVEDOR"):
        os.environ[nome] = ""

    from PIL import Image

    import llm
    import slm
    from agente import Agente

    print("=" * 62)
    print("teste de aceitacao: sistema sem rede")
    print("=" * 62)
    print(f"SLM local:  {'no ar' if slm.disponivel() else 'FORA (simulado)'}")
    print(f"Nuvem:      {'no ar' if llm.disponivel() else 'FORA (simulado)'}")
    if slm.disponivel() or llm.disponivel():
        print("\nnao consegui simular a queda; o teste nao prova nada assim")
        return 1

    foto_exemplo = "data/oxford-iiit-pet/images/pug_52.jpg"
    if os.path.exists(foto_exemplo):
        foto, esperado = Image.open(foto_exemplo), "Pug"
    else:
        # Sem dataset o teste ainda roda: o que importa e nada quebrar.
        foto, esperado = Image.new("RGB", (400, 400), (130, 120, 110)), None

    agente = Agente()
    r = agente.ver_foto(foto)

    print(f"\n1. classificacao .. {r['palpites'][0]['raca']} "
          f"({r['topo']:.1%})")
    assert r["palpites"], "a rede nao devolveu palpites"
    if esperado:
        assert r["palpites"][0]["raca"] == esperado, \
            f"esperava {esperado}, veio {r['palpites'][0]['raca']}"

    print(f"2. grad-cam ....... {len(r['mapa_atencao']):,} caracteres de PNG")
    assert r["mapa_atencao"].startswith("data:image/png;base64,"), "mapa invalido"

    print(f"3. decisao ........ {r['decisao']}")
    assert r["decisao"] in {"afirmar", "comparar", "perguntar"}

    print(f"4. frase curta .... [{r['rota_linguagem']}] {r['mensagem']}")
    assert r["rota_linguagem"] == "reserva", \
        f"sem SLM a frase devia vir da reserva, veio de {r['rota_linguagem']}"
    termo = slm.violou_guardrail(r["mensagem"])
    assert termo is None, f"guardrail violado pelo texto de reserva: {termo}"

    raca = r["palpites"][0]["raca"]
    p = agente.perfil(r["sessao"], raca)
    print(f"5. perfil longo ... rota '{p['rota']}' (degradado, como previsto)")
    assert p["rota"] == "base", f"esperava rota 'base', veio '{p['rota']}'"
    assert raca in p["texto"], "o perfil devia conter a propria base da raca"

    m = agente.roteador.metricas.to_dict()
    print(f"6. metricas ....... {m['fotos']} foto, decisoes {m['decisoes']}")
    assert m["fotos"] == 1

    print("\n" + "=" * 62)
    print("passou. Classificacao, mapa, decisao e respostas curtas")
    print("funcionam sem rede. So o perfil longo degradou, com aviso.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
