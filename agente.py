"""O agente: orquestra percepcao, linguagem e memoria por sessao.

FERRAMENTAS
-----------
    classificar(foto)     -> Leitura      (predict.ler)
    explicar(foto)        -> PNG base64   (gradcam.explain_image)
    buscar_perfil(raca)   -> texto        (rag + roteador -> llm ou slm)

O comportamento vem da `decisao` que `predict.ler()` devolve. A regra NAO e
reimplementada aqui -- isso esta dito no CLAUDE.md e importa: se a regra
existisse em dois lugares, calibrar os limiares deixaria de funcionar.

    afirmar    -> apresenta a raca e busca o perfil
    comparar   -> declara a ambiguidade e explica como distinguir as duas
    perguntar  -> formula uma pergunta, recebe texto livre, extrai campo
                  estruturado pelo SLM e reclassifica

O QUE "RECLASSIFICAR" SIGNIFICA AQUI
------------------------------------
A CNN nao aceita texto. O que o refinamento faz e REPONDERAR os palpites que
ela ja produziu, usando o campo extraido contra a base de conhecimento: se o
visitante diz que o pelo e longo, as racas cuja pelagem a base descreve como
longa sobem. A rede continua sendo a unica fonte de evidencia visual; o texto
so desempata. Vale dizer isso na arguicao -- e honesto e e o que de fato ocorre.

MEMORIA E PRIVACIDADE
---------------------
A sessao guarda o VETOR DE PROBABILIDADES, nao a foto. Depois que
`/classificar` responde, a imagem nao existe mais em lugar nenhum do processo:
nao foi para o disco (restricao 6), nao foi para a nuvem (restricao 2) e nao
ficou na memoria. O refinamento nao precisa dela, porque reponderar opera
sobre numeros que a rede ja calculou.
"""

from __future__ import annotations

import base64
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from PIL import Image

import gradcam
import predict
import rag
from roteador import Roteador

# Teto de sessoes vivas. Um evento de um dia nao deve acumular memoria sem
# limite; as mais antigas saem primeiro.
MAX_SESSOES = 200

# Campos que o refinamento sabe usar, com os termos que aparecem na base.
# Mapear por palavra, e nao por campo estruturado, mantem racas.json legivel
# por humanos -- que e quem a revisa.
CAMPOS_REFINAMENTO = {
    "pelagem": {
        "curta": ("curta", "curto", "rente", "sem pelo"),
        "media": ("semilonga", "media", "ondulada"),
        "longa": ("longa", "longo", "juba", "franjas"),
    },
    "porte": {
        "pequeno": ("muito pequeno", "pequeno", "1,5", "2 a 3", "3 a 6"),
        "medio": ("medio", "11 a", "12 a", "14 a", "16 a", "20 a"),
        "grande": ("grande", "39 a", "41 a", "45 a", "59 a", "27 a 45"),
    },
    "energia": {
        "baixa": ("baixa",),
        "media": ("media",),
        "alta": ("alta", "muito alta"),
    },
}

# UM enum so, com o campo embutido no valor.
#
# A versao anterior tinha dois campos -- `campo` e `valor`, com `valor` sendo
# um enum unico com todos os valores possiveis de todos os campos. O modelo
# entao respondia {"campo": "pelagem", "valor": "grande"} para "pelo enorme":
# escolhia o campo certo e, em seguida, o valor errado, porque "grande" estava
# disponivel e descreve bem um pelo enorme. O JSON Schema garantia saida
# valida, mas validade nao e correcao.
#
# Com os valores autodescritivos o erro nao e representavel: nao existe token
# "grande" solto para "pelagem" escolher.
VALORES_REFINAMENTO = {
    "pelo_curto": ("pelagem", "curta"),
    "pelo_medio": ("pelagem", "media"),
    "pelo_longo": ("pelagem", "longa"),
    "porte_pequeno": ("porte", "pequeno"),
    "porte_medio": ("porte", "medio"),
    "porte_grande": ("porte", "grande"),
    "energia_baixa": ("energia", "baixa"),
    "energia_media": ("energia", "media"),
    "energia_alta": ("energia", "alta"),
}

ESQUEMA_REFINAMENTO = {
    "type": "object",
    "properties": {
        "atributo": {
            "type": "string",
            # "nao_sei" existe para o modelo poder recusar. Sem essa saida ele
            # e obrigado a escolher algo, e passa a inventar.
            "enum": [*VALORES_REFINAMENTO, "nao_sei"],
        },
    },
    "required": ["atributo"],
}

INSTRUCAO_REFINAMENTO = (
    "O visitante esta descrevendo o pet dele. Identifique qual UNICO atributo "
    "ele mencionou. Se ele nao mencionou nenhum destes, responda nao_sei."
)


@dataclass
class Sessao:
    id: str
    criada_em: float = field(default_factory=time.time)
    probabilidades: list[float] = field(default_factory=list)
    leitura: predict.Leitura | None = None
    refinamentos: list[dict] = field(default_factory=list)
    perfis_entregues: set[str] = field(default_factory=set)


class Agente:
    def __init__(self, modelo: str = "mobilenet", checkpoints: str = "./checkpoints",
                 roteador: Roteador | None = None):
        self.modelo, self.class_names = predict.load_checkpoint(modelo, checkpoints)
        self.nome_modelo = modelo
        self.roteador = roteador or Roteador()
        self.indice = rag.indice()
        self._sessoes: OrderedDict[str, Sessao] = OrderedDict()

    # -- memoria -------------------------------------------------------
    def sessao(self, sessao_id: str | None = None) -> Sessao:
        if sessao_id and sessao_id in self._sessoes:
            self._sessoes.move_to_end(sessao_id)
            return self._sessoes[sessao_id]
        s = Sessao(id=sessao_id or str(uuid.uuid4()))
        self._sessoes[s.id] = s
        while len(self._sessoes) > MAX_SESSOES:
            self._sessoes.popitem(last=False)
        return s

    # -- ferramentas ---------------------------------------------------
    def classificar(self, foto: Image.Image) -> tuple[predict.Leitura, list[float], float]:
        t0 = time.perf_counter()
        probs = predict.probabilidades(self.modelo, foto)
        leitura = predict.ler(self.modelo, self.class_names, foto)
        return leitura, probs.tolist(), (time.perf_counter() - t0) * 1000

    def explicar(self, foto: Image.Image) -> tuple[str, float]:
        """Mapa de atencao como data URI. Nunca gravado em disco."""
        t0 = time.perf_counter()
        overlay, _, _ = gradcam.explain_image(self.modelo, foto)
        dados = base64.b64encode(gradcam.to_png_bytes(overlay)).decode("ascii")
        return f"data:image/png;base64,{dados}", (time.perf_counter() - t0) * 1000

    def buscar_perfil(self, raca: str) -> tuple[str, dict]:
        trechos = [str(t) for t in self.indice.perfil_bruto(raca)]
        return self.roteador.perfil(raca, trechos)

    # -- reponderacao --------------------------------------------------
    def _combina(self, raca: str, campo: str, valor: str) -> bool:
        """A base descreve essa raca com esse valor de campo?"""
        termos = CAMPOS_REFINAMENTO.get(campo, {}).get(valor, ())
        texto = " ".join(t.texto.lower() for t in self.indice.trechos
                         if t.raca == raca and t.campo == campo)
        return any(termo in texto for termo in termos)

    def reponderar(self, sessao: Sessao, campo: str, valor: str) -> predict.Leitura:
        """Reordena os palpites da rede com o que o visitante contou.

        Multiplicar e renormalizar, em vez de descartar as racas que nao
        combinam: o visitante pode se enganar sobre o proprio pet, e a
        evidencia visual nao deve ser jogada fora por causa de uma frase.
        """
        probs = torch.tensor(sessao.probabilidades)
        pesos = torch.tensor([2.5 if self._combina(n, campo, valor) else 1.0
                              for n in self.class_names])
        novas = probs * pesos
        novas = novas / novas.sum()

        sessao.probabilidades = novas.tolist()
        ordem = novas.argsort(descending=True)
        palpites = [predict.Palpite(self.class_names[int(i)], float(novas[int(i)]))
                    for i in ordem[:3]]
        topo = palpites[0].probabilidade
        margem = topo - palpites[1].probabilidade if len(palpites) > 1 else topo

        # A regra de decisao continua sendo a do predict.py -- so os numeros
        # mudaram. Reimplementa-la aqui e exatamente o que o CLAUDE.md proibe,
        # entao os limiares sao lidos de la.
        if topo < predict.CONFIANCA_BAIXA:
            decisao = "perguntar"
        elif margem < predict.MARGEM_AMBIGUA:
            decisao = "comparar"
        elif topo >= predict.CONFIANCA_ALTA:
            decisao = "afirmar"
        else:
            decisao = "comparar"

        leitura = predict.Leitura(palpites=palpites, decisao=decisao,
                                  topo=topo, margem=margem)
        sessao.leitura = leitura
        return leitura

    # -- fluxos --------------------------------------------------------
    def ver_foto(self, foto: Image.Image, sessao_id: str | None = None) -> dict:
        """Fluxo completo de POST /classificar."""
        s = self.sessao(sessao_id)
        leitura, probs, ms_percepcao = self.classificar(foto)
        mapa, ms_gradcam = self.explicar(foto)

        s.probabilidades = probs
        s.leitura = leitura

        raca_topo = leitura.palpites[0].raca
        mensagem, tel_msg = self.roteador.legenda(leitura, raca_topo)

        pergunta = None
        tel_perg = None
        if leitura.decisao == "perguntar":
            pergunta, tel_perg = self.roteador.pergunta(leitura, raca_topo)

        self.roteador.metricas.registrar_foto(leitura.decisao, ms_percepcao, ms_gradcam)

        return {
            "sessao": s.id,
            **leitura.to_dict(),
            "mapa_atencao": mapa,
            "mensagem": mensagem,
            "pergunta": pergunta,
            "latencia_ms": {
                "percepcao": round(ms_percepcao),
                "gradcam": round(ms_gradcam),
                "linguagem": round(tel_msg["latencia_ms"]
                                   + (tel_perg["latencia_ms"] if tel_perg else 0)),
            },
            "rota_linguagem": tel_msg["rota"],
        }

    def responder(self, sessao_id: str, texto: str) -> dict:
        """Fluxo de POST /responder: texto livre -> campo -> reponderacao."""
        s = self.sessao(sessao_id)
        if not s.probabilidades:
            raise ValueError("sessao sem foto classificada")

        dados, tel = self.roteador.extrair(texto, ESQUEMA_REFINAMENTO,
                                           INSTRUCAO_REFINAMENTO)
        campo, valor = VALORES_REFINAMENTO.get(dados.get("atributo", ""), (None, None))

        if campo is None:
            # Extracao nao deu em nada util: devolve a leitura como estava,
            # sem fingir que o refinamento funcionou.
            return {"sessao": s.id, **s.leitura.to_dict(), "entendido": None,
                    "mensagem": "Nao consegui aproveitar essa resposta. "
                                "Pode tentar descrever o pelo ou o tamanho?",
                    "latencia_ms": {"linguagem": round(tel["latencia_ms"])},
                    "rota_linguagem": tel["rota"]}

        leitura = self.reponderar(s, campo, valor)
        s.refinamentos.append({"campo": campo, "valor": valor, "texto": texto})

        raca_topo = leitura.palpites[0].raca
        mensagem, tel_msg = self.roteador.legenda(leitura, raca_topo)

        return {
            "sessao": s.id,
            **leitura.to_dict(),
            "entendido": {"campo": campo, "valor": valor},
            "mensagem": mensagem,
            "latencia_ms": {"linguagem": round(tel["latencia_ms"] + tel_msg["latencia_ms"])},
            "rota_linguagem": tel_msg["rota"],
        }

    def perfil(self, sessao_id: str, raca: str) -> dict:
        s = self.sessao(sessao_id)
        texto, tel = self.buscar_perfil(raca)
        s.perfis_entregues.add(raca)
        return {"texto": texto, "rota": tel["rota"],
                "latencia_ms": round(tel["latencia_ms"]),
                "fonte": "conhecimento/racas.json"}


if __name__ == "__main__":
    import json

    print("carregando agente...")
    a = Agente()
    print(f"modelo {a.nome_modelo}, {len(a.class_names)} classes, "
          f"indice por {a.indice.rota}")

    # Imagem sintetica: o modulo precisa se exercitar sozinho, sem depender de
    # o usuario ter uma foto a mao.
    foto = Image.new("RGB", (400, 400), (120, 110, 100))
    r = a.ver_foto(foto)
    print(f"\ndecisao: {r['decisao']}  topo {r['topo']:.0%}  margem {r['margem']:.0%}")
    for p in r["palpites"]:
        print(f"   {p['raca']:28} {p['probabilidade']:.1%}")
    print(f"mensagem: {r['mensagem']}")
    if r["pergunta"]:
        print(f"pergunta: {r['pergunta']}")
    print(f"latencias: {r['latencia_ms']}")
    print(f"mapa de atencao: {len(r['mapa_atencao'])} caracteres de data URI")

    print("\nrefinamento com texto livre:")
    r2 = a.responder(r["sessao"], "ele tem um pelo enorme, bem comprido mesmo")
    print(f"  entendido: {r2['entendido']}")
    print(f"  nova decisao: {r2['decisao']}")
    for p in r2["palpites"]:
        print(f"   {p['raca']:28} {p['probabilidade']:.1%}")

    print("\nperfil da raca do topo:")
    p = a.perfil(r["sessao"], r2["palpites"][0]["raca"])
    print(f"  rota {p['rota']} ({p['latencia_ms']} ms)")
    print(f"  {p['texto'][:200]}...")

    print("\nmetricas:")
    print(json.dumps(a.roteador.metricas.to_dict(), ensure_ascii=False, indent=2)[:600])
