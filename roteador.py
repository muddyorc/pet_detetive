"""Camada de roteamento: decide local ou nuvem, por tarefa, e mede tudo.

Esta e a entrega visivel de ESW489. A tabela abaixo e a especificacao do
CLAUDE.md virada em codigo:

    legenda curta, provocacao      local    dezenas de chamadas, precisa ser rapida
    extracao estruturada           local    tarefa fechada, saida em JSON
    pergunta de refinamento        local    latencia percebida pelo visitante
    perfil completo da raca        nuvem    texto longo, uma vez por visitante
    qualquer tarefa, sem rede      local    modo degradado

O principio e simples e vale dizer na bancada: o caminho quente e local. O
visitante interage dezenas de vezes por sessao, e cada uma dessas interacoes
tem que responder em menos de um segundo. So o perfil longo -- uma vez por
visitante -- justifica a ida a nuvem.

Os contadores nao sao enfeite: GET /metricas os expoe, e e com eles que se
defende a arquitetura ("87% das chamadas foram resolvidas localmente").
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import llm
import slm

# Tarefa -> rota pretendida. "pretendida" porque sem rede tudo vira local.
ROTA_PRETENDIDA = {
    "legenda": "local",
    "extracao": "local",
    "pergunta": "local",
    "perfil": "nuvem",
}

MOTIVO = {
    "legenda": "dezenas de chamadas por sessao, precisa ser rapida",
    "extracao": "tarefa fechada, saida em JSON, nao precisa de modelo grande",
    "pergunta": "latencia percebida diretamente pelo visitante",
    "perfil": "texto longo, uma vez por visitante",
}


@dataclass
class Contador:
    chamadas: int = 0
    erros: int = 0
    latencia_total_ms: float = 0.0
    tokens_entrada: int = 0
    tokens_saida: int = 0

    @property
    def latencia_media_ms(self) -> float:
        return self.latencia_total_ms / self.chamadas if self.chamadas else 0.0

    def to_dict(self) -> dict:
        return {"chamadas": self.chamadas, "erros": self.erros,
                "latencia_media_ms": round(self.latencia_media_ms, 1),
                "tokens_entrada": self.tokens_entrada,
                "tokens_saida": self.tokens_saida}


@dataclass
class Metricas:
    """Contadores do evento. A api.py mantem uma instancia para o dia todo."""

    por_rota: dict[str, Contador] = field(default_factory=dict)
    por_tarefa: dict[str, Contador] = field(default_factory=dict)
    fotos: int = 0
    decisoes: dict[str, int] = field(default_factory=dict)
    # Latencias da percepcao, que nao passam pelo roteador mas aparecem no
    # mesmo painel: sao elas que o visitante sente primeiro.
    percepcao: Contador = field(default_factory=Contador)
    gradcam: Contador = field(default_factory=Contador)
    _trava: threading.Lock = field(default_factory=threading.Lock)

    def registrar(self, tarefa: str, rota: str, latencia_ms: float,
                  tokens_entrada: int = 0, tokens_saida: int = 0,
                  erro: bool = False) -> None:
        with self._trava:
            for mapa, chave in ((self.por_rota, rota), (self.por_tarefa, tarefa)):
                c = mapa.setdefault(chave, Contador())
                c.chamadas += 1
                c.latencia_total_ms += latencia_ms
                c.tokens_entrada += tokens_entrada
                c.tokens_saida += tokens_saida
                if erro:
                    c.erros += 1

    def registrar_foto(self, decisao: str, ms_percepcao: float, ms_gradcam: float) -> None:
        with self._trava:
            self.fotos += 1
            self.decisoes[decisao] = self.decisoes.get(decisao, 0) + 1
            self.percepcao.chamadas += 1
            self.percepcao.latencia_total_ms += ms_percepcao
            self.gradcam.chamadas += 1
            self.gradcam.latencia_total_ms += ms_gradcam

    def to_dict(self) -> dict:
        with self._trava:
            total = sum(c.chamadas for c in self.por_rota.values())
            locais = self.por_rota.get("local", Contador()).chamadas
            # O numero que vale a pena estar grande no painel.
            pct_local = (100.0 * locais / total) if total else 100.0
            return {
                "fotos": self.fotos,
                "decisoes": dict(self.decisoes),
                "chamadas_linguagem": total,
                "percentual_local": round(pct_local, 1),
                "por_rota": {k: v.to_dict() for k, v in self.por_rota.items()},
                "por_tarefa": {k: v.to_dict() for k, v in self.por_tarefa.items()},
                "percepcao_ms": round(self.percepcao.latencia_media_ms, 1),
                "gradcam_ms": round(self.gradcam.latencia_media_ms, 1),
            }


class Roteador:
    def __init__(self, metricas: Metricas | None = None):
        self.metricas = metricas or Metricas()

    # ------------------------------------------------------------------
    def estado(self) -> dict:
        """O que o GET /saude mostra. Consultado a cada carga da pagina."""
        return {"slm_disponivel": slm.disponivel(),
                "nuvem_disponivel": llm.disponivel()}

    def rota_de(self, tarefa: str) -> tuple[str, str]:
        """Rota efetiva e o porque. Sem rede, tudo cai para local."""
        pretendida = ROTA_PRETENDIDA.get(tarefa, "local")
        if pretendida == "nuvem" and not llm.disponivel():
            return "local", "nuvem indisponivel: modo degradado"
        return pretendida, MOTIVO.get(tarefa, "padrao local")

    # ------------------------------------------------------------------
    def _curto(self, tarefa: str, prompt: str, reserva: str,
               max_tokens: int = 70) -> tuple[str, dict]:
        """Tarefa curta e local. `reserva` e usada se o SLM estiver fora.

        Toda funcao que fala com rede tem fallback -- aqui o fallback e um
        texto fixo, escrito a mao, que mantem a interface coerente.
        """
        try:
            texto, tel = slm.gerar(prompt, max_tokens=max_tokens)
            self.metricas.registrar(tarefa, "local", tel["latencia_ms"],
                                    tel["tokens_entrada"], tel["tokens_saida"])

            # Guardrail da restricao 1, verificado e nao so pedido. Um modelo
            # de 3B escorrega; o texto de reserva e escrito a mao e nunca
            # escorrega. Melhor uma frase mais simples do que a palavra errada
            # na tela na frente do avaliador.
            termo = slm.violou_guardrail(texto)
            if termo:
                self.metricas.registrar(tarefa, "local", 0.0, erro=True)
                return reserva, {"rota": "reserva", "latencia_ms": tel["latencia_ms"],
                                 "tokens_entrada": tel["tokens_entrada"],
                                 "tokens_saida": tel["tokens_saida"],
                                 "guardrail": termo}
            return texto, {"rota": "local", **tel}
        except slm.SLMIndisponivel:
            self.metricas.registrar(tarefa, "local", 0.0, erro=True)
            return reserva, {"rota": "reserva", "latencia_ms": 0.0,
                             "tokens_entrada": 0, "tokens_saida": 0}

    def legenda(self, leitura, raca_topo: str) -> tuple[str, dict]:
        """Frase curta que acompanha o resultado na tela."""
        if leitura.decisao == "afirmar":
            pedido = (f"A rede achou {leitura.topo:.0%} de semelhanca com {raca_topo}. "
                      "Diga isso ao visitante em uma frase.")
            reserva = f"A foto se parece bastante com um {raca_topo}."
        elif leitura.decisao == "comparar":
            segundo = leitura.palpites[1].raca if len(leitura.palpites) > 1 else "outra raca"
            pedido = (f"A rede ficou dividida entre {raca_topo} e {segundo}, com "
                      f"{leitura.topo:.0%} e {leitura.palpites[1].probabilidade:.0%}. "
                      "Diga em uma frase que ha ambiguidade e cite as duas.")
            reserva = f"A foto fica entre {raca_topo} e {segundo} — as duas se parecem muito."
        else:
            pedido = ("A rede nao teve confianca em nenhuma raca. Diga isso com "
                      "honestidade e sem drama, em uma frase.")
            reserva = "Nao consegui me decidir por nenhuma raca com essa foto."
        return self._curto("legenda", pedido, reserva)

    def pergunta(self, leitura, raca_topo: str) -> tuple[str, dict]:
        """Pergunta de refinamento, feita ao visitante quando ha duvida."""
        pedido = (
            "Faca UMA pergunta curta e objetiva ao visitante sobre o pet dele, "
            "para ajudar a distinguir entre racas parecidas. Pergunte sobre "
            "pelagem, porte ou energia. Nao cumprimente, nao explique: so a pergunta."
        )
        return self._curto("pergunta", pedido, "O pelo dele e curto, medio ou longo?", 40)

    def extrair(self, texto: str, esquema: dict,
                instrucao: str | None = None) -> tuple[dict, dict]:
        """Texto livre do visitante -> JSON. O caso canonico de ESW489."""
        try:
            dados, tel = slm.extrair(texto, esquema, instrucao)
            self.metricas.registrar("extracao", "local", tel["latencia_ms"],
                                    tel["tokens_entrada"], tel["tokens_saida"])
            return dados, {"rota": "local", **tel}
        except slm.SLMIndisponivel:
            self.metricas.registrar("extracao", "local", 0.0, erro=True)
            return {}, {"rota": "reserva", "latencia_ms": 0.0,
                        "tokens_entrada": 0, "tokens_saida": 0}

    def perguntar(self, pergunta: str, trechos: list[str],
                  historico: list[dict] | None = None,
                  aprofundar: bool = False) -> tuple[str, dict]:
        """Pergunta aberta do visitante.

        POR QUE A ROTA NAO E ADIVINHADA
        -------------------------------
        A ideia obvia seria o roteador decidir sozinho se a pergunta precisa da
        nuvem. Tentamos duas formas e as duas falharam, entao a decisao ficou
        visivel em vez de automatica:

        1. Pela similaridade do RAG. Nao separa: perguntas respondidas pela
           base deram 0,027 a 0,343, e perguntas que exigem raciocinio, 0,044 a
           0,129. Faixas sobrepostas. "quanto ele pesa?" pontua 0,039 porque a
           base diz "6 a 8 kg", nao "pesa".

        2. Pedindo ao SLM que classifique a pergunta. Com quatro classes:
           6/12. Reduzindo para um booleano `precisa_pensar`: 7/12, e olhando
           as respostas ele diz `true` quase sempre -- classificador
           degenerado, nao roteador. Um 3B colapsa nesse tipo de juizo.

        O QUE FAZEMOS
        -------------
        Toda pergunta e respondida LOCALMENTE primeiro, em um a tres segundos.
        Sob a resposta, o visitante pode pedir para aprofundar, e so ai a
        nuvem entra (`aprofundar=True`).

        Isso e melhor que acertar o roteamento escondido: o visitante ve as
        duas rotas e sente a diferenca de latencia e de profundidade. A tese
        de ESW489 -- o caminho quente e local, a nuvem e excecao -- deixa de
        ser uma afirmacao do banner e vira algo demonstravel na bancada. E
        nenhum roteamento errado e possivel.
        """
        if aprofundar and llm.disponivel():
            try:
                texto, tel = llm.responder(pergunta, trechos, historico)
                termo = slm.violou_guardrail(texto)
                if termo:
                    texto = ("Nao consegui responder isso sem escorregar no "
                             "vocabulario de genetica. O que a base diz:\n\n"
                             + "\n".join(trechos[:3]))
                    tel = {**tel, "rota": "base", "guardrail": termo}
                self.metricas.registrar("pergunta_aberta", tel["rota"],
                                        tel["latencia_ms"], tel["tokens_entrada"],
                                        tel["tokens_saida"])
                return texto, tel
            except llm.SemNuvem:
                pass            # cai no local, como todo o resto do sistema

        corpo = "\n".join(f"- {t}" for t in trechos)
        prompt = (f"Base de conhecimento:\n{corpo}\n\n"
                  f"Pergunta do visitante: {pergunta}\n\n"
                  "Responda em no maximo duas frases, direto ao ponto, usando a base.")
        reserva = "O que a base diz sobre isso:\n\n" + "\n".join(trechos[:2])
        return self._curto("pergunta_aberta", prompt, reserva, max_tokens=150)

    def perfil(self, raca: str, trechos: list[str]) -> tuple[str, dict]:
        """Perfil longo. A unica tarefa que tenta a nuvem."""
        texto, tel = llm.perfil(raca, trechos)

        # O mesmo guardrail do texto curto. Aqui a reserva e a propria base de
        # conhecimento: texto escrito a mao, revisado, e que por construcao
        # nao fala de genetica.
        termo = slm.violou_guardrail(texto)
        if termo:
            texto = "\n".join(trechos)
            tel = {**tel, "rota": "base", "guardrail": termo}

        self.metricas.registrar("perfil", tel["rota"], tel["latencia_ms"],
                                tel.get("tokens_entrada", 0), tel.get("tokens_saida", 0))
        return texto, tel


if __name__ == "__main__":
    import json

    r = Roteador()
    print("estado:", json.dumps(r.estado(), ensure_ascii=False))
    print("\ntabela de roteamento:")
    for tarefa in ROTA_PRETENDIDA:
        rota, motivo = r.rota_de(tarefa)
        marca = "" if rota == ROTA_PRETENDIDA[tarefa] else "  <- degradado"
        print(f"  {tarefa:10} -> {rota:6} {motivo}{marca}")

    print("\nexercitando extracao e contadores:")
    esquema = {"type": "object",
               "properties": {"porte": {"type": "string",
                                        "enum": ["pequeno", "medio", "grande"]}},
               "required": ["porte"]}
    dados, tel = r.extrair("ele e enorme, quase do tamanho de um bezerro", esquema)
    print(f"  extracao -> {dados}  [{tel['rota']}, {tel['latencia_ms']:.0f} ms]")

    r.metricas.registrar_foto("comparar", 84.0, 120.0)
    print("\nmetricas:")
    print(json.dumps(r.metricas.to_dict(), ensure_ascii=False, indent=2))
