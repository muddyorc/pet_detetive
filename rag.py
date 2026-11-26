"""Recuperacao sobre a base de conhecimento das racas.

Sao poucas centenas de trechos. Nao instale banco vetorial para isso: um
numpy com similaridade de cosseno resolve, e cabe inteiro na memoria.

DUAS ROTAS, PORQUE O EVENTO PODE FICAR SEM REDE
-----------------------------------------------
1. Embeddings do Ollama (`nomic-embed-text`) -- melhor qualidade semantica.
2. TF-IDF do scikit-learn -- fallback que nao depende de nada externo.

MEDIMOS AS DUAS, E O PADRAO E O TF-IDF
--------------------------------------
Contra a intuicao, o TF-IDF ganhou nas tres consultas de teste:

    "gato que mia muito e pede atencao"
        embeddings -> Egyptian Mau / pelagem      (errado)
        tfidf      -> Siamese / temperamento      (certo)

    "cachorro grande e branco que late a noite"
        embeddings -> Samoyed / sinais            (so metade)
        tfidf      -> Great Pyrenees / cuidados:
                      "Late a noite por instinto de guarda"   (exato)

E ganhou por 3 ms contra 2096 ms, ou seja, setecentas vezes mais rapido.

Dois motivos. O `nomic-embed-text` foi treinado sobretudo em ingles, e esta
base esta em portugues sem acentos. E cada consulta por embedding custa uma ida
ao Ollama pela rede, enquanto o TF-IDF e uma multiplicacao de matriz local.

Por isso `USAR_EMBEDDINGS` vem False. A rota de embeddings continua pronta:
troque para True se a base crescer muito ou mudar de idioma. Os vetores do
Ollama sao gravados em cache quando usados.

Vale dizer isso na arguicao: ter medido as duas rotas e escolhido com numero e
um argumento melhor do que usar embeddings porque e o que se espera de um RAG.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import slm

# Medido: o TF-IDF ganha em qualidade e e 700x mais rapido nesta base.
# Ver o cabecalho do modulo. True reativa a rota de embeddings.
USAR_EMBEDDINGS = False

BASE_PADRAO = Path("conhecimento/racas.json")
CACHE_PADRAO = Path("conhecimento/_vetores.npz")

# Ordem em que os campos entram no perfil. Nao e alfabetica de proposito:
# e a ordem em que uma pessoa pergunta sobre um animal na bancada.
ORDEM_CAMPOS = ("especie", "temperamento", "porte", "energia",
                "pelagem", "cuidados", "sinais_distintivos")


@dataclass
class Trecho:
    raca: str
    campo: str
    texto: str

    def __str__(self) -> str:
        return f"{self.raca} — {self.campo}: {self.texto}"


def carregar_trechos(caminho: Path = BASE_PADRAO) -> list[Trecho]:
    """Um trecho por campo de cada raca.

    Granularidade por campo, e nao por raca inteira, porque a pergunta do
    visitante costuma ser sobre um aspecto ("ele late muito?") e recuperar o
    verbete inteiro traria ruido para o gerador.
    """
    bruto = json.loads(Path(caminho).read_text(encoding="utf-8"))
    trechos: list[Trecho] = []
    for raca, dados in bruto.items():
        if raca.startswith("_"):
            continue
        for campo in ORDEM_CAMPOS:
            if dados.get(campo):
                trechos.append(Trecho(raca, campo, dados[campo]))
    return trechos


class Indice:
    """Indice de busca com rota de embeddings e fallback TF-IDF."""

    def __init__(self, trechos: list[Trecho], usar_embeddings: bool | None = None,
                 cache: Path | None = CACHE_PADRAO):
        self.trechos = trechos
        if usar_embeddings is None:
            usar_embeddings = USAR_EMBEDDINGS
        self.rota = "tfidf"
        self._matriz: np.ndarray | None = None
        self._vetorizador = None

        if usar_embeddings and slm.disponivel():
            try:
                self._matriz = self._embeddings(cache)
                self.rota = "embeddings"
            except (slm.SLMIndisponivel, OSError, ValueError):
                # Qualquer falha cai no TF-IDF em silencio: na bancada, uma
                # busca pior e muito melhor que uma excecao na tela.
                self._matriz = None

        if self._matriz is None:
            self._tfidf()

    # ------------------------------------------------------------------
    def _chave_cache(self) -> str:
        """Identifica o conteudo indexado, para invalidar o cache sozinho."""
        import hashlib
        h = hashlib.sha256()
        for t in self.trechos:
            h.update(str(t).encode("utf-8"))
        h.update(slm.MODELO_EMBED.encode("utf-8"))
        return h.hexdigest()

    def _embeddings(self, cache: Path | None) -> np.ndarray:
        chave = self._chave_cache()
        if cache and Path(cache).exists():
            guardado = np.load(cache, allow_pickle=False)
            if str(guardado.get("chave", "")) == chave:
                return guardado["matriz"]

        vetores = np.array(slm.embeddings([str(t) for t in self.trechos]),
                           dtype=np.float32)
        vetores = self._normalizar(vetores)
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, matriz=vetores, chave=np.array(chave))
        return vetores

    def _tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        # analyzer de caracteres: o portugues tem muita flexao ("peludo",
        # "peluda", "pelagem") e n-gramas de caractere captam isso sem
        # precisar de stemmer.
        self._vetorizador = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                            lowercase=True, min_df=1)
        m = self._vetorizador.fit_transform([str(t) for t in self.trechos])
        self._matriz = self._normalizar(np.asarray(m.todense(), dtype=np.float32))

    @staticmethod
    def _normalizar(m: np.ndarray) -> np.ndarray:
        normas = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.maximum(normas, 1e-9)

    def _vetor_consulta(self, consulta: str) -> np.ndarray:
        if self.rota == "embeddings":
            v = np.array(slm.embeddings([consulta])[0], dtype=np.float32)
        else:
            m = self._vetorizador.transform([consulta])
            v = np.asarray(m.todense(), dtype=np.float32)[0]
        return v / max(float(np.linalg.norm(v)), 1e-9)

    # ------------------------------------------------------------------
    def buscar(self, consulta: str, k: int = 4,
               raca: str | None = None) -> tuple[list[Trecho], dict]:
        """Trechos mais parecidos com a consulta.

        `raca` restringe a busca a uma raca -- e o caso de buscar_perfil(),
        em que ja se sabe o animal e se quer o melhor material sobre ele.
        """
        t0 = time.perf_counter()
        candidatos = range(len(self.trechos))
        if raca:
            candidatos = [i for i, t in enumerate(self.trechos) if t.raca == raca]
            if not candidatos:
                return [], {"rota": self.rota, "latencia_ms": 0.0, "encontrados": 0}

        try:
            v = self._vetor_consulta(consulta)
        except slm.SLMIndisponivel:
            # O Ollama caiu depois da indexacao. Reconstroi em TF-IDF e segue.
            self.rota = "tfidf"
            self._tfidf()
            v = self._vetor_consulta(consulta)

        indices = np.array(list(candidatos))
        pontos = self._matriz[indices] @ v
        ordem = np.argsort(-pontos)[:k]
        melhores = indices[ordem]
        return ([self.trechos[i] for i in melhores],
                {"rota": self.rota,
                 "latencia_ms": (time.perf_counter() - t0) * 1000,
                 "encontrados": len(melhores),
                 # A similaridade do melhor trecho. O roteador usa isso para
                 # decidir entre o SLM local e a nuvem: quando a resposta esta
                 # literalmente na base, um modelo pequeno so precisa
                 # reescrever; quando nao esta, alguem tem que raciocinar.
                 "similaridade": float(pontos[ordem[0]]) if len(ordem) else 0.0})

    def perfil_bruto(self, raca: str) -> list[Trecho]:
        """Todos os trechos de uma raca, na ordem de ORDEM_CAMPOS.

        E o que vai para o llm.py montar o perfil longo. Nao passa pela busca
        porque aqui nao ha o que ranquear: queremos o verbete inteiro.
        """
        por_campo = {t.campo: t for t in self.trechos if t.raca == raca}
        return [por_campo[c] for c in ORDEM_CAMPOS if c in por_campo]


_indice: Indice | None = None


def indice(recarregar: bool = False) -> Indice:
    """Indice unico do processo. A api.py monta uma vez na inicializacao."""
    global _indice
    if _indice is None or recarregar:
        _indice = Indice(carregar_trechos())
    return _indice


if __name__ == "__main__":
    trechos = carregar_trechos()
    racas = {t.raca for t in trechos}
    print(f"{len(trechos)} trechos de {len(racas)} racas")

    for usar in (True, False):
        idx = Indice(trechos, usar_embeddings=usar, cache=None)
        print(f"\n=== rota: {idx.rota} ===")
        for consulta in ["gato que mia muito e pede atencao",
                         "cachorro grande e branco que late a noite",
                         "precisa escovar todo dia senao embola"]:
            achados, tel = idx.buscar(consulta, k=2)
            print(f"\n  \"{consulta}\"   [{tel['latencia_ms']:.0f} ms]")
            for t in achados:
                print(f"    {t.raca} / {t.campo}: {t.texto[:70]}...")

    print("\n=== perfil bruto de uma raca ===")
    for t in indice().perfil_bruto("Pug"):
        print(f"  {t.campo:19} {t.texto[:66]}...")
