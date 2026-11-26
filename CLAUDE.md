# Pet Detetive — contexto do projeto

Leia este arquivo antes de qualquer alteração. Ele descreve o que o projeto é,
o que já existe, o que falta e as decisões que **não** devem ser revertidas.

---

## 1. O que é

Aplicação interativa para o 5º Workshop de Trabalhos de IA da FASOFT (UniRV),
em **30/11/2026**. O visitante mostra uma foto do próprio pet e o sistema:

1. identifica **com quais raças o animal mais se parece** (nunca "composição genética");
2. exibe um **mapa de atenção** Grad-CAM sobre a foto, mostrando onde a rede olhou;
3. **declara ambiguidade** quando as duas primeiras raças estão próximas, e **pergunta**
   ao visitante quando não tem confiança em nada;
4. apresenta um **perfil da raça** ancorado em base de conhecimento;
5. faz tudo isso com a **imagem nunca saindo da máquina local**.

O projeto é avaliado simultaneamente por três disciplinas, com votação popular
separada em cada uma. Cada camada precisa existir e ser defensável:

| Disciplina | O que ela avalia aqui |
| --- | --- |
| ESW456 Redes Neurais | Duas CNNs comparadas, aumento de dados, regularização, métricas, Grad-CAM |
| ESW488 Agentes Inteligentes | Agente com tool calling, memória, prompts, RAG, API de LLM |
| ESW489 SLM e IA Corporativa | SLM quantizado local, camada de roteamento, latência, privacidade |

---

## 2. Restrições que não podem ser quebradas

Estas não são preferências de estilo. Quebrar qualquer uma destrói a entrega de
alguma das três disciplinas, ou torna o projeto indefensável numa arguição.

1. **Nunca afirmar ascendência genética.** O que a rede produz é uma softmax sobre
   raças puras, ou seja, uma medida de *semelhança visual*. Todo texto de interface
   e todo prompt deve dizer "com quais raças se parece", nunca "composição" ou
   "% de raça X". Isso está no prompt do agente como guardrail explícito.

2. **A foto nunca sai da máquina.** O LLM em nuvem recebe apenas o *nome da raça* e
   o texto recuperado do RAG. Nunca a imagem, nunca o mapa de atenção, nunca bytes
   de foto. Esse é o argumento central de privacidade do projeto; se a imagem subir
   para a nuvem, a tese cai.

2b. **Precisao de linguagem sobre a restricao 2.** "A foto nunca sai da
   maquina" vale ao pe da letra quando o visitante usa o notebook da bancada.
   Se ele acessar do proprio celular (`python api.py --rede`), a foto cruza o
   Wi-Fi local ate o notebook. Ela continua **nunca tocando a internet**, que
   e o argumento que importa, mas na arguicao diga "a foto nunca sai da rede
   local" nesse caso. Afirmar o mais forte e ser pego na diferenca custa mais
   do que a diferenca vale.

   Pelo mesmo motivo, **nao use tunel** (ngrok e parecidos) para liberar a
   camera ao vivo no celular: daria HTTPS, mas as fotos passariam a trafegar
   pela internet, e a tese cai.

3. **As duas redes compartilham o pipeline.** Mesmo split (semente 42), mesmo aumento
   de dados, mesma rotina de avaliação. Só o modelo muda. É isso que torna a
   comparação do banner honesta. Não "otimize" uma das duas separadamente.

4. **O caminho quente é local.** Legendas curtas, extração estruturada de respostas
   e perguntas de refinamento são tarefas do SLM local. Só o perfil longo da raça
   vai para a nuvem, uma vez por visitante.

5. **Sem rede, o sistema continua funcionando.** Classificação, Grad-CAM, decisão e
   respostas curtas não dependem de internet. Só o perfil longo fica indisponível,
   e isso deve ser exibido com elegância, não como erro.

6. **A foto também não fica na máquina.** A restrição 2 impede a imagem de subir
   para a nuvem; esta impede que ela seja gravada em disco. `POST /classificar`
   lê o upload para `BytesIO` e trabalha com `PIL.Image` em memória — nunca
   `arquivo.save(...)`, nunca pasta de temporários. Sem isso, no fim do dia 30/11
   o notebook acumula fotos dos pets de dezenas de desconhecidos: a imagem não
   foi para a nuvem, mas também não precisava ter ficado. O mapa de atenção vai
   para a resposta como base64 e morre com a sessão.

---

## 3. Estado atual — o que já existe e está testado

```
baixar_dataset.py  monta o dataset rápido (HF + anotações de Oxford)
data.py        Oxford-IIIT Pet, split estratificado, aumento de dados
models.py      PetCNN (do zero) e MobileNetV3Transfer (transferência)
train.py       loop de treino, parada antecipada por F1 macro, checkpoints
evaluate.py    acurácia, F1 macro, matriz de confusão PNG, tabela comparativa
gradcam.py     mapa de atenção, sobreposição na foto original
predict.py     inferência + regra de decisão por confiança
smoke_test.py  verificação sem dataset — roda em menos de 1 min
```

`python smoke_test.py` passa integralmente. Rode-o depois de qualquer alteração.

**Dataset:** use `python baixar_dataset.py` na primeira vez, não `data.py` sozinho.
O servidor de Oxford limita a banda por conexao (57 KB/s medidos numa linha de
47 MB/s = mais de 3 h) e ignora requisições de faixa. O script pega as imagens
do CDN do Hugging Face e leva 6 min 25 s (quase tudo sao as anotacoes de
Oxford; as imagens do HF levam menos de 1 min). Resultado esperado: 37 classes,
`treino 3129 | val 551 | teste 3669`.

**Treinado (25/11, GTX 1660 SUPER, torch 2.11+cu128):**

| Modelo | F1 val | Acurácia teste | F1 teste | Latência | Épocas | Tempo |
| --- | --- | --- | --- | --- | --- | --- |
| mobilenet | 0,9330 | **90,4%** | **0,903** | 0,26 ms | parou na 16 de 20 | 8,1 min |
| cnn | 0,5230 | 40,2% | 0,394 | 0,16 ms | 40 de 40 | 26,8 min |

A CNN própria **não** parou antecipadamente: ela ainda estava melhorando na
época 40, o orçamento de épocas é que acabou. Se sobrar tempo, `--epochs 60`
deve render mais. Mas a comparação já é a que interessa para ESW456 — a
diferença entre 0,903 e 0,394 é o argumento da transferência de aprendizado.

**Não use `--num-classes 12`.** O plano de contorno existe para o caso de 37
classes não convergirem, e elas convergiram. Reduzir agora obrigaria a reduzir
as duas redes (restrição 3) e trocaria um 0,903 defensável por um número menor.

As confusões são o material mais interessante da bancada, e as duas redes erram
de maneiras qualitativamente diferentes:

- **mobilenet** erra *dentro* de grupos visualmente próximos: American Pit Bull
  Terrier ↔ Staffordshire Bull Terrier (25x), Bengal → Egyptian Mau (11x). São
  erros que um humano também comete.
- **cnn** erra *atravessando* a fronteira cão/gato: Pomeranian → Ragdoll (27x),
  Samoyed → Ragdoll (23x). Ela não aprendeu nem a distinção mais grosseira.

Os pesos ficam em `checkpoints/` e **estão versionados de propósito** (~14 MB,
sem estado do otimizador). Com eles no repositório, `predict.py` roda em
qualquer máquina sem dataset e sem treinar — é a contingência real para 30/11.

**Ollama:** `llama3.2:3b` (geração e extração) e `nomic-embed-text` (RAG).
Para a extração estruturada do `slm.py`, passe um JSON Schema no campo `format`
da API do Ollama — é o que torna a saída confiável sem pós-processamento.
Verificado: *"ele é meio peludo, sabe"* → `{"pelagem": "media"}` em 0,7 s com o
modelo quente (a primeira chamada leva ~75 s, carregando o modelo; a `api.py`
deve aquecer o SLM na inicialização para o visitante não pagar essa espera).

A regra de decisão já está implementada em `predict.py:ler()` e devolve um objeto
`Leitura` com `palpites`, `decisao` (`afirmar` | `comparar` | `perguntar`), `topo` e
`margem`. **O agente consome esse objeto; não reimplemente a regra.**

Limiares em `predict.py` (calibrar depois de ver a distribuição real):

```python
CONFIANCA_ALTA  = 0.55
MARGEM_AMBIGUA  = 0.15
CONFIANCA_BAIXA = 0.30
```

---

## 3b. Etapas 4 e 5 — construídas em 25/11, funcionando

```
detector.py      acha o cao/gato na foto ANTES de classificar
slm.py           cliente do Ollama: gerar() e extrair() com JSON Schema
rag.py           busca sobre conhecimento/racas.json (TF-IDF, embeddings opcionais)
llm.py           cliente da nuvem; sem chave cai no SLM local
roteador.py      decide local/nuvem por tarefa e conta tudo
agente.py        orquestra, com memória por sessão e reponderação
api.py           FastAPI, 6 rotas + /exemplos
web/index.html   interface única, sem framework
teste_offline.py teste de aceitação da etapa 5
conhecimento/racas.json   37 raças × 7 campos, validado contra class_names
```

`python teste_offline.py` passa: sem Ollama e sem nuvem, classificação, mapa,
decisão e frases curtas continuam; só o perfil longo degrada, com aviso.

**Três decisões que divergem da especificação original, cada uma por medição:**

1. **O RAG usa TF-IDF, não embeddings.** Medimos as duas rotas: o TF-IDF
   ganhou nas três consultas de teste *e* é 700x mais rápido (3 ms contra
   2096 ms). O `nomic-embed-text` é treinado sobretudo em inglês, e a base
   está em português. `rag.USAR_EMBEDDINGS = True` reativa a outra rota. Ter
   medido e escolhido é um argumento melhor na arguição do que usar embeddings
   por serem o esperado.

2. **O guardrail da restrição 1 é verificado, não só pedido.**
   `slm.violou_guardrail()` barra o texto se aparecer "ascendência",
   "genética", "composição", "DNA", "linhagem" e afins; o roteador então usa
   um texto de reserva escrito à mão. Isso existe porque aconteceu: o prompt
   dizia "nunca diga ascendência genética" e o modelo de 3B respondeu
   *"não conseguiu identificar a ascendência genética"*. Citar a palavra
   proibida a torna disponível — o prompt agora é só positivo.

3. **O esquema de refinamento usa um enum autodescritivo.** Com `campo` e
   `valor` separados, "pelo enorme" virava `{"campo":"pelagem","valor":"grande"}`:
   JSON válido, resposta errada. Com `pelo_longo` / `porte_grande` o erro não é
   representável. 5 de 5 casos corretos, incluindo recusar com `nao_sei`.

**O detector existe por causa da bancada.** O visitante nao traz o pet: ele
mostra uma foto na tela do proprio celular. O quadro vem com a pessoa, a
parede e a moldura do aparelho, e o `CenterCrop(224)` do pipeline entrega o
rosto da pessoa para a rede. Medido numa cena em que o pet ocupa 8% do quadro:

| | Resultado | Decisao |
| --- | --- | --- |
| sem detector | Siamese 8,5% | perguntar (inutil) |
| com detector | **Pug 81,0%** | afirmar |

E ele nao custa nada no caso normal — em 40 fotos do dataset, onde o animal ja
ocupa o quadro, o acerto subiu de 34/40 para **37/40**. Custa ~45 ms.

De quebra, reforca a privacidade: a pessoa que segura o celular e descartada
em memoria antes de qualquer outra coisa. O rosto dela nao chega no
classificador nem aparece no mapa de atencao.

A especie que o detector devolve (`cao`/`gato`) e so informativa e **nao vai
para a tela**: o COCO e generico e erra entre as duas com frequencia — na cena
de teste chamou um Pug de gato. Quem sabe a especie e a raca classificada.

**O chat nao adivinha a rota, e isso foi medido.** `POST /perguntar` responde
toda pergunta LOCALMENTE primeiro; sob a resposta o visitante pode pedir para
aprofundar, e so ai a nuvem entra. Tentamos decidir sozinhos de duas formas,
e as duas falharam:

| Tentativa | Resultado |
| --- | --- |
| limiar de similaridade do RAG | nao separa: base 0,027–0,343, raciocinio 0,044–0,129 |
| SLM classificando em 4 tipos | 6/12 |
| SLM com booleano `precisa_pensar` | 7/12, dizendo `true` quase sempre (degenerado) |

Deixar a escolha visivel e melhor que acertar escondido: o visitante ve as
duas rotas, sente a diferenca de latencia e profundidade, e nenhum roteamento
errado e possivel. A tese de ESW489 deixa de ser afirmacao de banner e vira
demonstracao.

A unica regra automatica e deterministica: se a pergunta citar OUTRA raca pelo
nome, os trechos das duas entram na recuperacao. Casamento exato contra
`class_names`, entao nao erra.

**Latencia do chat local: ~3 s** no `llama3.2:3b` com a GTX 1660 SUPER (2-3 s
tipicos, 62 tok/s de geracao, 100% GPU confirmado por `ollama ps`). Reduzir o
contexto do RAG nao ajuda -- medimos k=4, 3 e 2 e deu 2,8 / 2,9 / 3,1 s: o
custo esta na geracao, nao no prompt. Para reduzir a espera PERCEBIDA o
caminho e streaming, nao menos contexto.

**"Reclassificar" significa reponderar.** A CNN não aceita texto. O campo
extraído multiplica os pesos das raças que a base descreve daquele jeito e
renormaliza — a rede continua sendo a única evidência visual, o texto desempata.
Não descarta candidatas: o visitante pode se enganar sobre o próprio pet.

**O que ainda falta:** a chave da API (o sistema funciona sem ela, com o perfil
longo pelo SLM local) e a calibração dos limiares em fotos de celular.

---

## 4. Especificação original das etapas 4 e 5

Mantida como referência do que foi pedido.

### Etapa 4 — API e interface web

**`api.py`** (FastAPI), carregando o modelo uma vez na inicialização:

| Rota | Entrada | Saída |
| --- | --- | --- |
| `GET /` | — | serve `web/index.html` |
| `GET /saude` | — | `{modelo, dispositivo, classes, slm_disponivel, nuvem_disponivel}` |
| `POST /classificar` | multipart `foto` | ver contrato abaixo |
| `POST /responder` | `{sessao, texto}` | nova `Leitura` após o refinamento |
| `POST /perfil` | `{sessao, raca}` | `{texto, rota, latencia_ms, fonte}` |
| `GET /metricas` | — | contadores acumulados do evento |

Contrato de `POST /classificar`:

```json
{
  "sessao": "uuid",
  "palpites": [{"raca": "Pug", "probabilidade": 0.62}],
  "decisao": "comparar",
  "topo": 0.62,
  "margem": 0.08,
  "mapa_atencao": "data:image/png;base64,...",
  "mensagem": "texto curto gerado pelo SLM local",
  "pergunta": "o pelo dele é curto ou longo?",
  "latencia_ms": {"percepcao": 84, "gradcam": 120, "linguagem": 60},
  "rota_linguagem": "local"
}
```

`pergunta` vem preenchida apenas quando `decisao == "perguntar"`.

**`web/index.html`** — página única, sem framework. Deve conter:

- envio por arquivo **e** captura por câmera (`<input type="file" accept="image/*" capture>`);
- foto original e mapa de atenção lado a lado;
- barras de confiança dos três primeiros palpites, **com o percentual visível**;
- caixa de pergunta de refinamento quando `decisao == "perguntar"`;
- painel de métricas no rodapé: total de fotos, % resolvido localmente, latência média
  por rota — esse painel é a entrega visível de ESW489, não corte;
- galeria de fotos de exemplo, para visitantes sem pet.

### Etapa 5 — agente, RAG, SLM local e roteamento

**`conhecimento/racas.json`** — uma entrada por raça com os campos
`temperamento`, `porte`, `energia`, `pelagem`, `cuidados`, `sinais_distintivos`.
Escreva de forma sóbria e verificável. É esta base que impede alucinação.

**`rag.py`** — embeddings via Ollama (`nomic-embed-text`), similaridade de cosseno
sobre numpy. São poucas dezenas de trechos; não instale banco vetorial para isso.
Mantenha um fallback TF-IDF do scikit-learn para quando o Ollama estiver fora.

**`slm.py`** — cliente do Ollama. Duas funções:
`gerar(prompt)` para texto curto e `extrair(texto, esquema)` para saída estruturada
em JSON. A segunda é o que converte "ele é meio peludo, sabe" em
`{"pelagem": "media"}` — esse é o caso de uso canônico de SLM e a entrega mais
concreta de ESW489.

**`llm.py`** — cliente do LLM em nuvem, com timeout curto e fallback para o SLM
local. Recebe nome da raça + trechos do RAG. **Nunca recebe imagem.**

**`roteador.py`** — decide a rota por tarefa:

| Tarefa | Rota | Motivo |
| --- | --- | --- |
| legenda curta, provocação | local | dezenas de chamadas, precisa ser rápida |
| extração estruturada | local | tarefa fechada, saída em JSON |
| pergunta de refinamento | local | latência percebida pelo visitante |
| perfil completo da raça | nuvem | texto longo, uma vez por visitante |
| qualquer tarefa, sem rede | local | modo degradado |

Registre em todas as rotas: contagem, latência, tokens estimados. Exponha em
`GET /metricas`.

**`agente.py`** — orquestra tudo, com memória por sessão. Ferramentas:
`classificar(foto)`, `explicar(foto)` e `buscar_perfil(raca)`. O comportamento vem
da `decisao` devolvida por `predict.ler()`:

- `afirmar` → apresenta a raça e busca o perfil;
- `comparar` → declara a ambiguidade e explica como distinguir as duas candidatas;
- `perguntar` → formula uma pergunta, recebe texto livre, extrai campo estruturado
  pelo SLM e reclassifica.

---

## 5. Convenções do código

- Português nos nomes de domínio (`Leitura`, `palpites`, `decisao`, `roteador`) e
  nos comentários. Inglês só onde já é padrão da biblioteca.
- Comentários explicam **por que**, não o que a linha faz.
- Sem framework de frontend. Sem framework de agente. O fluxo é pequeno e o tempo
  é curto; código próprio é mais rápido de depurar numa bancada.
- Todo módulo novo ganha um bloco `if __name__ == "__main__":` que o exercita sozinho.
- Toda função que fala com rede tem timeout e fallback.

## 6. Como verificar

```bash
python smoke_test.py                       # depois de qualquer alteração
python evaluate.py --compare               # números do banner
uvicorn api:app --reload                   # sobe a API
```

Teste de aceitação da etapa 5: **desligue a rede e use o sistema inteiro**.
Classificação, mapa, decisão e respostas curtas continuam. Só o perfil longo
mostra um aviso. Se qualquer outra coisa quebrar, o roteamento está errado.

## 7. Prazo

| Data | Entrega |
| --- | --- |
| 25/11 qua | Cadastro na planilha. CNN treinando. |
| 26/11 qui | Duas redes treinadas e comparadas. Grad-CAM validado em fotos reais. |
| 27/11 sex | **Trava do cadastro.** Etapa 4 funcionando. Banner para impressão. |
| 28/11 sáb | Etapa 5: agente, RAG, SLM local, roteamento. |
| 29/11 dom | Testes com pessoas de fora. Vídeo de contingência. Pitch. |
| 30/11 seg | Exposição. |

**Plano de contorno:** se 37 raças não convergirem, use `--num-classes 12`. Um
sistema que acerta consistentemente em 12 classes demonstra melhor do que um que
erra muito em 37. A flag já existe; não é preciso mexer em mais nada.

**Ordem inegociável:** nada da camada de linguagem antes da classificação funcionar
com fotos de celular de verdade.
