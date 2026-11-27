# STATUS — Pet Detetive

Levantamento do que existe no código em **27/11/2026**. Descreve o estado real,
não o planejado. Onde algo não existe, está escrito "não implementado".

---

## 1. Repositório

| | |
| --- | --- |
| Remote | `https://github.com/muddyorc/pet_detetive.git` (fetch e push) |
| Branch | `main` |
| Commits | 19 |
| HEAD | `3cec1c6`, igual ao `origin/main` |
| Pendências locais | nenhuma |
| Tamanho do clone | 19 MB |

### Árvore (excluindo `data/`, `checkpoints/`, `outputs/`, `.venv/`, `__pycache__/`)

```
.gitignore
.vscode/launch.json
CLAUDE.md
INSTALAR.md
PetDetetive_Briefing_ClaudeCode.pdf
README.md
STATUS.md
agente.py
api.py
baixar_dataset.py
conhecimento/racas.json
data.py
detector.py
evaluate.py
gradcam.py
llm.py
models.py
predict.py
rag.py
requirements.txt
roteador.py
slm.py
smoke_test.py
teste_offline.py
train.py
web/exemplos/LEIA.md
web/index.html
```

`checkpoints/` é versionado por decisão (ver seção 4): contém `mobilenet.pt`,
`cnn.pt`, os dois `*_history.json` e `class_names.json`.

---

## 2. Módulos

| Arquivo | O que faz | No caminho da demo? |
| --- | --- | --- |
| `api.py` | FastAPI: 13 rotas, carrega modelo/detector/RAG e aquece o SLM na inicialização | **sim** |
| `agente.py` | Orquestra percepção e linguagem, memória por sessão, reponderação | **sim** |
| `detector.py` | Localiza o cão/gato na foto e recorta antes de classificar | **sim** |
| `predict.py` | Carrega checkpoint, infere e aplica a regra de decisão | **sim** |
| `gradcam.py` | Mapa de atenção sobre a imagem classificada | **sim** |
| `models.py` | `PetCNN` e `MobileNetV3Transfer` | **sim** (carregado por `predict`) |
| `roteador.py` | Decide local/nuvem por tarefa, aplica guardrail, conta métricas | **sim** |
| `slm.py` | Cliente do Ollama: `gerar`, `extrair`, `embeddings`, guardrail | **sim** |
| `llm.py` | Cliente de nuvem, 5 provedores, detecção de chave, fallback local | **sim** |
| `rag.py` | Busca sobre `conhecimento/racas.json` | **sim** |
| `conhecimento/racas.json` | 37 raças × 7 campos | **sim** |
| `web/index.html` | Interface única, sem framework | **sim** |
| `data.py` | Split estratificado e dataloaders | não (só treino/avaliação) |
| `train.py` | Loop de treino com parada antecipada por F1 macro | não |
| `evaluate.py` | Métricas, matriz de confusão, tabela comparativa | não |
| `baixar_dataset.py` | Monta o Oxford-IIIT Pet via CDN do Hugging Face | não |
| `smoke_test.py` | Verificação sem dataset | não |
| `teste_offline.py` | Teste de aceitação sem Ollama e sem nuvem | não |

---

## 3. Dependências reais

`pip freeze` filtrado pelos pacotes de terceiros que o código de fato importa:

```
anthropic==2.14.0
fastapi==0.165.0
matplotlib==3.11.2
numpy==2.5.4
pillow==13.0.0
pyarrow==26.0.1
pydantic==2.14.4
python-dotenv==1.2.4
requests==2.34.2
scikit-learn==1.9.1
torch==2.11.0+cu128
torchvision==0.26.0+cu128
uvicorn==0.54.0
```

Python 3.14.7, Windows 10.

---

## 4. Treino

Houve treino das duas redes.

### Checkpoints

| Modelo | Melhor época | F1 macro (validação) | Classes |
| --- | --- | --- | --- |
| `mobilenet.pt` | 10 | **0,9330** | 37 |
| `cnn.pt` | 47 | **0,6019** | 37 |

### Históricos

**`mobilenet_history.json`** — 16 épocas rodadas (de 20; parada antecipada
após 6 sem ganho de F1 macro)

| | Perda treino | Perda val | Acurácia val | F1 macro val |
| --- | --- | --- | --- | --- |
| Época 1 | 1,839 | 1,104 | 0,897 | 0,894 |
| Época 16 | 0,730 | 0,891 | 0,931 | 0,931 |

**`cnn_history.json`** — 57 épocas rodadas (de 60; parada antecipada após 10
sem ganho)

| | Perda treino | Perda val | Acurácia val | F1 macro val |
| --- | --- | --- | --- | --- |
| Época 1 | 3,584 | 3,514 | 0,045 | 0,025 |
| Época 57 | 1,871 | 1,877 | 0,593 | 0,588 |

A CNN própria foi treinada duas vezes: a primeira com 40 épocas terminou em
F1 0,523 **sem** parada antecipada — o orçamento de épocas acabou antes da
convergência. A segunda, com 60 épocas e paciência 10, convergiu em 0,6019.

Os pesos são versionados de propósito: ~14 MB e ~4,8 MB, sem estado do
otimizador. Com eles no repositório, `predict.py` roda em qualquer máquina sem
dataset e sem treinar.

---

## 5. Avaliação

Conjunto de teste: **3669 imagens**, 37 classes, inferência em CUDA.

### `outputs/mobilenet_metricas.json`

```
acuracia                 0.9043336058871627
f1_macro                 0.9026164598289405
latencia_ms_por_imagem   0.2818455709558606
parametros               3 482 965
dispositivo              cuda
piores_confusoes:
  American Pit Bull Terrier   -> Staffordshire Bull Terrier   25x
  Staffordshire Bull Terrier  -> American Bulldog             14x
  Staffordshire Bull Terrier  -> American Pit Bull Terrier    12x
  American Pit Bull Terrier   -> American Bulldog             11x
  Bengal                      -> Egyptian Mau                 11x
```

### `outputs/cnn_metricas.json`

```
acuracia                 0.4769692014172799
f1_macro                 0.46363351043737194
latencia_ms_por_imagem   0.17099264101956613
parametros               1 182 725
dispositivo              cuda
piores_confusoes:
  Great Pyrenees              -> Samoyed              31x
  Miniature Pinscher          -> Boxer                30x
  Scottish Terrier            -> Newfoundland         28x
  Staffordshire Bull Terrier  -> German Shorthaired   28x
  Pug                         -> American Bulldog     27x
```

### `outputs/comparacao.md`

| Modelo | Acuracia | F1 macro | Latencia | Parametros |
| --- | --- | --- | --- | --- |
| cnn | 47.7% | 0.464 | 0.17 ms | 1,182,725 |
| mobilenet | 90.4% | 0.903 | 0.28 ms | 3,482,965 |

### Figuras existentes em `outputs/`

| Arquivo | Tamanho |
| --- | --- |
| `mobilenet_confusao.png` | 173 737 B |
| `cnn_confusao.png` | 176 622 B |
| `mobilenet_por_classe.csv` | 1 458 B |
| `cnn_por_classe.csv` | 1 458 B |

`outputs/` está no `.gitignore` — as figuras não vêm no clone e são
regeneradas com `python evaluate.py --compare`, que exige o dataset.

**Observação sobre as confusões:** as duas redes erram de formas
qualitativamente diferentes. A MobileNetV3 erra dentro de grupos visualmente
próximos (Pit Bull ↔ Staffordshire, Bengal → Egyptian Mau). A CNN própria erra
atravessando a fronteira cão/gato em versões anteriores e, nesta, confunde
raças de portes e espécies distintas (Scottish Terrier → Newfoundland, Pug →
American Bulldog).

---

## 6. Camada de linguagem

### SLM local

| | |
| --- | --- |
| Modelo | `llama3.2:3b` |
| Parâmetros | 3,2 B |
| Quantização | **Q4_K_M** |
| Contexto | 131 072 |
| Servidor | Ollama em `http://localhost:11434` |
| Modelo de embedding baixado | `nomic-embed-text` |

Configurável por `PET_SLM_MODELO`, `PET_SLM_EMBED` e `OLLAMA_HOST`.

### LLM de nuvem

| | |
| --- | --- |
| Provedor ativo | **Google Gemini** |
| Modelo ativo | **`gemini-3.5-flash-lite`** |
| Onde está gravado | `.env` (`PET_LLM_CHAVE`, `PET_LLM_PROVEDOR`, `PET_LLM_MODELO`), fora do git |

Cinco provedores suportados, detectados por prefixo de chave, com lista manual
para quando a detecção falhar:

| Prefixo | Provedor | Transporte |
| --- | --- | --- |
| `sk-ant-` | Anthropic | SDK `anthropic` |
| `sk-or-` | OpenRouter | HTTP `/chat/completions` |
| `gsk_` | Groq | HTTP `/chat/completions` |
| `AQ.`, `AIza` | Google Gemini | REST `generateContent` |
| `sk-proj-`, `sk-` | OpenAI | HTTP `/chat/completions` |

### RAG

Existe.

| | |
| --- | --- |
| Rota ativa | **TF-IDF** (`sklearn.TfidfVectorizer`, `analyzer="char_wb"`, n-gramas 3–5) |
| Trechos | **259**, de 37 raças (7 campos por raça) |
| Armazenamento | matriz numpy densa em memória, shape **(259, 7678)**, normalizada; similaridade de cosseno |
| Embeddings | implementados mas **desligados** (`rag.USAR_EMBEDDINGS = False`) |
| Cache de vetores | `conhecimento/_vetores.npz`, só usado na rota de embeddings; fora do git |

Não há banco vetorial. A rota de embeddings usa `nomic-embed-text` via Ollama
quando ligada.

### Roteamento — pseudocódigo

```
legenda, pergunta de refinamento, extração estruturada  -> SLM local
perfil da raça       -> cache se já gerado; senão SLM local; senão base crua
pergunta aberta      -> nuvem se houver chave; senão SLM local
sem Ollama           -> texto de reserva escrito à mão
toda saída de modelo -> violou_guardrail(texto)? então descarta e usa a reserva
```

### O que não existe na camada de linguagem

- **Roteamento automático por tipo de pergunta** — não implementado. Foi
  tentado de duas formas e as duas falharam (seção 10). A escolha é do
  visitante, por botão.
- **Explicação comparativa** no caminho `comparar` — **não implementado**. O
  código declara a ambiguidade e cita as duas raças, mas não explica como
  distingui-las.
- **Streaming de resposta** — não implementado.
- **Persistência de sessões entre reinícios** — não implementado; a memória é
  um `OrderedDict` em processo, limitado a 200 sessões.

---

## 7. API e interface

### Rotas implementadas

| Método | Caminho | Entrada | Saída |
| --- | --- | --- | --- |
| GET | `/` | — | `web/index.html` |
| GET | `/saude` | — | modelo, dispositivo, classes, detector, SLM, nuvem, RAG |
| POST | `/classificar` | multipart `foto`, query `sessao` opcional | palpites, decisão, topo, margem, mapa de atenção, recorte, detecção, mensagem, pergunta, latências, rota |
| POST | `/responder` | `{sessao, texto}` | nova leitura após reponderação, `entendido` |
| POST | `/perfil` | `{sessao, raca}` | `{texto, rota, latencia_ms, fonte}` |
| POST | `/perguntar` | `{sessao, pergunta, local}` | `{texto, rota, latencia_ms, comparou_com, trechos_usados, rag_ms, pode_comparar}` |
| GET | `/configuracao` | — (só localhost) | estado da chave: provedor, modelo, máscara, lista de provedores |
| POST | `/configuracao` | `{chave, modelo, provedor, base}` (só localhost) | resultado do teste da chave |
| POST | `/configuracao/modelos` | `{chave, provedor, base}` (só localhost) | modelos que a chave pode usar |
| GET | `/metricas` | — | contadores por rota e por tarefa |
| GET | `/exemplos` | — | quantidade de fotos sorteáveis e fonte |
| GET | `/exemplos/aleatorio` | — | uma foto, com `X-Raca-Verdadeira` no cabeçalho |
| GET | `/exemplos/{nome}` | nome do arquivo | foto de `web/exemplos/` |

### Interface

Existe, em `web/index.html`, página única sem framework. Verificado por
inspeção dos identificadores no HTML:

| Elemento | Estado |
| --- | --- |
| Envio por arquivo | implementado |
| Câmera ao vivo com recorte quadrado | implementado |
| Sorteio de foto do banco | implementado |
| Gabarito no sorteio | implementado |
| Barras de confiança com percentual | implementado |
| Mapa de atenção lado a lado com o recorte | implementado |
| Pergunta de refinamento | implementado |
| Perfil da raça | implementado |
| Chat sobre o animal | implementado |
| Botão de comparar rotas | implementado |
| Diálogo da chave de API | implementado |
| Painel de métricas no rodapé | implementado |
| Pastilhas de estado no cabeçalho | implementado |

---

## 8. Fluxo real de uma requisição

O que o código executa quando o visitante envia uma foto:

1. **`api.classificar`** lê o upload para `io.BytesIO` e abre com
   `PIL.Image.open`. Recusa acima de 12 MB. Nunca grava em disco.
2. **`agente.Agente.ver_foto`** recebe a `PIL.Image` e a sessão.
3. **`detector.localizar`** roda o SSDLite320-MobileNetV3 (COCO), pega a caixa
   de maior confiança entre `cat` e `dog`, folga 12% e recorta. Se não achar
   nada, segue com a foto inteira.
4. **`agente.classificar`** chama `predict.probabilidades` e `predict.ler`
   sobre o **recorte**. O `ler` aplica a regra de decisão por confiança.
5. **`agente.explicar`** chama `gradcam.explain_image` sobre o mesmo recorte e
   converte para data URI com `gradcam.to_png_bytes`.
6. A sessão guarda o **vetor de probabilidades**, não a imagem.
7. **`roteador.legenda`** monta o pedido conforme a decisão e chama
   `slm.gerar`. A saída passa por `slm.violou_guardrail`; se violar, usa o
   texto de reserva.
8. Se a decisão foi `perguntar`, **`roteador.pergunta`** gera a pergunta de
   refinamento pelo mesmo caminho local.
9. **`roteador.metricas.registrar_foto`** registra decisão e latências.
10. A resposta volta com palpites, decisão, mapa de atenção, recorte em base64,
    dados da detecção, mensagem, pergunta e latências por etapa.

A imagem original sai de escopo ao fim da requisição e é coletada pelo garbage
collector.

---

## 9. Status honesto por etapa

| Etapa | Status |
| --- | --- |
| Dados | **implementado e testado** — `baixar_dataset.py` validado ponta a ponta (7349 imagens, MD5 das anotações conferido); `data.py` imprime 37 classes, 3129/551/3669 |
| CNN própria | **implementado e testado** — 57 épocas, F1 val 0,6019, avaliada no teste |
| MobileNetV3 | **implementado e testado** — 16 épocas, F1 val 0,9330, avaliada no teste |
| Avaliação | **implementado e testado** — métricas, matrizes, CSV por classe e tabela gerados |
| Grad-CAM | **implementado sem teste** — coberto pelo `smoke_test.py` com imagens sintéticas e exercitado em fotos do dataset, mas **não validado em fotos de celular reais** |
| Regra de decisão | **implementado sem teste** — a lógica é coberta pelo `smoke_test.py`, mas os limiares **não foram calibrados** com fotos reais |
| API | **implementado e testado** — 13 rotas, exercitadas manualmente; sem suíte automatizada |
| Interface web | **implementado sem teste** — verificada por inspeção e uso manual; sem teste automatizado |
| Agente | **implementado e testado** — bloco `__main__` exercita o fluxo completo |
| RAG | **implementado e testado** — bloco `__main__` compara as duas rotas |
| SLM local | **implementado e testado** — bloco `__main__` cobre geração, extração e embeddings |
| Roteamento | **implementado e testado** — `teste_offline.py` cobre o modo degradado |
| Métricas na tela | **implementado sem teste** — painel existe e atualiza a cada 8 s; sem teste automatizado |
| Explicação comparativa | **não implementado** |

Testes automatizados que existem e passam hoje: `smoke_test.py` e
`teste_offline.py`. Não há suíte de testes unitários nem CI.

---

## 10. Problemas

### Bugs conhecidos e abertos

1. **Limiares não calibrados.** `CONFIANCA_ALTA = 0.55`, `MARGEM_AMBIGUA = 0.15`
   e `CONFIANCA_BAIXA = 0.30` são valores de partida. Foram pensados para a
   distribuição do conjunto de teste, que é de raças puras bem enquadradas — a
   bancada terá mestiços e fotos de celular. Risco: o sistema afirmar uma raça
   quando deveria perguntar.
2. **Explicação comparativa ausente.** O caminho `comparar` declara a
   ambiguidade e para.
3. **Inferência roda em CPU.** `predict.load_checkpoint` tem `device=None` que
   resolve para `torch.device("cpu")`, e `predict.probabilidades` monta o
   tensor sem mover para o dispositivo do modelo. Passar `device="cuda"`
   quebraria. Não é problema na prática (0,28 ms/imagem medidos em CUDA no
   `evaluate.py`; na API a percepção fica em 23–171 ms em CPU), mas é uma
   limitação real do código.
4. **Sessões em memória.** Reiniciar a API perde todas as sessões. Teto de 200.

### Gargalos com números medidos

| Gargalo | Medido | Contorno |
| --- | --- | --- |
| Download do dataset pelo servidor de Oxford | 57 KB/s numa linha de 47 MB/s = **3 h 33** | `baixar_dataset.py` usa o CDN do Hugging Face: **6 min 25 s** |
| Primeira chamada ao SLM (carga do modelo) | **~75 s** | `api.py` aquece o SLM na inicialização |
| Geração do SLM local | 2,8–4,6 s por resposta | reduzir contexto **não ajuda** (k=4/3/2 → 2,8/2,9/3,1 s); o custo é geração, não prompt |
| Perfil regerado a cada visitante | 4,7 s | cache por raça: **2 ms** a partir da segunda vez |

### O que falhou e como foi contornado

| Falhou | Contorno |
| --- | --- |
| Roteamento por similaridade do RAG | faixas sobrepostas (base 0,027–0,343; raciocínio 0,044–0,129). Abandonado |
| SLM classificando o tipo da pergunta (4 classes) | 6/12. Abandonado |
| SLM com booleano `precisa_pensar` | 7/12, respondendo `true` quase sempre — classificador degenerado. Abandonado |
| Escolha da rota pelo visitante | adotado: botão visível, sem roteamento errado possível |
| Embeddings `nomic-embed-text` no RAG | perdeu do TF-IDF em 3 de 3 consultas e é 700x mais lento (2096 ms × 3 ms). Trocado por TF-IDF |
| Esquema de extração com `campo` + `valor` | "pelo enorme" → `{"campo":"pelagem","valor":"grande"}`: JSON válido, resposta errada. Trocado por enum autodescritivo (`pelo_longo`), 5/5 |
| Prompt pedindo a palavra "pedigree" | `pedigree` está em `TERMOS_PROIBIDOS`; o guardrail descartou a resposta que o prompt pediu. Palavra removida do prompt |
| Limite de assunto apertado demais | pet vomitando virou "só sei falar do bicho da foto". Saúde explicitada como dentro do tema |
| Guardrail por instrução negativa | o prompt dizia "nunca diga ascendência genética" e o modelo respondeu "não conseguiu identificar a ascendência genética". Prompt reescrito em forma positiva + filtro verificado |
| `teste_offline.py` deixou de simular queda | `load_dotenv` repunha a chave após o `pop`. Agora define as variáveis como string vazia |
| Fallback do perfil perdido | ao mover o perfil para local, `SLMIndisponivel` deixou de ser capturado. O `teste_offline.py` cobrou; captura restaurada |

### Hardware

| | |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1660 SUPER, 6144 MiB, driver 616.64 |
| Treino | CUDA (`torch 2.11.0+cu128`) |
| Avaliação | CUDA |
| Inferência na API | **CPU** (ver bug 3) |
| SLM | 100% GPU, 2,9 GB de 6 GB ocupados |
| SO | Windows 10, Python 3.14.7 |

---

## 11. Números medidos

Todos medidos nesta máquina. Nada estimado.

### Treino

| | |
| --- | --- |
| MobileNetV3, 16 épocas | **8,1 min** (~29 s/época) |
| PetCNN, 57 épocas | **38,9 min** (~39 s/época) |
| PetCNN, execução anterior de 40 épocas | 26,8 min |

### Dados

| | |
| --- | --- |
| `baixar_dataset.py` completo | **6 min 25 s** |
| Download pelo servidor de Oxford (medido, não usado) | 57 KB/s, ETA 3 h 33 |
| Velocidade da linha | 47 MB/s |
| CDN do Hugging Face | 37 MB/s |
| Imagens escritas | 7349 |

### Inferência

| | |
| --- | --- |
| MobileNetV3 em CUDA (`evaluate.py`) | **0,28 ms/imagem** |
| PetCNN em CUDA | **0,17 ms/imagem** |
| Detecção na API (CPU) | 36–52 ms |
| Percepção na API (CPU) | 23–171 ms |
| Grad-CAM na API | 46–245 ms |

### Detector

| | |
| --- | --- |
| Acerto sem detector, 40 fotos do dataset | 34/40 |
| Acerto com detector | **37/40** |
| Confiança média sem / com | 81,0% / 81,6% |
| Animal localizado | 39/40 |
| Cena com pet a 8% do quadro, sem detector | Siamese **8,5%**, decisão `perguntar` |
| Mesma cena, com detector | Pug **81,0%**, decisão `afirmar` |

### Linguagem

| | |
| --- | --- |
| Primeira chamada ao SLM (carga) | **~75 s** |
| Extração estruturada, modelo quente | **0,7 s** (2,5–3,9 s com a GPU dividida com treino) |
| Geração do SLM local | 2,8–4,6 s; 62 tok/s de geração, ~184 tok/s de prompt |
| Pergunta aberta pela nuvem | **0,9–1,1 s** |
| Pergunta aberta pelo SLM local | **3,0–3,5 s** |
| Perfil, primeira geração | 4,7 s |
| Perfil em cache | **2 ms** |
| Busca no RAG, TF-IDF | **3 ms** |
| Busca no RAG, embeddings | 2096 ms |

### API

| | |
| --- | --- |
| Inicialização completa | **3,3–4,0 s** |
| Classificação ponta a ponta | 3,3 s (dominada pela legenda do SLM) |
| Página servida | ~34 KB |

### Acertos verificados pela API

| | |
| --- | --- |
| 10 fotos aleatórias do dataset | **10/10** no topo |
| 10 sorteios seguidos | **10/10**, sem foto repetida |
| 12 fotos, imagem inteira vs quadrado central | 10/12 em ambos |
