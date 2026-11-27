# Pet Detetive

Identificação de raças de cães e gatos com redes neurais convolucionais e agente inteligente local.
5º Workshop de Trabalhos de IA da FASOFT · UniRV · 30/11/2026.

---

## Instalação

```bash
cd petdetetive
python -m venv .venv
source .venv/bin/activate        # Windows:  .venv\Scripts\activate
```

O PyTorch vem antes do resto, porque a roda depende do hardware. Escolha **uma**:

```bash
# Com GPU NVIDIA — confira a arquitetura com nvidia-smi:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Sem GPU NVIDIA — muito menor (~200 MB em vez de ~2,5 GB), mas o treino
# passa de 8 minutos para algo entre 20 e 60:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

### Camada de linguagem (etapas 4 e 5)

O SLM local roda no [Ollama](https://ollama.com). Dois modelos, ~2,3 GB no total:

```bash
# Windows:  winget install Ollama.Ollama
ollama pull llama3.2:3b         # geração e extração estruturada
ollama pull nomic-embed-text    # embeddings do RAG
```

O `llama3.2:3b` é a escolha mínima que ainda atende às duas tarefas: o Llama 3.2
tem português entre os idiomas oficialmente suportados, e o Ollama aceita um
JSON Schema no campo `format`, o que dá saída estruturada confiável sem
pós-processamento. Medido nesta máquina: **0,7 s** por extração com o modelo
quente, ~75 s na primeira chamada (carregamento). Se a VRAM apertar, o
`llama3.2:1b` serve de contorno, com perda de qualidade na extração.

### Rodando em outra máquina

O repositório traz os pesos treinados em `checkpoints/`, cerca de 14 MB cada e
sem estado do otimizador. Por isso:

| O que você quer fazer | Precisa do dataset (790 MB)? | Precisa treinar? |
| --- | --- | --- |
| `predict.py`, `gradcam.py`, `api.py` | não | não |
| `evaluate.py` | sim | não |
| `train.py` | sim | — |

Para classificar fotos numa máquina nova, bastam o `.venv` e o Ollama. O dataset
só entra se você for treinar de novo ou recalcular as métricas do banner.

## Antes de qualquer coisa: o teste de fumaça

Roda sem dataset, com imagens sintéticas, em menos de um minuto. Se ele passar,
o código está sadio e você pode gastar tempo baixando os dados.

```bash
python smoke_test.py
```

## Etapa 1 — dados

```bash
python baixar_dataset.py
python data.py
```

Deve imprimir 37 classes e os tamanhos dos três conjuntos:
`treino 3129 | val 551 | teste 3669`.

**Não use `python data.py` sozinho na primeira vez.** Ele funciona, mas baixa as
imagens do servidor de Oxford, que limita a banda por conexão: medimos 57 KB/s
numa linha de 47 MB/s, o que dá mais de **três horas**. O servidor também ignora
requisições de faixa, então paralelizar não resolve.

O `baixar_dataset.py` pega as mesmas imagens do CDN do Hugging Face e só as
anotações (19 MB) de Oxford, montando a árvore que o `torchvision` espera. Ele
confere MD5 das anotações, a contagem de imagens e as linhas dos arquivos de
split. Depois dele, `data.py` não toca na rede.

Medido numa linha de 47 MB/s: **6 min 25 s**. Quase tudo são os 19 MB de
anotações vindos de Oxford — as 790 MB de imagens do Hugging Face levam menos de
um minuto. Não há como fugir desse trecho: os arquivos de split não existem em
outra fonte.

## Etapa 2 — treinar as duas redes

A ordem importa. Comece pela MobileNetV3: ela converge rápido e te diz em
vinte minutos se o pipeline está correto. A CNN própria demora mais e serve
para a comparação, não para descobrir bugs.

```bash
python train.py --model mobilenet --epochs 20
python train.py --model cnn       --epochs 40
```

Opções úteis:

| Flag | Para que serve |
| --- | --- |
| `--num-classes 12` | **Plano de contorno.** Treina só com as 12 primeiras raças. Use se 37 não convergir a tempo. |
| `--batch-size 16` | Se faltar memória. |
| `--device cpu` | Força CPU. |
| `--patience 4` | Para antes, se o tempo apertar. |

Sem GPU, a MobileNetV3 leva algo entre 20 e 60 minutos para 20 épocas.
Se estiver inviável, use o Google Colab com GPU: suba a pasta, rode o mesmo
comando e baixe a pasta `checkpoints/`.

## Etapa 3 — avaliar e gerar as figuras do banner

```bash
python evaluate.py --compare
```

Gera em `outputs/`:

- `<modelo>_confusao.png` — matriz de confusão, vai direto para o banner
- `<modelo>_metricas.json` — acurácia, F1 macro, latência, parâmetros
- `<modelo>_por_classe.csv` — precisão e revocação por raça
- `comparacao.md` — a tabela comparativa, pronta para colar

Olhe o campo `piores_confusoes` do JSON. É o material mais interessante que
você vai ter para explicar na bancada: quais raças a rede troca entre si e por quê.

## Etapa 4 — classificar uma foto e ver o mapa de atenção

```bash
python predict.py minha_foto.jpg --model mobilenet
python gradcam.py minha_foto.jpg --model mobilenet --out mapa.png
```

O `predict.py` já imprime a decisão do agente: `afirmar`, `comparar` ou
`perguntar`, conforme a confiança.

---

## Estrutura

```
Percepção (etapas 1 a 3)
  baixar_dataset.py  monta o Oxford-IIIT Pet rápido (HF + anotações)
  data.py            split estratificado, aumento de dados
  models.py          PetCNN (do zero) e MobileNetV3Transfer (transferência)
  train.py           loop de treino, parada antecipada, checkpoints
  evaluate.py        métricas, matriz de confusão, tabela comparativa
  gradcam.py         mapa de atenção sobre a foto original
  predict.py         inferência e a regra de decisão por confiança

Linguagem (etapas 4 e 5)
  slm.py             Ollama: gerar() e extrair() com JSON Schema
  rag.py             busca sobre conhecimento/racas.json
  llm.py             nuvem; sem chave cai no SLM local
  roteador.py        decide local/nuvem por tarefa e conta tudo
  agente.py          orquestra, memória por sessão, reponderação
  api.py             FastAPI
  web/index.html     interface única, sem framework

Verificação
  smoke_test.py      rápida, sem dataset
  teste_offline.py   aceitação da etapa 5: sistema sem rede

Dados
  conhecimento/racas.json   37 raças × 7 campos
```

## Ajuste da regra de decisão

Os limiares ficam no topo de `predict.py`:

```python
CONFIANCA_ALTA  = 0.55   # acima disso, afirma a raça
MARGEM_AMBIGUA  = 0.15   # 1º e 2º próximos: declara a ambiguidade
CONFIANCA_BAIXA = 0.30   # topo abaixo disso: pergunta ao visitante
```

Calibre depois de ver a distribuição real de confianças no conjunto de teste.
Se o sistema ficar perguntando demais na bancada, suba `CONFIANCA_BAIXA`.

## Etapas 4 e 5 — subir o sistema

```bash
uvicorn api:app --reload          # só nesta máquina
python api.py --rede              # aceita celulares da mesma rede
```

E abra `http://localhost:8000`. A API carrega o modelo e o índice uma vez e
**aquece o SLM na inicialização** — sem isso o primeiro visitante esperaria uns
75 segundos pela primeira frase. Sobe em cerca de 4 segundos.

### Pelo celular

Com `--rede`, a API imprime o endereço (algo como `http://192.168.0.14:8000`) e
qualquer celular no mesmo Wi-Fi abre a página. O layout é responsivo.

Mas **a câmera ao vivo não funciona por IP**: o `getUserMedia` exige contexto
seguro — HTTPS ou `localhost`. É regra do navegador, não configuração. Pelo
celular o caminho é "Escolher uma foto", e o seletor de arquivo do celular já
oferece a câmera no próprio menu do sistema. Por isso não existe um botão
separado de câmera para celular: seria redundante.

| Como você acessa | Escolher uma foto | Ligar a câmera |
| --- | --- | --- |
| `localhost` no notebook da bancada | sim | **sim** |
| IP da rede, pelo celular | sim (abre a câmera pelo menu) | não |

**Não use túnel (ngrok e parecidos) para contornar isso.** Daria HTTPS e
liberaria a câmera, mas as fotos passariam a trafegar pela internet — o que
quebra exatamente a tese de privacidade do projeto.

| Rota | O que faz |
| --- | --- |
| `GET /` | a interface |
| `GET /saude` | modelo, dispositivo, e se SLM e nuvem estão no ar |
| `POST /classificar` | foto → palpites, mapa de atenção, decisão, frase |
| `POST /responder` | texto livre do visitante → reponderação dos palpites |
| `POST /perfil` | perfil longo da raça |
| `POST /perguntar` | pergunta aberta sobre o animal; `aprofundar: true` manda para a nuvem |
| `GET /metricas` | contadores do evento |
| `GET /exemplos` | quantas fotos há para sortear |
| `GET /exemplos/aleatorio` | uma foto qualquer do banco, com a raça verdadeira no cabeçalho |

### O teste que vale mais que os outros

```bash
python teste_offline.py
```

Simula o Ollama e a nuvem fora do ar. Classificação, mapa, decisão e frases
curtas têm que continuar funcionando; só o perfil longo degrada, e com aviso.
Se qualquer outra coisa quebrar, o roteamento está errado.

### A chave da API

Clique na **engrenagem**, no canto do cabeçalho, e cole a chave. O provedor é
reconhecido pelo prefixo:

| Prefixo | Provedor | Modelo padrão |
| --- | --- | --- |
| `sk-ant-` | Anthropic (Claude) | `claude-opus-5-5` |
| `sk-or-` | OpenRouter | `anthropic/claude-sonnet-4.5` |
| `gsk_` | Groq | `llama-3.3-70b-versatile` |
| `AIza` | Google Gemini | `gemini-2.0-flash` |
| `sk-proj-`, `sk-` | OpenAI | `gpt-4o-mini` |

A chave é **testada antes de ser gravada** — uma chamada mínima ao provedor. Se
não servir, o erro dele aparece na tela e nada é salvo. O campo de modelo é
opcional: vazio usa o padrão da tabela.

Fica em `.env`, que está no `.gitignore`. **A tela só abre em `localhost`**: com
`--rede`, a API recusa `/configuracao` vindo de fora, porque ler ou gravar a
chave pela rede seria entregá-la a quem estiver na sala. E a chave nunca volta
do servidor — só a máscara (`sk-ant-…1234`), o provedor e o modelo.

Sem chave o sistema **continua funcionando**: perfil e respostas saem do modelo
local, e a interface mostra a rota usada. Essa degradação é por especificação,
não um defeito — `GET /saude` informa `nuvem_disponivel: false`.

## Uma regra que vale mais que o código

Nada da camada de linguagem é tocado antes de a classificação funcionar com
fotos de celular de verdade. Tire cinco fotos de pets conhecidos, rode o
`predict.py` nelas e veja o que acontece. É aí que aparecem os problemas que
importam.
