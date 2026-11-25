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
data.py        download, split estratificado, aumento de dados
models.py      PetCNN (do zero) e MobileNetV3Transfer (transferência)
train.py       loop de treino, parada antecipada, checkpoints
evaluate.py    métricas, matriz de confusão, tabela comparativa
gradcam.py     mapa de atenção sobre a foto original
predict.py     inferência e a regra de decisão por confiança
smoke_test.py  verificação rápida sem dataset
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

## O que ainda falta (etapa 5)

API FastAPI, interface web, base de RAG, SLM local via Ollama e a camada de
roteamento. Essa parte entra depois que a classificação estiver funcionando
com fotos reais — não antes.

## Uma regra que vale mais que o código

Nada da camada de linguagem é tocado antes de a classificação funcionar com
fotos de celular de verdade. Tire cinco fotos de pets conhecidos, rode o
`predict.py` nelas e veja o que acontece. É aí que aparecem os problemas que
importam.
