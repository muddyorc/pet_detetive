# Instalar o Pet Detetive em outra máquina

Passo a passo completo, do zero até classificar uma foto. Os tempos são os
medidos numa GTX 1660 SUPER com linha de 47 MB/s.

Decida primeiro **o que você vai fazer na máquina nova**, porque isso muda
metade dos passos:

| Objetivo | Passos necessários | Tempo |
| --- | --- | --- |
| Só classificar fotos (bancada do evento) | 1 a 5 | ~10 min |
| Também recalcular as métricas do banner | 1 a 6 | ~17 min |
| Também treinar de novo | 1 a 7 | ~30 min |

Os pesos treinados vêm no repositório, então **classificar não exige dataset
nem treino**.

---

## 1. Pré-requisitos

- **Python 3.11 ou mais novo** — testado em 3.14.7
- **Git**
- **GPU NVIDIA** (opcional). Sem ela tudo funciona, mas o treino passa de
  8 minutos para algo entre 20 e 60. Classificar uma foto continua rápido.

Se tiver GPU, anote a arquitetura:

```bash
nvidia-smi
```

---

## 2. Clonar

```bash
git clone https://github.com/muddyorc/pet_detetive.git
cd pet_detetive
```

Confira que os pesos vieram — são o que te poupa o treino:

```bash
ls -l checkpoints/
# mobilenet.pt      ~14 MB
# cnn.pt            ~5 MB
# class_names.json
```

Se a pasta estiver vazia, o `git clone` não trouxe os pesos e você vai precisar
treinar (passo 7).

---

## 3. Ambiente virtual

```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

O prompt deve passar a mostrar `(.venv)`. Todos os comandos seguintes assumem o
ambiente ativo.

---

## 4. PyTorch

**Este passo vem antes do `requirements.txt`**, porque a roda certa depende do
hardware. Escolha **uma** linha:

```bash
# Com GPU NVIDIA (CUDA 12.8 — cobre Turing, Ampere, Ada e Blackwell):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Sem GPU NVIDIA — ~200 MB em vez de ~2,5 GB:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Depois, o resto:

```bash
pip install -r requirements.txt
```

Confirme que a GPU foi reconhecida:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Se imprimir `False` numa máquina com NVIDIA, você instalou a roda de CPU.
Desinstale e repita:

```bash
pip uninstall -y torch torchvision
```

---

## 4b. Pesos do detector de pet (baixa sozinho)

Antes de classificar, o sistema localiza o cão ou gato na foto e recorta — sem
isso, o `CenterCrop(224)` do pipeline entrega o fundo ou o rosto de quem segura
o celular. Os pesos (SSDLite320-MobileNetV3, ~14 MB) vêm do torchvision e são
baixados **na primeira execução**, depois ficam em cache.

Na bancada isso já aconteceu, porque a `api.py` carrega o detector na
inicialização de propósito — não é na frente do visitante que esse download
deve rodar. Mas **numa máquina nova, rode a API uma vez com internet** antes do
evento.

Sem os pesos e sem rede, o sistema usa a foto inteira: funciona pior, mas
funciona.

## 5. Verificar sem dataset

```bash
python smoke_test.py
```

Roda em menos de um minuto, com imagens sintéticas. Exercita transformações,
split, as duas arquiteturas, o loop de treino, o Grad-CAM e a regra de decisão.

**Se este passo passar, o código está sadio.** Rode-o sempre depois de qualquer
alteração — é a regra do `CLAUDE.md`.

Agora já é possível classificar uma foto:

```bash
python predict.py minha_foto.jpg --model mobilenet
python gradcam.py minha_foto.jpg --model mobilenet --out mapa.png
```

Para a bancada do evento, **você pode parar aqui.**

---

## 6. Dataset (só para `evaluate.py` ou `train.py`)

```bash
python baixar_dataset.py
python data.py
```

Deve imprimir:

```
37 classes
treino 3129 | val 551 | teste 3669
```

**Não rode `python data.py` sozinho na primeira vez.** Ele funciona, mas baixa
as imagens do servidor de Oxford, que limita a banda por conexão: 57 KB/s
medidos numa linha de 47 MB/s, ou seja, **mais de três horas**. O servidor
também ignora requisições de faixa, então paralelizar não ajuda.

O `baixar_dataset.py` pega as imagens do CDN do Hugging Face e só as anotações
(19 MB) de Oxford. Leva **6 min 25 s**, e é idempotente: se o dataset já está
montado, não faz nada.

Com o dataset no lugar:

```bash
python evaluate.py --compare
```

Gera em `outputs/` as matrizes de confusão, as métricas em JSON, a precisão e
revocação por raça, e a tabela comparativa.

---

## 7. Treinar de novo (opcional)

Só é necessário se `checkpoints/` estiver vazio ou se você mudar a arquitetura.
A ordem importa: a MobileNetV3 converge rápido e revela erros de pipeline; a CNN
própria é lenta e serve para a comparação, não para achar bugs.

```bash
python train.py --model mobilenet --epochs 20   # ~8 min na 1660 SUPER
python train.py --model cnn       --epochs 40   # ~26 min
```

Se faltar memória de vídeo: `--batch-size 16`. Para forçar CPU: `--device cpu`.

---

## 8. Camada de linguagem (etapas 4 e 5)

Necessária para a `api.py` e o agente; **não** para classificar fotos.

### Ollama

```bash
# Windows:
winget install Ollama.Ollama
# Linux:
curl -fsSL https://ollama.com/install.sh | sh
# macOS: baixe em https://ollama.com
```

Depois, os dois modelos (~2,3 GB no total):

```bash
ollama pull llama3.2:3b         # geração e extração estruturada
ollama pull nomic-embed-text    # embeddings do RAG
```

Confira que o serviço responde:

```bash
curl http://localhost:11434/api/tags
```

Por que esses dois: o Llama 3.2 tem português entre os idiomas oficialmente
suportados, e o Ollama aceita um JSON Schema no campo `format`, o que dá saída
estruturada confiável sem pós-processamento. Se a VRAM apertar, o `llama3.2:1b`
serve de contorno, com perda de qualidade na extração.

**A primeira chamada ao modelo leva cerca de 75 segundos** (carregamento); as
seguintes, menos de 1 segundo. Por isso a `api.py` aquece o SLM na
inicialização — senão o primeiro visitante da bancada paga essa espera.

### Chave do LLM em nuvem

**Pela interface**, não editando arquivo: clique na **engrenagem** no canto do
cabeçalho, cole a chave e clique em *Testar e salvar*. O provedor é reconhecido
pelo prefixo (`sk-ant-`, `AQ.`/`AIza`, `gsk_`, `sk-or-`, `sk-proj-`/`sk-`), e
há uma lista manual para quando ele não for.

A chave é testada antes de gravar. Se não souber o nome do modelo, o botão
*Listar modelos* pergunta ao provedor quais a sua chave pode usar.

Fica em `.env`, que está no `.gitignore` — **a chave não vem no clone**. Cada
máquina configura a sua.

**A engrenagem só funciona em `localhost`.** Com `python api.py --rede`, a API
recusa a configuração vinda de fora: ler ou gravar a chave pela rede seria
entregá-la a quem estiver na sala.

Sem chave o sistema **continua funcionando**: a pergunta aberta cai no SLM
local, mais lenta e com qualidade menor, e a interface mostra a rota usada.

---

## 9. Subir a aplicação

```bash
uvicorn api:app --reload          # só nesta máquina
python api.py --rede              # aceita celulares da mesma rede
```

E abra `http://localhost:8000`. A API carrega modelo, detector e índice uma vez
e **aquece o SLM** — sem isso o primeiro visitante esperaria ~75 s pela primeira
frase. Sobe em cerca de 4 segundos.

O que você vai encontrar na tela:

| Elemento | O que faz |
| --- | --- |
| **Escolher uma foto** | arquivo; no celular o próprio seletor oferece a câmera |
| **Ligar a câmera** | câmera ao vivo, quadro quadrado; só em `localhost` ou HTTPS |
| **Sortear um animal** | puxa uma das 7349 fotos do dataset e mostra o gabarito |
| **Ver o perfil** | texto da raça, gerado localmente e guardado em cache |
| **Pergunte sobre este animal** | chat; vai para a nuvem, com botão para comparar com o local |
| **engrenagem** | chave de API |
| rodapé | métricas do evento: fotos, % resolvido localmente, latências |

As pastilhas do cabeçalho mostram o estado: modelo e dispositivo, SLM, nuvem e
RAG. Verde é saudável; amarelo é degradado, não quebrado.

---

## Teste de aceitação

O teste que vale mais que todos os outros: **desligue a rede e use o sistema
inteiro.** Classificação, mapa de atenção, decisão e respostas curtas continuam
funcionando. Só o perfil longo da raça mostra um aviso. Se qualquer outra coisa
quebrar, o roteamento está errado.

---

## Problemas comuns

| Sintoma | Causa | Solução |
| --- | --- | --- |
| "Ligar a câmera" não aparece | acessou pelo IP da rede | o navegador só libera câmera em `localhost` ou HTTPS; use "Escolher uma foto" |
| a engrenagem dá 403 | acessou pelo IP da rede | a chave só se configura na própria máquina |
| classifica o fundo em vez do pet | pesos do detector não baixaram | rode a API uma vez com internet |
| "prefixo não reconhecido" na chave | formato novo do provedor | escolha o provedor na lista do diálogo |
| nome de modelo inválido | catálogo mudou | use *Listar modelos* e clique num nome |
| `torch.cuda.is_available()` dá `False` com NVIDIA presente | roda de CPU instalada | `pip uninstall -y torch torchvision` e repita o passo 4 com `cu128` |
| `CUDA out of memory` no treino | lote grande demais | `--batch-size 16` |
| `data.py` começa a baixar e fica lento | `baixar_dataset.py` não rodou | interrompa, rode o passo 6 na ordem |
| `checkpoints/` vazio após o clone | pesos não foram versionados | rode o passo 7 |
| Primeira resposta do SLM demora ~75 s | carregamento do modelo | normal; aqueça o modelo na inicialização |
| `ollama: command not found` | serviço não instalado ou fora do PATH | reabra o terminal após instalar |
| Configurações de depuração não aparecem no VS Code | `launch.json` fora de `.vscode/` | o arquivo tem que estar em `.vscode/launch.json` |

---

## Onde está o contexto do projeto

O `CLAUDE.md` na raiz descreve o que o projeto é, as cinco restrições que não
podem ser quebradas, o estado atual e o que falta construir. **Leia antes de
alterar qualquer coisa.** O Claude Code carrega esse arquivo automaticamente ao
abrir a pasta.
