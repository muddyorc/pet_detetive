# Galeria de exemplos

Fotos que aparecem na interface para visitantes que não têm um pet por perto.
A `GET /exemplos` lê esta pasta sozinha; a galeria **some da tela** se não
houver nenhuma imagem, em vez de mostrar quadros quebrados na bancada.

Formatos aceitos: `.jpg`, `.jpeg`, `.png`, `.webp`.

## Por que os arquivos não estão no repositório

As imagens aqui vêm do Oxford-IIIT Pet, que é distribuído para pesquisa e
ensino. Usá-las na sua máquina é tranquilo; **redistribuí-las** num repositório
público é outra conversa, e não vale o risco por uma galeria. Por isso
`web/exemplos/*.jpg` está no `.gitignore`.

## Como repovoar numa máquina nova

Depois de montar o dataset (`python baixar_dataset.py`):

```bash
python - <<'EOF'
import shutil, pathlib
src = pathlib.Path("data/oxford-iiit-pet/images")
dst = pathlib.Path("web/exemplos"); dst.mkdir(parents=True, exist_ok=True)
for nome in ["pug_52", "Maine_Coon_1", "beagle_30", "Sphynx_8",
             "samoyed_10", "boxer_15", "Persian_1"]:
    p = src / f"{nome}.jpg"
    if p.exists():
        shutil.copy(p, dst / p.name)
EOF
```

Melhor ainda: use **fotos suas e de colegas**. São mais parecidas com o que os
visitantes vão trazer — celular, luz ruim, enquadramento torto — e é
exatamente nesse material que o sistema precisa se sair bem.

## Sobre o `beagle_30.jpg`

Mantenha esse. Ele é um Beagle que a rede só reconhece com 14% de confiança, o
que faz o sistema cair no caminho **perguntar** em vez de afirmar. É a melhor
demonstração de que a camada de decisão existe e funciona — e é bem mais
interessante de mostrar numa bancada do que mais um acerto de 97%.
