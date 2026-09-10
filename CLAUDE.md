# Contexto do projeto (lido automaticamente pelo Claude Code)

Este arquivo existe para retomar o trabalho de onde parou. Leia ele primeiro, depois `achados_dos_zips.md` se precisar de detalhes.

## O que é este projeto

Repositório de estudo da disciplina de Visão Computacional do Professor Vitor Amadeu (PVA). O usuário baixa do Google Drive do professor um zip por aula e organiza aqui os códigos dados em aula, em Python, prontos para rodar.

Idioma de trabalho: português do Brasil, em tudo (mensagens, documentos, nomes de arquivos novos).

## Estrutura

```
aula_NN_Tema/
├── referencia_de_aula/      README.txt + zip original do professor (não mexer)
├── exemplos_em_python/      README.txt + scripts .py prontos para uso
└── tutorial_da_aula.md      tutorial da aula em 8 seções, gerado a partir dos scripts
achados_dos_zips.md          relatório completo do que veio em cada zip e o que falta
requirements.txt             ambiente único para todas as aulas (Python 3.13); justificativas na seção 6 do achados
requirements_face_recognition.txt  só face-recognition, instalar depois com --no-deps
preparar_ambiente.py         detecta Windows ou Linux (WSL), acha ou baixa Python 3.13, cria o venv, instala e verifica
tutoriais.md                 índice dos 12 tutoriais, conceitos por aula e defeitos achados na leitura
nome_pastas.txt              lista original das 12 aulas, base para os nomes das pastas
CLAUDE.md                    este arquivo
```

As 12 aulas: 01 OpenCV, 02 Camera, 03 ROI_SURF_etc, 04 Tensor, 05 HSV, 06 Reconhecimento_Facial, 07 Homografia, 08 TensorFlow_Keras, 09 HOG_SVM, 10 Rastreamento, 11 MNIST, 12 DNN_Caffe.

Os dois README.txt são iguais em todas as aulas. O de `referencia_de_aula` diz que a pasta contém os códigos baixados do Drive do PVA. O de `exemplos_em_python` diz que contém os arquivos extraídos do zip e transformados em `.py`. Ao criar uma pasta nova, copie os READMEs da aula 01 byte a byte.

## Convenções que o usuário aprovou

- **Fidelidade ao professor.** Os scripts ficam como ele escreveu, inclusive imports duplicados e argumentos não usados. Só se corrige defeito de transcrição (tab perdido antes de um `"""`, CRLF), e toda correção é relatada.
- **Nomes dos .py convertidos de docx.** Se o docstring cita o nome do arquivo, usar esse nome, porque os scripts se referenciam entre si. Senão, prefixo com o número do título "Exemplo NN" e o resto derivado do título. A aula 02 ficou numerada de 09 a 19 por isso.
- **Encoding.** UTF-8 sem BOM, quebra de linha LF.
- **Zip com .py prontos.** Achatar a pasta de topo do zip para dentro de `exemplos_em_python/`, preservar subpastas, ignorar `__pycache__`, copiar arquivos de apoio junto. Se vier um `README.txt` na raiz do zip, salvar como `README_do_professor.txt`. Exceção: quando uma aula vem em vários zips que são subpastas de uma raiz comum e os scripts referenciam essas subpastas pelo nome (caso da aula 09), manter as subpastas dentro de `exemplos_em_python/`.
- **Verificação obrigatória.** `py_compile` em todos os `.py` e, na conversão de docx, diff linha a linha contra o texto do docx.
- **Pastas sem acento.** Espaços viram `_`.

## Como converter docx em .py

Não há `python-docx` instalado. A conversão usa só a biblioteca padrão: abrir o docx com `zipfile`, ler `word/document.xml` com `xml.etree`, e para cada parágrafo `w:p` concatenar os `w:t` (texto), `w:tab` (tab) e `w:br` (quebra). Cada parágrafo vira uma linha. Os docx do professor vêm um parágrafo por linha, com indentação em espaços, então o resultado é o script inteiro sem retoque.

## Estado atual (10/09/2026)

Concluído:

- 12 pastas de aula criadas com a estrutura acima.
- Zips de todas as aulas processados. Aulas 01 e 02 vieram como docx e foram convertidas (8 + 11 scripts). Aulas 03 a 12 vieram com `.py` prontos e foram copiadas. Total de 219 scripts, todos compilando.
- Aula 09 reconstruída em 10/09 com os três zips do professor: `exemplos_em_python/01_fundamentos_convolucao/`, `02_cnn_basica/`, `03_hog_svm_atividade/` (com `modelo_hog_svm.joblib`), mais `dados/positivas/`, `dados/negativas/` e `saidas/` vazias. Rodar cada parte de dentro da subpasta.
- `achados_dos_zips.md` escrito, com tabela resumo, método, detalhes por aula, checklist de pendências e bibliotecas por aula.
- 12 tutoriais escritos (`aula_NN_Tema/tutorial_da_aula.md`), um por aula, todos com as mesmas 8 seções: visão geral, ambiente, conceitos-chave, roteiro dos scripts, passo a passo, exercícios e itens, armadilhas, glossário. O da aula 12 tem uma seção 7b com as funções esperadas do pacote `utils/` ausente. Índice em `tutoriais.md`. Verificação feita: UTF-8 sem BOM, LF, 8 seções presentes, todo script citado ao menos uma vez.

Venv `.venv` (Windows) criado na raiz com Python 3.13.5 e requirements instalados em 10/09/2026, com testes de import e de backend do Keras passando (seção 6.1 do achados). `preparar_ambiente.py` automatiza isso nos dois sistemas: no Windows usa `.venv`, no Linux ou WSL usa `.venv_linux`, e se não houver Python 3.13 instala o `uv` na pasta do usuário e baixa um 3.13 isolado. Nenhum script do professor foi executado ainda. Não é repositório git.

## Pendências (resumo, detalhes em achados_dos_zips.md seção 4)

Material do professor que não veio nos zips:

- Aula 04: `exemplos_em_python/assets/carro_real.png`.
- Aula 05: pasta `recursos/` (par estéreo, `alvo_hsv.png`, `video_alvo.mp4`) ou o script `gerar_recursos.py`.
- Aula 06: pasta `recursos/` (`foto_grupo.jpg`, `identidades/`, `video_teste.mp4`).
- Aula 11: fotos `0.jpg` a `9.jpg` em `exemplos_em_python/data/real_digits/`. Podem ser feitas por nós.
- Aula 12: pacote `utils/` (`face_utils.py`, `tf_utils.py`), 4 arquivos dos modelos Caffe, `data/faces_teste/`, `data/utkface_sample/`. Sem o pacote `utils/` quase nada da aula 12 roda.

Material para a atividade da aula 09: vídeo com pedestres (item A) e dataset real com 100 positivas e 100 negativas (item B).

Dúvidas para o professor:

- Aula 03: o zip é uma pasta agregada com 28 exemplos de vários temas.

Ambiente:

- Usar **Python 3.13** (`py -3.13`), porque o TensorFlow 2.21 não tem wheel para o 3.14 da máquina. `requirements.txt` fixa OpenCV Contrib 4.14 (não o 5), TF 2.21, Keras 3.15, pandas 2.3, dlib-bin. Também `setuptools<82`, porque o 82 removeu `pkg_resources` e quebra o `face_recognition`. Instalado e testado no `.venv` em 10/09/2026. Rodar scripts com `.\.venv\Scripts\python.exe` ou ativando o venv.
- Nas aulas 05, 06, 07, 08 e 12, os scripts consideram a raiz do projeto como a pasta acima de `exemplos_em_python/`, ou seja, `aula_NN_Tema/`. Pastas `recursos/`, `saidas/`, `data/`, `models/` ficam lá.
- Nas aulas 04, 10 e 11 os caminhos são relativos ao diretório de trabalho. Rodar de dentro de `exemplos_em_python/`. Na aula 09, rodar de dentro de cada subpasta.

## Próximos passos prováveis

1. Usuário consegue o material faltante e pede para colocar nas pastas certas.
2. Venv pronto. Para recriar ou preparar outra máquina: `py preparar_ambiente.py` (Windows) ou `python3 preparar_ambiente.py` (Linux). Flags: `--recriar`, `--sem-verificacao`, `--venv-dir`, `--python`.
3. Rodar os exemplos aula a aula, começando pelas autossuficientes: 01, 02, 03, 07, 08, 09, 10.
4. Se chegar um zip novo com docx, repetir a conversão pelo método acima.

## Histórico

- 06/09/2026: estrutura criada, 12 zips processados, `achados_dos_zips.md` e este arquivo escritos.
- 10/09/2026: 12 tutoriais escritos a partir da leitura integral dos scripts, mais o índice `tutoriais.md`. Nada executado.
- 10/09/2026: chegaram os dois zips que faltavam da aula 09. Aula reestruturada em três subpastas, 11 scripts novos, tutorial reescrito.
- 10/09/2026: `requirements.txt` único escrito, `.venv` criado com Python 3.13, instalação e testes de runtime feitos (Keras 3.15 com backend TF 2.21, OpenCV 4.14 contrib, face_recognition). Pino `setuptools<82` adicionado.
- 10/09/2026: `preparar_ambiente.py` escrito e testado no Windows (`.venv`) e no WSL Ubuntu 22.04 (`.venv_linux`, Python 3.13 obtido via uv).
