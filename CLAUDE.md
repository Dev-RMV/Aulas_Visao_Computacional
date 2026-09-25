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
requirements_ultralytics.txt só ultralytics (aula 13), instalar depois com --no-deps
preparar_ambiente.py         detecta Windows ou Linux (WSL), acha ou baixa Python 3.13, cria o venv, instala os três requirements e verifica
tutoriais.md                 índice dos 16 tutoriais, conceitos por aula e defeitos achados na leitura
tutorial_html/index.html     as 16 aulas em uma página só, com laboratórios interativos em canvas
nome_pastas.txt              lista original das aulas, base para os nomes das pastas
README.md                    apresentação do repositório no GitHub, com créditos ao professor
CLAUDE.md                    este arquivo
```

As 16 aulas: 01 OpenCV, 02 Camera, 03 ROI_SURF_etc, 04 Tensor, 05 HSV, 06 Reconhecimento_Facial, 07 Homografia, 08 TensorFlow_Keras, 09 HOG_SVM, 10 Rastreamento, 11 MNIST, 12 DNN_Caffe, 13 YOLO_SSD, 14 OpenCV_DNN, 15 HSV_ORB_DeepLabV3_ResNet50, 16 Calibracao.

Os dois README.txt são iguais em todas as aulas. O de `referencia_de_aula` diz que a pasta contém os códigos baixados do Drive do PVA. O de `exemplos_em_python` diz que contém os arquivos extraídos do zip e transformados em `.py`. Ao criar uma pasta nova, copie os READMEs da aula 01 byte a byte.

## Convenções que o usuário aprovou

- **Fidelidade ao professor.** Os scripts ficam como ele escreveu, inclusive imports duplicados e argumentos não usados. Só se corrige defeito de transcrição (tab perdido antes de um `"""`, CRLF), e toda correção é relatada.
- **Nomes dos .py convertidos de docx.** Se o docstring cita o nome do arquivo, usar esse nome, porque os scripts se referenciam entre si. Senão, prefixo com o número do título "Exemplo NN" e o resto derivado do título. A aula 02 ficou numerada de 09 a 19 por isso.
- **Encoding.** UTF-8 sem BOM, quebra de linha LF. Conferir CR e BOM com Python: o `grep -c $'\r'` do Git Bash desta máquina dá falso positivo em todo arquivo.
- **Zip com .py prontos.** Achatar a pasta de topo do zip para dentro de `exemplos_em_python/`, preservar subpastas, ignorar `__pycache__`, copiar arquivos de apoio junto. Se vier um `README.txt` na raiz do zip, salvar como `README_do_professor.txt`. Exceção: quando uma aula vem em vários zips que são subpastas de uma raiz comum e os scripts referenciam essas subpastas pelo nome (caso da aula 09), manter as subpastas dentro de `exemplos_em_python/`.
- **Pastas de saída.** Pasta em que os scripts gravam sem `mkdir` recebe um `.gitkeep` (caso de `saidas/` nas aulas 13 a 16, `relatorios/` na 13, `relatorio/` na 15). O que os scripts geram ou baixam (`dados/`, `modelos/`, `data/`, `imagens/`, pesos `.pt`, `.onnx`, `.weights`, `.tflite`) fica fora do git pelo `.gitignore`. Arquivos que vieram no zip do professor ficam no git, mesmo que os scripts os regravem (caso de `dados_sinteticos/` da aula 16).
- **Pacotes que declaram `opencv-python`.** Nunca instalar por cima do `opencv-contrib-python`. O pacote vai para um `requirements_*.txt` próprio instalado com `--no-deps`, e as dependências dele entram no `requirements.txt`. Feito para `face-recognition` e `ultralytics`.
- **Verificação obrigatória.** `py_compile` em todos os `.py` com o Python do venv e, na conversão de docx, diff linha a linha contra o texto do docx; na cópia de `.py` prontos, comparação byte a byte com o zip.
- **Pastas sem acento.** Espaços viram `_`. Comas e a conjunção "e" caem; a grafia dos nomes de modelo segue a usual (`ResNet50`, embora a lista escreva "ResNET50").
- **Tutoriais.** Um por aula, 8 seções fixas, cada conceito com trecho literal do professor e "Onde aparece". Para aula nova, delegar a um fork com a lista de fatos da leitura (material faltante, pastas, dependências, defeitos), e depois conferir: 8 seções, sem travessão fora do H1, sem CR, sem BOM, todo arquivo da pasta citado.

## Como converter docx em .py

Não há `python-docx` instalado. A conversão usa só a biblioteca padrão: abrir o docx com `zipfile`, ler `word/document.xml` com `xml.etree`, e para cada parágrafo `w:p` concatenar os `w:t` (texto), `w:tab` (tab) e `w:br` (quebra). Cada parágrafo vira uma linha. Os docx do professor vêm um parágrafo por linha, com indentação em espaços, então o resultado é o script inteiro sem retoque.

## Estado atual (22/09/2026)

Concluído:

- 16 pastas de aula criadas com a estrutura acima.
- Zips de todas as aulas processados. Aulas 01 e 02 vieram como docx e foram convertidas (8 + 11 scripts). Aulas 03 a 16 vieram com `.py` prontos e foram copiadas. Total de 353 scripts (219 das aulas 01 a 12, 134 das aulas 13 a 16), todos compilando no Python 3.13 do venv.
- Aula 09 reconstruída em 10/09 com os três zips do professor: `exemplos_em_python/01_fundamentos_convolucao/`, `02_cnn_basica/`, `03_hog_svm_atividade/` (com `modelo_hog_svm.joblib`), mais `dados/positivas/`, `dados/negativas/` e `saidas/` vazias. Rodar cada parte de dentro da subpasta.
- Aulas 13 a 16 processadas em 22/09/2026 a partir de quatro zips que o usuário colocou em `novos_zips/` (pasta removida depois, vazia). Nenhuma correção foi necessária: os arquivos já vinham em LF e UTF-8 sem BOM. A aula 13 traz `GUIA_PROFESSOR.md` e um notebook; a 14, o `requirements.txt` do professor; a 16, `dados_sinteticos/` com 20 PNG e uma saída do professor em `saidas/`. Nas quatro, a raiz dos caminhos é a própria pasta dos scripts.
- `achados_dos_zips.md` escrito, com tabela resumo, método, detalhes por aula (01 a 16), checklist de pendências, bibliotecas por aula e a seção 6 do ambiente, ampliada em 22/09 com torch, torchvision, tabulate, psutil e ultralytics.
- 16 tutoriais escritos (`aula_NN_Tema/tutorial_da_aula.md`), um por aula, todos com as mesmas 8 seções: visão geral, ambiente, conceitos-chave, roteiro dos scripts, passo a passo, exercícios e itens, armadilhas, glossário. O da aula 12 tem uma seção 7b com as funções esperadas do pacote `utils/` ausente. Os das aulas 13 a 16 têm 25 a 28 conceitos e 51 a 66 KB. Índice em `tutoriais.md`, com conceitos por aula e defeitos achados na leitura. Verificação feita: UTF-8 sem BOM, LF, 8 seções presentes, todo script citado ao menos uma vez.
- `tutorial_html/index.html`: página única com as 16 aulas, cada uma com visão geral, conceitos-chave, laboratório interativo, armadilhas e links para os scripts e o `tutorial_da_aula.md`. São 21 laboratórios, 50 canvas e 50 sliders, tudo em JavaScript no próprio arquivo, sem biblioteca externa e sem internet. As 12 primeiras aulas (14/09/2026) compartilham uma cena sintética; as quatro novas (22/09/2026) desenham a rua sintética do `vision_utils.py` (NMS, IoU arrastável e rastreador por IoU com linha virtual), o frame do `dnn_utils.py` (ROI por HSV, cantos por Harris no lugar do ORB e cronômetro por etapa), a cena do `seg_utils.py` com a máscara ideal (IoU da pista contra a máscara HSV) e a câmera virtual do `synthetic_utils.py` (pose, eixos, cubo e distorção radial com deslocamento dos cantos). Três demonstrações das aulas 06 e 09 são esquemáticas e estão rotuladas como tais. Verificado no Chrome por `file://` em 22/09: sem erro de console vindo da página, 21 de 21 laboratórios ativos, sem rolagem horizontal na menor largura que a janela do teste permitiu (485 px; as seções novas usam as mesmas classes das antigas, verificadas a 400 px em 14/09), UTF-8 sem BOM e LF.

Venv `.venv` (Windows) na raiz com Python 3.13.5, requirements instalados em 10/09/2026 e ampliados em 22/09/2026 (torch 2.14.0+cpu, torchvision 0.29.0, tabulate, psutil, dependências do ultralytics; ultralytics 8.4.160 com `--no-deps`). `py preparar_ambiente.py` rodado de novo em 22/09: venv reaproveitado, três requirements instalados, verificação sem falhas. Nenhum script do professor foi executado ainda. Repositório git em `main`, remoto público `https://github.com/Dev-RMV/Aulas_Visao_Computacional` (criado em 10/09/2026). Fim de linha LF fixado por `.gitattributes` e `core.autocrlf=false` local. Venvs, `__pycache__`, `saidas/`, `resultados/`, modelos `.keras`/`.h5`/`.tflite`/`.pt`/`.onnx`/`.weights`, `aula_12_DNN_Caffe/models/` e as pastas geradas das aulas 13 a 16 ficam fora do git.

**Commitado e publicado:** as 16 aulas, `tutorial_html/` e toda a documentação estão em `main` e no GitHub desde o commit `e935ac1` do usuário (23/09/2026, "Update com os 4 últimos ZIPs, sem conferência humana."). **Sem commit:** só as correções de animação de 24/09 em `tutorial_html/index.html` e este `CLAUDE.md`. As pastas `.idea/` citadas antes não existem mais.

## Pendências (resumo, detalhes em achados_dos_zips.md seção 4)

Material do professor que não veio nos zips:

- Aula 04: `exemplos_em_python/assets/carro_real.png`.
- Aula 05: pasta `recursos/` (par estéreo, `alvo_hsv.png`, `video_alvo.mp4`) ou o script `gerar_recursos.py`.
- Aula 06: pasta `recursos/` (`foto_grupo.jpg`, `identidades/`, `video_teste.mp4`).
- Aula 11: fotos `0.jpg` a `9.jpg` em `exemplos_em_python/data/real_digits/`. Podem ser feitas por nós.
- Aula 12: pacote `utils/` (`face_utils.py`, `tf_utils.py`), 4 arquivos dos modelos Caffe, `data/faces_teste/`, `data/utkface_sample/`. Sem o pacote `utils/` quase nada da aula 12 roda.
- Aula 13: mídia real (`dados/imagem_teste.jpg`, `dados/video_real.mp4`) e modelos em `modelos/` (`yolov8n.onnx` exportado pelo ultralytics; `yolov4-tiny.cfg`, `.weights` e `coco.names`; `frozen_inference_graph.pb` e `ssd_mobilenet_v2_coco.pbtxt`). Só os scripts 11 a 15 e 29 dependem disso.
- Aula 14: o notebook `notebooks/pipeline_integrado_dnn.ipynb` citado pelo script 28. Para a atividade, 10 fotos reais em `data/classificacao/` e o `data/labels_top1.csv` preenchido.
- Aula 15: `relatorio/RELATORIO_INTEGRATIVO_MODELO.md` citado pelo script 34.
- Aula 16: nada, veio completa.

Material para a atividade da aula 09: vídeo com pedestres (item A) e dataset real com 100 positivas e 100 negativas (item B).

Dúvidas para o professor:

- Aula 03: o zip é uma pasta agregada com 28 exemplos de vários temas.

Ambiente:

- Usar **Python 3.13** (`py -3.13`), porque o TensorFlow 2.21 não tem wheel para o 3.14 da máquina. `requirements.txt` fixa OpenCV Contrib 4.14 (não o 5), TF 2.21, Keras 3.15, pandas 2.3, dlib-bin, torch 2.14 (CPU) e torchvision 0.29. Também `setuptools<82`, porque o 82 removeu `pkg_resources` e quebra o `face_recognition`. Rodar scripts com `.\.venv\Scripts\python.exe` ou ativando o venv. `pip check` reclama de `dlib` e de `opencv-python`; os dois são esperados.
- Nas aulas 05, 06, 07, 08 e 12, os scripts consideram a raiz do projeto como a pasta acima de `exemplos_em_python/`, ou seja, `aula_NN_Tema/`. Pastas `recursos/`, `saidas/`, `data/`, `models/` ficam lá.
- Nas aulas 04, 10 e 11 os caminhos são relativos ao diretório de trabalho. Rodar de dentro de `exemplos_em_python/`. Na aula 09, rodar de dentro de cada subpasta. Nas aulas 13 a 16 a raiz é a pasta dos scripts (`Path(__file__).parent`), mas rodar de dentro de `exemplos_em_python/` mesmo assim, porque os módulos de apoio são importados como locais e o `yolov8n.pt` cai no diretório de trabalho.
- Aulas 14 e 15 precisam de internet na primeira execução (pesos da MobileNetV2 e rótulos do ImageNet; pesos do FCN-ResNet50 e do DeepLabV3 do torchvision, 135 e 160 MB).
- Aula 16: os scripts 25, 27 e 28 sobrescrevem `calib_01.png` a `calib_18.png` com versões ruidosas; rodar o 03 depois deles para restaurar (as imagens são deterministas e iguais às do zip) e o `git status` volta a ficar limpo.

## Próximos passos prováveis

1. Commitar as correções de animação de 24/09 (`tutorial_html/index.html` e `CLAUDE.md`), quando o usuário pedir. Antes de afirmar o que está ou não commitado, conferir com `git status` e `git log`, não com este arquivo.
2. Usuário consegue o material faltante e pede para colocar nas pastas certas.
3. Venv pronto. Para recriar ou preparar outra máquina: `py preparar_ambiente.py` (Windows) ou `python3 preparar_ambiente.py` (Linux). Flags: `--recriar`, `--sem-verificacao`, `--venv-dir`, `--python`.
4. Rodar os exemplos aula a aula, começando pelas autossuficientes: 01, 02, 03, 07, 08, 09, 10, 16, e as que só precisam de internet na primeira vez: 14, 15. A aula 13 roda sem material até o 10 e de 16 a 35.
5. Se chegar um zip novo, repetir o processo: pasta com a estrutura acima, zip em `referencia_de_aula/`, cópia verificada, `.gitkeep` onde os scripts gravam sem `mkdir`, `.gitignore` para o que é gerado, seção no `achados`, tutorial em 8 seções, linha em `tutoriais.md` e no `README.md`, seção e laboratório na página HTML. Se vier docx, converter pelo método acima.
6. **Tutorial online.** O `tutorial_html/` pode virar o tutorial público do repositório, servido por GitHub Pages a partir da branch `main`. A página já é um arquivo único, autossuficiente e responsivo, então a publicação é só ligar o Pages. Antes disso vale decidir o que fazer com os links relativos para os `.md` e as pastas de scripts, que hoje funcionam no sistema de arquivos e, publicados, apontariam para arquivos não renderizados; o caminho simples é trocá-los pelas URLs do GitHub. Se o formato agradar, os tutoriais em markdown e a página passam a ser duas saídas do mesmo conteúdo, e é preciso decidir qual das duas é a fonte.

## Histórico

- 06/09/2026: estrutura criada, 12 zips processados, `achados_dos_zips.md` e este arquivo escritos.
- 10/09/2026: 12 tutoriais escritos a partir da leitura integral dos scripts, mais o índice `tutoriais.md`. Nada executado.
- 10/09/2026: chegaram os dois zips que faltavam da aula 09. Aula reestruturada em três subpastas, 11 scripts novos, tutorial reescrito.
- 10/09/2026: `requirements.txt` único escrito, `.venv` criado com Python 3.13, instalação e testes de runtime feitos (Keras 3.15 com backend TF 2.21, OpenCV 4.14 contrib, face_recognition). Pino `setuptools<82` adicionado.
- 10/09/2026: `preparar_ambiente.py` escrito e testado no Windows (`.venv`) e no WSL Ubuntu 22.04 (`.venv_linux`, Python 3.13 obtido via uv).
- 10/09/2026: git iniciado, `README.md`, `.gitignore` e `.gitattributes` criados, repositório público `Dev-RMV/Aulas_Visao_Computacional` publicado com 289 arquivos.
- 14/09/2026: projeto restaurado ao estado do GitHub a pedido do usuário (`reset --hard origin/main` e `clean -fd`), o que descartou o notebook da aula 01 e alterações locais em `CLAUDE.md`, `README.md`, `requirements.txt` e `tutoriais.md`, nenhuma delas commitada. Em seguida, `tutorial_html/index.html` criado. Sem commit.
- 22/09/2026: aulas 13 a 16 processadas a partir de quatro zips novos (134 scripts, nenhuma correção), quatro tutoriais escritos por forks paralelos, `achados_dos_zips.md`, `tutoriais.md`, `README.md` e `.gitignore` atualizados, `requirements.txt` ampliado com torch, torchvision, tabulate, psutil e as dependências do ultralytics, `requirements_ultralytics.txt` criado (`--no-deps`), `preparar_ambiente.py` atualizado e rodado com sucesso, página HTML ampliada com quatro seções e quatro laboratórios e verificada no Chrome. Sem commit.
- 23/09/2026: o usuário commitou e publicou tudo de 14/09 e 22/09 no commit `e935ac1` (189 arquivos), sem atualizar este arquivo.
- 24/09/2026: laboratórios das aulas 01, 08 e 09 da página HTML não animavam com `prefers-reduced-motion: reduce` (Windows com "Efeitos de animação" desligado, caso da máquina do usuário). Agora os três têm botão de animar e pausar e começam parados só quando o movimento reduzido está ligado, como os das aulas 10 e 13 já faziam. Testado no Chrome com a preferência ativa. Para testar sem mexer no Windows: DevTools › Rendering › "Emulate CSS media feature prefers-reduced-motion". Sem commit.
