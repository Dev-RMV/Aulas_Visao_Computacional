# Achados dos zips de referência

Registro do processamento dos zips baixados do Google Drive do Professor Vitor Amadeu (PVA), feito em 06/09/2026.

Cada `aula_NN_Tema/` tem duas subpastas:

- `referencia_de_aula/`: README e o zip original, intocado.
- `exemplos_em_python/`: README e os scripts prontos para uso.

Os nomes das pastas seguem `nome_pastas.txt`, sem acentos e com `_` no lugar de espaços. A aula 3 ficou `aula_03_ROI_SURF_etc`.

## 1. Resumo

| Aula | Zip | Formato no zip | Scripts | Apoio incluído | Roda sem material extra? |
|---|---|---|---|---|---|
| 01 OpenCV | `exemplos-20260906T162749Z-1-001.zip` | 8 docx | 8 | – | Sim (webcam em 4 scripts) |
| 02 Câmera | `exemplos-20260906T162818Z-1-001.zip` | 11 docx | 11 | – | Sim (webcam gera a imagem-base) |
| 03 ROI SURF etc | `exemplos-20260906T165410Z-1-001.zip` | 29 .py | 29 | – | Sim (gera as próprias imagens) |
| 04 Tensor | `exemplos-20260906T165549Z-1-001.zip` | 12 .py | 12 | `requirements.txt`, README do professor | **Não**: falta `assets/carro_real.png` |
| 05 HSV | `exemplos-20260906T165635Z-1-001.zip` | 12 .py | 12 | – | **Não**: falta a pasta `recursos/` |
| 06 Reconhecimento facial | `exemplos-20260906T165725Z-1-001.zip` | 12 .py | 12 | – | **Não**: falta a pasta `recursos/` |
| 07 Homografia | `exemplos-20260906T165810Z-1-001.zip` | 14 .py | 14 | – | Sim (gera as próprias imagens) |
| 08 TensorFlow Keras | `exemplos-20260906T165922Z-1-001.zip` | 14 .py | 14 | – | Sim (gera o próprio dataset) |
| 09 HOG e SVM | 3 zips: `01_fundamentos_convolucao-…`, `02_cnn_basica-…`, `03_hog_svm_atividade-…` | 25 .py | 25 | `modelo_hog_svm.joblib` | Parcial: item A pede vídeo de pedestres, item B pede dataset real |
| 10 Rastreamento | `rastreamento_aula_exemplos-20260906T170117Z-1-001.zip` | 13 .py | 13 | `README.md`, `requirements.txt`, `data/synthetic_motion.mp4` | Sim |
| 11 MNIST | `exemplos-20260906T170224Z-1-001.zip` | 33 .py | 33 | – | Parcial: falta `data/real_digits/` |
| 12 DNN e Caffe | `exemplos-20260906T170305Z-1-001.zip` | 36 .py | 36 | – | **Não**: falta o pacote `utils/`, os modelos Caffe e os dados |

Total: 219 scripts (208 em 06/09/2026, mais 11 da aula 09 em 10/09/2026). Todos passam em `py_compile` sem erro de sintaxe.

Só as aulas 01 e 02 vieram como documentos Word. As aulas 03 a 12 já vieram com `.py` prontos e foram apenas copiadas.

## 2. Método

### Conversão docx para .py (aulas 01 e 02)

- Cada docx contém um script inteiro, um parágrafo por linha, indentação em espaços.
- Não há `python-docx` instalado. A leitura foi feita com `zipfile` + `xml.etree` direto em `word/document.xml`, tratando `w:t`, `w:tab` e `w:br`.
- Gravação em UTF-8 sem BOM, quebra de linha LF, uma quebra final.
- Verificação: `ast.parse` do texto e diff linha a linha entre o docx e o `.py` gerado. Todos ficaram idênticos.
- Nomes: quando o docstring cita o nome do arquivo, esse nome é usado, porque os scripts se referenciam entre si. Sem citação, o prefixo é o número do título "Exemplo NN" e o resto vem do título.
- Só defeitos de transcrição foram corrigidos e estão listados abaixo. O resto ficou fiel ao professor, inclusive imports duplicados e argumentos não usados.

### Cópia dos .py prontos (aulas 03 a 12)

- A pasta de topo do zip (`exemplos/`, `rastreamento_aula_exemplos/`, `01_fundamentos_convolucao/`) foi achatada para dentro de `exemplos_em_python/`. Subpastas abaixo dela foram preservadas.
- Pastas `__pycache__` foram ignoradas.
- Arquivos de apoio (`utils.py`, `requirements.txt`, README, dados) foram copiados junto.
- Cada arquivo foi conferido por checksum contra o original. Única alteração: um arquivo com CRLF foi normalizado para LF.
- Se o zip traz um `README.txt` na raiz, ele é salvo como `README_do_professor.txt` para não sobrescrever o nosso.

## 3. Detalhes por aula

### Aula 01 – OpenCV

| docx | Arquivo criado | Origem do nome |
|---|---|---|
| ex0 | `00_verificar_ambiente.py` | citado no docstring do ex0 e do ex1 |
| ex1 | `01_imagem_e_array_numpy.py` | citado no docstring |
| ex2 | `02_gerar_video_teste.py` | citado no docstring |
| ex3 | `03_camera_metadados_fps.py` | derivado do título |
| ex4 | `04_capturar_e_salvar_frame.py` | citado na mensagem de erro do ex5 |
| ex5 | `05_recarregar_cinza_hconcat.py` | citado no docstring |
| ex6 | `06_item_b_completo_camera.py` | derivado do título |
| ex7 | `07_pipeline_integrado_a_b.py` | derivado do título |

- **Correção feita:** o ex4 tinha um tab antes do `"""` que abre o docstring. Daria erro de indentação na primeira linha. Removido.
- **Mantido como está:** o ex5 importa `argparse` sem usar, e o docstring cita um argumento `--imagem` que o código não implementa. O caminho fica fixo em `frame_capturado.png`.
- Ordem de execução: 00 e 02 rodam sem câmera e geram `teste_ambiente.png` e `video_teste.mp4`. 01 lê a imagem do 00. 04 gera `frame_capturado.png`, que o 05 lê. 03, 04, 06 e 07 usam a webcam.

### Aula 02 – Câmera

Os docstrings dizem "Exemplo 09" a "Exemplo 19", continuando a numeração da aula 01. Não existe "Exemplo 08". Nenhum doc cita nome de arquivo, então o prefixo é o número do professor.

| docx | Arquivo criado | Título no docstring |
|---|---|---|
| ex01 | `09_capturar_imagem_base.py` | Captura de uma imagem-base com a câmera |
| ex02 | `10_canais_hsv_lab.py` | Canais dos espaços de cor HSV e LAB |
| ex03 | `11_alterar_saturacao_hsv.py` | Alteração programática da saturação em HSV |
| ex04 | `12_exercicio2_item_a_completo.py` | Solução completa do Exercício 2, Item A |
| ex05 | `13_sharpening_filter2d.py` | Sharpening manual com cv2.filter2D |
| ex06 | `14_unsharp_masking_laplaciano.py` | Unsharp masking e variância do Laplaciano |
| ex07 | `15_comparar_sharpening_unsharp.py` | Exemplo 15 (compara original, sharpening e unsharp) |
| ex08 | `16_limiarizacao_global_adaptativa_otsu.py` | Limiarização global, adaptativa e Otsu |
| ex09 | `17_canny_dois_thresholds.py` | Comparação do detector Canny com dois pares de thresholds |
| ex10 | `18_contornos_por_area.py` | Exemplo 18 corrigido: detecção e classificação de contornos por área |
| ex11 | `19_exercicio3_solucao_integrada.py` | Solução integrada do Exercício 3 |

- **Mantido como está:** de 10 a 19, `import cv2` aparece duas vezes e um bloco `obter_imagem_base` está repetido em vários scripts. É o código do professor.
- Fluxo: o 09 gera `imagem_base.png` na pasta dos scripts. Os demais leem essa imagem, e se ela não existir abrem a câmera para capturar uma nova.

### Aula 03 – ROI SURF etc

- 29 scripts numerados de 00 a 27. Há dois com prefixo 09: `09_gerar_mapa_disparidade.py` e `09_disparidade_para_profundidade.py`.
- O conteúdo vai muito além de ROI e SURF. Parece uma pasta agregada do professor cobrindo várias aulas:
  - 01 a 06: cena sintética, ROI, HSV, Otsu, morfologia, GrabCut.
  - 07 a 09: par estéreo, disparidade SGBM, profundidade.
  - 10 a 15: faces com Haar, LBPH, embeddings.
  - 16 a 21: SIFT, ORB, matching, homografia RANSAC, SURF opcional, recuperação de imagens.
  - 22 a 27: MLP e CNN no Fashion-MNIST, callbacks, features, transfer learning MobileNetV2, pipeline OpenCV + CNN na webcam.
- Autossuficiente: o 01 gera `cena.png`, `objeto_consulta.png` e `objeto_transformado.png`. O 07 gera o par estéreo. Fashion-MNIST é baixado pelo Keras. O Haar cascade vem de `cv2.data`.
- **Correção feita:** `09_gerar_mapa_disparidade.py` estava com CRLF e foi normalizado para LF.
- O 15 precisa de `face_recognition`. O 20 (SURF) precisa de `opencv-contrib` com módulos nonfree, e o nome sugere que tem fallback.

### Aula 04 – Tensor

- 12 scripts numerados de 01 a 12. Importam só `tensorflow`, sem OpenCV.
- Vieram `requirements.txt` e o README do professor, salvo como `README_do_professor.txt`.
- O README diz: ambiente testado com Python 3.12.2 e TensorFlow 2.17 no Windows, com instruções de venv no PowerShell. No Windows nativo o TensorFlow usa só CPU.
- **Falta:** `assets/carro_real.png`, usado pelos exemplos 04 a 10 e 12. O caminho é relativo ao diretório de trabalho, então o arquivo deve ficar em `aula_04_Tensor/exemplos_em_python/assets/carro_real.png` e os scripts devem ser executados de dentro de `exemplos_em_python/`.

### Aula 05 – HSV

- 11 scripts numerados de 01 a 11 mais `utils.py`. Conteúdo: 01 a 04 par estéreo e disparidade (BM e SGBM), 05 a 08 HSV, segmentação e ROI, 09 a 11 vídeo e integração.
- `utils.py` define a raiz do projeto como a pasta acima da dos scripts. No nosso layout isso é `aula_05_HSV/`. Ele espera `aula_05_HSV/recursos/` e cria `aula_05_HSV/saidas/` sozinho.
- **Falta a pasta `recursos/`** com: `estereo_esquerda.png`, `estereo_direita.png`, `alvo_hsv.png`, `video_alvo.mp4`.
- `utils.py` manda executar `gerar_recursos.py` primeiro, mas esse script **não veio no zip**.
- Os scripts 09 e 10 aceitam `--fonte 0` para usar a webcam no lugar do vídeo.

### Aula 06 – Reconhecimento facial

- 11 scripts numerados de 00 a 10 mais `utils.py`. Mesmo layout da aula 05: `aula_06_Reconhecimento_Facial/recursos/` e `saidas/`.
- **Falta a pasta `recursos/`** com: `foto_grupo.jpg`, `identidades/Ana.jpg` (e provavelmente outras fotos de identidades para o script 06), `video_teste.mp4`. O `gabarito_video.csv` é opcional, o código devolve vazio se ele não existir.
- O Haar cascade usa cópia local em `recursos/` se houver, senão cai em `cv2.data`. Isso está resolvido.
- O script 04 aceita `--source 0` para usar a webcam.
- Precisa de `dlib` e `face_recognition`, que são difíceis de instalar no Windows (exigem CMake e compilador C++, ou wheel pré-compilado).
- O zip trazia `__pycache__` de Python 3.12, ou seja, o professor rodou com 3.12.

### Aula 07 – Homografia

- 13 scripts numerados de 00 a 12 mais `utils.py`. Layout `recursos/` e `saidas/` sob `aula_07_Homografia/`.
- Autossuficiente: `01_gerar_imagens_teste.py` gera `cena_referencia.png`, `cena_transformada.png`, `homografia_real.json` e `video_localizacao.mp4`. Rode ele primeiro.
- Só precisa de `cv2` e `numpy`.

### Aula 08 – TensorFlow Keras

- 13 scripts numerados de 00 a 12 mais `utils.py`. Layout `dataset/`, `modelos/`, `recursos/` e `saidas/` sob `aula_08_TensorFlow_Keras/`.
- Autossuficiente: `01_gerar_dataset.py` gera o dataset de formas geométricas. O 05 treina e salva `cnn_geometrica.keras`, que os seguintes usam. Rode em ordem.
- Precisa de `tensorflow`, `sklearn`, `matplotlib`, `cv2`.

### Aula 09 – HOG e SVM

- A aula vem em **três zips**, um por subpasta do Drive do professor. Em 06/09 só o primeiro tinha sido baixado, por isso parecia zip errado. Os outros dois chegaram em 10/09/2026.
- Os scripts do terceiro zip usam `Path(__file__).resolve().parents[1]` e referenciam `03_hog_svm_atividade/modelo_hog_svm.joblib`, `dados/` e `saidas/` a partir da pasta pai. Achatar quebraria esses caminhos. Por isso, exceção à regra geral: as três subpastas foram **preservadas** dentro de `exemplos_em_python/`, e os 14 scripts do primeiro zip, que estavam achatados, foram movidos para `01_fundamentos_convolucao/`.
- Layout final: `01_fundamentos_convolucao/` (01 a 14, convolução, Sobel, blur, stride, padding, ReLU, pooling), `02_cnn_basica/` (15 a 20, CNN no MNIST e CIFAR-10, kernels, feature maps, MobileNetV2, conceito de detecção), `03_hog_svm_atividade/` (item A com o detector de pedestres HOG do OpenCV, item B com dataset 64x128, SVM e janela deslizante, gerador de dataset sintético e o modelo `modelo_hog_svm.joblib` de 2,3 MB treinado pelo professor).
- Pastas `dados/positivas/`, `dados/negativas/` e `saidas/` foram criadas vazias em `exemplos_em_python/`, porque `gerar_dataset_sintetico_demo.py` e `item_b_janela_deslizante.py` gravam nelas sem `mkdir`.
- Rodar cada parte de dentro da sua subpasta. Os scripts 01 a 20 gravam em caminho relativo ao diretório de trabalho; o 15 salva `modelo_mnist.keras`, lido por 16 e 17.
- **Falta para a atividade:** um vídeo com pedestres para o item A (`--video`) e um dataset real com no mínimo 100 positivas e 100 negativas para o item B. O `gerar_dataset_sintetico_demo.py` cria um dataset de demonstração que, segundo o próprio docstring, não substitui o real.
- Checksum conferido contra os zips, todos iguais. Nenhuma correção necessária, os arquivos já vinham em LF.

### Aula 10 – Rastreamento

- 11 scripts numerados de 00 a 10 mais `generate_synthetic_video.py` e `utils.py`.
- Veio completo: `README.md` do professor, `requirements.txt`, `data/synthetic_motion.mp4` (6,3 MB) e `output/.gitkeep`. As pastas `data/` e `output/` foram mantidas em `exemplos_em_python/` porque os scripts usam caminhos relativos.
- Todos aceitam `--source` (0 para webcam ou caminho de vídeo). Autossuficiente.
- Precisa de `cv2`, `numpy`, `pandas`, `matplotlib`. O `__pycache__` do zip era de Python 3.13.

### Aula 11 – MNIST

- 32 scripts numerados de 00 a 31 mais `mnist_utils.py`.
- O MNIST é baixado pelo Keras na primeira execução.
- **Falta `data/real_digits/`** com fotos de dígitos manuscritos nomeadas `0.jpg` a `9.jpg`. Usado pelos scripts 22, 23, 24, 25, 28 e 30. O caminho é relativo, então a pasta deve ficar em `aula_11_MNIST/exemplos_em_python/data/real_digits/` e os scripts rodam de dentro de `exemplos_em_python/`. Essas fotos podem ser feitas por nós mesmos.
- O 31 usa a webcam.
- Precisa de `tensorflow`, `sklearn`, `pandas`, `matplotlib`, `cv2`.

### Aula 12 – DNN e Caffe

- 35 scripts numerados de 01 a 35 mais `_bootstrap.py`.
- `_bootstrap.py` define a raiz do projeto como a pasta acima da dos scripts (`aula_12_DNN_Caffe/`), coloca ela no `sys.path` e faz `chdir` para lá. Então `data/`, `models/` e `resultados/` ficam em `aula_12_DNN_Caffe/`, não em `exemplos_em_python/`.
- **Falta o pacote `utils/`** com `face_utils.py` e `tf_utils.py`. A maioria dos scripts importa funções dele (`detect_faces_haar`, `load_caffe_age_gender`, `predict_age_gender`, `parse_utkface_filename`, `build_mobilenetv2_gender`, entre outras). Sem esse pacote quase nada roda. Ele deve ficar em `aula_12_DNN_Caffe/utils/`.
- **Faltam os modelos Caffe** em `models/opencv_age_gender/`: `age_deploy.prototxt`, `age_net.caffemodel`, `gender_deploy.prototxt`, `gender_net.caffemodel`. O script 08 diz que o zip não os inclui por tamanho e licença, e lista as URLs: os `.caffemodel` vêm do repositório GilLevi/AgeGenderDeepLearning e os `.prototxt` de spmallick/learnopencv.
- **Faltam os dados:** `data/faces_teste/` com `pessoa.jpg` e `grupo.jpg`, e `data/utkface_sample/` com até 1000 imagens do dataset UTKFace.
- O script 02 cria as pastas vazias. O 03 localiza o Haar via `cv2.data`.
- Precisa de `tensorflow`, `sklearn`, `pandas`, `matplotlib`, `cv2`. O `__pycache__` do zip era de Python 3.12.

## 4. Pendências

Material a providenciar antes de rodar:

- [ ] **Aula 04:** `exemplos_em_python/assets/carro_real.png`.
- [ ] **Aula 05:** pasta `aula_05_HSV/recursos/` com o par estéreo, `alvo_hsv.png` e `video_alvo.mp4`, ou o script `gerar_recursos.py` que os cria.
- [ ] **Aula 06:** pasta `aula_06_Reconhecimento_Facial/recursos/` com `foto_grupo.jpg`, `identidades/`, `video_teste.mp4` e, se houver, `gabarito_video.csv`.
- [x] **Aula 09:** resolvido em 10/09/2026, faltavam dois zips. Ainda faltam para a atividade: vídeo de pedestres (item A) e dataset real de positivas e negativas (item B).
- [ ] **Aula 11:** fotos `0.jpg` a `9.jpg` em `exemplos_em_python/data/real_digits/`.
- [ ] **Aula 12:** pacote `aula_12_DNN_Caffe/utils/` (`face_utils.py`, `tf_utils.py`), os 4 arquivos Caffe, `data/faces_teste/` e `data/utkface_sample/`.
- [ ] **Aula 03:** confirmar se esse zip agregado é mesmo o da aula 3.

Ambiente:

- [ ] O professor usou Python 3.12 (aulas 04, 05, 06, 12) e 3.13 (aula 10). A máquina local tem Python 3.14.3. Confirmar se TensorFlow, dlib e face_recognition têm wheel para 3.14, senão criar um venv com 3.12.
- [ ] `dlib` e `face_recognition` (aulas 03 e 06) costumam exigir wheel pré-compilado no Windows.
- [ ] `opencv-contrib-python` para o SURF opcional da aula 03.

## 5. Bibliotecas de terceiros por aula

Levantado por análise dos imports. Módulos locais e da biblioteca padrão foram excluídos.

| Aula | Bibliotecas |
|---|---|
| 01 OpenCV | cv2, numpy |
| 02 Câmera | cv2, numpy |
| 03 ROI SURF etc | cv2, numpy, matplotlib, sklearn, tensorflow, face_recognition |
| 04 Tensor | tensorflow |
| 05 HSV | cv2, numpy |
| 06 Reconhecimento facial | cv2, numpy, dlib, face_recognition |
| 07 Homografia | cv2, numpy |
| 08 TensorFlow Keras | cv2, numpy, matplotlib, sklearn, tensorflow |
| 09 HOG e SVM | cv2, numpy, tensorflow, matplotlib, sklearn, joblib |
| 10 Rastreamento | cv2, numpy, pandas, matplotlib |
| 11 MNIST | cv2, numpy, pandas, matplotlib, sklearn, tensorflow |
| 12 DNN e Caffe | cv2, numpy, pandas, matplotlib, sklearn, tensorflow |

Os nomes de pacote e as versões fixadas estão em `requirements.txt` na raiz. A justificativa está na seção 6.

## 6. Ambiente unificado (`requirements.txt`, 10/09/2026)

Objetivo: um único venv que rode todos os 219 scripts sem conflito. Versões conferidas no PyPI em 10/09/2026. O venv `.venv` foi criado na raiz com o Python 3.13.5 da máquina, os dois requirements foram instalados e os testes de runtime abaixo passaram.

| Decisão | Motivo |
|---|---|
| **Python 3.13** | O TensorFlow 2.21.0 publica wheel para Windows só de 3.10 a 3.13. O 3.14.3 da máquina não serve. O professor usou 3.12 e 3.13. |
| **opencv-contrib-python==4.14.0.94** | Último 4.x (28/07/2026). O 5.0.0.93 (02/07/2026) mudou a API e o código do professor é 4.x. O contrib é um superconjunto do `opencv-python` e traz `cv2.face` (LBPH, aula 03). Nunca instalar os dois juntos. O SURF (aula 03, script 20): a função `cv2.xfeatures2d.SURF_create` existe, mas ao chamar dá erro "patented and excluded", porque a wheel é compilada sem nonfree. O script tem fallback. |
| **tensorflow==2.21.0** | Mais recente. Exige `keras>=3.12`, `numpy>=1.26` sem teto, `h5py<3.15`, `protobuf<8`. CPU no Windows nativo. |
| **keras==3.15.1** | Mais recente, compatível com o TF 2.21 (o próprio TF instala o Keras 3). Os scripts usam só `tensorflow.keras`. `ImageDataGenerator` (aula 11) ainda existe, com aviso de depreciação. |
| **numpy>=2.3,<3** | Todas as wheels binárias da lista são compiladas contra a ABI do numpy 2. O pip escolheu 2.5.3. |
| **pandas==2.3.3** | O 3.0 (2026) mudou padrões (strings, copy-on-write). O código do professor é da geração 2.x. |
| **matplotlib==3.11.1, scikit-learn==1.9.0, joblib** | Mais recentes, sem restrição especial. O `modelo_hog_svm.joblib` da aula 09 pode não carregar em versão de sklearn diferente da do professor; nesse caso retreinar com `item_b_treinar_svm.py`. |
| **dlib-bin==20.0.1** | O `dlib` do PyPI é só código-fonte e exige CMake e Visual Studio. O `dlib-bin` traz o mesmo módulo compilado, com wheel para 3.10 a 3.14. |
| **face-recognition==1.3.0 com `--no-deps`** | Declara dependência de `dlib`, que o pip tentaria compilar. Por isso fica em `requirements_face_recognition.txt`, instalado depois com `--no-deps`. As dependências dele (`face-recognition-models`, `Click`, `Pillow`, `numpy`) já estão no principal. Efeito colateral esperado: `pip check` reclama que falta `dlib`. Ignorar. |
| **setuptools<82** | O setuptools 82 (fevereiro de 2026) removeu `pkg_resources`, que o `face-recognition-models` 0.3.0 usa para achar os `.dat`. Sem esse pino o `import face_recognition` falha. O TF exige setuptools sem teto, então não há conflito. |

Instalação, no PowerShell, a partir da raiz do projeto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -r requirements_face_recognition.txt
```

### 6.1 Verificação feita no venv (10/09/2026)

- `import keras` com `KERAS_BACKEND=tensorflow`: Keras 3.15.1, backend `tensorflow`, tensores do TF, modelo `Sequential` compila e treina uma época.
- `import keras` sem variável: mesmo resultado, porque o `~/.keras/keras.json` padrão já traz `"backend": "tensorflow"`.
- `tf.keras.__version__` é 3.15.1 e `tf.keras.Sequential` é o mesmo `keras.Sequential`: a partir do TF 2.16 `tensorflow.keras` é o Keras 3. `tf.keras.preprocessing.image.ImageDataGenerator` (aula 11) ainda existe.
- OpenCV 4.14.0 com `cv2.face` (LBPH cria), `cv2.dnn`, `SIFT_create`, `HOGDescriptor` com o detector de pedestres. SURF presente mas bloqueado (nonfree).
- `dlib` 20.0.1 e `face_recognition` importam e `face_locations` roda.

### 6.2 Script `preparar_ambiente.py` (Windows e Linux)

Automatiza a seção 6 inteira: detecta o sistema (Windows, Linux ou WSL), procura um Python 3.13 (`py -3.13`, `python3.13`, `python3`), e se não achar instala o `uv` na pasta do usuário e baixa com ele um Python 3.13 isolado, sem administrador. Cria `.venv` no Windows e `.venv_linux` no Linux (nomes diferentes porque a pasta do projeto pode ser vista pelos dois via `/mnt/c`), instala os dois requirements, no Linux confere `libGL.so.1` e `libglib-2.0.so.0` via `ldconfig`, e roda a verificação da seção 6.1. Flags: `--recriar`, `--sem-verificacao`, `--venv-dir`, `--python`.

Testado em 10/09/2026:

| Sistema | Python encontrado | Venv | Tempo | Resultado |
|---|---|---|---|---|
| Windows 11, `py preparar_ambiente.py` | 3.13.5 do sistema | `.venv` já existente, reaproveitado | 0,1 min | verificação sem falhas |
| WSL Ubuntu 22.04 (só tinha Python 3.10), `python3 preparar_ambiente.py` | nenhum; instalou `uv` 0.12.12 e baixou CPython 3.13.15 | `.venv_linux` criado em `/mnt/c` | 8,7 min na primeira vez, 0,5 min na segunda | verificação sem falhas, libGL e libglib já presentes |

No WSL a instalação em `/mnt/c` é mais lenta que no disco do Linux, mas funciona. As janelas do OpenCV no WSL dependem do WSLg, que existe no Windows 11 e estava ativo (`DISPLAY=:0`).

### 6.3 Backend do Keras

O Keras 3 escolhe o backend **antes** do `import keras`, por uma destas vias, em ordem de prioridade:

1. Variável de ambiente `KERAS_BACKEND` (`tensorflow`, `jax` ou `torch`).
2. Arquivo `~/.keras/keras.json`, chave `"backend"`. O padrão criado na primeira importação é `tensorflow`.

Para definir de forma dinâmica no próprio script, basta setar a variável antes da importação:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print(keras.backend.backend())
```

Trocar depois de importado só com `keras.config.set_backend("...")`, que a própria documentação chama de perigoso: exige `del keras` e reimportar, e invalida camadas e tensores já criados. Os scripts do professor usam `tensorflow.keras`, que é o próprio Keras 3 rodando com backend TensorFlow, então não precisam de configuração. A exigência mínima do Keras 3 para esse backend é TF 2.16.1; o TF 2.21.0 atende.
