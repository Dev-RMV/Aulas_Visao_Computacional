# Achados dos zips de referência

Registro do processamento dos zips baixados do Google Drive do Professor Vitor Amadeu (PVA), feito em 06/09/2026 (aulas 01 a 12) e em 22/09/2026 (aulas 13 a 16).

Cada `aula_NN_Tema/` tem duas subpastas:

- `referencia_de_aula/`: README e o zip original, intocado.
- `exemplos_em_python/`: README e os scripts prontos para uso.

Os nomes das pastas seguem `nome_pastas.txt`, sem acentos e com `_` no lugar de espaços. A aula 3 ficou `aula_03_ROI_SURF_etc`. As aulas 13 a 16, acrescentadas à lista em 22/09/2026, ficaram `aula_13_YOLO_SSD`, `aula_14_OpenCV_DNN`, `aula_15_HSV_ORB_DeepLabV3_ResNet50` (a lista escreve "ResNET50"; a pasta usa a grafia usual do modelo) e `aula_16_Calibracao`.

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
| 13 YOLO e SSD | `exemplos_yolo_ssd_tracking_incrementais-20260923T023509Z-1-001.zip` | 36 .py | 36 | `GUIA_PROFESSOR.md`, notebook `.ipynb` | Parcial: 11 a 15 e 29 pedem `ultralytics`, modelos e mídia reais |
| 14 OpenCV DNN | `exemplos_opencv_dnn_incrementais-20260923T023413Z-1-001.zip` | 31 .py | 31 | `requirements.txt` do professor | Sim, com internet na primeira vez (MobileNetV2 e rótulos); falta o notebook que o 28 cita |
| 15 HSV, ORB, DeepLabV3, ResNet50 | `exemplos_segmentacao_integrativa_incrementais-20260923T023649Z-1-001.zip` | 36 .py | 36 | – | Sim, com internet na primeira vez (pesos do torchvision); falta o relatório modelo que o 34 cita |
| 16 Calibração | `exemplos_calibracao_ar_sinteticos-20260923T023602Z-1-001.zip` | 31 .py | 31 | `dados_sinteticos/` (20 PNG), `saidas/04_cantos_detectados.png` | Sim |

Total: 353 scripts (208 em 06/09/2026, mais 11 da aula 09 em 10/09/2026, mais 134 das aulas 13 a 16 em 22/09/2026). Todos passam em `py_compile` sem erro de sintaxe, no Python 3.13 do venv.

Só as aulas 01 e 02 vieram como documentos Word. As aulas 03 a 16 já vieram com `.py` prontos e foram apenas copiadas.

## 2. Método

### Conversão docx para .py (aulas 01 e 02)

- Cada docx contém um script inteiro, um parágrafo por linha, indentação em espaços.
- Não há `python-docx` instalado. A leitura foi feita com `zipfile` + `xml.etree` direto em `word/document.xml`, tratando `w:t`, `w:tab` e `w:br`.
- Gravação em UTF-8 sem BOM, quebra de linha LF, uma quebra final.
- Verificação: `ast.parse` do texto e diff linha a linha entre o docx e o `.py` gerado. Todos ficaram idênticos.
- Nomes: quando o docstring cita o nome do arquivo, esse nome é usado, porque os scripts se referenciam entre si. Sem citação, o prefixo é o número do título "Exemplo NN" e o resto vem do título.
- Só defeitos de transcrição foram corrigidos e estão listados abaixo. O resto ficou fiel ao professor, inclusive imports duplicados e argumentos não usados.

### Cópia dos .py prontos (aulas 03 a 16)

- A pasta de topo do zip (`exemplos/`, `rastreamento_aula_exemplos/`, `01_fundamentos_convolucao/`) foi achatada para dentro de `exemplos_em_python/`. Subpastas abaixo dela foram preservadas.
- Pastas `__pycache__` foram ignoradas.
- Arquivos de apoio (`utils.py`, `requirements.txt`, README, dados) foram copiados junto.
- Cada arquivo foi conferido por checksum contra o original. Única alteração: um arquivo com CRLF foi normalizado para LF.
- Se o zip traz um `README.txt` na raiz, ele é salvo como `README_do_professor.txt` para não sobrescrever o nosso.
- Aulas 13 a 16 (22/09/2026): os quatro zips tinham uma única pasta de topo, achatada para `exemplos_em_python/`. Todos os arquivos de texto já vinham em UTF-8 sem BOM e com LF, então nenhuma alteração foi necessária: cada arquivo foi conferido byte a byte contra o zip. Só o `__pycache__` da aula 16 (Python 3.12) foi ignorado. Pastas de saída que os scripts usam sem criar (`saidas/` nas quatro aulas, `relatorios/` na 13, `relatorio/` na 15) receberam um `.gitkeep`, e o `.gitignore` deixa de fora o que os scripts geram ou baixam (`dados/`, `modelos/`, `data/`, `imagens/`, pesos `.pt`, `.onnx`, `.weights`).

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

### Aula 13 – YOLO e SSD

- 35 scripts numerados de 01 a 35 mais `vision_utils.py`, `GUIA_PROFESSOR.md` e `notebook_pipeline_yolo_ssd_tracking.ipynb`. Tema: detecção de objetos em tempo real (caixas, confiança, NMS, FPS e latência), modelos reais (YOLOv8 via Ultralytics, YOLOv4-tiny e SSD MobileNet via `cv2.dnn`), rastreamento por IoU com IDs persistentes, trilhas, contagem em linha virtual, ID switches, e análise ética de contagem por drone.
- Raiz dos caminhos é a pasta dos scripts (`Path(__file__).resolve().parent`). O `01_ambiente_e_pastas.py` cria `dados/`, `saidas/`, `modelos/` e `relatorios/`. Os scripts 04 e 10 gravam em `saidas/` e o 33 em `relatorios/` sem criar a pasta, por isso `saidas/.gitkeep` e `relatorios/.gitkeep` foram adicionados. `dados/` e `modelos/` ficam fora do git.
- Autossuficiente até o 10 e de 16 a 35: `vision_utils.py` desenha uma rua sintética (960x540, MJPG a 20 fps) com pessoa, carro, bicicleta e ônibus, e um `fake_detector` que devolve caixas com jitter e uma duplicata de cada uma para o NMS ter o que suprimir. O `IoUTracker` faz associação gulosa por IoU.
- **Precisam de material extra:** 11 (`ultralytics` mais `dados/imagem_teste.jpg`), 12 e 29 (`ultralytics` mais `dados/video_real.mp4`), 13 (`modelos/yolov8n.onnx`, exportado com `yolo export model=yolov8n.pt format=onnx imgsz=640`, mais a imagem), 14 (`modelos/yolov4-tiny.cfg`, `yolov4-tiny.weights` e `coco.names`, do repositório darknet de AlexeyAB), 15 (`modelos/frozen_inference_graph.pb` do SSD MobileNet v2 COCO do TensorFlow model zoo e `ssd_mobilenet_v2_coco.pbtxt` gerado pelo `tf_text_graph_ssd.py` do OpenCV). Os scripts 11, 12 e 29 avisam e encerram sem o `ultralytics`; 13, 14 e 15 avisam e encerram sem os arquivos.
- `YOLO("yolov8n.pt")` baixa o peso no diretório de trabalho, não em `modelos/`; o 16, que mede tamanho de arquivo, só encontra o `.pt` se ele for copiado para lá.
- 17, 18, 30 e 32 são tabelas-modelo com zeros ou `None` para o aluno preencher com medições; 20 e 33 usam números fixos de exemplo (24,5 e 31,2 FPS). O 27 (CSRT) só imprime três linhas de exemplo. O notebook usa `Path.cwd()`, então o Jupyter deve ser aberto de dentro de `exemplos_em_python/`; o pacote `notebook` não está no ambiente unificado.
- Precisa de `cv2`, `numpy`, `pandas`, `matplotlib` e, opcionalmente, `ultralytics`. Nenhum `__pycache__` veio no zip.

### Aula 14 – OpenCV DNN

- 30 scripts numerados de 01 a 30 mais `dnn_utils.py` e o `requirements.txt` do professor (`opencv-python`, `numpy`, `pillow`, `psutil`, `tensorflow`, `pandas`, `matplotlib`, `notebook`). Tema: classificação com MobileNetV2 do Keras convertida para TensorFlow Lite e executada pelo `cv2.dnn` (`blobFromImage`, `readNetFromTFLite`, `forward`, top-3), comparação de latência, memória e acurácia entre OpenCV DNN e Keras, e um pipeline integrado com `undistort`, ROI por HSV, ORB, HOG ou Haar e DNN na ROI.
- Raiz dos caminhos é a pasta dos scripts. `dnn_utils.ensure_dirs()` cria `data/classificacao/`, `data/pipeline/`, `modelos/` e `saidas/`, e é chamado por quase todos os scripts. `data/` e `modelos/` ficam fora do git; `saidas/` tem `.gitkeep`.
- Fluxo obrigatório: o 07 baixa a MobileNetV2 com pesos ImageNet (para `~/.keras/models/`, na primeira vez) e grava `modelos/mobilenetv2_imagenet.tflite`; 08 a 12, 14, 16, 18, 24, 25, 26 e 30 dependem desse arquivo. O 05 baixa `imagenet_labels.txt` do GitHub; sem ele os rótulos viram `classe_000` a `classe_999`. O `cv2.dnn.readNetFromTFLite` existe no OpenCV 4.14 do venv.
- **Defeito:** a mensagem de erro de `get_opencv_net` em `dnn_utils.py` manda executar "06_converter_keras_para_tflite.py", mas o conversor é o 07. Mantido como está.
- **Falta:** o notebook `notebooks/pipeline_integrado_dnn.ipynb` que o 28 diz ter sido fornecido não veio no zip. Também faltam as 10 fotos reais que o 01 pede em `data/classificacao/` e o preenchimento de `data/labels_top1.csv` (o 03 cria o CSV com a coluna de rótulos vazia e o 18 termina com 0/0 sem ela). Sem fotos reais o 03 gera 10 imagens sintéticas, que servem só para testar o fluxo.
- O 29 é pseudocódigo da webcam, deixado como desafio de laboratório. O 20, 25, 26 e 30 usam `DEFAULT_K` e `DEFAULT_DIST` fixos para o `undistort`. Precisa de `cv2`, `numpy`, `tensorflow`, `pandas`, `psutil`. Nenhum `__pycache__` veio no zip.

### Aula 15 – HSV, ORB, DeepLabV3, ResNet50

- 35 scripts numerados de 01 a 35 mais `seg_utils.py`. Tema: segmentação semântica para robótica e veículos autônomos, partindo de cenas sintéticas de rua com máscara ideal de 8 classes, passando por segmentação HSV da pista com morfologia, IoU contra a máscara ideal, FCN-ResNet50 e DeepLabV3-ResNet50 pré-treinados do torchvision, medição de tempo e memória, e terminando em um pipeline integrativo (undistort, HSV, ORB, detector clássico) com relatório, diagrama Mermaid, análise de orçamento de 5 W e checklist de entrega.
- Raiz dos caminhos é a pasta dos scripts. `seg_utils.ensure()` cria `imagens/` e `saidas/`, chamado por `salvar_cenas` em quase todos os scripts. **A pasta `relatorio/` nunca é criada pelos scripts**, e 22, 29 e 30 gravam nela; por isso `relatorio/.gitkeep` foi adicionado. `imagens/` (cenas geradas) e o conteúdo de `relatorio/` ficam fora do git.
- Precisam de `torch` e `torchvision`: 09, 10, 11, 15, 17 e 18. Os pesos (`FCN_ResNet50_Weights.DEFAULT` e `DeepLabV3_ResNet50_Weights.DEFAULT`, COCO com os 21 rótulos VOC) são baixados para `~/.cache/torch/hub/checkpoints/` na primeira execução, cerca de 135 MB e 160 MB. O 20 usa `tabulate`, que não aparece em nenhum requirements do professor. O 19 usa `psutil`.
- 12, 13, 14 e 16 dependem de `saidas/11_mask_fcn.png`, gerado pelo 11, e param com aviso se ele não existir.
- **Falta:** `relatorio/RELATORIO_INTEGRATIVO_MODELO.md`, que o 34 lê; sem ele o script só imprime um aviso. O termo "DR4" (33 e 35) não é definido no material.
- Como a cena sintética não tem objetos que a rede reconheça, a máscara do FCN sai quase toda classe 0; os overlays só fazem sentido com fotos reais de rua. Precisa de `cv2`, `numpy`, `Pillow`, `torch`, `torchvision`, `psutil`, `tabulate`. Nenhum `__pycache__` veio no zip.

### Aula 16 – Calibração

- 30 scripts numerados de 01 a 30 mais `synthetic_utils.py`. Tema: calibração de câmera com tabuleiro de xadrez em um laboratório virtual (câmera sintética com K conhecida), `findChessboardCorners`, `cornerSubPix`, `calibrateCamera`, erro de reprojeção, `undistort`, `solvePnP`, `Rodrigues`, `projectPoints`, eixos e cubo em realidade aumentada, efeito do ruído e da pouca variedade de poses, e a ponte para a webcam real.
- Veio completo: `dados_sinteticos/` com 20 PNG (`calib_01.png` a `calib_18.png`, `tabuleiro_base.png`, `vista_unica.png`) e `saidas/04_cantos_detectados.png`, saída do próprio professor. Os PNG de `dados_sinteticos/` ficam no git; `saidas/` é ignorada e recebeu `.gitkeep`. O zip trazia `__pycache__` de Python 3.12.
- Raiz dos caminhos é a pasta dos scripts. Só precisa de `cv2` e `numpy`. Autossuficiente: o 03 (e todo script que chama `generate_dataset`) regera as 18 imagens, deterministas e iguais às do zip.
- **Detalhes:** 25, 27 e 28 chamam `generate_dataset(noisy=True)` e sobrescrevem `calib_01.png` a `calib_18.png` com versões ruidosas (blur 0,8 e ruído gaussiano 2,0, sem semente); depois deles os scripts 08 a 16 calibram com imagens ruidosas até o 03 regravar as limpas, e o `git status` acusa os PNG modificados. O 26 grava `pouca_var_00.png` a `pouca_var_11.png`, o 23 grava `video_tabuleiro.avi` e o 13 grava `camera_sintetica.npz`, todos fora do git. O 21 não checa o retorno de `find_refined_corners` antes do `solvePnP`.

## 4. Pendências

Material a providenciar antes de rodar:

- [ ] **Aula 04:** `exemplos_em_python/assets/carro_real.png`.
- [ ] **Aula 05:** pasta `aula_05_HSV/recursos/` com o par estéreo, `alvo_hsv.png` e `video_alvo.mp4`, ou o script `gerar_recursos.py` que os cria.
- [ ] **Aula 06:** pasta `aula_06_Reconhecimento_Facial/recursos/` com `foto_grupo.jpg`, `identidades/`, `video_teste.mp4` e, se houver, `gabarito_video.csv`.
- [x] **Aula 09:** resolvido em 10/09/2026, faltavam dois zips. Ainda faltam para a atividade: vídeo de pedestres (item A) e dataset real de positivas e negativas (item B).
- [ ] **Aula 11:** fotos `0.jpg` a `9.jpg` em `exemplos_em_python/data/real_digits/`.
- [ ] **Aula 12:** pacote `aula_12_DNN_Caffe/utils/` (`face_utils.py`, `tf_utils.py`), os 4 arquivos Caffe, `data/faces_teste/` e `data/utkface_sample/`.
- [ ] **Aula 03:** confirmar se esse zip agregado é mesmo o da aula 3.
- [ ] **Aula 13:** mídia real em `exemplos_em_python/dados/` (`imagem_teste.jpg`, `video_real.mp4`) e modelos em `exemplos_em_python/modelos/` (`yolov8n.onnx` exportado pelo ultralytics; `yolov4-tiny.cfg`, `yolov4-tiny.weights`, `coco.names`; `frozen_inference_graph.pb` e `ssd_mobilenet_v2_coco.pbtxt`). Jupyter à parte para abrir o notebook.
- [ ] **Aula 14:** o notebook `notebooks/pipeline_integrado_dnn.ipynb` citado pelo 28 não veio no zip. Para a atividade: 10 fotos reais em `exemplos_em_python/data/classificacao/` e a coluna `label_esperado` de `data/labels_top1.csv` preenchida. Internet na primeira execução do 05 e do 07.
- [ ] **Aula 15:** `exemplos_em_python/relatorio/RELATORIO_INTEGRATIVO_MODELO.md` citado pelo 34 não veio no zip. Fotos reais de rua para os overlays do FCN fazerem sentido. Internet na primeira execução de 10, 11, 15, 17 e 18.
- [x] **Aula 16:** veio completa, nada a providenciar.

Ambiente:

- [x] O professor usou Python 3.12 (aulas 04, 05, 06, 12, 16) e 3.13 (aula 10). A máquina local tem Python 3.14.3. Resolvido em 10/09/2026 com o venv de Python 3.13 (seção 6).
- [x] `dlib` e `face_recognition` (aulas 03 e 06) costumam exigir wheel pré-compilado no Windows. Resolvido com `dlib-bin` e `--no-deps` (seção 6).
- [x] `opencv-contrib-python` para o SURF opcional da aula 03. Instalado; o SURF continua bloqueado por ser nonfree, o script cai no ORB (seção 6).
- [x] Aulas 13 a 16 (22/09/2026): `torch`, `torchvision`, `tabulate` e as dependências do `ultralytics` entraram no `requirements.txt`; o `ultralytics` entrou em `requirements_ultralytics.txt` com `--no-deps` (seção 6).

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
| 13 YOLO e SSD | cv2, numpy, pandas, matplotlib, ultralytics (opcional: 11, 12, 29) |
| 14 OpenCV DNN | cv2, numpy, tensorflow, pandas, psutil |
| 15 HSV, ORB, DeepLabV3, ResNet50 | cv2, numpy, PIL, torch, torchvision, psutil, tabulate |
| 16 Calibração | cv2, numpy |

Os nomes de pacote e as versões fixadas estão em `requirements.txt` na raiz. A justificativa está na seção 6.

## 6. Ambiente unificado (`requirements.txt`, 10/09/2026, ampliado em 22/09/2026)

Objetivo: um único venv que rode todos os 353 scripts sem conflito. Versões conferidas no PyPI em 10/09/2026 (aulas 01 a 12) e em 22/09/2026 (aulas 13 a 16). O venv `.venv` foi criado na raiz com o Python 3.13.5 da máquina, os requirements foram instalados e os testes de runtime abaixo passaram.

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
| **torch==2.14.0, torchvision==0.29.0** (22/09/2026) | Aula 15 (FCN-ResNet50 e DeepLabV3 do torchvision) e base do `ultralytics` da aula 13. Mais recentes no PyPI em 22/09/2026, com wheel para 3.13; o torchvision 0.29 é o par do torch 2.14. A wheel `torch` do PyPI é só CPU no Windows (aparece como `2.14.0+cpu`); para GPU, usar o índice de pytorch.org. Os pesos dos modelos são baixados para `~/.cache/torch/hub/checkpoints/` na primeira execução. Instalado com `-c requirements.txt` como restrição, para não mexer nos pinos anteriores: numpy continuou 2.5.3 e nada foi rebaixado. |
| **ultralytics==8.4.160 com `--no-deps`** (22/09/2026) | Aula 13, scripts 11, 12 e 29 (YOLOv8n). Declara dependência de `opencv-python!=4.13.0.90,>=4.7.0`, que o pip instalaria por cima do `opencv-contrib-python` (mesmo módulo `cv2`, e o `cv2.face` da aula 03 sumiria). Mesma solução do face-recognition: fica em `requirements_ultralytics.txt` e entra com `--no-deps`; as dependências dele (`torch`, `torchvision`, `numpy`, `matplotlib`, `Pillow`, `PyYAML`, `requests`, `psutil`, `polars`, `cloudpickle`, `filelock`, `nvidia-ml-py`, `ultralytics-thop`, `ultralytics-platform`) estão no `requirements.txt`, fixadas nas versões instaladas. Efeito colateral esperado: `pip check` também reclama que falta `opencv-python`. Ignorar. Os extras de exportação (`onnx`, para o `yolo export` do script 13) não foram instalados. |
| **tabulate==0.10.0, psutil==7.2.2** (22/09/2026) | `tabulate` é importado pelo script 20 da aula 15 e não aparece em nenhum requirements do professor. `psutil` (aulas 14 e 15) já estava no venv como dependência indireta e passou a ser pino explícito. |

Instalação, no PowerShell, a partir da raiz do projeto:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -r requirements_face_recognition.txt
pip install --no-deps -r requirements_ultralytics.txt
```

### 6.1 Verificação feita no venv (10/09/2026)

- `import keras` com `KERAS_BACKEND=tensorflow`: Keras 3.15.1, backend `tensorflow`, tensores do TF, modelo `Sequential` compila e treina uma época.
- `import keras` sem variável: mesmo resultado, porque o `~/.keras/keras.json` padrão já traz `"backend": "tensorflow"`.
- `tf.keras.__version__` é 3.15.1 e `tf.keras.Sequential` é o mesmo `keras.Sequential`: a partir do TF 2.16 `tensorflow.keras` é o Keras 3. `tf.keras.preprocessing.image.ImageDataGenerator` (aula 11) ainda existe.
- OpenCV 4.14.0 com `cv2.face` (LBPH cria), `cv2.dnn`, `SIFT_create`, `HOGDescriptor` com o detector de pedestres. SURF presente mas bloqueado (nonfree).
- `dlib` 20.0.1 e `face_recognition` importam e `face_locations` roda.

Acréscimo de 22/09/2026 (aulas 13 a 16), no mesmo venv:

- `torch` 2.14.0+cpu e `torchvision` 0.29.0+cpu importam; `fcn_resnet50`, `deeplabv3_resnet50` e `transforms.Compose` com `Normalize` funcionam; os pesos `DEFAULT` das duas redes declaram 21 categorias. Uma operação de tensor roda. `torch.cuda.is_available()` é falso, como esperado na wheel do PyPI.
- `ultralytics` 8.4.160 importa e `from ultralytics import YOLO` funciona; na primeira importação ele grava `%APPDATA%\Ultralytics\settings.json`. O peso `yolov8n.pt` só é baixado quando um script chama `YOLO("yolov8n.pt")`.
- `tabulate` 0.10.0 e `psutil` 7.2.2 importam.
- Depois da instalação: OpenCV continua 4.14.0 com `cv2.face`, `cv2.dnn.readNetFromTFLite` (aula 14) e `SIFT_create`; TensorFlow 2.21 e Keras 3.15 intactos; `face_recognition` intacto; só `opencv-contrib-python` instalado, nenhum `opencv-python`.
- `pip check`: as duas únicas queixas são `face-recognition requires dlib` e `ultralytics requires opencv-python`, ambas esperadas.
- `py preparar_ambiente.py` rodado de novo depois das mudanças: venv reaproveitado, os três requirements instalados, verificação (agora com torch, torchvision, tabulate, psutil e ultralytics) concluída sem falhas em 0,1 min.

### 6.2 Script `preparar_ambiente.py` (Windows e Linux)

Automatiza a seção 6 inteira: detecta o sistema (Windows, Linux ou WSL), procura um Python 3.13 (`py -3.13`, `python3.13`, `python3`), e se não achar instala o `uv` na pasta do usuário e baixa com ele um Python 3.13 isolado, sem administrador. Cria `.venv` no Windows e `.venv_linux` no Linux (nomes diferentes porque a pasta do projeto pode ser vista pelos dois via `/mnt/c`), instala os três requirements (o principal e os dois com `--no-deps`, desde 22/09/2026), no Linux confere `libGL.so.1` e `libglib-2.0.so.0` via `ldconfig`, e roda a verificação da seção 6.1, que desde 22/09/2026 também cobre torch, torchvision, tabulate, psutil e ultralytics. Flags: `--recriar`, `--sem-verificacao`, `--venv-dir`, `--python`.

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
