# Tutoriais das aulas

Índice dos tutoriais escritos a partir dos scripts do Professor Vitor Amadeu (PVA). Cada aula tem um `tutorial_da_aula.md` na raiz da sua pasta, com a mesma estrutura em oito seções: visão geral, ambiente, conceitos-chave (com trecho do código do professor e onde aparece), roteiro dos scripts, passo a passo, exercícios e itens, armadilhas e glossário.

Gerados em 10/09/2026. Nenhum script foi executado nem alterado para escrever os tutoriais. Tudo foi tirado da leitura integral dos 219 scripts.

## Índice

| Aula | Tutorial | Scripts | Conceitos | Roda sem material extra? |
|---|---|---|---|---|
| 01 OpenCV | [aula_01_OpenCV/tutorial_da_aula.md](aula_01_OpenCV/tutorial_da_aula.md) | 8 | 16 | Sim, webcam em 03, 04, 06 e 07 |
| 02 Câmera | [aula_02_Camera/tutorial_da_aula.md](aula_02_Camera/tutorial_da_aula.md) | 11 | 18 | Sim, webcam gera a imagem-base |
| 03 ROI, SURF e outros | [aula_03_ROI_SURF_etc/tutorial_da_aula.md](aula_03_ROI_SURF_etc/tutorial_da_aula.md) | 28 | 31 | Quase: 10, 15 e 26 pedem fotos ou dataset próprios |
| 04 Tensores | [aula_04_Tensor/tutorial_da_aula.md](aula_04_Tensor/tutorial_da_aula.md) | 12 | 12 | Não, falta `assets/carro_real.png` |
| 05 HSV, segmentação e estéreo | [aula_05_HSV/tutorial_da_aula.md](aula_05_HSV/tutorial_da_aula.md) | 11 + utils | 20 | Não, falta `recursos/` |
| 06 Reconhecimento facial | [aula_06_Reconhecimento_Facial/tutorial_da_aula.md](aula_06_Reconhecimento_Facial/tutorial_da_aula.md) | 11 + utils | 20 | Não, falta `recursos/` |
| 07 Homografia | [aula_07_Homografia/tutorial_da_aula.md](aula_07_Homografia/tutorial_da_aula.md) | 13 + utils | 24 | Sim |
| 08 TensorFlow e Keras | [aula_08_TensorFlow_Keras/tutorial_da_aula.md](aula_08_TensorFlow_Keras/tutorial_da_aula.md) | 13 + utils | 19 | Sim |
| 09 Convolução, CNN básica e HOG + SVM | [aula_09_HOG_SVM/tutorial_da_aula.md](aula_09_HOG_SVM/tutorial_da_aula.md) | 25 | 30 | Parcial: item A pede vídeo de pedestres, item B pede dataset real |
| 10 Rastreamento | [aula_10_Rastreamento/tutorial_da_aula.md](aula_10_Rastreamento/tutorial_da_aula.md) | 11 + 2 apoio | 19 | Sim |
| 11 MNIST | [aula_11_MNIST/tutorial_da_aula.md](aula_11_MNIST/tutorial_da_aula.md) | 32 + utils | 30 | Parcial, falta `data/real_digits/` |
| 12 DNN e Caffe | [aula_12_DNN_Caffe/tutorial_da_aula.md](aula_12_DNN_Caffe/tutorial_da_aula.md) | 35 + bootstrap | 32 | Não, falta `utils/`, modelos e dados |

## Conceitos por aula

- **Aula 01 OpenCV.** Imagem como array NumPy em BGR, primitivas de desenho, `imwrite` e `imread`, `cvtColor`, `hconcat`, `imshow` e `waitKey`, `VideoWriter` com fourcc, `VideoCapture`, FPS com `getTickCount` e média móvel com `deque`, captura por tecla, `try/finally` para liberar a câmera.
- **Aula 02 Câmera.** Espaços de cor HSV e LAB e seus canais, alteração de saturação com `clip`, sharpening com `filter2D`, unsharp masking, variância do Laplaciano como medida de nitidez, limiarização global, adaptativa e Otsu, Canny com dois pares de thresholds, morfologia, contornos e classificação por área.
- **Aula 03 ROI, SURF e outros.** Cinco blocos: ROI, HSV, Otsu, morfologia e GrabCut; par estéreo, disparidade SGBM e profundidade; faces com Haar, LBPH e embeddings; SIFT, ORB, matching com ratio test, homografia com RANSAC, SURF opcional e recuperação de imagens; MLP e CNN no Fashion-MNIST, augmentation, callbacks, extração de features e transfer learning com MobileNetV2.
- **Aula 04 Tensores.** Escalar, vetor, matriz e tensor, rank, shape e dtype, pixel RGB, imagem HWC, leitura e decodificação com `tf.io`, dimensão de lote, RGB para cinza, split e concat de canais, normalização para 0 a 1, resize, `Conv2D` e pipeline `tf.data`.
- **Aula 05 HSV, segmentação e estéreo.** StereoBM e StereoSGBM com seus parâmetros, máscara de validade e normalização por percentis, relação Z igual a f vezes B sobre d, HSV e `inRange`, vermelho cruzando o zero do Hue, morfologia, bounding box e proporção da ROI, vídeo com `VideoWriter`, FPS, trackbars, integração cor mais disparidade.
- **Aula 06 Reconhecimento facial.** Haar Cascade e `detectMultiScale`, `equalizeHist`, ROI 48x48, comparação de parâmetros contra gabarito, embeddings com `face_recognition`, cadastro de identidades, tolerância e varredura, benchmark para embarcado, ética em biometria.
- **Aula 07 Homografia.** SIFT, ORB e AKAZE, BFMatcher com crossCheck, FLANN com teste de Lowe, `findHomography` com RANSAC, inliers, reprojeção e erro contra a homografia real, robustez a rotação, escala e luz, localização visual em vídeo.
- **Aula 08 TensorFlow e Keras.** Sementes, dataset sintético de formas, imagem como tensor, convolução passo a passo, CNN com Keras, treino e avaliação, filtros e ativações, features com PCA, ORB nas mesmas imagens, bag of visual words, comparação CNN contra ORB, latência para embarcado.
- **Aula 09 Convolução, CNN básica e HOG + SVM.** Três partes. Convolução: imagem como matriz, kernel deslizante, `filter2D`, Sobel, blur, sharpen, stride, padding, ReLU, max pooling. CNN básica: CNN no MNIST e no CIFAR-10, kernels e feature maps, MobileNetV2 congelada, classificação contra detecção. HOG + SVM: detector de pedestres do OpenCV com cenários rápido e preciso, dataset 64x128, `HOGDescriptor` com vetor de 3780, `SVC` com probabilidade, janela deslizante com pirâmide e NMS.
- **Aula 10 Rastreamento.** MOG2 e KNN, limpeza por threshold e morfologia, contornos e bounding box, `selectROI`, histograma HSV e backprojection, CamShift, filtro de Kalman com predict e correct, varredura de parâmetros, CSV e gráfico de trajetórias.
- **Aula 11 MNIST.** Carregamento e normalização, one-hot, MLP e CNN, curvas de treino, matriz de confusão, filtros e feature maps, overfitting, dropout, early stopping, salvar e carregar, dígitos reais pré-processados com OpenCV, data augmentation, domain gap, webcam.
- **Aula 12 DNN e Caffe.** Bootstrap e estrutura de pastas, Haar e ROI, `cv2.dnn` com `blobFromImage`, inferência de gênero e idade com modelos Caffe, FPS em imagem, webcam e vídeo, CSV, UTKFace com `tf.data`, MobileNetV2 com cabeça nova e fine-tuning, avaliação, comparação, TFLite, embarcado e ética. Inclui a tabela das 14 funções esperadas do pacote `utils/` que não veio no zip.

## Defeitos e detalhes encontrados na leitura

Nada foi corrigido nos scripts. Fica registrado para quando forem executados.

- Aula 02: `12_exercicio2_item_a_completo.py` chama `destroyWindow` com um nome de janela que não existe. `18_contornos_por_area.py` sempre usa a webcam, não lê `imagem_base.png`.
- Aula 03: `09_gerar_mapa_disparidade.py` sobrescreve o `disparidade_float.npy` do 08. `25_extrair_features_cnn.py` falha com o modelo MLP do 22 por diferença de forma de entrada. Os scripts 00 e 15 citam arquivos de requirements que não vieram.
- Aula 06: `05_comparar_parametros.py` dá `KeyError` sem `gabarito_video.csv`. `10_varrer_tolerancia.py` falha sem mensagem se o 06 não rodou antes.
- Aula 07: `06_bfmatcher_crosscheck.py` dá `IndexError` se não houver matches.
- Aula 08: `01_gerar_dataset.py` apaga `dataset/` inteira antes de regenerar. As mensagens de erro do `utils.py` citam a pasta `exemplos/`, nome original do professor.
- Aula 09: `item_b_treinar_svm.py` sobrescreve o `modelo_hog_svm.joblib` do professor. `gerar_dataset_sintetico_demo.py` e `item_b_janela_deslizante.py` gravam em `dados/` e `saidas/` sem criar as pastas, por isso elas foram criadas vazias. O item A encerra com Esc, não com q. O `.joblib` depende da versão do scikit-learn para carregar.
- Aula 10: o README do professor cita a tecla `p` para pausar, mas nenhum script implementa. `--roi` sem valores dá `ValueError`.
- Aula 11: scripts 26, 27, 28 e 30 usam `ImageDataGenerator`, API legada que avisa depreciação no Keras 3. Sem semente fixa, os resultados variam. O 30 sobrescreve o modelo salvo pelo 13.
- Aula 12: `33_pipeline_final_item_b.py` chama scripts em `exemplos/` e falha no nosso layout. O 26 plota curvas simuladas e o 31 usa números fixos de exemplo. Scripts 20 a 26 gravam em `resultados/` sem criar a pasta, dependem do 02.
