# Aula 03 – ROI, SURF e outros temas

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula é, na prática, um percurso completo de visão computacional em 28 scripts. Ela começa no nível do pixel (ler uma imagem, recortar uma região de interesse, segmentar por cor e por limiar, limpar máscaras com morfologia, extrair contornos e caixas delimitadoras, separar primeiro plano com GrabCut), passa pela visão estéreo (par de imagens, mapa de disparidade com StereoSGBM e conversão para profundidade métrica), entra em faces (detecção com Haar Cascade em imagem e webcam, coleta de amostras, reconhecimento com LBPH e com embeddings do pacote `face_recognition`), segue para features locais (keypoints e descritores SIFT, ORB e SURF, matching por força bruta com cross-check e com ratio test de Lowe, homografia com RANSAC, recuperação de imagens por similaridade) e termina em aprendizado profundo (MLP e CNN no Fashion-MNIST, aumento de dados, callbacks, extração de features com PCA, transfer learning com MobileNetV2 e um pipeline OpenCV + Keras rodando na webcam).

O fio condutor é a noção de região de interesse (ROI): primeiro a ROI é recortada à mão, depois é encontrada por cor, por contorno, por GrabCut, por detector de faces, por matching de descritores e, no final, é a entrada de uma rede neural. Quase tudo é autossuficiente: o script 01 gera a cena sintética que alimenta os exemplos de segmentação e de features, o 07 gera o par estéreo, o Haar cascade vem embutido no OpenCV e o Fashion-MNIST é baixado pelo Keras.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas nos scripts:

| Biblioteca | Pacote pip | Scripts |
|---|---|---|
| OpenCV | `opencv-contrib-python` | quase todos. O módulo `cv2.face` (LBPH, scripts 13 e 14) só existe no pacote contrib. SURF (script 20) exige build com módulos nonfree, que não vem nos wheels oficiais; o script tem fallback para SIFT. |
| NumPy | `numpy` | quase todos |
| TensorFlow / Keras | `tensorflow` | 22 a 27 |
| scikit-learn | `scikit-learn` | 22 (matriz de confusão) e 25 (PCA) |
| Matplotlib | `matplotlib` | 22, 23, 25 |
| face_recognition | `face-recognition` (depende de `dlib`) | 15 apenas |

Instalação sugerida, de preferência em um ambiente virtual:

```
pip install opencv-contrib-python numpy tensorflow scikit-learn matplotlib
pip install face-recognition
```

Observações importantes sobre o ambiente:

- Os scripts citam `requirements-base.txt` e `requirements-opcionais.txt` no `00_verificar_ambiente.py` e no `15_reconhecimento_embeddings.py`. Esses arquivos não vieram no zip. A tabela acima substitui os dois.
- `dlib` e `face-recognition` são difíceis de instalar no Windows: exigem CMake e compilador C++ ou wheel pré-compilado. Só o script 15 depende deles.
- A máquina local tem Python 3.14. O professor usou 3.12 e 3.13 em outras aulas. Se `tensorflow` ou `dlib` não tiverem wheel para 3.14, crie o venv com 3.12.
- No Windows nativo o TensorFlow roda só em CPU. Os treinos dos scripts 22 a 24 levam alguns minutos cada.

De que pasta rodar: todos os caminhos são relativos ao diretório de trabalho (`dados/gerados/`, `dados/faces_treino/`, `modelos/`). Os docstrings do professor mostram `python exemplos/nome.py`, ou seja, ele rodava a partir da pasta acima de `exemplos/`. Aqui, a forma mais simples é entrar em `exemplos_em_python/` e rodar `python nome.py`. As pastas `dados/` e `modelos/` serão criadas ali dentro. O que importa é usar sempre a mesma pasta de trabalho, senão um script não encontra o que o outro gerou.

Material que falta (não veio no zip):

- Fotos com rostos para os scripts 10 (`--imagem` é obrigatório) e 15 (`--conhecidas` e `--consulta`). Use fotos próprias.
- Dataset por pastas `dados/dataset_custom/treino/<classe>/` e `validacao/<classe>/` para o script 26. O docstring diz "conforme o README", mas o README do professor não veio. Monte o dataset com fotos próprias em duas ou mais classes.

Webcam: usada pelos scripts 00 (teste), 11, 12, 14 e 27. Todos aceitam `--camera N` para trocar o índice.

## 3. Conceitos-chave

Os conceitos estão agrupados em cinco blocos, na ordem dos scripts: fundamentos e segmentação (3.1 a 3.8), estéreo (3.9 a 3.11), faces (3.12 a 3.17), features locais (3.18 a 3.23) e aprendizado profundo (3.24 a 3.31).

### Bloco A: fundamentos, ROI e segmentação

### 3.1 Verificação do ambiente

Antes de qualquer aula prática vale confirmar as versões instaladas, os recursos opcionais do OpenCV e se a câmera responde. O professor faz isso com `importlib.import_module` dentro de um `try`, lendo o atributo `__version__`, e com `hasattr(cv2, ...)` para descobrir se SIFT, ORB, `cv2.face` e SURF existem no build. A câmera é testada abrindo `cv2.VideoCapture(0)`, lendo um frame e liberando o dispositivo.

```python
# 00_verificar_ambiente.py
def versao(modulo: str) -> str:
    try:
        pacote = importlib.import_module(modulo)
        return str(getattr(pacote, "__version__", "instalado"))
    except Exception as erro:
        return f"não disponível ({type(erro).__name__})"

print("SIFT:", hasattr(cv2, "SIFT_create"))
print("Módulo cv2.face:", hasattr(cv2, "face"))
camera = cv2.VideoCapture(0)
ok, _ = camera.read() if camera.isOpened() else (False, None)
camera.release()
```

Onde aparece: `00_verificar_ambiente.py`.

### 3.2 Imagem como array NumPy e desenho de primitivas

No OpenCV uma imagem colorida é um array NumPy de forma `(altura, largura, 3)` com `dtype=uint8` e canais na ordem BGR (azul, verde, vermelho), não RGB. O script 01 cria uma tela cinza-clara com `np.full` e desenha objetos com `cv2.rectangle`, `cv2.circle`, `cv2.fillPoly`, `cv2.line` e `cv2.putText`. Espessura `-1` preenche a forma. Linhas brancas e o texto "CV" existem de propósito: criam cantos e bordas que viram keypoints para SIFT e ORB mais adiante.

Para simular uma foto real, o professor soma ruído gaussiano com `np.random.default_rng(semente).normal(0, 10, forma)`. O truque é converter para `int16` antes de somar (para permitir valores negativos e acima de 255), aplicar `np.clip(..., 0, 255)` e voltar para `uint8`. Sem isso o `uint8` estoura e a imagem fica corrompida.

O mesmo script produz uma versão rotacionada e reduzida do objeto com `cv2.getRotationMatrix2D(centro, angulo, escala)` seguida de `cv2.warpAffine`. O parâmetro `borderValue` preenche os cantos vazios com a cor do fundo.

```python
# 01_gerar_cena_sintetica.py
cena = np.full((480, 640, 3), (238, 238, 238), dtype=np.uint8)
cv2.rectangle(cena, (60, 80), (250, 250), (255, 80, 20), -1)  # azul em BGR
cv2.circle(cena, (430, 160), 85, (40, 210, 40), -1)
ruido = rng.normal(0, 10, imagem.shape).astype(np.int16)
cena = np.clip(imagem.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

objeto = cena[70:260, 50:260].copy()
matriz = cv2.getRotationMatrix2D((objeto.shape[1] / 2, objeto.shape[0] / 2), 18, 0.9)
transformado = cv2.warpAffine(objeto, matriz, (objeto.shape[1], objeto.shape[0]),
                              borderValue=(238, 238, 238))
```

Onde aparece: `01_gerar_cena_sintetica.py`, `07_criar_par_estereo_sintetico.py`.

### 3.3 Leitura, escrita, exibição e recorte de ROI

`cv2.imread(caminho)` devolve `None` quando o arquivo não existe, então o professor sempre testa `if imagem is None` e lança `FileNotFoundError`. `cv2.imwrite` devolve `False` em caso de falha. `cv2.imshow` abre uma janela, `cv2.waitKey(0)` espera uma tecla e `cv2.destroyAllWindows` fecha tudo.

Uma ROI (Region of Interest, região de interesse) é só um fatiamento NumPy: `imagem[y1:y2, x1:x2]`. A ordem é linha (y) antes de coluna (x), o oposto da convenção `(x, y)` usada nas funções de desenho do OpenCV. O script 02 ainda limita as coordenadas aos limites da imagem com `max`/`min` para não gerar fatia vazia. Note que a fatia é uma vista do array original: o script 01 usa `.copy()` quando precisa de um objeto independente.

```python
# 02_ler_e_recortar_roi.py
imagem = cv2.imread(args.imagem)
if imagem is None:
    raise FileNotFoundError(f"Imagem não encontrada: {args.imagem}. Execute o exemplo 01 primeiro.")
altura_img, largura_img = imagem.shape[:2]
x1 = max(0, min(args.x, largura_img - 1))
y1 = max(0, min(args.y, altura_img - 1))
x2 = max(x1 + 1, min(x1 + args.largura, largura_img))
y2 = max(y1 + 1, min(y1 + args.altura, altura_img))
roi = imagem[y1:y2, x1:x2]
print("Lembrete: NumPy indexa como imagem[y, x], não imagem[x, y].")
```

Onde aparece: `02_ler_e_recortar_roi.py` e, como padrão, em todos os scripts que leem imagem.

### 3.4 Espaço de cor HSV e segmentação por cor

Segmentar por cor em BGR é ruim porque a iluminação altera os três canais ao mesmo tempo. Em HSV a cor fica isolada no canal H (matiz), enquanto S (saturação) e V (valor, brilho) absorvem as variações de luz. No OpenCV, H vai de 0 a 179 (não 0 a 359) e S e V vão de 0 a 255.

`cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)` faz a conversão. `cv2.inRange(hsv, inferior, superior)` gera uma máscara binária (0 ou 255) com os pixels dentro do intervalo. `cv2.bitwise_and(imagem, imagem, mask=mascara)` mantém só os pixels da máscara. `cv2.countNonZero` conta os pixels selecionados. O intervalo padrão `H` de 90 a 135 seleciona o retângulo azul da cena sintética.

```python
# 03_segmentar_cor_hsv.py
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
limite_inferior = np.array([args.h_min, args.s_min, args.v_min], dtype=np.uint8)
limite_superior = np.array([args.h_max, 255, 255], dtype=np.uint8)
mascara = cv2.inRange(hsv, limite_inferior, limite_superior)
resultado = cv2.bitwise_and(imagem, imagem, mask=mascara)
proporcao = 100.0 * cv2.countNonZero(mascara) / mascara.size
```

Para exibir máscara (1 canal) ao lado da imagem (3 canais) com `np.hstack`, o professor converte a máscara com `cv2.COLOR_GRAY2BGR`.

Onde aparece: `03_segmentar_cor_hsv.py`, `05_morfologia_contornos_roi.py`.

### 3.5 Limiarização global, de Otsu e adaptativa

Limiarizar (threshold) é transformar uma imagem em tons de cinza em binária: pixels acima do limiar viram 255, abaixo viram 0. O script 04 compara três estratégias, sempre depois de um `cv2.GaussianBlur` com kernel 5x5 para reduzir ruído:

- **Global**: `cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)` usa um valor fixo. Falha quando a iluminação não é uniforme.
- **Otsu**: `cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)` escolhe o limiar automaticamente maximizando a separação entre as duas classes do histograma. O valor de entrada (0) é ignorado e o limiar escolhido é devolvido como primeiro retorno.
- **Adaptativa**: `cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)` calcula um limiar diferente para cada pixel a partir da média ponderada gaussiana de uma vizinhança de 31x31, menos a constante 5. Lida bem com sombras e gradientes.

```python
# 04_limiarizacao_otsu.py
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)
_, global_ = cv2.threshold(suavizada, 127, 255, cv2.THRESH_BINARY)
limiar_otsu, otsu = cv2.threshold(suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptativa = cv2.adaptiveThreshold(suavizada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
mosaico = np.hstack([cinza, global_, otsu, adaptativa])
```

Onde aparece: `04_limiarizacao_otsu.py`, `27_pipeline_opencv_cnn_webcam.py` (Otsu invertido na ROI da webcam).

### 3.6 Morfologia matemática: abertura e fechamento

Máscaras vindas de `inRange` ou `threshold` costumam ter pontos isolados de ruído e buracos dentro dos objetos. A morfologia corrige isso com um elemento estruturante (kernel), aqui uma elipse 7x7 criada por `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))`.

- **Abertura** (`cv2.MORPH_OPEN`): erosão seguida de dilatação. Remove pontos brancos pequenos sem mudar o tamanho dos objetos grandes.
- **Fechamento** (`cv2.MORPH_CLOSE`): dilatação seguida de erosão. Fecha buracos e fendas pretas dentro dos objetos.

O parâmetro `iterations` repete a operação. O professor usa uma abertura e dois fechamentos.

```python
# 05_morfologia_contornos_roi.py
mascara = cv2.inRange(hsv, np.array([85, 60, 30]), np.array([140, 255, 255]))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
abertura = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
limpa = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel, iterations=2)
```

Onde aparece: `05_morfologia_contornos_roi.py`.

### 3.7 Contornos, área e bounding boxes

Com a máscara limpa, `cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` devolve a lista de contornos. `RETR_EXTERNAL` pega só os contornos mais externos (ignora buracos). `CHAIN_APPROX_SIMPLE` comprime segmentos retos guardando só os vértices. O segundo retorno (hierarquia) é descartado com `_`.

Para cada contorno, `cv2.contourArea` dá a área em pixels, usada para descartar regiões pequenas (`--area-minima`, padrão 1000). `cv2.boundingRect` devolve `(x, y, largura, altura)` do retângulo alinhado aos eixos, que vira a ROI recortada por fatiamento e salva em `dados/gerados/rois/roi_NN.png`. O professor ordena os contornos por área decrescente com `sorted(contornos, key=cv2.contourArea, reverse=True)`.

```python
# 05_morfologia_contornos_roi.py
contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contorno in sorted(contornos, key=cv2.contourArea, reverse=True):
    area = cv2.contourArea(contorno)
    if area < args.area_minima:
        continue
    x, y, largura, altura = cv2.boundingRect(contorno)
    roi = imagem[y : y + altura, x : x + largura]
    cv2.imwrite(str(saida / f"roi_{aceitos:02d}.png"), roi)
    cv2.rectangle(anotada, (x, y), (x + largura, y + altura), (0, 0, 0), 3)
```

Onde aparece: `05_morfologia_contornos_roi.py`.

### 3.8 GrabCut: segmentação interativa de primeiro plano

GrabCut separa primeiro plano de fundo a partir de um retângulo inicial que contém o objeto e uma borda de fundo. O algoritmo modela as cores de cada lado com misturas gaussianas (os arrays `modelo_fundo` e `modelo_frente`, de forma `(1, 65)`) e refina a divisão iterativamente com corte em grafo.

`cv2.grabCut(imagem, mascara, retangulo, modelo_fundo, modelo_frente, iteracoes, cv2.GC_INIT_WITH_RECT)` escreve na `mascara` um de quatro rótulos por pixel: `GC_BGD` (fundo certo), `GC_FGD` (frente certa), `GC_PR_BGD` (provável fundo) e `GC_PR_FGD` (provável frente). Para obter o objeto, mantém-se frente certa ou provável e multiplica-se a imagem pela máscara binária expandida para 3 canais com `[:, :, np.newaxis]`.

```python
# 06_grabcut_roi.py
mascara = np.zeros(imagem.shape[:2], np.uint8)
modelo_fundo = np.zeros((1, 65), np.float64)
modelo_frente = np.zeros((1, 65), np.float64)
retangulo = (args.x, args.y, args.largura, args.altura)
cv2.grabCut(imagem, mascara, retangulo, modelo_fundo, modelo_frente,
            args.iteracoes, cv2.GC_INIT_WITH_RECT)
frente = np.where((mascara == cv2.GC_FGD) | (mascara == cv2.GC_PR_FGD), 1, 0).astype("uint8")
resultado = imagem * frente[:, :, np.newaxis]
```

Onde aparece: `06_grabcut_roi.py`.

### Bloco B: visão estéreo

### 3.9 Par estéreo e disparidade

Duas câmeras lado a lado, separadas por uma distância chamada baseline, veem o mesmo ponto em colunas diferentes. Essa diferença horizontal, em pixels, é a disparidade. Objetos próximos têm disparidade grande, objetos distantes têm disparidade pequena. Para que a busca de correspondência seja só horizontal, as imagens precisam estar retificadas: pontos correspondentes ficam na mesma linha.

O script 07 constrói esse par de forma controlada: um fundo texturizado com `rng.integers` (textura é essencial, porque regiões lisas não têm correspondência confiável), três objetos rotulados PERTO, MEIO e LONGE, e cada objeto colado na imagem direita deslocado para a esquerda por 48, 28 e 12 pixels. Assim o resultado do SGBM pode ser conferido contra valores conhecidos.

```python
# 07_criar_par_estereo_sintetico.py
textura = rng.integers(70, 180, size=(h, w, 1), dtype=np.uint8)
esquerda = np.repeat(textura, 3, axis=2)
direita = esquerda.copy()
# (x, y, disparidade): perto desloca mais entre esquerda e direita.
posicoes = [(90, 190, 48), (300, 120, 28), (500, 55, 12)]
for obj, (x, y, disparidade) in zip(objetos, posicoes):
    colar(esquerda, obj, x, y)
    colar(direita, obj, x - disparidade, y)
```

Onde aparece: `07_criar_par_estereo_sintetico.py`.

### 3.10 Mapa de disparidade com StereoSGBM

`cv2.StereoSGBM_create` (Semi-Global Block Matching) compara blocos de pixels entre as duas imagens em tons de cinza e devolve a disparidade por pixel. Parâmetros que o professor usa:

- `numDisparities`: faixa de busca, obrigatoriamente múltiplo de 16 (padrão 96). Deve ser maior que a maior disparidade esperada.
- `blockSize`: lado do bloco, ímpar e no mínimo 3 (padrão 5). Bloco maior é mais robusto a ruído, mas borra bordas.
- `P1` e `P2`: penalidades de suavidade, calculadas pelas fórmulas usuais `8 * canais * bloco**2` e `32 * canais * bloco**2`.
- `uniquenessRatio`, `speckleWindowSize`, `speckleRange`, `disp12MaxDiff`: filtros que rejeitam correspondências ambíguas, manchas pequenas e inconsistências esquerda-direita.
- `mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY`: variante mais rápida.

O resultado de `compute` vem em ponto fixo com 4 bits fracionários: divide-se por 16 para obter pixels. Valores menores ou iguais a zero são inválidos. Para visualizar, o professor normaliza pela faixa entre os percentis 2 e 98 (evita que outliers achatem o contraste) e aplica `cv2.applyColorMap(..., cv2.COLORMAP_TURBO)`, zerando os pixels inválidos. O mapa em float é salvo com `np.save` para o próximo script.

```python
# 08_mapa_disparidade_sgbm.py
sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=args.num_disparidades, blockSize=args.bloco,
    P1=8 * canais * args.bloco**2, P2=32 * canais * args.bloco**2,
    disp12MaxDiff=1, uniquenessRatio=8, speckleWindowSize=80, speckleRange=2,
    preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disparidade = sgbm.compute(esquerda, direita).astype(np.float32) / 16.0
valida = disparidade > 0
minimo, maximo = np.percentile(disparidade[valida], [2, 98])
visual = np.clip((disparidade - minimo) * 255 / max(maximo - minimo, 1e-6), 0, 255).astype(np.uint8)
colorida = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)
np.save(saida / "disparidade_float.npy", disparidade)
```

Onde aparece: `08_mapa_disparidade_sgbm.py`.

### 3.11 De disparidade para profundidade métrica

Com câmeras calibradas e retificadas vale a relação `Z = f * B / d`, onde `Z` é a profundidade em metros, `f` a distância focal em pixels, `B` a baseline em metros e `d` a disparidade em pixels. O script 09 aplica a fórmula só onde `d > 0.5`, deixa `NaN` no resto e produz três saídas: um PNG de 16 bits em milímetros (`uint16`, 0 significa inválido, limite de 65,535 m), um `.npy` em metros e uma visualização colorida em que mais perto é mais brilhante (`255 * (1 - Z / max_m)`).

O script `09_gerar_mapa_disparidade.py` é o inverso didático: parte de profundidades conhecidas (1, 2 e 4 m) e calcula `d = f * B / Z` (60, 30 e 15 px) para montar um mapa sintético e conferir o script anterior. Atenção: ele grava no mesmo arquivo `disparidade_float.npy` que o SGBM gera, então sobrescreve o resultado do script 08.

```python
# 09_disparidade_para_profundidade.py
disparidade = np.load(args.disparidade).astype(np.float32)
valida = disparidade > 0.5
profundidade = np.full(disparidade.shape, np.nan, dtype=np.float32)
profundidade[valida] = args.focal_px * args.baseline_m / disparidade[valida]

profundidade_mm = np.zeros(disparidade.shape, dtype=np.uint16)
metros_limitados = np.clip(profundidade[valida], 0, 65.535)
profundidade_mm[valida] = np.round(metros_limitados * 1000).astype(np.uint16)
```

Onde aparece: `09_disparidade_para_profundidade.py`, `09_gerar_mapa_disparidade.py`.

### Bloco C: faces

### 3.12 Detecção de faces com Haar Cascade

Detecção responde "onde há uma face"; não responde "quem é". O classificador Haar em cascata (Viola-Jones) desliza janelas de vários tamanhos pela imagem em tons de cinza e rejeita rapidamente as que não parecem rosto. O arquivo XML treinado vem com o OpenCV em `cv2.data.haarcascades + "haarcascade_frontalface_default.xml"`. `detector.empty()` indica que o XML não carregou.

Antes de detectar, o professor aplica `cv2.equalizeHist` para espalhar o histograma e reduzir o efeito de iluminação. `detectMultiScale` recebe:

- `scaleFactor` (padrão 1.1): fator de redução da imagem entre escalas. Menor é mais preciso e mais lento.
- `minNeighbors` (padrão 5): quantas detecções sobrepostas são necessárias para aceitar uma face. Maior reduz falsos positivos.
- `minSize`: menor face aceita, em pixels.

O retorno é uma lista de `(x, y, w, h)` que se desenha com `cv2.rectangle`.

```python
# 10_detectar_faces_haar_imagem.py
cinza = cv2.equalizeHist(cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY))
caminho = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(caminho)
if detector.empty():
    raise RuntimeError(f"Não foi possível carregar o cascade: {caminho}")
faces = detector.detectMultiScale(cinza, scaleFactor=args.escala,
                                  minNeighbors=args.vizinhos, minSize=(40, 40))
for indice, (x, y, w, h) in enumerate(faces, start=1):
    cv2.rectangle(saida, (x, y), (x + w, y + h), (40, 210, 40), 3)
```

Onde aparece: `10_detectar_faces_haar_imagem.py`, `11_detectar_faces_haar_webcam.py`, `12_coletar_faces_webcam.py`, `14_reconhecer_faces_lbph.py`.

### 3.13 Laço de vídeo em tempo real, FPS e teclas

O padrão de vídeo do professor: abrir `cv2.VideoCapture(indice)`, verificar `isOpened`, opcionalmente ajustar `camera.set(cv2.CAP_PROP_FRAME_WIDTH, largura)`, ler frames em `while True` com `ok, quadro = camera.read()`, processar, exibir e ler teclas com `cv2.waitKey(1) & 0xFF`. Tudo dentro de `try/finally` para garantir `camera.release()` e `cv2.destroyAllWindows()` mesmo em erro.

O FPS é medido com `time.perf_counter` e suavizado por média exponencial (`0.9 * anterior + 0.1 * atual`), que evita o número saltar a cada frame. A tecla `e` alterna a equalização para mostrar, ao vivo, o efeito dela na detecção.

```python
# 11_detectar_faces_haar_webcam.py
camera = cv2.VideoCapture(args.camera)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.largura)
try:
    while True:
        ok, quadro = camera.read()
        if not ok:
            break
        agora = time.perf_counter()
        fps = 1.0 / max(agora - instante_anterior, 1e-6)
        instante_anterior = agora
        fps_suave = fps if fps_suave == 0 else 0.9 * fps_suave + 0.1 * fps
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("e"):
            equalizar = not equalizar
finally:
    camera.release()
    cv2.destroyAllWindows()
```

Onde aparece: `11_detectar_faces_haar_webcam.py`, `12_coletar_faces_webcam.py`, `14_reconhecer_faces_lbph.py`, `27_pipeline_opencv_cnn_webcam.py`.

### 3.14 Coleta consentida de amostras faciais

Reconhecimento supervisionado precisa de um conjunto de treino. O script 12 monta a pasta `dados/faces_treino/<nome>/` e, a cada tecla `c`, salva a maior face do frame (escolhida com `max(faces, key=lambda f: f[2] * f[3])`) como recorte em tons de cinza, redimensionado para 160x160 com `cv2.INTER_AREA` (a melhor interpolação para reduzir) e equalizado. O nome é sanitizado por `normalizar_nome` (minúsculas, só letras, números, `_` e `-`), e o contador continua de onde parou se a pasta já tiver arquivos. O docstring pede variação de expressão e pose e enfatiza o consentimento da pessoa.

```python
# 12_coletar_faces_webcam.py
maior = max(faces, key=lambda f: f[2] * f[3]) if len(faces) else None
if tecla == ord("c") and maior is not None:
    x, y, w, h = maior
    face = cinza[y : y + h, x : x + w]
    face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
    face = cv2.equalizeHist(face)
    cv2.imwrite(str(pasta / f"{contador:03d}.png"), face)
    contador += 1
```

Onde aparece: `12_coletar_faces_webcam.py`.

### 3.15 Treinamento do reconhecedor LBPH

LBPH (Local Binary Patterns Histograms) descreve cada face por histogramas de padrões binários locais: para cada pixel compara-se o valor com os vizinhos em um raio, gerando um código binário; a imagem é dividida em uma grade e cada célula gera um histograma. `cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)` define raio 1, 8 vizinhos e grade 8x8. Só existe no `opencv-contrib-python`.

O script percorre as subpastas de `dados/faces_treino/`, atribui um índice inteiro a cada pessoa, carrega as imagens em cinza e redimensiona para 160x160. `reconhecedor.train(imagens, rotulos_int32)` treina, `write` salva em YAML e o mapa índice para nome vai para um JSON. O professor exige ao menos duas pessoas e dez amostras.

```python
# 13_treinar_reconhecedor_lbph.py
if not hasattr(cv2, "face"):
    raise RuntimeError("cv2.face não está disponível. Instale apenas opencv-contrib-python.")
for indice, pasta in enumerate(pessoas):
    mapa[indice] = pasta.name
    for caminho in sorted(pasta.glob("*.png")):
        face = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
        imagens.append(cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA))
        rotulos.append(indice)
reconhecedor = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
reconhecedor.train(imagens, np.asarray(rotulos, dtype=np.int32))
reconhecedor.write(args.modelo)
```

Onde aparece: `13_treinar_reconhecedor_lbph.py`.

### 3.16 Reconhecimento LBPH em tempo real e limiar de distância

`reconhecedor.predict(face)` devolve `(rotulo, distancia)`. A "confiança" do LBPH é uma distância entre histogramas: quanto menor, mais parecida. Por isso o script aceita a identidade apenas quando `distancia <= limiar` (padrão 65) e escreve "desconhecido" caso contrário. O pré-processamento do frame deve ser idêntico ao da coleta: cinza, recorte da face, `resize` para 160x160 e `equalizeHist`. O mapa JSON tem chaves em texto, daí o `mapa.get(str(rotulo))`.

```python
# 14_reconhecer_faces_lbph.py
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read(args.modelo)
for x, y, w, h in faces:
    face = cv2.equalizeHist(cv2.resize(cinza[y : y + h, x : x + w], (160, 160)))
    rotulo, distancia = reconhecedor.predict(face)
    nome = mapa.get(str(rotulo), "desconhecido")
    aceito = distancia <= args.limiar
    cor = (40, 210, 40) if aceito else (40, 40, 230)
```

Onde aparece: `14_reconhecer_faces_lbph.py`.

### 3.17 Reconhecimento por embeddings com face_recognition

A abordagem moderna substitui histogramas por um vetor de 128 números (embedding) produzido por uma rede neural treinada para que faces da mesma pessoa fiquem próximas no espaço vetorial. O pacote `face_recognition` (sobre `dlib`) oferece:

- `load_image_file`: lê a imagem em RGB (atenção: o OpenCV usa BGR, por isso o professor converte a consulta com `COLOR_BGR2RGB`).
- `face_locations(rgb, model="hog")`: detecta faces e devolve `(topo, direita, base, esquerda)`, ordem diferente da do OpenCV.
- `face_encodings(rgb, localizacoes)`: calcula o embedding de cada face.
- `face_distance(vetores_conhecidos, vetor)`: distância euclidiana para cada referência.

O nome do arquivo de referência vira o rótulo (`caminho.stem`). Referências com zero ou mais de uma face são ignoradas. A identidade é o menor `argmin` das distâncias, aceita se `distancia <= tolerancia` (padrão 0.55).

```python
# 15_reconhecimento_embeddings.py
rgb = face_recognition.load_image_file(str(caminho))
codigos = face_recognition.face_encodings(rgb)
if len(codigos) != 1:
    continue
nomes.append(caminho.stem); vetores.append(codigos[0])

localizacoes = face_recognition.face_locations(rgb, model="hog")
consultas = face_recognition.face_encodings(rgb, localizacoes)
for (topo, direita, base, esquerda), vetor in zip(localizacoes, consultas):
    distancias = face_recognition.face_distance(vetores, vetor)
    melhor = int(np.argmin(distancias))
    aceito = float(distancias[melhor]) <= args.tolerancia
```

Onde aparece: `15_reconhecimento_embeddings.py`.

### Bloco D: features locais e matching

### 3.18 Keypoints e descritores SIFT

Um keypoint é um ponto da imagem fácil de reencontrar (canto, mancha, borda com textura), com posição, escala e orientação. Um descritor é um vetor que resume a vizinhança do keypoint e permite compará-lo com keypoints de outra imagem. SIFT (Scale-Invariant Feature Transform) é invariante a escala e rotação e produz descritores de 128 valores `float32`.

`cv2.SIFT_create(nfeatures=N)` limita o número de pontos. `detectAndCompute(cinza, None)` devolve a lista de keypoints e a matriz de descritores `(N, 128)`. `cv2.drawKeypoints` com a flag `DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` desenha círculos cujo raio é a escala e um raio interno que mostra a orientação.

```python
# 16_pontos_chave_sift.py
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create(nfeatures=args.max_pontos)
pontos, descritores = sift.detectAndCompute(cinza, None)
desenhada = cv2.drawKeypoints(imagem, pontos, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(f"Keypoints: {len(pontos)}; descritores: {descritores.shape}")
```

Onde aparece: `16_pontos_chave_sift.py`, `18_matching_sift_ratio_test.py`, `19_homografia_ransac.py`, `20_surf_opcional.py`.

### 3.19 ORB, distância de Hamming e cross-check

ORB (Oriented FAST and Rotated BRIEF) é a alternativa livre e rápida ao SIFT: detecta cantos com FAST e descreve com BRIEF, gerando descritores binários de 32 bytes. Descritores binários se comparam pela distância de Hamming (número de bits diferentes), não pela euclidiana. `cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)` constrói uma pirâmide de 8 níveis com fator 1.2 entre eles para obter invariância a escala.

`cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)` faz força bruta: para cada descritor de A acha o mais próximo em B. Com `crossCheck` o par só é aceito se também for o mais próximo no sentido inverso, o que elimina muitos falsos matches. `match` devolve objetos `DMatch` com `distance`, `queryIdx` e `trainIdx`; ordená-los por distância e desenhar os 40 melhores com `cv2.drawMatches` mostra a qualidade da correspondência.

```python
# 17_matching_orb.py
orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
kp_a, des_a = orb.detectAndCompute(a, None)
kp_b, des_b = orb.detectAndCompute(b, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
correspondencias = sorted(matcher.match(des_a, des_b), key=lambda m: m.distance)
melhores = correspondencias[: args.mostrar]
desenho = cv2.drawMatches(a, kp_a, b, kp_b, melhores, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

Onde aparece: `17_matching_orb.py`, `21_recuperacao_imagens_orb.py`.

### 3.20 Matching k-NN e ratio test de Lowe

Em vez de aceitar só o vizinho mais próximo, `knnMatch(des_a, des_b, k=2)` devolve os dois melhores candidatos de B para cada descritor de A. O ratio test de Lowe aceita o par apenas se o melhor for claramente melhor que o segundo: `m.distance < ratio * n.distance`. Com ratio 0.75, um match ambíguo (dois candidatos parecidos) é rejeitado. Ratio menor é mais seletivo; maior aceita mais ambiguidade. Para SIFT usa-se `cv2.NORM_L2` (euclidiana), sem cross-check. `cv2.drawMatchesKnn` espera uma lista de listas, por isso `bons = [[m] ...]`.

```python
# 18_matching_sift_ratio_test.py
sift = cv2.SIFT_create(nfeatures=1000)
kp_a, des_a = sift.detectAndCompute(a, None)
kp_b, des_b = sift.detectAndCompute(b, None)
matcher = cv2.BFMatcher(cv2.NORM_L2)
pares = matcher.knnMatch(des_a, des_b, k=2)
bons = [[m] for m, n in pares if m.distance < args.ratio * n.distance]
desenho = cv2.drawMatchesKnn(a, kp_a, b, kp_b, bons[:80], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

Onde aparece: `18_matching_sift_ratio_test.py`, `19_homografia_ransac.py`, `21_recuperacao_imagens_orb.py`.

### 3.21 Homografia com RANSAC e localização de objeto

Uma homografia é uma matriz 3x3 `H` que mapeia pontos de um plano para outro sob perspectiva (rotação, escala, translação e inclinação). Com pelo menos 4 pares de pontos correspondentes é possível estimá-la. Como os matches contêm erros, `cv2.findHomography(origem, destino, cv2.RANSAC, reprojecao)` usa RANSAC: sorteia subconjuntos de 4 pares, estima `H`, conta quantos pares ficam a menos de `reprojecao` pixels (padrão 4.0) da posição prevista (inliers) e fica com a melhor. Devolve `H` e uma máscara de inliers.

As coordenadas vêm dos keypoints: `kp_o[m.queryIdx].pt` no objeto e `kp_c[m.trainIdx].pt` na cena, no formato `(N, 1, 2)` `float32`. Para localizar o objeto na cena, os quatro cantos do objeto são projetados por `cv2.perspectiveTransform(cantos, H)` e ligados com `cv2.polylines`. `drawMatches` com `matchesMask` desenha só os inliers.

```python
# 19_homografia_ransac.py
origem = np.float32([kp_o[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
destino = np.float32([kp_c[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
H, mascara = cv2.findHomography(origem, destino, cv2.RANSAC, args.reprojecao)
h, w = objeto.shape
cantos = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
projetados = cv2.perspectiveTransform(cantos, H)
cena_marcada = cv2.polylines(cena_bgr, [np.int32(projetados)], True, (40, 210, 40), 4, cv2.LINE_AA)
inliers = mascara.ravel().astype(bool)
```

Onde aparece: `19_homografia_ransac.py`.

### 3.22 SURF e tratamento de indisponibilidade

SURF (Speeded-Up Robust Features) é um detector mais rápido que o SIFT, baseado no determinante da matriz Hessiana. `cv2.xfeatures2d.SURF_create(hessianThreshold=400)` só existe em builds do OpenCV compilados com os módulos nonfree, o que não acontece nos wheels do pip. O script demonstra a prática de testar `hasattr(cv2, "xfeatures2d")`, capturar `AttributeError` e `cv2.error`, e cair para `cv2.SIFT_create` com uma mensagem clara. O `hessianThreshold` controla quantos pontos são aceitos: maior, menos pontos e mais fortes.

```python
# 20_surf_opcional.py
try:
    if not hasattr(cv2, "xfeatures2d"):
        raise AttributeError("cv2.xfeatures2d ausente")
    detector = cv2.xfeatures2d.SURF_create(hessianThreshold=args.hessian)
    pontos, descritores = detector.detectAndCompute(cinza, None)
except (AttributeError, cv2.error) as erro:
    print("SURF não está disponível neste build:", erro)
    algoritmo = "SIFT (fallback)"
    detector = cv2.SIFT_create(nfeatures=600)
    pontos, descritores = detector.detectAndCompute(cinza, None)
```

Onde aparece: `20_surf_opcional.py`.

### 3.23 Recuperação de imagens por similaridade local

Recuperação de imagens (image retrieval) é procurar, em um banco de imagens, as mais parecidas com uma consulta. O script 21 usa ORB + `BFMatcher(NORM_HAMMING)` + k-NN com ratio test (0.78) contra cada imagem do banco e define o score como a fração de descritores da consulta aprovados no ratio test (`len(bons) / len(des_q)`). O ranking é ordenado por score e, em empate, por número de bons matches. O próprio arquivo de consulta é excluído do banco comparando caminhos com `resolve()`. Rodando com o padrão, o banco é `dados/gerados/`, então a melhor resposta tende a ser `objeto_transformado.png` ou `cena.png`.

```python
# 21_recuperacao_imagens_orb.py
for caminho in sorted(Path(args.banco).iterdir()):
    if caminho.suffix.lower() not in EXTENSOES or caminho.resolve() == consulta_resolvida:
        continue
    kp_c, des_c = descritores_orb(candidata, orb)
    pares = matcher.knnMatch(des_q, des_c, k=2)
    bons = [m for m, n in pares if m.distance < args.ratio * n.distance]
    score = len(bons) / max(len(des_q), 1)
    resultados.append((score, len(bons), caminho.name, candidata, len(kp_c)))
resultados.sort(key=lambda item: (item[0], item[1]), reverse=True)
```

Onde aparece: `21_recuperacao_imagens_orb.py`.

### Bloco E: aprendizado profundo com Keras

### 3.24 Fashion-MNIST, normalização e MLP

Fashion-MNIST é um conjunto de 70 mil imagens 28x28 em tons de cinza de 10 classes de roupas (camiseta, calça, pulôver, vestido, casaco, sandália, camisa, tênis, bolsa, bota). `keras.datasets.fashion_mnist.load_data()` baixa e devolve `(x_treino, y_treino), (x_teste, y_teste)` com 60 mil e 10 mil amostras. Os pixels `uint8` são convertidos para `float32` e divididos por 255 para ficar em `[0, 1]`, o que estabiliza o treino.

Uma MLP (Multi-Layer Perceptron) recebe a imagem achatada em um vetor de 784 valores (`layers.Flatten`) e perde a topologia 2D: dois pixels vizinhos são tratados como quaisquer outros dois. A rede do professor tem `Dense(256, relu)`, `Dropout(0.25)`, `Dense(128, relu, name="features")` e `Dense(10, softmax)`. O nome `features` na penúltima camada é intencional: o script 25 extrai ativações dela.

`compile` define otimizador (Adam com taxa 1e-3), perda (`sparse_categorical_crossentropy`, porque os rótulos são inteiros e não one-hot) e métrica. `fit` com `validation_split=0.1` separa 10% do treino para validação. `evaluate` mede no teste. `keras.utils.set_random_seed` torna o experimento reprodutível e `modelo.save("*.keras")` grava o modelo completo.

```python
# 22_mlp_fashion_mnist.py
(x_treino, y_treino), (x_teste, y_teste) = keras.datasets.fashion_mnist.load_data()
x_treino = x_treino.astype("float32") / 255.0
modelo = keras.Sequential([
    keras.Input(shape=(28, 28), name="imagem"),
    layers.Flatten(name="vetor_784"),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(128, activation="relu", name="features"),
    layers.Dense(10, activation="softmax", name="classes"),
], name="mlp_fashion_mnist")
modelo.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss="sparse_categorical_crossentropy", metrics=["accuracy"])
historico = modelo.fit(x_treino, y_treino, validation_split=0.1, epochs=args.epocas,
                       batch_size=args.batch, verbose=2)
```

Onde aparece: `22_mlp_fashion_mnist.py`.

### 3.25 Curvas de treino e matriz de confusão

O objeto retornado por `fit` guarda em `historico.history` as listas `loss`, `val_loss`, `accuracy` e `val_accuracy` por época. Plotar treino e validação lado a lado revela overfitting: a perda de treino cai enquanto a de validação sobe. O professor usa `matplotlib` com `fig.savefig` e `plt.close(fig)` para gravar sem abrir janela.

A matriz de confusão (`sklearn.metrics.confusion_matrix`) cruza classe real com classe prevista e mostra onde o modelo se confunde (camisa com camiseta, por exemplo). `modelo.predict(...).argmax(axis=1)` transforma probabilidades em índice de classe. `ConfusionMatrixDisplay` desenha a matriz com os nomes das classes.

```python
# 22_mlp_fashion_mnist.py
eixos[0].plot(historico.history["loss"], label="treino")
eixos[0].plot(historico.history["val_loss"], label="validação")
previsoes = modelo.predict(x_teste[:2000], verbose=0).argmax(axis=1)
matriz = confusion_matrix(y_teste[:2000], previsoes)
ConfusionMatrixDisplay(matriz, display_labels=CLASSES).plot(
    ax=eixo, cmap="Blues", xticks_rotation=45, colorbar=False)
```

Onde aparece: `22_mlp_fashion_mnist.py`, `23_cnn_fashion_mnist.py` (curva de acurácia).

### 3.26 CNN compacta e API funcional do Keras

Uma CNN (rede neural convolucional) preserva a vizinhança espacial: `layers.Conv2D(32, 3, padding="same", activation="relu")` aplica 32 filtros 3x3 que deslizam pela imagem, compartilhando pesos em todas as posições. `padding="same"` mantém o tamanho. `layers.MaxPooling2D()` reduz cada dimensão pela metade guardando o máximo de cada janela 2x2, o que dá invariância a pequenos deslocamentos e reduz custo. Depois de dois blocos, `Flatten` e camadas densas fecham a classificação.

A CNN exige o eixo de canal: a entrada passa de `(28, 28)` para `(28, 28, 1)` com `[..., None]`. O script usa a API funcional (`keras.Input` seguido de chamadas encadeadas e `keras.Model(entradas, saidas)`), mais flexível que `Sequential`. As camadas `mapas_64` e `features` são nomeadas para inspeção posterior.

```python
# 23_cnn_fashion_mnist.py
def construir_modelo() -> keras.Model:
    entradas = keras.Input(shape=(28, 28, 1), name="imagem")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(entradas)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="mapas_64")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(96, activation="relu", name="features")(x)
    x = layers.Dropout(0.3)(x)
    saidas = layers.Dense(10, activation="softmax", name="classes")(x)
    return keras.Model(entradas, saidas, name="cnn_fashion_mnist")

x_treino = (x_treino.astype("float32") / 255.0)[..., None]
```

Onde aparece: `23_cnn_fashion_mnist.py`.

### 3.27 Aumento de dados, BatchNormalization e GlobalAveragePooling

Aumento de dados (data augmentation) gera variações aleatórias das imagens durante o treino para a rede não decorar o conjunto. No Keras isso é feito por camadas dentro do próprio modelo: `RandomRotation(0.06)` (fração de 2π), `RandomTranslation(0.08, 0.08)` e `RandomZoom(0.08)`. Elas só agem em treino; em inferência são identidade.

`layers.BatchNormalization()` normaliza as ativações de cada lote, acelera a convergência e ajuda a regularizar. `layers.GlobalAveragePooling2D()` substitui o `Flatten`: tira a média de cada mapa de ativação, reduzindo drasticamente o número de parâmetros da camada densa seguinte e tornando o modelo menos sensível à posição.

```python
# 24_cnn_augmentation_callbacks.py
aumento = keras.Sequential([
    layers.RandomRotation(0.06),
    layers.RandomTranslation(0.08, 0.08),
    layers.RandomZoom(0.08),
], name="aumento_de_dados")
x = aumento(entradas)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
```

Onde aparece: `24_cnn_augmentation_callbacks.py`, `26_transfer_learning_mobilenetv2.py` (RandomFlip, RandomRotation, RandomZoom).

### 3.28 Callbacks de treinamento

Callbacks são objetos chamados pelo Keras a cada época para monitorar e intervir no treino. O professor usa quatro:

- `ModelCheckpoint(caminho, monitor="val_accuracy", mode="max", save_best_only=True)`: salva o modelo só quando a acurácia de validação melhora.
- `EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)`: para o treino se a perda de validação não melhorar por 4 épocas e restaura os melhores pesos.
- `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)`: reduz a taxa de aprendizado pela metade quando a validação estagna por 2 épocas.
- `CSVLogger(caminho)`: grava as métricas de cada época em CSV.

```python
# 24_cnn_augmentation_callbacks.py
callbacks = [
    keras.callbacks.ModelCheckpoint(pasta / "cnn_fashion_melhor.keras",
                                    monitor="val_accuracy", mode="max", save_best_only=True),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    keras.callbacks.CSVLogger(pasta / "treinamento_cnn.csv"),
]
modelo.fit(x_treino, y_treino, validation_split=0.1, epochs=args.epocas,
           batch_size=args.batch, callbacks=callbacks, verbose=2)
```

Onde aparece: `24_cnn_augmentation_callbacks.py`, `26_transfer_learning_mobilenetv2.py` (EarlyStopping).

### 3.29 Extração de features de camada intermediária e PCA

Uma rede treinada pode servir como extrator de características: basta criar um submodelo cuja saída é uma camada interna. `keras.models.load_model` carrega o `.keras`, `modelo.get_layer("features")` localiza a camada pelo nome e `keras.Model(inputs=modelo.inputs, outputs=camada.output)` monta o extrator. `predict` devolve um vetor por imagem, salvo em `.npz` comprimido.

Para enxergar se as classes se separam, o vetor de features é projetado em 2 dimensões por PCA (`sklearn.decomposition.PCA(n_components=2)`), que encontra as direções de maior variância. O gráfico de dispersão colorido por classe mostra agrupamentos; `explained_variance_ratio_` diz quanta variância os dois componentes retêm.

```python
# 25_extrair_features_cnn.py
modelo = keras.models.load_model(args.modelo)
camada = modelo.get_layer(args.camada)
extrator = keras.Model(inputs=modelo.inputs, outputs=camada.output)
features = extrator.predict(x, batch_size=256, verbose=0).reshape(n, -1)
pca = PCA(n_components=2, random_state=42)
projecao = pca.fit_transform(features)
eixo.scatter(projecao[:, 0], projecao[:, 1], c=y_teste[:n], cmap="tab10", s=9, alpha=0.65)
```

Onde aparece: `25_extrair_features_cnn.py`.

### 3.30 Transfer learning com MobileNetV2

Transfer learning reaproveita uma rede treinada em um conjunto enorme (ImageNet) como base para um problema pequeno. `keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights="imagenet")` carrega a rede sem a camada de classificação original. `base.trainable = False` congela os pesos: só a "cabeça" nova (`GlobalAveragePooling2D`, `Dropout`, `Dense(n_classes, softmax)`) é treinada. `base(x, training=False)` mantém o BatchNormalization interno em modo de inferência.

O dataset vem de pastas com `keras.utils.image_dataset_from_directory(pasta, image_size=(160, 160), batch_size=32, label_mode="int")`: cada subpasta é uma classe e `class_names` lista os nomes. `keras.applications.mobilenet_v2.preprocess_input` escala os pixels para `[-1, 1]` como a rede espera. `prefetch(tf.data.AUTOTUNE)` sobrepõe carregamento e treino. As classes são salvas em texto para uso posterior.

```python
# 26_transfer_learning_mobilenetv2.py
treino = keras.utils.image_dataset_from_directory(
    treino_dir, image_size=TAMANHO, batch_size=args.batch, label_mode="int", seed=42)
base = keras.applications.MobileNetV2(input_shape=(*TAMANHO, 3), include_top=False, weights="imagenet")
base.trainable = False
entradas = keras.Input(shape=(*TAMANHO, 3))
x = aumento(entradas)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D(name="features")(x)
x = layers.Dropout(0.25)(x)
saidas = layers.Dense(len(classes), activation="softmax")(x)
```

Onde aparece: `26_transfer_learning_mobilenetv2.py`.

### 3.31 Pipeline integrado OpenCV + CNN na webcam

O último script fecha o ciclo: OpenCV cuida da captura e do pré-processamento, Keras da classificação. Uma ROI quadrada central com 55% do menor lado do frame é recortada; dentro dela a função `preparar` converte para cinza, suaviza, binariza com Otsu (invertido por padrão, porque o Fashion-MNIST tem objeto claro sobre fundo preto), redimensiona para 28x28 com `INTER_AREA` e monta o tensor `(1, 28, 28, 1)` normalizado. `modelo.predict(tensor)[0]` devolve as 10 probabilidades; `argmax` dá a classe e o valor máximo é a certeza. A miniatura 28x28 ampliada com `INTER_NEAREST` é colada no canto do frame para mostrar exatamente o que a rede vê. A tecla `i` inverte a binarização.

```python
# 27_pipeline_opencv_cnn_webcam.py
def preparar(roi, inverter):
    cinza = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    tipo = cv2.THRESH_BINARY_INV if inverter else cv2.THRESH_BINARY
    _, binaria = cv2.threshold(cinza, 0, 255, tipo + cv2.THRESH_OTSU)
    imagem_28 = cv2.resize(binaria, (28, 28), interpolation=cv2.INTER_AREA)
    tensor = (imagem_28.astype("float32") / 255.0)[None, ..., None]
    return tensor, imagem_28

lado = int(min(h, w) * 0.55)
roi = quadro[y1:y2, x1:x2]
probabilidades = modelo.predict(tensor, verbose=0)[0]
indice = int(np.argmax(probabilidades))
quadro[10:150, 10:150] = cv2.cvtColor(cv2.resize(imagem_28, (140, 140),
                                      interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
```

Onde aparece: `27_pipeline_opencv_cnn_webcam.py`.

## 4. Roteiro dos scripts

Caminhos relativos ao diretório de trabalho. `dg/` abrevia `dados/gerados/`.

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `00_verificar_ambiente.py` | Mostra versões, recursos do OpenCV e testa a câmera 0 | nenhuma | só terminal | nada |
| `01_gerar_cena_sintetica.py` | Desenha a cena com 4 objetos, ruído, e recorta o objeto de consulta e uma versão rotacionada | nenhuma | `dg/cena.png`, `dg/objeto_consulta.png`, `dg/objeto_transformado.png` | nada |
| `02_ler_e_recortar_roi.py` | Recorta uma ROI por coordenadas e marca na cena | `--imagem`, `--x --y --largura --altura` | `dg/roi.png`, `dg/cena_com_roi.png` | 01 |
| `03_segmentar_cor_hsv.py` | Máscara HSV do retângulo azul | `--imagem`, `--h-min --h-max --s-min --v-min` | `dg/mascara_hsv.png`, `dg/segmentacao_hsv.png` | 01 |
| `04_limiarizacao_otsu.py` | Compara limiar global, Otsu e adaptativo | `--imagem` | `dg/comparacao_limiares.png` | 01 |
| `05_morfologia_contornos_roi.py` | Limpa máscara HSV, acha contornos, salva ROIs por bounding box | `--imagem`, `--area-minima` | `dg/mascara_morfologica.png`, `dg/contornos_rois.png`, `dg/rois/roi_NN.png` | 01 |
| `06_grabcut_roi.py` | Segmenta primeiro plano a partir de um retângulo | `--imagem`, `--x --y --largura --altura --iteracoes` | `dg/mascara_grabcut.png`, `dg/resultado_grabcut.png` | 01 |
| `07_criar_par_estereo_sintetico.py` | Gera par esquerda/direita com 3 objetos de disparidade 48, 28 e 12 px | nenhuma | `dg/estereo_esquerda.png`, `dg/estereo_direita.png` | nada |
| `08_mapa_disparidade_sgbm.py` | Calcula disparidade com StereoSGBM e colore | `--esquerda --direita --num-disparidades --bloco` | `dg/disparidade_float.npy`, `dg/disparidade_colorida.png` | 07 |
| `09_gerar_mapa_disparidade.py` | Gera mapa de disparidade sintético com 3 blocos (1, 2 e 4 m) | nenhuma | `dg/disparidade_float.npy` (sobrescreve o do 08), `dg/disparidade_teste.png` | nada |
| `09_disparidade_para_profundidade.py` | Converte disparidade em profundidade Z = fB/d | `--disparidade --focal-px --baseline-m --max-m` | `dg/profundidade_mm.png` (16 bits), `dg/profundidade_colorida.png`, `dg/profundidade_m.npy` | 08 ou 09_gerar |
| `10_detectar_faces_haar_imagem.py` | Detecta faces em uma foto com Haar | `--imagem` (obrigatório), `--escala --vizinhos` | `dg/faces_haar.png` | foto própria |
| `11_detectar_faces_haar_webcam.py` | Haar em tempo real com FPS e tecla de equalização | `--camera --largura` | só tela | webcam |
| `12_coletar_faces_webcam.py` | Salva recortes 160x160 da maior face ao apertar `c` | `--nome` (obrigatório), `--quantidade --camera` | `dados/faces_treino/<nome>/NNN.png` | webcam |
| `13_treinar_reconhecedor_lbph.py` | Treina LBPH com as pastas coletadas | `--dados --modelo --mapa` | `modelos/lbph.yml`, `modelos/lbph_rotulos.json` | 12 (2+ pessoas, 10+ amostras), contrib |
| `14_reconhecer_faces_lbph.py` | Reconhece na webcam com limiar de distância | `--modelo --mapa --limiar --camera` | só tela | 13, webcam, contrib |
| `15_reconhecimento_embeddings.py` | Reconhece por embeddings do face_recognition | `--conhecidas --consulta` (obrigatórios), `--tolerancia` | `dg/reconhecimento_embeddings.png` | fotos próprias, `face_recognition` |
| `16_pontos_chave_sift.py` | Detecta e desenha keypoints SIFT | `--imagem --max-pontos` | `dg/keypoints_sift.png` | 01 |
| `17_matching_orb.py` | Matching ORB com Hamming e cross-check | `--imagem-a --imagem-b --mostrar` | `dg/matching_orb.png` | 01 |
| `18_matching_sift_ratio_test.py` | Matching SIFT k-NN com ratio test | `--imagem-a --imagem-b --ratio` | `dg/matching_sift_ratio.png` | 01 |
| `19_homografia_ransac.py` | Localiza o objeto na cena por homografia + RANSAC | `--objeto --cena --ratio --reprojecao` | `dg/homografia_ransac.png` | 01 |
| `20_surf_opcional.py` | Tenta SURF, cai para SIFT | `--imagem --hessian` | `dg/keypoints_surf_ou_fallback.png` | 01 |
| `21_recuperacao_imagens_orb.py` | Ranqueia imagens do banco por similaridade ORB | `--consulta --banco --ratio --top-k` | `dg/melhor_recuperacao_orb.png` | 01 (banco = `dg/`) |
| `22_mlp_fashion_mnist.py` | Treina MLP no Fashion-MNIST, curvas e matriz de confusão | `--epocas --batch --semente` | `modelos/mlp_fashion.keras`, `modelos/historico_mlp.png`, `modelos/matriz_confusao_mlp.png` | download do Keras |
| `23_cnn_fashion_mnist.py` | Treina CNN compacta | `--epocas --batch` | `modelos/cnn_fashion.keras`, `modelos/historico_cnn.png` | download do Keras |
| `24_cnn_augmentation_callbacks.py` | CNN com aumento de dados, BatchNorm e 4 callbacks | `--epocas --batch` | `modelos/cnn_fashion_melhor.keras`, `modelos/treinamento_cnn.csv` | download do Keras |
| `25_extrair_features_cnn.py` | Extrai a camada `features` e projeta com PCA | `--modelo --camada --amostras` | `modelos/features_fashion.npz`, `modelos/features_pca.png` | 23 (ou 24 com `--modelo`) |
| `26_transfer_learning_mobilenetv2.py` | Transfer learning em dataset por pastas | `--dados --epocas --batch` | `modelos/mobilenetv2_custom.keras`, `modelos/mobilenetv2_classes.txt` | dataset próprio, download dos pesos ImageNet |
| `27_pipeline_opencv_cnn_webcam.py` | Classifica a ROI central da webcam com a CNN | `--modelo --camera` | só tela | 23, webcam |

## 5. Passo a passo para executar

Todos os comandos abaixo assumem que você está dentro de `aula_03_ROI_SURF_etc/exemplos_em_python/`.

```
cd C:\visao_computacional_PVA\aula_03_ROI_SURF_etc\exemplos_em_python
```

**Passo 0: ambiente.** Confira versões e recursos. Espere ver `SIFT: True`, `ORB: True`, `Módulo cv2.face: True` (se instalou o contrib) e `SURF: False` na maioria dos casos.

```
python 00_verificar_ambiente.py
```

**Passo 1: gerar a cena.** Cria `dados/gerados/` com três PNGs. O terminal imprime a forma de cada um.

```
python 01_gerar_cena_sintetica.py
```

**Passo 2: ROI, cor, limiar, morfologia, GrabCut.** Cada script abre janelas; pressione qualquer tecla para fechar. Os resultados vão para `dados/gerados/`.

```
python 02_ler_e_recortar_roi.py
python 02_ler_e_recortar_roi.py --x 340 --y 60 --largura 180 --altura 200
python 03_segmentar_cor_hsv.py
python 03_segmentar_cor_hsv.py --h-min 40 --h-max 80
python 04_limiarizacao_otsu.py
python 05_morfologia_contornos_roi.py --area-minima 500
python 06_grabcut_roi.py
```

No 03, a janela mostra original, máscara e resultado lado a lado, e o terminal informa a porcentagem de pixels selecionados. Com `--h-min 40 --h-max 80` o círculo verde é selecionado no lugar do retângulo azul. No 04, o mosaico tem quatro painéis (cinza, global, Otsu, adaptativa) e o terminal imprime o limiar de Otsu. No 05, as ROIs aceitas aparecem com a área anotada e são salvas em `dados/gerados/rois/`.

**Passo 3: estéreo.** Gere o par, calcule a disparidade e converta em profundidade.

```
python 07_criar_par_estereo_sintetico.py
python 08_mapa_disparidade_sgbm.py
python 09_disparidade_para_profundidade.py --focal-px 500 --baseline-m 0.12
```

No 08 o terminal mostra a porcentagem de pixels válidos e a faixa de disparidade. O objeto PERTO deve aparecer com cor mais quente no mapa TURBO. Para testar o 09 com valores exatos, rode antes o gerador sintético (ele sobrescreve o `.npy` do SGBM):

```
python 09_gerar_mapa_disparidade.py
python 09_disparidade_para_profundidade.py
```

O terminal deve informar profundidades próximas de 1, 2 e 4 m nos percentis.

**Passo 4: faces.** O 10 exige uma foto com rostos. O 11 usa a webcam (`q` sai, `e` alterna equalização). Para o LBPH, colete ao menos duas pessoas (`c` captura, `q` encerra), treine e reconheça.

```
python 10_detectar_faces_haar_imagem.py --imagem C:\caminho\para\foto.jpg
python 11_detectar_faces_haar_webcam.py --camera 0
python 12_coletar_faces_webcam.py --nome pessoa_a --quantidade 30
python 12_coletar_faces_webcam.py --nome pessoa_b --quantidade 30
python 13_treinar_reconhecedor_lbph.py
python 14_reconhecer_faces_lbph.py --limiar 65
```

No 14 o nome aparece em verde quando a distância é menor ou igual ao limiar e "desconhecido" em vermelho caso contrário. Para o 15, monte uma pasta com uma foto por pessoa (o nome do arquivo é o rótulo) e uma foto de consulta:

```
python 15_reconhecimento_embeddings.py --conhecidas dados/faces_referencia --consulta dados/consulta_faces.jpg --tolerancia 0.55
```

**Passo 5: features locais.** Todos usam as imagens do passo 1.

```
python 16_pontos_chave_sift.py
python 17_matching_orb.py --mostrar 40
python 18_matching_sift_ratio_test.py --ratio 0.75
python 19_homografia_ransac.py --reprojecao 4.0
python 20_surf_opcional.py
python 21_recuperacao_imagens_orb.py --top-k 5
```

No 16 espere círculos de tamanhos diferentes com um raio de orientação. No 17 e 18 aparecem linhas ligando pontos correspondentes entre o objeto e a versão rotacionada. No 19 um quadrilátero verde deve envolver o retângulo azul na cena e o terminal imprime a matriz `H` e o número de inliers. No 20 o texto no canto informa se rodou SURF ou o fallback SIFT. No 21 o terminal lista o ranking com score.

**Passo 6: aprendizado profundo.** Os treinos baixam o Fashion-MNIST na primeira vez e demoram alguns minutos em CPU. Rode em ordem.

```
python 22_mlp_fashion_mnist.py --epocas 8
python 23_cnn_fashion_mnist.py --epocas 8
python 24_cnn_augmentation_callbacks.py --epocas 20
python 25_extrair_features_cnn.py --modelo modelos/cnn_fashion.keras
python 25_extrair_features_cnn.py --modelo modelos/cnn_fashion_melhor.keras
```

Compare a acurácia de teste impressa pelo 22 (MLP) e pelo 23 (CNN): a CNN deve ficar acima. O 24 pode parar antes das 20 épocas por causa do EarlyStopping. O 25 gera `modelos/features_pca.png` com um agrupamento por classe.

Transfer learning só com dataset próprio em `dados/dataset_custom/treino/<classe>/` e `validacao/<classe>/`:

```
python 26_transfer_learning_mobilenetv2.py --dados dados/dataset_custom --epocas 10
```

**Passo 7: pipeline na webcam.** Mostre um desenho ou uma peça de roupa dentro do quadrado verde. `i` inverte a binarização, `q` sai. A miniatura no canto superior esquerdo mostra a imagem 28x28 que a rede recebe.

```
python 27_pipeline_opencv_cnn_webcam.py --modelo modelos/cnn_fashion.keras
```

## 6. Exercícios e itens da aula

Os scripts desta aula não referenciam itens ou exercícios numerados. Exercícios propostos para praticar:

1. **Segmentar cada objeto da cena.** Use `03_segmentar_cor_hsv.py` com quatro intervalos de H diferentes para isolar o retângulo azul, o círculo verde, o triângulo vermelho e o retângulo amarelo. Anote os intervalos que funcionaram e explique por que o vermelho é o mais difícil (dica: H próximo de 0 e de 179).
2. **Contar objetos sem cor.** Adapte a ideia do `05_morfologia_contornos_roi.py` trocando `inRange` pela binarização de Otsu do script 04 e verifique se as quatro formas aparecem como contornos separados. Varie `--area-minima` e o tamanho do kernel morfológico.
3. **Validar o SGBM contra a verdade.** Depois de `07` e `08`, carregue `dados/gerados/disparidade_float.npy` e calcule a mediana da disparidade dentro de cada objeto (as posições estão no script 07). Compare com 48, 28 e 12 px. Repita com `--bloco 3` e `--bloco 9` e observe o efeito nas bordas.
4. **Curva de limiar do LBPH.** Com o modelo do 13, rode o 14 com `--limiar 40`, `65` e `90`. Registre quando uma pessoa não cadastrada passa a ser aceita e quando uma cadastrada passa a ser rejeitada.
5. **Robustez do matching.** Gere versões do `objeto_consulta.png` com rotação de 45 e 90 graus e escala 0.5 usando `getRotationMatrix2D` e `warpAffine`, e compare quantos inliers o `19_homografia_ransac.py` encontra com SIFT em cada caso. Troque o SIFT por ORB no mesmo script e repita.
6. **MLP contra CNN com menos dados.** Nos scripts 22 e 23, treine com apenas 5 mil imagens (`x_treino[:5000]`) e compare a queda de acurácia. Use o 25 nos dois modelos para ver se as features da CNN continuam mais separáveis.

## 7. Armadilhas e dicas

- **Pasta de trabalho.** Os caminhos `dados/` e `modelos/` são relativos. Rode sempre da mesma pasta, ou o script 02 não acha a cena do 01, o 25 não acha o modelo do 23 e assim por diante.
- **Dois scripts 09.** `09_gerar_mapa_disparidade.py` e `08_mapa_disparidade_sgbm.py` gravam no mesmo `dados/gerados/disparidade_float.npy`. O último que rodar vence. Se quiser converter o resultado real do SGBM, rode o 08 de novo antes do `09_disparidade_para_profundidade.py`.
- **Arquivos de requisitos citados e ausentes.** `requirements-base.txt` e `requirements-opcionais.txt` aparecem em mensagens dos scripts 00 e 15, mas não vieram no zip. Instale pela tabela da seção 2.
- **README citado no 26.** O docstring diz "conforme o README" para a estrutura do dataset, mas o README do professor não veio. A estrutura é `dados/dataset_custom/treino/<classe>/*.jpg` e `validacao/<classe>/*.jpg`, com no mínimo duas classes.
- **`cv2.face` exige contrib.** Instale `opencv-contrib-python` no lugar de `opencv-python`, nunca os dois juntos, senão um sobrescreve o outro.
- **SURF quase nunca existe.** Os wheels do pip não trazem `cv2.xfeatures2d.SURF_create`. O script 20 cai para SIFT de propósito. Não é erro.
- **`--imagem` obrigatório no 10.** Diferente dos outros, não há valor padrão. Use uma foto sua.
- **Webcam.** Scripts 00, 11, 12, 14 e 27 abrem a câmera. Se `VideoCapture(0)` não abrir, tente `--camera 1`. No Windows o primeiro acesso pode pedir permissão de privacidade.
- **Modelo errado no 25.** O padrão é `modelos/cnn_fashion.keras` (do 23). O 24 salva como `cnn_fashion_melhor.keras`, então passe `--modelo`. O modelo do 22 (MLP) tem entrada `(28, 28)` e o 25 envia `(28, 28, 1)`, o que gera erro de forma. Para inspecionar a MLP seria preciso ajustar o script.
- **Modelo no 27.** O padrão também é `cnn_fashion.keras`. O modelo do 24 funciona se passado por `--modelo`, porque tem a mesma entrada.
- **Inversão no 27.** O Fashion-MNIST é objeto claro em fundo preto. Com objeto escuro sobre papel branco, deixe a inversão ligada (padrão). Se o fundo for escuro, aperte `i`.
- **Limite do `uint8`.** Ao somar ruído ou brilho, converta para `int16` ou `float32` antes e use `np.clip`, como o script 01 faz. Somar direto em `uint8` "dá a volta" (250 + 10 vira 4).
- **Ordem (x, y) contra [y, x].** Funções de desenho usam `(x, y)`; fatiamento NumPy usa `[y, x]`. `face_recognition` devolve `(topo, direita, base, esquerda)`. Misturar essas convenções é a fonte mais comum de recorte errado.
- **HSV do OpenCV.** H vai de 0 a 179. Valores de tabelas na internet (0 a 360) precisam ser divididos por 2.
- **Parâmetros que valem variar.** `minNeighbors` e `scaleFactor` no Haar; `--limiar` no LBPH e `--tolerancia` nos embeddings; `--ratio` e `--reprojecao` no matching; `numDisparities` e `blockSize` no SGBM; `--epocas` e `--batch` nos treinos; o tamanho do kernel morfológico no 05.
- **Tempo de treino.** Em CPU, 8 épocas do 22 levam poucos minutos; o 24 com 20 épocas e aumento de dados demora bem mais. Reduza `--epocas` para testar o fluxo antes.
- **Fidelidade ao professor.** Os docstrings usam `python exemplos/...` porque a pasta original se chamava `exemplos`. Aqui ela é `exemplos_em_python`, e o mais simples é rodar de dentro dela. Os scripts não foram alterados.

## 8. Glossário

| Termo | Significado |
|---|---|
| Adam | Otimizador de gradiente com taxa de aprendizado adaptativa, padrão em redes neurais. |
| Aumento de dados (data augmentation) | Geração de variações aleatórias (rotação, translação, zoom, espelho) das imagens de treino para reduzir overfitting. |
| Baseline | Distância entre as duas câmeras de um sistema estéreo. |
| BatchNormalization | Camada que normaliza as ativações de cada lote, acelerando e estabilizando o treino. |
| BFMatcher | Brute-Force Matcher: compara cada descritor de uma imagem com todos da outra. |
| BGR | Ordem de canais do OpenCV: azul, verde, vermelho. |
| Bounding box | Retângulo alinhado aos eixos que envolve um contorno ou objeto. |
| Callback | Objeto que o Keras chama a cada época para salvar, parar ou ajustar o treino. |
| CNN | Rede neural convolucional: usa filtros deslizantes que preservam a estrutura espacial da imagem. |
| Cross-check | Critério do BFMatcher que só aceita um par se ele for o melhor nos dois sentidos. |
| Descritor | Vetor numérico que resume a vizinhança de um keypoint para comparação. |
| Disparidade | Diferença horizontal, em pixels, entre a posição de um ponto na imagem esquerda e na direita. |
| Distância de Hamming | Número de bits diferentes entre dois descritores binários. |
| Dropout | Camada que zera aleatoriamente uma fração das ativações no treino, como regularização. |
| EarlyStopping | Callback que interrompe o treino quando a métrica de validação para de melhorar. |
| Elemento estruturante (kernel morfológico) | Forma (elipse, retângulo) usada pela erosão e dilatação. |
| Embedding | Vetor de baixa dimensão produzido por uma rede, no qual a distância reflete similaridade. |
| Equalização de histograma | Redistribuição dos níveis de cinza para aumentar contraste. |
| Fashion-MNIST | Dataset de 70 mil imagens 28x28 de 10 classes de roupas. |
| FPS | Frames por segundo. |
| GrabCut | Algoritmo de segmentação de primeiro plano baseado em corte em grafo e modelos de cor. |
| GlobalAveragePooling2D | Camada que substitui cada mapa de ativação pela sua média. |
| Haar Cascade | Detector de objetos em cascata baseado em features Haar (Viola-Jones). |
| Homografia | Matriz 3x3 que mapeia um plano para outro sob projeção perspectiva. |
| HSV | Espaço de cor matiz, saturação e valor. No OpenCV H vai de 0 a 179. |
| Inlier | Par de pontos consistente com o modelo estimado pelo RANSAC. |
| Keypoint | Ponto distintivo da imagem, com posição, escala e orientação. |
| k-NN (no matching) | Busca dos k descritores mais próximos; aqui k = 2 para o ratio test. |
| LBPH | Local Binary Patterns Histograms, reconhecedor facial clássico do módulo `cv2.face`. |
| Limiarização (threshold) | Conversão de imagem em cinza para binária por comparação com um limiar. |
| MLP | Multi-Layer Perceptron, rede de camadas densas totalmente conectadas. |
| MobileNetV2 | Rede convolucional leve pré-treinada no ImageNet, usada como base de transfer learning. |
| Morfologia (abertura e fechamento) | Operações de erosão e dilatação combinadas para limpar máscaras binárias. |
| ORB | Oriented FAST and Rotated BRIEF, detector e descritor binário livre e rápido. |
| Otsu | Método que escolhe o limiar automaticamente a partir do histograma. |
| Overfitting | Quando o modelo decora o treino e piora na validação. |
| PCA | Análise de componentes principais, projeção linear que preserva a maior variância. |
| RANSAC | Random Sample Consensus, estimativa robusta que ignora outliers por amostragem aleatória. |
| Ratio test de Lowe | Aceita um match só se a melhor distância for menor que `ratio` vezes a segunda melhor. |
| Retificação | Transformação do par estéreo para que pontos correspondentes fiquem na mesma linha. |
| ROI | Region of Interest, região de interesse: sub-array da imagem. |
| SGBM | Semi-Global Block Matching, algoritmo de disparidade do OpenCV. |
| SIFT | Scale-Invariant Feature Transform, detector e descritor invariante a escala e rotação. |
| Softmax | Ativação que transforma a saída em probabilidades que somam 1. |
| Sparse categorical crossentropy | Perda para classificação com rótulos inteiros. |
| SURF | Speeded-Up Robust Features, detector baseado na Hessiana; exige build nonfree. |
| Transfer learning | Reaproveitar uma rede pré-treinada, congelando a base e treinando só a cabeça nova. |
| uint8 | Inteiro sem sinal de 8 bits (0 a 255), tipo padrão dos pixels. |
| Validação (validation_split) | Fração do treino reservada para medir generalização a cada época. |
