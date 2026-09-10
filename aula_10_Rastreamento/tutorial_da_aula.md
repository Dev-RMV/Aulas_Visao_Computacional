# Aula 10 – Rastreamento

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula ensina rastreamento clássico de objetos em movimento com OpenCV, sem redes neurais. O problema central é: dado um vídeo (webcam ou arquivo), como separar o que se move do que fica parado, como seguir um objeto específico de frame em frame, e como suavizar a trajetória medida quando a medição falha ou fica ruidosa. O professor responde com três famílias de técnicas: subtração de fundo (background subtraction) com os algoritmos MOG2 e KNN, rastreamento por cor com CamShift a partir de uma região de interesse (ROI), e filtragem de Kalman para prever e suavizar a posição do objeto.

O fio condutor dos scripts é incremental. O `00_check_environment.py` e o `01_webcam_viewer_metadata.py` mostram como abrir uma fonte de vídeo e medir FPS. O `generate_synthetic_video.py` cria um vídeo de teste controlado, com um círculo vermelho, um retângulo azul e uma oclusão temporária. Os scripts 02 a 04 aplicam MOG2 e KNN, limpam a máscara com morfologia e contam objetos por contornos. Os scripts 05 e 06 seguem o círculo com CamShift e depois acoplam um filtro de Kalman. Os scripts 07 a 09 tratam de medição: varredura de parâmetros, registro em CSV e gráfico de trajetória. O `10_full_lab_pipeline.py` junta tudo em um único laço. Um módulo `utils.py` concentra as funções compartilhadas.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas nos scripts, conforme o `requirements.txt` do professor:

| Biblioteca | Pacote pip | Onde é usada |
|---|---|---|
| OpenCV | `opencv-python` | todos os scripts |
| NumPy | `numpy` | `utils.py`, 04, 05, 06, 10, gerador de vídeo |
| Matplotlib | `matplotlib` | 06, 09, 10 |
| pandas | `pandas` | 09 |

Instalação, de dentro de `exemplos_em_python/`:

```
pip install -r requirements.txt
```

Pasta de execução: todos os caminhos são relativos ao diretório de trabalho (`data/synthetic_motion.mp4`, `output/...`). Rode sempre de dentro de `aula_10_Rastreamento/exemplos_em_python/`. As pastas `data/` e `output/` já vêm no zip do professor. O vídeo `data/synthetic_motion.mp4` (6,3 MB) veio pronto, então não é obrigatório rodar o gerador.

Material faltante: nenhum. Segundo `achados_dos_zips.md`, esta é uma das aulas que veio completa.

Webcam: só o `00_check_environment.py` e o `01_webcam_viewer_metadata.py` usam a câmera por padrão. Todos os outros usam o vídeo sintético por padrão e aceitam `--source 0` para trocar pela webcam.

Python: o `__pycache__` do zip era de Python 3.13, então o professor rodou a aula com essa versão.

## 3. Conceitos-chave

### 3.1 Abertura de fonte de vídeo com `cv2.VideoCapture`

Tudo começa com a leitura de frames. O OpenCV abre tanto uma câmera (índice inteiro) quanto um arquivo ou URL (string) pela mesma classe `cv2.VideoCapture`. O professor encapsula isso em `utils.py`: `parse_source` tenta converter o argumento para inteiro, e se falhar trata como caminho. `open_capture` lança `RuntimeError` se a fonte não abrir, o que evita laços que rodam sem frame nenhum.

O laço padrão é `ok, frame = cap.read()` até `ok` ser falso (fim do arquivo ou câmera desligada), com `cv2.waitKey(1)` para desenhar as janelas e ler a tecla `q`. Ao final, `cap.release()` e `cv2.destroyAllWindows()`.

```python
# utils.py
def parse_source(value):
    """Converte '0' em câmera 0; demais valores são tratados como caminho/URL."""
    try:
        return int(value)
    except ValueError:
        return value

def open_capture(source):
    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {source}")
    return cap
```

Onde aparece: `utils.py`, `00_check_environment.py` (usa `cv2.VideoCapture(0)` direto), e todos os scripts de 01 a 10 via `open_capture`.

### 3.2 Argumentos de linha de comando com `argparse`

Cada script define seus parâmetros com `argparse.ArgumentParser`, o que permite trocar fonte, tamanho, thresholds e caminhos de saída sem editar código. O argumento `--source` é a convenção da aula: `"0"` para webcam, ou caminho de vídeo. Outros argumentos recorrentes: `--width` (largura máxima de exibição), `--min-area` (área mínima de contorno), `--roi x y w h` (região inicial do CamShift, com `nargs="*"`), `--q` e `--r` (ruídos do Kalman), `--output` (arquivo CSV).

```python
# 02_background_mog2_basic.py
parser = argparse.ArgumentParser(description="Background subtraction com MOG2.")
parser.add_argument("--source", default="data/synthetic_motion.mp4")
parser.add_argument("--history", type=int, default=500)
parser.add_argument("--var-threshold", type=float, default=16)
parser.add_argument("--detect-shadows", action="store_true")
parser.add_argument("--min-area", type=float, default=500)
parser.add_argument("--width", type=int, default=800)
args = parser.parse_args()
```

Onde aparece: todos os scripts numerados de 01 a 10 e `generate_synthetic_video.py`.

### 3.3 Redimensionamento mantendo proporção

Frames grandes deixam o processamento lento. `resize_keep_aspect` reduz o frame para uma largura máxima calculando um fator de escala único para os dois eixos com `cv2.resize(frame, None, fx=scale, fy=scale)`. Se o frame já é menor ou igual à largura pedida, ele volta intacto: a função só encolhe, nunca amplia. O vídeo sintético tem 640 pixels de largura, então com `--width 800` (padrão nos scripts 01, 02, 03, 05 e 06) ele não muda de tamanho.

```python
# utils.py
def resize_keep_aspect(frame, width=None):
    if width is None or frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, None, fx=scale, fy=scale)
```

Onde aparece: `utils.py` e todos os scripts de 01 a 10.

### 3.4 Medição de FPS e tempo de laço com `time.perf_counter`

Rastreamento é um problema de tempo real, então a aula mede desempenho desde o início. O `01_webcam_viewer_metadata.py` calcula o FPS médio como número de frames dividido pelo tempo desde o início, e o tempo do laço como diferença entre `perf_counter()` antes e depois do processamento, em milissegundos. Os scripts seguintes usam um `collections.deque(maxlen=120)` para guardar só os últimos 120 tempos e exibir uma média móvel, que reflete o desempenho recente e não o acumulado.

```python
# 01_webcam_viewer_metadata.py
tic = time.perf_counter()
ok, frame = cap.read()
...
elapsed = time.perf_counter() - t0
fps = frame_count / max(elapsed, 1e-6)
proc_ms = (time.perf_counter() - tic) * 1000
```

```python
# 02_background_mog2_basic.py
times = deque(maxlen=120)
...
times.append(elapsed_ms)
avg_ms = sum(times) / len(times)
```

Onde aparece: `01_webcam_viewer_metadata.py` (FPS), scripts 02 a 08 e 10 (deque com média móvel), `07_parameter_sweep_background.py` (soma total dividida por frames).

### 3.5 Sobreposição de texto legível com `cv2.putText`

Para mostrar métricas no próprio frame, `overlay_text` desenha cada linha duas vezes: primeiro em preto com espessura 4, depois em branco com espessura 2. O resultado é um texto com contorno escuro que continua legível sobre fundos claros ou escuros. A função trabalha em uma cópia do frame para não alterar o original, e desloca 24 pixels por linha.

```python
# utils.py
def overlay_text(frame, lines, x=10, y=24, color=(255, 255, 255)):
    out = frame.copy()
    for i, line in enumerate(lines):
        yy = y + i * 24
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    return out
```

Onde aparece: `utils.py` e scripts 01 a 06, 08 e 10.

### 3.6 Geração de vídeo sintético com `cv2.VideoWriter`

Para ter um caso de teste reproduzível, o professor gera um vídeo em vez de depender de câmera. `cv2.VideoWriter` recebe caminho, codec (`cv2.VideoWriter_fourcc(*"mp4v")`), FPS e tamanho (largura, altura). Cada frame é um array NumPy `uint8` de forma `(altura, largura, 3)` desenhado do zero e gravado com `writer.write(frame)`.

O conteúdo foi projetado para exercitar cada técnica da aula:

- Fundo cinza escuro (valor 35) com ruído gaussiano de desvio 7, gerado por `np.random.default_rng(7)` com semente fixa. Isso simula o ruído de sensor real e obriga a subtração de fundo a lidar com pequenas variações.
- Grade de linhas fixas a cada 80 e 60 pixels, que devem ser aprendidas como fundo.
- Círculo vermelho (BGR `(35, 35, 220)`) com raio 28 que anda para a direita e oscila em seno. Saturado e de matiz único, é o alvo ideal para o CamShift em HSV.
- Retângulo azul que anda para a esquerda e oscila em cosseno, para testar a contagem por subtração de fundo com dois objetos.
- Retângulo preto fixo entre os frames 110 e 145, que cobre parcialmente o círculo. Essa oclusão é o que justifica o filtro de Kalman.
- Texto `frame N` no rodapé.

O operador `% args.width` faz os objetos reaparecerem do outro lado quando saem da tela.

```python
# generate_synthetic_video.py
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
for t in range(args.frames):
    frame = np.full((args.height, args.width, 3), 35, dtype=np.uint8)
    noise = rng.normal(0, 7, frame.shape).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cx = 60 + int(2.0 * t)
    cy = 145 + int(45 * np.sin(t / 25))
    cv2.circle(frame, (cx % args.width, cy), 28, (35, 35, 220), -1)
    if 110 < t < 145:
        cv2.rectangle(frame, (260, 95), (340, 220), (20, 20, 20), -1)
    writer.write(frame)
writer.release()
```

Onde aparece: `generate_synthetic_video.py`.

### 3.7 Subtração de fundo com MOG2

Subtração de fundo é a técnica de construir um modelo estatístico do que é "fundo" em cada pixel e classificar como "primeiro plano" (foreground) o que foge desse modelo. MOG2 (Mixture of Gaussians, versão 2) modela cada pixel como uma mistura de distribuições gaussianas, o que permite fundos com pequenas variações (ruído, folhas balançando). O modelo se adapta ao longo do tempo: um objeto que para de se mover acaba absorvido pelo fundo.

A API é `cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows)`:

- `history`: quantos frames recentes pesam no modelo (padrão do professor: 500). Maior é mais estável, mas se adapta mais devagar a mudanças de iluminação.
- `varThreshold`: limiar sobre a distância de Mahalanobis ao quadrado entre o pixel e o modelo. Menor é mais sensível (mais ruído vira objeto); maior é mais conservador (objetos fracos somem). Padrão 16.
- `detectShadows`: se verdadeiro, pixels de sombra recebem valor 127 na máscara em vez de 255.

A cada frame, `subtractor.apply(frame)` devolve a máscara em escala de cinza e atualiza o modelo.

```python
# 02_background_mog2_basic.py
subtractor = cv2.createBackgroundSubtractorMOG2(
    history=args.history,
    varThreshold=args.var_threshold,
    detectShadows=args.detect_shadows,
)
while True:
    ok, frame = cap.read()
    ...
    mask = subtractor.apply(frame)
    clean = clean_foreground_mask(mask)
    contours = contours_from_mask(clean, min_area=args.min_area)
```

Onde aparece: `02_background_mog2_basic.py`, `04_compare_mog2_knn.py`, `07_parameter_sweep_background.py`, `08_logging_to_csv.py`, `10_full_lab_pipeline.py`.

### 3.8 Subtração de fundo com KNN

KNN (K-Nearest Neighbors) é a alternativa não paramétrica. Em vez de gaussianas, cada pixel guarda um conjunto de amostras recentes, e o pixel atual é fundo se tiver vizinhos suficientes dentro de uma distância. A API é `cv2.createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows)`. O parâmetro `dist2Threshold` é a distância ao quadrado no espaço de cor (padrão 400, ou seja, distância 20). O papel dele é análogo ao `varThreshold` do MOG2, mas a escala é diferente, por isso a varredura do script 07 testa valores de 100 a 1200 para KNN e de 8 a 32 para MOG2.

A aula compara os dois lado a lado: mesma máscara de limpeza, mesma contagem de contornos, e o tempo médio de cada um. O script 04 mostra as duas máscaras em uma janela (`np.hstack`) e os dois frames anotados em outra, verde para MOG2 e laranja para KNN.

```python
# 04_compare_mog2_knn.py
mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
knn = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)
...
mask_mog2, ann_mog2, cont_mog2, t_mog2 = process("MOG2", mog2, frame, args.min_area)
mask_knn, ann_knn, cont_knn, t_knn = process("KNN", knn, frame, args.min_area)
masks = np.hstack([cv2.cvtColor(mask_mog2, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mask_knn, cv2.COLOR_GRAY2BGR)])
```

Onde aparece: `03_background_knn_basic.py`, `04_compare_mog2_knn.py`, `07_parameter_sweep_background.py`, `08_logging_to_csv.py`, `10_full_lab_pipeline.py`.

### 3.9 Limpeza da máscara: limiarização e morfologia

A máscara bruta dos subtratores tem três problemas: pixels de sombra em 127, pontos isolados de ruído, e buracos dentro dos objetos. `clean_foreground_mask` resolve os três em sequência:

1. `cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)`: tudo abaixo de 200 vira 0. Isso descarta as sombras (127) e mantém só o foreground pleno (255).
2. `cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)`: abertura (erosão seguida de dilatação) remove pontos pequenos.
3. `cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)`: fechamento (dilatação seguida de erosão) preenche buracos e une fragmentos próximos.

O elemento estruturante é uma elipse 5x5 criada por `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))`. O parâmetro `min_area` existe na assinatura mas não é usado dentro da função; a filtragem por área acontece em `contours_from_mask`.

```python
# utils.py
def clean_foreground_mask(mask, kernel_size=5, min_area=500):
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
```

Onde aparece: `utils.py`, scripts 02, 03, 04, 07, 08 e 10.

### 3.10 Contornos, filtro por área e caixas delimitadoras

Com a máscara limpa, cada mancha branca conectada vira um objeto. `cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)` extrai só os contornos externos (ignora buracos internos) e comprime segmentos retos em seus extremos. `cv2.contourArea` mede a área, e contornos abaixo de `min_area` (padrão 500 pixels) são descartados como ruído. Isso é o que transforma "pixels em movimento" em "número de objetos".

Para anotar, `cv2.boundingRect(c)` devolve o retângulo alinhado aos eixos `(x, y, w, h)`, desenhado com `cv2.rectangle`, e cada objeto recebe um rótulo `obj N` acima da caixa.

```python
# utils.py
def contours_from_mask(mask, min_area=500):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            valid.append(c)
    return valid

def annotate_contours(frame, contours, color=(0, 255, 0)):
    annotated = frame.copy()
    for idx, c in enumerate(contours, start=1):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"obj {idx}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated
```

Onde aparece: `utils.py`, scripts 02, 03, 04, 07, 08 e 10.

### 3.11 Seleção de ROI com `cv2.selectROI`

CamShift precisa saber o que rastrear. O professor oferece dois caminhos: passar a ROI pela linha de comando (`--roi x y w h`, validada por `parse_roi`) ou selecionar com o mouse na primeira imagem. `cv2.selectROI(nome_janela, frame, fromCenter=False, showCrosshair=True)` abre uma janela, deixa o usuário arrastar um retângulo, e devolve `(x, y, w, h)` ao pressionar Enter ou Espaço. A janela é fechada em seguida com `cv2.destroyWindow`. A tecla `r` durante o rastreamento reabre a seleção para reiniciar.

```python
# 05_camshift_roi.py
roi = parse_roi(args.roi)
if roi is None:
    roi = cv2.selectROI("Selecione a ROI do objeto", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Selecione a ROI do objeto")
```

Onde aparece: `05_camshift_roi.py`, `06_camshift_kalman.py`, `10_full_lab_pipeline.py`.

### 3.12 Histograma de matiz em HSV e backprojection

O CamShift rastreia por cor. O espaço HSV separa matiz (H, a cor em si, de 0 a 180 no OpenCV), saturação (S) e valor (V, brilho), então o histograma só do canal H descreve "a cor do objeto" de forma razoavelmente robusta a mudanças de iluminação.

`initialize_camshift` faz quatro passos sobre a ROI:

1. Converte para HSV com `cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)`.
2. Cria uma máscara com `cv2.inRange(roi_hsv, (0, 40, 30), (180, 255, 255))`: só pixels com saturação de pelo menos 40 e brilho de pelo menos 30 entram no histograma. Isso exclui cinzas e pretos, cujo matiz é indefinido.
3. Calcula o histograma de 180 bins do canal 0 com `cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])`.
4. Normaliza para o intervalo 0 a 255 com `cv2.normalize(..., cv2.NORM_MINMAX)`.

A cada frame novo, `cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)` gera a backprojection: uma imagem em que cada pixel recebe o valor do histograma no bin do seu matiz. Pixels com a cor do objeto ficam claros, o resto fica escuro. É essa imagem de probabilidade que o CamShift usa.

```python
# 05_camshift_roi.py
def initialize_camshift(frame, roi):
    x, y, w, h = roi
    track_window = (x, y, w, h)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi_hsv = hsv[y:y + h, x:x + w]
    mask = cv2.inRange(roi_hsv, np.array((0, 40, 30)), np.array((180, 255, 255)))
    hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return track_window, hist
...
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
```

Onde aparece: `05_camshift_roi.py`, `06_camshift_kalman.py`, `10_full_lab_pipeline.py`.

### 3.13 Rastreamento com CamShift

CamShift (Continuously Adaptive Mean Shift) é o Mean Shift com janela adaptativa. Mean Shift desloca uma janela iterativamente para o centro de massa da backprojection dentro dela, até convergir. CamShift faz o mesmo e ainda ajusta o tamanho e a orientação da janela conforme o objeto se aproxima, se afasta ou gira.

`cv2.CamShift(backproj, track_window, term_criteria)` devolve dois valores: um `RotatedRect` na forma `((cx, cy), (w, h), ângulo)`, e a nova janela `(x, y, w, h)` que deve ser passada de volta no próximo frame. O critério de parada `(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)` interrompe as iterações após 10 passos ou quando o deslocamento fica abaixo de 1 pixel.

O resultado é desenhado com `cv2.ellipse(out, rotated_rect, ...)`, que aceita o `RotatedRect` direto, e o centro `rotated_rect[0]` é a medição de posição usada pelo Kalman em seguida.

```python
# 05_camshift_roi.py
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
...
rotated_rect, track_window = cv2.CamShift(backproj, track_window, term_criteria)
out = frame.copy()
cv2.ellipse(out, rotated_rect, (0, 255, 0), 2)
cx, cy = map(int, rotated_rect[0])
cv2.circle(out, (cx, cy), 4, (0, 255, 255), -1)
```

Onde aparece: `05_camshift_roi.py`, `06_camshift_kalman.py`, `10_full_lab_pipeline.py`.

### 3.14 Filtro de Kalman: modelo de velocidade constante

O CamShift mede a posição, mas a medição falha quando o objeto é ocluído e treme quando a backprojection é ruidosa. O filtro de Kalman combina um modelo de movimento com as medições, ponderando cada um pela sua incerteza, e entrega uma estimativa mais suave que ainda acompanha o objeto.

O professor usa `cv2.KalmanFilter(4, 2)`: estado de 4 dimensões `[x, y, vx, vy]` e medição de 2 dimensões `[x, y]`. As matrizes configuradas são:

- `transitionMatrix` (F): modelo de velocidade constante. A posição nova é a antiga mais a velocidade vezes `dt` (1 frame). A velocidade se mantém.
- `measurementMatrix` (H): diz que só se mede x e y, não as velocidades.
- `processNoiseCov` (Q): incerteza do modelo, `np.eye(4) * q`. Se Q sobe, o filtro confia menos na velocidade constante e reage mais rápido a manobras, mas suaviza menos.
- `measurementNoiseCov` (R): incerteza da medição, `np.eye(2) * r`. Se R sobe, o filtro confia menos no CamShift e suaviza mais.
- `errorCovPost` (P): incerteza inicial do estado, `np.eye(4) * p`. Maior P aceita correções fortes no começo.
- `statePost`: estado inicial, colocado no centro da ROI com velocidade zero.

Todas as matrizes devem ser `np.float32`, exigência do OpenCV.

```python
# 06_camshift_kalman.py
kf = cv2.KalmanFilter(4, 2)
kf.transitionMatrix = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)
kf.measurementMatrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
kf.errorCovPost = np.eye(4, dtype=np.float32) * p
```

Onde aparece: `06_camshift_kalman.py` (função `create_kalman` com docstring explicando Q, R e P), `10_full_lab_pipeline.py` (mesma função, com `dt` fixo em 1 e `p` fixo em 1.0).

### 3.15 Ciclo predição e correção do Kalman

O filtro funciona em dois passos por frame. `kf.predict()` propaga o estado pelo modelo de movimento e devolve a predição a priori: "onde o objeto deveria estar se continuasse no mesmo ritmo". `kf.correct(measurement)` incorpora a medição do CamShift e atualiza o estado a posteriori. A medição precisa ser um array coluna `float32` de forma `(2, 1)`.

Nos scripts do professor, o valor registrado como `kalman_x`, `kalman_y` é a predição, capturada antes da correção. Por isso, durante a oclusão do vídeo sintético (frames 111 a 144), quando o CamShift perde o círculo, o ponto azul continua seguindo a tendência anterior enquanto o ponto vermelho salta. Depois da oclusão, a correção puxa o filtro de volta.

```python
# 06_camshift_kalman.py
measured_x, measured_y = rotated_rect[0]
measurement = np.array([[np.float32(measured_x)], [np.float32(measured_y)]])

prediction = kf.predict()
predicted_x, predicted_y = float(prediction[0]), float(prediction[1])
kf.correct(measurement)

cv2.circle(out, (int(measured_x), int(measured_y)), 7, (0, 0, 255), -1)   # vermelho: medição CamShift
cv2.circle(out, (int(predicted_x), int(predicted_y)), 7, (255, 0, 0), -1) # azul: predição Kalman
```

Onde aparece: `06_camshift_kalman.py`, `10_full_lab_pipeline.py`.

### 3.16 Varredura de parâmetros

Escolher `varThreshold` ou `dist2Threshold` "no olho" é pouco confiável. O `07_parameter_sweep_background.py` automatiza: para cada valor de uma lista, reabre o vídeo, cria um subtrator novo, processa até `--frames` frames (padrão 180) e acumula tempo total e número total de objetos. No fim calcula média de milissegundos por frame e média de objetos por frame. As listas testadas são `[8, 12, 16, 24, 32]` para MOG2 e `[100, 200, 400, 800, 1200]` para KNN. Não há janela; o script só imprime e grava CSV.

A leitura do resultado: se `avg_objects` está muito acima de 2 (o vídeo tem dois objetos), o threshold está baixo demais e o ruído vira objeto. Se cai abaixo de 2, está alto demais e objetos somem.

```python
# 07_parameter_sweep_background.py
for threshold in thresholds:
    cap = open_capture(source)
    if method == "mog2":
        subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=threshold, detectShadows=True)
    else:
        subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=threshold, detectShadows=True)
    while frame_idx < frames_limit:
        ...
        total_ms += (time.perf_counter() - tic) * 1000
        total_objects += len(contours)
    rows.append({"method": method, "threshold": threshold, "frames": frame_idx,
                 "avg_ms": total_ms / max(frame_idx, 1), "avg_objects": total_objects / max(frame_idx, 1)})
```

Onde aparece: `07_parameter_sweep_background.py`.

### 3.17 Registro de métricas em CSV

Três scripts gravam CSV com o módulo `csv` da biblioteca padrão. O padrão é: `Path(args.output)`, `parent.mkdir(parents=True, exist_ok=True)` para garantir a pasta `output/`, abrir com `newline=""` e `encoding="utf-8"`, escrever cabeçalho e depois uma linha por frame (ou por configuração). O script 07 usa `csv.DictWriter` com `fieldnames` e `writeheader()`; os scripts 06, 08 e 10 usam `csv.writer` com `writerow`.

Os arquivos gerados e suas colunas:

| Script | Arquivo padrão | Colunas |
|---|---|---|
| 06 | `output/trajectory_camshift_kalman.csv` | frame, measured_x, measured_y, kalman_x, kalman_y, loop_ms |
| 07 | `output/background_parameter_sweep.csv` | method, threshold, frames, avg_ms, avg_objects |
| 08 | `output/background_metrics.csv` | frame, method, objects, loop_ms, avg_loop_ms |
| 10 | `output/full_pipeline_trajectory.csv` | frame, mog2_objects, knn_objects, measured_x, measured_y, kalman_x, kalman_y, track_ms |

```python
# 08_logging_to_csv.py
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "method", "objects", "loop_ms", "avg_loop_ms"])
    while True:
        ...
        writer.writerow([frame_idx, args.method, len(contours), loop_ms, avg_ms])
```

Onde aparece: `06_camshift_kalman.py`, `07_parameter_sweep_background.py`, `08_logging_to_csv.py`, `10_full_lab_pipeline.py`.

### 3.18 Gráfico de trajetórias com Matplotlib e pandas

A trajetória medida e a predita são comparadas em um gráfico 2D. Os pontos vermelhos (`"r.-"`) são o CamShift e os azuis (`"b.-"`) o Kalman. Um detalhe importante: `plt.gca().invert_yaxis()` inverte o eixo vertical, porque em imagens o y cresce para baixo e no Matplotlib cresce para cima. Sem isso o gráfico sai espelhado. O gráfico é salvo com `plt.savefig(..., dpi=160, bbox_inches="tight")` e exibido com `plt.show()`.

O `09_plot_trajectories_csv.py` faz o mesmo gráfico lendo o CSV do script 06 com `pd.read_csv`, e valida que as quatro colunas obrigatórias existem antes de plotar. Isso separa a coleta da análise: pode-se rodar o rastreamento uma vez e replotar quantas vezes quiser.

```python
# 09_plot_trajectories_csv.py
df = pd.read_csv(args.csv)
required = {"measured_x", "measured_y", "kalman_x", "kalman_y"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"CSV sem colunas obrigatórias: {missing}")
plt.plot(df["measured_x"], df["measured_y"], "r.-", label="CamShift medido")
plt.plot(df["kalman_x"], df["kalman_y"], "b.-", label="Kalman predito")
plt.gca().invert_yaxis()
```

Onde aparece: `06_camshift_kalman.py`, `09_plot_trajectories_csv.py`, `10_full_lab_pipeline.py`.

### 3.19 Pipeline integrado

O `10_full_lab_pipeline.py` roda as três técnicas no mesmo frame, cada uma com seu cronômetro: MOG2 (máscara mais contornos), KNN (máscara mais contornos) e CamShift com Kalman. Abre cinco janelas (duas máscaras, dois frames anotados e o rastreamento), imprime uma linha por frame com tudo, grava o CSV com oito colunas e, ao final, salva e mostra o gráfico de trajetórias. É a demonstração de que as técnicas são complementares: a subtração de fundo conta objetos sem saber quem são, e o CamShift com Kalman segue um objeto escolhido.

```python
# 10_full_lab_pipeline.py
tic = time.perf_counter()
mask_mog2 = clean_foreground_mask(mog2.apply(frame))
cont_mog2 = contours_from_mask(mask_mog2, args.min_area)
times_mog2.append((time.perf_counter() - tic) * 1000)

tic = time.perf_counter()
mask_knn = clean_foreground_mask(knn.apply(frame))
cont_knn = contours_from_mask(mask_knn, args.min_area)
times_knn.append((time.perf_counter() - tic) * 1000)

tic = time.perf_counter()
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
backproj = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
rotated_rect, track_window = cv2.CamShift(backproj, track_window, term_criteria)
pred = kf.predict()
kf.correct(np.array([[np.float32(measured_x)], [np.float32(measured_y)]]))
```

Onde aparece: `10_full_lab_pipeline.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `00_check_environment.py` | Imprime versões de OpenCV e NumPy e testa a câmera 0 | webcam 0 (fixa) | texto no terminal | nada |
| `01_webcam_viewer_metadata.py` | Exibe a fonte de vídeo com shape, contador de frames, FPS médio e tempo do laço | `--source` (padrão `0`), `--width` (800) | janela | `utils.py` |
| `generate_synthetic_video.py` | Gera vídeo com círculo vermelho, retângulo azul, grade, ruído e oclusão | `--output`, `--frames` (260), `--width` (640), `--height` (360), `--fps` (30) | `data/synthetic_motion.mp4` | nada |
| `02_background_mog2_basic.py` | Subtração de fundo MOG2, limpeza, contagem de objetos, tempo médio | `--source`, `--history`, `--var-threshold`, `--detect-shadows`, `--min-area`, `--width` | duas janelas e log no terminal | `utils.py`, vídeo |
| `03_background_knn_basic.py` | Mesmo que o 02 com KNN | `--source`, `--history`, `--dist2-threshold`, `--detect-shadows`, `--min-area`, `--width` | duas janelas e log | `utils.py`, vídeo |
| `04_compare_mog2_knn.py` | MOG2 e KNN lado a lado, máscaras e frames anotados, tempos comparados | `--source`, `--min-area`, `--width` (640) | duas janelas e log | `utils.py`, vídeo |
| `05_camshift_roi.py` | CamShift por histograma de matiz a partir de uma ROI; mostra backprojection e elipse | `--source`, `--roi x y w h` ou seleção com mouse, `--width` | duas janelas | `utils.py`, vídeo |
| `06_camshift_kalman.py` | CamShift mais filtro de Kalman; grava trajetória e plota | `--source`, `--roi`, `--width`, `--q` (0.03), `--r` (8.0), `--output` | janela, `output/trajectory_camshift_kalman.csv` e `.png` | `utils.py`, vídeo |
| `07_parameter_sweep_background.py` | Testa 5 thresholds de MOG2 e 5 de KNN, mede tempo e objetos médios | `--source`, `--frames` (180), `--min-area`, `--width`, `--output` | `output/background_parameter_sweep.csv` e log | `utils.py`, vídeo |
| `08_logging_to_csv.py` | Subtração de fundo (mog2 ou knn) com registro por frame em CSV | `--source`, `--method`, `--output`, `--min-area`, `--width` | duas janelas, `output/background_metrics.csv` | `utils.py`, vídeo |
| `09_plot_trajectories_csv.py` | Lê o CSV do 06 e replota as trajetórias | `--csv`, `--output` | `output/trajectory_plot_from_csv.png` | CSV do 06, pandas |
| `10_full_lab_pipeline.py` | MOG2, KNN, CamShift e Kalman no mesmo laço; CSV e gráfico | `--source`, `--roi`, `--width`, `--min-area`, `--q`, `--r`, `--output` | cinco janelas, `output/full_pipeline_trajectory.csv` e `.png` | `utils.py`, vídeo |
| `utils.py` | Funções compartilhadas: abrir fonte, redimensionar, limpar máscara, contornos, anotação, texto | importado | nada | cv2, numpy |

## 5. Passo a passo para executar

Todos os comandos abaixo são executados de dentro de `aula_10_Rastreamento/exemplos_em_python/`. Em todas as janelas do OpenCV, a tecla `q` encerra.

Passo 1. Verificar o ambiente. Se não houver webcam, a mensagem final diz para usar `--source` com arquivo.

```
python 00_check_environment.py
```

Passo 2. Ver a fonte com metadados. Com webcam, observe o FPS médio e o tempo do laço no canto superior esquerdo. Sem webcam, aponte para o vídeo sintético.

```
python 01_webcam_viewer_metadata.py
python 01_webcam_viewer_metadata.py --source data/synthetic_motion.mp4
```

Passo 3. Gerar o vídeo sintético. Opcional, porque `data/synthetic_motion.mp4` já veio no zip. Rode se quiser mudar tamanho, duração ou FPS. Regerar sobrescreve o arquivo original.

```
python generate_synthetic_video.py
python generate_synthetic_video.py --frames 400 --fps 25
```

Passo 4. MOG2. Abrem duas janelas: a máscara limpa (branco sobre preto) e o frame com caixas verdes e rótulos `obj N`. Espere ver dois objetos na maior parte do tempo. O terminal imprime uma linha por frame. Experimente os thresholds.

```
python 02_background_mog2_basic.py
python 02_background_mog2_basic.py --var-threshold 8 --detect-shadows
python 02_background_mog2_basic.py --source 0
```

Passo 5. KNN. Mesma cena, caixas laranja.

```
python 03_background_knn_basic.py
python 03_background_knn_basic.py --dist2-threshold 200
```

Passo 6. Comparação lado a lado. Uma janela com as duas máscaras e outra com os dois frames anotados. Compare a contagem de objetos e o tempo médio de cada método.

```
python 04_compare_mog2_knn.py
```

Passo 7. CamShift. Sem `--roi`, aparece uma janela para arrastar um retângulo sobre o círculo vermelho; confirme com Enter. Com `--roi`, a região vem pronta. Observe a janela de backprojection (o círculo aparece claro) e a elipse verde que segue o objeto. Tecla `r` reinicia a ROI.

```
python 05_camshift_roi.py --roi 70 120 80 80
python 05_camshift_roi.py
```

Passo 8. CamShift com Kalman. Ponto vermelho é a medição, ponto azul é a predição. Preste atenção entre os frames 110 e 145, quando o retângulo preto cobre o círculo: o vermelho salta e o azul continua a trajetória. Ao fechar a janela com `q` (ou no fim do vídeo) o script grava o CSV, salva o PNG e abre o gráfico. Feche o gráfico para o script terminar. Varie Q e R para sentir o efeito.

```
python 06_camshift_kalman.py --roi 70 120 80 80
python 06_camshift_kalman.py --roi 70 120 80 80 --q 0.5 --r 1
python 06_camshift_kalman.py --roi 70 120 80 80 --q 0.001 --r 50
```

Passo 9. Varredura de parâmetros. Sem janela. Imprime uma linha por configuração e salva o CSV. Leva cerca de 10 execuções de 180 frames.

```
python 07_parameter_sweep_background.py
```

Passo 10. Registro por frame em CSV.

```
python 08_logging_to_csv.py --method mog2
python 08_logging_to_csv.py --method knn --output output/background_metrics_knn.csv
```

Passo 11. Replotar a trajetória do passo 8 a partir do CSV.

```
python 09_plot_trajectories_csv.py
```

Passo 12. Pipeline completo. Cinco janelas. Ao final, CSV com oito colunas e gráfico.

```
python 10_full_lab_pipeline.py --roi 70 120 80 80
```

Ao terminar todos os passos, a pasta `output/` contém: `trajectory_camshift_kalman.csv`, `trajectory_camshift_kalman.png`, `background_parameter_sweep.csv`, `background_metrics.csv`, `trajectory_plot_from_csv.png`, `full_pipeline_trajectory.csv` e `full_pipeline_trajectory.png`.

## 6. Exercícios e itens da aula

Os scripts desta aula não referenciam itens ou exercícios numerados. O `README.md` do professor chama a aula de "aula prática" e o script 10 de "full lab pipeline", mas não há divisão em itens. Exercícios sugeridos para praticar:

1. Abra `output/background_parameter_sweep.csv` e escolha, para cada método, o threshold que deixa `avg_objects` mais próximo de 2. Rode o `02` e o `03` com esses valores e confirme visualmente.
2. No `06_camshift_kalman.py`, rode com três combinações de `--q` e `--r` e compare os três PNGs gerados (use `--output` com nomes diferentes). Descreva em uma frase o efeito de cada parâmetro na oclusão.
3. Modifique uma cópia do `generate_synthetic_video.py` para deixar o círculo parado a partir do frame 150. Rode o `02` e observe em quantos frames o círculo é absorvido pelo fundo. Repita com `--history 100` e `--history 2000`.
4. No `05_camshift_roi.py`, selecione o retângulo azul em vez do círculo. Explique por que a backprojection fica diferente e o que acontece quando os dois objetos se cruzam.
5. Escreva um script que leia `output/background_metrics.csv` com pandas e plote `objects` e `loop_ms` por frame, no estilo do `09_plot_trajectories_csv.py`.

## 7. Armadilhas e dicas

- Rode sempre de dentro de `exemplos_em_python/`. Os caminhos `data/synthetic_motion.mp4` e `output/` são relativos ao diretório de trabalho, e `utils.py` é importado como módulo local.
- `00_check_environment.py` não aceita `--source`; ele sempre tenta a câmera 0. Se não houver câmera, apenas imprime o aviso e termina, sem erro.
- `01_webcam_viewer_metadata.py` é o único cujo `--source` tem padrão `0` (webcam). Os demais têm padrão no vídeo sintético.
- `--roi` sem valores (`--roi` seguido de nada) gera `ValueError: ROI deve ser x y w h`, porque `nargs="*"` devolve lista vazia. Passe os quatro inteiros ou omita a opção para selecionar com o mouse.
- A ROI `70 120 80 80` do README foi pensada para o vídeo sintético com largura 640. Como `resize_keep_aspect` só reduz, e `--width` padrão é 800 (scripts 05 e 06) ou 640 (script 10), o vídeo não é redimensionado e a ROI bate. Se passar `--width 320`, a ROI precisa ser dividida por 2.
- Ao selecionar ROI com o mouse, confirme com Enter ou Espaço. Cancelar com `c` devolve `(0, 0, 0, 0)`, e o histograma sai vazio: o CamShift não rastreia nada.
- O README do professor lista a tecla `p` para pausar "em alguns exemplos", mas nenhum script implementa essa tecla. Só `q` e `r` funcionam.
- Sem `--detect-shadows`, MOG2 e KNN não geram o valor 127; com a opção, a limiarização em 200 da `clean_foreground_mask` remove as sombras. Nos scripts 04, 07, 08 e 10 `detectShadows=True` está fixo no código.
- `clean_foreground_mask` recebe `min_area` mas não usa; o filtro por área está em `contours_from_mask`. Mantido como o professor escreveu.
- `utils.py` define `moving_average` e `ensure_output_dir`, mas nenhum script as usa. Mantidas por fidelidade ao professor.
- As funções `parse_roi`, `initialize_camshift` e `create_kalman` estão duplicadas nos scripts 05, 06 e 10 em vez de ficarem em `utils.py`. É o código do professor. Note que `create_kalman` no 06 recebe `(dt, q, r, p)` por nome e no 10 recebe `(q, r)` por posição.
- Todas as matrizes do `cv2.KalmanFilter` precisam ser `np.float32`. Passar `float64` gera erro de tipo dentro de `predict` ou `correct`.
- O valor gravado como `kalman_x`, `kalman_y` é a predição antes da correção, não o estado corrigido. Por isso a curva azul sempre "atrasa" um passo em relação à vermelha em movimento uniforme.
- Os scripts 06, 09 e 10 terminam com `plt.show()`, que bloqueia até a janela do gráfico ser fechada. O CSV e o PNG já foram salvos antes disso.
- No vídeo sintético, os objetos dão a volta pela borda (`% width`). Quando o círculo some à direita e reaparece à esquerda, o CamShift perde o alvo. Use `r` para reselecionar.
- Com webcam (`--source 0`), o modelo de fundo leva os primeiros segundos para estabilizar. Fique parado no início ou tudo vira objeto. Iluminação oscilante e câmera com auto-exposição também geram falsos positivos; suba o threshold ou a `--min-area`.
- Parâmetros que valem a pena variar: `--var-threshold` (MOG2), `--dist2-threshold` (KNN), `--history`, `--min-area`, `--q` e `--r` (Kalman), e a faixa de saturação e valor em `cv2.inRange` dentro de `initialize_camshift` se o objeto for pouco saturado.

## 8. Glossário

| Termo | Significado |
|---|---|
| Background subtraction | Subtração de fundo. Técnica que modela o fundo estático de cada pixel e marca como foreground o que desvia do modelo. |
| Foreground | Primeiro plano. Pixels classificados como pertencentes a objetos em movimento. |
| Máscara | Imagem binária (0 ou 255) que marca quais pixels são foreground. |
| MOG2 | Mixture of Gaussians 2. Subtrator de fundo que modela cada pixel com uma mistura de gaussianas adaptativa. |
| KNN | K-Nearest Neighbors. Subtrator de fundo não paramétrico baseado em amostras vizinhas no espaço de cor. |
| history | Número de frames recentes que influenciam o modelo de fundo. |
| varThreshold | Limiar do MOG2 sobre a distância de Mahalanobis ao quadrado entre pixel e modelo. |
| dist2Threshold | Limiar do KNN sobre a distância ao quadrado no espaço de cor. |
| Sombra | Pixel marcado com valor 127 pelos subtratores quando `detectShadows=True`. |
| Threshold | Limiar. Valor de corte que separa duas classes; em `cv2.threshold`, separa 0 de 255. |
| Morfologia | Operações sobre imagens binárias com um elemento estruturante: erosão, dilatação, abertura, fechamento. |
| Abertura (MORPH_OPEN) | Erosão seguida de dilatação. Remove pontos pequenos. |
| Fechamento (MORPH_CLOSE) | Dilatação seguida de erosão. Preenche buracos e une fragmentos. |
| Kernel / elemento estruturante | Pequena matriz (aqui, elipse 5x5) que define a vizinhança das operações morfológicas. |
| Contorno | Curva que delimita uma região conectada em uma máscara binária. |
| RETR_EXTERNAL | Modo do `findContours` que devolve só contornos externos. |
| CHAIN_APPROX_SIMPLE | Modo do `findContours` que guarda só os vértices dos segmentos retos. |
| Bounding box | Caixa delimitadora. Retângulo alinhado aos eixos que envolve um contorno, dado por `cv2.boundingRect`. |
| ROI | Region of Interest. Região de interesse, retângulo `(x, y, w, h)` que define o objeto a rastrear. |
| HSV | Espaço de cor com matiz (Hue), saturação (Saturation) e valor (Value). No OpenCV, H vai de 0 a 180. |
| Histograma | Contagem de pixels por faixa de valor. Aqui, 180 bins do canal de matiz. |
| Backprojection | Retroprojeção. Imagem em que cada pixel recebe a probabilidade da sua cor segundo o histograma do objeto. |
| Mean Shift | Algoritmo que move uma janela para o centro de massa da densidade dentro dela até convergir. |
| CamShift | Continuously Adaptive Mean Shift. Mean Shift com janela que adapta tamanho e orientação. |
| RotatedRect | Retângulo rotacionado `((cx, cy), (w, h), ângulo)` devolvido pelo CamShift. |
| Critério de parada | Tupla `(flags, max_iter, eps)` que limita as iterações do CamShift. |
| Filtro de Kalman | Estimador recursivo que combina um modelo de movimento com medições ruidosas, ponderando pela incerteza. |
| Estado | Vetor `[x, y, vx, vy]` estimado pelo Kalman: posição e velocidade. |
| Modelo de velocidade constante | Suposição de que a velocidade não muda entre frames; base da `transitionMatrix`. |
| transitionMatrix (F) | Matriz que propaga o estado de um frame para o próximo. |
| measurementMatrix (H) | Matriz que projeta o estado no espaço das medições (só x e y). |
| processNoiseCov (Q) | Covariância do ruído de processo. Incerteza do modelo de movimento. |
| measurementNoiseCov (R) | Covariância do ruído de medição. Incerteza do centroide do CamShift. |
| errorCovPost (P) | Covariância do erro do estado. Incerteza inicial da estimativa. |
| Predição (predict) | Passo do Kalman que propaga o estado pelo modelo, sem usar medição. |
| Correção (correct) | Passo do Kalman que incorpora a medição e atualiza o estado. |
| Oclusão | Momento em que o objeto rastreado fica escondido por outro. No vídeo sintético, frames 111 a 144. |
| FPS | Frames per second. Quadros por segundo processados. |
| deque | Fila de tamanho máximo fixo da biblioteca padrão, usada para média móvel dos tempos. |
| Varredura de parâmetros | Execução sistemática com vários valores de um parâmetro para comparar resultados. |
| CSV | Comma-Separated Values. Arquivo de texto tabular usado para registrar métricas. |
| FOURCC | Código de quatro letras do codec de vídeo, aqui `mp4v`. |
| Frame | Quadro individual de um vídeo, representado como array NumPy `(altura, largura, 3)` em BGR. |
| BGR | Ordem dos canais de cor do OpenCV: azul, verde, vermelho. |
