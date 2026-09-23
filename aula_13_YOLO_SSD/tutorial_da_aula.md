# Aula 13 – YOLO, SSD e rastreamento por IoU

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula muda o ponto de vista em relação à aula 10. Lá, o objeto em movimento era encontrado por subtração de fundo e seguido por cor. Aqui, o objeto é encontrado por um detector: um modelo que recebe um frame e devolve uma lista de caixas delimitadoras, cada uma com uma classe e uma confiança. O professor trabalha com as duas famílias mais usadas em sistemas embarcados, YOLO (You Only Look Once, na versão YOLOv8 da Ultralytics e na YOLOv4-tiny em Darknet) e SSD (Single Shot MultiBox Detector, na variante SSD MobileNet), e mostra que detectar não é só desenhar caixas: envolve filtrar por confiança, remover duplicatas com Non-Maximum Suppression (NMS), medir FPS e latência, comparar tamanho em disco e número de parâmetros, e decidir qual modelo cabe em uma placa embarcada.

A segunda metade da aula responde a uma pergunta que a detecção sozinha não resolve: a caixa deste frame é o mesmo objeto da caixa do frame anterior? Detectar frame a frame não garante identidade. O professor constrói um rastreador simples por associação de IoU (Intersection over Union) entre frames consecutivos, com IDs persistentes, trilhas dos últimos 30 frames, contagem de entradas e saídas por uma linha virtual e uma estimativa de ID switches por minuto. A aula termina com o pipeline detecção mais NMS mais rastreamento, templates para preencher com medições reais e uma discussão técnica e ética sobre contagem de pessoas por drone.

O fio condutor é incremental e sintético. Um módulo `vision_utils.py` desenha uma rua com uma pessoa, um carro, uma bicicleta e um ônibus em movimento, com caixas conhecidas, e simula um detector com ruído e caixas duplicadas. Assim, os scripts 01 a 10 (fundamentos), 16 a 28 (comparação e rastreamento) e 30 a 35 (análise) rodam sem câmera, sem dataset e sem modelo pesado. Os scripts 11 a 15 e 29 são a ponte para modelos reais e dependem de arquivos que o aluno precisa obter. O `GUIA_PROFESSOR.md` organiza tudo em quatro momentos, e o `notebook_pipeline_yolo_ssd_tracking.ipynb` repete o pipeline em células.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas nos scripts:

| Biblioteca | Pacote pip | Onde é usada |
|---|---|---|
| OpenCV | `opencv-contrib-python` (no `requirements.txt` da raiz) | `vision_utils.py`, 03, 04, 10, 11 a 15, 24, 27, 28, 29, 35 |
| NumPy | `numpy` | `vision_utils.py` |
| pandas | `pandas` | 18, 30 |
| Matplotlib | `matplotlib` | 33 |
| Ultralytics | `ultralytics` (em `requirements_ultralytics.txt`, instalado com `--no-deps`) | 11, 12, 29, opcional |
| PyTorch e torchvision | `torch`, `torchvision` | dependências do Ultralytics, não importadas diretamente |

O repositório usa um único ambiente para todas as aulas: `requirements.txt` na raiz, venv `.venv` com Python 3.13, criado por `py preparar_ambiente.py`. Nesta rodada entraram `torch` 2.14 e `torchvision` 0.29 no `requirements.txt`. O `ultralytics` 8.4.160 fica em um arquivo separado, `requirements_ultralytics.txt`, instalado com `--no-deps`, pelo mesmo motivo do `face-recognition` das aulas 03 e 06: o pacote declara dependência de `opencv-python`, que conflita com o `opencv-contrib-python` já instalado (os dois disputam o mesmo módulo `cv2`). As dependências restantes do Ultralytics já estão no ambiente. O `preparar_ambiente.py` executa os dois passos. O Jupyter (`notebook`) não faz parte do ambiente unificado; para abrir o notebook do professor é preciso instalar à parte com `pip install notebook`.

Os três scripts que usam o Ultralytics fazem `try/except ImportError` e encerram com a mensagem `Instale com: pip install ultralytics` se o pacote faltar. Em um ambiente com contrib, não siga essa instrução ao pé da letra: use o `requirements_ultralytics.txt` com `--no-deps`.

Pasta de execução: todos os scripts definem `ROOT = Path(__file__).resolve().parent`, ou seja, a raiz dos caminhos é a própria pasta dos scripts, `aula_13_YOLO_SSD/exemplos_em_python/`. As subpastas são `dados/` (vídeos e imagens), `saidas/` (resultados), `modelos/` (pesos) e `relatorios/` (CSV e gráficos). O `01_ambiente_e_pastas.py` cria as quatro. No repositório, `saidas/` e `relatorios/` já existem com um `.gitkeep`; `dados/` e `modelos/` ficam fora do git porque guardam vídeos e pesos grandes. Como `ROOT` vem de `__file__`, os scripts podem ser chamados de qualquer diretório, mas o `yolov8n.pt` baixado pelo Ultralytics cai no diretório de trabalho atual, então a recomendação é rodar sempre de dentro de `exemplos_em_python/`.

Material faltante, segundo `achados_dos_zips.md`: nada para os scripts sintéticos. Para os scripts de modelos reais, o aluno precisa providenciar:

- `dados/imagem_teste.jpg` (script 11 e 13) e `dados/video_real.mp4` (scripts 12 e 29): qualquer foto ou vídeo com pessoas, carros ou objetos das 80 classes do COCO.
- `yolov8n.pt`: baixado automaticamente pelo Ultralytics na primeira chamada de `YOLO("yolov8n.pt")`.
- `modelos/yolov8n.onnx` (script 13): exportado com `yolo export model=yolov8n.pt format=onnx imgsz=640`, que exige o extra de exportação do Ultralytics (`onnx`, `onnxslim`).
- `modelos/yolov4-tiny.cfg`, `modelos/yolov4-tiny.weights` e `modelos/coco.names` (script 14): do repositório darknet de AlexeyAB.
- `modelos/frozen_inference_graph.pb` e `modelos/ssd_mobilenet_v2_coco.pbtxt` (script 15): o `.pb` vem do TensorFlow model zoo (SSD MobileNet v2 COCO 2018_03_29) e o `.pbtxt` é gerado pelo `tf_text_graph_ssd.py` do OpenCV ou obtido do repositório opencv_extra.

Webcam: nenhum script abre a câmera. Tudo é vídeo sintético ou arquivo.

## 3. Conceitos-chave

### 3.1 Estrutura de pastas com `pathlib`

Cada script começa com `ROOT = Path(__file__).resolve().parent`, o que ancora todos os caminhos na pasta do próprio script, independentemente de onde o Python foi chamado. A função `ensure_dirs` cria as quatro subpastas com `mkdir(exist_ok=True)`, que não reclama se a pasta já existe. É a primeira coisa a rodar, porque os scripts 04, 10, 24 e 28 gravam em `saidas/` e o 33 em `relatorios/` sem criar a pasta.

```python
# vision_utils.py
def ensure_dirs(root):
    root = Path(root)
    for name in ["dados", "saidas", "modelos", "relatorios"]:
        (root / name).mkdir(exist_ok=True)
```

```python
# 01_ambiente_e_pastas.py
ROOT = Path(__file__).resolve().parent
ensure_dirs(ROOT)

print("Estrutura preparada:")
for pasta in ["dados", "saidas", "modelos", "relatorios"]:
    print(" -", ROOT / pasta)
```

Onde aparece: `vision_utils.py`, `01_ambiente_e_pastas.py`, e a linha `ROOT = ...` em quase todos os scripts.

### 3.2 Objetos sintéticos com caixas conhecidas

A base de toda a aula é uma cena em que o professor sabe exatamente onde cada objeto está. `synthetic_objects(frame_idx)` devolve, para um índice de frame, uma lista de dicionários com classe, confiança e caixa `[x1, y1, x2, y2]`. A pessoa anda da esquerda para a direita com um balanço senoidal, o carro anda da direita para a esquerda, a bicicleta desce em diagonal, e o ônibus só entra depois do frame 35. No final, as caixas são recortadas para caber em `VIDEO_SIZE` e as que saíram da tela são descartadas. Isso é o "gabarito" (ground truth) que qualquer detector real não tem.

```python
# vision_utils.py
def synthetic_objects(frame_idx):
    """Retorna objetos sintéticos no formato dict com bbox, classe e confiança."""
    objs = []
    # pessoa andando da esquerda para a direita
    x = 40 + frame_idx * 5
    y = 280 + int(15 * np.sin(frame_idx / 8))
    objs.append({"cls": "person", "conf": 0.82, "bbox": [x, y, x + 48, y + 110]})

    # carro andando da direita para a esquerda
    x2 = 850 - frame_idx * 4
    y2 = 345
    objs.append({"cls": "car", "conf": 0.88, "bbox": [x2, y2, x2 + 120, y2 + 60]})
    ...
    # ônibus aparece depois de alguns frames
    if frame_idx > 35:
        x4 = 720 - (frame_idx - 35) * 2
        y4 = 165
        objs.append({"cls": "bus", "conf": 0.79, "bbox": [x4, y4, x4 + 150, y4 + 85]})
```

Onde aparece: `vision_utils.py`, chamado por `draw_scene`, `fake_detector` e diretamente em `04_desenhar_bbox.py`.

### 3.3 Desenho da rua sintética

`draw_scene(frame_idx)` monta um frame BGR `uint8` de 540 por 960 com `np.full`, pinta céu, pista e calçadas com `cv2.rectangle`, desenha faixas que se deslocam com `frame_idx*3 % 160` para dar sensação de movimento, prédios com janelas, e por fim cada objeto de `synthetic_objects` como formas simples (círculo mais retângulo para a pessoa, retângulos e rodas para carro e ônibus, dois aros e um quadro para a bicicleta). As cores por classe ficam no dicionário `COLORS`. É a mesma ideia do vídeo sintético da aula 10, só que com objetos rotulados.

```python
# vision_utils.py
VIDEO_SIZE = (960, 540)
FPS_VIDEO = 20
...
def draw_scene(frame_idx):
    """Desenha uma rua sintética com objetos em movimento."""
    w, h = VIDEO_SIZE
    frame = np.full((h, w, 3), (35, 40, 45), dtype=np.uint8)

    # céu / fundo
    cv2.rectangle(frame, (0, 0), (w, 150), (80, 100, 120), -1)
    # pista
    cv2.rectangle(frame, (0, 150), (w, h), (55, 55, 55), -1)
    ...
    # faixas
    for x in range(-100, w + 100, 160):
        cv2.line(frame, (x + frame_idx*3 % 160, 300), (x + 70 + frame_idx*3 % 160, 300), (220, 220, 220), 4)
```

Onde aparece: `vision_utils.py`, usado por `create_synthetic_video` e pelos scripts 04, 05, 10, 19, 24, 28, 35.

### 3.4 Gravação de vídeo com `cv2.VideoWriter` e codec MJPG

`create_synthetic_video` grava `n_frames` frames em um arquivo `.avi` com o codec MJPG (Motion JPEG, cada frame é um JPEG). O `VideoWriter` recebe caminho, FOURCC, FPS (20) e tamanho `(largura, altura)`. A pasta pai é criada com `mkdir(exist_ok=True)`. MJPG em AVI é a combinação mais portátil no Windows sem codecs extras; o preço é arquivo grande, que o desafio do script 02 pede para medir.

```python
# vision_utils.py
def create_synthetic_video(path, n_frames=120):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, FPS_VIDEO, VIDEO_SIZE)
    for i in range(n_frames):
        writer.write(draw_scene(i))
    writer.release()
    return path
```

Onde aparece: `vision_utils.py`, `02_criar_video_sintetico.py` (120 frames), `03_ler_frames_video.py` (30 frames, no mesmo arquivo), notebook. Os scripts 10, 24 e 28 usam `cv2.VideoWriter` diretamente para gravar vídeos anotados em `saidas/`.

### 3.5 Leitura de frames com `cv2.VideoCapture`

O laço padrão de leitura é o mesmo da aula 10: `ok, frame = cap.read()` até `ok` ser falso, e `cap.release()` no fim. O script 03 conta os frames e imprime o `shape` de cada um, `(540, 960, 3)`. O script 12 e o 29 usam o mesmo laço sobre um vídeo real, sem janela: só medem tempo.

```python
# 03_ler_frames_video.py
cap = cv2.VideoCapture(str(video))
contador = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    contador += 1
    print("Frame", contador, "-> shape:", frame.shape)

cap.release()
print("Total de frames lidos:", contador)
```

Onde aparece: `03_ler_frames_video.py`, `12_yolov8_ultralytics_video.py`, `29_pipeline_video_real_com_yolo_template.py`.

### 3.6 Formato de uma detecção e conversão de caixas

A aula fixa um contrato: uma detecção é um dicionário `{"cls": str, "conf": float, "bbox": [x1, y1, x2, y2]}`. Tudo que produz detecções (o simulador, o YOLO no script 29) e tudo que consome (desenho, NMS, rastreador) fala esse formato, e o notebook chama isso de "ponte para modelos reais". Como o `cv2.dnn.NMSBoxes` espera caixas em `[x, y, w, h]`, há duas funções de conversão. O script 09 mostra como formatar o rótulo com porcentagem.

```python
# vision_utils.py
def bbox_to_xywh(box):
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2-x1), int(y2-y1)]


def xywh_to_bbox(box):
    x, y, w, h = box
    return [int(x), int(y), int(x+w), int(y+h)]
```

```python
# 09_formatar_rotulos_confianca.py
for d in fake_detector(15, duplicate=False):
    label = f"{d['cls']} | {d['conf']*100:.1f}%"
    print(label)
```

Onde aparece: `vision_utils.py`, `06_deteccoes_sinteticas_por_frame.py` (imprime os dicionários crus), `09_formatar_rotulos_confianca.py`, `29_pipeline_video_real_com_yolo_template.py` (monta o dicionário a partir do YOLO).

### 3.7 Desenho de caixas, rótulos e trilhas

`draw_detections` trabalha em uma cópia do frame, desenha cada caixa com `cv2.rectangle` na cor da classe e escreve `classe confiança` com `cv2.putText` logo acima, usando `max(20, y1-8)` para o texto não sair pela borda superior. Com `show_id=True`, o rótulo ganha o prefixo `ID N`. Se receber o dicionário `tracks` (as trilhas do rastreador), liga os centros consecutivos com `cv2.line` em amarelo.

```python
# vision_utils.py
def draw_detections(frame, detections, show_id=False, tracks=None):
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"])
        cls = d.get("cls", "obj")
        conf = d.get("conf", 0.0)
        color = COLORS.get(cls, (0, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        if show_id and "id" in d:
            label = f"ID {d['id']} | " + label
        cv2.putText(out, label, (x1, max(20, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    if tracks:
        for tid, pts in tracks.items():
            pts = list(pts)
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 255), 2)
    return out
```

Onde aparece: `vision_utils.py`, scripts 04, 10, 24, 28, 29, notebook.

### 3.8 FPS e latência com `time.perf_counter`

Detecção em tempo real é um problema de orçamento de tempo, então a aula mede desde o início. Latência é o tempo de processar um frame, em milissegundos; FPS é quantos frames cabem em um segundo. O script 05 mede os dois de formas independentes: a latência por frame com um cronômetro em volta de cada iteração, e o FPS como número de frames dividido pelo tempo total. Os scripts 12, 19, 28 e 29 fazem a conta `1000 / latência_média` para estimar o FPS.

```python
# 05_medicao_fps_latencia.py
n_frames = 80
t0 = time.perf_counter()
latencias = []

for i in range(n_frames):
    inicio = time.perf_counter()
    frame = draw_scene(i)
    detections = fake_detector(i)
    fim = time.perf_counter()
    latencias.append((fim - inicio) * 1000)

total = time.perf_counter() - t0
fps = n_frames / total

print(f"FPS médio: {fps:.2f}")
print(f"Latência média por frame: {sum(latencias)/len(latencias):.2f} ms")
```

Onde aparece: `05_medicao_fps_latencia.py`, `12_yolov8_ultralytics_video.py`, `19_benchmark_yolo_ssd_template.py`, `28_pipeline_yolo_tracking_sintetico.py`, `29_pipeline_video_real_com_yolo_template.py`. O `vision_utils.py` define uma classe `Timer` com o mesmo propósito, mas nenhum script a usa.

### 3.9 Detector simulado com ruído e duplicatas

Um detector real nunca devolve a caixa exata: ela treme de frame em frame e, muitas vezes, o mesmo objeto sai em duas ou três caixas quase iguais. `fake_detector` imita isso: para cada objeto do gabarito, soma um ruído inteiro em `[-jitter, jitter]` a cada coordenada (gerador com semente `1000 + frame_idx`, então o resultado é reproduzível) e, se `duplicate=True`, acrescenta uma segunda caixa deslocada 5 pixels em x e 3 em y com confiança 0,10 menor. Essa duplicata é o que dá trabalho ao NMS.

```python
# vision_utils.py
def fake_detector(frame_idx, duplicate=True, jitter=3):
    """Simula saídas de detector com pequenas variações e caixas duplicadas."""
    rng = np.random.default_rng(seed=1000 + frame_idx)
    dets = []
    for o in synthetic_objects(frame_idx):
        x1, y1, x2, y2 = o["bbox"]
        noise = rng.integers(-jitter, jitter+1, size=4)
        b = [x1+int(noise[0]), y1+int(noise[1]), x2+int(noise[2]), y2+int(noise[3])]
        dets.append({"cls": o["cls"], "conf": float(o["conf"]), "bbox": b})
        if duplicate:
            b2 = [b[0]+5, b[1]+3, b[2]+5, b[3]+3]
            dets.append({"cls": o["cls"], "conf": float(o["conf"] - 0.10), "bbox": b2})
    return dets
```

Onde aparece: `vision_utils.py`, scripts 05 a 10, 19, 23 a 26, 28, 34, notebook.

### 3.10 Non-Maximum Suppression com `cv2.dnn.NMSBoxes`

NMS é o algoritmo que elimina caixas redundantes: ordena por confiança, guarda a melhor, descarta todas as que tenham IoU acima de um limiar com ela, e repete com a próxima que sobrou. O OpenCV implementa em `cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)`, que recebe caixas em `[x, y, w, h]`, descarta antes as que têm confiança abaixo de `score_threshold`, e devolve os índices das caixas mantidas. `apply_nms` converte as caixas, chama a função e reconstrói a lista de dicionários. O `np.array(idxs).flatten()` existe porque versões antigas do OpenCV devolviam os índices em uma matriz coluna.

Com `nms_thr=0.40`, as duplicatas do simulador (IoU em torno de 0,8 com a original) somem. Com `nms_thr=0.80`, o limiar fica próximo do IoU das duplicatas e algumas sobrevivem, principalmente a da pessoa, que é uma caixa estreita e por isso perde mais IoU com o mesmo deslocamento. É isso que o script 34 mostra ao varrer 0,1 a 0,8.

```python
# vision_utils.py
def apply_nms(detections, score_thr=0.25, nms_thr=0.40):
    boxes = [bbox_to_xywh(d["bbox"]) for d in detections]
    scores = [float(d["conf"]) for d in detections]
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_thr, nms_thr)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).flatten().tolist()
    return [detections[i] for i in idxs]
```

```python
# 34_testar_thresholds_nms.py
frame_idx = 30
for thr in [0.1, 0.3, 0.4, 0.6, 0.8]:
    dets = fake_detector(frame_idx, duplicate=True)
    filtradas = apply_nms(dets, score_thr=0.25, nms_thr=thr)
    print(f"NMS threshold={thr:.1f}: {len(filtradas)} caixas")
```

Onde aparece: `vision_utils.py`, `07_aplicar_nms_opencv.py`, `10_salvar_video_anotado_sintetico.py`, scripts 23 a 26, 28, `34_testar_thresholds_nms.py`, notebook.

### 3.11 Filtro por confiança

Antes ou junto do NMS, descarta-se o que o modelo não tem certeza. O script 08 faz isso com uma list comprehension e limiar 0,80. No frame 42 o simulador devolve oito caixas (quatro objetos, cada um com duplicata); só a pessoa (0,82) e o carro (0,88) passam. O ônibus, com 0,79, fica fora por um centésimo, e é um bom exemplo de como um limiar arbitrário derruba um objeto real. A confiança também entra no YOLO como `conf=0.25` e no `apply_nms` como `score_thr=0.25`.

```python
# 08_filtrar_por_confianca.py
threshold = 0.80
detections = fake_detector(42, duplicate=True)
filtered = [d for d in detections if d["conf"] >= threshold]

print("Threshold de confiança:", threshold)
print("Detecções originais:", len(detections))
print("Detecções filtradas:", len(filtered))
```

Onde aparece: `08_filtrar_por_confianca.py`; como parâmetro em `apply_nms` (`vision_utils.py`) e em `model(..., conf=0.25, iou=0.40)` nos scripts 11, 12 e 29.

### 3.12 YOLOv8 pela biblioteca Ultralytics

YOLO trata detecção como uma regressão única: a imagem passa uma vez pela rede e sai uma grade de caixas com classe e confiança. A biblioteca `ultralytics` esconde todo o pré e pós-processamento. `YOLO("yolov8n.pt")` carrega o modelo nano (o menor da família, 3,2 milhões de parâmetros) e baixa o arquivo na primeira vez; `model(imagem, conf=0.25, iou=0.40)` roda a inferência já com filtro de confiança e NMS (o `iou` é o limiar do NMS); `results[0].plot()` devolve o frame anotado como array BGR. Para vídeo, `verbose=False` silencia a linha que a biblioteca imprime por frame. O script 29 extrai as caixas com `b.xyxy[0]`, a classe com `b.cls[0]` e o nome com `model.names[cls_id]`, e converte para o dicionário da aula.

```python
# 11_yolov8_ultralytics_imagem.py
try:
    from ultralytics import YOLO
except ImportError:
    print("Instale com: pip install ultralytics")
    raise SystemExit
...
model = YOLO("yolov8n.pt")
results = model(str(imagem), conf=0.25, iou=0.40)
annotated = results[0].plot()
```

```python
# 29_pipeline_video_real_com_yolo_template.py
    results = model(frame, conf=0.25, iou=0.40, verbose=False)[0]
    dets = []
    for b in results.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
        cls_id = int(b.cls[0])
        dets.append({"cls": model.names[cls_id], "conf": float(b.conf[0]), "bbox": [x1,y1,x2,y2]})
    tracked = tracker.update(dets)
```

Onde aparece: `11_yolov8_ultralytics_imagem.py`, `12_yolov8_ultralytics_video.py`, `29_pipeline_video_real_com_yolo_template.py`.

### 3.13 YOLOv8 exportado para ONNX no OpenCV DNN

Em sistemas embarcados, muitas vezes não se quer o PyTorch inteiro. O caminho é exportar o modelo para ONNX (`yolo export model=yolov8n.pt format=onnx imgsz=640`) e carregar com `cv2.dnn.readNetFromONNX`. A entrada é montada com `cv2.dnn.blobFromImage`: escala `1/255` para levar os pixels a `[0, 1]`, tamanho 640 por 640, `swapRB=True` porque o YOLO foi treinado em RGB e o OpenCV lê BGR, e `crop=False`. A saída bruta de um YOLOv8n tem forma `(1, 84, 8400)`: 8400 candidatos, cada um com 4 coordenadas e 80 pontuações de classe. Decodificar isso (transpor, pegar a classe de maior pontuação, converter centro e tamanho em cantos, aplicar NMS) é o desafio do script; o professor só imprime a forma.

```python
# 13_yolov8_opencv_onnx.py
net = cv2.dnn.readNetFromONNX(str(onnx))
img = cv2.imread(str(img_path))
blob = cv2.dnn.blobFromImage(img, 1/255.0, (640,640), swapRB=True, crop=False)
net.setInput(blob)
out = net.forward()

print("Saída bruta do modelo:", out.shape)
print("A decodificação pode variar conforme a exportação do YOLO.")
```

Onde aparece: `13_yolov8_opencv_onnx.py`.

### 3.14 YOLOv4-tiny em Darknet

A geração anterior do YOLO é distribuída no formato Darknet: um `.cfg` com a arquitetura e um `.weights` com os pesos, mais um `coco.names` com os 80 nomes de classe, um por linha. `cv2.dnn.readNetFromDarknet(cfg, weights)` monta a rede, e `net.getUnconnectedOutLayersNames()` lista as camadas de saída (o YOLOv4-tiny tem duas, para objetos de escalas diferentes), que são os nomes a passar em `net.forward(...)`. O script só carrega e imprime; o forward com blob 416 por 416 fica como desafio.

```python
# 14_yolov4_tiny_opencv_dnn.py
classes = names.read_text(encoding="utf-8").strip().splitlines()
net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))

print("Modelo YOLOv4-tiny carregado.")
print("Número de classes:", len(classes))
print("Camadas de saída:", net.getUnconnectedOutLayersNames())
```

Onde aparece: `14_yolov4_tiny_opencv_dnn.py`.

### 3.15 SSD MobileNet em TensorFlow

SSD é a outra família de detectores de um estágio. Prediz caixas a partir de várias camadas de resolução diferente, usando caixas-âncora, e o MobileNet como tronco o deixa leve. O formato clássico é um grafo congelado do TensorFlow 1 (`frozen_inference_graph.pb`) mais um `.pbtxt` que descreve o grafo para o OpenCV. `cv2.dnn.readNetFromTensorflow(pb, pbtxt)` carrega. A saída, depois de um `forward`, é um tensor `(1, 1, N, 7)` com `[batch, classe, confiança, x1, y1, x2, y2]` normalizados; montar o blob e rodar é o desafio.

```python
# 15_ssd_mobilenet_opencv_dnn.py
pb = ROOT / "modelos" / "frozen_inference_graph.pb"
pbtxt = ROOT / "modelos" / "ssd_mobilenet_v2_coco.pbtxt"

if not (pb.exists() and pbtxt.exists()):
    print("Coloque os arquivos do SSD MobileNet em modelos/.")
    raise SystemExit

net = cv2.dnn.readNetFromTensorflow(str(pb), str(pbtxt))
print("SSD MobileNet carregado com OpenCV DNN.")
```

Onde aparece: `15_ssd_mobilenet_opencv_dnn.py`.

### 3.16 Tamanho em disco, parâmetros e tabelas comparativas

Comparar modelos exige mais que precisão. O script 16 mede o tamanho de cada arquivo com `Path.stat().st_size` e converte para MB. O 17 registra o número de parâmetros em milhões em uma tabela manual, porque o OpenCV DNN não expõe essa contagem para todo formato; só o YOLOv8n vem preenchido (3,2 M) e o resto é `None` para o aluno completar. O 18 e o 30 montam um `pandas.DataFrame` com colunas de FPS, latência, parâmetros, tamanho e ID switches por minuto, todas zeradas ou `None`: são templates. O 30 ainda enuncia a regra de comparação justa: mesmo vídeo, mesma resolução de entrada, mesma máquina, mesmo critério de confiança e NMS.

```python
# 16_comparar_tamanho_arquivos.py
for m in modelos:
    if m.exists():
        mb = m.stat().st_size / (1024*1024)
        print(f"{m.name}: {mb:.2f} MB")
    else:
        print(f"{m.name}: arquivo não encontrado")
```

```python
# 18_tabela_comparativa_modelos.py
linhas = [
    {"modelo": "YOLOv8n", "fps": 0.0, "latencia_ms": 0.0, "parametros_M": 3.2, "tamanho_MB": 0.0},
    {"modelo": "SSD MobileNetV2", "fps": 0.0, "latencia_ms": 0.0, "parametros_M": None, "tamanho_MB": 0.0},
]

df = pd.DataFrame(linhas)
print(df.to_string(index=False))
```

Onde aparece: `16_comparar_tamanho_arquivos.py`, `17_estimar_parametros_manual.py`, `18_tabela_comparativa_modelos.py`, `30_comparar_yolo_ssd_tracking_template.py`.

### 3.17 Template de benchmark e conclusão para embarcado

O script 19 isola a inferência em uma função `inferencia_simulada(frame, idx)` que devolve `fake_detector(idx)`; o cronômetro envolve só ela, sem contar o desenho do frame. A ideia é trocar o corpo por `model(frame)` ou `net.forward()` e reaproveitar o resto. O script 20 mostra como transformar a tabela em decisão: `max` por FPS e `min` por latência, mais um texto de conclusão. Os números 24,5 FPS e 31,2 FPS que aparecem ali (e no gráfico do script 33) são exemplos fixos, não medições.

```python
# 19_benchmark_yolo_ssd_template.py
def inferencia_simulada(frame, idx):
    # Troque esta função por: model(frame) ou net.forward().
    return fake_detector(idx)

latencias = []
frames = 100

for i in range(frames):
    frame = draw_scene(i)
    t0 = time.perf_counter()
    detections = inferencia_simulada(frame, i)
    latencias.append((time.perf_counter() - t0) * 1000)

media = sum(latencias) / len(latencias)
print(f"Latência média: {media:.3f} ms")
print(f"FPS estimado: {1000/media:.2f}")
```

```python
# 20_concluir_modelo_embarcado.py
melhor_fps = max(resultados, key=lambda x: x["fps"])
menor_lat = min(resultados, key=lambda x: x["latencia_ms"])

print("Modelo com maior FPS:", melhor_fps["modelo"])
print("Modelo com menor latência:", menor_lat["modelo"])
```

Onde aparece: `19_benchmark_yolo_ssd_template.py`, `20_concluir_modelo_embarcado.py`.

### 3.18 IoU entre duas caixas

IoU (Intersection over Union) é a medida de sobreposição entre duas caixas: área da interseção dividida pela área da união. Vale 1 para caixas idênticas, 0 para caixas disjuntas. A interseção é o retângulo entre o maior dos cantos superiores esquerdos e o menor dos cantos inferiores direitos; se a largura ou a altura sair negativa, `max(0, ...)` zera. A união é a soma das áreas menos a interseção, para não contar duas vezes. No script 21, A e B se sobrepõem em 70 por 70 pixels: IoU de 4900 sobre 21600, cerca de 0,227. A e C não se tocam: 0,0. A mesma medida serve ao NMS (redundância) e ao rastreamento (continuidade).

```python
# vision_utils.py
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter
    return 0.0 if union == 0 else inter / union
```

Onde aparece: `vision_utils.py`, `21_iou_basico.py`, `22_associacao_por_iou.py`, e dentro de `IoUTracker.update`.

### 3.19 Associação entre frames por IoU

Se um objeto se move pouco entre dois frames consecutivos, a caixa nova se sobrepõe muito à caixa antiga. Essa é a hipótese do rastreamento por IoU: calcular a IoU de cada track do frame anterior com cada detecção do frame atual e associar os pares de maior sobreposição. O script 22 faz a conta na mão com dois tracks e duas detecções: o track 1 tem IoU de cerca de 0,80 com a detecção 0 e 0 com a detecção 1; o track 2, cerca de 0,83 com a detecção 1. A associação é evidente. Quando dois objetos se cruzam, as IoUs cruzadas sobem e a associação pode trocar, o que é a origem dos ID switches.

```python
# 22_associacao_por_iou.py
tracks_anteriores = {
    1: [100, 100, 200, 220],
    2: [400, 160, 520, 230],
}

deteccoes_atuais = [
    [108, 104, 208, 224],
    [392, 162, 512, 232],
]

for tid, box_track in tracks_anteriores.items():
    print("\nTrack ID", tid)
    for i, box_det in enumerate(deteccoes_atuais):
        print(f"  IoU com detecção {i}:", iou(box_track, box_det))
```

Onde aparece: `22_associacao_por_iou.py`.

### 3.20 Rastreador com IDs persistentes: `IoUTracker`

A classe `IoUTracker` automatiza a associação. Em `update(detections)`, ela monta todos os pares (IoU, track, detecção), ordena por IoU decrescente e percorre de forma gulosa: aceita o par se a IoU está acima de `iou_thr` e nem o track nem a detecção já foram usados. Detecções sem par viram tracks novos com `next_id`, e `total_created` conta quantos IDs já nasceram. Tracks sem par ganham um `missing` a mais e são apagados quando passam de `max_missing`; enquanto isso, continuam na saída com a última caixa conhecida. A associação não olha a classe: um track de carro pode ser continuado por uma detecção de ônibus se a IoU for alta, e a classe do track é sobrescrita.

```python
# vision_utils.py
        track_ids = list(self.tracks.keys())
        pairs = []
        for tid in track_ids:
            for di, det in enumerate(detections):
                pairs.append((iou(self.tracks[tid]["bbox"], det["bbox"]), tid, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for score, tid, di in pairs:
            if score < self.iou_thr:
                continue
            if tid in assigned_tracks or di in assigned_dets:
                continue
            self.tracks[tid].update({"bbox": detections[di]["bbox"], "cls": detections[di]["cls"], "conf": detections[di]["conf"], "missing": 0})
            assigned_tracks.add(tid)
            assigned_dets.add(di)

        for di, det in enumerate(detections):
            if di not in assigned_dets:
                tid = self.next_id
                self.next_id += 1
                self.total_created += 1
                self.tracks[tid] = {"bbox": det["bbox"], "cls": det["cls"], "conf": det["conf"], "missing": 0}
                self.trails[tid] = deque(maxlen=self.trail_len)
                assigned_tracks.add(tid)
```

```python
# 23_tracker_iou_ids_persistentes.py
tracker = IoUTracker(iou_thr=0.30)

for frame_idx in range(0, 15):
    dets = apply_nms(fake_detector(frame_idx), 0.25, 0.40)
    tracked = tracker.update(dets)
    print("\nFrame", frame_idx)
    for t in tracked:
        print("ID", t["id"], t["cls"], t["bbox"])
```

Onde aparece: `vision_utils.py`, scripts 23, 24, 25, 26, 28, 29, notebook.

### 3.21 Trilhas com `deque` de tamanho fixo

Cada track tem uma trilha: um `collections.deque(maxlen=trail_len)` com os centros das caixas dos últimos frames. Quando o deque enche, o ponto mais antigo sai sozinho, então a trilha tem sempre no máximo 30 pontos (padrão) e some aos poucos atrás do objeto. O centro é anexado só nos frames em que o track foi associado; em frames perdidos, a trilha congela. `draw_detections` recebe `tracker.trails` e liga os pontos.

```python
# vision_utils.py
            else:
                x1, y1, x2, y2 = self.tracks[tid]["bbox"]
                center = ((x1+x2)//2, (y1+y2)//2)
                self.trails.setdefault(tid, deque(maxlen=self.trail_len)).append(center)
```

```python
# 24_trilhas_ultimos_30_frames.py
tracker = IoUTracker(iou_thr=0.30, trail_len=30)

for i in range(100):
    frame = draw_scene(i)
    dets = apply_nms(fake_detector(i), 0.25, 0.40)
    tracked = tracker.update(dets)
    out = draw_detections(frame, tracked, show_id=True, tracks=tracker.trails)
    writer.write(out)
```

Onde aparece: `vision_utils.py`, `24_trilhas_ultimos_30_frames.py`, `28_pipeline_yolo_tracking_sintetico.py`, `29_pipeline_video_real_com_yolo_template.py`.

### 3.22 Contagem de entradas e saídas por linha virtual

Com IDs persistentes, contar vira simples: para cada track, calcular o centro em x, classificar em `esq` ou `dir` em relação a uma linha vertical em `x=480` (o meio do frame), e comparar com o lado registrado no frame anterior para aquele ID. Mudou de `esq` para `dir` é entrada; o contrário é saída. O dicionário `lado_anterior` é indexado pelo ID, por isso a contagem depende da identidade ser estável: se o ID troca no meio do cruzamento, o cruzamento não é contado, ou é contado duas vezes.

```python
# 25_entrada_saida_linha_virtual.py
linha_x = 480
lado_anterior = {}
entradas = 0
saidas = 0

for i in range(100):
    dets = apply_nms(fake_detector(i), 0.25, 0.40)
    tracked = tracker.update(dets)

    for t in tracked:
        x1, y1, x2, y2 = t["bbox"]
        cx = (x1 + x2) // 2
        lado = "esq" if cx < linha_x else "dir"
        tid = t["id"]
        if tid in lado_anterior and lado_anterior[tid] != lado:
            if lado == "dir":
                entradas += 1
            else:
                saidas += 1
        lado_anterior[tid] = lado
```

Onde aparece: `25_entrada_saida_linha_virtual.py`.

### 3.23 ID switches por minuto

ID switch é quando o mesmo objeto físico recebe um ID novo ou troca de ID com outro. Medir de verdade exige gabarito de identidade. O professor usa uma aproximação: com quatro objetos na cena, todo ID criado além de quatro é um switch. Para provocar o problema, o script 26 aperta o rastreador (`iou_thr=0.65`, `max_missing=1`) e aumenta o tremor do detector (`jitter=8`); a IoU entre frames cai abaixo do limiar, o track morre e renasce com ID novo. A taxa é normalizada por minuto de vídeo: 120 frames a 20 FPS são 6 segundos, 0,1 minuto.

```python
# 26_id_switches_por_minuto.py
tracker = IoUTracker(iou_thr=0.65, max_missing=1)
frames = 120

for i in range(frames):
    dets = apply_nms(fake_detector(i, jitter=8), 0.25, 0.40)
    tracker.update(dets)

duracao_min = frames / FPS_VIDEO / 60
switches_aprox = max(0, tracker.total_created - 4)
taxa = switches_aprox / duracao_min
```

Onde aparece: `26_id_switches_por_minuto.py`; a coluna `idswitch_min` e `id_switches_min` nos templates 30 e 32.

### 3.24 Rastreadores do OpenCV: CSRT, KCF e MIL

O OpenCV traz rastreadores de objeto único que, uma vez inicializados com uma caixa, seguem a região por aparência (CSRT usa filtros de correlação com máscara espacial, KCF filtros de correlação kernelizados, MIL aprendizado de múltiplas instâncias). O script 27 é apenas conceitual: imprime as três chamadas. O professor observa que esses rastreadores não substituem um detector quando objetos novos entram na cena o tempo todo, porque não sabem que um objeto apareceu. No OpenCV 4.14 contrib do projeto, `cv2.TrackerCSRT_create` e `cv2.legacy.TrackerCSRT_create` existem os dois.

```python
# 27_tracker_csrt_opcional.py
print("Exemplo conceitual de inicialização CSRT:")
print("tracker = cv2.legacy.TrackerCSRT_create()  # em algumas versões")
print("tracker.init(frame, bbox)")
print("ok, bbox = tracker.update(frame)")
```

Onde aparece: `27_tracker_csrt_opcional.py`.

### 3.25 Pipeline integrado: detecção, NMS, rastreamento, desenho

O script 28 junta tudo em um laço de 120 frames: desenhar (ou ler) o frame, detectar, aplicar NMS, atualizar o rastreador, desenhar caixas com ID e trilhas, gravar. O cronômetro envolve o laço inteiro, então a latência inclui o desenho da cena e a gravação. Os comentários indicam exatamente o ponto de troca: `fake_detector(i)` vira a chamada do YOLO. O script 29 é a mesma estrutura com o YOLOv8n de verdade sobre `dados/video_real.mp4`, sem gravar nem exibir.

```python
# 28_pipeline_yolo_tracking_sintetico.py
for i in range(120):
    t0 = time.perf_counter()
    frame = draw_scene(i)
    dets = fake_detector(i)                 # substituível por YOLO real
    dets = apply_nms(dets, 0.25, 0.40)       # NMS com threshold 0.4
    tracked = tracker.update(dets)           # IDs persistentes
    out = draw_detections(frame, tracked, show_id=True, tracks=tracker.trails)
    latencias.append((time.perf_counter() - t0) * 1000)
    writer.write(out)

writer.release()
print("Vídeo final salvo em:", saida)
print(f"Latência total média: {sum(latencias)/len(latencias):.2f} ms")
```

Onde aparece: `28_pipeline_yolo_tracking_sintetico.py`, `29_pipeline_video_real_com_yolo_template.py`, notebook.

### 3.26 Registro em CSV e gráfico de barras

Para relatórios, `write_csv` usa `csv.DictWriter` com a lista de cabeçalhos, cria a pasta pai e grava uma linha por dicionário. O script 32 gera `relatorios/metricas_modelos.csv` com zeros para o aluno preencher. O 33 faz um gráfico de barras de FPS com Matplotlib puro (`plt.bar`, `plt.savefig`), sem Seaborn, como o nome do script anuncia, e salva PNG em `relatorios/` sem criar a pasta.

```python
# vision_utils.py
def write_csv(path, rows, headers):
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
```

```python
# 33_plotar_fps_sem_seaborn.py
plt.figure(figsize=(7,4))
plt.bar(modelos, fps)
plt.ylabel("FPS médio")
plt.title("Comparação de FPS")
plt.tight_layout()

saida = ROOT / "relatorios" / "33_grafico_fps.png"
plt.savefig(saida, dpi=150)
```

Onde aparece: `vision_utils.py`, `32_gerar_csv_metricas.py`, `33_plotar_fps_sem_seaborn.py`.

### 3.27 Resolução de entrada do detector

Todo detector redimensiona o frame para um tamanho fixo antes da rede (320, 416 ou 640 pixels são os valores usuais). Reduzir a entrada aumenta o FPS quase proporcionalmente à área, mas objetos pequenos viram poucos pixels e somem. O script 35 mostra só a parte do `cv2.resize`; medir o tempo de cada tamanho é o desafio. O `imgsz=640` da exportação ONNX e o `(640,640)` do `blobFromImage` no script 13 são esse mesmo parâmetro.

```python
# 35_experimento_resolucao_entrada.py
frame = draw_scene(20)
for size in [(320, 320), (416, 416), (640, 640)]:
    resized = cv2.resize(frame, size)
    print("Entrada para o modelo:", resized.shape)
```

Onde aparece: `35_experimento_resolucao_entrada.py`, `13_yolov8_opencv_onnx.py`.

### 3.28 Técnica e ética em contagem por drone

O script 31 é um texto, não código: argumenta que contar pedestres por drone com detecção e rastreamento exige finalidade clara, minimização de dados, sinalização, proteção contra identificação indevida, auditoria de vieses, limitação de retenção e avaliação de impacto. Contagem agregada é menos invasiva que identificação individual, mas o vídeo bruto ainda expõe pessoas, rotinas e locais. A recomendação prática é processar localmente, descartar frames após a contagem e reportar só estatísticas. É o fecho do quarto momento do `GUIA_PROFESSOR.md`, junto com as perguntas sobre falso positivo, falso negativo e ID switches.

```python
# 31_relatorio_etica_contagem_drone.py
texto = """
Discussão técnica e ética para contagem por drone:

Um sistema de detecção e rastreamento pode estimar fluxo de pedestres ao detectar
pessoas, atribuir IDs persistentes e contar cruzamentos em regiões de interesse.
Em vigilância urbana, esse uso exige finalidade clara, minimização de dados,
sinalização quando aplicável, proteção contra identificação indevida, auditoria
de vieses, limitação de retenção das imagens e avaliação de impacto.
...
"""

print(texto)
```

Onde aparece: `31_relatorio_etica_contagem_drone.py`, `GUIA_PROFESSOR.md`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `vision_utils.py` | Cena sintética, vídeo, detector simulado, NMS, IoU, `IoUTracker`, desenho, CSV, `Timer` | importado | nada | cv2, numpy |
| `GUIA_PROFESSOR.md` | Roteiro do professor em quatro momentos e oito perguntas | leitura | nada | nada |
| `01_ambiente_e_pastas.py` | Cria `dados/`, `saidas/`, `modelos/`, `relatorios/` | nada | quatro pastas | `vision_utils.py` |
| `02_criar_video_sintetico.py` | Grava a rua sintética com 120 frames | nada | `dados/rua_sintetica.avi` | `vision_utils.py` |
| `03_ler_frames_video.py` | Regrava o vídeo com 30 frames e lê frame a frame imprimindo o shape | nada | `dados/rua_sintetica.avi` (sobrescrito) | `vision_utils.py` |
| `04_desenhar_bbox.py` | Desenha as caixas do gabarito no frame 12 | nada | `saidas/04_bounding_boxes.png` | `vision_utils.py`, pasta `saidas/` |
| `05_medicao_fps_latencia.py` | Mede FPS e latência do desenho mais detector simulado em 80 frames | nada | texto | `vision_utils.py` |
| `06_deteccoes_sinteticas_por_frame.py` | Imprime as detecções cruas dos frames 0, 10, 30 e 50 | nada | texto | `vision_utils.py` |
| `07_aplicar_nms_opencv.py` | Conta caixas antes e depois do NMS no frame 20 | nada | texto | `vision_utils.py` |
| `08_filtrar_por_confianca.py` | Filtra as detecções do frame 42 com limiar 0,80 | nada | texto | `vision_utils.py` |
| `09_formatar_rotulos_confianca.py` | Formata rótulos `classe | conf%` do frame 15 | nada | texto | `vision_utils.py` |
| `10_salvar_video_anotado_sintetico.py` | Vídeo de 100 frames com detecções após NMS | nada | `saidas/10_video_anotado.avi` | `vision_utils.py`, pasta `saidas/` |
| `11_yolov8_ultralytics_imagem.py` | YOLOv8n em uma imagem, salva anotada | `dados/imagem_teste.jpg` | `saidas/11_yolov8_imagem.jpg`, `yolov8n.pt` no cwd | ultralytics |
| `12_yolov8_ultralytics_video.py` | YOLOv8n em vídeo, latência média e FPS | `dados/video_real.mp4` | texto | ultralytics |
| `13_yolov8_opencv_onnx.py` | Carrega o ONNX no OpenCV DNN e imprime a forma da saída | `modelos/yolov8n.onnx`, `dados/imagem_teste.jpg` | texto | cv2.dnn |
| `14_yolov4_tiny_opencv_dnn.py` | Carrega YOLOv4-tiny em Darknet e lista camadas de saída | `modelos/yolov4-tiny.cfg`, `.weights`, `coco.names` | texto | cv2.dnn |
| `15_ssd_mobilenet_opencv_dnn.py` | Carrega SSD MobileNet v2 do TensorFlow | `modelos/frozen_inference_graph.pb`, `ssd_mobilenet_v2_coco.pbtxt` | texto | cv2.dnn |
| `16_comparar_tamanho_arquivos.py` | Tamanho em MB de quatro arquivos de modelo | arquivos em `modelos/` (opcionais) | texto | nada |
| `17_estimar_parametros_manual.py` | Tabela manual de parâmetros em milhões | nada | texto | nada |
| `18_tabela_comparativa_modelos.py` | DataFrame modelo com FPS, latência, parâmetros e tamanho | nada | texto | pandas |
| `19_benchmark_yolo_ssd_template.py` | Cronometra só a inferência (simulada) em 100 frames | nada | texto | `vision_utils.py` |
| `20_concluir_modelo_embarcado.py` | Escolhe modelo por FPS e latência a partir de valores fixos | nada | texto | nada |
| `21_iou_basico.py` | IoU entre três caixas de exemplo | nada | texto | `vision_utils.py` |
| `22_associacao_por_iou.py` | IoU de dois tracks contra duas detecções | nada | texto | `vision_utils.py` |
| `23_tracker_iou_ids_persistentes.py` | Rastreia 15 frames e imprime ID, classe e caixa | nada | texto | `vision_utils.py` |
| `24_trilhas_ultimos_30_frames.py` | Vídeo de 100 frames com IDs e trilhas | nada | `saidas/24_trilhas_30_frames.avi` | `vision_utils.py`, pasta `saidas/` |
| `25_entrada_saida_linha_virtual.py` | Conta cruzamentos da linha x=480 em 100 frames | nada | texto | `vision_utils.py` |
| `26_id_switches_por_minuto.py` | Estima ID switches com rastreador apertado e detector trêmulo | nada | texto | `vision_utils.py` |
| `27_tracker_csrt_opcional.py` | Imprime o esqueleto de uso do CSRT | nada | texto | cv2 |
| `28_pipeline_yolo_tracking_sintetico.py` | Detecção simulada, NMS, rastreamento e trilhas em 120 frames, com latência | nada | `saidas/28_pipeline_tracking_sintetico.avi` | `vision_utils.py`, pasta `saidas/` |
| `29_pipeline_video_real_com_yolo_template.py` | Mesmo pipeline com YOLOv8n em vídeo real, só latência | `dados/video_real.mp4` | texto | ultralytics, `vision_utils.py` |
| `30_comparar_yolo_ssd_tracking_template.py` | DataFrame modelo YOLO contra SSD com ID switches | nada | texto | pandas |
| `31_relatorio_etica_contagem_drone.py` | Imprime a discussão técnica e ética | nada | texto | nada |
| `32_gerar_csv_metricas.py` | CSV modelo de métricas por modelo | nada | `relatorios/metricas_modelos.csv` | `vision_utils.py` |
| `33_plotar_fps_sem_seaborn.py` | Gráfico de barras de FPS com valores fixos | nada | `relatorios/33_grafico_fps.png` | matplotlib, pasta `relatorios/` |
| `34_testar_thresholds_nms.py` | Conta caixas após NMS para cinco limiares | nada | texto | `vision_utils.py` |
| `35_experimento_resolucao_entrada.py` | Redimensiona o frame para 320, 416 e 640 | nada | texto | `vision_utils.py` |
| `notebook_pipeline_yolo_ssd_tracking.ipynb` | Vídeo sintético, NMS, rastreador e ponte para modelos reais em células | nada | `dados/notebook_video.avi`, `saidas/notebook_deteccoes.png` | Jupyter, `vision_utils.py` |

## 5. Passo a passo para executar

Todos os comandos abaixo são executados de dentro de `aula_13_YOLO_SSD/exemplos_em_python/`, com o venv da raiz ativo. Nenhum script abre janela; os resultados são texto no terminal e arquivos em `saidas/` e `relatorios/`. Os vídeos `.avi` abrem em qualquer player ou com um `cv2.VideoCapture`.

Passo 1. Preparar as pastas. Obrigatório antes de tudo, porque 04, 10, 24, 28 e 33 gravam em pastas que não criam.

```
python 01_ambiente_e_pastas.py
```

Passo 2. Gerar o vídeo sintético (120 frames, 20 FPS, 960 por 540) e depois ler frame a frame. Atenção: o 03 regrava o mesmo arquivo com apenas 30 frames. Nenhum script posterior lê esse arquivo (todos desenham os frames direto), mas se quiser o vídeo de 120 frames, rode o 02 de novo depois do 03.

```
python 02_criar_video_sintetico.py
python 03_ler_frames_video.py
```

Passo 3. Ver o gabarito desenhado. Abra `saidas/04_bounding_boxes.png`: pessoa, carro e bicicleta com caixa e rótulo `classe conf`. O ônibus não aparece porque o frame é o 12.

```
python 04_desenhar_bbox.py
```

Passo 4. Medir FPS e latência do laço sintético. Os números são altos (centenas de FPS) porque não há rede neural; servem como linha de base do custo do desenho.

```
python 05_medicao_fps_latencia.py
```

Passo 5. Ver as detecções cruas, com jitter e duplicatas, e o efeito do NMS e do filtro de confiança. No 07, espere 6 caixas antes e 3 depois (frame 20, sem ônibus). No 08, só pessoa e carro passam do limiar 0,80. O 09 imprime rótulos formatados.

```
python 06_deteccoes_sinteticas_por_frame.py
python 07_aplicar_nms_opencv.py
python 08_filtrar_por_confianca.py
python 09_formatar_rotulos_confianca.py
```

Passo 6. Gravar o vídeo anotado com NMS. Abra `saidas/10_video_anotado.avi` e observe o ônibus entrando depois do frame 35.

```
python 10_salvar_video_anotado_sintetico.py
```

Passo 7. Modelos reais (opcional, exige material). Coloque uma foto em `dados/imagem_teste.jpg` e um vídeo em `dados/video_real.mp4`. Na primeira chamada o Ultralytics baixa `yolov8n.pt` no diretório atual. O 11 salva `saidas/11_yolov8_imagem.jpg`; o 12 imprime latência e FPS.

```
python 11_yolov8_ultralytics_imagem.py
python 12_yolov8_ultralytics_video.py
```

Para o 13, exporte o ONNX e mova para `modelos/`; para o 14 e o 15, baixe os arquivos listados na seção 2. Sem os arquivos, os três scripts apenas imprimem o que esperam e encerram.

```
yolo export model=yolov8n.pt format=onnx imgsz=640
move yolov8n.onnx modelos\
python 13_yolov8_opencv_onnx.py
python 14_yolov4_tiny_opencv_dnn.py
python 15_ssd_mobilenet_opencv_dnn.py
```

Passo 8. Comparação de modelos. O 16 imprime "arquivo não encontrado" para o que não estiver em `modelos/` (copie o `yolov8n.pt` para lá se quiser vê-lo). O 17, 18 e 20 imprimem tabelas com valores fixos ou vazios. O 19 cronometra a inferência simulada.

```
python 16_comparar_tamanho_arquivos.py
python 17_estimar_parametros_manual.py
python 18_tabela_comparativa_modelos.py
python 19_benchmark_yolo_ssd_template.py
python 20_concluir_modelo_embarcado.py
```

Passo 9. Rastreamento. O 21 imprime IoU de 0,227 e 0,0. O 22 mostra as IoUs cruzadas. O 23 lista os IDs por frame durante 15 frames: espere IDs 1, 2 e 3 estáveis. O 24 grava o vídeo com trilhas amarelas. O 25 imprime entradas e saídas cumulativas. O 26 imprime IDs criados, switches aproximados e taxa por minuto.

```
python 21_iou_basico.py
python 22_associacao_por_iou.py
python 23_tracker_iou_ids_persistentes.py
python 24_trilhas_ultimos_30_frames.py
python 25_entrada_saida_linha_virtual.py
python 26_id_switches_por_minuto.py
python 27_tracker_csrt_opcional.py
```

Passo 10. Pipeline completo sintético. Gera `saidas/28_pipeline_tracking_sintetico.avi` e imprime latência e FPS do laço inteiro.

```
python 28_pipeline_yolo_tracking_sintetico.py
```

Passo 11. Pipeline com YOLO real (opcional, exige `dados/video_real.mp4` e Ultralytics) e template de comparação.

```
python 29_pipeline_video_real_com_yolo_template.py
python 30_comparar_yolo_ssd_tracking_template.py
```

Passo 12. Análise e relatório. O 31 imprime o texto de ética. O 32 cria o CSV modelo, o 33 salva o gráfico em `relatorios/`, o 34 varre os limiares do NMS (a contagem cresce quando o limiar se aproxima de 0,8) e o 35 imprime os três tamanhos de entrada.

```
python 31_relatorio_etica_contagem_drone.py
python 32_gerar_csv_metricas.py
python 33_plotar_fps_sem_seaborn.py
python 34_testar_thresholds_nms.py
python 35_experimento_resolucao_entrada.py
```

Passo 13. Notebook (opcional, exige `pip install notebook`). Abra o Jupyter de dentro de `exemplos_em_python/`, porque o notebook usa `Path.cwd()` como raiz e importa `vision_utils`.

```
jupyter notebook notebook_pipeline_yolo_ssd_tracking.ipynb
```

Ao terminar, `saidas/` contém `04_bounding_boxes.png`, `10_video_anotado.avi`, `24_trilhas_30_frames.avi` e `28_pipeline_tracking_sintetico.avi`; `relatorios/` contém `metricas_modelos.csv` e `33_grafico_fps.png`; `dados/` contém `rua_sintetica.avi`.

## 6. Exercícios e itens da aula

Nesta aula não há itens A e B. Cada script termina com um bloco `# DESAFIO DO ALUNO`, e o `GUIA_PROFESSOR.md` organiza a condução em quatro momentos com oito perguntas.

Desafios do aluno, script a script:

1. `01`: crie uma subpasta chamada `testes` dentro de `saidas`.
2. `02`: aumente `n_frames` para 200 e observe o tamanho do arquivo gerado.
3. `03`: salve o quinto frame como imagem PNG em `saidas`.
4. `04`: altere `frame_idx` para 40 e compare a posição dos objetos.
5. `05`: imprima também a menor e a maior latência observada.
6. `06`: conte quantas detecções aparecem em cada frame antes de aplicar NMS.
7. `07`: mude `nms_thr` para 0.1 e 0.8. O que acontece com a quantidade de caixas?
8. `08`: teste thresholds 0.5, 0.7 e 0.9 e compare os resultados.
9. `09`: acrescente as coordenadas da bbox no rótulo impresso.
10. `10`: escreva FPS e número de objetos detectados sobre cada frame.
11. `11`: teste com 3 imagens diferentes e registre quais classes aparecem.
12. `12`: salve um vídeo com as caixas desenhadas usando `results[0].plot()`.
13. `13`: pesquise o formato da saída do ONNX exportado e implemente a decodificação.
14. `14`: faça forward em uma imagem usando `blobFromImage` com tamanho 416x416.
15. `15`: crie o blob de uma imagem e rode `net.forward()`.
16. `16`: acrescente uma coluna `modelo` e salve esses dados em CSV.
17. `17`: complete os parâmetros do modelo SSD escolhido e registre a fonte usada.
18. `18`: acrescente uma coluna chamada `observacao_visual`.
19. `19`: substitua `inferencia_simulada` por uma função de inferência real.
20. `20`: modifique os valores usando medições reais e gere uma conclusão diferente.
21. `21`: crie uma box D que tenha IoU maior que 0.5 com a box A.
22. `22`: associe manualmente cada detecção ao ID mais provável.
23. `23`: diminua `iou_thr` para 0.1 e depois aumente para 0.7. O que muda?
24. `24`: mude `trail_len` para 10 e depois 60. Compare visualmente.
25. `25`: troque a linha vertical por uma linha horizontal e conte cruzamentos.
26. `26`: reduza `iou_thr` para 0.30 e compare a taxa.
27. `27`: verifique sua versão do OpenCV e descubra se o CSRT está em `cv2` ou `cv2.legacy`.
28. `28`: troque `fake_detector` por uma chamada real do YOLOv8n.
29. `29`: salve o vídeo de saída com as anotações.
30. `30`: preencha a tabela com os valores obtidos experimentalmente.
31. `31`: acrescente um parágrafo sobre risco de falso positivo e falso negativo.
32. `32`: preencha o CSV com medições reais.
33. `33`: crie outro gráfico para latência média.
34. `34`: explique por que threshold muito alto pode deixar caixas duplicadas.
35. `35`: meça o tempo de resize para cada resolução usando `time.perf_counter()`.

Os quatro momentos do `GUIA_PROFESSOR.md`:

1. Fundamentos do detector, exemplos 01 a 10: detecção não é só desenhar caixas; envolve confiança, classe, NMS, FPS e latência.
2. Modelos reais, exemplos 11 a 20: YOLOv8 via Ultralytics e SSD ou YOLO via OpenCV DNN quando os arquivos estiverem disponíveis; os alunos preenchem uma tabela com FPS, latência, tamanho em disco e observações visuais.
3. Rastreamento, exemplos 21 a 30: detecção frame a frame não garante identidade persistente; IoU, associação entre frames, trilhas e contagem de entradas e saídas.
4. Análise e responsabilidade, exemplos 31 a 35 e o notebook: limitações técnicas, erros, ID switches, privacidade, viés, sinalização e minimização de dados.

As oito perguntas do guia, para responder por escrito depois de rodar os scripts:

1. Qual a diferença entre detectar e rastrear?
2. Por que aplicar Non-Maximum Suppression?
3. Um FPS maior sempre é melhor?
4. O que acontece quando dois objetos se cruzam?
5. Por que ID switches são problemáticos em contagem?
6. Em quais cenários SSD ainda pode ser interessante?
7. O que deve ser medido no próprio ambiente de operação?
8. Quais cuidados éticos são necessários ao usar drones em áreas urbanas?

## 7. Armadilhas e dicas

- Rode o `01_ambiente_e_pastas.py` antes de qualquer outro. Os scripts 04, 10, 24 e 28 gravam em `saidas/` e o 33 em `relatorios/` sem criar a pasta; se ela não existir, `cv2.imwrite` e `VideoWriter` falham em silêncio (arquivo não aparece e nenhum erro é impresso) e `plt.savefig` dá `FileNotFoundError`. No repositório as duas pastas já vêm com `.gitkeep`, mas `dados/` e `modelos/` não.
- O `03_ler_frames_video.py` regrava `dados/rua_sintetica.avi` com 30 frames por cima dos 120 do script 02. Nenhum outro script lê esse arquivo, mas se for usá-lo, rode o 02 de novo.
- `YOLO("yolov8n.pt")` baixa o peso no diretório de trabalho atual, não em `modelos/`. O `16_comparar_tamanho_arquivos.py` procura `modelos/yolov8n.pt` e imprime "arquivo não encontrado" até você copiar o arquivo para lá.
- Não instale o `ultralytics` com `pip install ultralytics` seco, como sugere a mensagem de erro dos scripts 11, 12 e 29: ele puxa `opencv-python`, que conflita com o `opencv-contrib-python` do projeto. Use `pip install --no-deps -r requirements_ultralytics.txt` a partir da raiz, ou o `preparar_ambiente.py`.
- Exportar para ONNX (`yolo export`) exige `onnx` e `onnxslim`, que não estão no ambiente unificado. Instale à parte só se for fazer o script 13.
- O `13_yolov8_opencv_onnx.py` não decodifica a saída: `out.shape` será `(1, 84, 8400)` e nada é desenhado. A decodificação é o desafio do aluno.
- Os scripts 14 e 15 só carregam o modelo. Não há blob, forward nem desenho; isso também fica como desafio.
- Os números 24,5 e 31,2 FPS dos scripts 20 e 33, e os 3,2 milhões de parâmetros do 17 e do 18, são exemplos fixos, não medições. Se entregar o gráfico do 33 sem trocar os valores, o relatório está inventado.
- Os scripts 17, 18, 30 e 32 imprimem ou gravam tabelas com zeros e `None`. São templates para preencher com as medições dos scripts 12, 19 e 29.
- O `apply_nms` é agnóstico de classe: junta todas as caixas em uma lista só. Duas classes diferentes muito sobrepostas (uma pessoa na frente de um carro) podem se suprimir. Detectores reais normalmente fazem NMS por classe.
- `IoUTracker` também ignora a classe na associação: se um track de `car` tiver IoU alta com uma detecção de `bus`, ele continua com o ID antigo e a classe é sobrescrita. Na cena sintética, pessoa e carro se cruzam por volta do frame 90, e é aí que os IDs podem trocar.
- Tracks perdidos (sem associação) continuam na saída do `update` com a última caixa, até `missing` passar de `max_missing` (padrão 8). Por isso o objeto que saiu da tela ainda aparece desenhado por alguns frames.
- As trilhas de tracks apagados não são removidas de `tracker.trails`, então continuam sendo desenhadas até o fim do vídeo, congeladas onde o objeto sumiu.
- No `26_id_switches_por_minuto.py`, `switches_aprox = total_created - 4` assume que há exatamente quatro objetos. Com outra cena, ou com `n_frames` menor que 36 (o ônibus ainda não entrou), a conta muda.
- O `27_tracker_csrt_opcional.py` não rastreia nada; só imprime três linhas. Nesta instalação (OpenCV 4.14 contrib) tanto `cv2.TrackerCSRT_create()` quanto `cv2.legacy.TrackerCSRT_create()` funcionam.
- `vision_utils.py` define `xywh_to_bbox` e a classe `Timer`, que nenhum script usa. Mantidos por fidelidade ao professor.
- O `28_pipeline_yolo_tracking_sintetico.py` cronometra o laço inteiro, com desenho da cena e gravação. Comparar essa latência com a do 19 (só inferência) mostra quanto custa o restante do pipeline.
- Vídeos MJPG em `.avi` ficam grandes: 100 frames de 960 por 540 dão dezenas de MB. Se o disco for um problema, reduza `n_frames` ou troque o FOURCC.
- O notebook usa `Path.cwd()` e `from vision_utils import ...`. Abra o Jupyter de dentro de `exemplos_em_python/`, senão a importação e os caminhos falham. O Jupyter não está no ambiente unificado.
- O `12_yolov8_ultralytics_video.py` e o `29_...` processam o vídeo inteiro sem janela nem gravação. Em um vídeo longo isso demora, sem nenhuma saída no terminal até o fim. Corte o vídeo ou limite o número de frames para testar.
- Parâmetros que valem a pena variar: `nms_thr` (0,1 a 0,8), o limiar de confiança (0,5 a 0,9), `iou_thr` do rastreador (0,1 a 0,7), `max_missing`, `trail_len`, `jitter` do detector simulado e o tamanho de entrada (320, 416, 640).

## 8. Glossário

| Termo | Significado |
|---|---|
| Detecção de objetos | Tarefa de localizar objetos em uma imagem com caixas delimitadoras, cada uma com classe e confiança. |
| Bounding box (bbox) | Caixa delimitadora. Nesta aula, lista `[x1, y1, x2, y2]` com os cantos superior esquerdo e inferior direito. |
| xywh | Formato de caixa por canto superior esquerdo, largura e altura, exigido pelo `cv2.dnn.NMSBoxes`. |
| Classe (cls) | Categoria do objeto detectado, por exemplo `person`, `car`, `bicycle`, `bus`. |
| Confiança (conf) | Pontuação entre 0 e 1 que o detector atribui à caixa. |
| Threshold de confiança | Limiar abaixo do qual a detecção é descartada. |
| NMS | Non-Maximum Suppression. Remoção de caixas redundantes que se sobrepõem muito a uma caixa de maior confiança. |
| nms_thr | Limiar de IoU do NMS: caixas com IoU acima dele em relação à caixa mantida são suprimidas. |
| IoU | Intersection over Union. Área da interseção dividida pela área da união de duas caixas, entre 0 e 1. |
| iou_thr | Limiar mínimo de IoU para um track ser associado a uma detecção no rastreador. |
| YOLO | You Only Look Once. Família de detectores de um estágio que prediz todas as caixas em uma passada. |
| YOLOv8n | Variante nano do YOLOv8 da Ultralytics, com cerca de 3,2 milhões de parâmetros. |
| YOLOv4-tiny | Versão reduzida do YOLOv4, distribuída em formato Darknet (`.cfg` e `.weights`). |
| SSD | Single Shot MultiBox Detector. Detector de um estágio que prediz caixas a partir de várias camadas com âncoras. |
| MobileNet | Rede convolucional leve usada como tronco do SSD em sistemas embarcados. |
| Ultralytics | Biblioteca Python que empacota treino, inferência e exportação do YOLOv8. |
| ONNX | Open Neural Network Exchange. Formato aberto de modelo que o OpenCV DNN consegue carregar. |
| Darknet | Framework original do YOLO; seus arquivos `.cfg` e `.weights` são lidos por `readNetFromDarknet`. |
| Grafo congelado (.pb) | Modelo do TensorFlow 1 com pesos embutidos, lido por `readNetFromTensorflow` junto com um `.pbtxt`. |
| Blob | Tensor de entrada da rede no formato `(N, C, H, W)`, criado por `cv2.dnn.blobFromImage`. |
| swapRB | Parâmetro do `blobFromImage` que troca BGR por RGB, necessário quando o modelo foi treinado em RGB. |
| Forward | Passagem da entrada pela rede para obter a saída, `net.forward()`. |
| Camadas de saída | Camadas não conectadas a outras, cujos tensores são o resultado do detector. |
| COCO | Dataset de 80 classes em que YOLOv8n, YOLOv4-tiny e SSD MobileNet foram treinados; os nomes ficam em `coco.names`. |
| Parâmetros | Número de pesos treináveis do modelo, medido em milhões (M). |
| FPS | Frames per second. Quadros processados por segundo. |
| Latência | Tempo para processar um frame, em milissegundos. |
| Benchmark | Medição sistemática de FPS, latência, memória e tamanho nas mesmas condições. |
| Embarcado | Sistema com processador, memória e energia limitados, como uma placa em um robô ou drone. |
| Rastreamento (tracking) | Manter a identidade de cada objeto ao longo dos frames. |
| Track | Objeto acompanhado pelo rastreador, com ID, última caixa, classe, confiança e contador de frames perdidos. |
| ID persistente | Identificador que permanece o mesmo enquanto o objeto é acompanhado. |
| Associação | Escolha de qual detecção do frame atual corresponde a qual track do frame anterior. |
| Associação gulosa | Estratégia que aceita os pares em ordem decrescente de IoU, sem reconsiderar. |
| max_missing | Número de frames sem associação que um track tolera antes de ser apagado. |
| Trilha (trail) | Sequência dos centros recentes de um track, guardada em um `deque`. |
| deque | Fila de tamanho máximo fixo da biblioteca padrão; descarta o item mais antigo ao encher. |
| ID switch | Troca indevida de identidade: o mesmo objeto recebe um ID novo ou troca de ID com outro. |
| Linha virtual | Reta fixa na imagem usada para contar objetos que a cruzam. |
| CSRT, KCF, MIL | Rastreadores de objeto único do OpenCV, inicializados com uma caixa e atualizados por aparência. |
| Jitter | Tremor. Ruído somado às coordenadas da caixa pelo detector simulado. |
| Detector simulado | `fake_detector`, que devolve o gabarito com ruído e duplicatas, sem rede neural. |
| Ground truth | Gabarito. Caixas verdadeiras conhecidas, aqui geradas por `synthetic_objects`. |
| Falso positivo | Caixa detectada onde não há objeto. |
| Falso negativo | Objeto presente que o detector não encontrou. |
| MJPG | Motion JPEG. Codec em que cada frame é um JPEG independente; portátil, mas gera arquivos grandes. |
| FOURCC | Código de quatro letras do codec de vídeo, aqui `MJPG`. |
| CSV | Comma-Separated Values. Arquivo de texto tabular para registrar métricas. |
| DataFrame | Tabela do pandas, usada nos templates de comparação. |
| Minimização de dados | Princípio de coletar e reter só o necessário para a finalidade declarada. |
