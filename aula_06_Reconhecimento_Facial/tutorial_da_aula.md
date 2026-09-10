# Aula 06 – Reconhecimento facial

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula percorre o caminho completo de um sistema de visão facial em duas etapas. A primeira é a **detecção**: encontrar onde há rostos em uma imagem ou em um vídeo, usando o classificador Haar Cascade do OpenCV, e recortar cada rosto como uma região de interesse (ROI) normalizada para 48 por 48 pixels. A segunda é o **reconhecimento**: descobrir de quem é cada rosto, comparando um vetor de 128 números (embedding) gerado pela biblioteca `face_recognition` com uma base de identidades cadastradas.

O fio condutor é o compromisso entre sensibilidade e especificidade. Na detecção, os parâmetros `scaleFactor` e `minNeighbors` decidem se o sistema encontra mais rostos ao custo de mais falsos positivos. No reconhecimento, a tolerância de distância decide se o sistema aceita mais correspondências ao custo de mais falsos aceites. Os scripts medem esse compromisso com um vídeo de gabarito, medem a latência por frame e discutem como escolher os parâmetros em um alvo embarcado (robô ou drone), inclusive com uma nota ética sobre biometria.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas pelos scripts:

| Biblioteca | Pacote pip | Usada em |
|---|---|---|
| `cv2` | `opencv-python` | todos |
| `numpy` | `numpy` | 00, 07, 08, 09, 10 |
| `face_recognition` | `face-recognition` | 00, 06, 07, 08, 09, 10 |
| `dlib` | `dlib` | 00 (importado só para checar a versão; `face_recognition` depende dele) |

Instalação sugerida:

```
pip install opencv-python numpy face-recognition
```

Observações sobre o ambiente:

- `dlib` e `face_recognition` são difíceis de instalar no Windows. Eles exigem CMake e um compilador C++ ou um wheel pré-compilado. O professor rodou os scripts com Python 3.12 (o zip trazia `__pycache__` dessa versão). A máquina local tem Python 3.14, então vale criar um venv com 3.12 se o `pip install dlib` falhar.
- Os scripts 00 a 05 só precisam de OpenCV e NumPy. Dá para começar por eles enquanto o `dlib` não instala.
- **Pasta de execução:** rode os scripts de dentro de `exemplos_em_python/`, porque todos fazem `from utils import ...`. O `utils.py` define a raiz do projeto como a pasta acima, ou seja, `aula_06_Reconhecimento_Facial/`. É lá que ficam `recursos/` (entrada) e `saidas/` (criada automaticamente).
- **Material que falta** (não veio no zip, ver `achados_dos_zips.md`): a pasta `aula_06_Reconhecimento_Facial/recursos/` com `foto_grupo.jpg`, `identidades/Ana.jpg`, `identidades/Bruno.jpg`, `identidades/Carla.jpg` e `video_teste.mp4`. O arquivo `gabarito_video.csv` é opcional (o código devolve um dicionário vazio se ele não existir), mas sem ele o script 05 quebra e o 04 não calcula taxa de detecção. A cópia local `recursos/haarcascade_frontalface_default.xml` também é opcional: se não existir, o `utils.py` usa o XML que vem no OpenCV.
- **Webcam:** os scripts 04 e 08 aceitam `--source 0` para usar a webcam no lugar do vídeo. Os demais leem arquivos.

Formato esperado do `gabarito_video.csv`, deduzido de `load_ground_truth` no `utils.py`: cabeçalho `frame,rostos_esperados`, uma linha por frame do vídeo, com o índice do frame começando em 0.

## 3. Conceitos-chave

### 3.1 Estrutura do projeto e módulo utilitário

Todos os scripts compartilham o `utils.py`, que centraliza caminhos e funções repetidas. `PROJECT` é a pasta acima da pasta dos scripts, `RESOURCES` é `PROJECT/recursos` e `OUTPUTS` é `PROJECT/saidas`. A função `ensure_outputs()` cria `saidas/` se necessário e devolve o caminho. Isso permite que os scripts gravem resultados sem se preocupar com a existência da pasta.

```python
# utils.py
PROJECT = Path(__file__).resolve().parents[1]
RESOURCES = PROJECT / "recursos"
OUTPUTS = PROJECT / "saidas"

def ensure_outputs() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS
```

Onde aparece: `utils.py`, importado por todos os scripts de 02 a 10.

### 3.2 Validação do ambiente

Antes de qualquer processamento, o script 00 imprime as versões de OpenCV e NumPy, tenta importar `face_recognition` e `dlib` dentro de um `try/except` e verifica se os arquivos de `recursos/` existem. Essa checagem evita descobrir a falta de uma dependência só no meio de um vídeo.

```python
# 00_validar_ambiente.py
try:
    import face_recognition
    import dlib
    print(f"dlib: {getattr(dlib, '__version__', 'OK')}")
except Exception as exc:
    print(f"face_recognition/dlib: FALHA — {exc}")

for path in required:
    print(f"{path.name}: {'OK' if path.exists() else 'AUSENTE'}")
```

Onde aparece: `00_validar_ambiente.py`.

### 3.3 Haar Cascade: o detector de rostos do OpenCV

O Haar Cascade é um detector clássico (Viola-Jones) que usa características retangulares simples (diferenças de soma de pixels entre regiões claras e escuras) avaliadas em cascata: cada estágio rejeita rapidamente janelas que claramente não são rosto e só as candidatas sobrevivem até os estágios finais. O OpenCV distribui modelos já treinados como arquivos XML, e o caminho da pasta deles está em `cv2.data.haarcascades`.

A classe `cv2.CascadeClassifier` carrega o XML. O método `empty()` devolve `True` se a carga falhou, e o professor usa isso para lançar um erro claro em vez de deixar o script seguir com um detector vazio.

```python
# 01_carregar_haar.py
xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(str(xml))
print(f"Classificador carregado: {not detector.empty()}")
if detector.empty():
    raise RuntimeError("O Haar Cascade não foi carregado. Verifique a instalação do OpenCV.")
```

A função `load_haar()` do `utils.py` faz o mesmo, mas prefere uma cópia local em `recursos/` e cai no XML do OpenCV como reserva.

Onde aparece: `01_carregar_haar.py`, `utils.py` (`load_haar`), e via `load_haar` em 02, 03, 04, 05.

### 3.4 Pré-processamento: escala de cinza e equalização de histograma

O Haar trabalha sobre intensidades, não sobre cor, então o primeiro passo é `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`. Lembre que o OpenCV lê imagens em ordem BGR, não RGB.

Em seguida, `cv2.equalizeHist(gray)` redistribui os níveis de cinza para que o histograma fique mais uniforme. Isso aumenta o contraste em imagens escuras ou lavadas e torna o detector menos dependente da iluminação. O professor aplica a equalização nos scripts 02, 04 e 05, mas não no 03.

```python
# 02_detectar_imagem.py
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
```

Onde aparece: `02_detectar_imagem.py`, `03_extrair_roi_48x48.py` (só cinza), `04_item_a_tempo_real.py`, `05_comparar_parametros.py`.

### 3.5 detectMultiScale e seus parâmetros

`detector.detectMultiScale(gray, scaleFactor, minNeighbors, minSize)` varre a imagem em várias escalas (pirâmide) e devolve uma lista de retângulos `(x, y, w, h)`. Os três parâmetros controlam o compromisso central da aula:

- `scaleFactor`: quanto a janela de busca cresce a cada passo da pirâmide. Um valor de 1,05 significa passos de 5 por cento, ou seja, muitas escalas testadas (mais sensível, mais lento). Um valor de 1,20 pula escalas (mais rápido, pode perder rostos de tamanhos intermediários).
- `minNeighbors`: quantos retângulos vizinhos precisam concordar para uma detecção ser aceita. Valores baixos aceitam mais detecções, inclusive falsas. Valores altos exigem consenso e rejeitam falsos positivos, mas podem descartar rostos reais.
- `minSize`: menor tamanho de rosto considerado, em pixels. Descarta ruído pequeno e reduz o custo.

```python
# 02_detectar_imagem.py
faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.10,
    minNeighbors=5,
    minSize=(50, 50),
)
for i, (x, y, w, h) in enumerate(faces, start=1):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (38, 226, 167), 3)
```

No script 03 os mesmos parâmetros são passados por posição: `detectMultiScale(gray, 1.08, 4, minSize=(45, 45))`.

Onde aparece: `02_detectar_imagem.py`, `03_extrair_roi_48x48.py`, `04_item_a_tempo_real.py`, `05_comparar_parametros.py`.

### 3.6 Desenho de anotações: retângulo e rótulo com fundo

Para visualizar as detecções, o professor desenha um retângulo com `cv2.rectangle` e um rótulo com a função `put_label` do `utils.py`. Essa função mede o texto com `cv2.getTextSize`, pinta um retângulo escuro preenchido (espessura -1) atrás do texto e escreve com `cv2.putText` usando `cv2.LINE_AA` para bordas suavizadas. O fundo garante que o texto continue legível sobre qualquer imagem. A cor padrão `(38, 226, 167)` é um verde em BGR.

```python
# utils.py
def put_label(frame, text, origin, color=(38, 226, 167)):
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y + 4), (20, 25, 31), -1)
    cv2.putText(frame, text, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)
```

A chamada `put_label(frame, f"rosto {i}", (x, max(28, y)))` usa `max(28, y)` para o rótulo não sair pelo topo da imagem quando o rosto está muito alto.

Onde aparece: `utils.py`, `02_detectar_imagem.py`, `04_item_a_tempo_real.py`, `07_reconhecer_imagem.py`, `08_item_b_video.py`.

### 3.7 Extração de ROI e normalização para 48 por 48

Uma ROI (Region of Interest, região de interesse) é o recorte do rosto dentro da imagem. Em NumPy o recorte é uma fatia do array: `gray[y:y + h, x:x + w]`, com linhas primeiro (eixo y) e colunas depois (eixo x). Em seguida o recorte é redimensionado para um tamanho fixo de 48 por 48 pixels com `cv2.resize` e interpolação `cv2.INTER_AREA`, que é a recomendada para reduzir imagens porque faz média dos pixels e evita serrilhado.

O tamanho fixo é importante porque modelos de classificação (por exemplo, de expressão facial) esperam entradas de dimensão constante. Cada ROI é salva em PNG com nome numerado, e o script imprime shape, mínimo e máximo para confirmar que a imagem é 48x48 em uint8 de 0 a 255.

```python
# 03_extrair_roi_48x48.py
for number, (x, y, w, h) in enumerate(faces, start=1):
    roi = gray[y:y + h, x:x + w]
    roi_48 = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
    filename = folder / f"face_{number:03d}.png"
    cv2.imwrite(str(filename), roi_48)
    print(f"{filename.name}: shape={roi_48.shape}, min={roi_48.min()}, max={roi_48.max()}")
```

Onde aparece: `03_extrair_roi_48x48.py`, `04_item_a_tempo_real.py`.

### 3.8 Captura de vídeo e webcam com argparse

A função `open_source(source)` do `utils.py` aceita uma string: se for só dígitos, converte para inteiro e abre a webcam desse índice; senão trata como caminho de vídeo. `cv2.VideoCapture` serve para os dois casos. O laço padrão de vídeo é `ok, frame = cap.read()` até `ok` ser `False`, seguido de `cap.release()` e `cv2.destroyAllWindows()`.

Os scripts 04 e 08 expõem isso via `argparse`, com `--source` (padrão `recursos/video_teste.mp4`), `--headless` (não abre janela, útil em servidor sem tela), `--max-frames` (0 processa até o fim) e `--output-video` (grava o resultado anotado).

```python
# utils.py
def open_source(source: str) -> cv2.VideoCapture:
    parsed = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(parsed)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a fonte: {source}")
    return cap
```

Onde aparece: `utils.py`, `04_item_a_tempo_real.py`, `08_item_b_video.py`. Os scripts 05 e 09 abrem o vídeo diretamente com `cv2.VideoCapture`.

### 3.9 Exibição em tempo real e tecla de saída

`cv2.imshow(titulo, frame)` mostra o frame em uma janela e `cv2.waitKey(1)` espera 1 milissegundo por uma tecla, o que mantém a janela responsiva. A expressão `cv2.waitKey(1) & 0xFF == ord("q")` isola o byte da tecla e compara com `q`. Sem o `waitKey` a janela não atualiza.

```python
# 04_item_a_tempo_real.py
if not args.headless:
    cv2.imshow("Item A - Haar", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

Onde aparece: `04_item_a_tempo_real.py`, `08_item_b_video.py`.

### 3.10 Gravação de vídeo anotado com VideoWriter

Quando `--output-video` é informado, o script cria um `cv2.VideoWriter` na primeira iteração, quando já conhece a altura e largura do frame. O codec é dado por `cv2.VideoWriter_fourcc(*"mp4v")`, a taxa é fixa em 20 quadros por segundo e o tamanho é `(w, h)` (largura antes da altura, ao contrário do `shape`). Cada frame anotado é escrito com `writer.write(frame)` e no fim `writer.release()` fecha o arquivo.

```python
# 08_item_b_video.py
if args.output_video:
    if writer is None:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    writer.write(frame)
```

Onde aparece: `04_item_a_tempo_real.py`, `08_item_b_video.py`.

### 3.11 Sensibilidade versus especificidade na detecção

O script 04 define dois perfis de parâmetros e o docstring explica o compromisso. O perfil `sensivel` (scaleFactor 1,05, minNeighbors 3, minSize 35) tem pirâmide densa e pouca exigência de consenso: encontra mais rostos, aceita mais falsos positivos e gasta mais CPU. O perfil `especifica` (scaleFactor 1,20, minNeighbors 6, minSize 50) faz o oposto: rejeita falsos positivos, mas pode perder rostos pequenos, inclinados ou mal iluminados.

O professor conecta a escolha ao custo do erro na aplicação. Para um robô acompanhar uma pessoa ou evitar colisão, perder uma detecção é pior, então prioriza-se a sensibilidade. Para liberar uma porta, um falso aceite é mais grave, então prioriza-se a especificidade.

```python
# 04_item_a_tempo_real.py e 05_comparar_parametros.py
CONFIGS = {
    "sensivel": dict(scaleFactor=1.05, minNeighbors=3, minSize=(35, 35)),
    "especifica": dict(scaleFactor=1.20, minNeighbors=6, minSize=(50, 50)),
}
faces = detector.detectMultiScale(gray, **CONFIGS[args.config])
```

Onde aparece: `04_item_a_tempo_real.py` (`--config sensivel|especifica`), `05_comparar_parametros.py`.

### 3.12 Avaliação contra gabarito: taxa de detecção e falsos positivos

Para medir os dois perfis de forma objetiva, o professor usa um gabarito por frame, lido por `load_ground_truth()` do CSV `recursos/gabarito_video.csv` (dicionário `frame -> rostos_esperados`). A cada frame, com `found` detecções e `expected` esperadas:

- acertos estimados: `min(found, expected)`;
- falsos positivos visíveis estimados: `max(0, found - expected)`;
- taxa de detecção estimada: soma dos acertos dividida pela soma dos esperados, em porcentagem.

É uma estimativa porque só compara contagens, sem conferir se cada retângulo está na posição certa. O script 05 roda os dois perfis no vídeo inteiro, imprime uma tabela e grava `saidas/05_comparacao_parametros.csv` com `csv.DictWriter`. O script 04 faz a mesma contagem em tempo real e mostra o status no frame, mas só quando a fonte é um arquivo (com webcam não há gabarito).

```python
# 05_comparar_parametros.py
found = len(detector.detectMultiScale(gray, **params))
expected = truth[frame_idx]
detected_total += found
expected_total += expected
matched_total += min(found, expected)
fp_total += max(0, found - expected)
```

Onde aparece: `utils.py` (`load_ground_truth`), `04_item_a_tempo_real.py`, `05_comparar_parametros.py`.

### 3.13 Embeddings faciais de 128 dimensões

Detecção diz onde há um rosto; reconhecimento diz de quem é. A biblioteca `face_recognition` (construída sobre `dlib`) transforma cada rosto em um vetor de 128 números, o embedding, gerado por uma rede neural treinada para que rostos da mesma pessoa fiquem próximos e de pessoas diferentes fiquem distantes. Comparar identidades vira comparar distâncias entre vetores.

O fluxo de cadastro tem três chamadas:

- `face_recognition.load_image_file(path)`: lê a imagem já em RGB (diferente do OpenCV).
- `face_recognition.face_locations(image, model="hog")`: localiza rostos com o detector HOG (Histogram of Oriented Gradients), que roda em CPU. Devolve tuplas `(top, right, bottom, left)`.
- `face_recognition.face_encodings(image, locations)`: gera um embedding por rosto localizado.

O script 06 exige exatamente um rosto por foto de identidade e salva a base com `pickle` em `saidas/encodings.pkl`, como um dicionário com `names` e `encodings`.

```python
# 06_cadastrar_identidades.py
image = face_recognition.load_image_file(str(path))  # RGB
locations = face_recognition.face_locations(image, model="hog")
vectors = face_recognition.face_encodings(image, locations)
if len(vectors) != 1:
    raise RuntimeError(f"Esperado 1 rosto em {path.name}; encontrados {len(vectors)}")
encodings.append(vectors[0])
database = {"names": KNOWN_NAMES, "encodings": encodings}
with output.open("wb") as f:
    pickle.dump(database, f)
```

Onde aparece: `06_cadastrar_identidades.py`, `07_reconhecer_imagem.py`, `08_item_b_video.py`, `09_benchmark_embarcado.py`, `10_varrer_tolerancia.py`.

### 3.14 Conversão BGR para RGB ao misturar OpenCV e face_recognition

Quando a imagem vem do OpenCV (`cv2.imread` ou `cap.read()`), ela está em BGR, mas `face_recognition` espera RGB. Por isso os scripts 07, 08 e 09 chamam `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` antes de localizar e codificar rostos. Esquecer essa conversão degrada o embedding em silêncio. Os scripts 06 e 10 usam `load_image_file`, que já devolve RGB, e não precisam converter.

Outro detalhe é a ordem das coordenadas: `face_locations` devolve `(top, right, bottom, left)`, então o retângulo do OpenCV é desenhado como `cv2.rectangle(frame, (left, top), (right, bottom), ...)`.

```python
# 07_reconhecer_imagem.py
frame = cv2.imread(str(RESOURCES / "foto_grupo.jpg"))
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
locations = face_recognition.face_locations(rgb, model="hog")
vectors = face_recognition.face_encodings(rgb, locations)
for (top, right, bottom, left), vector in zip(locations, vectors):
    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
```

Onde aparece: `07_reconhecer_imagem.py`, `08_item_b_video.py`, `09_benchmark_embarcado.py`.

### 3.15 Distância entre embeddings e tolerância

`face_recognition.face_distance(base, vetor)` devolve a distância euclidiana entre o vetor do rosto detectado e cada vetor cadastrado. `np.argmin` escolhe o cadastrado mais próximo. A decisão final compara essa menor distância com uma tolerância: se `distances[best] <= TOLERANCE`, o rosto recebe o nome; senão, vira `Desconhecido`. A tolerância padrão nos scripts é 0,50.

Tolerância menor é mais rígida: menos falsos aceites (aceitar um estranho como cadastrado), mais falsos rejeites (recusar um cadastrado). Tolerância maior é o inverso. É o mesmo compromisso da detecção, agora no reconhecimento. O script 07 usa cores diferentes para reconhecido (verde) e desconhecido (vermelho em BGR, `(84, 91, 231)`) e escreve a distância no rótulo.

```python
# 07_reconhecer_imagem.py
TOLERANCE = 0.50  # menor = comparação mais rígida = menos falsos aceites
distances = face_recognition.face_distance(database["encodings"], vector)
best = int(np.argmin(distances))
name = database["names"][best] if distances[best] <= TOLERANCE else "Desconhecido"
color = (38, 226, 167) if name != "Desconhecido" else (84, 91, 231)
```

Onde aparece: `07_reconhecer_imagem.py`, `08_item_b_video.py` (`--tolerance`), `10_varrer_tolerancia.py`.

### 3.16 Varredura de tolerância: falso aceite versus falso rejeite

O script 10 aplica três tolerâncias (0,40, 0,50 e 0,60) aos mesmos rostos da foto de grupo e imprime, para cada uma, o rótulo e a distância de cada rosto. O objetivo é ver na prática um rosto mudar de `Desconhecido` para um nome conforme a tolerância cresce. A mensagem final do professor é que o limiar não deve ser escolhido pela demonstração, e sim por validação em dados separados e pelo custo do erro na aplicação.

```python
# 10_varrer_tolerancia.py
for tolerance in [0.40, 0.50, 0.60]:
    labels = []
    for vector in vectors:
        distances = face_recognition.face_distance(database["encodings"], vector)
        best = int(np.argmin(distances))
        label = database["names"][best] if distances[best] <= tolerance else "Desconhecido"
        labels.append(f"{label}(d={distances[best]:.3f})")
    print(f"tolerância={tolerance:.2f}: " + ", ".join(labels))
```

Onde aparece: `10_varrer_tolerancia.py`.

### 3.17 Redução de escala para acelerar a inferência

Localizar e codificar rostos com `face_recognition` é caro. O script 08 reduz o frame com `cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)` antes da inferência (padrão 0,50, metade da largura e da altura, um quarto dos pixels). As coordenadas devolvidas estão na imagem pequena, então antes de desenhar no frame original elas são multiplicadas por `inv = 1.0 / args.scale`.

```python
# 08_item_b_video.py
small = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
locations = face_recognition.face_locations(rgb, model="hog")
...
inv = 1.0 / args.scale
for (top, right, bottom, left), (name, distance) in zip(locations, predictions):
    top, right, bottom, left = [int(v * inv) for v in (top, right, bottom, left)]
```

Onde aparece: `08_item_b_video.py`, `09_benchmark_embarcado.py`.

### 3.18 Medição de latência: média, p95 e FPS teórico

Para saber se o sistema serve para tempo real, o script 08 mede o tempo de cada inferência com `time.perf_counter()` (relógio de alta resolução) antes e depois do bloco de localização e codificação, converte para milissegundos e acumula em uma lista. No fim imprime a média, o percentil 95 com `np.percentile(values, 95)` (o tempo abaixo do qual ficam 95 por cento dos frames, útil para captar picos) e o FPS teórico, calculado como `1000 / média`. O rótulo no frame mostra a latência do frame atual e a média acumulada.

```python
# 08_item_b_video.py
start = time.perf_counter()
...  # localizar e codificar rostos
latency = (time.perf_counter() - start) * 1000
latencies_ms.append(latency)
...
print(f"Latência média de inferência: {values.mean():.2f} ms/frame")
print(f"Latência p95: {np.percentile(values, 95):.2f} ms/frame")
print(f"FPS teórico da inferência: {1000 / values.mean():.1f}")
```

Onde aparece: `08_item_b_video.py`, `09_benchmark_embarcado.py`.

### 3.19 Benchmark embarcado: escala e salto de frames

O script 09 combina duas alavancas de desempenho: a escala da imagem e o salto de frames (processar só um a cada `every` frames, testado com `read % every == 0`). Define quatro perfis (`qualidade` 1,00 e 1; `equilibrado` 0,50 e 1; `economico` 0,50 e 2; `minimo` 0,25 e 2), roda cada um nos 80 primeiros frames do vídeo e imprime latência média e FPS efetivo, calculado como `1000 / mean_ms * every`, porque pular frames multiplica a taxa efetiva.

O comentário final do professor lembra que, em controle robótico, além do FPS é preciso definir a idade máxima aceitável da última identidade reconhecida, ou seja, por quanto tempo uma decisão antiga ainda pode ser usada.

```python
# 09_benchmark_embarcado.py
EXPERIMENTS = [
    ("qualidade", 1.00, 1),
    ("equilibrado", 0.50, 1),
    ("economico", 0.50, 2),
    ("minimo", 0.25, 2),
]
if read % every == 0:
    start = time.perf_counter()
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ...
effective_fps = 1000 / mean_ms * every
```

Onde aparece: `09_benchmark_embarcado.py`.

### 3.20 Ética em biometria

O docstring do script 08 traz o alerta ético da aula. Biometria em robôs e drones pode viabilizar vigilância e rastreamento sem consentimento, e os erros do sistema podem atingir grupos de pessoas de forma desigual. Uma implantação responsável exige finalidade legítima e explícita, minimização e proteção dos dados, prazo de retenção, auditoria, teste por subgrupos e revisão humana. O professor é categórico: uma correspondência biométrica não deve disparar punição autônoma.

Onde aparece: `08_item_b_video.py` (docstring), reforçado pelas mensagens finais de `09_benchmark_embarcado.py` e `10_varrer_tolerancia.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `utils.py` | Define `PROJECT`, `RESOURCES`, `OUTPUTS`, `load_haar`, `open_source`, `load_ground_truth`, `put_label`, `ensure_outputs` | nenhuma | nenhuma | OpenCV |
| `00_validar_ambiente.py` | Imprime versões, testa import de `face_recognition` e `dlib`, checa arquivos de `recursos/` | `recursos/` | só terminal | nada |
| `01_carregar_haar.py` | Carrega o XML do Haar direto de `cv2.data` e confirma que não está vazio | XML do OpenCV | só terminal | nada |
| `02_detectar_imagem.py` | Detecta rostos em `identidades/Ana.jpg` com equalização e desenha caixas | `recursos/identidades/Ana.jpg` | `saidas/02_deteccao_imagem.jpg` | `utils.py` |
| `03_extrair_roi_48x48.py` | Detecta rostos na foto de grupo, recorta e salva cada ROI em 48x48 | `recursos/foto_grupo.jpg` | `saidas/capturas_48x48/face_NNN.png` | `utils.py` |
| `04_item_a_tempo_real.py` | Item A: Haar em vídeo ou webcam com perfil sensível ou específico, salva ROIs, mede taxa contra gabarito | `--source` (padrão `video_teste.mp4`), `gabarito_video.csv` opcional | `saidas/capturas_48x48/<config>/face_NNNNN.png`, vídeo opcional | `utils.py` |
| `05_comparar_parametros.py` | Roda os dois perfis no vídeo inteiro e tabula taxa de detecção e falsos positivos | `recursos/video_teste.mp4`, `gabarito_video.csv` (obrigatório) | `saidas/05_comparacao_parametros.csv` | `utils.py` |
| `06_cadastrar_identidades.py` | Gera embeddings de Ana, Bruno e Carla e salva a base | `recursos/identidades/{Ana,Bruno,Carla}.jpg` | `saidas/encodings.pkl` | `face_recognition`, `utils.py` |
| `07_reconhecer_imagem.py` | Reconhece rostos na foto de grupo com tolerância 0,50 e marca desconhecidos | `saidas/encodings.pkl`, `recursos/foto_grupo.jpg` | `saidas/07_grupo_reconhecido.jpg` | script 06 |
| `08_item_b_video.py` | Item B: reconhecimento em vídeo ou webcam com escala reduzida, latência média, p95 e FPS | `saidas/encodings.pkl`, `--source` | vídeo opcional via `--output-video` | script 06 |
| `09_benchmark_embarcado.py` | Compara quatro perfis de escala e salto de frames em 80 frames | `recursos/video_teste.mp4` | só terminal | `face_recognition`, `utils.py` |
| `10_varrer_tolerancia.py` | Aplica tolerâncias 0,40, 0,50 e 0,60 na foto de grupo | `saidas/encodings.pkl`, `recursos/foto_grupo.jpg` | só terminal | script 06 |

## 5. Passo a passo para executar

Antes de tudo, monte a pasta de recursos na raiz da aula, ao lado de `exemplos_em_python/`:

```
aula_06_Reconhecimento_Facial/
├── recursos/
│   ├── foto_grupo.jpg                       (grupo com os três cadastrados e uma quarta pessoa)
│   ├── video_teste.mp4
│   ├── gabarito_video.csv                   (opcional: frame,rostos_esperados)
│   ├── haarcascade_frontalface_default.xml  (opcional)
│   └── identidades/
│       ├── Ana.jpg
│       ├── Bruno.jpg
│       └── Carla.jpg
└── exemplos_em_python/
```

Abra o terminal dentro de `exemplos_em_python/`.

**Passo 1: validar o ambiente.**

```
python 00_validar_ambiente.py
python 01_carregar_haar.py
```

Espere ver as versões, `face_recognition: OK` e `OK` para os três arquivos. Se aparecer `FALHA` para `dlib`, os scripts 06 a 10 não vão rodar, mas os de 02 a 05 sim.

**Passo 2: detecção em imagem estática.**

```
python 02_detectar_imagem.py
python 03_extrair_roi_48x48.py
```

O 02 grava `saidas/02_deteccao_imagem.jpg` com a caixa em `Ana.jpg` e imprime o número de rostos. O 03 cria `saidas/capturas_48x48/` com um PNG de 48x48 por rosto da foto de grupo e imprime shape, mínimo e máximo de cada um.

**Passo 3: Item A, detecção em vídeo ou webcam.**

```
python 04_item_a_tempo_real.py
python 04_item_a_tempo_real.py --config especifica
python 04_item_a_tempo_real.py --source 0
python 04_item_a_tempo_real.py --headless --max-frames 200 --output-video ../saidas/item_a.mp4
```

Abre a janela `Item A - Haar` com caixas verdes e um rótulo de status no canto. Tecla `q` encerra. Cada ROI detectada é salva em `saidas/capturas_48x48/sensivel/` ou `especifica/` (use `--save-every 5` para salvar menos). No fim imprime frames processados, ROIs salvas e, se houver gabarito, taxa de detecção e falsos positivos estimados.

**Passo 4: comparação controlada dos parâmetros.**

```
python 05_comparar_parametros.py
```

Exige `gabarito_video.csv`. Imprime uma tabela com os dois perfis e grava `saidas/05_comparacao_parametros.csv`.

**Passo 5: cadastrar identidades.**

```
python 06_cadastrar_identidades.py
```

Imprime `Cadastrado: Ana | embedding=(128,) | Ana.jpg` para cada pessoa e grava `saidas/encodings.pkl`. Cada foto precisa ter exatamente um rosto.

**Passo 6: reconhecer em imagem.**

```
python 07_reconhecer_imagem.py
python 10_varrer_tolerancia.py
```

O 07 grava `saidas/07_grupo_reconhecido.jpg` com nomes em verde e `Desconhecido` em vermelho, e imprime a melhor distância de cada rosto. O 10 imprime três linhas, uma por tolerância, mostrando como os rótulos mudam.

**Passo 7: Item B, reconhecimento em vídeo ou webcam.**

```
python 08_item_b_video.py
python 08_item_b_video.py --source 0 --tolerance 0.45 --scale 0.5
python 08_item_b_video.py --headless --max-frames 100 --output-video ../saidas/item_b.mp4
```

Abre a janela `Item B - reconhecimento facial` com nomes, distâncias e latência no topo. Tecla `q` encerra. No fim imprime latência média, p95 e FPS teórico.

**Passo 8: benchmark embarcado.**

```
python 09_benchmark_embarcado.py
```

Imprime uma tabela com os quatro perfis e a latência e o FPS efetivo de cada um. Demora alguns minutos em CPU.

## 6. Exercícios e itens da aula

Os scripts referenciam dois itens:

- **Item A (detecção):** `04_item_a_tempo_real.py`. Pede para aplicar o Haar Cascade em vídeo ou webcam, desenhar a bounding box de cada rosto, extrair a ROI normalizada em 48x48 e comparar um perfil sensível com um específico, medindo taxa de detecção e falsos positivos. Os scripts 01, 02, 03 e 05 são as etapas preparatórias e a medição controlada.
- **Item B (reconhecimento):** `08_item_b_video.py`. Pede para reconhecer, em vídeo ou webcam, três pessoas cadastradas e rejeitar uma quarta, medindo a latência por frame. Os scripts 06, 07, 09 e 10 são o cadastro, a versão em imagem, o benchmark de desempenho e a análise do limiar.

Exercícios extras para fixar:

1. Crie seu próprio `gabarito_video.csv` para um vídeo curto contando os rostos em cada frame, e rode o 05 para ver qual perfil vence no seu vídeo.
2. Adicione um terceiro perfil em `CONFIGS` (por exemplo, scaleFactor 1,10 e minNeighbors 5) e compare com os outros dois.
3. Troque `model="hog"` por `model="cnn"` em `face_locations` e meça a diferença de latência com o 08 (sem GPU vai ficar muito mais lento).
4. No 08, grave a menor distância de cada frame em um CSV e trace um histograma para escolher a tolerância com base em dados, como o 10 sugere.
5. Cadastre uma quarta pessoa no 06 e verifique se ela deixa de ser `Desconhecido` no 07.

## 7. Armadilhas e dicas

- **Rodar da pasta errada.** `from utils import ...` só funciona com o terminal em `exemplos_em_python/`. E `recursos/` e `saidas/` ficam um nível acima, em `aula_06_Reconhecimento_Facial/`.
- **Recursos ausentes.** Sem `recursos/` nada de 02 em diante roda. O 02 lança `FileNotFoundError`, o 03 lança `RuntimeError`. Use o 00 para listar o que falta.
- **Script 05 sem gabarito.** `load_ground_truth` devolve `{}` se o CSV não existir, e o 05 faz `truth[frame_idx]`, que lança `KeyError` no primeiro frame. O 04 usa `truth.get` e tolera a ausência. Já com `--source 0` o 04 nem tenta carregar o gabarito.
- **Script 05 com vídeo mais longo que o gabarito.** O mesmo `truth[frame_idx]` quebra se o CSV tiver menos linhas do que frames.
- **Ordem obrigatória do reconhecimento.** 07, 08 e 10 leem `saidas/encodings.pkl`. Rode o 06 primeiro. O 07 e o 08 avisam com mensagem clara; o 10 lança `FileNotFoundError` direto.
- **Uma pessoa por foto de identidade.** O 06 aborta se encontrar zero ou mais de um rosto em `Ana.jpg`, `Bruno.jpg` ou `Carla.jpg`.
- **BGR versus RGB.** Sempre converta com `COLOR_BGR2RGB` antes de passar um frame do OpenCV para `face_recognition`. O resultado errado não dá erro, só piora as distâncias.
- **Ordem das coordenadas.** Haar devolve `(x, y, w, h)`; `face_recognition` devolve `(top, right, bottom, left)`. Misturar as duas produz caixas em lugar errado.
- **Escala e coordenadas.** Se reduzir o frame para inferir, multiplique as coordenadas por `1/scale` antes de desenhar, como o 08 faz.
- **Instalação do dlib no Windows.** Precisa de CMake e compilador C++, ou de um wheel pré-compilado para a sua versão de Python. O professor usou Python 3.12.
- **Mensagens com travessão.** O 00 imprime `FALHA — {exc}` com um travessão Unicode. Em terminal com codificação antiga isso pode aparecer trocado, mas não afeta o resultado. Mantido por fidelidade ao professor.
- **Haar local opcional.** Se copiar o XML para `recursos/`, `load_haar` usa essa cópia. O 01 sempre usa a do OpenCV.
- **Equalização não é universal.** O 03 detecta sem `equalizeHist`. Experimente adicionar e ver se o número de rostos muda na sua foto de grupo.
- **Parâmetros que valem variar:** `scaleFactor`, `minNeighbors`, `minSize` (detecção); `--tolerance`, `--scale`, salto de frames `every` (reconhecimento); taxa de 20 FPS fixa no `VideoWriter`, que pode não bater com a do vídeo original.
- **Webcam.** Índice 0 é a câmera padrão. Se não abrir, `open_source` lança `RuntimeError`; tente `--source 1`.

## 8. Glossário

| Termo | Significado |
|---|---|
| Haar Cascade | Detector clássico de objetos (Viola-Jones) baseado em características retangulares e uma cascata de estágios de rejeição rápida. |
| `cv2.data.haarcascades` | Pasta, dentro da instalação do OpenCV, com os XML dos modelos Haar já treinados. |
| `detectMultiScale` | Método que varre a imagem em várias escalas e devolve retângulos `(x, y, w, h)` dos objetos encontrados. |
| scaleFactor | Fator de crescimento da janela de busca entre escalas. Menor é mais sensível e mais lento. |
| minNeighbors | Número mínimo de retângulos vizinhos que precisam concordar para aceitar uma detecção. |
| minSize | Menor tamanho de objeto considerado, em pixels. |
| Pirâmide de escalas | Sequência de versões da imagem em tamanhos decrescentes, usada para achar objetos de vários tamanhos. |
| Equalização de histograma | Redistribuição dos níveis de cinza para aumentar o contraste. `cv2.equalizeHist`. |
| BGR / RGB | Ordem dos canais de cor. OpenCV usa BGR; `face_recognition` usa RGB. |
| ROI | Region of Interest, região de interesse. Recorte de uma parte da imagem, aqui o rosto. |
| Bounding box | Retângulo que delimita um objeto detectado. |
| `INTER_AREA` | Interpolação do `cv2.resize` que faz média de pixels, indicada para reduzir imagens. |
| Frame | Um quadro individual de um vídeo. |
| `VideoCapture` / `VideoWriter` | Classes do OpenCV para ler e gravar vídeo ou webcam. |
| fourcc | Código de quatro letras que identifica o codec de vídeo, aqui `mp4v`. |
| Headless | Execução sem janela gráfica, para servidores ou scripts automatizados. |
| argparse | Módulo padrão do Python para ler argumentos de linha de comando. |
| Sensibilidade | Capacidade de encontrar os casos positivos (rostos). Alta sensibilidade perde poucos rostos. |
| Especificidade | Capacidade de rejeitar os casos negativos. Alta especificidade gera poucos falsos positivos. |
| Falso positivo | Detecção onde não há rosto. |
| Falso aceite | Reconhecer um estranho como pessoa cadastrada. |
| Falso rejeite | Recusar uma pessoa cadastrada como desconhecida. |
| Gabarito (ground truth) | Contagem verdadeira de rostos por frame, usada para avaliar o detector. |
| `face_recognition` | Biblioteca Python sobre `dlib` que localiza rostos e gera embeddings de 128 dimensões. |
| dlib | Biblioteca C++ de aprendizado de máquina que fornece o detector HOG e a rede de embeddings. |
| HOG | Histogram of Oriented Gradients, descritor de forma usado pelo detector de rostos em CPU (`model="hog"`). |
| Embedding | Vetor numérico (aqui 128-D) que representa um rosto de modo que rostos parecidos fiquem próximos. |
| `face_locations` | Função que devolve tuplas `(top, right, bottom, left)` dos rostos. |
| `face_encodings` | Função que gera um embedding por rosto localizado. |
| `face_distance` | Distância euclidiana entre um embedding e cada embedding de uma lista. |
| Tolerância | Limiar de distância abaixo do qual dois embeddings são considerados a mesma pessoa. |
| pickle | Módulo padrão do Python para serializar objetos em arquivo. Usado em `encodings.pkl`. |
| Latência | Tempo gasto para processar um frame, aqui em milissegundos. |
| p95 | Percentil 95: valor abaixo do qual ficam 95 por cento das medições. |
| FPS | Frames por segundo. O FPS teórico é `1000 / latência média em ms`. |
| Salto de frames | Processar só um a cada N frames para reduzir carga de inferência. |
| `perf_counter` | Relógio de alta resolução do módulo `time`, usado para medir tempo. |
| Biometria | Identificação de pessoas por características físicas, como o rosto. |
