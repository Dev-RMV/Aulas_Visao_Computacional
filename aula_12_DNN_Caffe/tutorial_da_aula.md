# Aula 12 – DNN e Caffe

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

A aula ensina a montar um sistema completo de estimativa de gênero aparente e faixa etária a partir de rostos, e a comparar duas estratégias para isso. A primeira estratégia usa modelos prontos no formato Caffe (age_net e gender_net), carregados pelo módulo `cv2.dnn` do OpenCV, sem nenhum treinamento. A segunda estratégia faz transfer learning: pega uma MobileNetV2 pré-treinada no Keras, congela o tronco e treina apenas uma cabeça nova para classificar gênero usando o dataset UTKFace.

O fio condutor dos 35 scripts é o pipeline "detectar rosto com Haar Cascade, recortar a ROI, classificar, desenhar o rótulo, medir FPS". Os scripts 01 a 17 constroem esse pipeline com os modelos Caffe (Item A da aula). Os scripts 18 a 31 preparam o UTKFace, treinam e avaliam a MobileNetV2 e produzem a tabela comparativa (Item B). Os scripts 32 a 35 fecham com os pipelines finais, um comentário sobre uso embarcado e ética, e a conversão para TensorFlow Lite.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros importadas pelos scripts: `cv2` (OpenCV, com o módulo `dnn`), `numpy`, `matplotlib`, `tensorflow` (Keras e TFLite), `pandas` e `sklearn` (scikit-learn).

```
pip install opencv-python numpy matplotlib tensorflow pandas scikit-learn
```

O professor rodou com Python 3.12 (o zip trazia `__pycache__` dessa versão). O TensorFlow no Windows nativo usa só CPU, e é isso que o script 01 confirma ao listar as GPUs.

Layout de pastas. Todo script começa com `from _bootstrap import PROJECT_ROOT`. O `_bootstrap.py` define a raiz do projeto como a pasta acima da dos scripts, insere essa raiz no `sys.path` e faz `os.chdir` para lá. No nosso layout a raiz é `aula_12_DNN_Caffe/`. Portanto:

- `data/`, `models/` e `resultados/` ficam em `aula_12_DNN_Caffe/`, não dentro de `exemplos_em_python/`.
- O pacote `utils/` precisa ficar em `aula_12_DNN_Caffe/utils/`.
- Pode-se rodar os scripts de qualquer pasta, porque o bootstrap ajusta o diretório de trabalho. O comando mais simples é `python exemplos_em_python/NN_nome.py` a partir de `aula_12_DNN_Caffe/`.

Material que não veio no zip (ver `achados_dos_zips.md`):

- O pacote `utils/` com `face_utils.py` e `tf_utils.py`. Sem ele, só rodam os scripts 01, 02, 03, 22, 23, 26, 27, 28, 34 e 35. A seção 7b documenta cada função esperada para permitir reconstruir o pacote.
- Os quatro arquivos dos modelos Caffe em `models/opencv_age_gender/`: `age_deploy.prototxt`, `age_net.caffemodel`, `gender_deploy.prototxt`, `gender_net.caffemodel`. O script 08 lista as URLs de origem.
- `data/faces_teste/` com `pessoa.jpg` e `grupo.jpg`.
- `data/utkface_sample/` com até 1000 imagens do UTKFace.
- Um arquivo de vídeo para os scripts 14, 15 e 32 (o docstring sugere `data/video.mp4`).

Webcam: usada pelos scripts 05, 13 e 32 (este último aceita vídeo no lugar da webcam).

## 3. Conceitos-chave

### 3.1 Bootstrap: raiz do projeto, sys.path e diretório de trabalho

Quando um projeto tem um pacote próprio (`utils/`) e pastas de dados, dois problemas aparecem ao rodar scripts soltos: o Python não encontra o pacote, e os caminhos relativos mudam conforme a pasta de onde se chamou o script. O professor resolve os dois com um módulo importado na primeira linha de cada exemplo. `Path(__file__).resolve().parents[1]` sobe um nível a partir do arquivo do script. `sys.path.insert(0, ...)` faz `import utils` funcionar. `os.chdir` fixa o diretório de trabalho na raiz, então `data/faces_teste/pessoa.jpg` sempre resolve para o mesmo lugar.

Trecho de `_bootstrap.py`:

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]

root_str = str(PROJECT_ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

os.chdir(PROJECT_ROOT)
```

Repare que em todos os scripts a linha `from _bootstrap import PROJECT_ROOT` vem antes do docstring. Por isso o docstring não é reconhecido como docstring do módulo, é só uma string solta. O único script em que isso importa é o 34, que faz `print(__doc__)`.

Onde aparece: `_bootstrap.py` e todos os 35 scripts.

### 3.2 Verificação de ambiente e detecção de GPU

Antes de qualquer coisa, imprimir as versões das bibliotecas evita perder tempo com incompatibilidades. O TensorFlow é importado dentro de `try/except` porque é a dependência mais pesada e a que mais falha ao instalar. `tf.config.list_physical_devices('GPU')` retorna a lista de GPUs visíveis; no Windows nativo a lista vem vazia.

Trecho de `01_verificar_ambiente.py`:

```python
import cv2, numpy as np, matplotlib
print("OpenCV:", cv2.__version__)
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("GPU detectada:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("TensorFlow nao carregou:", e)
```

Onde aparece: `01_verificar_ambiente.py`.

### 3.3 Estrutura de pastas com pathlib

`Path(p).mkdir(parents=True, exist_ok=True)` cria a pasta e todos os pais, sem erro se já existir. O script 02 cria as quatro pastas do projeto de uma vez. Vários scripts posteriores (20, 21, 22, 23, 25) gravam em `resultados/` sem chamar `ensure_dir`, então o 02 precisa ter rodado antes.

Trecho de `02_estrutura_de_pastas.py`:

```python
for p in ["data/faces_teste", "data/utkface_sample", "models/opencv_age_gender", "resultados"]:
    Path(p).mkdir(parents=True, exist_ok=True)
    print("OK:", p)
```

Onde aparece: `02_estrutura_de_pastas.py`.

### 3.4 Haar Cascade: localizar o classificador e detectar rostos

O Haar Cascade é um detector clássico de objetos treinado com características de Haar (diferenças de soma de pixels em retângulos) e um classificador em cascata. O OpenCV distribui o XML `haarcascade_frontalface_default.xml` na pasta `cv2.data.haarcascades`, então não é preciso baixar nada. A detecção em si fica encapsulada em `detect_faces_haar(img)` do pacote `utils/`, que recebe a imagem BGR e retorna uma lista de caixas. Cada caixa é usada como `box` em `draw_label` e `crop_face`. O formato exato (provavelmente uma tupla `(x, y, w, h)`) é definido no pacote.

Trecho de `03_localizar_haar_cascade.py`:

```python
path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
print("Arquivo Haar Cascade:", path)
print("Existe?", path.exists())
```

Trecho de `04_detectar_rosto_imagem.py`:

```python
img = load_image(args.imagem)
faces = detect_faces_haar(img)
print(f"Rostos detectados: {len(faces)}")
for i, box in enumerate(faces, start=1):
    draw_label(img, box, f"Rosto {i}")
ensure_dir("resultados")
cv2.imwrite("resultados/04_rostos_detectados.jpg", img)
```

Onde aparece: `03_localizar_haar_cascade.py`, `04_detectar_rosto_imagem.py`, e depois em 05, 06, 07, 12, 13, 14, 15, 16, 17 e 32.

### 3.5 Argumentos de linha de comando com argparse

Quase todos os scripts aceitam a imagem, o vídeo ou a pasta de entrada como argumento, com um valor padrão apontando para `data/`. `parser.add_argument("--imagem", default=...)` cria a opção; `required=True` obriga o usuário a informar (usado em `--video`). No script 32, `--fonte` aceita tanto `0` (webcam) quanto um caminho, e a conversão `int(args.fonte) if args.fonte.isdigit() else args.fonte` decide qual passar ao `VideoCapture`.

Trecho de `32_pipeline_final_item_a.py`:

```python
parser = argparse.ArgumentParser(); parser.add_argument("--fonte", default="0")
args = parser.parse_args()
source = int(args.fonte) if args.fonte.isdigit() else args.fonte
cap = cv2.VideoCapture(source)
```

Onde aparece: 04, 06, 07, 10, 11, 12, 14, 15, 17, 19, 20, 21 e 32.

### 3.6 Captura de webcam, loop de frames e FPS médio

`cv2.VideoCapture(0)` abre a câmera principal. `cap.isOpened()` confere se abriu. O loop lê com `cap.read()`, que retorna `(ok, frame)`. O FPS médio é calculado de forma simples: número de frames dividido pelo tempo decorrido desde o início, medido com `time.perf_counter()`. O `max(1e-6, ...)` evita divisão por zero no primeiro frame. `cv2.putText` escreve o FPS no canto, `cv2.imshow` exibe, e `cv2.waitKey(1) & 0xFF == ord('q')` encerra com a tecla Q. Ao final, `release()` e `destroyAllWindows()` liberam a câmera e as janelas.

Trecho de `05_detectar_rosto_webcam.py`:

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Nao foi possivel abrir a webcam.")
frames, t0 = 0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok: break
    faces = detect_faces_haar(frame)
    for box in faces:
        draw_label(frame, box, "Rosto")
    frames += 1
    fps = frames / max(1e-6, (time.perf_counter() - t0))
    cv2.putText(frame, f"FPS medio: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.imshow("Haar Cascade - Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
```

Onde aparece: `05_detectar_rosto_webcam.py`, `13_webcam_caffe_fps.py`, `14_video_arquivo_caffe_fps.py`, `15_salvar_video_anotado.py`, `32_pipeline_final_item_a.py`.

### 3.7 ROI facial com padding

A rede de idade e gênero não recebe a imagem inteira, recebe só o rosto. `crop_face(img, box, padding=20)` recorta o retângulo detectado com uma margem de 20 pixels em volta, o que ajuda porque o Haar tende a apertar demais a caixa e os modelos Caffe foram treinados com rostos um pouco mais folgados. As ROIs são salvas em `resultados/rois/face_NN.jpg` e servem de entrada para o script 11.

Trecho de `06_recortar_roi_facial.py`:

```python
faces = detect_faces_haar(img)
ensure_dir("resultados/rois")
for i, box in enumerate(faces):
    roi = crop_face(img, box, padding=20)
    out = f"resultados/rois/face_{i:02d}.jpg"
    cv2.imwrite(out, roi)
    print("ROI salva:", out, roi.shape)
```

Onde aparece: `06_recortar_roi_facial.py`, e `crop_face` sem padding explícito em 12, 13, 14, 15, 16, 17 e 32.

### 3.8 Desenhar bounding box e rótulo (mock antes da rede)

O professor separa a parte visual da parte de inferência. O script 07 desenha o rótulo "Genero: ? | Idade: ?" sobre cada rosto, sem carregar nenhuma rede, só para validar que `draw_label(img, box, texto)` funciona. Depois, nos scripts 12 em diante, o texto passa a ser o resultado real da predição.

Trecho de `07_desenhar_rotulos_mock.py`:

```python
for box in faces:
    draw_label(img, box, "Genero: ? | Idade: ?")
ensure_dir("resultados")
cv2.imwrite("resultados/07_rotulos_mock.jpg", img)
```

Onde aparece: `07_desenhar_rotulos_mock.py`.

### 3.9 Modelos Caffe: arquivos prototxt e caffemodel

Um modelo Caffe é distribuído em dois arquivos: o `.prototxt` descreve a arquitetura da rede (camadas, tamanhos) em texto, e o `.caffemodel` contém os pesos treinados em binário. Os modelos de idade e gênero desta aula são os de Gil Levi e Tal Hassner, e os `.prototxt` de deploy vêm do repositório learnopencv. O zip não os inclui por tamanho e licença. `required_caffe_files()` devolve um dicionário nome -> `Path` para os quatro arquivos, e o script 08 imprime se cada um existe.

Trecho de `08_baixar_modelos_instrucoes.py`:

```python
#https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
#https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel
#https://github.com/spmallick/learnopencv/blob/master/AgeGender/age_deploy.prototxt
#https://github.com/spmallick/learnopencv/blob/master/AgeGender/gender_deploy.prototxt
for name, path in required_caffe_files().items():
    print(f"{name:12s}: {path} | existe={path.exists()}")
```

Onde aparece: `08_baixar_modelos_instrucoes.py`.

### 3.10 Carregar redes com cv2.dnn

O módulo `cv2.dnn` executa redes treinadas em outros frameworks (Caffe, TensorFlow, ONNX, Darknet) sem precisar instalar esses frameworks. Para Caffe a chamada é `cv2.dnn.readNetFromCaffe(prototxt, caffemodel)`, que devolve um objeto `cv2.dnn.Net`. Nos scripts isso está encapsulado em `load_caffe_age_gender()`, que retorna a tupla `(age_net, gender_net)`. O script 09 só imprime o tipo dos objetos para confirmar o carregamento.

Trecho de `09_carregar_modelos_caffe.py`:

```python
from utils.face_utils import load_caffe_age_gender
age_net, gender_net = load_caffe_age_gender()
print("age_net:", type(age_net))
print("gender_net:", type(gender_net))
```

Onde aparece: `09_carregar_modelos_caffe.py`, e depois em 11, 12, 13, 14, 15, 16, 17 e 32.

### 3.11 Blob: pré-processamento para a rede

Uma rede convolucional espera a entrada em um formato fixo. O "blob" é o tensor de 4 dimensões `(N, C, H, W)`: lote, canais, altura, largura. `make_age_gender_blob(img)` faz o que `cv2.dnn.blobFromImage` faria: redimensiona o rosto para 227 x 227 (tamanho de entrada das redes de Levi e Hassner), subtrai a média por canal usada no treinamento e reorganiza os canais de HWC para CHW. O script 10 imprime o shape esperado `(1, 3, 227, 227)`, o `dtype` e a faixa de valores, para o aluno enxergar o efeito da subtração de média (valores negativos aparecem).

Trecho de `10_blob_para_caffe.py`:

```python
img = load_image(args.imagem)
blob = make_age_gender_blob(img)
print("Shape do blob:", blob.shape)  # esperado: (1, 3, 227, 227)
print("dtype:", blob.dtype, "min:", blob.min(), "max:", blob.max())
```

Onde aparece: `10_blob_para_caffe.py`. Internamente, `predict_age_gender` deve usar a mesma função.

### 3.12 Inferência de gênero e faixa etária

Com o blob pronto, a inferência em `cv2.dnn` é `net.setInput(blob)` seguido de `net.forward()`, que devolve um vetor de probabilidades. gender_net tem 2 saídas (Masculino, Feminino) e age_net tem 8 saídas, uma por faixa etária no formato `(25-32)`. `predict_age_gender(face, age_net, gender_net)` encapsula isso e devolve um objeto com os atributos `gender`, `gender_conf`, `age` e `age_conf`. O script 11 roda sobre uma ROI já recortada (a saída do 06) e imprime o objeto inteiro.

Trecho de `11_inferir_genero_idade_roi.py`:

```python
face = load_image(args.imagem)          # padrão: resultados/rois/face_00.jpg
age_net, gender_net = load_caffe_age_gender()
pred = predict_age_gender(face, age_net, gender_net)
print(pred)
```

Trecho de `12_imagem_completa_caffe.py`, onde os quatro atributos viram o texto do rótulo:

```python
for box in faces:
    roi = crop_face(img, box)
    pred = predict_age_gender(roi, age_net, gender_net)
    label = f"{pred.gender} {pred.gender_conf:.2f} | {pred.age} {pred.age_conf:.2f}"
    draw_label(img, box, label)
```

Onde aparece: 11, 12, 13, 14, 15, 16, 17 e 32.

### 3.13 Pipeline completo em imagem estática

O script 12 junta todas as peças anteriores: carrega a imagem, carrega as duas redes uma única vez (fora do loop, porque carregar é caro), detecta os rostos, e para cada rosto recorta, prediz e desenha. A imagem anotada vai para `resultados/12_pipeline_imagem.jpg`. O padrão de entrada é `grupo.jpg`, para exercitar várias detecções na mesma foto.

Onde aparece: `12_imagem_completa_caffe.py`.

### 3.14 Pipeline em vídeo: webcam e arquivo

Os scripts 13 e 14 aplicam o mesmo pipeline a cada frame. A única diferença entre eles é a fonte: `VideoCapture(0)` no 13 e `VideoCapture(args.video)` no 14. O 13 escreve o FPS no frame durante a execução; o 14 só calcula o FPS ao final, sobre o total de frames processados. Ambos encerram com Q ou quando o vídeo acaba.

Trecho de `14_video_arquivo_caffe_fps.py`:

```python
cap = cv2.VideoCapture(args.video)
frames, t0 = 0, time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok: break
    for box in detect_faces_haar(frame):
        pred = predict_age_gender(crop_face(frame, box), age_net, gender_net)
        draw_label(frame, box, f"{pred.gender} | {pred.age}")
    frames += 1
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
fps = frames / max(1e-6, time.perf_counter() - t0)
```

Onde aparece: `13_webcam_caffe_fps.py`, `14_video_arquivo_caffe_fps.py`.

### 3.15 Gravar vídeo anotado com VideoWriter

Para salvar o resultado em disco, é preciso ler as propriedades do vídeo de entrada (`CAP_PROP_FPS`, `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT`) e criar um `cv2.VideoWriter` com o codec `mp4v`, o mesmo FPS e o mesmo tamanho. O `or 25` cobre vídeos que não informam FPS. Cada frame anotado é gravado com `out.write(frame)`. Aqui não há `imshow`, o processamento é em lote. Não esquecer `out.release()`, senão o arquivo fica corrompido.

Trecho de `15_salvar_video_anotado.py`:

```python
fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(args.saida, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w,h))
while True:
    ok, frame = cap.read()
    if not ok: break
    for box in detect_faces_haar(frame):
        pred = predict_age_gender(crop_face(frame, box), age_net, gender_net)
        draw_label(frame, box, f"{pred.gender} | {pred.age}")
    out.write(frame); frames += 1
cap.release(); out.release()
```

Onde aparece: `15_salvar_video_anotado.py`.

### 3.16 Relatório de validação em CSV

Para saber se o modelo Caffe acerta, o professor pede um gabarito manual: um CSV em `data/relatorio_5_rostos.csv` com as colunas `imagem`, `genero_real`, `idade_real`. Se o arquivo não existe, o script cria um modelo com uma linha de exemplo e para (`raise SystemExit`), para o aluno preencher com cinco rostos reais. Na segunda execução, ele lê o CSV com `csv.DictReader`, processa cada imagem (usa só o primeiro rosto detectado), compara predição com gabarito e grava um CSV de saída com `csv.DictWriter`, adicionando as colunas `pred_genero`, `pred_idade`, `acerto_genero` e `acerto_idade`. Imagens sem rosto detectado recebem "NA".

Trecho de `16_relatorio_5_rostos_csv.py`:

```python
for row in csv.DictReader(f):
    img = load_image(Path("data/faces_teste") / row["imagem"])
    faces = detect_faces_haar(img)
    if not faces:
        rows_out.append({**row, "pred_genero":"NA", "pred_idade":"NA", "acerto_genero":False, "acerto_idade":False})
        continue
    pred = predict_age_gender(crop_face(img, faces[0]), age_net, gender_net)
    rows_out.append({**row, "pred_genero":pred.gender, "pred_idade":pred.age,
                     "acerto_genero":pred.gender == row["genero_real"], "acerto_idade":pred.age == row["idade_real"]})
```

Repare que a comparação de idade é por igualdade de string, então o gabarito precisa usar exatamente o formato das faixas do modelo, como `(25-32)`.

Onde aparece: `16_relatorio_5_rostos_csv.py`.

### 3.17 Benchmark de latência com aquecimento

Medir o tempo de uma única inferência engana, porque a primeira execução inclui alocações e inicializações. O padrão correto é fazer algumas chamadas de aquecimento (3 no script), depois cronometrar N repetições (30 por padrão, ajustável com `--repeticoes`) e dividir. O resultado é em milissegundos por imagem; o inverso dá o FPS teórico para uma única face, sem contar a detecção Haar.

Trecho de `17_benchmark_fps_imagem.py`:

```python
for _ in range(3): predict_age_gender(face, age_net, gender_net)
t0 = time.perf_counter()
for _ in range(args.repeticoes): predict_age_gender(face, age_net, gender_net)
lat_ms = (time.perf_counter() - t0) * 1000 / args.repeticoes
print(f"Latencia media: {lat_ms:.2f} ms/imagem")
print(f"FPS teorico em uma face: {1000/lat_ms:.2f}")
```

Onde aparece: `17_benchmark_fps_imagem.py`. O mesmo padrão é usado por `measure_keras_latency_ms` no script 30.

### 3.18 Dataset UTKFace: rótulos no nome do arquivo

O UTKFace é um dataset de rostos em que o rótulo está codificado no nome do arquivo: `idade_genero_etnia_data.jpg`. Gênero 0 é masculino e 1 é feminino. `parse_utkface_filename(nome)` faz o split do nome e devolve `(idade, genero)` como inteiros. Aceita tanto string quanto `Path` (o script 19 passa um `Path`). `collect_utkface_paths(pasta, limite)` lista até `limite` imagens válidas da pasta, como objetos `Path`.

Trecho de `18_parse_utkface_nome.py`:

```python
amostras = ["25_0_0_20170116174525125.jpg", "31_1_2_201701040202.jpg"]
for nome in amostras:
    idade, genero = parse_utkface_filename(nome)
    print(nome, "-> idade:", idade, "genero:", genero, "label:", "Feminino" if genero == 1 else "Masculino")
```

Onde aparece: `18_parse_utkface_nome.py`, `19_listar_utkface_1000.py`, `20_criar_dataframe_utkface.py`, `21_dividir_treino_validacao.py`.

### 3.19 DataFrame com pandas e balanceamento das classes

Antes de treinar, os caminhos e rótulos são organizados em um `pandas.DataFrame` com colunas `path`, `age` e `gender`. `df["gender"].value_counts()` mostra quantas imagens há de cada classe, o que serve para avaliar o balanceamento. O DataFrame vai para CSV em `resultados/20_utkface_dataframe.csv`.

Trecho de `20_criar_dataframe_utkface.py`:

```python
rows = []
for p in collect_utkface_paths(args.pasta, args.limite):
    age, gender = parse_utkface_filename(p)
    rows.append({"path": str(p), "age": age, "gender": gender})
df = pd.DataFrame(rows)
print(df["gender"].value_counts(dropna=False))
df.to_csv("resultados/20_utkface_dataframe.csv", index=False)
```

Onde aparece: `20_criar_dataframe_utkface.py`.

### 3.20 Divisão treino, validação e teste com estratificação

`train_test_split` do scikit-learn divide um DataFrame de forma aleatória. Duas chamadas produzem três conjuntos: primeiro 70 % treino e 30 % temporário, depois o temporário é dividido ao meio em validação e teste (15 % cada). `stratify=df["gender"]` mantém a proporção de classes igual em todos os conjuntos, e `random_state=42` torna a divisão reproduzível. Os três conjuntos vão para `resultados/21_train.csv`, `21_val.csv` e `21_test.csv`, que são a entrada de todos os scripts de treino e avaliação.

Trecho de `21_dividir_treino_validacao.py`:

```python
train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["gender"])
val, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["gender"])
print(len(train), len(val), len(test))
train.to_csv("resultados/21_train.csv", index=False)
```

Onde aparece: `21_dividir_treino_validacao.py`.

### 3.21 Pipeline de entrada com tf.data

`tf.data.Dataset.from_tensor_slices((caminhos, rotulos))` cria um dataset a partir de duas colunas. `map(load_sample)` aplica uma função que lê o arquivo (`tf.io.read_file`), decodifica o JPEG com 3 canais (`tf.image.decode_jpeg`), redimensiona para 160 x 160 (`tf.image.resize`) e converte o rótulo para float32, que é o formato esperado pela perda binária. `batch(16)` agrupa e `prefetch(tf.data.AUTOTUNE)` prepara o próximo lote enquanto o atual é consumido. `ds.take(1)` retira um único lote para inspecionar os shapes: imagens `(16, 160, 160, 3)` e rótulos `(16,)`.

Trecho de `22_tfdata_utkface.py`:

```python
IMG_SIZE = (160, 160)
def load_sample(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img, tf.cast(label, tf.float32)

df = pd.read_csv("resultados/21_train.csv")
ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["gender"].values))
ds = ds.map(load_sample).batch(16).prefetch(tf.data.AUTOTUNE)
```

Observação importante: nos scripts 22, 25, 27, 28 e 30 as imagens vão para o modelo na faixa 0 a 255, sem normalização. Só o 23 divide por 255, e apenas para exibir com matplotlib. Isso significa que `build_mobilenetv2_gender` precisa incluir o pré-processamento dentro do modelo (uma camada `Rescaling` ou o `preprocess_input` da MobileNetV2, que leva para a faixa -1 a 1).

Onde aparece: `22_tfdata_utkface.py`, `23_visualizar_batch_utkface.py`, `25_treinar_cabeca_10_epocas.py`, `27_avaliar_modelo_keras.py`, `28_matriz_confusao_genero.py`, `30_latencia_modelo_keras.py`.

### 3.22 Visualizar um lote com matplotlib

Olhar as imagens com os rótulos antes de treinar é a forma mais barata de pegar erro de parse ou de caminho. O script 23 monta um lote de 9 e desenha uma grade 3 x 3 com `plt.subplot`, título "F" ou "M" conforme o rótulo, e salva em `resultados/23_batch_utkface.png` com `dpi=150`. `next(iter(ds))` pega o primeiro lote sem loop.

Trecho de `23_visualizar_batch_utkface.py`:

```python
imgs, labels = next(iter(ds))
plt.figure(figsize=(7,7))
for i in range(min(9, len(imgs))):
    plt.subplot(3,3,i+1); plt.imshow(imgs[i]); plt.axis('off')
    plt.title("F" if labels[i].numpy()==1 else "M")
plt.tight_layout(); plt.savefig("resultados/23_batch_utkface.png", dpi=150)
```

Onde aparece: `23_visualizar_batch_utkface.py`.

### 3.23 Transfer learning: MobileNetV2 congelada com cabeça nova

Transfer learning reaproveita uma rede treinada em um problema grande (ImageNet) para um problema pequeno. A MobileNetV2 é escolhida por ser leve, adequada a sistemas embarcados. O tronco convolucional é congelado (`trainable = False`) e recebe por cima uma "cabeça": camadas `Dense` com `Dropout` de 0,3 e uma saída sigmoide para classificação binária de gênero. `build_mobilenetv2_gender(input_shape=(160,160,3), dropout=0.3)` constrói e compila esse modelo. O script 24 imprime o `summary()` e conta quantas camadas e quantos parâmetros são treináveis, para o aluno ver que só a cabeça treina.

Trecho de `24_modelo_mobilenetv2_cabeca.py`:

```python
model = build_mobilenetv2_gender(input_shape=(160,160,3), dropout=0.3)
model.summary()
print("Camadas treinaveis:", sum(1 for l in model.layers if l.trainable))
print("Parametros treinaveis:", sum(w.numpy().size for w in model.trainable_weights))
```

Onde aparece: `24_modelo_mobilenetv2_cabeca.py`, `25_treinar_cabeca_10_epocas.py`.

### 3.24 Treinar a cabeça com EarlyStopping e medir o tempo

O treino usa `model.fit` com o dataset de treino embaralhado (`shuffle(len(df), seed=42)`) e o de validação, por até 10 épocas. O callback `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)` interrompe quando a perda de validação não melhora por 3 épocas seguidas e restaura os melhores pesos, o que combate overfitting. O tempo total é cronometrado porque entra na tabela comparativa do Item B. O modelo é salvo no formato nativo `.keras` em `resultados/25_mobilenetv2_genero.keras`, e todos os scripts seguintes o carregam de lá.

Trecho de `25_treinar_cabeca_10_epocas.py`:

```python
train_ds=make_ds("resultados/21_train.csv", True); val_ds=make_ds("resultados/21_val.csv")
model=build_mobilenetv2_gender((160,160,3), 0.3)
early=tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
t0=time.perf_counter()
hist=model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=[early])
train_time=time.perf_counter()-t0
model.save("resultados/25_mobilenetv2_genero.keras")
```

Onde aparece: `25_treinar_cabeca_10_epocas.py`.

### 3.25 Curvas de treino e diagnóstico de overfitting

As curvas de acurácia e perda por época, para treino e validação, são a ferramenta básica de diagnóstico. O sinal de overfitting é a perda de treino continuar caindo enquanto a de validação sobe. O script 26 não lê o histórico do 25 (a variável `hist` não é salva em disco); ele usa um dicionário com valores simulados que exibem exatamente esse padrão a partir da época 4. O docstring avisa que o script deve ser adaptado para ler o histórico real.

Trecho de `26_plotar_curvas_finetuning.py`:

```python
history = {
    "accuracy": [0.60,0.70,0.76,0.80,0.83,0.85],
    "val_accuracy": [0.58,0.67,0.72,0.75,0.74,0.73],
    "loss": [0.68,0.55,0.46,0.40,0.36,0.33],
    "val_loss": [0.69,0.60,0.52,0.48,0.50,0.55]
}
plt.plot(history["loss"], label="treino"); plt.plot(history["val_loss"], label="validacao")
plt.savefig("resultados/26_curva_loss.png", dpi=150)
```

Onde aparece: `26_plotar_curvas_finetuning.py`.

### 3.26 Avaliação no conjunto de teste

`tf.keras.models.load_model` recarrega o `.keras` salvo. `model.evaluate(ds)` percorre o conjunto de teste e devolve `(loss, acc)`. O conjunto de teste nunca foi visto no treino nem na escolha da parada, então essa acurácia é a estimativa honesta de desempenho.

Trecho de `27_avaliar_modelo_keras.py`:

```python
df=pd.read_csv("resultados/21_test.csv")
ds=tf.data.Dataset.from_tensor_slices((df["path"].values, df["gender"].values)).map(load_sample).batch(BATCH)
model=tf.keras.models.load_model("resultados/25_mobilenetv2_genero.keras")
loss, acc = model.evaluate(ds)
```

Onde aparece: `27_avaliar_modelo_keras.py`.

### 3.27 Matriz de confusão e relatório de classificação

A acurácia esconde para qual classe o modelo erra. A matriz de confusão do scikit-learn cruza rótulo real com predito. `classification_report` dá precisão, recall e F1 por classe. Como a saída do modelo é uma probabilidade (sigmoide), a predição é obtida com o limiar 0,5: `(prob >= 0.5).astype(int)`. Aqui o professor não usa `tf.data`; ele empilha todas as imagens de teste em um único tensor com `tf.stack` e chama `model.predict` uma vez. `.ravel()` achata a saída `(N, 1)` para `(N,)`.

Trecho de `28_matriz_confusao_genero.py`:

```python
imgs=tf.stack([load_img(p) for p in df["path"].values])
prob=model.predict(imgs, verbose=0).ravel()
pred=(prob>=0.5).astype(int)
print(confusion_matrix(df["gender"].values, pred))
print(classification_report(df["gender"].values, pred, target_names=["Masculino", "Feminino"]))
```

Onde aparece: `28_matriz_confusao_genero.py`.

### 3.28 Tamanho do modelo em disco e latência Keras

Para sistemas embarcados, dois números importam além da acurácia: quanto o modelo ocupa e quanto tempo leva por imagem. `model_size_mb(path)` lê o tamanho do arquivo `.keras` em megabytes. `measure_keras_latency_ms(model, sample, repeat=30)` repete a inferência sobre uma imagem (array NumPy 160 x 160 x 3) e devolve a média em milissegundos, seguindo o mesmo padrão de aquecimento do script 17.

Trecho de `30_latencia_modelo_keras.py`:

```python
sample=load_img(df.iloc[0]["path"])     # devolve img.numpy()
lat=measure_keras_latency_ms(model, sample, repeat=30)
print(f"Latencia MobileNetV2 fine-tuned: {lat:.2f} ms/imagem")
```

Onde aparece: `29_salvar_modelo_e_tamanho.py`, `30_latencia_modelo_keras.py`.

### 3.29 Tabela comparativa: Caffe pré-treinado versus MobileNetV2 fine-tuned

O Item B pede uma tabela com acurácia, tempo de treino, tamanho e latência dos dois modelos. O script 31 monta a tabela em um DataFrame e salva em `resultados/31_tabela_comparativa.csv`. Os valores de acurácia, tempo de treino e latência estão fixos como exemplo (0,80 e 0,87 de acurácia, 320 s, 35 ms e 18 ms) e devem ser substituídos pelos números obtidos nos scripts 16, 17, 25, 27 e 30. Só o tamanho do modelo Keras é lido de verdade, com `model_size_mb`.

Trecho de `31_comparar_caffe_finetuned.py`:

```python
acc_caffe = 0.80
acc_ft = 0.87
train_time_s = 320.0
size_ft = model_size_mb(keras_model) if keras_model.exists() else None
rows = [
    {"modelo":"OpenCV Caffe pre-treinado", "acuracia":acc_caffe, "tempo_treino_s":0, "tamanho_MB":"depende dos .caffemodel", "latencia_ms":lat_caffe_ms},
    {"modelo":"MobileNetV2 fine-tuned", "acuracia":acc_ft, "tempo_treino_s":train_time_s, "tamanho_MB":size_ft, "latencia_ms":lat_ft_ms},
]
```

Onde aparece: `31_comparar_caffe_finetuned.py`.

### 3.30 Pipelines finais: fonte unificada e orquestração com subprocess

O script 32 é a versão definitiva do pipeline Caffe, com uma única opção `--fonte` que aceita webcam ou arquivo (ver 3.5). O script 33 é a versão definitiva do Item B: ele não repete código, chama os scripts 19, 20, 21, 24, 25, 27, 29, 30 e 31 em sequência com `subprocess.run([sys.executable, ...], check=True)`. `sys.executable` garante que o mesmo interpretador (o do venv) seja usado, e `check=True` interrompe a cadeia no primeiro erro.

Trecho de `33_pipeline_final_item_b.py`:

```python
scripts = ["19_listar_utkface_1000.py", "20_criar_dataframe_utkface.py", "21_dividir_treino_validacao.py",
           "24_modelo_mobilenetv2_cabeca.py", "25_treinar_cabeca_10_epocas.py", "27_avaliar_modelo_keras.py",
           "29_salvar_modelo_e_tamanho.py", "30_latencia_modelo_keras.py", "31_comparar_caffe_finetuned.py"]
for s in scripts:
    print("\n=== Executando", s, "===")
    subprocess.run([sys.executable, "exemplos/" + s], check=True)
```

Atenção: o caminho está fixo em `exemplos/`, o nome da pasta do professor. No nosso layout a pasta é `exemplos_em_python/`, então o 33 falha como está (ver seção 7).

Onde aparece: `32_pipeline_final_item_a.py`, `33_pipeline_final_item_b.py`.

### 3.31 Modelo fixo ou fine-tuning: critérios técnicos e éticos

O script 34 é um texto, não código. Ele lista quando preferir um modelo pré-treinado fixo (pouca base rotulada, prototipagem rápida, hardware limitado, manutenção barata) e quando fazer fine-tuning (ambiente do robô muito diferente do dataset original em câmera, iluminação, ângulo, população e resolução; amostras locais rotuladas com consentimento; erro do modelo fixo inaceitável; possibilidade de validar viés, desempenho por subgrupo e latência no hardware real). Fecha com três cuidados: gênero inferido por aparência não é identidade de gênero; a idade é uma faixa com alta incerteza; em robótica esses atributos só devem ser usados com justificativa, consentimento e plano de mitigação de viés. Por causa da linha de import antes do docstring (ver 3.1), o `print(__doc__)` imprime `None`, não o texto.

Onde aparece: `34_comentario_embarcado_etica.py`.

### 3.32 Conversão para TensorFlow Lite

O TFLite é o formato do TensorFlow para dispositivos móveis e embarcados. `tf.lite.TFLiteConverter.from_keras_model(model)` cria o conversor; `converter.optimizations = [tf.lite.Optimize.DEFAULT]` ativa a quantização padrão, que reduz o tamanho (tipicamente para cerca de um quarto, ao converter pesos float32 para int8). O resultado de `convert()` é um `bytes` gravado com `write_bytes`. O script compara o tamanho do `.tflite` com o do `.keras` medido no 29.

Trecho de `35_converter_para_tflite.py`:

```python
model = tf.keras.models.load_model(model_path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
out = Path("resultados/35_genero_mobilenetv2.tflite")
out.write_bytes(tflite_model)
print(f"Tamanho TFLite: {out.stat().st_size / (1024*1024):.2f} MB")
```

Onde aparece: `35_converter_para_tflite.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `_bootstrap.py` | Define a raiz do projeto, ajusta `sys.path` e `chdir` | nenhuma | nenhuma | nada |
| `01_verificar_ambiente.py` | Imprime versões de OpenCV, NumPy, Matplotlib, TensorFlow e GPUs | nenhuma | terminal | nada |
| `02_estrutura_de_pastas.py` | Cria `data/faces_teste`, `data/utkface_sample`, `models/opencv_age_gender`, `resultados` | nenhuma | pastas | nada |
| `03_localizar_haar_cascade.py` | Mostra o caminho do XML Haar em `cv2.data` | nenhuma | terminal | nada |
| `04_detectar_rosto_imagem.py` | Detecta rostos e desenha caixas | `--imagem` (padrão `data/faces_teste/pessoa.jpg`) | `resultados/04_rostos_detectados.jpg` | utils, pessoa.jpg |
| `05_detectar_rosto_webcam.py` | Detecção Haar em tempo real com FPS | webcam | janela | utils, webcam |
| `06_recortar_roi_facial.py` | Recorta cada rosto com padding 20 | `--imagem` | `resultados/rois/face_NN.jpg` | utils, pessoa.jpg |
| `07_desenhar_rotulos_mock.py` | Desenha rótulo fictício sobre os rostos | `--imagem` | `resultados/07_rotulos_mock.jpg` | utils, pessoa.jpg |
| `08_baixar_modelos_instrucoes.py` | Confere se os 4 arquivos Caffe existem e lista URLs | nenhuma | terminal | utils |
| `09_carregar_modelos_caffe.py` | Carrega age_net e gender_net | modelos Caffe | terminal | utils, modelos |
| `10_blob_para_caffe.py` | Gera o blob 1x3x227x227 e imprime shape e faixa | `--imagem` | terminal | utils, pessoa.jpg |
| `11_inferir_genero_idade_roi.py` | Prediz gênero e idade de uma ROI | `--imagem` (padrão `resultados/rois/face_00.jpg`) | terminal | utils, modelos, 06 |
| `12_imagem_completa_caffe.py` | Haar + ROI + Caffe + rótulo em imagem | `--imagem` (padrão `grupo.jpg`) | `resultados/12_pipeline_imagem.jpg` | utils, modelos, grupo.jpg |
| `13_webcam_caffe_fps.py` | Pipeline Caffe na webcam com FPS | webcam | janela | utils, modelos, webcam |
| `14_video_arquivo_caffe_fps.py` | Pipeline Caffe em arquivo de vídeo com FPS | `--video` (obrigatório) | janela | utils, modelos, vídeo |
| `15_salvar_video_anotado.py` | Grava vídeo anotado | `--video` (obrigatório), `--saida` | `resultados/video_anotado.mp4` | utils, modelos, vídeo |
| `16_relatorio_5_rostos_csv.py` | Compara predição com gabarito de 5 rostos | `data/relatorio_5_rostos.csv` e imagens em `data/faces_teste/` | `resultados/16_relatorio_5_rostos_resultado.csv` | utils, modelos, CSV manual |
| `17_benchmark_fps_imagem.py` | Latência média por imagem do Caffe | `--imagem`, `--repeticoes` | terminal | utils, modelos, pessoa.jpg |
| `18_parse_utkface_nome.py` | Extrai idade e gênero de nomes UTKFace | nomes fixos no script | terminal | utils |
| `19_listar_utkface_1000.py` | Lista até 1000 imagens válidas | `--pasta`, `--limite` | terminal | utils, UTKFace |
| `20_criar_dataframe_utkface.py` | DataFrame path, age, gender | `--pasta`, `--limite` | `resultados/20_utkface_dataframe.csv` | utils, UTKFace, pasta resultados |
| `21_dividir_treino_validacao.py` | Split 70/15/15 estratificado | `--pasta`, `--limite` | `resultados/21_train.csv`, `21_val.csv`, `21_test.csv` | utils, UTKFace, sklearn |
| `22_tfdata_utkface.py` | Pipeline tf.data e shapes de um lote | `resultados/21_train.csv` | terminal | 21 |
| `23_visualizar_batch_utkface.py` | Grade 3x3 de um lote com rótulos | `resultados/21_train.csv` | `resultados/23_batch_utkface.png` | 21 |
| `24_modelo_mobilenetv2_cabeca.py` | Constrói e resume o modelo | nenhuma | terminal | utils |
| `25_treinar_cabeca_10_epocas.py` | Treina a cabeça com EarlyStopping | `21_train.csv`, `21_val.csv` | `resultados/25_mobilenetv2_genero.keras` | utils, 21 |
| `26_plotar_curvas_finetuning.py` | Curvas de acurácia e loss (simuladas) | dicionário fixo | `resultados/26_curva_acc.png`, `26_curva_loss.png` | pasta resultados |
| `27_avaliar_modelo_keras.py` | Loss e acurácia no teste | `21_test.csv`, modelo 25 | terminal | 21, 25 |
| `28_matriz_confusao_genero.py` | Matriz de confusão e relatório | `21_test.csv`, modelo 25 | terminal | 21, 25, sklearn |
| `29_salvar_modelo_e_tamanho.py` | Tamanho do `.keras` em MB | modelo 25 | terminal | utils, 25 |
| `30_latencia_modelo_keras.py` | Latência por imagem do modelo Keras | `21_test.csv`, modelo 25 | terminal | utils, 21, 25 |
| `31_comparar_caffe_finetuned.py` | Tabela comparativa (valores de exemplo) | modelo 25 opcional | `resultados/31_tabela_comparativa.csv` | utils |
| `32_pipeline_final_item_a.py` | Pipeline Caffe com `--fonte` webcam ou vídeo | `--fonte` (padrão `0`) | janela | utils, modelos |
| `33_pipeline_final_item_b.py` | Executa 19, 20, 21, 24, 25, 27, 29, 30, 31 em sequência | nenhuma | as saídas desses scripts | pasta `exemplos/` (ver seção 7) |
| `34_comentario_embarcado_etica.py` | Texto sobre modelo fixo, fine-tuning e ética | nenhuma | terminal | nada |
| `35_converter_para_tflite.py` | Converte o modelo para TFLite quantizado | modelo 25 | `resultados/35_genero_mobilenetv2.tflite` | 25 |

## 5. Passo a passo para executar

Todos os comandos abaixo assumem que o terminal está em `aula_12_DNN_Caffe/` e que o pacote `utils/` está em `aula_12_DNN_Caffe/utils/`. Como o bootstrap faz `chdir`, rodar de outra pasta também funciona, exceto o script 33.

Etapa 0, preparação:

```
python exemplos_em_python/01_verificar_ambiente.py
python exemplos_em_python/02_estrutura_de_pastas.py
python exemplos_em_python/03_localizar_haar_cascade.py
```

Espere ver as versões no terminal, as quatro pastas criadas e "Existe? True" para o Haar. Coloque `pessoa.jpg` e `grupo.jpg` em `data/faces_teste/`, os quatro arquivos Caffe em `models/opencv_age_gender/` e as imagens UTKFace em `data/utkface_sample/`.

Etapa 1, detecção e ROI (sem os modelos Caffe):

```
python exemplos_em_python/04_detectar_rosto_imagem.py
python exemplos_em_python/05_detectar_rosto_webcam.py
python exemplos_em_python/06_recortar_roi_facial.py
python exemplos_em_python/07_desenhar_rotulos_mock.py
```

O 04 e o 07 gravam imagens em `resultados/`. O 05 abre uma janela com a webcam; pressione Q para sair. O 06 grava `resultados/rois/face_00.jpg` e imprime o shape de cada ROI.

Etapa 2, modelos Caffe:

```
python exemplos_em_python/08_baixar_modelos_instrucoes.py
python exemplos_em_python/09_carregar_modelos_caffe.py
python exemplos_em_python/10_blob_para_caffe.py
python exemplos_em_python/11_inferir_genero_idade_roi.py
python exemplos_em_python/12_imagem_completa_caffe.py
```

O 08 deve mostrar `existe=True` para os quatro arquivos. O 10 deve imprimir `(1, 3, 227, 227)`. O 11 imprime a predição da ROI gerada no 06. O 12 grava `resultados/12_pipeline_imagem.jpg` com os rostos de `grupo.jpg` rotulados.

Etapa 3, vídeo e métricas do Item A:

```
python exemplos_em_python/13_webcam_caffe_fps.py
python exemplos_em_python/14_video_arquivo_caffe_fps.py --video data/video.mp4
python exemplos_em_python/15_salvar_video_anotado.py --video data/video.mp4 --saida resultados/video_anotado.mp4
python exemplos_em_python/16_relatorio_5_rostos_csv.py
python exemplos_em_python/17_benchmark_fps_imagem.py --repeticoes 50
python exemplos_em_python/32_pipeline_final_item_a.py --fonte 0
```

Q encerra o 13, o 14 e o 32. O 16 na primeira execução cria `data/relatorio_5_rostos.csv` com uma linha de exemplo e para; edite com cinco linhas reais (nomes de arquivos em `data/faces_teste/`, gênero como `Masculino` ou `Feminino`, idade como faixa no formato `(25-32)`) e rode de novo. Anote o FPS do 13 ou 32 e a latência do 17 para a tabela do 31.

Etapa 4, UTKFace e treino do Item B:

```
python exemplos_em_python/18_parse_utkface_nome.py
python exemplos_em_python/19_listar_utkface_1000.py
python exemplos_em_python/20_criar_dataframe_utkface.py
python exemplos_em_python/21_dividir_treino_validacao.py
python exemplos_em_python/22_tfdata_utkface.py
python exemplos_em_python/23_visualizar_batch_utkface.py
python exemplos_em_python/24_modelo_mobilenetv2_cabeca.py
python exemplos_em_python/25_treinar_cabeca_10_epocas.py
```

O 21 imprime três números (com 1000 imagens: 700, 150 e 150). O 22 imprime `(16, 160, 160, 3)` e `(16,)`. O 23 grava a grade em `resultados/23_batch_utkface.png`. O 25 treina por até 10 épocas em CPU (minutos), salva o `.keras` e imprime o tempo total. Anote esse tempo.

Etapa 5, avaliação e comparação:

```
python exemplos_em_python/26_plotar_curvas_finetuning.py
python exemplos_em_python/27_avaliar_modelo_keras.py
python exemplos_em_python/28_matriz_confusao_genero.py
python exemplos_em_python/29_salvar_modelo_e_tamanho.py
python exemplos_em_python/30_latencia_modelo_keras.py
python exemplos_em_python/31_comparar_caffe_finetuned.py
```

Edite as constantes `acc_caffe`, `acc_ft`, `train_time_s`, `lat_caffe_ms` e `lat_ft_ms` do script 31 com os números obtidos antes de rodar. A tabela sai no terminal e em `resultados/31_tabela_comparativa.csv`.

Etapa 6, fechamento:

```
python exemplos_em_python/34_comentario_embarcado_etica.py
python exemplos_em_python/35_converter_para_tflite.py
```

O 35 grava o `.tflite` e imprime o tamanho, para comparar com o valor do 29. O 33 só funciona se os scripts estiverem em uma pasta chamada `exemplos/` (ver seção 7).

## 6. Exercícios e itens da aula

Os scripts referenciam dois itens.

Item A: pipeline com modelo pré-treinado. Detectar rostos com Haar Cascade, recortar a ROI, estimar gênero e faixa etária com os modelos Caffe via `cv2.dnn`, desenhar os rótulos e medir o FPS em imagem, vídeo e webcam. A validação é o relatório de 5 rostos com gabarito manual. Scripts que resolvem: 04 a 17, com a versão final em `32_pipeline_final_item_a.py` e a validação em `16_relatorio_5_rostos_csv.py`.

Item B: fine-tuning e comparação. Preparar até 1000 imagens do UTKFace, dividir em treino, validação e teste, treinar a cabeça de uma MobileNetV2 congelada para gênero, avaliar no teste, medir tamanho e latência, e montar a tabela comparativa contra o modelo Caffe (acurácia, tempo de treino, tamanho e latência). Scripts que resolvem: 18 a 31, orquestrados por `33_pipeline_final_item_b.py`. O comentário técnico e ético do 34 é o texto que acompanha a entrega, e o 35 é opcional para embarcado.

## 7. Armadilhas e dicas

- Nada roda sem o pacote `utils/`. Ele não veio no zip. A seção 7b descreve o que cada função precisa fazer para reconstruí-lo. Coloque-o em `aula_12_DNN_Caffe/utils/` com um `__init__.py`.
- Os quatro arquivos Caffe precisam ter exatamente os nomes `age_deploy.prototxt`, `age_net.caffemodel`, `gender_deploy.prototxt` e `gender_net.caffemodel` em `models/opencv_age_gender/`. As URLs do script 08 apontam para páginas do GitHub; para os `.prototxt` use a versão raw do arquivo, não a página HTML.
- O script 33 chama `exemplos/NN.py`, mas nossa pasta se chama `exemplos_em_python/`. Ele falha com "No such file". Os docstrings dos scripts 04, 14 e 32 também citam `exemplos/`. Mantido por fidelidade ao professor; para rodar o 33, ou crie uma junção de diretório chamada `exemplos` apontando para `exemplos_em_python`, ou rode os nove scripts na mão.
- Em todos os scripts o docstring vem depois do `from _bootstrap import`. Isso é inofensivo, exceto no 34, onde `print(__doc__)` imprime `None`. Para ler o texto, abra o arquivo.
- Os scripts 20, 21, 22, 23, 25 e 26 gravam em `resultados/` sem criar a pasta. Rode o 02 antes.
- Os scripts 22, 25, 27, 28 e 30 entregam pixels de 0 a 255 ao modelo. Ao reconstruir `build_mobilenetv2_gender`, inclua o pré-processamento dentro do modelo (`tf.keras.applications.mobilenet_v2.preprocess_input` ou uma camada `Rescaling`), senão a acurácia despenca.
- O script 26 plota curvas simuladas, não o histórico real. Para usar o real, salve `hist.history` em JSON no fim do 25 e carregue no 26 (o `import json` já está lá).
- O script 31 tem acurácia, tempo e latência fixos como exemplo. Substitua pelos valores medidos, senão a tabela entregue é inventada.
- No 16, a comparação de idade é por string exata. Escreva o gabarito no mesmo formato das faixas do modelo: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`.
- O 16 usa apenas o primeiro rosto detectado por imagem. Use fotos com uma pessoa só.
- O 17 assume que há pelo menos um rosto na imagem (`detect_faces_haar(img)[0]`); com zero rostos dá `IndexError`.
- Webcam: 05, 13 e 32 (padrão). Se a câmera 0 não abrir, troque o índice no código ou use `--fonte` no 32.
- O 14 e o 15 exigem `--video`. Não há vídeo no zip; grave um curto com a webcam ou use o do 15 de outra aula.
- O treino do 25 é em CPU no Windows. Com 1000 imagens de 160 x 160 e o tronco congelado, cada época leva de dezenas de segundos a alguns minutos. Reduza `--limite` no 19 a 21 para testes rápidos.
- O EarlyStopping com `patience=3` pode parar antes das 10 épocas; isso é esperado.
- Parâmetros que valem variar: `padding` do `crop_face` (afeta muito a predição Caffe), `dropout` da cabeça, `BATCH` e `IMG_SIZE` no treino, o limiar 0,5 do script 28, e `--repeticoes` do 17.
- O `.tflite` do 35 é quantizado; para comparar latência de forma justa seria preciso rodar com o interpretador TFLite, o que o script não faz.

## 7b. Funções esperadas do pacote utils/ (não incluído no zip)

Assinaturas deduzidas das chamadas nos scripts. Tipos entre parênteses são inferidos do uso.

| Função | Módulo | Assinatura deduzida | O que os scripts esperam que ela faça | Scripts que usam |
|---|---|---|---|---|
| `load_image` | face_utils | `load_image(path: str \| Path) -> np.ndarray` | Lê a imagem com `cv2.imread` (BGR) e provavelmente lança erro se o arquivo não existir ou vier `None`. Recebe `str` e `Path`. | 04, 06, 07, 10, 11, 12, 16, 17 |
| `detect_faces_haar` | face_utils | `detect_faces_haar(img_bgr: np.ndarray) -> list[box]` | Converte para cinza, roda `CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale`, retorna lista de caixas (uso sugere `(x, y, w, h)`). Lista vazia quando não há rosto (`if not faces`). Indexável (`faces[0]`). | 04, 05, 06, 07, 12, 13, 14, 15, 16, 17, 32 |
| `draw_label` | face_utils | `draw_label(img: np.ndarray, box, text: str) -> None` | Desenha o retângulo da caixa e o texto sobre ele, modificando `img` no lugar (os scripts gravam `img` logo depois, sem usar retorno). | 04, 05, 07, 12, 13, 14, 15, 32 |
| `ensure_dir` | face_utils | `ensure_dir(path: str) -> None` | `Path(path).mkdir(parents=True, exist_ok=True)`. | 04, 06, 07, 12, 15, 16 |
| `crop_face` | face_utils | `crop_face(img: np.ndarray, box, padding: int = <padrão>) -> np.ndarray` | Recorta a caixa com margem `padding`, limitando aos limites da imagem. Retorna array com `.shape`. Chamado com `padding=20` no 06 e sem padding nos demais, então há um valor padrão. | 06, 12, 13, 14, 15, 16, 17, 32 |
| `required_caffe_files` | face_utils | `required_caffe_files() -> dict[str, Path]` | Dicionário com os quatro arquivos em `models/opencv_age_gender/`: `age_deploy.prototxt`, `age_net.caffemodel`, `gender_deploy.prototxt`, `gender_net.caffemodel`. Chave é um nome curto (formatado com `:12s`), valor é `Path`. | 08 |
| `load_caffe_age_gender` | face_utils | `load_caffe_age_gender() -> tuple[cv2.dnn.Net, cv2.dnn.Net]` | `cv2.dnn.readNetFromCaffe` para age e para gender, usando os caminhos de `required_caffe_files`. Retorna `(age_net, gender_net)`. | 09, 11, 12, 13, 14, 15, 16, 17, 32 |
| `make_age_gender_blob` | face_utils | `make_age_gender_blob(face_bgr: np.ndarray) -> np.ndarray` | `cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)` com a média `(78.4263377603, 87.7689143744, 114.895847746)`. Shape `(1, 3, 227, 227)`, `dtype` float32, com `.min()` e `.max()`. | 10 (e internamente por `predict_age_gender`) |
| `predict_age_gender` | face_utils | `predict_age_gender(face_bgr, age_net, gender_net) -> objeto` | Gera o blob, faz `setInput` e `forward` nas duas redes, pega o argmax. Retorna objeto (dataclass ou namedtuple) com atributos `gender` (str, `"Masculino"` ou `"Feminino"`, comparável ao CSV do 16), `gender_conf` (float), `age` (str no formato `(25-32)`), `age_conf` (float). `print(pred)` deve ser legível. | 11, 12, 13, 14, 15, 16, 17, 32 |
| `parse_utkface_filename` | tf_utils | `parse_utkface_filename(name: str \| Path) -> tuple[int, int]` | Faz split do nome do arquivo por `_` e retorna `(idade, genero)` como inteiros; gênero 0 masculino, 1 feminino. Aceita `Path` (o 19 e o 21 passam `Path`). | 18, 19, 20, 21 |
| `collect_utkface_paths` | tf_utils | `collect_utkface_paths(folder: str, limit: int) -> list[Path]` | Lista os arquivos de imagem da pasta cujo nome faz parse válido, ordenados, até `limit`. Cada item tem `.name` e vira `str(p)`. | 19, 20, 21 |
| `build_mobilenetv2_gender` | tf_utils | `build_mobilenetv2_gender(input_shape=(160,160,3), dropout=0.3) -> tf.keras.Model` | Monta `MobileNetV2(include_top=False, weights="imagenet")` congelada, com pré-processamento incluído (entrada 0 a 255), `GlobalAveragePooling`, `Dropout(dropout)`, `Dense(1, sigmoid)`. Já compilado com perda `binary_crossentropy` e métrica `accuracy` (o 27 espera `evaluate` retornar exatamente `(loss, acc)`). Chamado posicionalmente `((160,160,3), 0.3)` no 25. | 24, 25 |
| `model_size_mb` | tf_utils | `model_size_mb(path: Path) -> float` | `path.stat().st_size / (1024*1024)`. | 29, 31 |
| `measure_keras_latency_ms` | tf_utils | `measure_keras_latency_ms(model, sample: np.ndarray, repeat: int = 30) -> float` | Recebe uma imagem `(160,160,3)` em NumPy, adiciona a dimensão de lote, faz aquecimento, repete `model.predict` `repeat` vezes e retorna a média em milissegundos. | 30 |

## 8. Glossário

| Termo | Significado |
|---|---|
| Argmax | Índice da maior probabilidade em um vetor de saída; define a classe predita. |
| Blob | Tensor 4D `(N, C, H, W)` que o `cv2.dnn` usa como entrada da rede. |
| Bounding box | Retângulo `(x, y, w, h)` que delimita um objeto detectado. |
| Caffe | Framework de deep learning cujos modelos são distribuídos em `.prototxt` (arquitetura) e `.caffemodel` (pesos). |
| Cabeça (head) | Camadas finais adicionadas sobre um tronco pré-treinado, treinadas para a nova tarefa. |
| CHW / HWC | Ordem dos eixos de uma imagem: canais, altura, largura (redes) ou altura, largura, canais (OpenCV, NumPy). |
| CSV | Arquivo de texto com valores separados por vírgula. |
| cv2.dnn | Módulo do OpenCV para executar redes neurais treinadas em outros frameworks. |
| DataFrame | Tabela em memória do pandas. |
| Dropout | Camada que zera aleatoriamente uma fração das ativações durante o treino, para reduzir overfitting. |
| EarlyStopping | Callback do Keras que para o treino quando a métrica monitorada deixa de melhorar. |
| Época (epoch) | Uma passagem completa pelo conjunto de treino. |
| Estratificação | Divisão de dados que preserva a proporção das classes em cada subconjunto. |
| F1 | Média harmônica entre precisão e recall. |
| Fine-tuning | Ajuste de um modelo pré-treinado para uma nova tarefa; nesta aula, treinando apenas a cabeça. |
| FPS | Frames por segundo processados. |
| Haar Cascade | Detector clássico de objetos baseado em características de Haar e classificador em cascata. |
| ImageNet | Dataset de milhões de imagens usado para pré-treinar redes como a MobileNetV2. |
| Keras | API de alto nível do TensorFlow para construir e treinar redes. |
| Latência | Tempo de uma única inferência, em milissegundos. |
| Matriz de confusão | Tabela que cruza classe real com classe predita. |
| MobileNetV2 | Rede convolucional leve, projetada para dispositivos móveis. |
| Overfitting | Modelo que decora o treino e piora na validação. |
| Padding (na ROI) | Margem extra em pixels ao redor da caixa detectada. |
| Precisão e recall | Fração dos preditos como positivo que são positivos, e fração dos positivos que foram encontrados. |
| prefetch / AUTOTUNE | Otimização do tf.data que prepara o próximo lote em paralelo, com tamanho de buffer escolhido automaticamente. |
| Quantização | Redução da precisão numérica dos pesos (float32 para int8) para diminuir tamanho e latência. |
| ROI | Region of Interest, região de interesse; aqui, o recorte do rosto. |
| Sigmoide | Função de ativação que mapeia para (0, 1); usada na saída de classificação binária. |
| Subtração de média | Pré-processamento que subtrai a média por canal usada no treinamento da rede. |
| tf.data | API do TensorFlow para pipelines de entrada eficientes. |
| TFLite | TensorFlow Lite, formato e runtime para dispositivos móveis e embarcados. |
| Transfer learning | Reaproveitar uma rede treinada em um problema para outro problema. |
| Tronco (backbone) | Parte convolucional pré-treinada de uma rede, congelada no transfer learning. |
| UTKFace | Dataset de rostos com idade, gênero e etnia codificados no nome do arquivo. |
| VideoWriter | Classe do OpenCV para gravar vídeo em disco. |
| Viés (bias) | Erro sistemático do modelo que prejudica determinados subgrupos. |
