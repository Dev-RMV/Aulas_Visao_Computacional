# Aula 14 – OpenCV DNN: MobileNetV2 do Keras ao cv2.dnn

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula responde a uma pergunta prática: depois que uma rede neural está treinada, como colocá-la para rodar dentro de um programa OpenCV, sem carregar o TensorFlow inteiro, e como saber se valeu a pena? O professor usa uma rede pronta, a MobileNetV2 com pesos do ImageNet, e a executa por dois caminhos. O primeiro é o Keras, o framework em que a rede foi treinada. O segundo é o módulo `cv2.dnn`, que só faz inferência (o forward pass) e aceita modelos em vários formatos, entre eles o TensorFlow Lite. Para chegar ao segundo caminho é preciso converter a rede do Keras para `.tflite`, e é essa conversão que liga as duas metades da aula.

O fio condutor está escrito no docstring de todos os scripts: "imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado". Os scripts 01 a 03 preparam as pastas e criam imagens sintéticas para o fluxo funcionar sem dataset. O 04 explica o blob, o tensor que a rede espera. O 05 baixa os nomes das 1000 classes do ImageNet. Os scripts 06 a 08 carregam a rede no Keras, convertem para TFLite e carregam o arquivo no OpenCV. Os scripts 09 a 12 fazem o forward, extraem o top-3 e desenham o resultado sobre a imagem. O 13 repete a inferência no Keras para comparar. Os scripts 14 a 19 medem latência, memória e acurácia top-1 e montam uma tabela comparativa entre os dois backends. Os scripts 20 a 27 montam um pipeline clássico em volta da rede: correção de distorção da câmera, segmentação por cor em HSV, features ORB, detectores HOG e Haar, classificação da ROI e cronômetro por etapa, primeiro em imagem e depois em um vídeo sintético. O 28 aponta para um notebook, o 29 deixa em pseudocódigo o desafio de fazer tudo isso na webcam e o 30 é o pipeline final em uma única imagem. O módulo `dnn_utils.py` concentra as funções repetidas.

Cada script termina com um bloco `# DESAFIO DO ALUNO`, que é a parte prática da aula. A seção 6 lista todos.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas nos scripts, conforme o `requirements.txt` do professor:

| Biblioteca | Pacote pip do professor | No projeto | Onde é usada |
|---|---|---|---|
| OpenCV | `opencv-python` | `opencv-contrib-python` 4.14 | `dnn_utils.py` e quase todos os scripts |
| NumPy | `numpy` | 2.x | `dnn_utils.py`, 13, 20, 27 |
| TensorFlow e Keras | `tensorflow` | 2.21 com Keras 3.15 | 06, 07, 13, 15, 17 |
| pandas | `pandas` | 2.3 | 19 |
| psutil | `psutil` | 7.2 | 16, 17 |
| Pillow | `pillow` | instalado | não é importado por nenhum script; o Keras usa internamente |
| Matplotlib | `matplotlib` | 3.11 | não é importado por nenhum script |
| Jupyter | `notebook` | não está no ambiente unificado | só para o notebook citado pelo 28, que não veio no zip |

O repositório tem um único `requirements.txt` na raiz, que atende esta aula, e o venv `.venv` com Python 3.13 é criado por `py preparar_ambiente.py`. Não instale `opencv-python` por cima do `opencv-contrib-python`: os dois pacotes disputam o mesmo módulo `cv2`. O `cv2.dnn.readNetFromTFLite`, que esta aula exige, existe na versão instalada.

Pasta de execução: `dnn_utils.py` define a raiz como a pasta do próprio arquivo (`Path(__file__).resolve().parent`), então todos os caminhos são relativos a `aula_14_OpenCV_DNN/exemplos_em_python/`, independentemente de onde o comando é digitado. Mesmo assim, rode de dentro dessa pasta, para que o `import dnn_utils` e os comandos deste tutorial funcionem sem ajuste. A função `ensure_dirs()` cria `data/classificacao/`, `data/pipeline/`, `modelos/` e `saidas/` na primeira chamada, e quase todo script passa por ela via `list_classification_images()` ou `read_pipeline_frame()`. No repositório, `saidas/` já existe com um `.gitkeep`; `data/`, `modelos/` e os arquivos `.tflite` ficam fora do git.

Internet: o 05 baixa `imagenet_labels.txt` do GitHub. Os scripts 06, 07, 13, 15 e 17 carregam a MobileNetV2 com `weights='imagenet'`, e na primeira vez o Keras baixa os pesos (cerca de 14 MB) para `~/.keras/models/`. Depois disso tudo roda offline.

Material faltante: o `28_gerar_notebook_pipeline.py` diz que o notebook está em `notebooks/pipeline_integrado_dnn.ipynb`, mas essa pasta não veio no zip. Também não vêm as 10 fotos reais que o 01 pede para `data/classificacao/` nem o `data/pipeline/frame_real.jpg` opcional; sem elas, os scripts geram imagens sintéticas e continuam funcionando.

Webcam: nenhum script abre a câmera. O 29 é um pseudocódigo para o aluno implementar.

## 3. Conceitos-chave

### 3.1 Raiz relativa ao script e estrutura de pastas

Todos os caminhos partem de `ROOT`, a pasta onde `dnn_utils.py` está. Isso torna a aula independente do diretório de trabalho: `data/`, `modelos/` e `saidas/` ficam sempre ao lado dos scripts. `ensure_dirs()` cria as quatro pastas com `mkdir(parents=True, exist_ok=True)`, o que não dá erro se elas já existirem. O 01 chama a função e imprime os caminhos; os demais a chamam indiretamente.

```python
# dnn_utils.py
ROOT = Path(__file__).resolve().parent
DATA_CLASS = ROOT / 'data' / 'classificacao'
DATA_PIPE = ROOT / 'data' / 'pipeline'
MODELS = ROOT / 'modelos'
OUT = ROOT / 'saidas'

IMAGE_SIZE = (224, 224)


def ensure_dirs():
    for p in [DATA_CLASS, DATA_PIPE, MODELS, OUT]:
        p.mkdir(parents=True, exist_ok=True)
```

Onde aparece: `dnn_utils.py`, `01_configurar_pastas.py`, `05_baixar_labels_imagenet.py`, `07_converter_keras_para_tflite.py` e, por meio de `list_classification_images` e `read_pipeline_frame`, quase todos os outros.

### 3.2 Imagens sintéticas de classificação e o CSV de rótulos

Para que o fluxo de arquivos funcione antes de o aluno ter fotos, `create_synthetic_classification_images` desenha 10 imagens de 480x360 com fundo claro, um retângulo, um círculo e o nome de um objeto escrito por cima (`carro`, `caneca`, `bola`, e assim por diante). Elas não se parecem com nada do ImageNet, então a classificação nelas serve só para testar o código. A função também cria `data/labels_top1.csv` com a coluna `label_esperado` vazia, que o aluno preenche depois para o script 18.

`list_classification_images` procura `.jpg`, `.jpeg`, `.png` e `.bmp` em `data/classificacao/`, devolve a lista ordenada por nome e, se a pasta estiver vazia, gera as sintéticas. É essa função que quase todo script usa para pegar a primeira imagem (`[0]`).

```python
# dnn_utils.py
def create_synthetic_classification_images(n=10):
    """Cria imagens simples para testar fluxo de arquivos quando o aluno ainda não tem dataset real."""
    ensure_dirs()
    names = ['carro','caneca','bola','livro','garrafa','teclado','robo','caixa','ferramenta','capacete']
    ...
    for i, name in enumerate(names[:n]):
        img = np.full((360, 480, 3), colors[i], np.uint8)
        cv2.rectangle(img, (80,80), (400,280), (40+i*12,80+i*8,160+i*5), 3)
        cv2.circle(img, (240,180), 55+i*2, (80+i*10,50+i*8,200-i*5), -1)
        cv2.putText(img, name, (95,330), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30,30,30), 2)
        path = DATA_CLASS / f'{i+1:02d}_{name}.jpg'
        cv2.imwrite(str(path), img)
        paths.append(path)
    csv_path = ROOT / 'data' / 'labels_top1.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['arquivo','label_esperado'])
        for p in paths:
            w.writerow([p.name, ''])
    return paths


def list_classification_images():
    ensure_dirs()
    exts = ['*.jpg','*.jpeg','*.png','*.bmp']
    paths=[]
    for e in exts:
        paths.extend(DATA_CLASS.glob(e))
    paths = sorted(paths)
    if not paths:
        paths = create_synthetic_classification_images(10)
    return paths
```

Onde aparece: `dnn_utils.py`, `03_criar_imagens_teste_sinteticas.py` (chamada explícita), e `list_classification_images` nos scripts 04, 09, 10, 11, 12, 13, 14, 15, 16, 17 e 18.

### 3.3 Frame sintético do pipeline e o fallback para um frame real

A segunda metade da aula trabalha sobre um único frame de 640x480. `create_pipeline_frame` desenha um retângulo vermelho grande (em BGR, `(0,0,255)`) para a segmentação HSV, uma moldura branca e uma fileira de círculos e traços pretos dentro dele para dar cantos ao ORB, e um texto. `read_pipeline_frame` devolve `data/pipeline/frame_real.jpg` se o aluno colocou uma foto lá; senão, gera o sintético. Ela devolve a imagem já lida e o caminho de origem, e o 30 imprime esse caminho.

```python
# dnn_utils.py
def create_pipeline_frame():
    """Cria um frame sintético com região colorida, detalhes ORB e área para anotação."""
    ensure_dirs()
    img = np.full((480, 640, 3), (235, 235, 235), np.uint8)
    cv2.rectangle(img, (35,35), (605,445), (210,210,210), 2)
    cv2.rectangle(img, (120,120), (380,340), (0,0,255), -1)  # ROI vermelha em BGR
    cv2.rectangle(img, (150,150), (350,310), (255,255,255), 3)
    for x in range(160, 340, 35):
        cv2.circle(img, (x, 190), 8, (0,0,0), -1)
        cv2.line(img, (x,230), (x+20,260), (0,0,0), 2)
    cv2.putText(img, 'ROI HSV + ORB + DNN', (95,410), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2)
    cv2.imwrite(str(DATA_PIPE/'frame_sintetico.jpg'), img)
    return DATA_PIPE/'frame_sintetico.jpg'


def read_pipeline_frame():
    real = DATA_PIPE / 'frame_real.jpg'
    if real.exists():
        return cv2.imread(str(real)), real
    p = create_pipeline_frame()
    return cv2.imread(str(p)), p
```

Onde aparece: `dnn_utils.py`, `02_criar_frame_sintetico_pipeline.py`, `27_pipeline_video_sintetico.py` (chama `create_pipeline_frame` direto) e `read_pipeline_frame` nos scripts 20 a 26 e 30.

### 3.4 O blob: `cv2.dnn.blobFromImage` para a MobileNetV2

Uma rede não recebe a imagem como o OpenCV a guarda. Ela espera um tensor de 4 dimensões, `(N, C, H, W)`: lote, canais, altura, largura, em `float32`, com os pixels na escala em que a rede foi treinada. `cv2.dnn.blobFromImage` faz essa conversão em uma chamada. A ordem interna é: redimensiona para `size`, subtrai `mean` de cada canal, multiplica por `scalefactor`, troca R e B se `swapRB` for verdadeiro e reorganiza os eixos. Com `mean=127.5` e `scalefactor=1/127.5`, o resultado é `(pixel - 127.5) / 127.5`, ou seja, o intervalo 0 a 255 vira -1 a 1, que é o que a MobileNetV2 espera. `swapRB=True` é necessário porque o OpenCV lê em BGR e a rede foi treinada em RGB. `crop=False` redimensiona sem recortar, então a proporção da imagem é deformada para 224x224.

```python
# dnn_utils.py
def mobilenet_blob_bgr(img):
    """
    Cria blob para MobileNetV2.
    MobileNetV2 espera pixels normalizados aproximadamente em [-1, 1].
    No blobFromImage fazemos: pixel * 1/127.5 - 1.
    """
    return cv2.dnn.blobFromImage(
        img, scalefactor=1/127.5, size=IMAGE_SIZE,
        mean=(127.5,127.5,127.5), swapRB=True, crop=False
    )
```

O script 04 imprime o antes e o depois: a imagem `(360, 480, 3)` em `uint8` vira um blob `(1, 3, 224, 224)` em `float32` com valores entre -1 e 1.

```python
# 04_entender_blob_from_image.py
img = cv2.imread(str(list_classification_images()[0]))
blob = mobilenet_blob_bgr(img)

print('Formato da imagem original:', img.shape)
print('Formato do blob:', blob.shape)
print('dtype:', blob.dtype)
print('menor valor:', blob.min())
print('maior valor:', blob.max())
```

Onde aparece: `dnn_utils.py`, `04_entender_blob_from_image.py`, `09_forward_pass_opencv.py` e, via `predict_opencv`, todos os scripts que classificam com OpenCV.

### 3.5 Rótulos do ImageNet

A rede devolve 1000 números, um por classe do ImageNet, e sem uma lista de nomes o índice 817 não diz nada. O 05 baixa o arquivo `imagenet_classes.txt` do repositório do PyTorch Hub com `urllib.request.urlretrieve` e salva como `modelos/imagenet_labels.txt`. `load_labels` lê esse arquivo, ignora linhas vazias e devolve as 1000 primeiras entradas. Se o arquivo não existir ou tiver menos de 1000 linhas, devolve nomes genéricos `classe_000` a `classe_999`, e o fluxo segue sem quebrar.

```python
# 05_baixar_labels_imagenet.py
url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
out = MODELS / 'imagenet_labels.txt'

print('Baixando labels ImageNet...')
urllib.request.urlretrieve(url, out)
print('Arquivo salvo em:', out)
```

```python
# dnn_utils.py
def load_labels():
    """Carrega labels ImageNet se disponíveis; caso contrário cria labels genéricos."""
    label_file = MODELS / 'imagenet_labels.txt'
    if label_file.exists():
        labels = [line.strip() for line in label_file.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(labels) >= 1000:
            return labels[:1000]
    return [f'classe_{i:03d}' for i in range(1000)]
```

Onde aparece: `05_baixar_labels_imagenet.py`, `dnn_utils.py`, e `load_labels` nos scripts 10, 18, 26 e 30 (o `draw_top3` também a chama quando não recebe a lista).

### 3.6 MobileNetV2 no Keras com pesos do ImageNet

`tensorflow.keras.applications.MobileNetV2(weights='imagenet')` monta a arquitetura completa, com a cabeça de classificação de 1000 classes, e baixa os pesos treinados no ImageNet na primeira execução. O 06 só carrega e imprime as formas: entrada `(None, 224, 224, 3)`, no formato do Keras (lote, altura, largura, canais, ou NHWC), e saída `(None, 1000)`. O `None` é a dimensão do lote, livre. Repare que o Keras usa NHWC e o OpenCV usa NCHW; o `blobFromImage` faz a troca por baixo dos panos.

```python
# 06_carregar_mobilenetv2_keras.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet')
print('Modelo Keras carregado.')
print('Entrada esperada:', model.input_shape)
print('Saída esperada:', model.output_shape)
```

Onde aparece: `06_carregar_mobilenetv2_keras.py`, `07_converter_keras_para_tflite.py`, `13_inferencia_keras_top3.py`, `15_medir_latencia_keras.py`, `17_medir_memoria_keras.py`.

### 3.7 Conversão do Keras para TensorFlow Lite

O OpenCV não lê um modelo Keras diretamente. O caminho escolhido pelo professor é o TensorFlow Lite, formato pensado para dispositivos móveis e embarcados. `tf.lite.TFLiteConverter.from_keras_model(model)` cria o conversor, `convert()` devolve os bytes do modelo, e `write_bytes` grava `modelos/mobilenetv2_imagenet.tflite`. Com `converter.optimizations = []` os pesos ficam em `float32` e o arquivo tem cerca de 14 MB, o mesmo tamanho dos pesos originais. O desafio do script é ativar `tf.lite.Optimize.DEFAULT`, que quantiza os pesos para 8 bits e reduz o arquivo para cerca de um quarto disso.

```python
# 07_converter_keras_para_tflite.py
ensure_dirs()
model = MobileNetV2(weights='imagenet')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = []
tflite_model = converter.convert()

out = MODELS / 'mobilenetv2_imagenet.tflite'
out.write_bytes(tflite_model)
print('Modelo TensorFlow Lite salvo em:', out)
print('Tamanho aproximado [MB]:', out.stat().st_size / (1024*1024))
```

Este script é o pré-requisito de toda a metade OpenCV da aula: os scripts 08 a 12, 14, 16, 18, 24, 25, 26 e 30 param com `FileNotFoundError` se o `.tflite` não existir.

Onde aparece: `07_converter_keras_para_tflite.py`.

### 3.8 Carregar o TFLite no OpenCV DNN: `readNetFromTFLite`, backend e target

`cv2.dnn.readNetFromTFLite(caminho)` lê o arquivo e monta a rede dentro do OpenCV. `setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)` escolhe a implementação própria do OpenCV (as alternativas seriam CUDA, OpenVINO e outras) e `setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)` manda executar na CPU. `get_opencv_net` encapsula isso e é usada em toda a aula. O 08 carrega a rede e imprime quantas camadas ela tem com `getLayerNames()`. O comentário do 08 resume a divisão de papéis: o OpenCV DNN serve para inferência; treinar e ajustar é trabalho para TensorFlow, Keras ou PyTorch.

Um detalhe: a mensagem de erro manda executar `06_converter_keras_para_tflite.py`, mas o conversor é o 07. O 06 só carrega o modelo e imprime as formas.

```python
# dnn_utils.py
def get_opencv_net():
    model = MODELS / 'mobilenetv2_imagenet.tflite'
    if not model.exists():
        raise FileNotFoundError('Modelo não encontrado. Execute primeiro 06_converter_keras_para_tflite.py')
    net = cv2.dnn.readNetFromTFLite(str(model))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net
```

```python
# 08_carregar_tflite_no_opencv_dnn.py
net = get_opencv_net()
print('Modelo carregado no OpenCV DNN.')
print('Camadas da rede:', len(net.getLayerNames()))
```

Onde aparece: `dnn_utils.py`, `08_carregar_tflite_no_opencv_dnn.py`, e `get_opencv_net` nos scripts 09, 10, 12, 14 e 16 (os demais a chamam por dentro de `predict_opencv`).

### 3.9 Forward pass: `setInput` e `forward`

Inferir com `cv2.dnn` são duas chamadas: `net.setInput(blob)` entrega o tensor de entrada e `net.forward()` executa a rede da entrada até a última camada e devolve a saída. Para a MobileNetV2 a saída tem forma `(1, 1000)`: uma linha (o lote de uma imagem) com um valor por classe. O 09 imprime a forma e os 10 primeiros valores. A posição do maior valor é a classe mais provável, e é isso que o desafio do script pede para explicar.

```python
# 09_forward_pass_opencv.py
img = cv2.imread(str(list_classification_images()[0]))
net = get_opencv_net()
blob = mobilenet_blob_bgr(img)

net.setInput(blob)
out = net.forward()

print('Formato da saída:', out.shape)
print('Exemplo dos 10 primeiros valores:')
print(out.reshape(-1)[:10])
```

Onde aparece: `09_forward_pass_opencv.py` e, dentro de `predict_opencv`, em todos os scripts que classificam com OpenCV.

### 3.10 Softmax e top-k

`predict_opencv` junta blob, `setInput`, `forward` e um `softmax` escrito à mão: subtrai o máximo antes de exponenciar (truque numérico para evitar overflow) e divide pelo somatório, para que os 1000 valores somem 1. `topk` ordena com `np.argsort`, inverte para ficar do maior para o menor e devolve os `k` primeiros como pares `(índice, probabilidade)`. Se `net` não for passado, `predict_opencv` carrega o modelo do disco a cada chamada; os scripts 10, 12 e 14 passam a rede para evitar isso, mas 11, 18, 24, 25, 26 e 30 não.

Um ponto de atenção: a MobileNetV2 do Keras com a cabeça padrão já termina em uma camada softmax, então o TFLite devolve probabilidades. O `softmax` do `dnn_utils` é aplicado por cima delas. A ordem do top-3 não muda, porque a função é monotônica, mas as confianças ficam achatadas: um valor 1,0 vira algo próximo de 0,27%, e os demais ficam perto de 0,1%. As porcentagens do 13, que vêm direto do `model.predict`, são as reais.

```python
# dnn_utils.py
def softmax(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    s = e / np.sum(e)
    return s
...
def predict_opencv(img, net=None):
    if net is None:
        net = get_opencv_net()
    blob = mobilenet_blob_bgr(img)
    net.setInput(blob)
    out = net.forward()
    prob = softmax(out)
    return prob


def topk(prob, k=3):
    idx = np.argsort(prob)[::-1][:k]
    return [(int(i), float(prob[i])) for i in idx]
```

```python
# 10_extrair_top3_opencv.py
img = cv2.imread(str(list_classification_images()[0]))
net = get_opencv_net()
prob = predict_opencv(img, net)
labels = load_labels()

for pos, (idx, conf) in enumerate(topk(prob, 3), start=1):
    print(f'{pos}) índice={idx} label={labels[idx]} confiança={conf*100:.2f}%')
```

Onde aparece: `dnn_utils.py`, `10_extrair_top3_opencv.py`, e `predict_opencv` com `topk` nos scripts 11, 12, 14, 16, 18, 24, 25, 26 e 30.

### 3.11 Desenhar o top-3 sobre a imagem

`draw_top3` trabalha em uma cópia, pinta um retângulo branco com borda preta no canto superior esquerdo e escreve o título e as três linhas `N) rótulo: XX.X%` com `cv2.putText`, cada uma 28 pixels abaixo da anterior. Se o índice não existir na lista de rótulos, escreve `classe_N`. O 11 salva uma imagem anotada; o 12 faz o mesmo para as 10 primeiras imagens da pasta, carregando a rede uma única vez.

```python
# dnn_utils.py
def draw_top3(img, top3, labels=None, x=20, y=35):
    if labels is None:
        labels = load_labels()
    out = img.copy()
    cv2.rectangle(out, (10, 10), (470, 120), (255,255,255), -1)
    cv2.rectangle(out, (10, 10), (470, 120), (0,0,0), 2)
    cv2.putText(out, 'Top-3 DNN', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)
    for j, (idx, conf) in enumerate(top3, start=1):
        label = labels[idx] if idx < len(labels) else f'classe_{idx}'
        txt = f'{j}) {label}: {conf*100:.1f}%'
        cv2.putText(out, txt, (x, y+28*j), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,180), 2)
    return out
```

```python
# 12_processar_10_imagens.py
paths = list_classification_images()[:10]
net = get_opencv_net()

for p in paths:
    img = cv2.imread(str(p))
    prob = predict_opencv(img, net)
    vis = draw_top3(img, topk(prob, 3))
    out = OUT / f'12_top3_{p.stem}.jpg'
    cv2.imwrite(str(out), vis)
    print('salvo:', out.name)
```

Onde aparece: `dnn_utils.py`, `11_sobrepor_top3_na_imagem.py`, `12_processar_10_imagens.py`, `24_classificar_roi_detectada.py`, `26_pipeline_integrado_imagem.py`, `30_pipeline_final_completo.py`.

### 3.12 A mesma inferência no Keras: `preprocess_input` e `decode_predictions`

Para comparar, o 13 classifica a mesma imagem sem sair do Keras. O pré-processamento é feito à mão e explicita o que o blob escondia: `cv2.resize` para 224x224, `cv2.cvtColor` de BGR para RGB, `preprocess_input` (que para a MobileNetV2 é exatamente a escala para -1 a 1) e `[None, ...]` para acrescentar a dimensão do lote. `model.predict` devolve `(1, 1000)`, e `decode_predictions` traduz os índices para nomes do ImageNet e devolve o top-3 como tuplas `(id, rótulo, confiança)`. O `verbose=0` silencia a barra de progresso.

```python
# 13_inferencia_keras_top3.py
model = MobileNetV2(weights='imagenet')
img = cv2.imread(str(list_classification_images()[0]))
rgb = cv2.cvtColor(cv2.resize(img, (224,224)), cv2.COLOR_BGR2RGB)
x = preprocess_input(rgb.astype('float32'))[None, ...]
pred = model.predict(x, verbose=0)

for rank, (_, label, conf) in enumerate(decode_predictions(pred, top=3)[0], start=1):
    print(f'{rank}) {label}: {conf*100:.2f}%')
```

Onde aparece: `13_inferencia_keras_top3.py`, e o mesmo pré-processamento em `15_medir_latencia_keras.py` e `17_medir_memoria_keras.py`.

### 3.13 Latência com aquecimento: `time_ms`

Medir uma única inferência engana: a primeira execução paga custos de inicialização (alocação de buffers, compilação de kernels) que não se repetem. `time_ms` recebe uma função sem argumentos, executa `warmup` vezes sem cronometrar, depois `loops` vezes com `time.perf_counter`, e devolve média e desvio padrão em milissegundos. Os scripts 14 e 15 aplicam a mesma função ao OpenCV DNN e ao Keras, com 5 aquecimentos e 20 medições, e imprimem `média ± desvio`. O desvio alto indica que a máquina está ocupada com outra coisa ou que o número de repetições é baixo demais; o desafio do 14 pede 100 repetições.

```python
# dnn_utils.py
def time_ms(func, loops=10, warmup=3):
    for _ in range(warmup):
        func()
    times=[]
    for _ in range(loops):
        t0=time.perf_counter()
        func()
        times.append((time.perf_counter()-t0)*1000)
    return float(np.mean(times)), float(np.std(times))
```

```python
# 14_medir_latencia_opencv.py
img = cv2.imread(str(list_classification_images()[0]))
net = get_opencv_net()
media, desvio = time_ms(lambda: predict_opencv(img, net), loops=20, warmup=5)

print(f'Latência OpenCV DNN: {media:.2f} ± {desvio:.2f} ms')
```

```python
# 15_medir_latencia_keras.py
media, desvio = time_ms(lambda: model.predict(x, verbose=0), loops=20, warmup=5)
print(f'Latência Keras: {media:.2f} ± {desvio:.2f} ms')
```

Onde aparece: `dnn_utils.py`, `14_medir_latencia_opencv.py`, `15_medir_latencia_keras.py`.

### 3.14 Memória do processo com `psutil`

Em sistema embarcado a memória conta tanto quanto o tempo. `psutil.Process(os.getpid())` representa o próprio processo Python, e `memory_info().rss` é o Resident Set Size, a memória física ocupada, em bytes. Os scripts 16 e 17 leem o RSS antes e depois de carregar o modelo e fazer uma inferência, e imprimem a diferença em MB. É uma medida do custo de carregar o modelo mais os buffers da primeira inferência, não do processo inteiro: no 17, o `import tensorflow` acontece antes da primeira leitura, então as centenas de MB do runtime do TensorFlow já estão no `mem_antes`.

```python
# 16_medir_memoria_opencv.py
process = psutil.Process(os.getpid())
mem_antes = process.memory_info().rss / (1024*1024)
net = get_opencv_net()
img = cv2.imread(str(list_classification_images()[0]))
_ = predict_opencv(img, net)
mem_depois = process.memory_info().rss / (1024*1024)

print(f'Memória antes:  {mem_antes:.1f} MB')
print(f'Memória depois: {mem_depois:.1f} MB')
print(f'Aumento aproximado: {mem_depois - mem_antes:.1f} MB')
```

Onde aparece: `16_medir_memoria_opencv.py`, `17_medir_memoria_keras.py`.

### 3.15 Acurácia top-1 contra um CSV de rótulos

Latência sem acurácia não decide nada. O 18 lê `data/labels_top1.csv` com `csv.DictReader`, pula as linhas cuja coluna `label_esperado` está vazia, classifica cada imagem com o OpenCV DNN e compara o rótulo esperado com o rótulo do top-1. A comparação é tolerante: tudo em minúsculas, espaços do rótulo do ImageNet trocados por `_`, e basta um ser substring do outro (`car` casa com `sports_car`). A acurácia é `corretos / total`. Como o 03 cria o CSV com a coluna vazia, sem preencher nada o script termina em `0/0` e imprime 0,0%. O desafio é justamente preencher o CSV com 10 fotos reais.

```python
# 18_calcular_acuracia_top1.py
with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        expected = row['label_esperado'].strip().lower()
        if not expected:
            continue
        img_path = DATA_CLASS / row['arquivo']
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        prob = predict_opencv(img)
        pred_idx = topk(prob, 1)[0][0]
        pred_label = labels[pred_idx].lower().replace(' ', '_')
        ok = expected in pred_label or pred_label in expected
        corretos += int(ok)
        total += 1
        print(row['arquivo'], 'esperado=', expected, 'predito=', pred_label, 'ok=', ok)

acc = corretos / total if total else 0.0
print(f'Acurácia top-1: {acc*100:.1f}% ({corretos}/{total})')
```

Onde aparece: `18_calcular_acuracia_top1.py`.

### 3.16 Tabela comparativa com pandas

O 19 fecha a comparação entre backends em uma tabela de `pandas.DataFrame` com colunas `backend`, `latencia_ms`, `memoria_MB` e `top1_acc`. Os valores vêm zerados de propósito: o aluno copia o que mediu nos scripts 14 a 18. O `to_string(index=False)` imprime sem a coluna de índice. O comentário técnico ao final é a conclusão da aula: OpenCV DNN para inferência leve e integrada ao pipeline OpenCV, Keras para treinar, ajustar e experimentar.

```python
# 19_tabela_comparativa.py
dados = [
    {'backend':'OpenCV DNN', 'latencia_ms':0.0, 'memoria_MB':0.0, 'top1_acc':0.0},
    {'backend':'Keras',      'latencia_ms':0.0, 'memoria_MB':0.0, 'top1_acc':0.0},
]

df = pd.DataFrame(dados)
print(df.to_string(index=False))
```

Onde aparece: `19_tabela_comparativa.py`.

### 3.17 Correção de distorção com `cv2.undistort`

A partir do 20 o professor monta o pipeline clássico em volta da rede, e a primeira etapa é a calibração da câmera. `cv2.undistort(imagem, K, dist)` desfaz a distorção da lente usando a matriz intrínseca `K` (focais `fx`, `fy` e ponto principal `cx`, `cy`) e os coeficientes de distorção (`k1`, `k2`, `p1`, `p2`, `k3`). O `dnn_utils` traz valores fixos de exemplo: `fx = fy = 850`, centro em `(320, 240)` para um frame 640x480, e `k1 = -0,20`, `k2 = 0,08`. Eles não vêm de nenhuma calibração; o desafio dos scripts 20 e 30 é trocá-los pelos da câmera real (a aula 16 ensina a obtê-los). Como o frame sintético não tem distorção nenhuma, aplicar a correção deforma as bordas em vez de endireitá-las, e é isso que o painel `20_undistort_painel.jpg` mostra lado a lado.

```python
# dnn_utils.py
DEFAULT_K = np.array([[850.0, 0.0, 320.0], [0.0, 850.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)
DEFAULT_DIST = np.array([[-0.20, 0.08, 0.0, 0.0, 0.0]], dtype=np.float32)
```

```python
# 20_undistort_frame_sintetico.py
frame, path = read_pipeline_frame()
undist = cv2.undistort(frame, DEFAULT_K, DEFAULT_DIST)
painel = np.hstack([frame, undist])
cv2.putText(painel, 'original', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
cv2.putText(painel, 'undistort', (frame.shape[1]+20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
cv2.imwrite(str(OUT/'20_undistort_painel.jpg'), painel)
```

Onde aparece: `dnn_utils.py`, `20_undistort_frame_sintetico.py`, `25_medir_tempo_etapas_pipeline.py`, `26_pipeline_integrado_imagem.py`, `30_pipeline_final_completo.py`.

### 3.18 Segmentação de vermelho em HSV e a ROI

`segment_red_hsv` converte para HSV e usa `cv2.inRange` duas vezes, porque o vermelho fica nas duas pontas do círculo de matiz do OpenCV (0 a 180): uma faixa de 0 a 10 e outra de 170 a 180, ambas com saturação mínima 80 e valor mínimo 70 para excluir cinzas e sombras. As duas máscaras são unidas com `|`. `cv2.medianBlur(mask, 5)` remove pontos isolados. Depois `findContours` com `RETR_EXTERNAL` acha as manchas, `max` com `contourArea` escolhe a maior, e se ela tiver pelo menos 500 pixels a função devolve a máscara e a caixa `(x, y, w, h)` de `boundingRect`; senão, devolve `None` como ROI. No frame sintético, a moldura branca e os círculos pretos ficam dentro do retângulo vermelho, mas `RETR_EXTERNAL` ignora buracos, então a ROI é o retângulo inteiro. O 21 salva a máscara e o frame com a ROI em amarelo.

```python
# dnn_utils.py
def segment_red_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0,80,70]); upper1 = np.array([10,255,255])
    lower2 = np.array([170,80,70]); upper2 = np.array([180,255,255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask = cv2.medianBlur(mask, 5)
    cnts,_=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None
    c=max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 500:
        return mask, None
    x,y,w,h = cv2.boundingRect(c)
    return mask, (x,y,w,h)
```

```python
# 21_segmentacao_hsv_roi.py
frame, _ = read_pipeline_frame()
mask, roi = segment_red_hsv(frame)
vis = frame.copy()

if roi:
    x,y,w,h = roi
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 3)
    cv2.putText(vis, 'ROI HSV', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
else:
    print('Nenhuma ROI encontrada.')
```

Onde aparece: `dnn_utils.py`, `21_segmentacao_hsv_roi.py`, e `segment_red_hsv` nos scripts 22, 24, 25, 26, 27 e 30.

### 3.19 Features ORB na ROI

ORB (Oriented FAST and Rotated BRIEF) é o detector e descritor de pontos-chave livre de patente que as aulas 07 e 08 já usaram. `orb_features` converte para cinza se a imagem tiver 3 canais, cria o detector com `cv2.ORB_create(nfeatures=250)` e chama `detectAndCompute`, que devolve a lista de keypoints e a matriz de descritores binários (uma linha de 32 bytes por ponto; `None` se nenhum ponto foi achado). O 22 aplica o ORB só ao recorte da ROI vermelha, desenha os pontos com `cv2.drawKeypoints` e imprime quantos foram encontrados. A fileira de círculos e traços pretos do frame sintético existe exatamente para dar cantos ao ORB.

```python
# dnn_utils.py
def orb_features(img, max_features=250):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    orb = cv2.ORB_create(nfeatures=max_features)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des
```

```python
# 22_extrair_features_orb.py
if roi:
    x,y,w,h = roi
    crop = frame[y:y+h, x:x+w]
else:
    crop = frame

kp, des = orb_features(crop)
vis = cv2.drawKeypoints(crop, kp, None, color=(0,255,0), flags=0)
cv2.imwrite(str(OUT/'22_orb_roi.jpg'), vis)
print('Quantidade de keypoints ORB:', len(kp))
print('Descritores:', None if des is None else des.shape)
```

Onde aparece: `dnn_utils.py`, `22_extrair_features_orb.py`, `25_medir_tempo_etapas_pipeline.py`, `26_pipeline_integrado_imagem.py`, `27_pipeline_video_sintetico.py`, `30_pipeline_final_completo.py`.

### 3.20 Detectores clássicos: HOG de pessoas e Haar de rosto

Antes de classificar com a rede, o pipeline tenta os detectores clássicos das aulas 06 e 09. `hog_or_haar_detector` roda os dois no mesmo frame: o `HOGDescriptor` com o detector de pedestres padrão do OpenCV (`detectMultiScale` com passo de 8 pixels, padding 8 e escala 1,05) e o Haar Cascade de rosto frontal carregado de `cv2.data.haarcascades`. Cada caixa entra na lista como `('HOG', x, y, w, h)` ou `('Haar', x, y, w, h)`. No frame sintético não há pessoa nem rosto, então a lista sai vazia e o 23 escreve um aviso na imagem; isso é o comportamento esperado, e o desafio pede uma foto real em `data/pipeline/frame_real.jpg`. Note que a função recria o HOG e recarrega o XML do cascade a cada chamada, o que pesa no tempo medido pelos scripts 25, 26 e 30.

```python
# dnn_utils.py
def hog_or_haar_detector(frame):
    """Executa HOG de pessoas e Haar de face; retorna caixas encontradas, se houver."""
    boxes=[]
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
    for (x,y,w,h) in rects:
        boxes.append(('HOG', x,y,w,h))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face = cv2.CascadeClassifier(cascade_path)
    faces = face.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces:
        boxes.append(('Haar', x,y,w,h))
    return boxes
```

```python
# 23_detector_hog_haar.py
frame, _ = read_pipeline_frame()
boxes = hog_or_haar_detector(frame)
vis = frame.copy()

if boxes:
    for nome,x,y,w,h in boxes:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(vis, nome, (x,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
else:
    cv2.putText(vis, 'HOG/Haar executado: sem deteccao neste frame', (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,180), 2)
```

Onde aparece: `dnn_utils.py`, `23_detector_hog_haar.py`, `25_medir_tempo_etapas_pipeline.py`, `26_pipeline_integrado_imagem.py`, `30_pipeline_final_completo.py`.

### 3.21 Classificar a ROI detectada

A rede não precisa ver o frame inteiro. O 24 recorta a ROI vermelha com fatiamento NumPy (`frame[y:y+h, x:x+w]`), classifica só o recorte e desenha o top-3 sobre o frame completo, com a ROI em amarelo. Se a segmentação não achar nada, o recorte vira o frame inteiro e a caixa passa a ser `(0, 0, largura, altura)`. É a ligação entre a segmentação por cor e a rede: a cor diz onde olhar, a rede diz o que é. O desafio propõe trocar a ROI de cor pela caixa do HOG ou do Haar quando houver detecção.

```python
# 24_classificar_roi_detectada.py
frame, _ = read_pipeline_frame()
mask, roi = segment_red_hsv(frame)

if roi:
    x,y,w,h = roi
    crop = frame[y:y+h, x:x+w]
else:
    crop = frame
    x,y,w,h = 0,0,frame.shape[1], frame.shape[0]

prob = predict_opencv(crop)
vis = draw_top3(frame, topk(prob, 3))
cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 3)
cv2.imwrite(str(OUT/'24_classificar_roi.jpg'), vis)
```

Onde aparece: `24_classificar_roi_detectada.py`, e a mesma lógica dentro de `26_pipeline_integrado_imagem.py` e `30_pipeline_final_completo.py`.

### 3.22 Cronometrar cada etapa do pipeline

O 25 executa as cinco etapas em sequência sobre o frame corrigido e guarda o tempo de cada uma em um dicionário, sempre com o mesmo padrão: `t0 = perf_counter()`, a chamada, e `(perf_counter() - t0) * 1000`. Ao final imprime a tabela e o total. Esse é o dado que diz onde o pipeline gasta tempo e o que precisa ser otimizado para um alvo de FPS. Duas observações de leitura: a etapa `DNN` chama `predict_opencv(undist)` sem passar a rede, então o tempo dela inclui ler o `.tflite` do disco e montar a rede, não só a inferência; e a etapa `HOG_Haar` inclui recriar o HOG e carregar o cascade, como visto em 3.20. O desafio do script, rodar 10 vezes e tirar a média, também ajuda a diluir a primeira execução.

```python
# 25_medir_tempo_etapas_pipeline.py
frame, _ = read_pipeline_frame()
tempos = {}

t0=time.perf_counter(); undist=cv2.undistort(frame, DEFAULT_K, DEFAULT_DIST); tempos['undistort']=(time.perf_counter()-t0)*1000

t0=time.perf_counter(); mask, roi=segment_red_hsv(undist); tempos['HSV']=(time.perf_counter()-t0)*1000

t0=time.perf_counter(); kp, des=orb_features(undist); tempos['ORB']=(time.perf_counter()-t0)*1000

t0=time.perf_counter(); boxes=hog_or_haar_detector(undist); tempos['HOG_Haar']=(time.perf_counter()-t0)*1000

t0=time.perf_counter(); prob=predict_opencv(undist); tempos['DNN']=(time.perf_counter()-t0)*1000

print('Tempo por etapa:')
for k,v in tempos.items():
    print(f'{k:12s}: {v:7.2f} ms')
print(f'Total       : {sum(tempos.values()):7.2f} ms')
```

Onde aparece: `25_medir_tempo_etapas_pipeline.py`, e o mesmo cronômetro em `26_pipeline_integrado_imagem.py` e `30_pipeline_final_completo.py`.

### 3.23 O pipeline integrado em uma imagem

Os scripts 26 e 30 juntam tudo em um único frame anotado: undistort, ROI por HSV em amarelo, keypoints ORB em verde desenhados na posição do frame (somando o deslocamento `x, y` da ROI), caixas HOG ou Haar em azul, top-3 da rede aplicada ao recorte, e um painel branco no canto inferior esquerdo com o tempo de cada etapa e o total. A diferença entre os dois é de acabamento: o 26 desenha até 40 keypoints e só quando há ROI, e o 30 desenha até 60, sempre, rotula a ROI com o texto `ROI HSV`, imprime a origem do frame (sintético ou real) e organiza o código em seis etapas comentadas. Os dois chamam `predict_opencv` sem a rede, com o efeito descrito em 3.22.

```python
# 30_pipeline_final_completo.py
# 2. Segmentar ROI por cor HSV.
t0=time.perf_counter()
mask, roi_hsv = segment_red_hsv(frame_corrigido)
tempos['HSV'] = (time.perf_counter()-t0)*1000

vis = frame_corrigido.copy()
if roi_hsv:
    x,y,w,h = roi_hsv
    roi_img = frame_corrigido[y:y+h, x:x+w]
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 3)
    cv2.putText(vis, 'ROI HSV', (x,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,120,120), 2)
else:
    roi_img = frame_corrigido
    x,y,w,h = 0,0,frame_corrigido.shape[1], frame_corrigido.shape[0]

# 3. Extrair features ORB.
t0=time.perf_counter()
kp, des = orb_features(roi_img)
tempos['ORB'] = (time.perf_counter()-t0)*1000
for p in kp[:60]:
    px, py = int(p.pt[0] + x), int(p.pt[1] + y)
    cv2.circle(vis, (px,py), 2, (0,255,0), -1)
...
# 5. Classificar ROI com OpenCV DNN.
t0=time.perf_counter()
prob = predict_opencv(roi_img)
tempos['DNN'] = (time.perf_counter()-t0)*1000
vis = draw_top3(vis, topk(prob, 3), load_labels())
```

Onde aparece: `26_pipeline_integrado_imagem.py`, `30_pipeline_final_completo.py`.

### 3.24 Vídeo sintético com `np.roll` e `cv2.VideoWriter`

Para testar o pipeline em movimento sem câmera, o 27 pega o frame sintético e o desloca horizontalmente com `np.roll(base, shift=i*2, axis=1)`, 2 pixels por frame, durante 60 frames. Em cada um, segmenta a ROI, conta os keypoints ORB dentro dela e escreve o número no canto. Os frames vão para `saidas/27_video_pipeline_sintetico.avi` com o codec MJPG a 15 fps, o que dá 4 segundos de vídeo. O deslocamento máximo é de 118 pixels, então o retângulo vermelho nunca cruza a borda. O desafio é acrescentar a classificação DNN em cada frame, e o custo disso é o que a aula quer que o aluno sinta.

```python
# 27_pipeline_video_sintetico.py
base_path = create_pipeline_frame()
base = cv2.imread(str(base_path))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_path = OUT / '27_video_pipeline_sintetico.avi'
writer = cv2.VideoWriter(str(out_path), fourcc, 15.0, (640,480))

for i in range(60):
    frame = np.roll(base, shift=i*2, axis=1)
    mask, roi = segment_red_hsv(frame)
    vis = frame.copy()
    if roi:
        x,y,w,h=roi
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 2)
        kp,_ = orb_features(frame[y:y+h, x:x+w])
        cv2.putText(vis, f'ORB: {len(kp)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    writer.write(vis)
writer.release()
```

Onde aparece: `27_pipeline_video_sintetico.py`.

### 3.25 Da imagem à webcam: o notebook e o pseudocódigo

Os dois últimos passos antes do pipeline final não executam nada. O 28 só imprime onde estaria o notebook `notebooks/pipeline_integrado_dnn.ipynb` e como abri-lo; o arquivo não veio no zip. O 29 imprime o pseudocódigo do laboratório: carregar `K` e `dist` da calibração real, carregar a rede, abrir `cv2.VideoCapture(0)` e, para cada frame, repetir as etapas do 30 e mostrar o resultado na tela. Tudo o que o aluno precisa para escrever esse programa já está nos scripts 20 a 27 e no 30; o que muda é a fonte do frame e o laço.

```python
# 29_desafio_webcam_real.py
PSEUDOCODIGO = r"""
carregar K e dist da calibração real
carregar modelo DNN
abrir cv2.VideoCapture(0)

para cada frame:
    corrigir distorção com cv2.undistort
    segmentar ROI por HSV
    extrair ORB na ROI
    aplicar HOG ou Haar
    classificar ROI com DNN
    desenhar top-3, caixas, keypoints e tempos
    mostrar frame

fechar câmera
"""
```

Onde aparece: `28_gerar_notebook_pipeline.py`, `29_desafio_webcam_real.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `01_configurar_pastas.py` | Cria `data/classificacao/`, `data/pipeline/`, `modelos/` e `saidas/` e imprime os caminhos | nada | as quatro pastas | `dnn_utils.py` |
| `02_criar_frame_sintetico_pipeline.py` | Desenha o frame 640x480 com retângulo vermelho e detalhes para ORB | nada | `data/pipeline/frame_sintetico.jpg` | `dnn_utils.py` |
| `03_criar_imagens_teste_sinteticas.py` | Cria 10 imagens sintéticas nomeadas e o CSV de rótulos vazio | nada | `data/classificacao/01_carro.jpg` a `10_capacete.jpg`, `data/labels_top1.csv` | `dnn_utils.py` |
| `04_entender_blob_from_image.py` | Monta o blob da primeira imagem e imprime forma, dtype, mínimo e máximo | primeira imagem de `data/classificacao/` | texto | `dnn_utils.py` |
| `05_baixar_labels_imagenet.py` | Baixa os 1000 nomes de classe do ImageNet | internet | `modelos/imagenet_labels.txt` | `dnn_utils.py` |
| `06_carregar_mobilenetv2_keras.py` | Carrega a MobileNetV2 com pesos ImageNet e imprime as formas de entrada e saída | pesos do Keras (baixados na primeira vez) | texto | `tensorflow` |
| `07_converter_keras_para_tflite.py` | Converte a MobileNetV2 para TensorFlow Lite e imprime o tamanho | pesos do Keras | `modelos/mobilenetv2_imagenet.tflite` | `tensorflow`, `dnn_utils.py` |
| `08_carregar_tflite_no_opencv_dnn.py` | Carrega o `.tflite` no OpenCV DNN e conta as camadas | `.tflite` | texto | `dnn_utils.py`, 07 |
| `09_forward_pass_opencv.py` | Faz `setInput` e `forward` e imprime a forma da saída e 10 valores | primeira imagem, `.tflite` | texto | `dnn_utils.py`, 07 |
| `10_extrair_top3_opencv.py` | Softmax, top-3 e rótulos da primeira imagem | primeira imagem, `.tflite`, rótulos | texto | `dnn_utils.py`, 07, 05 (opcional) |
| `11_sobrepor_top3_na_imagem.py` | Desenha o top-3 sobre a primeira imagem | primeira imagem, `.tflite` | `saidas/11_top3_<nome>.jpg` | `dnn_utils.py`, 07 |
| `12_processar_10_imagens.py` | Top-3 desenhado nas 10 primeiras imagens da pasta | `data/classificacao/`, `.tflite` | `saidas/12_top3_<nome>.jpg` (10 arquivos) | `dnn_utils.py`, 07 |
| `13_inferencia_keras_top3.py` | Mesma imagem classificada no Keras com `decode_predictions` | primeira imagem, pesos do Keras | texto | `tensorflow`, `dnn_utils.py` |
| `14_medir_latencia_opencv.py` | Latência média e desvio de 20 inferências no OpenCV DNN | primeira imagem, `.tflite` | texto | `dnn_utils.py`, 07 |
| `15_medir_latencia_keras.py` | Latência média e desvio de 20 `model.predict` no Keras | primeira imagem, pesos do Keras | texto | `tensorflow`, `dnn_utils.py` |
| `16_medir_memoria_opencv.py` | RSS antes e depois de carregar e inferir no OpenCV DNN | primeira imagem, `.tflite` | texto | `psutil`, `dnn_utils.py`, 07 |
| `17_medir_memoria_keras.py` | RSS antes e depois de carregar e inferir no Keras | primeira imagem, pesos do Keras | texto | `psutil`, `tensorflow`, `dnn_utils.py` |
| `18_calcular_acuracia_top1.py` | Compara o top-1 do OpenCV DNN com os rótulos do CSV e calcula a acurácia | `data/labels_top1.csv` preenchido, `.tflite`, rótulos | texto | `dnn_utils.py`, 03, 05, 07 |
| `19_tabela_comparativa.py` | Tabela pandas OpenCV DNN contra Keras, com zeros para preencher | nada | texto | `pandas` |
| `20_undistort_frame_sintetico.py` | Corrige a distorção do frame com `K` e `dist` fixos e monta painel lado a lado | frame do pipeline | `saidas/20_undistort_painel.jpg` | `dnn_utils.py` |
| `21_segmentacao_hsv_roi.py` | Segmenta o vermelho em HSV e desenha a ROI | frame do pipeline | `saidas/21_mascara_hsv.jpg`, `saidas/21_roi_hsv.jpg` | `dnn_utils.py` |
| `22_extrair_features_orb.py` | ORB no recorte da ROI, desenha os keypoints | frame do pipeline | `saidas/22_orb_roi.jpg` | `dnn_utils.py` |
| `23_detector_hog_haar.py` | HOG de pessoas e Haar de rosto no frame | frame do pipeline (ou `frame_real.jpg`) | `saidas/23_detector_hog_haar.jpg` | `dnn_utils.py` |
| `24_classificar_roi_detectada.py` | Classifica o recorte da ROI com OpenCV DNN e desenha o top-3 | frame do pipeline, `.tflite` | `saidas/24_classificar_roi.jpg` | `dnn_utils.py`, 07 |
| `25_medir_tempo_etapas_pipeline.py` | Tempo de undistort, HSV, ORB, HOG/Haar e DNN | frame do pipeline, `.tflite` | texto | `dnn_utils.py`, 07 |
| `26_pipeline_integrado_imagem.py` | As cinco etapas em uma imagem anotada com tempos | frame do pipeline, `.tflite` | `saidas/26_pipeline_integrado.jpg` | `dnn_utils.py`, 07 |
| `27_pipeline_video_sintetico.py` | 60 frames deslocados com `np.roll`, ROI e contagem ORB, gravados em AVI | frame sintético | `saidas/27_video_pipeline_sintetico.avi` | `dnn_utils.py` |
| `28_gerar_notebook_pipeline.py` | Imprime onde estaria o notebook (não veio no zip) | nada | texto | nada |
| `29_desafio_webcam_real.py` | Imprime o pseudocódigo do laboratório com webcam | nada | texto | nada |
| `30_pipeline_final_completo.py` | Pipeline final: undistort, HSV, ORB, HOG/Haar, DNN e tempos em uma imagem | frame do pipeline (ou real), `.tflite` | `saidas/30_pipeline_final_completo.jpg` | `dnn_utils.py`, 07 |
| `dnn_utils.py` | Funções compartilhadas: pastas, imagens sintéticas, blob, rótulos, rede, softmax, top-k, desenho, cronômetro, HSV, ORB, HOG e Haar | importado | nada | cv2, numpy |
| `requirements.txt` | Lista de pacotes do professor | nada | nada | nada |

## 5. Passo a passo para executar

Todos os comandos abaixo são executados de dentro de `aula_14_OpenCV_DNN/exemplos_em_python/`, com o venv do projeto ativado. Nenhum script abre janela: os resultados vão para o terminal e para `saidas/`. Os avisos do TensorFlow sobre oneDNN e absl no início dos scripts 06, 07, 13, 15 e 17 são normais.

Passo 1. Criar as pastas.

```
python 01_configurar_pastas.py
```

Passo 2. Gerar o frame do pipeline e as 10 imagens sintéticas. Abra `data/pipeline/frame_sintetico.jpg` e `data/classificacao/` para ver o que foi criado. Se você já tem 10 fotos reais, copie-as para `data/classificacao/` e apague as sintéticas; veja a seção 7 sobre a ordem dos nomes.

```
python 02_criar_frame_sintetico_pipeline.py
python 03_criar_imagens_teste_sinteticas.py
```

Passo 3. Entender o blob. Espere `(360, 480, 3)` virando `(1, 3, 224, 224)`, `float32`, valores entre -1 e 1.

```
python 04_entender_blob_from_image.py
```

Passo 4. Baixar os rótulos (precisa de internet). O terminal mostra as 10 primeiras classes.

```
python 05_baixar_labels_imagenet.py
```

Passo 5. Carregar a MobileNetV2 no Keras. Na primeira vez o Keras baixa os pesos. Espere `(None, 224, 224, 3)` e `(None, 1000)`.

```
python 06_carregar_mobilenetv2_keras.py
```

Passo 6. Converter para TensorFlow Lite. Este passo é obrigatório para tudo o que vem depois no OpenCV. O arquivo `modelos/mobilenetv2_imagenet.tflite` deve ter cerca de 14 MB.

```
python 07_converter_keras_para_tflite.py
```

Passo 7. Carregar no OpenCV e fazer o primeiro forward. Espere a contagem de camadas e uma saída `(1, 1000)`.

```
python 08_carregar_tflite_no_opencv_dnn.py
python 09_forward_pass_opencv.py
```

Passo 8. Top-3 no terminal, na imagem e nas 10 imagens. Nas imagens sintéticas os rótulos não têm relação com o nome escrito nelas; isso é esperado. Repare que as porcentagens saem muito baixas (seção 3.10).

```
python 10_extrair_top3_opencv.py
python 11_sobrepor_top3_na_imagem.py
python 12_processar_10_imagens.py
```

Passo 9. A mesma imagem no Keras. Compare os três rótulos com os do passo 8; a ordem deve coincidir, as porcentagens não.

```
python 13_inferencia_keras_top3.py
```

Passo 10. Latência dos dois backends. Rode cada um mais de uma vez para ver a variação.

```
python 14_medir_latencia_opencv.py
python 15_medir_latencia_keras.py
```

Passo 11. Memória dos dois backends.

```
python 16_medir_memoria_opencv.py
python 17_medir_memoria_keras.py
```

Passo 12. Acurácia top-1. Antes, abra `data/labels_top1.csv` e preencha a coluna `label_esperado` com o nome em inglês do objeto de cada foto (`car`, `mug`, `ball`...). Com o CSV vazio o resultado é `0/0`.

```
python 18_calcular_acuracia_top1.py
```

Passo 13. Tabela comparativa. Edite os zeros do script com os valores dos passos 10 a 12 antes de rodar.

```
python 19_tabela_comparativa.py
```

Passo 14. Etapas clássicas, uma por vez. Cada script grava uma imagem em `saidas/`. No 23, sem foto real, a imagem sai com o aviso de que nada foi detectado.

```
python 20_undistort_frame_sintetico.py
python 21_segmentacao_hsv_roi.py
python 22_extrair_features_orb.py
python 23_detector_hog_haar.py
```

Passo 15. Classificar só a ROI.

```
python 24_classificar_roi_detectada.py
```

Passo 16. Tempos por etapa e pipeline integrado em imagem.

```
python 25_medir_tempo_etapas_pipeline.py
python 26_pipeline_integrado_imagem.py
```

Passo 17. Vídeo sintético. Abra `saidas/27_video_pipeline_sintetico.avi` em um player que aceite MJPG (VLC, por exemplo).

```
python 27_pipeline_video_sintetico.py
```

Passo 18. Notebook e pseudocódigo. Os dois só imprimem texto.

```
python 28_gerar_notebook_pipeline.py
python 29_desafio_webcam_real.py
```

Passo 19. Pipeline final. Se houver `data/pipeline/frame_real.jpg`, ele é usado no lugar do sintético, e o terminal diz qual foi a origem.

```
python 30_pipeline_final_completo.py
```

Ao terminar todos os passos, `saidas/` contém: `11_top3_01_carro.jpg`, os 10 `12_top3_*.jpg`, `20_undistort_painel.jpg`, `21_mascara_hsv.jpg`, `21_roi_hsv.jpg`, `22_orb_roi.jpg`, `23_detector_hog_haar.jpg`, `24_classificar_roi.jpg`, `26_pipeline_integrado.jpg`, `27_video_pipeline_sintetico.avi` e `30_pipeline_final_completo.jpg`. Em `data/` ficam as imagens de teste, o CSV e o frame do pipeline; em `modelos/`, os rótulos e o `.tflite`.

## 6. Exercícios e itens da aula

Não há itens numerados A e B nesta aula. Os exercícios são os blocos `# DESAFIO DO ALUNO` no fim de cada script, transcritos abaixo em resumo. O desafio de laboratório é o do script 29.

| Script | Desafio |
|---|---|
| 01 | Colocar pelo menos 10 imagens reais, de categorias diferentes, em `data/classificacao/`. |
| 02 | Abrir a imagem e identificar visualmente a parte que será segmentada por cor. |
| 03 | Substituir as imagens sintéticas por 10 imagens reais de categorias distintas. |
| 04 | Explicar por que uma imagem BGR do OpenCV precisa de `swapRB=True`. |
| 05 | Contar quantas classes existem no arquivo baixado. |
| 06 | Identificar quantas classes a saída possui e relacionar com o ImageNet. |
| 07 | Ativar `converter.optimizations = [tf.lite.Optimize.DEFAULT]` e comparar o tamanho do arquivo. |
| 08 | Imprimir os nomes das 10 primeiras camadas. |
| 09 | Explicar por que a maior saída indica a classe mais provável. |
| 10 | Trocar `k=3` por `k=5` e observar a diferença. |
| 11 | Mover o texto do top-3 para o canto inferior esquerdo. |
| 12 | Garantir que as 10 imagens sejam de categorias realmente diferentes. |
| 13 | Comparar visualmente o top-3 do Keras com o do OpenCV DNN. |
| 14 | Repetir com `loops=100` e comparar a estabilidade da média. |
| 15 | Comparar CPU e GPU, se o computador tiver GPU configurada. |
| 16 | Medir de novo depois de várias inferências e ver se a memória estabiliza. |
| 17 | Comparar com o valor medido no 16. |
| 18 | Preencher `labels_top1.csv` com as 10 imagens reais e rodar de novo. |
| 19 | Substituir os zeros pelas medições reais do seu computador. |
| 20 | Substituir `DEFAULT_K` e `DEFAULT_DIST` pelos valores da sua câmera calibrada. |
| 21 | Alterar os limites HSV para segmentar outra cor. |
| 22 | Aumentar `nfeatures` em `dnn_utils.orb_features` e comparar a quantidade de pontos. |
| 23 | Testar com uma foto contendo pessoa ou rosto em `data/pipeline/frame_real.jpg`. |
| 24 | Trocar a ROI HSV pela caixa devolvida pelo HOG ou pelo Haar, quando houver detecção. |
| 25 | Rodar 10 vezes e calcular a média de cada etapa. |
| 26 | Usar uma imagem real e ajustar a segmentação HSV para uma cor presente nela. |
| 27 | Acrescentar a classificação DNN em cada frame do vídeo. |
| 28 | Acrescentar ao notebook uma célula que salve o frame final anotado (o notebook não veio no zip). |
| 29 | **Desafio de laboratório:** transformar o pseudocódigo da webcam em um programa funcional. |
| 30 | Substituir o frame sintético por uma captura real e usar a calibração real da câmera. |

Sugestão de ordem para o laboratório do 29: comece copiando o `30_pipeline_final_completo.py`, troque `read_pipeline_frame()` por um laço `while` com `cv2.VideoCapture(0)` e `cap.read()`, carregue a rede uma única vez antes do laço com `get_opencv_net()` e passe-a para `predict_opencv`, mostre o resultado com `cv2.imshow` e encerre com a tecla `q`. Só depois troque `DEFAULT_K` e `DEFAULT_DIST` pela calibração da sua câmera.

## 7. Armadilhas e dicas

- Rode o `07_converter_keras_para_tflite.py` antes de qualquer script que use o OpenCV DNN (08 a 12, 14, 16, 18, 24, 25, 26 e 30). A mensagem de erro `Modelo não encontrado` manda executar o 06, mas o conversor é o 07.
- O 05 precisa de internet. Sem `modelos/imagenet_labels.txt`, os rótulos viram `classe_000` a `classe_999` e nada quebra, mas o 18 não consegue comparar com os nomes esperados.
- A primeira execução de 06, 07, 13, 15 ou 17 baixa os pesos da MobileNetV2 (cerca de 14 MB) para `~/.keras/models/`. Isso demora e depende de internet uma única vez.
- As confianças impressas pelo OpenCV DNN saem perto de 0,1%, porque o `dnn_utils.softmax` é aplicado sobre uma saída que já é softmax (a MobileNetV2 do Keras inclui essa camada). O ranking está certo; as porcentagens reais são as do 13. Para corrigir seria preciso não aplicar o segundo softmax, mas o script fica como o professor escreveu.
- As imagens sintéticas do 03 são formas geométricas com texto; o top-3 nelas não significa nada. Acurácia só faz sentido com fotos reais.
- `list_classification_images` ordena por nome e os scripts pegam `[0]` (a primeira) ou `[:10]` (o 12). Os nomes sintéticos começam com dígitos e ficam antes de qualquer nome com letra, então fotos reais chamadas `gato.jpg` seriam ignoradas. Ao usar fotos reais, apague as sintéticas ou nomeie as suas com prefixo numérico.
- Rodar o `03_criar_imagens_teste_sinteticas.py` de novo recria as 10 imagens e sobrescreve `data/labels_top1.csv` com a coluna vazia. Preencha o CSV depois do 03, e não rode o 03 outra vez.
- No 18, o rótulo esperado é comparado por substring com o rótulo do ImageNet em minúsculas e com `_` no lugar de espaço. Escreva em inglês e de forma curta: `car` casa com `sports_car`, `mug` casa com `coffee_mug`. Linhas com `label_esperado` vazio são puladas, e com todas vazias o resultado é `0/0`.
- `predict_opencv` sem o argumento `net` recarrega o `.tflite` do disco a cada chamada. Isso acontece no 11, no 18 (uma vez por imagem) e nos scripts 24, 25, 26 e 30. Nos scripts 25, 26 e 30 o tempo da etapa `DNN` inclui esse carregamento.
- `hog_or_haar_detector` recria o `HOGDescriptor` e recarrega o XML do cascade a cada chamada. O tempo da etapa `HOG_Haar` inclui isso.
- Os scripts 16 e 17 medem só a diferença de RSS entre antes e depois de carregar o modelo. No 17 o `import tensorflow` já aconteceu antes da primeira medição, então o custo do runtime do TensorFlow não aparece na diferença.
- `DEFAULT_K` e `DEFAULT_DIST` são valores de exemplo, não calibração. No frame sintético, que não tem distorção, o `undistort` do 20 entorta as bordas em vez de corrigir. Com câmera real, use os valores obtidos pelo procedimento da aula 16.
- No 23 é normal não detectar nada no frame sintético: não há pessoa nem rosto. Coloque uma foto com uma pessoa em `data/pipeline/frame_real.jpg`; a partir daí todos os scripts de 20 a 26 e o 30 passam a usar essa foto, e a segmentação de vermelho pode não achar ROI nela (a caixa vira o frame inteiro).
- A segmentação usa duas faixas de matiz porque o vermelho está nas duas pontas da escala 0 a 180 do OpenCV. Para outra cor basta uma faixa; ajuste também os mínimos de saturação (80) e valor (70).
- O `blobFromImage` com `crop=False` deforma a imagem para 224x224 sem respeitar a proporção. Recortes muito alongados (uma ROI estreita, por exemplo) chegam distorcidos à rede.
- O 28 aponta para `notebooks/pipeline_integrado_dnn.ipynb`, que não veio no zip. O pacote `notebook` do `requirements.txt` do professor existe só para ele.
- O `requirements.txt` do professor lista `opencv-python`. No projeto o pacote é o `opencv-contrib-python`; não instale os dois.
- `matplotlib` e `pillow` estão no `requirements.txt` do professor, mas nenhum script os importa.
- O vídeo do 27 é AVI com MJPG. Alguns players do Windows não abrem; use o VLC ou leia com `cv2.VideoCapture`.
- Os scripts que gravam em `saidas/` não criam a pasta sozinhos, mas `ensure_dirs()` roda antes em todos eles, por dentro de `read_pipeline_frame` ou `list_classification_images`, então não há problema de ordem. No repositório `saidas/` já existe com `.gitkeep`; `data/`, `modelos/` e `*.tflite` ficam fora do git.
- Tudo roda em CPU (`DNN_TARGET_CPU` no OpenCV, TensorFlow sem GPU no Windows nativo). O desafio do 15 sobre GPU só se aplica a máquinas com CUDA configurado.

## 8. Glossário

| Termo | Significado |
|---|---|
| DNN | Deep Neural Network, rede neural profunda. Também o nome do módulo `cv2.dnn`. |
| Inferência | Uso de um modelo já treinado para produzir uma saída a partir de uma entrada. É tudo o que o `cv2.dnn` faz. |
| Forward pass | Passagem da entrada pela rede até a saída, sem cálculo de gradientes. `net.forward()`. |
| Blob | Tensor de entrada da rede no formato `(N, C, H, W)` em `float32`, criado por `cv2.dnn.blobFromImage`. |
| NCHW e NHWC | Ordens de eixos de um lote de imagens: lote, canais, altura, largura (OpenCV) ou lote, altura, largura, canais (Keras). |
| scalefactor | Fator multiplicado sobre cada pixel depois de subtrair a média. Aqui `1/127.5`. |
| mean | Valor subtraído de cada canal antes da escala. Aqui `127.5` nos três canais. |
| swapRB | Troca dos canais R e B no blob, necessária porque o OpenCV usa BGR e a rede foi treinada em RGB. |
| crop | Opção do `blobFromImage`: recortar o centro mantendo a proporção (True) ou deformar para o tamanho (False). |
| MobileNetV2 | Arquitetura de CNN leve, projetada para dispositivos móveis, com cerca de 3,5 milhões de parâmetros. |
| ImageNet | Conjunto de dados de classificação com 1000 classes em que a MobileNetV2 foi treinada. |
| Pesos pré-treinados | Parâmetros já ajustados em outro conjunto de dados, baixados com `weights='imagenet'`. |
| Keras | API de alto nível para redes neurais, integrada ao TensorFlow como `tensorflow.keras`. |
| TensorFlow Lite (TFLite) | Formato e runtime de modelos para dispositivos móveis e embarcados. Extensão `.tflite`. |
| Conversor | `tf.lite.TFLiteConverter`, que transforma um modelo Keras em bytes TFLite. |
| Quantização | Redução da precisão dos pesos (de 32 bits para 8, por exemplo) para diminuir tamanho e tempo. `tf.lite.Optimize.DEFAULT`. |
| readNetFromTFLite | Função do `cv2.dnn` que carrega um modelo `.tflite`. |
| Backend e target | Implementação usada pelo `cv2.dnn` (OpenCV, CUDA, OpenVINO) e dispositivo de execução (CPU, GPU). |
| Camada | Unidade de processamento da rede; `getLayerNames()` lista as que o OpenCV montou. |
| Logits | Saída bruta de uma rede antes do softmax. |
| Softmax | Função que transforma um vetor em probabilidades que somam 1, exponenciando e normalizando. |
| Top-k | As `k` classes de maior probabilidade. Aqui top-3 e top-1. |
| Acurácia top-1 | Fração de imagens cuja classe mais provável coincide com o rótulo esperado. |
| preprocess_input | Função do Keras que aplica o pré-processamento próprio de cada arquitetura; na MobileNetV2, escala para -1 a 1. |
| decode_predictions | Função do Keras que traduz o vetor de 1000 saídas em rótulos legíveis do ImageNet. |
| Latência | Tempo de uma inferência, medido em milissegundos com `time.perf_counter`. |
| Warmup | Execuções descartadas antes de medir, para não contar custos de inicialização. |
| RSS | Resident Set Size, memória física ocupada pelo processo, lida com `psutil`. |
| psutil | Biblioteca para consultar processos e recursos do sistema. |
| Calibração | Processo que obtém a matriz intrínseca `K` e os coeficientes de distorção de uma câmera. |
| Matriz intrínseca (K) | Matriz 3x3 com as focais `fx`, `fy` e o ponto principal `cx`, `cy`. |
| Coeficientes de distorção | Vetor `k1, k2, p1, p2, k3` que descreve a distorção radial e tangencial da lente. |
| undistort | `cv2.undistort`, que remove a distorção de uma imagem dados `K` e os coeficientes. |
| HSV | Espaço de cor com matiz (0 a 180 no OpenCV), saturação e valor. |
| inRange | Função que gera uma máscara binária com os pixels dentro de uma faixa de valores. |
| medianBlur | Filtro de mediana, usado aqui para limpar pontos isolados da máscara. |
| Contorno | Curva que delimita uma região conectada de uma máscara binária. |
| ROI | Region of Interest, região de interesse `(x, y, w, h)`. Aqui, a caixa da mancha vermelha. |
| ORB | Oriented FAST and Rotated BRIEF, detector e descritor de pontos-chave binário. |
| Keypoint | Ponto-chave: posição notável da imagem detectada pelo ORB. |
| Descritor | Vetor que descreve a vizinhança de um keypoint; no ORB, 32 bytes. |
| HOG | Histogram of Oriented Gradients, descritor usado com um SVM no detector de pedestres do OpenCV. |
| Haar Cascade | Detector em cascata de classificadores simples, aqui o de rosto frontal de `cv2.data.haarcascades`. |
| detectMultiScale | Método que varre a imagem em várias escalas e devolve as caixas detectadas. |
| Pipeline | Sequência de etapas encadeadas: undistort, HSV, ORB, HOG ou Haar, DNN. |
| MJPG | Motion JPEG, codec de vídeo usado no `VideoWriter` do 27. |
| np.roll | Função NumPy que desloca um array com retorno pela borda oposta, usada para animar o frame. |
| Frame sintético | Imagem desenhada por código para testar o pipeline sem foto ou câmera. |
| Pseudocódigo | Descrição em texto dos passos de um programa, sem sintaxe executável. O 29 é um. |
