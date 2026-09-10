# Aula 04 – Tensores

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula apresenta o tensor como a estrutura de dados fundamental da visão computacional moderna. Em vez de usar OpenCV, os doze scripts usam apenas TensorFlow e mostram, passo a passo, como uma imagem é representada, inspecionada e transformada como um tensor numérico: do escalar ao vetor, do vetor à matriz, da matriz ao pixel RGB, do pixel à imagem inteira com altura, largura e canais, e da imagem ao lote de imagens que uma rede neural consome.

O fio condutor é a progressão de dimensões. Os scripts 01 a 03 constroem tensores à mão. Os scripts 04 a 10 carregam uma imagem real do disco e aplicam a ela as operações mais comuns de pré-processamento: inspecionar shape, dtype e faixa de valores, acrescentar a dimensão de lote, converter para escala de cinza, separar e recombinar canais, normalizar para o intervalo [0, 1] e redimensionar. O script 11 mostra o que uma camada de convolução faz com o shape de um tensor de entrada. O script 12 fecha a aula juntando tudo em um pipeline `tf.data`, que é a forma padrão de alimentar modelos Keras.

## 2. Ambiente e pré-requisitos

- Biblioteca usada: apenas `tensorflow`. Nenhum script importa OpenCV, NumPy ou Matplotlib diretamente.
- Versão fixada pelo professor em `requirements.txt`: `tensorflow==2.17.0`.
- Python usado pelo professor: 3.12.2 no Windows, segundo `README_do_professor.txt`. A máquina local tem Python 3.14.3. Confirme se existe wheel de TensorFlow para 3.14 antes de instalar, ou crie um venv com Python 3.12.
- Instalação sugerida pelo professor, no PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install tensorflow
```

- De que pasta rodar: de dentro de `aula_04_Tensor/exemplos_em_python/`. Os scripts 04 a 10 e 12 usam o caminho relativo `assets/carro_real.png`, então o diretório de trabalho precisa ser essa pasta.
- Material que falta: o arquivo `assets/carro_real.png` não veio no zip. Ele deve ficar em `aula_04_Tensor/exemplos_em_python/assets/carro_real.png`. Qualquer PNG colorido serve para acompanhar a aula, pois os scripts só inspecionam e transformam a imagem, sem depender do conteúdo dela.
- Webcam: nenhum script usa câmera.
- GPU: no Windows nativo o TensorFlow 2.11 ou superior usa apenas CPU. Para GPU NVIDIA é preciso WSL2, conforme https://www.tensorflow.org/install/pip. Para esta aula a CPU é suficiente.

## 3. Conceitos-chave

### 3.1 Tensor, rank, shape e dtype

Um tensor é um arranjo numérico de N dimensões. Um escalar é um tensor de rank 0 (nenhum eixo), um vetor é rank 1 (um eixo), uma matriz é rank 2 (dois eixos), e assim por diante. Três propriedades descrevem qualquer tensor: o shape, que é a tupla com o tamanho de cada eixo; o rank, que é o número de eixos, ou seja, o comprimento do shape; e o dtype, que é o tipo numérico dos elementos (inteiro de 32 bits, inteiro sem sinal de 8 bits, ponto flutuante de 32 bits etc.).

No TensorFlow, `tf.constant(valor)` cria um tensor imutável a partir de um número ou lista Python. O dtype é inferido do valor quando não é informado: inteiros Python viram `int32`. O atributo `.shape` devolve o shape, `tf.rank(tensor)` devolve o rank como um tensor escalar, e `.numpy()` converte qualquer tensor para um valor NumPy legível.

Trecho de `01_compare_escalar_vetor_e_matriz.py`:

```python
itens = {
    "escalar": tf.constant(7),
    "vetor": tf.constant([10, 20, 30]),
    "matriz": tf.constant([[1, 2], [3, 4]]),
}

for nome, tensor in itens.items():
    print(nome, "shape=", tensor.shape,
          "rank=", tf.rank(tensor).numpy(),
          "dtype=", tensor.dtype)
```

Saída esperada: o escalar tem shape `()` e rank 0, o vetor tem shape `(3,)` e rank 1, a matriz tem shape `(2, 2)` e rank 2. Todos com dtype `int32`.

Onde aparece: `01_compare_escalar_vetor_e_matriz.py`, `02_represente_um_pixel_rgb.py`, `05_inspecione_shape_dtype_e_faixa.py`.

### 3.2 Pixel RGB como vetor uint8

Um pixel colorido é um vetor de três componentes, uma para cada canal: vermelho (R), verde (G) e azul (B). Em imagens de 8 bits cada componente é um inteiro de 0 a 255, e o dtype correto para isso é `tf.uint8` (inteiro sem sinal de 8 bits). Informar `dtype=tf.uint8` em `tf.constant` faz o tensor ocupar exatamente um byte por componente e deixa explícito que os valores são intensidades de imagem.

A indexação com colchetes funciona como em listas Python e NumPy: `pixel[0]` é o canal R, `pixel[1]` o canal G e `pixel[2]` o canal B. Cada índice devolve um tensor escalar, e `.numpy()` extrai o número.

Trecho de `02_represente_um_pixel_rgb.py`:

```python
pixel = tf.constant([230, 80, 35], dtype=tf.uint8)

print("pixel:", pixel.numpy())
print("shape:", pixel.shape)
print("rank:", tf.rank(pixel).numpy())
print("R:", pixel[0].numpy())
print("G:", pixel[1].numpy())
print("B:", pixel[2].numpy())
```

Saída esperada: shape `(3,)`, rank 1, e os valores 230, 80 e 35, que formam um tom de laranja avermelhado.

Onde aparece: `02_represente_um_pixel_rgb.py`.

### 3.3 Imagem como tensor (altura, largura, canais)

Uma imagem colorida é um tensor de rank 3 com shape `(H, W, C)`: H linhas de altura, W colunas de largura e C canais. Essa ordem, chamada de "channels last", é a convenção padrão do TensorFlow e do Keras. O eixo dos canais é o último, e por isso muitas operações da aula usam `axis=-1` para se referir a ele.

O script 03 monta uma imagem sintética de 4 por 6 pixels totalmente vermelha. Ele cria três tensores de shape `(4, 6, 1)`, um por canal, e os junta ao longo do último eixo. As funções usadas são:

- `tf.fill(shape, valor)`: cria um tensor com o shape informado, todo preenchido com um valor. O dtype vem do valor, aqui `int32`.
- `tf.zeros(shape, dtype=...)`: cria um tensor só de zeros.
- `tf.concat(lista, axis=-1)`: concatena tensores ao longo de um eixo. Com `axis=-1`, três tensores `(4, 6, 1)` viram um único `(4, 6, 3)`.
- `tf.cast(tensor, tf.uint8)`: converte o dtype. Os três canais nascem como `int32` e a conversão final deixa a imagem no formato de 8 bits.

Trecho de `03_crie_uma_imagem_sintetica.py`:

```python
r = tf.fill((4, 6, 1), 255)
g = tf.zeros((4, 6, 1), dtype=tf.int32)
b = tf.zeros((4, 6, 1), dtype=tf.int32)

imagem = tf.concat([r, g, b], axis=-1)
imagem = tf.cast(imagem, tf.uint8)

print("shape:", imagem.shape)
print("pixel [0,0]:", imagem[0, 0].numpy())
```

Saída esperada: shape `(4, 6, 3)` e o pixel do canto superior esquerdo igual a `[255 0 0]`. Note que `imagem[0, 0]` indexa linha 0 e coluna 0 e devolve o vetor de três canais.

Onde aparece: `03_crie_uma_imagem_sintetica.py`. O mesmo layout `(H, W, C)` aparece em todos os scripts de 04 a 10 e 12.

### 3.4 Leitura e decodificação de imagem com tf.io

Carregar uma imagem do disco no TensorFlow acontece em dois passos. Primeiro `tf.io.read_file(caminho)` lê o arquivo inteiro como um tensor de bytes, sem interpretar nada. Depois `tf.io.decode_png(bytes, channels=3)` decodifica esses bytes como PNG e devolve um tensor `uint8` de shape `(H, W, 3)`. O parâmetro `channels=3` força três canais na saída: um PNG em escala de cinza é expandido para RGB e um PNG com transparência (RGBA) tem o canal alfa descartado. Isso garante um shape previsível independentemente de como o arquivo foi salvo.

Trecho de `04_carregue_uma_imagem_real.py`:

```python
caminho = "assets/carro_real.png"
bytes_png = tf.io.read_file(caminho)
imagem = tf.io.decode_png(bytes_png, channels=3)

print("shape:", imagem.shape)
print("dtype:", imagem.dtype)
print("primeiro pixel:", imagem[0, 0].numpy())
```

Saída esperada: shape `(H, W, 3)` com as dimensões reais do PNG, dtype `uint8` e o vetor RGB do pixel `[0, 0]`.

Onde aparece: `04_carregue_uma_imagem_real.py` e, como primeiro passo, em `05`, `06`, `07`, `08`, `09`, `10` e `12`.

### 3.5 Inspeção de faixa de valores com reduções

Antes de alimentar um modelo é essencial saber em que faixa os pixels estão, porque uma rede treinada com entradas em [0, 1] se comporta mal se receber valores em [0, 255]. As reduções `tf.reduce_min(tensor)` e `tf.reduce_max(tensor)` percorrem todos os elementos e devolvem o menor e o maior valor como tensores escalares. Sem o argumento `axis`, elas reduzem todos os eixos de uma vez.

Trecho de `05_inspecione_shape_dtype_e_faixa.py`:

```python
bytes_png = tf.io.read_file("assets/carro_real.png")
imagem = tf.io.decode_png(bytes_png, channels=3)

print("shape:", imagem.shape)
print("rank:", tf.rank(imagem).numpy())
print("dtype:", imagem.dtype)
print("mínimo:", tf.reduce_min(imagem).numpy())
print("máximo:", tf.reduce_max(imagem).numpy())
```

Saída esperada: rank 3, dtype `uint8`, e mínimo e máximo dentro de 0 a 255. Se a imagem tiver preto puro e branco puro, os valores serão exatamente 0 e 255.

Onde aparece: `05_inspecione_shape_dtype_e_faixa.py`, `09_normalize_os_pixels_para_0_1.py`.

### 3.6 Dimensão de lote (batch)

Modelos de redes neurais não processam uma imagem por vez, mas um lote de N imagens. Por isso a entrada esperada por camadas como `Conv2D` tem rank 4, com shape `(N, H, W, C)`. Para passar uma única imagem por um modelo é preciso acrescentar um eixo de tamanho 1 na frente, transformando `(H, W, C)` em `(1, H, W, C)`.

A função `tf.expand_dims(tensor, axis=0)` insere um novo eixo de tamanho 1 na posição indicada. Com `axis=0` o eixo vai para a frente, que é a posição do lote. A operação não copia dados, só muda a interpretação do shape.

Trecho de `06_acrescente_a_dimensao_de_lote.py`:

```python
bytes_png = tf.io.read_file("assets/carro_real.png")
imagem = tf.io.decode_png(bytes_png, channels=3)
lote = tf.expand_dims(imagem, axis=0)

print("imagem:", imagem.shape)
print("lote:", lote.shape)
print("N:", lote.shape[0])
```

Saída esperada: `imagem: (H, W, 3)`, `lote: (1, H, W, 3)` e `N: 1`. O acesso `lote.shape[0]` lê o tamanho do primeiro eixo, que é o número de imagens no lote.

Onde aparece: `06_acrescente_a_dimensao_de_lote.py`. O script 11 já cria o tensor direto com rank 4, e o script 12 obtém o lote pelo método `.batch()` do `tf.data`.

### 3.7 Conversão RGB para escala de cinza

Converter para escala de cinza reduz os três canais a um único canal de luminância. O TensorFlow faz isso com `tf.image.rgb_to_grayscale(rgb)`, que aplica a combinação ponderada padrão dos canais (pesos maiores para o verde, ao qual o olho humano é mais sensível) e devolve um tensor de shape `(H, W, 1)`. Repare que o eixo dos canais não desaparece: ele passa a ter tamanho 1. Isso mantém o rank 3 e a compatibilidade com o restante das operações de imagem. O dtype de saída acompanha o de entrada, então uma imagem `uint8` produz cinza `uint8`.

Trecho de `07_converta_rgb_para_escala_de_cinza.py`:

```python
bytes_png = tf.io.read_file("assets/carro_real.png")
rgb = tf.io.decode_png(bytes_png, channels=3)
cinza = tf.image.rgb_to_grayscale(rgb)

print("RGB:", rgb.shape)
print("cinza:", cinza.shape)
print("dtype:", cinza.dtype)
```

Saída esperada: `RGB: (H, W, 3)`, `cinza: (H, W, 1)`, dtype `uint8`.

Onde aparece: `07_converta_rgb_para_escala_de_cinza.py`.

### 3.8 Separação e recombinação de canais

Muitas vezes é preciso tratar cada canal de cor separadamente, por exemplo para equalizar ou para visualizar um canal isolado. `tf.split(tensor, 3, axis=-1)` divide o tensor em 3 pedaços iguais ao longo do último eixo, devolvendo uma lista de três tensores de shape `(H, W, 1)`. A operação inversa é `tf.concat(lista, axis=-1)`, já vista no script 03, que junta os pedaços de volta em `(H, W, 3)`.

Para provar que a ida e a volta não alteram nada, o script compara as duas imagens elemento a elemento. A expressão `rgb == reconstruida` produz um tensor booleano do mesmo shape, e `tf.reduce_all(...)` devolve `True` só se todos os elementos forem `True`.

Trecho de `08_separe_e_recombine_os_canais.py`:

```python
bytes_png = tf.io.read_file("assets/carro_real.png")
rgb = tf.io.decode_png(bytes_png, channels=3)

r, g, b = tf.split(rgb, 3, axis=-1)
reconstruida = tf.concat([r, g, b], axis=-1)

print(r.shape, g.shape, b.shape)
print("igual:", tf.reduce_all(rgb == reconstruida).numpy())
```

Saída esperada: três shapes `(H, W, 1)` e `igual: True`.

Onde aparece: `08_separe_e_recombine_os_canais.py`. O `tf.concat` aparece também em `03_crie_uma_imagem_sintetica.py`.

### 3.9 Normalização para o intervalo [0, 1]

Redes neurais treinam melhor com entradas em uma faixa pequena e centrada, e a convenção mais simples é dividir os pixels por 255 para levá-los de [0, 255] para [0, 1]. Antes de dividir é obrigatório converter o dtype para ponto flutuante, porque dividir um tensor `uint8` produziria erro ou truncamento. `tf.cast(imagem, tf.float32)` faz a conversão, e a divisão por `255.0` é aplicada elemento a elemento pelo broadcasting do TensorFlow.

Trecho de `09_normalize_os_pixels_para_0_1.py`:

```python
bytes_png = tf.io.read_file("assets/carro_real.png")
imagem = tf.io.decode_png(bytes_png, channels=3)
normalizada = tf.cast(imagem, tf.float32) / 255.0

print("antes:", imagem.dtype,
      tf.reduce_min(imagem).numpy(),
      tf.reduce_max(imagem).numpy())
print("depois:", normalizada.dtype,
      tf.reduce_min(normalizada).numpy(),
      tf.reduce_max(normalizada).numpy())
```

Saída esperada: antes, `uint8` com valores entre 0 e 255; depois, `float32` com valores entre 0.0 e 1.0.

Onde aparece: `09_normalize_os_pixels_para_0_1.py` e, na forma `imagem / 255.0`, em `12_monte_um_pipeline_tf_data.py`.

### 3.10 Redimensionamento

Modelos pré-treinados esperam um tamanho fixo de entrada. O valor 224 por 224 é o padrão de arquiteturas clássicas como VGG, ResNet e MobileNet, e por isso a aula redimensiona para `(224, 224)`. A função `tf.image.resize(imagem, (altura, largura))` interpola os pixels para o novo tamanho. Dois detalhes importantes: ela aceita tensores de rank 3 `(H, W, C)` ou rank 4 `(N, H, W, C)`, e ela sempre devolve `float32`, mesmo que a entrada seja `uint8`. Isso acontece porque a interpolação bilinear padrão produz valores fracionários. Os valores continuam na faixa 0 a 255, só o dtype muda.

Trecho de `10_redimensione_a_imagem.py`:

```python
bytes_png = tf.io.read_file("assets/carro_real.png")
imagem = tf.io.decode_png(bytes_png, channels=3)
redimensionada = tf.image.resize(imagem, (224, 224))

print("antes:", imagem.shape, imagem.dtype)
print("depois:", redimensionada.shape,
      redimensionada.dtype)
```

Saída esperada: antes `(H, W, 3)` `uint8`, depois `(224, 224, 3)` `float32`.

Onde aparece: `10_redimensione_a_imagem.py`, `12_monte_um_pipeline_tf_data.py`.

### 3.11 Convolução 2D como camada Keras

A convolução é a operação central das redes convolucionais. Uma camada `Conv2D` desliza um conjunto de filtros (kernels) pequenos sobre a imagem e produz, para cada filtro, um mapa de ativação. O script 11 não treina nada: ele cria a camada, passa um tensor aleatório e mostra como os shapes se relacionam.

Parâmetros usados em `tf.keras.layers.Conv2D`:

- `filters=8`: número de filtros, que vira o número de canais da saída.
- `kernel_size=3`: cada filtro tem 3 por 3 pixels em cada canal de entrada.
- `padding="same"`: preenche as bordas com zeros para que altura e largura da saída sejam iguais às da entrada. A alternativa `"valid"` não preenche e encolhe a saída.
- `activation="relu"`: aplica a função ReLU, que zera valores negativos, logo após a convolução.

A entrada é criada com `tf.random.uniform((1, 32, 32, 3))`, um lote de uma imagem 32 por 32 com 3 canais e valores aleatórios em [0, 1). A camada é chamada como uma função, `conv(x)`, e só nesse momento os pesos são criados, com base no número de canais da entrada. O atributo `conv.kernel` guarda o tensor de pesos, cujo shape é `(altura_kernel, largura_kernel, canais_entrada, filtros)`.

Trecho de `11_aplique_uma_convolucao.py`:

```python
x = tf.random.uniform((1, 32, 32, 3))

conv = tf.keras.layers.Conv2D(
    filters=8, kernel_size=3,
    padding="same", activation="relu"
)
y = conv(x)

print("entrada:", x.shape)
print("saída:", y.shape)
print("kernel:", conv.kernel.shape)
```

Saída esperada: entrada `(1, 32, 32, 3)`, saída `(1, 32, 32, 8)` e kernel `(3, 3, 3, 8)`. A altura e a largura se mantêm por causa do `padding="same"`, e o número de canais passa de 3 para 8 por causa de `filters=8`.

Onde aparece: `11_aplique_uma_convolucao.py`.

### 3.12 Pipeline de dados com tf.data

`tf.data` é a API do TensorFlow para montar pipelines de entrada eficientes, que leem, transformam e agrupam dados sob demanda. O script 12 reúne tudo o que a aula ensinou em um único pipeline:

- `tf.data.Dataset.from_tensor_slices(lista)`: cria um dataset em que cada elemento é um item da lista. Aqui a lista tem um único caminho de arquivo.
- `.map(funcao)`: aplica a função a cada elemento. A função `carregar` recebe o caminho como tensor de string, lê, decodifica, redimensiona para 224 por 224 e normaliza dividindo por 255. Como `tf.image.resize` já devolve `float32`, a divisão funciona sem `tf.cast`.
- `.batch(1)`: agrupa os elementos em lotes de 1, acrescentando o eixo N na frente. É o equivalente, dentro do pipeline, ao `tf.expand_dims` do script 06.
- `.take(1)`: pega apenas o primeiro lote, útil para inspecionar sem percorrer tudo.

Trecho de `12_monte_um_pipeline_tf_data.py`:

```python
arquivos = tf.data.Dataset.from_tensor_slices(
    ["assets/carro_real.png"]
)

def carregar(caminho):
    dados = tf.io.read_file(caminho)
    imagem = tf.io.decode_png(dados, channels=3)
    imagem = tf.image.resize(imagem, (224, 224))
    return imagem / 255.0

dataset = arquivos.map(carregar).batch(1)
for lote in dataset.take(1):
    print(lote.shape, lote.dtype)
```

Saída esperada: `(1, 224, 224, 3) <dtype: 'float32'>`. Esse é exatamente o formato que um modelo Keras de classificação de imagens espera receber.

Onde aparece: `12_monte_um_pipeline_tf_data.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `01_compare_escalar_vetor_e_matriz.py` | Cria escalar, vetor e matriz com `tf.constant` e imprime shape, rank e dtype de cada um | Nenhuma | Texto no terminal | Nada |
| `02_represente_um_pixel_rgb.py` | Representa um pixel RGB como vetor `uint8` de 3 posições e acessa cada canal por índice | Nenhuma | Texto no terminal | Nada |
| `03_crie_uma_imagem_sintetica.py` | Monta uma imagem 4x6 vermelha com `tf.fill`, `tf.zeros`, `tf.concat` e `tf.cast` | Nenhuma | Texto no terminal | Nada |
| `04_carregue_uma_imagem_real.py` | Lê um PNG com `tf.io.read_file` e `tf.io.decode_png` e imprime shape, dtype e primeiro pixel | `assets/carro_real.png` | Texto no terminal | Arquivo `assets/carro_real.png` |
| `05_inspecione_shape_dtype_e_faixa.py` | Imprime shape, rank, dtype, mínimo e máximo da imagem | `assets/carro_real.png` | Texto no terminal | Arquivo `assets/carro_real.png` |
| `06_acrescente_a_dimensao_de_lote.py` | Usa `tf.expand_dims` para transformar `(H, W, 3)` em `(1, H, W, 3)` | `assets/carro_real.png` | Texto no terminal | Arquivo `assets/carro_real.png` |
| `07_converta_rgb_para_escala_de_cinza.py` | Converte com `tf.image.rgb_to_grayscale` e compara shapes | `assets/carro_real.png` | Texto no terminal | Arquivo `assets/carro_real.png` |
| `08_separe_e_recombine_os_canais.py` | Separa com `tf.split`, recombina com `tf.concat` e verifica igualdade com `tf.reduce_all` | `assets/carro_real.png` | Texto no terminal | Arquivo `assets/carro_real.png` |
| `09_normalize_os_pixels_para_0_1.py` | Converte para `float32` e divide por 255, mostrando faixa antes e depois | `assets/carro_real.png` | Texto no terminal | Arquivo `assets/carro_real.png` |
| `10_redimensione_a_imagem.py` | Redimensiona para 224x224 com `tf.image.resize` e mostra a mudança de dtype | `assets/carro_real.png` | Texto no terminal | Arquivo `assets/carro_real.png` |
| `11_aplique_uma_convolucao.py` | Cria uma `Conv2D` com 8 filtros 3x3 e mostra os shapes de entrada, saída e kernel | Tensor aleatório `(1, 32, 32, 3)` | Texto no terminal | Nada |
| `12_monte_um_pipeline_tf_data.py` | Monta um pipeline `tf.data` que lê, decodifica, redimensiona, normaliza e agrupa em lote | `assets/carro_real.png` | Texto no terminal | Arquivo `assets/carro_real.png` |

Nenhum script grava arquivo em disco. Toda a saída é impressa no terminal.

## 5. Passo a passo para executar

Passo 0. Prepare o ambiente e o material.

```powershell
cd C:\visao_computacional_PVA\aula_04_Tensor\exemplos_em_python
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Crie a pasta `assets` dentro de `exemplos_em_python` e coloque nela um PNG colorido chamado `carro_real.png`. Sem esse arquivo os scripts 04 a 10 e 12 falham com `NotFoundError`.

Passo 1. Scripts sem dependência de arquivo. Rode em qualquer ordem; eles mostram os fundamentos.

```powershell
python 01_compare_escalar_vetor_e_matriz.py
python 02_represente_um_pixel_rgb.py
python 03_crie_uma_imagem_sintetica.py
```

Espere ver no terminal: no 01, três linhas com shape `()`, `(3,)` e `(2, 2)` e ranks 0, 1 e 2; no 02, o vetor `[230 80 35]` e os canais separados; no 03, shape `(4, 6, 3)` e pixel `[255 0 0]`.

Passo 2. Carregamento e inspeção da imagem real.

```powershell
python 04_carregue_uma_imagem_real.py
python 05_inspecione_shape_dtype_e_faixa.py
python 06_acrescente_a_dimensao_de_lote.py
```

Espere ver o shape real do seu PNG com 3 canais, dtype `uint8`, mínimo e máximo entre 0 e 255, e o lote com shape `(1, H, W, 3)`.

Passo 3. Transformações de canal e de valor.

```powershell
python 07_converta_rgb_para_escala_de_cinza.py
python 08_separe_e_recombine_os_canais.py
python 09_normalize_os_pixels_para_0_1.py
python 10_redimensione_a_imagem.py
```

Espere ver: cinza com shape `(H, W, 1)`; três canais `(H, W, 1)` e `igual: True`; faixa `float32` de 0.0 a 1.0; imagem `(224, 224, 3)` em `float32`.

Passo 4. Convolução e pipeline.

```powershell
python 11_aplique_uma_convolucao.py
python 12_monte_um_pipeline_tf_data.py
```

Espere ver: no 11, saída `(1, 32, 32, 8)` e kernel `(3, 3, 3, 8)`; no 12, um lote `(1, 224, 224, 3)` em `float32`.

Observações sobre a execução:

- Nenhum script recebe argumentos de linha de comando nem usa teclas de atalho. Cada um roda até o fim e encerra sozinho.
- Na primeira importação do TensorFlow é normal aparecerem mensagens de aviso sobre CPU, oneDNN ou ausência de GPU. Elas não são erros.
- Rode sempre de dentro de `exemplos_em_python/`, senão o caminho relativo `assets/carro_real.png` não é encontrado.

## 6. Exercícios e itens da aula

Os scripts desta aula não referenciam itens ou exercícios numerados. Exercícios propostos para praticar os conceitos:

1. Tensor de rank 4 à mão. Estenda o script 03 para gerar uma imagem sintética com um gradiente (por exemplo, usando `tf.range` e `tf.reshape`) e depois acrescente o eixo de lote com `tf.expand_dims`. Confira o shape `(1, 4, 6, 3)`.
2. Normalização centrada. Modifique o script 09 para levar os pixels ao intervalo [-1, 1], usando `(x / 127.5) - 1.0`, que é a normalização esperada pela MobileNetV2. Imprima mínimo e máximo para confirmar.
3. Padding valid versus same. No script 11, troque `padding="same"` por `padding="valid"` e explique por que a saída passa a ter shape `(1, 30, 30, 8)`. Depois troque `kernel_size` para 5 e preveja o shape antes de rodar.
4. Pipeline com várias imagens. Coloque três PNGs na pasta `assets` e altere o script 12 para listar os três caminhos em `from_tensor_slices`, usar `.batch(3)` e imprimir o shape do lote. Em seguida acrescente `.shuffle(3)` antes do `.batch` e observe que a ordem muda.
5. Cinza por conta própria. Reproduza o resultado de `tf.image.rgb_to_grayscale` usando `tf.split` e uma combinação ponderada dos canais (0.299 R + 0.587 G + 0.114 B), e compare com `tf.reduce_max(tf.abs(diferenca))` para ver o quanto seu resultado se aproxima do oficial.

## 7. Armadilhas e dicas

- Arquivo ausente. `assets/carro_real.png` não veio no zip do professor. Sem ele, oito dos doze scripts falham. Qualquer PNG colorido serve.
- Caminho relativo. Todos os scripts usam `assets/carro_real.png` sem `Path(__file__)`. O diretório de trabalho precisa ser `exemplos_em_python/`. Se rodar de outra pasta, o TensorFlow lança `NotFoundError`.
- Versão do Python. O professor testou com Python 3.12.2 e TensorFlow 2.17.0. Se não houver wheel de TensorFlow para o Python 3.14 da máquina, crie o venv com `py -3.12`.
- `tf.image.resize` muda o dtype. A entrada `uint8` sai como `float32`, mas os valores continuam entre 0 e 255. Quem esquece isso e compara com uma imagem normalizada em [0, 1] tem resultados errados. No script 12 a divisão por 255 vem depois do resize justamente por isso.
- Divisão de `uint8`. No script 09, o `tf.cast` para `float32` antes de dividir não é opcional. Dividir um tensor inteiro por um float no TensorFlow gera erro de tipos incompatíveis.
- `tf.fill` infere `int32`. No script 03, `tf.fill((4, 6, 1), 255)` nasce `int32`, e por isso os `tf.zeros` são criados explicitamente com `dtype=tf.int32`. Se um dos três tivesse dtype diferente, `tf.concat` falharia. O `tf.cast` final para `uint8` corrige o dtype da imagem inteira.
- `rgb_to_grayscale` mantém o rank. A saída tem shape `(H, W, 1)`, não `(H, W)`. Para visualizar com outras bibliotecas pode ser necessário `tf.squeeze`.
- Pesos da `Conv2D` só existem depois da primeira chamada. No script 11, `conv.kernel` só pode ser lido após `conv(x)`, porque é nesse momento que o Keras descobre o número de canais de entrada e cria os pesos. Acessar antes gera `AttributeError`.
- Valores aleatórios. `tf.random.uniform` no script 11 gera dados diferentes a cada execução, mas os shapes impressos são sempre os mesmos, e é isso que o script quer mostrar.
- Avisos do TensorFlow. Mensagens sobre oneDNN, CPU instructions ou CUDA ausente na importação são informativas. No Windows nativo não há suporte a GPU, e isso não afeta a aula.
- Parâmetros que valem a pena variar: `channels` em `decode_png` (1 para cinza, 4 para RGBA), o tamanho em `tf.image.resize`, e `filters`, `kernel_size` e `padding` em `Conv2D`. Observe como cada um altera o shape de saída.

## 8. Glossário

| Termo | Significado |
|---|---|
| Tensor | Arranjo numérico de N dimensões, estrutura de dados básica do TensorFlow |
| Rank | Número de eixos (dimensões) de um tensor: 0 para escalar, 1 para vetor, 2 para matriz, 3 para imagem |
| Shape | Tupla com o tamanho de cada eixo do tensor, por exemplo `(224, 224, 3)` |
| dtype | Tipo numérico dos elementos do tensor, como `int32`, `uint8` ou `float32` |
| uint8 | Inteiro sem sinal de 8 bits, de 0 a 255, formato padrão de pixels de imagem |
| float32 | Número de ponto flutuante de 32 bits, formato padrão para cálculos em redes neurais |
| Escalar | Tensor de rank 0, um único número |
| Vetor | Tensor de rank 1, uma sequência de números |
| Matriz | Tensor de rank 2, uma grade de números com linhas e colunas |
| RGB | Espaço de cor com três canais: vermelho (Red), verde (Green) e azul (Blue) |
| Canal | Um dos componentes de cor de um pixel; o último eixo de uma imagem no layout `(H, W, C)` |
| Channels last | Convenção em que o eixo dos canais é o último do tensor, padrão do TensorFlow e do Keras |
| `(H, W, C)` | Shape de uma imagem: altura (Height), largura (Width) e canais (Channels) |
| `(N, H, W, C)` | Shape de um lote de N imagens, formato de entrada das camadas convolucionais |
| Lote (batch) | Conjunto de N amostras processadas juntas por um modelo |
| `axis=-1` | Referência ao último eixo do tensor, que em imagens é o eixo dos canais |
| Broadcasting | Regra que permite operar um tensor com um escalar ou com um tensor de shape compatível, aplicando a operação elemento a elemento |
| Redução | Operação que combina todos os elementos ao longo de um ou mais eixos, como `reduce_min`, `reduce_max` e `reduce_all` |
| Normalização | Transformação dos valores de pixel para uma faixa pequena, aqui de [0, 255] para [0, 1] |
| Escala de cinza | Imagem com um único canal de luminância, obtida de RGB por combinação ponderada dos canais |
| Redimensionamento (resize) | Alteração da altura e largura da imagem por interpolação |
| Interpolação bilinear | Método padrão de `tf.image.resize`, que estima cada novo pixel a partir dos quatro vizinhos mais próximos |
| Convolução | Operação que desliza um kernel sobre a imagem e calcula somas ponderadas locais |
| Kernel (filtro) | Pequena matriz de pesos usada pela convolução; em `Conv2D` tem shape `(kh, kw, canais_entrada, filtros)` |
| `Conv2D` | Camada Keras de convolução bidimensional |
| Padding | Preenchimento das bordas antes da convolução; `same` mantém o tamanho, `valid` não preenche e encolhe a saída |
| ReLU | Rectified Linear Unit, função de ativação que substitui valores negativos por zero |
| Mapa de ativação | Saída de um filtro convolucional aplicado à imagem; um `Conv2D` com 8 filtros produz 8 mapas |
| Keras | API de alto nível do TensorFlow para construir e treinar modelos |
| `tf.data` | API do TensorFlow para montar pipelines de leitura e transformação de dados |
| Dataset | Objeto do `tf.data` que representa uma sequência de elementos processados sob demanda |
| `map` | Método do Dataset que aplica uma função a cada elemento |
| `batch` | Método do Dataset que agrupa elementos em lotes, acrescentando o eixo N |
| `take` | Método do Dataset que limita o número de elementos ou lotes percorridos |
| Pipeline | Sequência encadeada de etapas de leitura, decodificação, transformação e agrupamento de dados |
| PNG | Portable Network Graphics, formato de imagem sem perdas usado nos scripts |
| RGBA | RGB com um quarto canal alfa de transparência; `channels=3` descarta esse canal |
| venv | Ambiente virtual do Python, usado para isolar as bibliotecas do projeto |
| WSL2 | Windows Subsystem for Linux, necessário para usar GPU com TensorFlow no Windows |
| CUDA | Plataforma da NVIDIA para computação em GPU |

