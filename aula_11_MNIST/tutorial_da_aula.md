# Aula 11 – MNIST

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula usa o MNIST, o conjunto clássico de 70 mil imagens de dígitos manuscritos de 28x28 pixels, para ensinar o ciclo completo de classificação de imagens com redes neurais em TensorFlow/Keras. O fio condutor é uma comparação: um MLP (rede totalmente conectada, que recebe a imagem achatada em um vetor de 784 números) contra uma CNN (rede convolucional, que preserva a estrutura 2D da imagem). Ao longo dos scripts o aluno aprende a carregar e preparar os dados, montar as duas arquiteturas, treinar, plotar curvas de treino e validação, diagnosticar overfitting, aplicar Dropout e EarlyStopping, ler matrizes de confusão, salvar e recarregar modelos e olhar para dentro da rede (filtros e mapas de ativação).

A segunda metade da aula sai do laboratório e vai para o mundo real: fotos de dígitos escritos à mão são pré-processadas com OpenCV (cinza, Otsu, recorte, resize 28x28, normalização) e passadas pela CNN. Isso expõe o chamado domain gap, a diferença entre a distribuição do treino (MNIST limpo) e a da aplicação (fotos com sombra, fundo, rotação e ruído). A resposta do professor é data augmentation com rotação e zoom, seguida de uma comparação de acurácia antes e depois. O script 30 reúne tudo em um pipeline de Item A e Item B, e o 31 fecha com inferência ao vivo pela webcam.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas nos scripts (levantadas pelos imports):

| Biblioteca | Pacote pip | Uso na aula |
|---|---|---|
| `tensorflow` (com `tf.keras`) | `tensorflow` | dataset MNIST, camadas, treino, callbacks, `ImageDataGenerator` |
| `numpy` | `numpy` | reshape, argmax, máscaras, `np.newaxis` |
| `matplotlib` | `matplotlib` | grade de amostras, curvas, filtros, erros |
| `sklearn` | `scikit-learn` | `confusion_matrix` e `ConfusionMatrixDisplay` |
| `pandas` | `pandas` | tabela comparativa MLP vs CNN em CSV |
| `cv2` | `opencv-python` | pré-processamento de fotos reais e webcam |

Instalação sugerida, de preferência em um ambiente virtual:

```
pip install tensorflow numpy matplotlib scikit-learn pandas opencv-python
```

Pontos de atenção:

- **Pasta de execução.** Todos os caminhos são relativos ao diretório de trabalho (`resultados/`, `data/real_digits/`). Rode sempre de dentro de `aula_11_MNIST/exemplos_em_python/`. A pasta `resultados/` é criada pelos próprios scripts.
- **Download do MNIST.** Na primeira execução, `tf.keras.datasets.mnist.load_data()` baixa cerca de 11 MB para o cache do Keras (`~/.keras/datasets/`). Precisa de internet só nessa primeira vez.
- **Material que falta.** O zip não trouxe `data/real_digits/` com as fotos `0.jpg` a `9.jpg`. Elas são usadas pelos scripts 22, 23, 24, 25, 28 e 30 (Item B). Segundo o `CLAUDE.md`, essas fotos podem ser feitas por nós: escreva cada dígito em papel, fotografe e salve em `exemplos_em_python/data/real_digits/` com o nome igual ao rótulo. A função `list_real_digit_images` aceita as extensões `jpg`, `jpeg`, `png` e `bmp`.
- **Webcam.** Só o script 31 usa a câmera (`cv2.VideoCapture(0)`).
- **Versão de Python.** O professor rodou as aulas vizinhas com Python 3.12 e 3.13. A máquina local tem 3.14.3; confirme que existe wheel de TensorFlow para essa versão antes de instalar, ou crie um venv com 3.12.
- **`ImageDataGenerator`.** Os scripts 26, 27, 28 e 30 importam `tensorflow.keras.preprocessing.image.ImageDataGenerator`, API marcada como legada no Keras 3. Ela ainda funciona em TensorFlow 2.16+, mas emite aviso de depreciação.
- **Tempo de treino.** Tudo roda em CPU. Cada época da CNN sobre 54 mil imagens leva de 10 a 40 segundos numa CPU comum. Os scripts 12 e 30 treinam vários modelos por 5 épocas; reserve alguns minutos.

## 3. Conceitos-chave

### Bloco 1: dados e pré-processamento

### 3.1 O dataset MNIST e sua forma

MNIST é um conjunto de 60.000 imagens de treino e 10.000 de teste, cada uma com 28 linhas por 28 colunas de intensidades inteiras entre 0 e 255 (`uint8`), e um rótulo inteiro de 0 a 9. O Keras entrega tudo pronto como arrays NumPy pela função `tf.keras.datasets.mnist.load_data()`, que devolve duas tuplas: `(x_train, y_train)` e `(x_test, y_test)`. O primeiro contato com qualquer dataset deve ser inspecionar `shape`, `min`, `max` e alguns rótulos, exatamente como faz o script 01.

```python
# 01_carregar_mnist.py
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Formato x_train:", x_train.shape)   # (60000, 28, 28)
print("Formato y_train:", y_train.shape)   # (60000,)
print("Valor mínimo de pixel:", x_train.min())   # 0
print("Valor máximo de pixel:", x_train.max())   # 255
print("Primeiro rótulo:", y_train[0])
```

Onde aparece: `01_carregar_mnist.py`, `02_visualizar_amostras.py`, `03_normalizacao_e_formas.py`, `04_rotulos_e_one_hot.py`, e indiretamente em todos os outros via `mnist_utils.load_mnist`.

### 3.2 Visualizar amostras com Matplotlib

Antes de treinar, vale olhar os dados. O script 02 monta uma grade de 4x5 com `plt.subplot`, exibe cada imagem com `plt.imshow(..., cmap="gray")` e coloca o rótulo no título. `cmap="gray"` é necessário porque a imagem tem um só canal; sem ele o Matplotlib aplica o mapa de cores padrão (viridis). `plt.axis("off")` esconde os eixos e `plt.tight_layout()` ajusta os espaçamentos.

```python
# 02_visualizar_amostras.py
plt.figure(figsize=(8, 4))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"rótulo: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
```

Onde aparece: `02_visualizar_amostras.py`, e o mesmo padrão em `14_visualizar_filtros_cnn.py`, `15_visualizar_feature_maps.py`, `20_erros_de_classificacao.py`, `22_opencv_preprocessar_digito_real.py`, `24_inferir_pasta_digitos_reais.py`, `26_visualizar_data_augmentation.py`.

### 3.3 Normalização para [0, 1]

Redes neurais treinam melhor quando as entradas ficam em uma faixa pequena e centrada. O professor converte os pixels para `float32` e divide por 255, levando a faixa de 0..255 para 0..1. A conversão de tipo é obrigatória: dividir `uint8` por 255 diretamente produziria zeros ou resultado inesperado. Essa normalização é a mesma que será aplicada às fotos reais no Item B, e é importante que treino e inferência usem exatamente a mesma escala.

```python
# 03_normalizacao_e_formas.py
x_train_norm = x_train.astype("float32") / 255.0
print("Faixa normalizada:", x_train_norm.min(), x_train_norm.max())   # 0.0 1.0
```

Onde aparece: `03_normalizacao_e_formas.py`, `mnist_utils.py` (`load_mnist` e `preprocess_digit_image`), `31_webcam_digito_tempo_real.py`.

### 3.4 Formas de entrada: vetor para MLP, tensor 3D para CNN

O mesmo dado precisa de duas formas diferentes conforme a arquitetura. O MLP recebe cada imagem achatada em um vetor de 784 posições (28 x 28), feito com `reshape((-1, 784))`, onde `-1` deixa o NumPy calcular o número de amostras. A CNN recebe a imagem com uma dimensão de canal explícita, `(28, 28, 1)`, acrescentada com `x[..., np.newaxis]`. As reticências `...` significam "todas as dimensões anteriores". A função `load_mnist(flatten=...)` em `mnist_utils.py` centraliza essa escolha para todos os scripts seguintes.

```python
# 03_normalizacao_e_formas.py
x_mlp = x_train_norm.reshape((-1, 28 * 28))    # (60000, 784)
x_cnn = x_train_norm[..., np.newaxis]          # (60000, 28, 28, 1)

# mnist_utils.py
def load_mnist(flatten: bool = False, one_hot: bool = False):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    if flatten:
        x_train = x_train.reshape((-1, 28 * 28))
        x_test = x_test.reshape((-1, 28 * 28))
    else:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]
```

Onde aparece: `03_normalizacao_e_formas.py`, `mnist_utils.py`, e todo script que chama `load_mnist(flatten=True)` (MLP) ou `load_mnist(flatten=False)` (CNN).

### 3.5 Rótulos inteiros e one-hot encoding

Há duas maneiras de representar a classe de uma amostra. A forma inteira guarda só o número (por exemplo `5`). A forma one-hot é um vetor de 10 posições com `1` na posição da classe e `0` nas demais. `tf.keras.utils.to_categorical(y, num_classes=10)` faz a conversão. A escolha determina a função de perda: `categorical_crossentropy` exige one-hot, enquanto `sparse_categorical_crossentropy` aceita inteiros. O professor usa a versão sparse em toda a aula, então os rótulos ficam como inteiros, o que economiza memória e código. A função `load_mnist` tem o parâmetro `one_hot`, mas nenhum script o ativa.

```python
# 04_rotulos_e_one_hot.py
y_one_hot = tf.keras.utils.to_categorical(y_train[:5], num_classes=10)
print("Rótulos inteiros:", y_train[:5])       # [5 0 4 1 9]
print("Rótulos one-hot:")
print(y_one_hot)                               # 5 linhas x 10 colunas com um único 1 por linha
# Neste pacote usamos loss='sparse_categorical_crossentropy',
# portanto os rótulos podem ficar como inteiros de 0 a 9.
```

Onde aparece: `04_rotulos_e_one_hot.py`, `mnist_utils.py`.

### Bloco 2: MLP (rede totalmente conectada)

### 3.6 Modelo Sequential, camada Input e camada Dense

`tf.keras.Sequential` monta uma rede como uma lista de camadas em que a saída de uma é a entrada da próxima. `Input(shape=...)` declara a forma de uma amostra (sem a dimensão de lote). `Dense(n, activation=...)` é a camada totalmente conectada: cada um dos `n` neurônios recebe todas as entradas, calcula uma soma ponderada mais um viés e aplica a ativação. Com `relu` (Rectified Linear Unit, que zera valores negativos) nas camadas ocultas e `softmax` na saída (transforma 10 números em probabilidades que somam 1), temos o classificador básico. O MLP mínimo do script 05 tem uma única camada oculta de 64 neurônios.

```python
# 05_mlp_minimo.py
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
```

Onde aparece: `05_mlp_minimo.py`, `06_mlp_duas_camadas_densas.py` (via `build_mlp`), `09_cnn_minima.py`, `mnist_utils.py`.

### 3.7 Compilar: otimizador, perda e métrica

`model.compile` fixa três coisas. O otimizador `adam` ajusta os pesos usando gradientes com taxa de aprendizado adaptativa, e é a escolha padrão para começar. A perda `sparse_categorical_crossentropy` mede o quanto a distribuição softmax se afasta do rótulo correto, aceitando rótulos inteiros. A métrica `accuracy` é a fração de acertos, mais fácil de interpretar do que a perda, mas não é usada para ajustar pesos. `model.summary()` imprime a lista de camadas, a forma de saída de cada uma e o número de parâmetros treináveis, que é o que o Item A pede para comparar.

```python
# 05_mlp_minimo.py
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
```

Onde aparece: todos os scripts que criam modelo, centralizado em `build_mlp` e `build_cnn` de `mnist_utils.py`.

### 3.8 Treinar com `fit`: épocas, batch e validation_split

`model.fit` roda o treino. `epochs` é o número de passagens completas pelos dados; `batch_size` é quantas amostras entram por passo de gradiente (128 na aula); `validation_split=0.1` separa automaticamente os últimos 10% do treino (6.000 imagens) para medir o desempenho a cada época em dados que a rede não vê no ajuste. O retorno é um objeto `History` cujo atributo `.history` é um dicionário com listas `loss`, `accuracy`, `val_loss` e `val_accuracy`, uma entrada por época. Depois, `model.evaluate(x_test, y_test)` mede a perda e a acurácia no conjunto de teste, que nunca participou de nada.

```python
# 05_mlp_minimo.py
model.fit(x_train, y_train, validation_split=0.1, epochs=2, batch_size=128)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia de teste do MLP mínimo: {acc:.4f}")
```

Onde aparece: todos os scripts de treino, do 05 ao 30.

### 3.9 MLP com duas camadas densas (exigência do Item A)

O Item A exige um MLP com pelo menos duas camadas densas antes da saída. A função `build_mlp` em `mnist_utils.py` atende com 256 e 128 neurônios ReLU, seguidos de softmax de 10. Isso dá cerca de 235 mil parâmetros (784x256 + 256 + 256x128 + 128 + 128x10 + 10). O ponto didático é que o MLP recebe o vetor de 784 posições e perde a vizinhança entre pixels: para ele, o pixel (10, 10) e o (10, 11) são atributos independentes.

```python
# mnist_utils.py
def build_mlp() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28 * 28,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="mlp_mnist",
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
```

Onde aparece: `mnist_utils.py`, `06_mlp_duas_camadas_densas.py`, `07_mlp_curvas_treino.py`, `08_mlp_matriz_confusao.py`, `12_comparar_mlp_cnn.py`, `30_pipeline_completo_item_a_b.py`.

### 3.10 Curvas de treino e validação

Plotar `accuracy` e `loss` de treino e de validação, época a época, é a ferramenta principal de diagnóstico. A função `plot_history` em `mnist_utils.py` desenha dois gráficos lado a lado (acurácia e perda), salva em PNG com `plt.savefig(output_path, dpi=150)` se um caminho for informado, criando a pasta com `mkdir(parents=True, exist_ok=True)`, e depois mostra na tela. A leitura esperada: as duas curvas sobem juntas em um treino saudável; se a de treino continua subindo e a de validação estagna ou cai, há overfitting (ver 3.18).

```python
# mnist_utils.py
def plot_history(history, title, output_path=None):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, hist["accuracy"], label="treino")
    plt.plot(epochs, hist["val_accuracy"], label="validação")
    ...
    plt.subplot(1, 2, 2)
    plt.plot(epochs, hist["loss"], label="treino")
    plt.plot(epochs, hist["val_loss"], label="validação")
    ...
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.show()
```

Onde aparece: `mnist_utils.py`, `07_mlp_curvas_treino.py`, `11_cnn_curvas_treino.py`, `12_comparar_mlp_cnn.py`, `16_overfitting_com_poucos_dados.py`, `17_regularizacao_dropout.py`, `18_early_stopping.py`, `27_treinar_cnn_com_augmentation.py`, `30_pipeline_completo_item_a_b.py`.

### 3.11 Predição e matriz de confusão

`model.predict(x)` devolve, para cada amostra, um vetor de 10 probabilidades. `.argmax(axis=1)` escolhe o índice da maior, que é a classe predita. A matriz de confusão cruza rótulo real (linhas) com predição (colunas): a diagonal são os acertos e as células fora dela mostram quais pares de dígitos a rede confunde (tipicamente 4 e 9, 3 e 5, 7 e 1). O scikit-learn calcula com `confusion_matrix(y_real, y_pred)` e desenha com `ConfusionMatrixDisplay(cm).plot(values_format="d")`, onde `"d"` imprime inteiros.

```python
# 08_mlp_matriz_confusao.py
pred = model.predict(x_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm).plot(values_format="d")
plt.title("Matriz de confusão - MLP")
plt.show()
```

Onde aparece: `08_mlp_matriz_confusao.py`, `19_matriz_confusao_cnn.py`.

### Bloco 3: CNN (rede convolucional)

### 3.12 Conv2D, MaxPooling2D e Flatten

Uma CNN substitui a primeira parte do MLP por operações que respeitam a geometria da imagem. `Conv2D(filtros, (3, 3), activation="relu")` desliza `filtros` kernels de 3x3 pela imagem, e cada kernel produz um mapa de características que responde a um padrão local (borda, traço, curva). Os pesos do kernel são os mesmos em toda a imagem (compartilhamento de pesos), então há poucos parâmetros: 16 filtros 3x3 em 1 canal dão 16x(9+1) = 160 parâmetros. `MaxPooling2D((2, 2))` reduz cada mapa pela metade em altura e largura ficando com o máximo de cada janela 2x2, o que dá tolerância a pequenos deslocamentos e diminui o custo. `Flatten` achata os mapas em um vetor para entrar nas camadas `Dense` finais. A CNN mínima do script 09 tem um bloco Conv + Pool.

```python
# 09_cnn_minima.py
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
```

Onde aparece: `09_cnn_minima.py`, `mnist_utils.py` (`build_cnn`).

### 3.13 CNN com dois blocos e `padding="same"` (exigência do Item A)

O Item A exige pelo menos dois blocos Conv2D + MaxPooling. `build_cnn` em `mnist_utils.py` usa 32 filtros no primeiro bloco e 64 no segundo, com `padding="same"`, que adiciona borda de zeros para que a saída da convolução tenha o mesmo tamanho da entrada (28x28 em vez de 26x26). Assim, a sequência de formas é 28x28x32, 14x14x32, 14x14x64, 7x7x64, Flatten com 3136 valores, Dense 128, Dense 10. O parâmetro opcional `dropout` insere uma camada `Dropout` antes da saída (ver 3.19). A justificativa que o professor escreve nos scripts 12 e 30: a CNN supera o MLP porque preserva a estrutura espacial 2D, aprende filtros locais úteis para bordas e traços e reutiliza pesos por toda a imagem.

```python
# mnist_utils.py
def build_cnn(dropout: float = 0.0) -> tf.keras.Model:
    layers = [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
    ]
    if dropout > 0:
        layers.append(tf.keras.layers.Dropout(dropout))
    layers.append(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    model = tf.keras.Sequential(layers, name="cnn_mnist")
```

Onde aparece: `mnist_utils.py` e todos os scripts de 10 a 31 que chamam `build_cnn`.

### 3.14 Callbacks: medir o tempo por época

Um callback é um objeto que o Keras chama em momentos do treino (início e fim de época, de lote, etc.). O professor escreve `EpochTimer`, subclasse de `tf.keras.callbacks.Callback`, que guarda `time.perf_counter()` em `on_epoch_begin` e acumula a diferença em `on_epoch_end`. O resultado, `epoch_times`, entra na tabela comparativa do Item A como tempo médio por época. O callback é passado em `model.fit(..., callbacks=[timer])`.

```python
# mnist_utils.py
class EpochTimer(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_times = []
    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_times.append(time.perf_counter() - self._start)

# 12_comparar_mlp_cnn.py
timer_mlp = EpochTimer()
h_mlp = mlp.fit(x_train_mlp, y_train, validation_split=0.1, epochs=EPOCHS,
                batch_size=BATCH, callbacks=[timer_mlp])
```

Onde aparece: `mnist_utils.py`, `12_comparar_mlp_cnn.py`, `30_pipeline_completo_item_a_b.py`. O `EarlyStopping` do script 18 é outro callback, pronto do Keras.

### 3.15 Tabela comparativa MLP vs CNN com pandas

O Item A pede para comparar número de parâmetros, tempo por época e acurácia de teste. `model.count_params()` dá o total de parâmetros. `model_summary_row` monta um dicionário por modelo, e `save_comparison_table` transforma a lista em `pandas.DataFrame`, imprime com `to_string(index=False)` e grava em `resultados/comparacao_modelos.csv` (ou no caminho informado). A leitura típica: a CNN tem menos parâmetros que o MLP (por volta de 420 mil contra 235 mil no caso da aula, então aqui a CNN tem mais, por causa da Dense de 3136x128), demora mais por época, mas alcança acurácia de teste maior (cerca de 99% contra 98%).

```python
# mnist_utils.py
def model_summary_row(name, model, epoch_times, test_acc):
    return {
        "modelo": name,
        "parametros": int(model.count_params()),
        "tempo_medio_epoca_s": float(np.mean(epoch_times)),
        "acuracia_teste": float(test_acc),
    }

def save_comparison_table(rows, path="resultados/comparacao_modelos.csv"):
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(path, index=False)
```

Onde aparece: `mnist_utils.py`, `12_comparar_mlp_cnn.py`, `30_pipeline_completo_item_a_b.py`.

### 3.16 Salvar e carregar o modelo (`.keras`)

Treinar custa tempo; guardar o modelo evita repetir. `model.save("caminho.keras")` grava arquitetura, pesos e estado do otimizador no formato nativo do Keras 3. `tf.keras.models.load_model(caminho)` reconstrói tudo. Os scripts do Item B (23, 24, 25, 28, 31) seguem o padrão "se `resultados/cnn_mnist.keras` existe, carrega; senão, treina 3 épocas e salva". O script 27 salva a versão com augmentation em `resultados/cnn_mnist_aug.keras`.

```python
# 13_salvar_e_carregar_cnn.py
model_path = "resultados/cnn_mnist.keras"
model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
model.save(model_path)

loaded = tf.keras.models.load_model(model_path)
loss, acc = loaded.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia do modelo carregado: {acc:.4f}")
```

Onde aparece: `13_salvar_e_carregar_cnn.py`, `23_inferir_digito_real_unico.py`, `24_inferir_pasta_digitos_reais.py`, `25_acuracia_imagens_reais.py`, `27_treinar_cnn_com_augmentation.py`, `28_comparar_real_antes_depois_aug.py`, `30_pipeline_completo_item_a_b.py`, `31_webcam_digito_tempo_real.py`.

### Bloco 4: visualização interna da CNN

### 3.17 Filtros aprendidos e mapas de ativação

Uma CNN não é uma caixa preta total. `model.layers[0].get_weights()` devolve `[weights, biases]` da primeira `Conv2D`; `weights` tem forma `(altura, largura, canais_entrada, filtros)`, ou seja `(3, 3, 1, 32)`. O script 14 pega `weights[:, :, 0, :]`, normaliza cada filtro para [0, 1] (com `+ 1e-8` para evitar divisão por zero) e mostra 16 deles como imagens 3x3: aparecem detectores de borda em várias orientações.

Para ver o que a camada produz, o script 15 cria um segundo modelo com `tf.keras.Model(inputs=model.inputs, outputs=model.layers[0].output)`, que compartilha os pesos do original mas para na primeira convolução. `predict` de uma amostra devolve 32 mapas 28x28 (`feature_maps[:, :, i]`), exibidos com `cmap="viridis"`. Cada mapa realça uma característica diferente do mesmo dígito.

```python
# 14_visualizar_filtros_cnn.py
weights, biases = model.layers[0].get_weights()
filters = weights[:, :, 0, :]          # (3, 3, 32)
f = (f - f.min()) / (f.max() - f.min() + 1e-8)

# 15_visualizar_feature_maps.py
activation_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[0].output)
feature_maps = activation_model.predict(x_train[0:1], verbose=0)[0]   # (28, 28, 32)
plt.imshow(feature_maps[:, :, i], cmap="viridis")
```

Onde aparece: `14_visualizar_filtros_cnn.py`, `15_visualizar_feature_maps.py`.

### Bloco 5: overfitting e regularização

### 3.18 Overfitting e como provocá-lo

Overfitting é quando a rede memoriza o treino em vez de aprender padrões gerais: a acurácia de treino sobe, a de validação estagna ou cai, e a perda de validação começa a subir enquanto a de treino continua caindo. Ele fica mais provável com poucos dados, muitas épocas ou modelo grande demais. O script 16 provoca isso de propósito: treina a CNN com só 1.000 imagens, `validation_split=0.3`, 12 épocas e `batch_size=64`. Nas curvas geradas, o treino chega perto de 100% e a validação fica bem abaixo.

```python
# 16_overfitting_com_poucos_dados.py
x_small = x_train[:1000]
y_small = y_train[:1000]
model = build_cnn()
history = model.fit(x_small, y_small, validation_split=0.3, epochs=12, batch_size=64)
plot_history(history, "CNN com poucos dados", "resultados/overfitting_poucos_dados.png")
```

Onde aparece: `16_overfitting_com_poucos_dados.py`, e a discussão em `11_cnn_curvas_treino.py`, `12_comparar_mlp_cnn.py`, `30_pipeline_completo_item_a_b.py`.

### 3.19 Dropout

`Dropout(p)` é uma camada que, só durante o treino, zera aleatoriamente uma fração `p` das ativações a cada passo. Isso impede que neurônios se especializem em memorizar combinações específicas e força redundância. Na inferência a camada não faz nada. O script 17 usa `build_cnn(dropout=0.4)`, que coloca o Dropout entre a Dense de 128 e a saída. O efeito esperado é aproximar as curvas de treino e validação, às custas de um treino um pouco mais lento em convergir.

```python
# 17_regularizacao_dropout.py
model = build_cnn(dropout=0.4)
history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
plot_history(history, "CNN com Dropout", "resultados/cnn_dropout_curvas.png")

# mnist_utils.py, dentro de build_cnn
if dropout > 0:
    layers.append(tf.keras.layers.Dropout(dropout))
```

Onde aparece: `17_regularizacao_dropout.py`, `mnist_utils.py`.

### 3.20 EarlyStopping

Em vez de adivinhar o número de épocas, o callback `tf.keras.callbacks.EarlyStopping` observa uma métrica de validação (`monitor="val_loss"`) e interrompe o treino quando ela deixa de melhorar por `patience` épocas seguidas. `restore_best_weights=True` devolve ao modelo os pesos da melhor época, não os da última. O script 18 pede 20 épocas mas termina antes; `len(history.history["loss"])` mostra quantas rodaram de fato.

```python
# 18_early_stopping.py
callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)
history = model.fit(x_train, y_train, validation_split=0.1, epochs=20,
                    batch_size=128, callbacks=[callback])
print("Épocas executadas:", len(history.history["loss"]))
```

Onde aparece: `18_early_stopping.py`.

### Bloco 6: avaliação e inferência

### 3.21 Inspecionar os erros

Depois da matriz de confusão (3.11, repetida para a CNN no script 19), o script 20 mostra as imagens que a rede errou. `np.where(pred != y_test)[0]` devolve os índices dos erros; `x_test[idx].squeeze()` remove a dimensão de canal para o `imshow`. Ver os erros ensina que muitos deles são dígitos mal escritos ou ambíguos até para humanos.

```python
# 20_erros_de_classificacao.py
pred = model.predict(x_test, verbose=0).argmax(axis=1)
wrong = np.where(pred != y_test)[0]
print("Total de erros:", len(wrong))
for i, idx in enumerate(wrong[:12]):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[idx].squeeze(), cmap="gray")
    plt.title(f"real={y_test[idx]}, pred={pred[idx]}")
```

Onde aparece: `19_matriz_confusao_cnn.py`, `20_erros_de_classificacao.py`.

### 3.22 Inferência em uma amostra e confiança

Para classificar uma única imagem, é preciso manter a dimensão de lote: `x_test[idx:idx+1]` tem forma `(1, 28, 28, 1)`, enquanto `x_test[idx]` teria `(28, 28, 1)` e falharia. `predict` devolve `(1, 10)`; `[0]` pega o vetor, `np.argmax` a classe e `probs[pred]` a confiança (a probabilidade softmax da classe escolhida). O mesmo padrão reaparece em `predict_real_digit` de `mnist_utils.py`, que devolve um dicionário com `prediction` e `confidence`.

```python
# 21_inferencia_em_um_mnist.py
idx = 0
probs = model.predict(x_test[idx:idx+1], verbose=0)[0]
pred = int(np.argmax(probs))
plt.title(f"Real: {y_test[idx]} | Predito: {pred} | Confiança: {probs[pred]:.1%}")
```

Onde aparece: `21_inferencia_em_um_mnist.py`, `mnist_utils.py` (`predict_real_digit`), `31_webcam_digito_tempo_real.py`.

### Bloco 7: dígitos reais com OpenCV

### 3.23 Pré-processamento de uma foto para o formato MNIST

Uma foto de dígito não se parece com o MNIST: é colorida, grande, com fundo claro e dígito escuro, muitas vezes descentrada. A função `preprocess_digit_image` em `mnist_utils.py` implementa o pipeline exigido no enunciado (cinza, Otsu, resize 28x28, normalização) com dois cuidados extras. Passo a passo:

1. `cv2.imread` lê em BGR; `cv2.cvtColor(..., cv2.COLOR_BGR2GRAY)` converte para cinza.
2. `cv2.GaussianBlur(gray, (5, 5), 0)` suaviza ruído antes de binarizar.
3. `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`: o método de Otsu escolhe sozinho o limiar que melhor separa fundo e traço (o `0` do segundo argumento é ignorado nesse modo).
4. Inversão de polaridade: MNIST tem fundo preto e dígito branco. Se a média da imagem binária passa de 127 (fundo branco), faz `255 - binary`.
5. Recorte automático: `cv2.findNonZero` lista os pixels brancos e `cv2.boundingRect` dá o retângulo que os envolve, eliminando fundo sobrando.
6. Centralização em um quadrado preto de lado `max(altura, largura)`, para não distorcer a proporção no resize.
7. `cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)`: `INTER_AREA` é a interpolação indicada para reduzir imagens.
8. Normalização para [0, 1] e `reshape(1, 28, 28, 1)` para virar lote de uma amostra.

A função devolve todas as etapas intermediárias, que o script 22 exibe lado a lado para inspeção. Note que o `original[:, :, ::-1]` no script 22 inverte BGR para RGB para o Matplotlib.

```python
# mnist_utils.py, preprocess_digit_image (condensado)
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if invert_if_needed and np.mean(binary) > 127:
    binary = 255 - binary
coords = cv2.findNonZero(binary)
x, y, w, h = cv2.boundingRect(coords)
digit = binary[y : y + h, x : x + w]
side = max(digit.shape[:2])
square = np.zeros((side, side), dtype=np.uint8)
square[y0 : y0 + digit.shape[0], x0 : x0 + digit.shape[1]] = digit
resized = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
model_input = (resized.astype("float32") / 255.0).reshape(1, 28, 28, 1)
```

Onde aparece: `mnist_utils.py`, `22_opencv_preprocessar_digito_real.py`, e via `predict_real_digit` em 23, 24, 25, 28, 30. O script 31 repete uma versão simplificada (sem blur, recorte nem centralização).

### 3.24 Argumentos de linha de comando com argparse

Os scripts do Item B recebem caminhos por linha de comando. `argparse.ArgumentParser()` mais `add_argument("--imagem", default=...)` define uma opção com valor padrão; `parser.parse_args()` lê `sys.argv` e devolve um objeto com atributos `args.imagem`, `args.modelo`, `args.pasta`. Sem argumentos, os padrões são `data/real_digits/5.jpg`, `resultados/cnn_mnist.keras` e `data/real_digits`.

```python
# 23_inferir_digito_real_unico.py
parser = argparse.ArgumentParser()
parser.add_argument("--imagem", default="data/real_digits/5.jpg")
parser.add_argument("--modelo", default="resultados/cnn_mnist.keras")
args = parser.parse_args()
```

Onde aparece: `22_opencv_preprocessar_digito_real.py`, `23_inferir_digito_real_unico.py`, `24_inferir_pasta_digitos_reais.py`, `25_acuracia_imagens_reais.py`, `28_comparar_real_antes_depois_aug.py`.

### 3.25 Anotar a predição na imagem com OpenCV

`annotate_prediction` copia a imagem original e escreve o texto com `cv2.putText(img, texto, (x, y), fonte, escala, cor_BGR, espessura, cv2.LINE_AA)`. A cor `(0, 255, 0)` é verde em BGR e `LINE_AA` é o antialiasing. Para exibir no Matplotlib é preciso `cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)`, senão as cores saem trocadas. O script 24 percorre os 10 arquivos com `list_real_digit_images`, que procura `0.jpg` a `9.jpg` (ou `.jpeg`, `.png`, `.bmp`) e devolve pares `(caminho, rótulo)`.

```python
# mnist_utils.py
def annotate_prediction(original_bgr, prediction, confidence):
    annotated = original_bgr.copy()
    text = f"Predito: {prediction} ({confidence:.1%})"
    cv2.putText(annotated, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    return annotated

# 24_inferir_pasta_digitos_reais.py
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
```

Onde aparece: `mnist_utils.py`, `23_inferir_digito_real_unico.py`, `24_inferir_pasta_digitos_reais.py`.

### 3.26 Acurácia nas imagens reais

O script 25 calcula acertos sobre as 10 fotos e imprime, por arquivo, real, predito, confiança e se acertou. Com só 10 amostras, cada erro vale 10 pontos percentuais, então o número é uma estimativa grosseira, mas suficiente para mostrar a queda em relação aos 99% do teste MNIST. A discussão impressa pelo professor lista as causas: iluminação, fundo, espessura do traço, escala, rotação e ruído.

```python
# 25_acuracia_imagens_reais.py
correct = 0
for path, label in samples:
    result = predict_real_digit(model, path)
    ok = result["prediction"] == label
    correct += int(ok)
    print(f"{path.name}: real={label}, predito={result['prediction']}, "
          f"confiança={result['confidence']:.2%}, correto={ok}")
acc_real = correct / len(samples) if samples else 0.0
```

Onde aparece: `25_acuracia_imagens_reais.py`, `28_comparar_real_antes_depois_aug.py` (função `real_accuracy`), `30_pipeline_completo_item_a_b.py`.

### Bloco 8: data augmentation e domain gap

### 3.27 Data augmentation com ImageDataGenerator

Data augmentation gera variações artificiais das imagens de treino a cada época, para a rede ver mais diversidade sem coletar dados novos. `ImageDataGenerator(rotation_range=15, zoom_range=0.10)` aplica rotação aleatória de até ±15 graus e zoom de até ±10%, os valores exigidos no enunciado. `datagen.flow(x, y, batch_size=...)` devolve um gerador infinito de lotes já transformados. O script 26 mostra 9 variações da mesma imagem; note o `break` em `i == 8`, necessário porque o gerador nunca termina sozinho.

Para treinar, o script 27 passa `validation_split=0.1` ao gerador e cria dois fluxos com `subset="training"` e `subset="validation"`. `model.fit(train_gen, validation_data=val_gen, epochs=5)` consome os geradores diretamente. A avaliação final continua sendo feita no `x_test` limpo, sem augmentation.

```python
# 27_treinar_cnn_com_augmentation.py
datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.10, validation_split=0.1)
train_gen = datagen.flow(x_train, y_train, batch_size=128, subset="training")
val_gen = datagen.flow(x_train, y_train, batch_size=128, subset="validation")

model = build_cnn()
history = model.fit(train_gen, validation_data=val_gen, epochs=5)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
model.save("resultados/cnn_mnist_aug.keras")
```

Onde aparece: `26_visualizar_data_augmentation.py`, `27_treinar_cnn_com_augmentation.py`, `28_comparar_real_antes_depois_aug.py`, `30_pipeline_completo_item_a_b.py`.

### 3.28 Domain gap e comparação antes/depois

Domain gap é a diferença entre a distribuição dos dados de treino e a dos dados de uso. O MNIST é limpo, centralizado, com fundo padronizado e 28x28; uma foto real tem sombra, papel amarelado, perspectiva, rotação, escala e traço variável. Resultado: alta acurácia no teste MNIST não garante acerto nas fotos. O script 28 garante que os dois modelos existam (`ensure_models`, treinando se os `.keras` não estiverem em `resultados/`), calcula a acurácia real de cada um e imprime a comparação. A conclusão do professor, impressa pelo script 29 e repetida no 28 e 30: augmentation aproxima o treino das variações reais, mas não resolve todo o gap; iluminação, perspectiva e segmentação ruins pedem melhor pré-processamento e exemplos reais para validação.

```python
# 28_comparar_real_antes_depois_aug.py
def real_accuracy(model, samples):
    correct = 0
    for path, label in samples:
        pred = predict_real_digit(model, path)["prediction"]
        correct += int(pred == label)
    return correct / len(samples) if samples else 0.0

clean_model, aug_model = ensure_models()
print(f"Acurácia real antes do augmentation:  {real_accuracy(clean_model, samples):.2%}")
print(f"Acurácia real após augmentation:      {real_accuracy(aug_model, samples):.2%}")
```

Onde aparece: `28_comparar_real_antes_depois_aug.py`, `29_relatorio_domain_gap.py`, `30_pipeline_completo_item_a_b.py`.

### Bloco 9: pipeline e webcam

### 3.29 Pipeline completo (Item A + Item B)

O script 30 encadeia tudo em uma execução: treina MLP e CNN por 5 épocas com `EpochTimer`, plota as curvas, salva a tabela em `resultados/pipeline_comparacao.csv` e o modelo em `resultados/cnn_mnist.keras`; depois, se houver fotos em `data/real_digits/`, mede a acurácia real, treina uma segunda CNN com augmentation, salva em `resultados/cnn_mnist_aug.keras` e mede de novo. Se a pasta de fotos estiver vazia, o Item B é pulado com uma mensagem, sem erro.

```python
# 30_pipeline_completo_item_a_b.py (trecho do Item B)
samples = list_real_digit_images(REAL_FOLDER)
if not samples:
    print("\nItem B: nenhuma imagem real encontrada.")
else:
    ...
    datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.10, validation_split=0.1)
    cnn_aug = build_cnn()
    h_aug = cnn_aug.fit(
        datagen.flow(x_train_cnn, y_train, batch_size=BATCH, subset="training"),
        validation_data=datagen.flow(x_train_cnn, y_train, batch_size=BATCH, subset="validation"),
        epochs=EPOCHS,
    )
```

Onde aparece: `30_pipeline_completo_item_a_b.py`.

### 3.30 Inferência ao vivo pela webcam

O script 31 abre a câmera com `cv2.VideoCapture(0)`, lê frames em loop com `cap.read()` e desenha um quadrado azul central (`cv2.rectangle`, cor `(255, 0, 0)` em BGR) com lado igual à metade da menor dimensão do frame. `cv2.waitKey(1) & 0xFF` lê a tecla: 27 é ESC (sair) e 32 é ESPAÇO (classificar). Ao classificar, recorta a ROI (region of interest, a região de interesse) do frame, aplica cinza, Otsu, inversão condicional, resize 28x28 com `INTER_AREA`, normalização e reshape, e roda `model.predict`. O texto da predição fica na tela até a próxima captura. No fim, `cap.release()` e `cv2.destroyAllWindows()` liberam a câmera e fecham as janelas.

```python
# 31_webcam_digito_tempo_real.py
key = cv2.waitKey(1) & 0xFF
if key == 27:
    break
if key == 32:
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary) > 127:
        binary = 255 - binary
    resized = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
    x = resized.astype("float32").reshape(1, 28, 28, 1) / 255.0
    probs = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(probs))
    prediction_text = f"Predito: {pred} ({probs[pred]:.1%})"
```

Onde aparece: `31_webcam_digito_tempo_real.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `mnist_utils.py` | Módulo compartilhado: `load_mnist`, `build_mlp`, `build_cnn`, `EpochTimer`, `plot_history`, `preprocess_digit_image`, `predict_real_digit`, `list_real_digit_images`, `annotate_prediction`, `model_summary_row`, `save_comparison_table` | Importado pelos outros | Nada por si só | tensorflow, cv2, numpy, pandas, matplotlib |
| `00_verificar_ambiente.py` | Imprime versões de TensorFlow, Keras, OpenCV, NumPy, Matplotlib e lista GPUs | Nada | Texto no terminal | Bibliotecas instaladas |
| `01_carregar_mnist.py` | Baixa/carrega MNIST e imprime shapes, min, max e primeiro rótulo | MNIST (download) | Texto | Internet na primeira vez |
| `02_visualizar_amostras.py` | Grade 4x5 com 20 imagens e rótulos | MNIST | Janela Matplotlib | 01 (cache) |
| `03_normalizacao_e_formas.py` | Normaliza para [0, 1] e mostra as formas para MLP (784) e CNN (28, 28, 1) | MNIST | Texto | Nada |
| `04_rotulos_e_one_hot.py` | Compara rótulos inteiros com one-hot via `to_categorical` | MNIST | Texto | Nada |
| `05_mlp_minimo.py` | MLP com uma camada oculta de 64, 2 épocas, avalia no teste | MNIST | Texto | `mnist_utils` |
| `06_mlp_duas_camadas_densas.py` | MLP do Item A (256 + 128), 3 épocas | MNIST | Texto | `mnist_utils` |
| `07_mlp_curvas_treino.py` | MLP por 5 épocas e curvas de treino/validação | MNIST | `resultados/mlp_curvas.png` e janela | `mnist_utils` |
| `08_mlp_matriz_confusao.py` | MLP por 3 épocas e matriz de confusão | MNIST | Janela Matplotlib | `mnist_utils`, sklearn |
| `09_cnn_minima.py` | CNN com um bloco Conv+Pool, 2 épocas | MNIST | Texto | `mnist_utils` |
| `10_cnn_dois_blocos.py` | CNN do Item A (dois blocos), 3 épocas | MNIST | Texto | `mnist_utils` |
| `11_cnn_curvas_treino.py` | CNN por 5 épocas e curvas | MNIST | `resultados/cnn_curvas.png` e janela | `mnist_utils` |
| `12_comparar_mlp_cnn.py` | Item A completo: MLP e CNN por 5 épocas, summary, curvas, tabela de parâmetros, tempo e acurácia | MNIST | `resultados/curvas_mlp.png`, `resultados/curvas_cnn.png`, `resultados/comparacao_modelos.csv` | `mnist_utils`, pandas |
| `13_salvar_e_carregar_cnn.py` | Treina CNN 3 épocas, salva, recarrega e avalia | MNIST | `resultados/cnn_mnist.keras` | `mnist_utils` |
| `14_visualizar_filtros_cnn.py` | Treina 2 épocas e mostra 16 filtros 3x3 da primeira Conv2D | MNIST | Janela Matplotlib | `mnist_utils` |
| `15_visualizar_feature_maps.py` | Treina 2 épocas e mostra 16 mapas de ativação da primeira Conv2D | MNIST | Janela Matplotlib | `mnist_utils` |
| `16_overfitting_com_poucos_dados.py` | CNN com 1000 amostras, 12 épocas, para provocar overfitting | MNIST | `resultados/overfitting_poucos_dados.png` | `mnist_utils` |
| `17_regularizacao_dropout.py` | CNN com Dropout 0.4, 5 épocas, curvas | MNIST | `resultados/cnn_dropout_curvas.png` | `mnist_utils` |
| `18_early_stopping.py` | CNN com EarlyStopping (val_loss, patience 2), até 20 épocas | MNIST | `resultados/cnn_early_stopping.png` | `mnist_utils` |
| `19_matriz_confusao_cnn.py` | CNN 3 épocas e matriz de confusão | MNIST | Janela Matplotlib | `mnist_utils`, sklearn |
| `20_erros_de_classificacao.py` | CNN 3 épocas e grade com 12 imagens erradas | MNIST | Janela Matplotlib | `mnist_utils` |
| `21_inferencia_em_um_mnist.py` | CNN 3 épocas e predição de uma amostra do teste com confiança | MNIST | Janela Matplotlib | `mnist_utils` |
| `22_opencv_preprocessar_digito_real.py` | Mostra as etapas do pré-processamento OpenCV de uma foto | `--imagem` (padrão `data/real_digits/5.jpg`) | Janela com 4 painéis | Foto real, `mnist_utils` |
| `23_inferir_digito_real_unico.py` | Carrega ou treina CNN e prevê o dígito de uma foto | `--imagem`, `--modelo` | Janela anotada, texto; cria `resultados/cnn_mnist.keras` se faltar | Foto real, `mnist_utils` |
| `24_inferir_pasta_digitos_reais.py` | Prevê os 10 arquivos `0.jpg` a `9.jpg` e mostra em grade | `--pasta`, `--modelo` | Janela 2x5; cria o modelo se faltar | Fotos reais, `mnist_utils` |
| `25_acuracia_imagens_reais.py` | Acurácia nas fotos reais antes do augmentation | `--pasta`, `--modelo` | Texto; cria o modelo se faltar | Fotos reais, `mnist_utils` |
| `26_visualizar_data_augmentation.py` | Mostra 9 variações (rotação, zoom) de uma imagem | MNIST | Janela Matplotlib | `ImageDataGenerator` |
| `27_treinar_cnn_com_augmentation.py` | Treina CNN com augmentation por 5 épocas e salva | MNIST | `resultados/cnn_aug_curvas.png`, `resultados/cnn_mnist_aug.keras` | `mnist_utils` |
| `28_comparar_real_antes_depois_aug.py` | Acurácia real do modelo limpo vs com augmentation | `--pasta` | Texto; cria os dois `.keras` se faltarem | Fotos reais, `mnist_utils` |
| `29_relatorio_domain_gap.py` | Imprime um texto de discussão sobre domain gap | Nada | Texto | Nada |
| `30_pipeline_completo_item_a_b.py` | Item A + Item B em uma execução | MNIST, `data/real_digits/` | `resultados/pipeline_mlp_curvas.png`, `pipeline_cnn_curvas.png`, `pipeline_cnn_aug_curvas.png`, `pipeline_comparacao.csv`, `cnn_mnist.keras`, `cnn_mnist_aug.keras` | `mnist_utils`, fotos reais para o Item B |
| `31_webcam_digito_tempo_real.py` | Classifica o dígito dentro do quadrado central da webcam ao apertar ESPAÇO | Webcam | Janela OpenCV; cria `resultados/cnn_mnist.keras` se faltar | Webcam, `mnist_utils` |

## 5. Passo a passo para executar

Tudo é executado de dentro de `aula_11_MNIST/exemplos_em_python/`.

```
cd C:\visao_computacional_PVA\aula_11_MNIST\exemplos_em_python
```

**Passo 1: ambiente e dados.** Confira as versões e baixe o MNIST.

```
python 00_verificar_ambiente.py
python 01_carregar_mnist.py
python 02_visualizar_amostras.py
python 03_normalizacao_e_formas.py
python 04_rotulos_e_one_hot.py
```

Espere ver as versões no terminal, o download do MNIST na primeira vez, uma janela com 20 dígitos e os shapes `(60000, 28, 28)`, `(60000, 784)` e `(60000, 28, 28, 1)`. Feche cada janela do Matplotlib para o script terminar.

**Passo 2: MLP.**

```
python 05_mlp_minimo.py
python 06_mlp_duas_camadas_densas.py
python 07_mlp_curvas_treino.py
python 08_mlp_matriz_confusao.py
```

O terminal mostra o `summary` e a barra de progresso por época. Acurácia de teste em torno de 0,97 a 0,98. O script 07 cria `resultados/mlp_curvas.png`. O 08 abre a matriz de confusão 10x10.

**Passo 3: CNN.**

```
python 09_cnn_minima.py
python 10_cnn_dois_blocos.py
python 11_cnn_curvas_treino.py
python 12_comparar_mlp_cnn.py
```

Acurácia de teste em torno de 0,99. O 12 é o Item A completo: treina os dois modelos, abre duas janelas de curvas e imprime a tabela comparativa, gravada em `resultados/comparacao_modelos.csv`.

**Passo 4: salvar o modelo.** Faça isso antes do Item B para não retreinar a cada script.

```
python 13_salvar_e_carregar_cnn.py
```

Gera `resultados/cnn_mnist.keras`.

**Passo 5: olhar dentro da CNN.**

```
python 14_visualizar_filtros_cnn.py
python 15_visualizar_feature_maps.py
```

Duas janelas: 16 filtros 3x3 em cinza e 16 mapas de ativação coloridos do primeiro dígito de treino. Cada um treina por 2 épocas antes de mostrar.

**Passo 6: overfitting e regularização.**

```
python 16_overfitting_com_poucos_dados.py
python 17_regularizacao_dropout.py
python 18_early_stopping.py
```

No 16, espere ver a curva de treino disparar e a de validação ficar abaixo. No 17, curvas mais próximas. No 18, o treino para antes das 20 épocas e o terminal diz quantas rodaram.

**Passo 7: avaliação da CNN.**

```
python 19_matriz_confusao_cnn.py
python 20_erros_de_classificacao.py
python 21_inferencia_em_um_mnist.py
```

Matriz de confusão, 12 erros com real e predito, e a predição do primeiro dígito do teste (um 7) com confiança.

**Passo 8: preparar as fotos reais.** Escreva os dígitos 0 a 9 em papel, fotografe um por vez, e salve como `data/real_digits/0.jpg` ... `data/real_digits/9.jpg`. Fundo claro, traço escuro e grosso, dígito centralizado. Depois:

```
python 22_opencv_preprocessar_digito_real.py --imagem data/real_digits/5.jpg
python 23_inferir_digito_real_unico.py --imagem data/real_digits/5.jpg
python 24_inferir_pasta_digitos_reais.py
python 25_acuracia_imagens_reais.py
```

O 22 mostra original, cinza, Otsu e 28x28 lado a lado; confira se o dígito aparece branco sobre preto no último painel. O 23 mostra a foto anotada. O 24 mostra as 10 em grade. O 25 imprime a acurácia real.

**Passo 9: augmentation e domain gap.**

```
python 26_visualizar_data_augmentation.py
python 27_treinar_cnn_com_augmentation.py
python 28_comparar_real_antes_depois_aug.py
python 29_relatorio_domain_gap.py
```

O 27 gera `resultados/cnn_mnist_aug.keras`. O 28 imprime as duas acurácias reais. O 29 imprime o texto de discussão.

**Passo 10: pipeline completo (opcional, repete tudo em uma execução).**

```
python 30_pipeline_completo_item_a_b.py
```

Treina MLP, CNN e CNN com augmentation por 5 épocas cada. Abre 3 janelas de curvas (feche cada uma para continuar) e grava tudo em `resultados/` com prefixo `pipeline_`.

**Passo 11: webcam.**

```
python 31_webcam_digito_tempo_real.py
```

Teclas: ESPAÇO classifica o que está dentro do quadrado azul, ESC sai. Segure um papel com um dígito escuro sobre fundo claro preenchendo o quadrado.

## 6. Exercícios e itens da aula

Os docstrings referenciam dois itens de um exercício.

**Item A: MLP versus CNN no MNIST.** Pede para treinar um MLP com pelo menos duas camadas densas (`06_mlp_duas_camadas_densas.py`) e uma CNN com pelo menos dois blocos Conv2D + MaxPooling (`10_cnn_dois_blocos.py`), imprimir `model.summary()` de cada um, plotar as curvas de treino e validação, comparar número de parâmetros, tempo por época e acurácia de teste em uma tabela, e comentar indícios de overfitting. Resolvido por completo em `12_comparar_mlp_cnn.py` e na primeira metade de `30_pipeline_completo_item_a_b.py`. A justificativa esperada, escrita pelo professor no docstring do 12: a CNN supera o MLP porque preserva a estrutura espacial 2D, aprende filtros locais e compartilha pesos; o MLP achata a imagem e perde a vizinhança entre pixels.

**Item B: dígitos reais e data augmentation.** Pede para fotografar 10 dígitos manuscritos (um por classe), pré-processar com OpenCV na sequência escala de cinza, binarização Otsu, resize 28x28 e normalização (`22_opencv_preprocessar_digito_real.py`), medir a acurácia da CNN nessas fotos (`25_acuracia_imagens_reais.py`), treinar de novo com augmentation de rotação ±15° e zoom ±10% (`27_treinar_cnn_com_augmentation.py`), comparar a acurácia real antes e depois (`28_comparar_real_antes_depois_aug.py`) e discutir o domain gap (`29_relatorio_domain_gap.py`). Resolvido na segunda metade de `30_pipeline_completo_item_a_b.py`. O script 13 existe para "evitar retreinar o modelo em todos os exemplos do Item B".

O script 31 é um extra que o professor descreve como aproximação do exercício de percepção robótica.

## 7. Armadilhas e dicas

- **Rode de dentro de `exemplos_em_python/`.** Os caminhos `resultados/` e `data/real_digits/` são relativos. Executar de outra pasta cria `resultados/` no lugar errado ou falha ao achar as fotos.
- **Ordem entre 13 e o Item B.** Os scripts 23, 24, 25, 28 e 31 treinam uma CNN de 3 épocas se `resultados/cnn_mnist.keras` não existir. Rodar o 13 antes economiza esse tempo em cada um. Repare que o 30 sobrescreve `cnn_mnist.keras` com uma versão de 5 épocas.
- **Fotos reais são obrigatórias para 24, 25, 28 e o Item B do 30.** Sem elas, 24 e 28 saem com `SystemExit`, 25 divide por zero protegido (imprime 0%), e o 30 pula o Item B com aviso. O 22 e o 23 falham com `FileNotFoundError` se `data/real_digits/5.jpg` não existir.
- **Polaridade e fundo.** O pré-processamento inverte a imagem quando a média da binária passa de 127, ou seja, assume fundo claro e traço escuro. Fotos com fundo escuro e caneta clara podem ficar invertidas de forma errada. Sombras fortes fazem o Otsu binarizar a sombra como traço; use luz uniforme.
- **Traço fino.** Depois do resize para 28x28, um traço fino pode sumir. Escreva com caneta grossa ou marcador, parecido com o MNIST.
- **`INTER_AREA` para reduzir.** É a interpolação certa para diminuir imagens. Trocar por `INTER_LINEAR` deixa o dígito mais serrilhado e piora a acurácia real.
- **Dimensão de lote na inferência.** `model.predict` espera `(N, 28, 28, 1)`. Passar `(28, 28, 1)` dá erro de forma. Use `x[idx:idx+1]` ou `reshape(1, 28, 28, 1)`.
- **`ImageDataGenerator` é legado.** Funciona, mas o Keras 3 avisa que a API está depreciada. A alternativa moderna são as camadas `RandomRotation` e `RandomZoom` dentro do modelo. Mantido como o professor escreveu.
- **O gerador nunca termina.** No script 26, o `break` em `i == 8` é obrigatório; sem ele o loop roda para sempre.
- **Janelas do Matplotlib bloqueiam.** Cada `plt.show()` pausa o script até fechar a janela. No 12 e no 30 há várias em sequência.
- **Resultados variam entre execuções.** Não há semente fixa (`seed`), então acurácias e curvas mudam um pouco a cada rodada. Isso é normal.
- **`validation_split` pega o final do array.** Sem embaralhar, os últimos 10% do treino sempre são a validação. No 16, com 1000 amostras e `validation_split=0.3`, ficam só 700 para treinar.
- **Webcam.** O script 31 usa `VideoCapture(0)`. Se não abrir, troque o índice para 1 ou 2. A ROI é o quadrado central; o pré-processamento ali não recorta nem centraliza, então preencha bem o quadrado com o dígito.
- **Parâmetros que valem a pena variar:** `epochs` (5 para 10 nos scripts 07 e 11 para ver overfitting real), `dropout` (0,2 a 0,5 no 17), `patience` (1 a 5 no 18), `rotation_range` e `zoom_range` (no 27, testar 10 a 30 graus e 0,05 a 0,2), o número de filtros de `build_cnn`, o tamanho do subconjunto no 16 (100, 500, 5000).
- **Fidelidade ao professor.** O parâmetro `one_hot` de `load_mnist` nunca é usado, e `from __future__ import annotations` em `mnist_utils.py` permite os tipos `str | Path` mesmo em versões antigas. Tudo foi mantido como está.

## 8. Glossário

| Termo | Significado |
|---|---|
| MNIST | Dataset de 70 mil dígitos manuscritos 28x28 em cinza, 60 mil de treino e 10 mil de teste |
| MLP | Multi-Layer Perceptron, rede totalmente conectada em que cada neurônio de uma camada liga-se a todos da seguinte |
| CNN | Convolutional Neural Network, rede que usa convoluções para explorar a estrutura espacial de imagens |
| Tensor | Array multidimensional; uma imagem em lote tem forma (N, altura, largura, canais) |
| Shape | Tupla com o tamanho de cada dimensão de um array |
| Normalização | Levar os pixels de 0..255 para 0..1 dividindo por 255 |
| Flatten | Achatar um tensor multidimensional em um vetor 1D |
| One-hot | Vetor com 1 na posição da classe e 0 nas demais |
| Sparse categorical crossentropy | Função de perda para classificação multiclasse que aceita rótulos inteiros |
| Softmax | Ativação de saída que converte 10 números em probabilidades que somam 1 |
| ReLU | Rectified Linear Unit, ativação que devolve max(0, x) |
| Dense | Camada totalmente conectada do Keras |
| Conv2D | Camada de convolução 2D que desliza kernels pela imagem e gera mapas de características |
| Kernel (filtro) | Pequena matriz de pesos (3x3 na aula) que a convolução aplica em cada posição da imagem |
| Padding same | Preenchimento com zeros na borda para a saída da convolução ter o mesmo tamanho da entrada |
| MaxPooling2D | Reduz cada mapa pela metade ficando com o máximo de cada janela 2x2 |
| Feature map (mapa de ativação) | Saída de um filtro convolucional sobre a imagem |
| Época | Uma passagem completa pelo conjunto de treino |
| Batch (lote) | Grupo de amostras processadas por passo de gradiente |
| Adam | Otimizador com taxa de aprendizado adaptativa, padrão para começar |
| Loss (perda) | Medida do erro que o otimizador minimiza |
| Accuracy (acurácia) | Fração de predições corretas |
| validation_split | Fração do treino reservada pelo Keras para medir desempenho a cada época |
| History | Objeto devolvido por `fit` com as métricas por época em `.history` |
| Overfitting | Modelo memoriza o treino e piora em dados novos; treino sobe, validação estagna ou cai |
| Dropout | Regularização que zera aleatoriamente ativações durante o treino |
| EarlyStopping | Callback que interrompe o treino quando a métrica de validação para de melhorar |
| Callback | Objeto chamado pelo Keras em momentos do treino (início e fim de época, etc.) |
| Patience | Número de épocas sem melhora tolerado pelo EarlyStopping |
| Matriz de confusão | Tabela real x predito que mostra quais classes são confundidas |
| Confiança | Probabilidade softmax da classe predita |
| Argmax | Índice do maior valor de um vetor |
| Inferência | Usar o modelo treinado para prever uma amostra nova |
| Formato .keras | Arquivo nativo do Keras 3 com arquitetura, pesos e otimizador |
| BGR / RGB | Ordem dos canais de cor no OpenCV (BGR) e no Matplotlib (RGB) |
| Escala de cinza | Imagem de um canal com intensidades de 0 a 255 |
| Gaussian blur | Suavização por média ponderada gaussiana, reduz ruído antes de binarizar |
| Otsu | Método que escolhe automaticamente o limiar de binarização separando fundo e objeto |
| Threshold (limiar) | Valor de corte que separa pixels em preto e branco |
| Binarização | Converter a imagem para só dois valores, 0 e 255 |
| Bounding rect | Menor retângulo que envolve os pixels não nulos |
| INTER_AREA | Interpolação do OpenCV indicada para reduzir imagens |
| ROI | Region of Interest, região de interesse recortada de uma imagem ou frame |
| Frame | Um quadro de vídeo capturado da câmera |
| argparse | Módulo padrão para ler argumentos de linha de comando |
| Data augmentation | Geração de variações artificiais (rotação, zoom) das imagens de treino |
| ImageDataGenerator | Classe legada do Keras que gera lotes aumentados |
| rotation_range | Rotação aleatória máxima em graus no augmentation |
| zoom_range | Zoom aleatório máximo (fração) no augmentation |
| Domain gap | Diferença entre a distribuição dos dados de treino e a dos dados de uso real |
| DataFrame | Tabela do pandas usada para a comparação de modelos |
| CSV | Arquivo de texto com valores separados por vírgula |
| GPU | Placa gráfica; o script 00 lista as disponíveis, mas a aula roda em CPU |
