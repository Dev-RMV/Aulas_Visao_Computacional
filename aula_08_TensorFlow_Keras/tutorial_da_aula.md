# Aula 08 – TensorFlow e Keras

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula ensina a construir, treinar, avaliar e interpretar uma rede neural convolucional (CNN) do zero com TensorFlow e Keras, usando um dataset sintético de três formas geométricas (círculo, quadrado e triângulo) gerado com OpenCV. O fio condutor é uma sequência de laboratório: primeiro o aluno gera as imagens, entende que uma imagem é um tensor e vê o que uma convolução faz com kernels fixos; depois monta a CNN, treina por pelo menos 10 épocas (Item A), avalia com matriz de confusão e olha os filtros e mapas de ativação aprendidos.

A segunda metade da aula trata a CNN como extrator de features: a penúltima camada Dense produz um vetor de 64 valores por imagem, que é reduzido a duas dimensões com PCA e comparado, sobre as mesmas 20 imagens, com o descritor clássico ORB convertido em Bag of Visual Words (Item B). Os dois últimos scripts fecham o ciclo com a classificação de uma imagem isolada e um benchmark de latência entre CNN e ORB, pensando em robótica embarcada.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas pelos scripts: `tensorflow` (com Keras embutido), `numpy`, `opencv-python` (`cv2`), `matplotlib` e `scikit-learn` (`sklearn`).

```powershell
pip install tensorflow numpy opencv-python matplotlib scikit-learn
```

Pontos importantes sobre o ambiente:

- O módulo `utils.py` define a raiz do projeto como a pasta acima da dos scripts, ou seja, `aula_08_TensorFlow_Keras/`. Nela ficam as pastas `dataset/`, `saidas/`, `modelos/` e `recursos/`, criadas automaticamente por `ensure_dirs()`.
- Os scripts importam `from utils import ...`, então devem ser executados de dentro de `exemplos_em_python/` (ou com essa pasta no `sys.path`).
- A aula é autossuficiente. Nenhum arquivo externo precisa ser baixado: `01_gerar_dataset.py` cria todas as imagens.
- Nenhum script usa webcam.
- O treino roda em CPU. O script `00_validar_ambiente.py` imprime as GPUs visíveis e avisa que a GPU apenas reduz o tempo.
- `utils.py` define `os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")` para silenciar avisos do TensorFlow.
- O CLAUDE.md registra que o professor usou Python 3.12 e 3.13 em outras aulas e que a máquina local tem Python 3.14. Confirme que existe wheel do TensorFlow para a versão instalada antes de rodar.

## 3. Conceitos-chave

### 3.1 Reprodutibilidade com sementes

Redes neurais são inicializadas com pesos aleatórios e o dataset é embaralhado a cada época. Sem fixar sementes, cada execução dá um resultado diferente, o que atrapalha a comparação entre experimentos. A função `configure()` fixa a semente do `random`, do NumPy e do Keras com um único valor, e tenta ativar operações determinísticas no TensorFlow.

```python
# utils.py
SEED = 2026

def configure(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
```

`tf.keras.utils.set_random_seed` fixa de uma vez as sementes do Python, do NumPy e do TensorFlow. `enable_op_determinism` força kernels determinísticos, ao custo de velocidade, e está dentro de `try` porque nem toda versão suporta.

Onde aparece: `utils.py`, `04_criar_modelo_cnn.py`, `05_item_a_treinar_cnn.py`, `06_avaliar_modelo.py`.

### 3.2 Geração de dataset sintético com OpenCV

Antes de treinar, é preciso ter dados. O professor gera um dataset próprio de 1620 imagens RGB de 64x64 pixels, com três classes e divisão em `train` (360 por classe), `val` (90) e `test` (90). Cada amostra é desenhada com primitivas do OpenCV e recebe variações que imitam uma câmera real: posição, rotação, escala, gradiente de iluminação, ruído gaussiano, desfoque e oclusão ocasional. Isso é importante porque a CNN precisa aprender a forma, e não decorar uma posição fixa.

Cada imagem é composta em duas camadas: a forma é desenhada em `layer` e sua silhueta em `mask`; as duas passam pela mesma rotação e escala com `cv2.getRotationMatrix2D` e `cv2.warpAffine`, e depois a máscara vira um canal alfa para fundir a forma com o fundo.

```python
# 01_gerar_dataset.py
angle = float(rng.uniform(-24, 24))
scale = float(rng.uniform(0.82, 1.10))
matrix = cv2.getRotationMatrix2D((32, 32), angle, scale)
layer = cv2.warpAffine(layer, matrix, (width, height), flags=cv2.INTER_LINEAR)
mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_LINEAR)
alpha = (mask.astype(np.float32) / 255.0)[..., None]
image = np.uint8(image * (1.0 - alpha) + layer * alpha)
```

Funções do OpenCV usadas: `cv2.circle`, `cv2.line`, `cv2.fillConvexPoly`, `cv2.polylines`, `cv2.rectangle` para desenhar; `cv2.GaussianBlur` para desfoque; `cv2.imwrite` para gravar. O gerador aleatório é `np.random.default_rng` com uma semente diferente por split e por classe, o que torna o dataset reproduzível. O script também apaga e recria a pasta `dataset/` com `shutil.rmtree`, monta uma prévia com 6 amostras por classe em `recursos/amostras_dataset.png` e grava `dataset/dataset_info.json`.

Onde aparece: `01_gerar_dataset.py`.

### 3.3 Imagem como tensor

Uma imagem colorida é um arranjo tridimensional altura x largura x canais. No OpenCV o `cv2.imread` devolve os canais na ordem BGR, então o script converte para RGB com `cv2.cvtColor`. Os valores são inteiros `uint8` de 0 a 255; para alimentar uma rede, converte-se para `float32` e divide-se por 255, deixando o intervalo entre 0 e 1.

```python
# 02_imagem_como_tensor.py
bgr = cv2.imread(str(path))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
normalized = rgb.astype(np.float32) / 255.0
print("shape do tensor:", rgb.shape, "= altura × largura × canais")
print("dtype antes:", rgb.dtype, "intervalo:", (rgb.min(), rgb.max()))
print("Pixel central RGB:", rgb[32, 32].tolist())
```

O script ainda recorta um patch de 8x8 pixels com fatiamento NumPy (`bgr[28:36, 28:36]`) e amplia para 320x320 com `cv2.resize` e interpolação `INTER_NEAREST`, para que cada pixel apareça como um bloco visível na discussão em sala.

Onde aparece: `02_imagem_como_tensor.py`, `utils.py` (função `load_rgb`).

### 3.4 Convolução com kernels fixos

Convolução é a operação que desliza uma pequena matriz (kernel) sobre a imagem e, em cada posição, calcula a soma ponderada da vizinhança. Antes de deixar a CNN aprender seus próprios kernels, o professor mostra três kernels conhecidos: Sobel vertical, Sobel horizontal (a transposta do primeiro) e média 3x3 (blur).

```python
# 03_convolucao_passo_a_passo.py
kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
kernel_horizontal = kernel_vertical.T
kernel_blur = np.ones((3, 3), np.float32) / 9.0

outputs = [
    image,
    cv2.convertScaleAbs(cv2.filter2D(image, cv2.CV_32F, kernel_vertical)),
    cv2.convertScaleAbs(cv2.filter2D(image, cv2.CV_32F, kernel_horizontal)),
    cv2.filter2D(image, -1, kernel_blur),
]
```

`cv2.filter2D(imagem, ddepth, kernel)` aplica a convolução. O segundo argumento é a profundidade da saída: `cv2.CV_32F` mantém valores negativos das bordas, que depois são convertidos para 0 a 255 com `cv2.convertScaleAbs` (valor absoluto e saturação); `-1` mantém o mesmo tipo da entrada, adequado para o blur. O resultado é uma montagem de quatro painéis com legendas escritas por `cv2.putText`.

Onde aparece: `03_convolucao_passo_a_passo.py`.

### 3.5 Arquitetura da CNN com a API funcional do Keras

A CNN é construída com a API funcional: cria-se um `keras.Input`, encadeiam-se camadas chamando cada uma como função sobre o tensor anterior, e fecha-se com `keras.Model(inputs, outputs)`. A rede tem dois blocos convolucionais (duas `Conv2D` seguidas de `MaxPooling2D`), um `Flatten`, uma `Dense` de 64 unidades chamada `features_dense`, um `Dropout` e a `Dense` final de 3 saídas com softmax.

```python
# utils.py, função build_cnn
inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="imagem_rgb")
x = layers.Rescaling(1.0 / 255.0, name="normalizacao")(inputs)
x = layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1a")(x)
x = layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1b")(x)
x = layers.MaxPooling2D(2, name="pool1")(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2a")(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2b")(x)
x = layers.MaxPooling2D(2, name="pool2")(x)
x = layers.Flatten(name="flatten")(x)
features = layers.Dense(64, activation="relu", name="features_dense")(x)
x = layers.Dropout(0.15, name="dropout")(features)
outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="classe")(x)
model = keras.Model(inputs, outputs, name="cnn_geometrica")
```

Papel de cada camada:

- `Rescaling(1/255)`: a normalização fica dentro do modelo, então a inferência recebe `uint8` de 0 a 255 sem pré-processamento externo.
- `Conv2D(filtros, 3, padding="same", activation="relu")`: 16 ou 32 kernels 3x3 aprendidos; `padding="same"` preserva o tamanho espacial; ReLU zera valores negativos.
- `MaxPooling2D(2)`: reduz altura e largura pela metade pegando o máximo de cada janela 2x2, o que dá invariância a pequenos deslocamentos.
- `Flatten`: transforma os mapas 16x16x32 em um vetor.
- `Dense(64, relu)`: o vetor de features usado no Item B.
- `Dropout(0.15)`: desliga 15% das unidades em cada passo de treino, como regularização.
- `Dense(3, softmax)`: probabilidades das três classes.

Dar nome às camadas (`name=`) é o que permite depois recuperá-las com `model.get_layer("features_dense")`.

Onde aparece: `utils.py`, `04_criar_modelo_cnn.py`.

### 3.6 Compilação: otimizador, loss e métrica

`model.compile` define como a rede aprende. O otimizador Adam com taxa de aprendizado 1e-3 ajusta os pesos; a loss `sparse_categorical_crossentropy` mede o erro entre a distribuição softmax e o rótulo inteiro (0, 1 ou 2), sem precisar de one-hot; a métrica `accuracy` acompanha a fração de acertos.

```python
# utils.py, função build_cnn
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
```

O script `04_criar_modelo_cnn.py` imprime `model.summary()` capturando as linhas com `print_fn=lines.append`, grava o resumo em `saidas/04_resumo_modelo.txt` e mostra `model.input_shape`, `model.output_shape` e o shape da saída de `features_dense`.

Onde aparece: `utils.py`, `04_criar_modelo_cnn.py`.

### 3.7 Carregamento de dataset a partir de pastas com tf.data

O Keras lê um dataset organizado em `pasta/split/classe/*.png` com `keras.utils.image_dataset_from_directory`, que infere os rótulos pelo nome da subpasta. O resultado é um `tf.data.Dataset` de lotes (imagens, rótulos).

```python
# utils.py, função load_datasets
common = dict(
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=batch_size,
)
train = keras.utils.image_dataset_from_directory(
    DATASET / "train", shuffle=True, seed=SEED, **common
)
val = keras.utils.image_dataset_from_directory(DATASET / "val", shuffle=False, **common)
test = keras.utils.image_dataset_from_directory(DATASET / "test", shuffle=False, **common)
autotune = tf.data.AUTOTUNE
return train.prefetch(autotune), val.prefetch(autotune), test.prefetch(autotune)
```

Parâmetros que importam: `label_mode="int"` entrega o índice da classe (combina com a loss sparse); `class_names` fixa a ordem círculo, quadrado, triângulo em vez da ordem alfabética implícita; `shuffle=True` só no treino, para que validação e teste mantenham a ordem e os rótulos possam ser concatenados depois; `prefetch(AUTOTUNE)` prepara o próximo lote enquanto a GPU ou CPU processa o atual.

Onde aparece: `utils.py`, `05_item_a_treinar_cnn.py`, `06_avaliar_modelo.py`.

### 3.8 Treinamento com model.fit e callbacks

`model.fit` executa o laço de treino: para cada época percorre os lotes de treino, atualiza os pesos e, ao final, avalia no conjunto de validação. O objeto `History` devolvido guarda `accuracy`, `loss`, `val_accuracy` e `val_loss` por época. O Item A exige no mínimo 10 épocas, e o script recusa valores menores.

```python
# 05_item_a_treinar_cnn.py
history_obj = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    verbose=2,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        )
    ],
)
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
model.save(MODELS / "cnn_geometrica.keras")
```

O callback `ReduceLROnPlateau` observa `val_loss` e, quando ela não melhora por 2 épocas (`patience`), multiplica a taxa de aprendizado por 0,5 (`factor`), sem descer de 1e-5. Depois do treino, `model.evaluate` mede loss e acurácia no teste, e `model.save` grava o modelo no formato `.keras`, que os scripts seguintes reabrem com `keras.models.load_model` (função `load_trained_model` em `utils.py`). O tempo de treino é medido com `time.perf_counter()`.

Onde aparece: `05_item_a_treinar_cnn.py`, `utils.py`.

### 3.9 Curvas de treino e diagnóstico de overfitting

Overfitting acontece quando a rede continua melhorando no treino enquanto piora na validação: ela está decorando exemplos em vez de generalizar. O professor plota acurácia e loss de treino e validação lado a lado com Matplotlib e aplica um critério numérico simples em `overfitting_diagnosis`.

```python
# utils.py, função overfitting_diagnosis
gap = float(train_acc[-1] - val_acc[-1])
best_val_loss_epoch = int(np.argmin(val_loss) + 1)
loss_rising = bool(
    val_loss[-1] > 1.10 * val_loss.min()
    and best_val_loss_epoch <= len(val_loss) - 2
)
has_signal = bool(gap > 0.08 or loss_rising)
```

Há indício de overfitting se a diferença final de acurácia entre treino e validação passa de 8 pontos percentuais, ou se a loss de validação final está mais de 10% acima do seu mínimo e o mínimo ocorreu pelo menos duas épocas antes do fim. O docstring do script frisa que isso é indício, não prova. O resultado vai para `saidas/05_resultado_treino.json`, junto com o histórico em `saidas/05_historico.json` e o gráfico `saidas/05_curvas_treino.png`. Sugestões do professor quando há sinal: mais dados, data augmentation, dropout ou regularização L2.

Onde aparece: `05_item_a_treinar_cnn.py`, `utils.py`.

### 3.10 Avaliação: matriz de confusão e relatório de classificação

A acurácia global esconde onde a rede erra. A matriz de confusão cruza classe real (linhas) com predição (colunas), e o relatório do scikit-learn traz precisão, revocação e F1 por classe. As previsões vêm de `model.predict`, que devolve as probabilidades softmax; a classe prevista é o `argmax` de cada linha.

```python
# 06_avaliar_modelo.py
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probabilities = model.predict(test_ds, verbose=0)
y_pred = probabilities.argmax(axis=1)
matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
```

Extrair `y_true` iterando o `tf.data.Dataset` só funciona porque o teste foi carregado com `shuffle=False`. A matriz é desenhada com `ax.imshow(matrix, cmap="Blues")` e os números escritos célula a célula com `ax.text`. Saídas: `saidas/06_matriz_confusao.png` e `saidas/06_relatorio_classificacao.json`.

Onde aparece: `06_avaliar_modelo.py`.

### 3.11 Filtros aprendidos e mapas de ativação

Os kernels da primeira convolução podem ser lidos direto dos pesos com `get_weights()[0]`, um tensor 3x3x3x16 (altura, largura, canais de entrada, filtros). O script tira a média sobre os canais de entrada para exibir cada filtro como uma imagem 3x3 com colormap `coolwarm`. Para ver as ativações, cria-se um modelo intermediário cuja saída é a de uma camada interna.

```python
# 07_visualizar_filtros_ativacoes.py
weights = model.get_layer("conv1a").get_weights()[0]  # 3×3×3×16
kernel = weights[:, :, :, index].mean(axis=2)

activation_model = keras.Model(model.input, model.get_layer("conv2a").output)
activations = activation_model.predict(batch, verbose=0)[0]
ax.imshow(activations[:, :, index], cmap="viridis")
```

`keras.Model(model.input, camada.output)` reaproveita os pesos já treinados e devolve o tensor da camada `conv2a` para a imagem de entrada. `batch = rgb[None, ...]` acrescenta a dimensão de lote, porque o modelo sempre espera um lote. Os 16 primeiros mapas são gravados em `saidas/07_mapas_ativacao.png` e os filtros em `saidas/07_filtros_conv1.png`.

Onde aparece: `07_visualizar_filtros_ativacoes.py`.

### 3.12 Seleção fixa de 20 imagens

O Item B e os scripts seguintes trabalham sobre exatamente 20 imagens de teste: 7 círculos, 7 quadrados e 6 triângulos, sempre as primeiras de cada pasta em ordem alfabética. A função `select_twenty` grava essa lista em `saidas/selecao_20_imagens.json` e `read_selection` a reaproveita, de modo que CNN, ORB e benchmark comparem as mesmas imagens.

```python
# utils.py, função select_twenty
counts = {"circulo": 7, "quadrado": 7, "triangulo": 6}
for label, class_name in enumerate(CLASS_NAMES):
    paths = sorted((DATASET / "test" / class_name).glob("*.png"))
    for path in paths[: counts[class_name]]:
        selected.append({"path": str(path.relative_to(PROJECT)), "classe": class_name, "label": label})
```

Onde aparece: `utils.py`, `07_visualizar_filtros_ativacoes.py`, `08_item_b_features_pca.py`, `09_orb_nas_mesmas_imagens.py`, `10_comparar_cnn_orb.py`, `11_classificar_uma_imagem.py`.

### 3.13 CNN como extrator de features

Uma rede treinada para classificar aprendeu, nas camadas internas, uma representação numérica das imagens. Cortando o classificador final e ficando com a saída de `features_dense`, obtém-se um vetor de 64 valores por imagem: são as features da CNN. A técnica é a mesma da seção anterior, um `keras.Model` cuja saída é uma camada interna.

```python
# 08_item_b_features_pca.py
extractor = keras.Model(model.input, model.get_layer("features_dense").output)
features = extractor.predict(images, verbose=0)   # shape (20, 64)
```

Os vetores completos são gravados em `saidas/08_features_cnn.csv`, com uma coluna por feature (`f00` a `f63`), para serem reutilizados no script 10.

Onde aparece: `08_item_b_features_pca.py`.

### 3.14 Padronização, PCA e silhouette

Vetores de 64 dimensões não cabem em um gráfico. O PCA (análise de componentes principais) projeta os dados nas duas direções de maior variância, permitindo um scatter plot 2D. Antes do PCA, `StandardScaler` coloca cada feature com média 0 e desvio 1, para que nenhuma domine pela escala. O `silhouette_score`, calculado no espaço original de 64 dimensões, mede quão compacta e separada está cada classe: vai de -1 a 1, e quanto maior, melhor.

```python
# 08_item_b_features_pca.py
scaled = StandardScaler().fit_transform(features)
pca = PCA(n_components=2, random_state=2026)
points = pca.fit_transform(scaled)
silhouette = float(silhouette_score(scaled, labels))
ax.set(xlabel=f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% da variância)")
```

`explained_variance_ratio_` diz que fração da variância cada componente conserva; a soma das duas aparece no JSON. O docstring do script alerta: sobreposição no PCA não prova que as classes sejam inseparáveis no espaço original, pois é apenas uma projeção. Saídas: `saidas/08_pca_features_cnn.png` e `saidas/08_resultado_pca_cnn.json`.

Onde aparece: `08_item_b_features_pca.py`, `10_comparar_cnn_orb.py`.

### 3.15 Keypoints e descritores ORB

ORB (Oriented FAST and Rotated BRIEF) é um detector e descritor clássico do OpenCV, rápido e sem treinamento. Ele encontra pontos de interesse (keypoints) e descreve a vizinhança de cada um com um vetor binário de 32 bytes (256 bits). O número de keypoints varia de imagem para imagem.

```python
# 09_orb_nas_mesmas_imagens.py
orb = cv2.ORB_create(nfeatures=180, edgeThreshold=7, patchSize=17, fastThreshold=6)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
keypoints, descriptors = orb.detectAndCompute(gray, None)
for keypoint in keypoints:
    point = tuple(np.rint(keypoint.pt).astype(int))
    cv2.circle(visual, point, 1, (0, 230, 255), -1, cv2.LINE_AA)
```

Parâmetros de `cv2.ORB_create`: `nfeatures` limita a 180 pontos; `edgeThreshold` e `patchSize` foram reduzidos (7 e 17) porque a imagem tem apenas 64 pixels de lado; `fastThreshold=6` torna o detector FAST mais sensível em imagens de baixo contraste. O professor desenha os pontos com raio 1 em vez de `DRAW_RICH_KEYPOINTS` para não poluir a montagem. Saídas: `saidas/09_orb_keypoints_20_imagens.jpg` e `saidas/09_orb_estatisticas.csv`.

Onde aparece: `09_orb_nas_mesmas_imagens.py`, `10_comparar_cnn_orb.py`, `12_benchmark_embarcado.py`.

### 3.16 Bag of Visual Words com k-means

Para comparar ORB com a CNN é preciso um vetor de tamanho fixo por imagem, e o ORB entrega uma quantidade variável de descritores. A solução é o Bag of Visual Words (BoVW): agrupa-se os descritores de treino em 24 clusters com k-means (as "palavras visuais") e cada imagem passa a ser representada pelo histograma normalizado de quantas vezes cada palavra aparece.

```python
# 10_comparar_cnn_orb.py
kmeans = MiniBatchKMeans(
    n_clusters=VOCABULARY_SIZE, random_state=2026, batch_size=2048,
    n_init=5, max_iter=120,
).fit(all_descriptors)

def bovw(path):
    desc = descriptors(path)
    histogram = np.zeros(VOCABULARY_SIZE, np.float32)
    if desc is not None and len(desc):
        words = kmeans.predict(desc.astype(np.float32))
        histogram = np.bincount(words, minlength=VOCABULARY_SIZE).astype(np.float32)
        histogram /= histogram.sum()
    return histogram
```

O vocabulário é aprendido só com as 45 primeiras imagens de treino de cada classe, nunca com as de teste, para não vazar informação. `MiniBatchKMeans` é a versão do k-means que processa lotes e converge mais rápido. Os descritores binários são convertidos para `float32` porque o k-means opera com distância euclidiana.

Onde aparece: `10_comparar_cnn_orb.py`.

### 3.17 Comparação CNN versus ORB nas mesmas imagens

O script 10 aplica o mesmo tratamento aos dois espaços (padronização, PCA 2D, silhouette em todas as dimensões) e desenha os dois scatter plots lado a lado. Os vetores da CNN são lidos de volta do CSV do script 08, garantindo que sejam exatamente os mesmos.

```python
# 10_comparar_cnn_orb.py
with (OUTPUTS / "08_features_cnn.csv").open(encoding="utf-8") as f:
    cnn_rows = list(csv.DictReader(f))
cnn_features = np.asarray([[float(row[f"f{i:02d}"]) for i in range(64)] for row in cnn_rows])
cnn_silhouette = float(silhouette_score(StandardScaler().fit_transform(cnn_features), labels))
result["melhor_separacao_neste_ensaio"] = "CNN" if cnn_silhouette >= orb_silhouette else "ORB+BoVW"
```

A interpretação esperada, segundo o docstring: a CNN aprende descritores supervisionados, otimizados para separar as três classes; o ORB descreve padrões locais genéricos sem conhecer os rótulos. Silhouette e PCA ajudam a comparar a organização dos espaços, mas não substituem a avaliação de classificação. Saídas: `saidas/10_comparacao_cnn_orb.png`, `saidas/10_features_orb_bovw.csv`, `saidas/10_resultado_comparacao.json`.

Onde aparece: `10_comparar_cnn_orb.py`.

### 3.18 Inferência em uma imagem isolada

Classificar uma única imagem exige acrescentar a dimensão de lote (`rgb[None, ...]`), chamar `model.predict` e ler as três probabilidades softmax. A classe prevista é o índice de maior probabilidade. Como a normalização está dentro do modelo (camada `Rescaling`), a imagem entra em `uint8` sem tratamento.

```python
# 11_classificar_uma_imagem.py
rgb = load_rgb(path)
model = load_trained_model()
probabilities = model.predict(rgb[None, ...], verbose=0)[0]
prediction = int(np.argmax(probabilities))
for class_name, probability in zip(CLASS_NAMES, probabilities):
    print(f"  {class_name:<10}: {100*probability:6.2f}%")
```

O script aceita `--imagem caminho` e, sem argumento, usa a primeira imagem da seleção de 20. Grava uma versão ampliada para 512x512 com a predição escrita no topo em `saidas/11_predicao.png`.

Onde aparece: `11_classificar_uma_imagem.py`.

### 3.19 Benchmark de latência para sistemas embarcados

Em um robô, acurácia não é o único critério: latência, memória, consumo de energia e estabilidade temporal pesam na escolha do método. O script mede o tempo médio por imagem da CNN com `batch_size=1` (o cenário de tempo real, uma imagem por vez) e da extração ORB, sobre 120 imagens de teste. Antes de medir, faz um aquecimento com 8 imagens para não contabilizar o custo de inicialização do TensorFlow.

```python
# 12_benchmark_embarcado.py
model.predict(images[:8], verbose=0)          # aquecimento
start = time.perf_counter()
model.predict(images, batch_size=1, verbose=0)
cnn_ms = 1000 * (time.perf_counter() - start) / len(images)

start = time.perf_counter()
for rgb in images:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    keypoints, _ = orb.detectAndCompute(gray, None)
orb_ms = 1000 * (time.perf_counter() - start) / len(images)
```

O resultado vai para `saidas/12_benchmark.json` com a observação de que os números dependem do hardware, threads, versão das bibliotecas e aquecimento. A recomendação final do professor: repita o benchmark no hardware-alvo.

Onde aparece: `12_benchmark_embarcado.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `utils.py` | Define caminhos (`dataset/`, `saidas/`, `modelos/`, `recursos/`), classes, sementes, `build_cnn`, carregamento de datasets, seleção de 20 imagens e diagnóstico de overfitting | Nenhuma | Nenhuma direta | Nada; é importado por todos |
| `00_validar_ambiente.py` | Imprime versões de Python, TensorFlow, Keras, NumPy, OpenCV, Matplotlib, scikit-learn e GPUs visíveis | Nenhuma | Só terminal | Nada |
| `01_gerar_dataset.py` | Gera 1620 imagens 64x64 de círculo, quadrado e triângulo em train/val/test, com variações | Nenhuma | `dataset/{train,val,test}/{classe}/*.png`, `dataset/dataset_info.json`, `recursos/amostras_dataset.png` | Nada. Apaga `dataset/` se já existir |
| `02_imagem_como_tensor.py` | Mostra shape, dtype, intervalo e pixel central de uma imagem; amplia um patch 8x8 | Primeira imagem de `dataset/train/circulo` | `saidas/02_patch_pixels.png` | 01 |
| `03_convolucao_passo_a_passo.py` | Aplica Sobel vertical, Sobel horizontal e blur com `cv2.filter2D` | Primeira imagem de `dataset/test/quadrado` | `saidas/03_convolucoes.png` | 01 |
| `04_criar_modelo_cnn.py` | Constrói a CNN e imprime o resumo sem treinar | Nenhuma | `saidas/04_resumo_modelo.txt` | Nada |
| `05_item_a_treinar_cnn.py` | Treina a CNN por N épocas (mínimo 10), avalia no teste, plota curvas e diagnostica overfitting | Dataset completo; `--epochs`, `--batch-size` | `modelos/cnn_geometrica.keras`, `saidas/05_curvas_treino.png`, `saidas/05_historico.json`, `saidas/05_resultado_treino.json` | 01 |
| `06_avaliar_modelo.py` | Matriz de confusão e relatório de classificação no teste | Modelo treinado, `dataset/test` | `saidas/06_matriz_confusao.png`, `saidas/06_relatorio_classificacao.json` | 01, 05 |
| `07_visualizar_filtros_ativacoes.py` | Plota os 16 filtros de `conv1a` e 16 mapas de ativação de `conv2a` | Modelo treinado, primeira imagem da seleção de 20 | `saidas/07_filtros_conv1.png`, `saidas/07_mapas_ativacao.png`, `saidas/selecao_20_imagens.json` (se ainda não existir) | 01, 05 |
| `08_item_b_features_pca.py` | Extrai 64 features de `features_dense` para 20 imagens, padroniza, PCA 2D, silhouette | Modelo treinado, `dataset/test` | `saidas/selecao_20_imagens.json`, `saidas/08_pca_features_cnn.png`, `saidas/08_features_cnn.csv`, `saidas/08_resultado_pca_cnn.json` | 01, 05 |
| `09_orb_nas_mesmas_imagens.py` | Detecta keypoints ORB nas mesmas 20 imagens e monta grade visual | Seleção de 20 imagens | `saidas/09_orb_keypoints_20_imagens.jpg`, `saidas/09_orb_estatisticas.csv` | 01, 08 (ou 07) |
| `10_comparar_cnn_orb.py` | Constrói BoVW com ORB e k-means, compara com features CNN via PCA e silhouette | `dataset/train` (vocabulário), seleção de 20, `saidas/08_features_cnn.csv` | `saidas/10_comparacao_cnn_orb.png`, `saidas/10_features_orb_bovw.csv`, `saidas/10_resultado_comparacao.json` | 01, 08 |
| `11_classificar_uma_imagem.py` | Classifica uma imagem e imprime as probabilidades softmax | `--imagem` opcional; senão a primeira da seleção | `saidas/11_predicao.png` | 01, 05 |
| `12_benchmark_embarcado.py` | Mede ms por imagem da CNN (batch 1) e do ORB em 120 imagens de teste | Modelo treinado, `dataset/test` | `saidas/12_benchmark.json` | 01, 05 |

## 5. Passo a passo para executar

Todos os comandos abaixo devem ser executados de dentro de `aula_08_TensorFlow_Keras/exemplos_em_python/`. Os resultados aparecem nas pastas irmãs `../dataset/`, `../modelos/`, `../recursos/` e `../saidas/`.

```powershell
cd C:\visao_computacional_PVA\aula_08_TensorFlow_Keras\exemplos_em_python
```

Passo 1. Confira o ambiente. Espere ver as versões de cada biblioteca e uma lista de GPUs (vazia é normal).

```powershell
python 00_validar_ambiente.py
```

Passo 2. Gere o dataset. Leva alguns segundos. Abra `../recursos/amostras_dataset.png` para ver 18 amostras.

```powershell
python 01_gerar_dataset.py
```

Passo 3. Entenda a imagem como tensor. O terminal mostra `(64, 64, 3)`, `uint8` e o pixel central. Abra `../saidas/02_patch_pixels.png`.

```powershell
python 02_imagem_como_tensor.py
```

Passo 4. Veja convoluções com kernels fixos. O terminal imprime as matrizes dos kernels; a montagem com quatro painéis fica em `../saidas/03_convolucoes.png`.

```powershell
python 03_convolucao_passo_a_passo.py
```

Passo 5. Monte a CNN e leia o resumo. Observe como o shape diminui em cada `MaxPooling2D` e o número de parâmetros de cada camada.

```powershell
python 04_criar_modelo_cnn.py
```

Passo 6. Treine (Item A). Padrão de 12 épocas com lote de 64. Valores abaixo de 10 épocas são rejeitados. Em CPU leva de um a poucos minutos. Ao final, o terminal imprime acurácia de teste, gap treino-validação e o comentário de overfitting.

```powershell
python 05_item_a_treinar_cnn.py
python 05_item_a_treinar_cnn.py --epochs 20 --batch-size 32
```

Passo 7. Avalie. Abra `../saidas/06_matriz_confusao.png` e o JSON do relatório.

```powershell
python 06_avaliar_modelo.py
```

Passo 8. Visualize filtros e ativações.

```powershell
python 07_visualizar_filtros_ativacoes.py
```

Passo 9. Extraia features e faça o PCA (Item B). O terminal informa silhouette em 64D e variância explicada por PC1 e PC2.

```powershell
python 08_item_b_features_pca.py
```

Passo 10. ORB nas mesmas 20 imagens e comparação com a CNN. O script 10 precisa do CSV gerado pelo 08.

```powershell
python 09_orb_nas_mesmas_imagens.py
python 10_comparar_cnn_orb.py
```

Passo 11. Classifique uma imagem. Sem argumento usa a primeira da seleção; caminhos relativos são resolvidos a partir de `aula_08_TensorFlow_Keras/`.

```powershell
python 11_classificar_uma_imagem.py
python 11_classificar_uma_imagem.py --imagem dataset/test/triangulo/triangulo_0003.png
```

Passo 12. Benchmark de latência.

```powershell
python 12_benchmark_embarcado.py
```

Nenhum script abre janela; tudo é gravado em disco ou impresso no terminal. Não há teclas de atalho.

## 6. Exercícios e itens da aula

Os scripts referenciam dois itens de um exercício:

**Item A: CNN do zero treinada por pelo menos 10 épocas.** Resolvido por `05_item_a_treinar_cnn.py`, com a arquitetura em `build_cnn` de `utils.py`. O docstring de `build_cnn` diz que Flatten "atende ao requisito", o que indica que o enunciado exige camadas Conv2D, MaxPooling2D, Flatten e Dense. Entregáveis: modelo salvo, curvas de acurácia e loss de treino e validação, acurácia de teste e comentário sobre overfitting. Os scripts 04, 06 e 07 dão apoio: resumo da arquitetura, matriz de confusão e visualização dos filtros.

**Item B: features da CNN e comparação com ORB.** Resolvido por `08_item_b_features_pca.py` e `10_comparar_cnn_orb.py`. O enunciado, conforme os docstrings, pede: extrair o vetor da penúltima Dense (64 valores) de 20 imagens, reduzir a 2D com PCA, plotar um scatter colorido por classe e discutir a separabilidade; depois extrair ORB das mesmas 20 imagens (`09_orb_nas_mesmas_imagens.py`), transformar em vetor fixo com BoVW e comparar os dois espaços com silhouette e PCA. Os scripts 11 e 12 complementam com inferência unitária e benchmark de latência.

Exercícios adicionais para praticar:

1. Rode o treino com `--epochs 30` e veja se o diagnóstico de overfitting muda. Compare `05_curvas_treino.png` das duas execuções.
2. Em `utils.py`, altere `Dropout(0.15)` para 0,5 e retreine. Observe o gap treino-validação.
3. Mude `VOCABULARY_SIZE` em `10_comparar_cnn_orb.py` para 8 e para 64 e veja o efeito no silhouette do ORB.
4. Reduza `SPLITS["train"]` em `01_gerar_dataset.py` para 60 por classe, regenere e retreine. A rede ainda generaliza?

## 7. Armadilhas e dicas

- **Rode os scripts de dentro de `exemplos_em_python/`.** Eles fazem `from utils import ...`; de outra pasta o import falha.
- **Mensagens de erro citam a pasta `exemplos/`.** As funções `require_dataset` e `load_trained_model` mandam executar `python exemplos/01_gerar_dataset.py` e `python exemplos/05_item_a_treinar_cnn.py`. Esse era o nome da pasta do professor; aqui ela se chama `exemplos_em_python/`. Foi mantido por fidelidade ao original.
- **`01_gerar_dataset.py` apaga `dataset/` inteira** antes de regenerar. Se você tiver colocado imagens próprias lá, faça backup.
- **Ordem obrigatória:** 01 antes de qualquer outro; 05 antes de 06, 07, 08, 11 e 12; 08 antes de 10 (o script 10 lê `saidas/08_features_cnn.csv` e falha sem ele).
- **`selecao_20_imagens.json` é criado por quem rodar primeiro** (07, 09, 10 ou 11 via `read_selection`, ou 08 via `select_twenty`). O script 08 sempre refaz a seleção. Se você regenerar o dataset, apague `saidas/` para evitar uma seleção antiga.
- **Épocas abaixo de 10 são rejeitadas** pelo script 05 com `ValueError`.
- **Caminho de `--imagem` no script 11.** Caminhos relativos são unidos a `aula_08_TensorFlow_Keras/`, não ao diretório atual. Caminhos absolutos do Windows (`C:\...`) funcionam porque `pathlib` descarta o prefixo quando o segundo operando é absoluto.
- **Tempo de treino em CPU.** Doze épocas com 1080 imagens de 64x64 são rápidas, mas `enable_op_determinism` pode deixar tudo mais lento. Se precisar de velocidade, comente a chamada em `configure`, aceitando perder a reprodutibilidade.
- **Python 3.14.** O TensorFlow pode ainda não ter wheel para a versão instalada na máquina. Se `pip install tensorflow` falhar, crie um venv com Python 3.12, que foi a versão usada pelo professor em aulas vizinhas.
- **ORB em imagens de 64 pixels.** Os parâmetros `edgeThreshold=7` e `patchSize=17` foram escolhidos para imagens pequenas. Com os padrões do OpenCV (31 e 31) quase nenhum keypoint seria detectado. Se `descriptors` vier `None`, o script trata como zero keypoints.
- **Silhouette em 20 amostras é instável.** Uma imagem mal posicionada muda bastante o valor. Leia sempre junto com o scatter e a matriz de confusão, como os docstrings recomendam.
- **Parâmetros que valem a pena variar:** `--epochs` e `--batch-size` no 05; `learning_rate` e `Dropout` em `build_cnn`; `factor` e `patience` do `ReduceLROnPlateau`; `VOCABULARY_SIZE` no 10; `nfeatures` e `fastThreshold` do ORB.

## 8. Glossário

| Termo | Significado |
|---|---|
| Adam | Otimizador que adapta a taxa de aprendizado por parâmetro usando médias móveis do gradiente. |
| API funcional do Keras | Estilo de construção de modelo em que cada camada é chamada como função sobre o tensor anterior, permitindo ramificações e reaproveitamento de camadas internas. |
| Ativação (mapa de) | Saída de uma camada convolucional para uma imagem: um conjunto de imagens 2D, uma por filtro. |
| BGR / RGB | Ordem dos canais de cor. OpenCV usa BGR; Keras e Matplotlib esperam RGB. |
| BoVW (Bag of Visual Words) | Representação de imagem por histograma de "palavras visuais", que são clusters de descritores locais obtidos com k-means. |
| Batch (lote) | Grupo de imagens processado de uma vez pela rede. A primeira dimensão do tensor de entrada. |
| Callback | Objeto que o `model.fit` chama em momentos do treino, como o fim de cada época. |
| CNN | Rede neural convolucional, composta por camadas de convolução, pooling e densas. |
| Conv2D | Camada que aplica vários kernels aprendidos sobre a imagem. |
| Convolução | Soma ponderada da vizinhança de cada pixel por um kernel deslizante. |
| Dense | Camada totalmente conectada: cada saída depende de todas as entradas. |
| Descritor | Vetor numérico que descreve a vizinhança de um keypoint. No ORB, 32 bytes. |
| Dropout | Regularização que zera aleatoriamente uma fração das unidades durante o treino. |
| Entropia cruzada (crossentropy) | Loss que mede a distância entre a distribuição prevista e o rótulo verdadeiro. A versão sparse aceita rótulos inteiros. |
| Época (epoch) | Uma passagem completa pelo conjunto de treino. |
| F1 | Média harmônica entre precisão e revocação, reportada por classe no relatório de classificação. |
| FAST | Detector de cantos usado pelo ORB para escolher keypoints. |
| Feature | Valor numérico que representa alguma propriedade da imagem. Aqui, as 64 saídas de `features_dense`. |
| Flatten | Camada que transforma um tensor 3D em vetor 1D. |
| Gap treino-validação | Diferença entre a acurácia de treino e a de validação; um gap grande sugere overfitting. |
| Kernel | Matriz pequena (aqui 3x3) usada na convolução. |
| Keypoint | Ponto de interesse detectado na imagem, com posição, escala e orientação. |
| k-means | Algoritmo de agrupamento que particiona vetores em k clusters. `MiniBatchKMeans` é sua versão por lotes. |
| Latência | Tempo por imagem de uma etapa de processamento, medido aqui em milissegundos. |
| Loss | Função de perda minimizada durante o treino. |
| MaxPooling2D | Camada que reduz a resolução tomando o máximo de cada janela. |
| ORB | Oriented FAST and Rotated BRIEF, detector e descritor binário rápido do OpenCV. |
| Overfitting | Quando o modelo decora o treino e piora na validação. |
| Padding "same" | Preenchimento de bordas que mantém o tamanho espacial após a convolução. |
| PCA | Análise de componentes principais, projeção linear nas direções de maior variância. |
| Precisão e revocação | Fração de acertos entre as predições de uma classe, e fração de exemplos da classe que foram encontrados. |
| ReduceLROnPlateau | Callback que reduz a taxa de aprendizado quando a métrica monitorada para de melhorar. |
| ReLU | Função de ativação que devolve zero para valores negativos e o próprio valor para positivos. |
| Rescaling | Camada Keras que multiplica a entrada por um fator, aqui 1/255. |
| Semente (seed) | Valor inicial dos geradores aleatórios; fixá-la torna a execução reproduzível. |
| Silhouette | Medida de -1 a 1 de quão compactos e separados são os grupos de um agrupamento. |
| Sobel | Kernel 3x3 que realça bordas verticais ou horizontais. |
| Softmax | Função que transforma as saídas finais em probabilidades que somam 1. |
| Split | Partição do dataset: train, val e test. |
| StandardScaler | Transformação do scikit-learn que deixa cada feature com média 0 e desvio 1. |
| Taxa de aprendizado (learning rate) | Tamanho do passo de atualização dos pesos a cada iteração. |
| Tensor | Arranjo multidimensional de números; uma imagem RGB é um tensor altura x largura x 3. |
| tf.data | API do TensorFlow para pipelines de dados eficientes; `prefetch` e `AUTOTUNE` fazem parte dela. |
| Validação | Conjunto usado durante o treino para acompanhar a generalização sem tocar no teste. |
