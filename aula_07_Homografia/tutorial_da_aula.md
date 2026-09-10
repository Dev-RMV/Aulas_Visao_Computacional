# Aula 07 – Homografia e descritores de pontos-chave

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula ensina o caminho completo que vai de duas fotos de uma mesma cena plana até uma transformação geométrica que alinha uma sobre a outra. Esse caminho tem quatro etapas: detectar pontos-chave (keypoints) e calcular descritores com SIFT, ORB e AKAZE; casar descritores entre as duas imagens com BFMatcher ou FLANN; filtrar os casamentos ruins com cross-check ou com o teste da razão de Lowe; e estimar a homografia com RANSAC, usando a proporção de inliers e o erro de reprojeção como medida de confiança. O professor fecha a aula aplicando esse pipeline frame a frame em um vídeo, o que é a base da localização visual em robótica.

O fio condutor é um par de imagens sintéticas gerado pelo próprio script `01_gerar_imagens_teste.py`: uma cena de referência cheia de cantos e uma versão dela deformada por perspectiva, escurecida, com ruído e com uma pequena oclusão. Como a homografia usada para deformar é conhecida e salva em JSON, cada script seguinte pode medir o quanto a homografia estimada se afasta da verdadeira. Os scripts 02 a 04 extraem keypoints com cada método, o 05 compara os três (Item A), os scripts 06 a 08 fazem matching e homografia, o 09 integra tudo e alinha as imagens (Item B), e os scripts 10 a 12 exploram comparação entre métodos, robustez a rotação, escala e luz, e localização em vídeo.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas: `cv2` (OpenCV) e `numpy`. Tudo o mais é biblioteca padrão (`argparse`, `csv`, `json`, `time`, `pathlib`).

```powershell
pip install opencv-python numpy
```

SIFT faz parte do pacote `opencv-python` comum desde a versão 4.4, quando a patente expirou. Não é preciso `opencv-contrib-python` nesta aula. O script `00_validar_ambiente.py` confirma se `SIFT_create`, `ORB_create`, `AKAZE_create` e `findHomography` existem na instalação.

Pasta de execução: rode os scripts de dentro de `aula_07_Homografia/exemplos_em_python/`. O arquivo `utils.py` define a raiz do projeto como a pasta acima da dos scripts, ou seja, `aula_07_Homografia/`. As pastas `recursos/` (entradas) e `saidas/` (resultados) são criadas ali, e não dentro de `exemplos_em_python/`.

```
aula_07_Homografia/
├── exemplos_em_python/   scripts e utils.py
├── recursos/             criado pelo 01: cena_referencia.png, cena_transformada.png, par_imagens.jpg, homografia_real.json, video_localizacao.mp4
└── saidas/               criado pelos scripts 02 a 12: jpg, png, csv e mp4 de resultado
```

Material faltante: nenhum. Segundo `achados_dos_zips.md`, a aula é autossuficiente, desde que `01_gerar_imagens_teste.py` seja executado primeiro.

Webcam: só o script `12_localizacao_visual_video.py` pode usar webcam, se for chamado com `--source 0`. Por padrão ele lê o vídeo sintético gerado pelo 01.

Versão de Python usada pelo professor: não há registro nesta aula. O código usa `from __future__ import annotations` e anotações de tipo `tuple[...]`, o que pede Python 3.9 ou mais novo.

## 3. Conceitos-chave

### 3.1 Validação do ambiente por introspecção

Antes de rodar uma aula que depende de algoritmos específicos, vale conferir se a instalação do OpenCV os expõe. O professor faz isso com `hasattr(cv2, nome)`, que testa se o módulo tem o atributo, sem precisar chamar a função. Se qualquer um dos três detectores faltar, o script lança `RuntimeError` e interrompe.

```python
# 00_validar_ambiente.py
print(f"SIFT disponível: {hasattr(cv2, 'SIFT_create')}")
print(f"ORB disponível: {hasattr(cv2, 'ORB_create')}")
print(f"AKAZE disponível: {hasattr(cv2, 'AKAZE_create')}")
print(f"findHomography disponível: {hasattr(cv2, 'findHomography')}")

if not all(hasattr(cv2, name) for name in ("SIFT_create", "ORB_create", "AKAZE_create")):
    raise RuntimeError("Instalação do OpenCV incompleta para esta aula.")
```

Onde aparece: `00_validar_ambiente.py`.

### 3.2 Layout de projeto com pathlib

Todos os scripts importam de `utils.py` três constantes de caminho. `PROJECT` é calculado a partir do próprio arquivo `utils.py`: `Path(__file__).resolve()` dá o caminho absoluto, e `.parents[1]` sobe dois níveis (o índice 0 seria a pasta do script). Assim o projeto funciona independentemente do diretório de trabalho atual, desde que o import de `utils` resolva, e por isso é preciso rodar de dentro de `exemplos_em_python/`.

A função `ensure_outputs()` cria `saidas/` se necessário e devolve o caminho, e `load_pair()` carrega as duas imagens da cena, com uma mensagem clara caso o script 01 ainda não tenha sido executado. Como `cv2.imread` devolve `None` em vez de lançar exceção quando o arquivo não existe, esse teste explícito é obrigatório.

```python
# utils.py
PROJECT = Path(__file__).resolve().parents[1]
RESOURCES = PROJECT / "recursos"
OUTPUTS = PROJECT / "saidas"

def load_pair() -> tuple[np.ndarray, np.ndarray]:
    image_a = cv2.imread(str(RESOURCES / "cena_referencia.png"))
    image_b = cv2.imread(str(RESOURCES / "cena_transformada.png"))
    if image_a is None or image_b is None:
        raise FileNotFoundError("Execute 01_gerar_imagens_teste.py para criar o par.")
    return image_a, image_b
```

Onde aparece: `utils.py` e todos os scripts de 01 a 12.

### 3.3 Cena sintética rica em cantos

Detectores de pontos-chave procuram regiões com variação forte de intensidade em duas direções, como cantos e blobs. Uma parede lisa não gera keypoints. Por isso o professor desenha, com primitivas do OpenCV, uma cena de referência de 960 por 720 pixels com um tabuleiro de xadrez (muitos cantos regulares), três padrões de localização no estilo dos códigos QR (`finder_pattern`), círculos, um triângulo, textos e uma constelação de 95 cruzes e pontinhos posicionados por um gerador aleatório com semente fixa. O comentário do código explica a intenção: regiões assimétricas evitam ambiguidades globais, isto é, evitam que um pedaço da imagem pareça igual a outro.

A semente `np.random.default_rng(42)` garante que a cena seja idêntica a cada execução, o que torna os resultados reproduzíveis. As funções de desenho usadas são `cv2.rectangle`, `cv2.circle`, `cv2.line`, `cv2.fillConvexPoly`, `cv2.polylines`, `cv2.putText` e `cv2.drawMarker`.

```python
# 01_gerar_imagens_teste.py, make_reference()
cell = 44
for row in range(6):
    for col in range(8):
        color = (25, 25, 25) if (row + col) % 2 == 0 else (242, 242, 242)
        x, y = 100 + col * cell, 170 + row * cell
        cv2.rectangle(image, (x, y), (x + cell, y + cell), color, -1)

for i, (x, y) in enumerate(rng.integers([85, 80], [880, 645], size=(95, 2))):
    if i % 2:
        cv2.drawMarker(image, (int(x), int(y)), color, cv2.MARKER_CROSS, 10, 2)
    else:
        cv2.circle(image, (int(x), int(y)), 4, color, -1)
```

Onde aparece: `01_gerar_imagens_teste.py`.

### 3.4 Homografia e transformação de perspectiva

Uma homografia é uma matriz 3 por 3 que mapeia pontos de um plano para pontos de outro plano visto de outra posição de câmera. Ela modela exatamente dois casos: uma cena plana fotografada de qualquer ângulo, ou uma câmera que apenas gira em torno do próprio centro. Cada ponto é tratado em coordenadas homogêneas `(x, y, 1)`, multiplicado pela matriz, e o resultado é dividido pela terceira coordenada. Por isso a matriz tem 8 graus de liberdade (a escala global não importa), e bastam 4 pares de pontos para determiná-la.

O professor gera a segunda imagem do par escolhendo à mão para onde os 4 cantos da referência devem ir. `cv2.getPerspectiveTransform(origem, destino)` calcula a homografia exata a partir desses 4 pares, e `cv2.warpPerspective(imagem, H, (largura, altura))` aplica a deformação a todos os pixels. O argumento `borderValue` define a cor do fundo onde a imagem deformada não cobre a tela. A homografia é salva em `homografia_real.json` para servir de gabarito nos scripts 08 e 09.

```python
# 01_gerar_imagens_teste.py, transform()
source = np.float32([[0, 0], [WIDTH - 1, 0], [WIDTH - 1, HEIGHT - 1], [0, HEIGHT - 1]])
target = np.float32([[92, 74], [875, 25], [925, 648], [44, 698]])
homography = cv2.getPerspectiveTransform(source, target)
warped = cv2.warpPerspective(reference, homography, (WIDTH, HEIGHT), borderValue=(35, 40, 46))
```

Onde aparece: `01_gerar_imagens_teste.py`, `09_item_b_completo.py` (alinhamento), `utils.py` (`corner_error`).

### 3.5 Alterações fotométricas: iluminação, ruído, desfoque e oclusão

Deformar a geometria não basta para testar um descritor de verdade. Fotos reais também mudam de brilho, contraste, nitidez e têm partes tampadas. O professor simula isso em quatro passos sobre a imagem deformada: multiplica cada coluna por um fator que cresce de 0,72 a 1,03 da esquerda para a direita (gradiente de iluminação) e soma 12 (deslocamento de brilho); soma ruído gaussiano de desvio 2,5; aplica `cv2.GaussianBlur` com kernel 3 por 3; e pinta um retângulo escuro com a palavra "OCLUSAO" por cima de parte da cena.

Dois detalhes importam: a conta é feita em `float32` e o resultado é levado de volta a `uint8` com `np.clip(..., 0, 255)`, para não estourar a faixa do pixel. O ruído tem semente própria (`default_rng(7)`).

```python
# 01_gerar_imagens_teste.py, transform()
gradient = np.linspace(0.72, 1.03, WIDTH, dtype=np.float32)[None, :, None]
adjusted = np.clip(warped.astype(np.float32) * gradient + 12, 0, 255)
noise = np.random.default_rng(7).normal(0, 2.5, adjusted.shape).astype(np.float32)
adjusted = np.clip(adjusted + noise, 0, 255).astype(np.uint8)
adjusted = cv2.GaussianBlur(adjusted, (3, 3), 0.45)
cv2.rectangle(adjusted, (735, 560), (900, 655), (48, 52, 58), -1)  # pequena oclusão
```

Onde aparece: `01_gerar_imagens_teste.py`; variação de brilho também em `11_robustez_rotacao_escala_luz.py` e `01_gerar_imagens_teste.py` (`make_video`) via `cv2.convertScaleAbs`.

### 3.6 Geração de vídeo sintético com VideoWriter

Para o exemplo de localização visual é preciso um vídeo em que a cena de referência se mova. O professor cria 120 frames em que os 4 cantos de destino oscilam com seno e cosseno, recalcula a homografia a cada frame com `getPerspectiveTransform`, deforma a referência e ainda varia o brilho com `cv2.convertScaleAbs(frame, alpha, beta)`, que calcula `alpha * pixel + beta` e satura em 0 e 255.

`cv2.VideoWriter(caminho, fourcc, fps, (largura, altura))` abre o arquivo. O código `mp4v` é o codec MPEG-4 parte 2, que costuma funcionar sem instalação extra. É obrigatório checar `writer.isOpened()`, pois um codec ausente falha em silêncio, e chamar `writer.release()` no fim para gravar o cabeçalho do arquivo.

```python
# 01_gerar_imagens_teste.py, make_video()
writer = cv2.VideoWriter(str(RESOURCES / "video_localizacao.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (WIDTH, HEIGHT))
if not writer.isOpened():
    raise RuntimeError("Falha ao criar vídeo")
for frame_index in range(120):
    t = frame_index / 119.0
    dx, dy = 28 * np.sin(t * 2 * np.pi), 20 * np.cos(t * 2 * np.pi)
    h = cv2.getPerspectiveTransform(src, dst)
    frame = cv2.warpPerspective(reference, h, (WIDTH, HEIGHT), borderValue=(30, 36, 42))
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=10)
    writer.write(frame)
writer.release()
```

Onde aparece: `01_gerar_imagens_teste.py`, `12_localizacao_visual_video.py` (grava o vídeo anotado).

### 3.7 Pontos-chave e descritores: detectAndCompute

Um ponto-chave (keypoint) é uma posição da imagem escolhida por ser distinta e repetível: se a mesma cena for fotografada de novo, o detector deve achar o mesmo ponto. Cada keypoint carrega coordenadas `pt`, um tamanho `size` (escala) e um ângulo `angle` (orientação). Um descritor é um vetor numérico que resume a vizinhança do keypoint de forma que vizinhanças parecidas gerem vetores parecidos, mesmo com mudança de escala, rotação ou brilho.

No OpenCV, todos os detectores desta aula seguem a mesma interface: cria-se o objeto com `XXX_create(...)`, e `feature.detectAndCompute(cinza, None)` devolve a lista de keypoints e a matriz de descritores, uma linha por keypoint. A imagem é convertida para escala de cinza antes, com `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`, porque os três algoritmos trabalham sobre intensidade. A função `create_feature` centraliza a criação, e `extract` embrulha a conversão, a criação e a medição de tempo.

```python
# utils.py
def create_feature(method: str):
    method = method.upper()
    if method == "SIFT":
        return cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.025)
    if method == "ORB":
        return cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=8)
    if method == "AKAZE":
        return cv2.AKAZE_create()
    raise ValueError(f"Método desconhecido: {method}")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
keypoints, descriptors = feature.detectAndCompute(gray, None)
```

Onde aparece: `utils.py` (`create_feature`, `extract`), `02_keypoints_sift.py`, `03_keypoints_orb.py`, `04_keypoints_akaze.py`, `05_item_a_comparar_descritores.py` e todos os seguintes; `12_localizacao_visual_video.py` chama `detectAndCompute` diretamente em cada frame.

### 3.8 SIFT

SIFT (Scale-Invariant Feature Transform) detecta extremos em uma pirâmide de diferenças de gaussianas, o que dá invariância a escala, atribui a cada ponto a orientação dominante do gradiente, o que dá invariância a rotação, e descreve a vizinhança com histogramas de orientação de gradiente em 4 por 4 células com 8 direções cada, totalizando 128 valores em ponto flutuante. É o método mais robusto da aula, e também o mais pesado.

Parâmetros usados pelo professor: `nfeatures=1800` limita o número de keypoints retornados aos 1800 mais fortes, e `contrastThreshold=0.025` é o limiar de contraste abaixo do qual um extremo é descartado (o padrão do OpenCV é 0,04; um valor menor aceita mais pontos em regiões de baixo contraste). Os descritores SIFT são comparados com distância euclidiana, `cv2.NORM_L2`.

```python
# 02_keypoints_sift.py
image_a, image_b = load_pair()
kp_a, desc_a, time_a = extract("SIFT", image_a)
kp_b, desc_b, time_b = extract("SIFT", image_b)
print(f"SIFT | A: {len(kp_a)} keypoints, {time_a:.2f} ms")
print(f"Descritor: shape={desc_a.shape}, dtype={desc_a.dtype}, dimensão={desc_a.shape[1]}")
```

Onde aparece: `02_keypoints_sift.py`, `05_item_a_comparar_descritores.py`, `08_homografia_ransac.py`, `09_item_b_completo.py`, `10_comparar_matching_tres_metodos.py`, `11_robustez_rotacao_escala_luz.py`, `12_localizacao_visual_video.py`, e como opção em `06` e `07`.

### 3.9 ORB

ORB (Oriented FAST and Rotated BRIEF) combina o detector de cantos FAST, aplicado em uma pirâmide de escalas, com o descritor binário BRIEF, ao qual acrescenta uma orientação calculada pelo momento de intensidade. O descritor é uma sequência de 256 bits, cada um resultado de comparar a intensidade de dois pixels da vizinhança, armazenada em 32 bytes `uint8`. Por ser binário, é comparado com distância de Hamming (`cv2.NORM_HAMMING`), que conta os bits diferentes e é muito rápida. É o método indicado para CPU embarcada.

Parâmetros usados: `nfeatures=1800`, `scaleFactor=1.2` (razão entre níveis consecutivos da pirâmide) e `nlevels=8` (quantidade de níveis). A tabela no docstring do script 05 resume a leitura de engenharia: muito rápido e compacto, menos robusto a grandes mudanças de escala do que SIFT.

```python
# 03_keypoints_orb.py
kp_a, desc_a, time_a = extract("ORB", image_a)
vis_a = cv2.drawKeypoints(image_a, kp_a, None, (49, 196, 108), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(f"Descritor: shape={desc_a.shape}, dtype={desc_a.dtype}, dimensão={desc_a.shape[1]} bytes")
```

Onde aparece: `03_keypoints_orb.py`, `05_item_a_comparar_descritores.py`, `10_comparar_matching_tres_metodos.py`, e como opção em `06_bfmatcher_crosscheck.py` e `07_flann_lowe.py`.

### 3.10 AKAZE

AKAZE (Accelerated-KAZE) substitui a pirâmide gaussiana do SIFT por difusão não linear, que suaviza a imagem preservando bordas, e usa o descritor binário MLDB (Modified-Local Difference Binary), com 486 bits guardados em 61 bytes na configuração padrão. Assim como o ORB, é comparado com distância de Hamming. O professor o apresenta como um equilíbrio entre a robustez do SIFT e o custo do ORB. É criado sem parâmetros, `cv2.AKAZE_create()`.

```python
# 04_keypoints_akaze.py
kp_a, desc_a, time_a = extract("AKAZE", image_a)
kp_b, desc_b, time_b = extract("AKAZE", image_b)
vis_a = cv2.drawKeypoints(image_a, kp_a, None, (69, 81, 237), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
output = ensure_outputs() / "04_keypoints_akaze.jpg"
cv2.imwrite(str(output), np.hstack([vis_a, vis_b]))
```

Onde aparece: `04_keypoints_akaze.py`, `05_item_a_comparar_descritores.py`, `10_comparar_matching_tres_metodos.py`, e como opção em `06` e `07`.

### 3.11 Tipo do descritor e norma de distância

Cada família de descritor exige a distância certa. Descritores contínuos (SIFT, `float32`) usam a norma L2, a distância euclidiana. Descritores binários (ORB e AKAZE, `uint8`) usam a norma de Hamming. Usar L2 em descritor binário funciona numericamente mas dá resultado sem sentido, pois trata bytes como números. A função `descriptor_info` devolve, para cada método, a norma, o tipo e uma descrição da dimensão, e é ela que alimenta o `BFMatcher` no script 06 e a tabela do script 05.

```python
# utils.py
def descriptor_info(method: str) -> tuple[int, str, str]:
    method = method.upper()
    if method == "SIFT":
        return cv2.NORM_L2, "float32", "128 valores"
    if method == "ORB":
        return cv2.NORM_HAMMING, "uint8", "32 bytes / 256 bits"
    if method == "AKAZE":
        return cv2.NORM_HAMMING, "uint8", "61 bytes / 486 bits (MLDB padrão)"
    raise ValueError(method)
```

Onde aparece: `utils.py`, `05_item_a_comparar_descritores.py`, `06_bfmatcher_crosscheck.py`, `09_item_b_completo.py`.

### 3.12 Medição de tempo com aquecimento e mediana

Medir o tempo de um algoritmo uma única vez engana: a primeira chamada paga custos de alocação e cache, e qualquer processo do sistema pode interferir. O professor faz uma chamada de aquecimento descartada, repete a extração 7 vezes com `time.perf_counter()` (o relógio de maior resolução do Python) e devolve a mediana em milissegundos, que é resistente a valores fora da curva. Nos scripts que só precisam dos descritores, `repeats=1` evita o custo extra.

```python
# utils.py, extract()
feature.detectAndCompute(gray, None)  # aquecimento
times = []
for _ in range(repeats):
    start = time.perf_counter()
    keypoints, descriptors = feature.detectAndCompute(gray, None)
    times.append((time.perf_counter() - start) * 1000)
return keypoints, descriptors, float(np.median(times))
```

Onde aparece: `utils.py`; usado com 7 repetições em `02`, `03`, `04`, `05`, `09`, `10`, e com `repeats=1` em `06`, `07`, `08`, `11`, `12`. O script 12 mede latência por frame do mesmo jeito, sem repetição.

### 3.13 Visualização de keypoints

`cv2.drawKeypoints(imagem, keypoints, None, cor, flags)` desenha os pontos sobre uma cópia da imagem. Com a flag `cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS`, cada ponto vira um círculo cujo raio é a escala do keypoint e com um traço indicando a orientação, o que deixa visível o que cada detector considera "tamanho" e "direção" de uma feature. As duas imagens são colocadas lado a lado com `np.hstack` e gravadas com `cv2.imwrite`. O professor usa uma cor diferente por método: laranja para SIFT, verde para ORB e vermelho para AKAZE (em BGR).

```python
# 02_keypoints_sift.py
vis_a = cv2.drawKeypoints(image_a, kp_a, None, (61, 141, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
vis_b = cv2.drawKeypoints(image_b, kp_b, None, (61, 141, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
output = ensure_outputs() / "02_keypoints_sift.jpg"
cv2.imwrite(str(output), np.hstack([vis_a, vis_b]))
```

Onde aparece: `02_keypoints_sift.py`, `03_keypoints_orb.py`, `04_keypoints_akaze.py`, `05_item_a_comparar_descritores.py`.

### 3.14 Matching por força bruta com cross-check

Casar descritores é achar, para cada descritor da imagem A, o descritor mais parecido da imagem B. O `cv2.BFMatcher` (brute force) compara todos contra todos com a norma informada. O problema é que todo descritor de A recebe um par em B, mesmo os que não têm correspondente real, como os da região ocluída. A opção `crossCheck=True` exige reciprocidade: o par (a, b) só é mantido se b é o melhor de a e a é o melhor de b. Isso elimina boa parte dos casamentos errados sem precisar de limiar.

`matcher.match(desc_a, desc_b)` devolve uma lista de objetos `DMatch`. O script ordena por `distance` (menor é melhor), exibe os 100 melhores com `cv2.drawMatches` e imprime a distância mediana. A flag `DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS` esconde os keypoints que não casaram.

```python
# 06_bfmatcher_crosscheck.py
norm, _, _ = descriptor_info(args.method)
# crossCheck=True mantém m apenas se A->B e B->A concordarem como melhor par.
matcher = cv2.BFMatcher(norm, crossCheck=True)
matches = sorted(matcher.match(desc_a, desc_b), key=lambda m: m.distance)
shown = matches[:100]
visual = cv2.drawMatches(image_a, kp_a, image_b, kp_b, shown, None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
```

Onde aparece: `06_bfmatcher_crosscheck.py`, `09_item_b_completo.py`.

### 3.15 FLANN: índice aproximado adequado ao tipo do descritor

Quando há milhares de descritores, comparar todos contra todos fica caro. FLANN (Fast Library for Approximate Nearest Neighbors) constrói um índice e devolve vizinhos aproximados muito mais rápido. O índice precisa combinar com o tipo do descritor. Para SIFT (`float32`), o professor usa KD-tree com 5 árvores (`algorithm=1`). Para ORB e AKAZE (binários), usa LSH (Locality Sensitive Hashing, `algorithm=6`) com 12 tabelas, chaves de 20 bits e `multi_probe_level=2`. O parâmetro de busca `checks=64` limita quantas folhas são visitadas: mais checks, mais precisão e mais tempo.

```python
# utils.py
def flann_for(method: str) -> cv2.FlannBasedMatcher:
    method = method.upper()
    if method == "SIFT":
        # KD-tree para descritores contínuos float32.
        return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    # LSH para descritores binários ORB/AKAZE.
    return cv2.FlannBasedMatcher(
        dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2),
        dict(checks=64),
    )
```

Onde aparece: `utils.py`, `07_flann_lowe.py`, `08_homografia_ransac.py`, `09_item_b_completo.py`, `10_comparar_matching_tres_metodos.py`, `11_robustez_rotacao_escala_luz.py`, `12_localizacao_visual_video.py`.

### 3.16 k-NN e teste da razão de Lowe

Em vez de pedir só o melhor vizinho, `matcher.knnMatch(desc_a, desc_b, k=2)` devolve os dois melhores de B para cada descritor de A. O teste da razão de Lowe (proposto no artigo original do SIFT) compara as duas distâncias: se o melhor está bem mais perto que o segundo (`m.distance < ratio * n.distance`, com `ratio=0.75`), o casamento é distintivo e é mantido. Se as duas distâncias são parecidas, o descritor é ambíguo, provavelmente parte de uma textura repetitiva como o tabuleiro, e é descartado.

A implementação do professor tem uma proteção importante: `if len(pair) == 2`. Com índice LSH, o FLANN pode devolver menos de 2 vizinhos para alguns descritores, e sem essa checagem o desempacotamento falharia. O script 07 imprime a taxa de retenção, isto é, a fração de pares que sobreviveu ao teste. Argumento `--ratio` permite experimentar outros limiares.

```python
# utils.py
def lowe_filter(knn_matches, ratio: float = 0.75):
    return [m for pair in knn_matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]

# 07_flann_lowe.py
matcher = flann_for(args.method)
knn = matcher.knnMatch(desc_a, desc_b, k=2)
good = lowe_filter(knn, args.ratio)
print(f"Retenção: {100 * len(good) / len(knn):.1f}%")
```

Onde aparece: `utils.py`, `07_flann_lowe.py`, e em todo script que estima homografia (`08` a `12`).

### 3.17 De DMatch a pares de coordenadas

Cada `DMatch` guarda `queryIdx`, o índice do keypoint na primeira lista (A, a "query"), `trainIdx`, o índice na segunda lista (B, o "train"), e `distance`. Para estimar geometria é preciso converter a lista de matches em duas matrizes de pontos alinhadas linha a linha: a i-ésima linha de `points_a` corresponde à i-ésima de `points_b`. O formato `(N, 1, 2)` em `float32` é o que as funções geométricas do OpenCV esperam.

```python
# 08_homografia_ransac.py
points_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
points_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
```

Onde aparece: `08_homografia_ransac.py`, `09_item_b_completo.py`, `10_comparar_matching_tres_metodos.py`, `11_robustez_rotacao_escala_luz.py`, `12_localizacao_visual_video.py`.

### 3.18 Estimação de homografia com RANSAC

Mesmo depois do teste de Lowe, alguns casamentos estão errados (outliers). Um ajuste por mínimos quadrados sobre todos os pontos seria arrastado por eles. RANSAC (Random Sample Consensus) resolve isso por sorteio: escolhe 4 pares ao acaso, calcula a homografia exata deles, conta quantos dos demais pares concordam com ela dentro de um erro de reprojeção máximo, e repete muitas vezes ficando com a hipótese de maior consenso. Ao final, a homografia é refinada só com os inliers.

`cv2.findHomography(pontos_a, pontos_b, cv2.RANSAC, 4.0)` devolve a matriz 3 por 3 e uma máscara `(N, 1)` com 1 para inlier e 0 para outlier. O terceiro argumento é o limiar de reprojeção em pixels: um par é inlier se, projetado pela homografia, cai a menos de 4 px do ponto observado. O resultado pode ser `None` quando não há solução, então o script testa isso. É preciso ter pelo menos 4 correspondências. A proporção de inliers (`mask.mean()`) é a principal medida de confiança na hipótese, e o docstring do script 09 reforça: um robô não deve aceitar a pose apenas porque existem muitos matches, mas exigir inliers suficientes, boa distribuição espacial e erro de reprojeção compatível.

```python
# 08_homografia_ransac.py
if len(good) < 4:
    raise RuntimeError("Homografia requer pelo menos quatro correspondências.")
homography, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 4.0)
if homography is None:
    raise RuntimeError("RANSAC não encontrou uma homografia válida")

inlier_mask = mask.ravel().astype(bool)
inlier_matches = [m for m, keep in zip(good, inlier_mask) if keep]
print(f"Inliers do RANSAC: {int(mask.sum())}")
print(f"Taxa de inliers: {100 * mask.mean():.1f}%")
```

Onde aparece: `08_homografia_ransac.py`, `09_item_b_completo.py`, `10_comparar_matching_tres_metodos.py`, `11_robustez_rotacao_escala_luz.py`, `12_localizacao_visual_video.py`.

### 3.19 Projeção dos cantos e erro contra a homografia verdadeira

Para visualizar o que a homografia diz, o professor projeta os 4 cantos da imagem de referência com `cv2.perspectiveTransform(cantos, H)` e desenha o quadrilátero resultante sobre a imagem B com `cv2.polylines`. Se a estimativa é boa, o polígono contorna exatamente a cena deformada. Note que `perspectiveTransform` transforma pontos, enquanto `warpPerspective` transforma imagens inteiras.

Como a homografia verdadeira está salva em JSON, é possível medir o erro de forma objetiva. A função `corner_error` projeta os 4 cantos com a homografia estimada e com a verdadeira e devolve a distância média em pixels entre os resultados. Antes disso, as duas matrizes são normalizadas dividindo por `h[2, 2]`, porque uma homografia é definida a menos de escala, e duas matrizes que diferem só por um fator representam a mesma transformação.

```python
# utils.py
def normalized_homography(h: np.ndarray) -> np.ndarray:
    return h / h[2, 2]

def corner_error(h_est, h_true, width, height) -> float:
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    est = cv2.perspectiveTransform(corners, normalized_homography(h_est))
    true = cv2.perspectiveTransform(corners, normalized_homography(h_true))
    return float(np.mean(np.linalg.norm(est.reshape(-1, 2) - true.reshape(-1, 2), axis=1)))

# 08_homografia_ransac.py
projected = cv2.perspectiveTransform(corners, homography)
cv2.polylines(visual_b, [np.int32(projected)], True, (38, 226, 167), 5, cv2.LINE_AA)
```

Onde aparece: `utils.py`, `08_homografia_ransac.py`, `09_item_b_completo.py`, `12_localizacao_visual_video.py` (projeção dos cantos em cada frame).

### 3.20 Inversão da homografia e alinhamento de imagens

A homografia estimada leva pontos de A (referência) para B (transformada). Para trazer B de volta ao referencial de A, basta usar a inversa da matriz, `np.linalg.inv(h_ab)`, em `cv2.warpPerspective`. O resultado é a imagem B "desentortada", que deveria coincidir pixel a pixel com A. Para julgar visualmente o alinhamento, o professor mistura as duas com `cv2.addWeighted(A, 0.5, alinhada, 0.5, 0)`: onde o alinhamento é bom a sobreposição fica nítida, onde é ruim aparecem bordas duplicadas como em uma foto fantasma. Esse é o mesmo princípio de costura de panoramas.

```python
# 09_item_b_completo.py
# 4) Inverter H para alinhar B sobre A com warpPerspective.
h_ba = np.linalg.inv(h_ab)
height, width = image_a.shape[:2]
aligned = cv2.warpPerspective(image_b, h_ba, (width, height))
overlay = cv2.addWeighted(image_a, 0.50, aligned, 0.50, 0)
cv2.imwrite(str(ensure_outputs() / "09_imagem_alinhada.png"), aligned)
cv2.imwrite(str(ensure_outputs() / "09_alinhamento_overlay.png"), overlay)
```

Onde aparece: `09_item_b_completo.py`.

### 3.21 Comparação quantitativa e exportação em CSV

Três scripts terminam produzindo uma tabela: o 05 compara a extração (keypoints, tempo, dimensão, tipo), o 10 compara o pipeline inteiro por método (tempo, keypoints, matches após Lowe, inliers e taxa de inliers), e o 11 compara condições de degradação. O padrão é o mesmo: acumular dicionários em uma lista `rows`, gravar com `csv.DictWriter` usando as chaves do primeiro dicionário como cabeçalho, e imprimir uma tabela alinhada com f-strings. Os argumentos `newline=""` e `encoding="utf-8"` são a forma correta de abrir CSV em Python no Windows, evitando linhas em branco extras.

A mensagem didática do script 05 é que a tabela qualitativa (SIFT robusto e lento, ORB rápido e compacto, AKAZE intermediário) precisa ser confirmada por medição, porque contagem e tempo dependem de imagem, hardware e versão do OpenCV.

```python
# 10_comparar_matching_tres_metodos.py
rows.append({
    "metodo": method,
    "tempo_total_extracao_ms": round(time_a + time_b, 2),
    "keypoints_total": len(kp_a) + len(kp_b),
    "matches_lowe": len(good),
    "inliers_ransac": inliers,
    "taxa_inliers_pct": round(100 * inliers / len(good), 1) if good else 0.0,
})
with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
```

Onde aparece: `05_item_a_comparar_descritores.py`, `10_comparar_matching_tres_metodos.py`, `11_robustez_rotacao_escala_luz.py`.

### 3.22 Robustez a rotação, escala e iluminação

Um descritor é dito invariante a rotação e escala se continua casando quando a imagem gira ou muda de tamanho. O script 11 mede isso de forma controlada: gera três variantes da referência com rotação de 5, 15 e 28 graus, escala de 0,95, 0,82 e 0,68, e brilho de 0,95, 0,78 e 0,62, e conta matches após Lowe e inliers do RANSAC em cada caso. Espera-se que os números caiam conforme a degradação aumenta, mas que o SIFT continue localizando a cena mesmo no caso "forte".

A rotação com escala é uma transformação afim, construída por `cv2.getRotationMatrix2D(centro, angulo, escala)`, que devolve uma matriz 2 por 3, e aplicada com `cv2.warpAffine`. Compare com a homografia, que é 3 por 3 e aplicada com `warpPerspective`: a afim preserva paralelismo, a projetiva não. O brilho é reduzido com `cv2.convertScaleAbs(alpha=brightness, beta=12)`.

```python
# 11_robustez_rotacao_escala_luz.py
conditions = [("leve", 5, 0.95, 0.95), ("moderada", 15, 0.82, 0.78), ("forte", 28, 0.68, 0.62)]
for label, angle, scale, brightness in conditions:
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    variant = cv2.warpAffine(reference, matrix, (w, h), borderValue=(32, 38, 44))
    variant = cv2.convertScaleAbs(variant, alpha=brightness, beta=12)
    kp_a, desc_a, _ = extract("SIFT", reference, repeats=1)
    kp_b, desc_b, _ = extract("SIFT", variant, repeats=1)
    good = lowe_filter(flann_for("SIFT").knnMatch(desc_a, desc_b, k=2), 0.75)
```

Onde aparece: `11_robustez_rotacao_escala_luz.py`.

### 3.23 Localização visual em vídeo

O script 12 aplica o pipeline SIFT, FLANN, Lowe e RANSAC a cada frame de um vídeo, comparando sempre contra a mesma referência, cujos keypoints são extraídos uma única vez antes do laço. Em cada frame o script detecta keypoints, casa com a referência, estima a homografia e, quando há confiança, desenha o quadrilátero da cena projetada. Isso é localização visual: saber onde um objeto ou marco conhecido está no campo de visão da câmera.

Há dois limiares de segurança, mais rígidos que nos scripts anteriores: só tenta a homografia com pelo menos 8 matches após Lowe, e só desenha o polígono com pelo menos 12 inliers. O docstring do script 09 explica o porquê: o robô não deve confiar em uma pose sem evidência suficiente. O script também mede a latência por frame com `perf_counter`, escreve `inliers` e `ms` no canto com `cv2.putText`, grava o vídeo anotado com `VideoWriter` e imprime, ao final, latência média, inliers médios e quantos frames foram localizados.

A leitura de vídeo segue o padrão OpenCV: `cv2.VideoCapture(fonte)`, laço com `ok, frame = cap.read()` até `ok` ser falso, `cv2.imshow` e `cv2.waitKey(1)` para exibir e capturar a tecla `q`, e `release()` no fim. A fonte pode ser um caminho ou o inteiro 0 para webcam; o script converte `--source` para inteiro quando a string é só dígitos. A opção `--headless` desliga a janela, útil para rodar sem interface gráfica, e `--max-frames` limita a duração.

```python
# 12_localizacao_visual_video.py
kp_ref, desc_ref, _ = extract("SIFT", reference, repeats=1)
while True:
    ok, frame = cap.read()
    if not ok or (args.max_frames and frame_index >= args.max_frames):
        break
    kp_frame, desc_frame = sift.detectAndCompute(gray, None)
    if desc_frame is not None:
        good = lowe_filter(flann_for("SIFT").knnMatch(desc_ref, desc_frame, k=2), 0.75)
        if len(good) >= 8:
            homography, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
            if homography is not None and mask is not None:
                inliers = int(mask.sum())
                if inliers >= 12:
                    projected = cv2.perspectiveTransform(corners, homography)
                    cv2.polylines(frame, [np.int32(projected)], True, (38, 226, 167), 5, cv2.LINE_AA)
```

Onde aparece: `12_localizacao_visual_video.py`.

### 3.24 Argumentos de linha de comando com argparse

Três scripts aceitam opções pela linha de comando com `argparse`. O `--method` usa `choices=["SIFT", "ORB", "AKAZE"]`, que rejeita qualquer outro valor com mensagem automática. O `--ratio` é `type=float`. No script 12, `--headless` é `action="store_true"` (uma bandeira sem valor), `--max-frames` é inteiro com padrão 0 (sem limite) e `--output-video` tem como padrão um caminho em `saidas/`. Isso permite reutilizar o mesmo script para experimentos sem editar código.

```python
# 07_flann_lowe.py
parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["SIFT", "ORB", "AKAZE"], default="SIFT")
parser.add_argument("--ratio", type=float, default=0.75)
args = parser.parse_args()
```

Onde aparece: `06_bfmatcher_crosscheck.py`, `07_flann_lowe.py`, `12_localizacao_visual_video.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `utils.py` | Define `PROJECT`, `RESOURCES`, `OUTPUTS` e as funções `ensure_outputs`, `load_pair`, `create_feature`, `descriptor_info`, `extract`, `flann_for`, `lowe_filter`, `normalized_homography`, `corner_error` | nenhuma | nenhuma | `cv2`, `numpy` |
| `00_validar_ambiente.py` | Imprime versões de OpenCV e NumPy e confere se SIFT, ORB, AKAZE e `findHomography` existem | nenhuma | texto no terminal | `cv2`, `numpy` |
| `01_gerar_imagens_teste.py` | Desenha a cena de referência, deforma com homografia conhecida mais ruído, luz e oclusão, e gera vídeo de 120 frames | nenhuma | `recursos/cena_referencia.png`, `cena_transformada.png`, `par_imagens.jpg`, `homografia_real.json`, `video_localizacao.mp4` | `utils.py` |
| `02_keypoints_sift.py` | Extrai keypoints SIFT nas duas imagens, mede tempo, desenha | par de imagens | `saidas/02_keypoints_sift.jpg` | 01 |
| `03_keypoints_orb.py` | Extrai keypoints ORB nas duas imagens, mede tempo, desenha | par de imagens | `saidas/03_keypoints_orb.jpg` | 01 |
| `04_keypoints_akaze.py` | Extrai keypoints AKAZE nas duas imagens, mede tempo, desenha | par de imagens | `saidas/04_keypoints_akaze.jpg` | 01 |
| `05_item_a_comparar_descritores.py` | Item A: compara SIFT, ORB e AKAZE em contagem, tempo (mediana de 7), dimensão e tipo | par de imagens | `saidas/05_sift_keypoints.jpg`, `05_orb_keypoints.jpg`, `05_akaze_keypoints.jpg`, `05_tabela_descritores.csv` | 01 |
| `06_bfmatcher_crosscheck.py` | Matching por força bruta com cross-check, mostra os 100 melhores | par de imagens, `--method` | `saidas/06_bf_crosscheck_<metodo>.jpg` | 01 |
| `07_flann_lowe.py` | Matching FLANN k-NN com teste da razão de Lowe, mostra os 140 melhores | par de imagens, `--method`, `--ratio` | `saidas/07_flann_lowe_<metodo>.jpg` | 01 |
| `08_homografia_ransac.py` | SIFT + FLANN + Lowe + `findHomography` RANSAC; desenha inliers e o quadrilátero projetado; mede erro nos cantos contra o gabarito | par de imagens, `homografia_real.json` | `saidas/08_homografia_inliers.jpg` | 01 |
| `09_item_b_completo.py` | Item B: BF cross-check, FLANN + Lowe, RANSAC, inversão da homografia e alinhamento de B sobre A com overlay | par de imagens, `homografia_real.json` | `saidas/09_bf_crosscheck.jpg`, `09_flann_lowe_inliers.jpg`, `09_imagem_alinhada.png`, `09_alinhamento_overlay.png` | 01 |
| `10_comparar_matching_tres_metodos.py` | Roda o pipeline FLANN + Lowe + RANSAC para os três métodos e tabula tempo, keypoints, matches e inliers | par de imagens | `saidas/10_comparacao_matching.csv` | 01 |
| `11_robustez_rotacao_escala_luz.py` | Gera três variantes da referência (rotação, escala, brilho) e mede matches e inliers com SIFT | `cena_referencia.png` | `saidas/11_variante_leve.jpg`, `11_variante_moderada.jpg`, `11_variante_forte.jpg`, `11_robustez.csv` | 01 |
| `12_localizacao_visual_video.py` | Localiza a referência em cada frame do vídeo (ou webcam) com SIFT + FLANN + RANSAC, desenha o polígono, mede latência e grava vídeo anotado | `cena_referencia.png`, vídeo ou `--source 0` | `saidas/12_localizacao_visual.mp4`, janela ao vivo | 01 |

## 5. Passo a passo para executar

Todos os comandos abaixo são executados de dentro de `aula_07_Homografia/exemplos_em_python/`.

```powershell
cd C:\visao_computacional_PVA\aula_07_Homografia\exemplos_em_python
```

**Passo 1. Validar o ambiente.**

```powershell
python 00_validar_ambiente.py
```

Esperado: versões do OpenCV e NumPy e quatro linhas com `True`. Se algum for `False`, reinstale `opencv-python` (versão 4.4 ou mais nova).

**Passo 2. Gerar os recursos.** Obrigatório antes de qualquer outro script.

```powershell
python 01_gerar_imagens_teste.py
```

Esperado: a pasta `aula_07_Homografia/recursos/` é criada com 5 arquivos. Abra `par_imagens.jpg` para ver a referência à esquerda e a versão deformada, escurecida e ocluída à direita. O vídeo `video_localizacao.mp4` tem 6 segundos a 20 fps.

**Passo 3. Ver os keypoints de cada método.**

```powershell
python 02_keypoints_sift.py
python 03_keypoints_orb.py
python 04_keypoints_akaze.py
```

Esperado: cada script imprime contagem de keypoints e tempo em ms para as imagens A e B, mais o shape do descritor (128 colunas `float32` para SIFT, 32 colunas `uint8` para ORB, 61 colunas `uint8` para AKAZE). As imagens em `saidas/` mostram círculos cujo raio é a escala do keypoint.

**Passo 4. Item A: comparar os descritores.**

```powershell
python 05_item_a_comparar_descritores.py
```

Esperado: tabela no terminal com método, keypoints em A e B, tempos, dimensão e tipo, e o mesmo em `saidas/05_tabela_descritores.csv`. Leia o docstring do script para a interpretação de engenharia de cada método.

**Passo 5. Matching por força bruta com cross-check.**

```powershell
python 06_bfmatcher_crosscheck.py
python 06_bfmatcher_crosscheck.py --method ORB
python 06_bfmatcher_crosscheck.py --method AKAZE
```

Esperado: número de matches recíprocos e distância mediana. A imagem em `saidas/` liga com linhas os 100 melhores pares. Observe as linhas que cruzam de forma incoerente: são casamentos errados que o cross-check não removeu.

**Passo 6. FLANN com teste de Lowe.**

```powershell
python 07_flann_lowe.py
python 07_flann_lowe.py --method ORB --ratio 0.7
python 07_flann_lowe.py --ratio 0.9
```

Esperado: número de pares k-NN, quantos passaram no teste e a taxa de retenção. Com `--ratio` maior, mais pares sobrevivem, inclusive ruins. Com menor, sobram menos e mais confiáveis.

**Passo 7. Homografia com RANSAC.**

```powershell
python 08_homografia_ransac.py
```

Esperado: matches após Lowe, inliers, taxa de inliers e erro médio nos cantos em pixels, que deve ser pequeno (poucos pixels). A imagem `08_homografia_inliers.jpg` mostra só os inliers e um quadrilátero verde contornando a cena na imagem B.

**Passo 8. Item B: pipeline completo e alinhamento.**

```powershell
python 09_item_b_completo.py
```

Esperado: bloco "ITEM B" no terminal com tempos, contagens de cada etapa e erro nos cantos. Abra `saidas/09_alinhamento_overlay.png`: se o alinhamento estiver correto, a sobreposição de A com B corrigida aparece nítida, sem bordas duplicadas, exceto na região da oclusão.

**Passo 9. Comparar o pipeline entre os três métodos.**

```powershell
python 10_comparar_matching_tres_metodos.py
```

Esperado: tabela com tempo de extração, keypoints, matches após Lowe, inliers e taxa de inliers por método, salva em `saidas/10_comparacao_matching.csv`.

**Passo 10. Robustez a rotação, escala e luz.**

```powershell
python 11_robustez_rotacao_escala_luz.py
```

Esperado: três linhas (leve, moderada, forte) com matches e inliers decrescentes, e as três variantes gravadas em `saidas/` para inspeção.

**Passo 11. Localização visual no vídeo.**

```powershell
python 12_localizacao_visual_video.py
python 12_localizacao_visual_video.py --headless --max-frames 40
python 12_localizacao_visual_video.py --source 0
```

Esperado: janela "Localização visual" com o vídeo, um polígono verde acompanhando a cena e o texto `inliers=N | X ms` no canto. Tecla `q` encerra. No terminal, frames processados, latência média, inliers médios e quantos frames tiveram 12 ou mais inliers. O vídeo anotado fica em `saidas/12_localizacao_visual.mp4`. Com `--source 0`, aponte a webcam para a cena de referência impressa ou exibida em outra tela.

## 6. Exercícios e itens da aula

Os scripts referenciam dois itens.

**Item A: comparar descritores.** Resolvido por `05_item_a_comparar_descritores.py`, preparado por `02`, `03` e `04`. O item pede uma comparação entre SIFT, ORB e AKAZE em quatro eixos: quantidade de keypoints detectados, tempo de extração, dimensão e tipo do descritor, e distância adequada. O docstring do script traz a tabela qualitativa esperada e a ressalva de que ela deve ser acompanhada da medição impressa, porque contagem e tempo variam com imagem, hardware e versão do OpenCV.

**Item B: matching, homografia e alinhamento.** Resolvido por `09_item_b_completo.py`, preparado por `06`, `07` e `08`. O item pede o pipeline em quatro etapas: BFMatcher com cross-check; FLANN k-NN com teste de Lowe; homografia com RANSAC e contagem de inliers; inversão da homografia e alinhamento da imagem transformada sobre a referência com `warpPerspective`, além do erro contra a homografia verdadeira. O docstring conecta o item à localização visual em robótica: os inliers são a evidência de consistência da pose, e o robô deve exigir inliers suficientes, boa distribuição espacial e erro de reprojeção compatível.

Os scripts 10, 11 e 12 são extensões: comparação do pipeline entre métodos, teste de robustez e aplicação em vídeo.

Exercícios extras sugeridos para praticar:

1. Rode `10_comparar_matching_tres_metodos.py` depois de trocar `nfeatures` de 1800 para 500 em `create_feature` e observe o efeito em tempo e inliers.
2. No script 11, acrescente uma condição "extrema" com 45 graus e escala 0,5 e verifique se o SIFT ainda localiza a cena. Troque o método para ORB e compare.
3. Substitua `cv2.RANSAC` por `cv2.LMEDS` no script 08 e compare a taxa de inliers e o erro nos cantos.
4. No script 12, mude o limiar de 12 inliers para 4 e observe quantos falsos polígonos aparecem nos frames mais escuros do vídeo.
5. Imprima uma foto sua com bastante textura, use `--source 0` no script 12 e ajuste `contrastThreshold` até a localização ficar estável.

## 7. Armadilhas e dicas

- **Rode o 01 primeiro.** Todos os outros scripts chamam `load_pair()` e falham com `FileNotFoundError` se `recursos/` não existir.
- **Rode de dentro de `exemplos_em_python/`.** Os scripts fazem `from utils import ...`, e o Python só encontra `utils.py` se ele estiver na pasta do script ou no `sys.path`.
- **As pastas `recursos/` e `saidas/` ficam em `aula_07_Homografia/`,** um nível acima dos scripts, por causa de `parents[1]` em `utils.py`. Não procure os resultados dentro de `exemplos_em_python/`.
- **SIFT sem contrib.** Não é preciso `opencv-contrib-python`. Se `SIFT_create` faltar, a instalação é anterior à 4.4.
- **Norma errada dá matching sem sentido.** Sempre use `NORM_L2` com SIFT e `NORM_HAMMING` com ORB e AKAZE. A função `descriptor_info` cuida disso; se você criar um `BFMatcher` à mão, cuide você.
- **FLANN com descritor binário precisa de LSH.** Passar descritores `uint8` para o índice KD-tree gera erro ou resultados ruins. Use `flann_for(metodo)`.
- **`knnMatch` pode devolver pares incompletos** com LSH. O `lowe_filter` do professor já testa `len(pair) == 2`. Mantenha essa proteção se reescrever.
- **`findHomography` pode devolver `None`** e a máscara também. Os scripts 10, 11 e 12 tratam isso; o 08 e o 09 lançam exceção. Sempre teste antes de usar.
- **Erro no script 06 com zero matches.** A linha que calcula a distância mediana indexa `[len(matches)//2]` e lança `IndexError` se a lista for vazia. Isso só acontece com imagens sem textura. Mantido como o professor escreveu.
- **Tempo depende da máquina.** Os valores de ms impressos variam entre computadores e entre execuções. Compare apenas a ordem relativa entre métodos, como o docstring do script 05 recomenda.
- **`mp4v` e o `VideoWriter`.** Se `isOpened()` for falso na criação do vídeo, o codec não está disponível; tente `XVID` com extensão `.avi`. Não esqueça `release()`, senão o arquivo fica corrompido.
- **Webcam no script 12.** Só com `--source 0`. Se a câmera principal não abrir, tente `--source 1`. A janela precisa de ambiente gráfico; sem ele use `--headless`.
- **Fidelidade ao professor.** Os docstrings usam travessão no título e o script 11 tem dois comandos na mesma linha separados por ponto e vírgula. Nada disso foi alterado.
- **Parâmetros que valem a pena variar:** `--ratio` no script 07 (0,6 a 0,9); o limiar de reprojeção 4,0 em `findHomography` (2,0 a 8,0); `contrastThreshold` do SIFT; `nfeatures`; `checks` do FLANN; os limiares de 8 matches e 12 inliers do script 12.

## 8. Glossário

| Termo | Significado |
|---|---|
| Keypoint (ponto-chave) | Posição distinta e repetível da imagem, com escala e orientação, escolhida por um detector. |
| Descritor | Vetor numérico que resume a vizinhança de um keypoint, comparável entre imagens. |
| SIFT | Scale-Invariant Feature Transform. Detector e descritor em ponto flutuante, 128 valores, invariante a escala e rotação. |
| ORB | Oriented FAST and Rotated BRIEF. Detector FAST com descritor binário de 256 bits, rápido e leve. |
| AKAZE | Accelerated-KAZE. Detector por difusão não linear com descritor binário MLDB de 486 bits. |
| FAST | Features from Accelerated Segment Test. Detector de cantos usado pelo ORB. |
| BRIEF | Binary Robust Independent Elementary Features. Descritor binário por comparação de pares de pixels. |
| MLDB | Modified-Local Difference Binary. Descritor binário usado pelo AKAZE. |
| Norma L2 | Distância euclidiana, usada para descritores em ponto flutuante. |
| Distância de Hamming | Número de bits diferentes entre dois vetores binários. |
| Matching | Casamento de descritores entre duas imagens. |
| BFMatcher | Brute Force Matcher. Compara todos os descritores contra todos. |
| Cross-check | Regra que só aceita um par se cada lado é o melhor vizinho do outro. |
| FLANN | Fast Library for Approximate Nearest Neighbors. Índice para busca aproximada de vizinhos. |
| KD-tree | Estrutura de índice para vetores contínuos, usada pelo FLANN com SIFT. |
| LSH | Locality Sensitive Hashing. Índice para vetores binários, usado pelo FLANN com ORB e AKAZE. |
| k-NN | k vizinhos mais próximos. Com k=2, devolve os dois melhores candidatos. |
| Teste da razão de Lowe | Aceita um par só se a melhor distância é menor que uma fração (0,75) da segunda melhor. |
| DMatch | Objeto do OpenCV com `queryIdx`, `trainIdx` e `distance` de um casamento. |
| Homografia | Matriz 3 por 3 que mapeia um plano em outro sob projeção de perspectiva. |
| Coordenadas homogêneas | Representação `(x, y, w)` em que o ponto real é `(x/w, y/w)`. |
| Transformação afim | Transformação 2 por 3 que preserva paralelismo (rotação, escala, translação, cisalhamento). |
| RANSAC | Random Sample Consensus. Estimação robusta por sorteio de amostras mínimas e contagem de consenso. |
| LMEDS | Least Median of Squares. Alternativa ao RANSAC em `findHomography`. |
| Inlier | Par de pontos coerente com a homografia dentro do limiar de reprojeção. |
| Outlier | Par de pontos incoerente com a homografia, descartado pelo RANSAC. |
| Erro de reprojeção | Distância entre o ponto projetado pela homografia e o ponto observado. |
| Limiar (threshold) | Valor de corte usado em uma decisão, como os 4 px do RANSAC. |
| Oclusão | Parte da cena tampada por outro objeto. |
| Invariância | Propriedade de um descritor de manter o casamento sob rotação, escala ou mudança de luz. |
| Pirâmide de escalas | Conjunto de versões da imagem em resoluções decrescentes. |
| Frame | Um quadro de vídeo. |
| Latência | Tempo gasto para processar um frame, em ms. |
| Localização visual | Estimar onde um marco conhecido está no campo de visão da câmera. |
| Overlay | Sobreposição de duas imagens com transparência para avaliar alinhamento. |
| fourcc | Código de 4 letras que identifica o codec de vídeo, como `mp4v`. |
| CSV | Comma-Separated Values. Arquivo de texto tabular. |
| argparse | Módulo padrão do Python para argumentos de linha de comando. |
| Headless | Execução sem janela gráfica. |
