# Aula 05 – HSV, segmentação e estéreo

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula junta dois temas que costumam aparecer separados em cursos de visão computacional, mas que aqui convergem num mesmo objetivo: descobrir onde está um objeto na cena e quão perto ele está. A primeira metade da aula (scripts 01 a 04) trata de visão estéreo. A partir de duas fotos tiradas por câmeras deslocadas lateralmente, calcula-se um mapa de disparidade com os algoritmos StereoBM e StereoSGBM do OpenCV, limpa-se esse mapa com filtros robustos e extrai-se uma noção de profundidade relativa, chegando a apontar o pixel mais próximo e o mais distante da câmera. A segunda metade (scripts 05 a 10) trata de segmentação por cor no espaço HSV: separar os canais de matiz, saturação e valor, definir uma faixa de cor com `cv2.inRange`, limpar a máscara com erosão e dilatação, achar o maior contorno, desenhar a caixa delimitadora (bounding box) e medir a proporção da região de interesse (ROI) em relação ao quadro. Isso é feito primeiro em imagem estática e depois em vídeo ou webcam, com gravação de saída, medição de FPS e ajuste interativo por trackbars.

O script 11 fecha o ciclo: segmenta o objeto vermelho da imagem esquerda do par estéreo e cruza a máscara com o mapa de disparidade, respondendo com um único número, a proximidade mediana da ROI. O fio condutor é o `utils.py`, que concentra as funções compartilhadas (carregar o par, criar o SGBM, normalizar a disparidade, segmentar o verde, destacar a ROI) e define onde ficam as pastas `recursos/` e `saidas/`. Os comentários do professor no script 04 contextualizam tudo isso num drone: disparidade alta sugere obstáculo próximo, mas vibração, motion blur e baixa textura degradam o resultado, e distância em metros só vem depois de calibrar e retificar o par.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas: apenas `cv2` (OpenCV) e `numpy`. As demais são da biblioteca padrão (`argparse`, `pathlib`, `time`).

```powershell
pip install opencv-python numpy
```

Estrutura de pastas esperada. O `utils.py` define `ROOT` como a pasta acima da dos scripts, ou seja, `aula_05_HSV/`. As pastas de entrada e saída ficam lá, não dentro de `exemplos_em_python/`:

```
aula_05_HSV/
├── recursos/                  entrada (NÃO veio no zip)
│   ├── estereo_esquerda.png
│   ├── estereo_direita.png
│   ├── alvo_hsv.png
│   └── video_alvo.mp4
├── saidas/                    criada automaticamente pelo utils.py
└── exemplos_em_python/
    ├── utils.py
    └── 01_... a 11_...
```

Os scripts fazem `from utils import ...`, então devem ser executados de dentro de `exemplos_em_python/`. A pasta `saidas/` é criada sozinha na primeira importação do `utils.py` (`SAIDAS.mkdir(exist_ok=True)`).

Material que falta, conforme `achados_dos_zips.md`: a pasta `recursos/` inteira. Os scripts mandam rodar `gerar_recursos.py` quando não acham os arquivos, mas esse script não veio no zip. Enquanto ele não chegar, é preciso providenciar por conta própria:

- `estereo_esquerda.png` e `estereo_direita.png`: par estéreo com a mesma resolução. Pelo comentário do script 11, o objeto mais próximo da cena é vermelho.
- `alvo_hsv.png`: imagem com um objeto verde (o range didático é H de 35 a 85, S de 90 a 255, V de 70 a 255).
- `video_alvo.mp4`: vídeo com o mesmo alvo verde em movimento.

Webcam: os scripts 09 e 10 aceitam `--fonte 0` para usar a câmera no lugar do vídeo. Os demais são todos offline. Todos os scripts aceitam `--sem-janelas` para não abrir janelas do OpenCV, o que é útil em terminal sem interface gráfica ou para rodar em lote.

Versão do Python: o professor usou 3.12 nas aulas vizinhas (05, 06 e 12 segundo o `achados_dos_zips.md`). Esta aula não tem dependência sensível de versão, só OpenCV e NumPy. O `utils.py` usa `from __future__ import annotations` e anotações de tipo como `tuple[np.ndarray, np.ndarray]`, que exigem Python 3.9 ou superior.

## 3. Conceitos-chave

### 3.1 Organização do projeto com `pathlib` e módulo utilitário

Antes de qualquer visão computacional, o professor estabelece um padrão de projeto. O `utils.py` usa `Path(__file__).resolve().parents[1]` para achar a raiz da aula de forma independente do diretório em que o terminal está, e deriva dela as pastas `recursos/` e `saidas/`. Cada script importa dessas constantes em vez de montar caminhos na mão. Isso evita o erro clássico de "arquivo não encontrado" quando se roda o script de outra pasta, e concentra num só lugar as funções reutilizadas.

Todos os scripts seguem a mesma estrutura: `argparse` com a flag `--sem-janelas`, processamento, `cv2.imwrite` do resultado em `saidas/` com prefixo numérico igual ao do script, e por fim `cv2.imshow` condicional.

```python
# utils.py
ROOT = Path(__file__).resolve().parents[1]
RECURSOS = ROOT / "recursos"
SAIDAS = ROOT / "saidas"
SAIDAS.mkdir(exist_ok=True)

# padrão repetido em todos os scripts
parser = argparse.ArgumentParser(description="...")
parser.add_argument("--sem-janelas", action="store_true")
args = parser.parse_args()
...
if not args.sem_janelas:
    cv2.imshow("...", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

Onde aparece: `utils.py` e todos os scripts de 01 a 11.

### 3.2 Par estéreo e sua inspeção

Visão estéreo imita a visão humana: duas câmeras separadas por uma distância horizontal (a linha de base, ou baseline) veem a mesma cena de pontos de vista ligeiramente diferentes. Um objeto próximo aparece deslocado horizontalmente entre as duas imagens muito mais do que um objeto distante. Esse deslocamento em pixels é a disparidade, e é ela que carrega a informação de profundidade.

A função `carregar_par` lê as duas imagens de `recursos/`, falha com mensagem clara se alguma faltar e valida que as duas têm o mesmo `shape`. Essa checagem importa porque os algoritmos de correspondência estéreo comparam linha por linha e exigem resolução idêntica. O script 01 apenas imprime `shape` e `dtype` e mostra as duas lado a lado com `cv2.hconcat`.

```python
# utils.py
def carregar_par() -> tuple[np.ndarray, np.ndarray]:
    esquerda = cv2.imread(str(RECURSOS / "estereo_esquerda.png"))
    direita = cv2.imread(str(RECURSOS / "estereo_direita.png"))
    if esquerda is None or direita is None:
        raise FileNotFoundError("Execute primeiro: python gerar_recursos.py")
    if esquerda.shape != direita.shape:
        raise ValueError("As imagens do par estéreo devem ter a mesma resolução.")
    return esquerda, direita

# 01_inspecionar_par_estereo.py
par = cv2.hconcat([esquerda, direita])
cv2.imwrite(str(SAIDAS / "01_par_estereo.png"), par)
```

Onde aparece: `utils.py`, `01_inspecionar_par_estereo.py`, e indiretamente 02, 03, 04 e 11.

### 3.3 Mapa de disparidade com StereoBM

StereoBM (Block Matching) é o algoritmo estéreo mais simples do OpenCV. Para cada pixel da imagem esquerda ele pega um bloco de vizinhança de `blockSize` x `blockSize` e procura, na mesma linha da imagem direita, o bloco mais parecido dentro de uma faixa de até `numDisparities` pixels de deslocamento. O deslocamento encontrado é a disparidade. Ele trabalha só em escala de cinza, por isso o script converte com `cv2.COLOR_BGR2GRAY`.

Dois detalhes práticos aparecem no código. Primeiro, `numDisparities` precisa ser múltiplo de 16 (o script usa 96). Segundo, `compute` devolve inteiros de 16 bits com ponto fixo: o valor real é o inteiro dividido por 16. Daí o `.astype(np.float32) / 16.0`. Para visualizar, `cv2.normalize` com `NORM_MINMAX` estica a faixa para 0 a 1 e depois o mapa é colorido.

```python
# 02_disparidade_bm.py
g_e = cv2.cvtColor(esquerda, cv2.COLOR_BGR2GRAY)
g_d = cv2.cvtColor(direita, cv2.COLOR_BGR2GRAY)
stereo = cv2.StereoBM_create(numDisparities=96, blockSize=15)
disparidade = stereo.compute(g_e, g_d).astype(np.float32) / 16.0
disp_vis = cv2.normalize(disparidade, None, 0, 1, cv2.NORM_MINMAX)
jet = colorir_jet(disp_vis)
```

Onde aparece: `02_disparidade_bm.py`.

### 3.4 Mapa de disparidade com StereoSGBM

StereoSGBM (Semi-Global Block Matching) melhora o BM ao acrescentar um termo de suavidade: além de procurar o melhor casamento local, ele penaliza mudanças bruscas de disparidade entre pixels vizinhos ao longo de vários caminhos na imagem. O resultado é um mapa mais denso e menos ruidoso, ao custo de mais processamento. O professor encapsula a criação em `criar_sgbm`, com os parâmetros que a documentação do OpenCV recomenda:

- `minDisparity=0` e `numDisparities=96`: faixa de busca, múltiplo de 16 (validado pela função).
- `blockSize=5`: janela de comparação, menor que a do BM porque o termo de suavidade compensa.
- `P1=8*canais*bloco**2` e `P2=32*canais*bloco**2`: penalidades por mudança de disparidade de 1 pixel (P1) e maior que 1 (P2). P2 maior que P1 força superfícies suaves mas ainda permite descontinuidades em bordas de objetos.
- `disp12MaxDiff=1`: tolerância na checagem esquerda-direita, que descarta pixels cujas disparidades nos dois sentidos não batem.
- `uniquenessRatio=8`: exige que o melhor casamento seja pelo menos 8% melhor que o segundo.
- `speckleWindowSize=80` e `speckleRange=2`: filtro de manchas pequenas de disparidade incoerente.
- `preFilterCap=31`: limite do pré-filtro de gradiente.
- `mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY`: variante mais rápida que agrega em 3 direções.

```python
# utils.py
def criar_sgbm(num_disparidades: int = 96, bloco: int = 5) -> cv2.StereoSGBM:
    if num_disparidades % 16 != 0:
        raise ValueError("num_disparidades deve ser múltiplo de 16.")
    canais = 1
    return cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=num_disparidades, blockSize=bloco,
        P1=8 * canais * bloco**2, P2=32 * canais * bloco**2,
        disp12MaxDiff=1, uniquenessRatio=8,
        speckleWindowSize=80, speckleRange=2, preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

def calcular_disparidade_sgbm(esquerda, direita):
    cinza_e = cv2.cvtColor(esquerda, cv2.COLOR_BGR2GRAY)
    cinza_d = cv2.cvtColor(direita, cv2.COLOR_BGR2GRAY)
    return criar_sgbm().compute(cinza_e, cinza_d).astype(np.float32) / 16.0
```

Onde aparece: `utils.py`, `03_disparidade_sgbm.py`, `04_item_a_completo.py`, `11_integrar_roi_e_disparidade.py`.

### 3.5 Máscara de validade e normalização robusta da disparidade

Um mapa de disparidade bruto tem muito lixo: pixels sem correspondência (valor negativo ou zero), a faixa esquerda da imagem onde não há sobreposição entre as duas vistas, e casamentos errados isolados. A função `normalizar_disparidade` constrói uma máscara booleana `validos` que rejeita tudo isso com três critérios: o valor deve ser finito e maior que 0,5; deve estar a menos de 1,5 pixel da mediana local (`cv2.medianBlur` com janela 5), o que elimina pontos isolados incoerentes com a vizinhança; e não pode estar nas bordas (10 pixels em cima, embaixo e à direita, e 100 pixels ou um quarto da largura à esquerda, que é a região sem sobreposição).

Depois, em vez de usar o mínimo e o máximo para escalar, usa os percentis 10 e 90 dos valores válidos. Um único casamento errado com disparidade absurda não consegue mais definir a escala inteira. A função devolve dois mapas normalizados entre 0 e 1 com convenções opostas: `proximidade` (0 longe, 1 perto, proporcional à disparidade) e `profundidade_rel` (0 perto, 1 longe, proporcional ao inverso da disparidade, já que a profundidade Z é inversamente proporcional a d: Z = f·B/d).

```python
# utils.py, dentro de normalizar_disparidade
mediana_local = cv2.medianBlur(disparidade, 5)
validos = (np.isfinite(disparidade) & (disparidade > 0.5)
           & (np.abs(disparidade - mediana_local) < 1.5))
h, w = disparidade.shape
validos[:10, :] = False
validos[h - 10:, :] = False
validos[:, : min(100, w // 4)] = False
validos[:, w - 10:] = False

d_validos = disparidade[validos]
d_min, d_max = np.percentile(d_validos, [10, 90]).astype(float)
d_limitada = np.clip(d_validos, d_min, d_max)
proximidade[validos] = (d_limitada - d_min) / max(d_max - d_min, 1e-6)
inversa[validos] = 1.0 / np.maximum(d_limitada, 1e-6)
```

Onde aparece: `utils.py`, `03_disparidade_sgbm.py`, `04_item_a_completo.py`, `11_integrar_roi_e_disparidade.py`.

### 3.6 Colorização de mapas com `cv2.applyColorMap`

Um mapa de disparidade é uma imagem de um canal em ponto flutuante, difícil de interpretar a olho nu. A função `colorir_jet` converte o mapa normalizado (0 a 1) para 8 bits e aplica `cv2.COLORMAP_JET`, que vai do azul (valores baixos, longe) ao vermelho (valores altos, perto). Os scripts 03, 04 e 11 ainda zeram os pixels inválidos (`jet[~validos] = 0`), deixando-os pretos para não confundir ruído com profundidade.

```python
# utils.py
def colorir_jet(mapa_01: np.ndarray) -> np.ndarray:
    oito_bits = np.clip(mapa_01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(oito_bits, cv2.COLORMAP_JET)

# 03_disparidade_sgbm.py
jet = colorir_jet(proximidade)
jet[~validos] = 0
print(f"Pixels válidos: {validos.sum()} de {validos.size} ({validos.mean():.1%})")
```

Onde aparece: `utils.py`, `02_disparidade_bm.py`, `03_disparidade_sgbm.py`, `04_item_a_completo.py`, `11_integrar_roi_e_disparidade.py`.

### 3.7 Pixel extremo robusto com componentes conexos

O Item A pede o ponto mais próximo e o mais distante da câmera. A resposta ingênua seria `argmax` e `argmin` da disparidade, mas isso quase sempre cai num pixel de ruído. A função `pixel_extremo_robusto` faz melhor: calcula o percentil pedido (90 para perto, 10 para longe) entre os válidos, cria uma máscara binária dos pixels cuja disparidade está a menos de `tolerancia` desse alvo, e roda `cv2.connectedComponentsWithStats` para agrupar esses pixels em regiões conectadas. Escolhe o maior componente (pela coluna `cv2.CC_STAT_AREA`, ignorando o rótulo 0 que é o fundo) e devolve o pixel desse componente mais próximo do centroide. Assim o ponto escolhido pertence a uma superfície real, não a um pixel solto. Se não houver nenhum componente, cai no pixel individual mais próximo do alvo.

```python
# utils.py
def pixel_extremo_robusto(disparidade, validos, percentil, tolerancia):
    alvo = float(np.percentile(disparidade[validos], percentil))
    faixa = (validos & (np.abs(disparidade - alvo) <= tolerancia)).astype(np.uint8)
    n, rotulos, stats, centroides = cv2.connectedComponentsWithStats(faixa)
    if n <= 1:
        ...  # fallback: pixel isolado mais próximo do alvo
    componente = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = centroides[componente]
    ys, xs = np.where(rotulos == componente)
    indice = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return int(xs[indice]), int(ys[indice])

# 04_item_a_completo.py
x_perto, y_perto = pixel_extremo_robusto(disparidade, validos, percentil=90, tolerancia=1.5)
x_longe, y_longe = pixel_extremo_robusto(disparidade, validos, percentil=10, tolerancia=0.75)
cv2.drawMarker(mapa, (x_perto, y_perto), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
```

Onde aparece: `utils.py`, `04_item_a_completo.py`.

### 3.8 Disparidade, profundidade e limitações práticas

O script 04 termina com um comentário do professor que vale como resumo teórico. A relação entre disparidade `d` e profundidade métrica `Z` é `Z = f·B/d`, em que `f` é a distância focal em pixels e `B` a baseline entre as câmeras. Só que essa fórmula pressupõe um par calibrado (parâmetros intrínsecos conhecidos) e retificado (linhas epipolares horizontais, para que a busca linha a linha faça sentido). Sem isso, o que temos é profundidade relativa, adimensional, e é isso que a aula produz.

Num drone, o professor cita os fatores que degradam o mapa: vibração, motion blur, rolling shutter, superfícies sem textura (onde não há o que casar), mudanças de iluminação entre as duas câmeras e atraso de processamento. Por isso, na prática, filtra-se a profundidade ao longo do tempo e rejeitam-se pixels inválidos, exatamente o que a máscara `validos` faz em cada quadro. O script grava também `04_profundidade_relativa.png`, o mapa `profundidade_rel` em escala de cinza (0 preto perto, 255 branco longe).

```python
# 04_item_a_completo.py
print("Convenção: proximidade 0=longe e 1=perto; profundidade relativa 0=perto e 1=longe")
cv2.imwrite(str(SAIDAS / "04_profundidade_relativa.png"),
            (profundidade_rel * 255).astype(np.uint8))
# Em um drone, a disparidade maior sugere obstáculo mais próximo e pode alimentar
# a evitação reativa. Porém, vibração, motion blur, rolling shutter, baixa textura,
# mudança de iluminação e atraso de processamento degradam o mapa. Antes de obter
# distância métrica Z=f*B/d, o par precisa ser calibrado e retificado.
```

Onde aparece: `04_item_a_completo.py`.

### 3.9 Espaço de cor HSV e separação de canais

BGR (a ordem que o OpenCV usa para RGB) mistura cor e brilho nos três canais: um objeto verde sob sombra tem valores muito diferentes do mesmo objeto sob sol. HSV separa esses aspectos. H (Hue, matiz) diz qual é a cor; S (Saturation, saturação) diz o quão pura ou desbotada ela é; V (Value, valor) diz o quão clara. Para segmentar por cor, isso é decisivo: basta fixar uma faixa estreita de H e faixas largas de S e V, e a segmentação passa a tolerar variações de iluminação.

Uma particularidade do OpenCV que o script imprime: H vai de 0 a 179 (não 0 a 359, para caber em 8 bits), enquanto S e V vão de 0 a 255. O script 05 converte com `cv2.cvtColor(..., cv2.COLOR_BGR2HSV)`, separa os canais com `cv2.split` e monta um painel com a imagem original mais os três canais em cinza, cada um reduzido para 320 x 240 com `cv2.resize` e convertido de volta para BGR (`COLOR_GRAY2BGR`) só para poder ser concatenado com a imagem colorida.

```python
# 05_explorar_hsv.py
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)
painel = cv2.hconcat([
    cv2.resize(imagem, (320, 240)),
    cv2.cvtColor(cv2.resize(h, (320, 240)), cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(cv2.resize(s, (320, 240)), cv2.COLOR_GRAY2BGR),
    cv2.cvtColor(cv2.resize(v, (320, 240)), cv2.COLOR_GRAY2BGR),
])
print("OpenCV usa H em [0,179] e S,V em [0,255].")
print("Range didático do verde: H=35..85, S=90..255, V=70..255")
```

Onde aparece: `05_explorar_hsv.py`, e a conversão BGR para HSV em `utils.py`, `10_item_b_avancado.py` e `11_integrar_roi_e_disparidade.py`.

### 3.10 Segmentação por faixa de cor com `cv2.inRange`

`cv2.inRange(hsv, inferior, superior)` devolve uma máscara binária de 8 bits: 255 onde o pixel está dentro da caixa definida pelos dois limites em todos os três canais, 0 fora. É a operação central da segmentação por cor. O professor fixa o verde em H de 35 a 85, S de 90 a 255 e V de 70 a 255. O limite inferior de S descarta cinzas e brancos (baixa saturação), e o de V descarta regiões escuras.

Para ver só o objeto, `cv2.bitwise_and(imagem, imagem, mask=mascara)` mantém a cor original onde a máscara é 255 e zera o resto. `cv2.countNonZero` conta quantos pixels foram classificados como alvo.

```python
# utils.py, dentro de segmentar_verde
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mascara_bruta = cv2.inRange(hsv, np.array([35, 90, 70]), np.array([85, 255, 255]))

# 06_segmentacao_hsv_imagem.py
mascara_bruta, _ = segmentar_verde(imagem)
segmentado = cv2.bitwise_and(imagem, imagem, mask=mascara_bruta)
print(f"Pixels classificados como alvo: {cv2.countNonZero(mascara_bruta)}")
```

Onde aparece: `utils.py`, `06_segmentacao_hsv_imagem.py`, `10_item_b_avancado.py`, `11_integrar_roi_e_disparidade.py`.

### 3.11 Faixa vermelha que cruza o zero do Hue

O matiz é circular: 0 e 179 são praticamente a mesma cor, o vermelho. Uma única chamada de `inRange` não consegue expressar "H perto de 0 ou perto de 179". A solução do script 11 é criar duas máscaras, uma para H de 0 a 12 e outra para H de 168 a 179, e uni-las com `cv2.bitwise_or`. Esse é um caso clássico que pega quem começa a trabalhar com HSV.

```python
# 11_integrar_roi_e_disparidade.py
hsv = cv2.cvtColor(esquerda, cv2.COLOR_BGR2HSV)
# O objeto PERTO do par sintético é vermelho; a faixa vermelha cruza o zero do Hue.
mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([12, 255, 255]))
mask2 = cv2.inRange(hsv, np.array([168, 100, 80]), np.array([179, 255, 255]))
mask = cv2.bitwise_or(mask1, mask2)
```

Onde aparece: `11_integrar_roi_e_disparidade.py`.

### 3.12 Morfologia: erosão, dilatação e abertura

A máscara bruta do `inRange` costuma vir com ruído: pontos isolados fora do objeto (falsos positivos) e furos dentro dele (falsos negativos). As operações morfológicas resolvem isso com um elemento estruturante, aqui uma elipse 7 x 7 criada por `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))`. `cv2.erode` encolhe as regiões brancas e apaga qualquer mancha menor que o kernel. `cv2.dilate` expande as regiões brancas, recompondo o objeto e fechando furos pequenos. O professor aplica uma erosão e duas dilatações, o que remove ruído e ainda engorda um pouco o alvo para conectar partes separadas.

O script 10 usa a forma equivalente com `cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)`, a abertura, que é uma erosão seguida de uma dilatação com o mesmo kernel, e depois mais uma dilatação. O script 11 usa abertura com kernel quadrado 5 x 5 feito com `np.ones`. O script 07 grava a comparação lado a lado, bruta e limpa, e imprime a contagem de pixels antes e depois.

```python
# utils.py, dentro de segmentar_verde
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
erodida = cv2.erode(mascara_bruta, kernel, iterations=1)
limpa = cv2.dilate(erodida, kernel, iterations=2)
return mascara_bruta, limpa

# 10_item_b_avancado.py
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.dilate(mask, kernel, iterations=1)

# 11_integrar_roi_e_disparidade.py
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
```

Onde aparece: `utils.py`, `07_morfologia_mascara.py`, `10_item_b_avancado.py`, `11_integrar_roi_e_disparidade.py`.

### 3.13 Contornos, maior contorno e área mínima

Com a máscara limpa, `cv2.findContours` extrai as fronteiras das regiões brancas. O professor usa `cv2.RETR_EXTERNAL` (só contornos externos, ignorando furos internos) e `cv2.CHAIN_APPROX_SIMPLE` (guarda só os vértices, comprimindo segmentos retos). A função `maior_contorno` filtra os que têm `cv2.contourArea` abaixo de uma área mínima (400 pixels por padrão) e devolve o maior. Isso assume que o alvo é o maior objeto verde da cena, uma premissa razoável para a aula. O script 10 relaxa essa premissa e itera sobre todos os contornos com área acima de 500, ordenados do maior para o menor.

```python
# utils.py
def maior_contorno(mask: np.ndarray, area_minima: float = 400.0):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = [c for c in contornos if cv2.contourArea(c) >= area_minima]
    return max(contornos, key=cv2.contourArea) if contornos else None

# 10_item_b_avancado.py
for c in sorted(contornos, key=cv2.contourArea, reverse=True):
    if cv2.contourArea(c) < 500:
        continue
```

Onde aparece: `utils.py`, `08_extrair_roi_imagem.py`, `09_item_b_video.py`, `10_item_b_avancado.py`, `11_integrar_roi_e_disparidade.py`.

### 3.14 Bounding box, ROI, overlay semitransparente e proporção da área

`cv2.boundingRect(contorno)` devolve `x, y, w, h` do menor retângulo alinhado aos eixos que contém o contorno. Esse retângulo é a ROI (Region of Interest, região de interesse). Recortar a ROI da imagem é só fatiar o array NumPy: `imagem[y:y+h, x:x+w]`.

Para visualizar, a função `destacar_roi` faz três coisas. Pinta de amarelo (0, 255, 255 em BGR) os pixels da máscara numa cópia da imagem e mistura essa cópia com a original usando `cv2.addWeighted(overlay, 0.38, saida, 0.62, 0)`, o que produz o efeito de máscara semitransparente. Desenha o retângulo branco com `cv2.rectangle`. E escreve com `cv2.putText` a proporção entre a área da bounding box (`w*h`) e a área do quadro (`altura*largura`), um número entre 0 e 1 que serve como estimativa grosseira de quão perto ou quão grande o objeto está. O `max(24, y - 10)` na posição do texto impede que ele saia da imagem quando o objeto está no topo.

```python
# utils.py, dentro de destacar_roi
x, y, w, h = cv2.boundingRect(contorno)
overlay = saida.copy()
overlay[mask > 0] = (0, 255, 255)
saida = cv2.addWeighted(overlay, 0.38, saida, 0.62, 0)
cv2.rectangle(saida, (x, y), (x + w, y + h), (255, 255, 255), 2)
proporcao_bbox = (w * h) / float(frame.shape[0] * frame.shape[1])
cv2.putText(saida, f"ROI/frame: {proporcao_bbox:.2%}", (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

# 08_extrair_roi_imagem.py
roi = imagem[y : y + h, x : x + w]
cv2.imwrite(str(SAIDAS / "08_roi_recortada.png"), roi)
```

Onde aparece: `utils.py`, `08_extrair_roi_imagem.py`, `09_item_b_video.py`, `10_item_b_avancado.py`, `11_integrar_roi_e_disparidade.py`.

### 3.15 Processamento de vídeo e webcam com `cv2.VideoCapture`

A passagem de imagem para vídeo é só um laço: `cv2.VideoCapture(fonte)` abre um arquivo ou uma câmera, `cap.read()` devolve `(ok, frame)` a cada iteração, e o laço termina quando `ok` é falso (fim do arquivo) ou quando o usuário aperta `q` ou Esc (código 27). O truque do professor para aceitar as duas fontes com um só argumento é `int(args.fonte) if args.fonte.isdigit() else args.fonte`: se o usuário passar `0`, vira o índice inteiro da webcam; se passar um caminho, fica string.

As propriedades `CAP_PROP_FPS`, `CAP_PROP_FRAME_WIDTH` e `CAP_PROP_FRAME_HEIGHT` são lidas com `cap.get`. O `or 30.0` no FPS cobre webcams que devolvem 0. O argumento `--limite-frames` permite parar depois de N quadros, útil para testes rápidos. Ao final, `cap.release()` libera a fonte.

```python
# 09_item_b_video.py
fonte = int(args.fonte) if args.fonte.isdigit() else args.fonte
cap = cv2.VideoCapture(fonte)
if not cap.isOpened():
    raise RuntimeError(f"Não foi possível abrir a fonte: {args.fonte}")
fps_entrada = cap.get(cv2.CAP_PROP_FPS) or 30.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    ...
    if not args.sem_janelas:
        cv2.imshow("Item B - ROI em tempo real", visual)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    if args.limite_frames and contador >= args.limite_frames:
        break
cap.release()
```

Onde aparece: `09_item_b_video.py`, `10_item_b_avancado.py`.

### 3.16 Gravação de vídeo com `cv2.VideoWriter`

Para salvar o vídeo processado, `cv2.VideoWriter` recebe o caminho de saída, o codec como FourCC (`cv2.VideoWriter_fourcc(*"mp4v")`, que é o MPEG-4 Part 2), o FPS e o tamanho do quadro como `(largura, altura)`. A ordem largura e altura é o inverso do `shape` do NumPy, um erro comum. Cada `writer.write(visual)` acrescenta um quadro, e `writer.release()` no fim fecha o arquivo. Se o release não for chamado, o arquivo pode ficar corrompido. O script também salva um quadro de referência (o de número 45) e sua máscara como PNG, para inspeção sem precisar abrir o vídeo.

```python
# 09_item_b_video.py
writer = cv2.VideoWriter(
    str(SAIDAS / "09_item_b_processado.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_entrada,
    (largura, altura),
)
...
writer.write(visual)
if contador == 45:
    cv2.imwrite(str(SAIDAS / "09_item_b_frame_referencia.png"), visual)
    cv2.imwrite(str(SAIDAS / "09_item_b_mascara_referencia.png"), mascara_limpa)
...
writer.release()
```

Onde aparece: `09_item_b_video.py`.

### 3.17 Medição de desempenho (FPS) com `time.perf_counter`

Em tempo real, o que importa é quantos quadros por segundo o pipeline processa. Os dois scripts de vídeo medem isso com `time.perf_counter()`, o relógio de maior resolução da biblioteca padrão. O script 09 mede o total ao final: quadros processados divididos pela duração. O script 10 calcula o FPS acumulado a cada quadro e escreve na imagem. O `max(..., 1e-9)` evita divisão por zero no primeiro instante.

```python
# 09_item_b_video.py
inicio = time.perf_counter()
...
duracao = max(time.perf_counter() - inicio, 1e-9)
print(f"Resumo: {contador} frames em {duracao:.2f}s; desempenho médio={contador/duracao:.1f} FPS")

# 10_item_b_avancado.py
fps = (contador + 1) / max(time.perf_counter() - t0, 1e-9)
cv2.putText(visual, f"ROIs={rois} area={proporcao:.2%} FPS={fps:.1f}", (18, 34), ...)
```

Onde aparece: `09_item_b_video.py`, `10_item_b_avancado.py`.

### 3.18 Trackbars para ajuste interativo da faixa HSV

Escolher a faixa HSV certa para um objeto real é tentativa e erro. O script 10 resolve isso com trackbars (controles deslizantes) do OpenCV. `cv2.namedWindow` cria a janela, `cv2.createTrackbar(nome, janela, valor_inicial, maximo, callback)` acrescenta cada controle, e dentro do laço `cv2.getTrackbarPos` lê o valor atual. O callback é uma função que o OpenCV chama a cada movimento; o professor passa `lambda _x: None` porque prefere ler os valores no laço em vez de reagir ao evento. Os quatro controles ajustam H mínimo, H máximo, S mínimo e V mínimo, com os valores iniciais do range didático do verde. Em modo `--sem-janelas` os valores ficam fixos nesses padrões.

```python
# 10_item_b_avancado.py
if not args.sem_janelas:
    cv2.namedWindow("Avancado")
    for nome, valor, maximo in [("H min", 35, 179), ("H max", 85, 179),
                                ("S min", 90, 255), ("V min", 70, 255)]:
        cv2.createTrackbar(nome, "Avancado", valor, maximo, lambda _x: None)
...
hmin = cv2.getTrackbarPos("H min", "Avancado")
hmax = cv2.getTrackbarPos("H max", "Avancado")
smin = cv2.getTrackbarPos("S min", "Avancado")
vmin = cv2.getTrackbarPos("V min", "Avancado")
mask = cv2.inRange(hsv, np.array([hmin, smin, vmin]), np.array([hmax, 255, 255]))
```

Onde aparece: `10_item_b_avancado.py`.

### 3.19 Múltiplas ROIs e área total

Enquanto o `destacar_roi` do `utils.py` assume um único alvo, o script 10 trata a cena com vários objetos da mesma cor. Ele percorre todos os contornos com área acima de 500 pixels, desenha um retângulo para cada um, conta quantas ROIs encontrou e soma as áreas das bounding boxes. A proporção reportada é essa soma dividida pela área do quadro. Isso muda a semântica da medida: de "quão grande é o alvo" para "quanto do quadro está ocupado por objetos dessa cor".

```python
# 10_item_b_avancado.py
area_total_bbox = 0
rois = 0
for c in sorted(contornos, key=cv2.contourArea, reverse=True):
    if cv2.contourArea(c) < 500:
        continue
    x, y, w, h = cv2.boundingRect(c)
    area_total_bbox += w * h
    rois += 1
    cv2.rectangle(visual, (x, y), (x + w, y + h), (255, 255, 255), 2)
proporcao = area_total_bbox / float(frame.shape[0] * frame.shape[1])
```

Onde aparece: `10_item_b_avancado.py`.

### 3.20 Integração de segmentação por cor com profundidade

O script 11 é a síntese da aula. Ele usa a imagem esquerda do par estéreo tanto para segmentar o objeto vermelho (via HSV, com a faixa dupla do vermelho e abertura morfológica) quanto como referência do mapa de disparidade (o SGBM produz a disparidade no sistema de coordenadas da imagem esquerda, por isso as duas máscaras se alinham pixel a pixel). A interseção lógica `(mask > 0) & validos` seleciona os pixels que são do objeto E têm disparidade confiável. Sobre esses pixels calcula-se a mediana da proximidade e a faixa entre os percentis 10 e 90. A mediana é uma estatística robusta: mesmo que parte da ROI tenha disparidades erradas, o valor central continua representativo. O resultado é um painel com a imagem anotada (overlay, bounding box e proximidade mediana escrita) ao lado do mapa colorido.

```python
# 11_integrar_roi_e_disparidade.py
roi_valida = (mask > 0) & validos
if not np.any(roi_valida):
    raise RuntimeError("A ROI não possui disparidades válidas.")
mediana = float(np.median(proximidade[roi_valida]))
p10, p90 = np.percentile(proximidade[roi_valida], [10, 90])
print(f"Proximidade mediana da ROI vermelha: {mediana:.3f}")
print(f"Faixa robusta P10..P90: {p10:.3f} .. {p90:.3f}")
...
painel = cv2.hconcat([resultado, mapa])
cv2.imwrite(str(SAIDAS / "11_roi_mais_disparidade.png"), painel)
```

Onde aparece: `11_integrar_roi_e_disparidade.py`.

## 4. Roteiro dos scripts

Todas as saídas vão para `aula_05_HSV/saidas/`. Todas as entradas vêm de `aula_05_HSV/recursos/`.

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `utils.py` | Define `ROOT`, `RECURSOS`, `SAIDAS` e as funções compartilhadas: `carregar_par`, `criar_sgbm`, `calcular_disparidade_sgbm`, `normalizar_disparidade`, `pixel_extremo_robusto`, `colorir_jet`, `maior_contorno`, `segmentar_verde`, `destacar_roi` | nenhuma | cria a pasta `saidas/` ao ser importado | cv2, numpy |
| `01_inspecionar_par_estereo.py` | Carrega o par estéreo, imprime shape e dtype, mostra lado a lado | `estereo_esquerda.png`, `estereo_direita.png` | `01_par_estereo.png` | `utils.py` |
| `02_disparidade_bm.py` | Mapa de disparidade com StereoBM, normalizado e colorido em JET | par estéreo | `02_disparidade_bm_jet.png` | `utils.py` |
| `03_disparidade_sgbm.py` | Mapa de disparidade com StereoSGBM, máscara de validade, colorido em JET com inválidos em preto | par estéreo | `03_disparidade_sgbm_jet.png` | `utils.py` |
| `04_item_a_completo.py` | Item A: disparidade SGBM, proximidade e profundidade relativa, pixel mais próximo e mais distante marcados com cruz | par estéreo | `04_item_a_mapa_jet.png`, `04_profundidade_relativa.png` | `utils.py` |
| `05_explorar_hsv.py` | Converte para HSV e mostra painel BGR, H, S, V | `alvo_hsv.png` | `05_canais_hsv.png` | `utils.py` |
| `06_segmentacao_hsv_imagem.py` | Máscara bruta por `inRange` no verde e objeto segmentado com `bitwise_and` | `alvo_hsv.png` | `06_mascara_hsv_bruta.png`, `06_objeto_segmentado.png` | `utils.py` |
| `07_morfologia_mascara.py` | Compara máscara bruta com máscara após erode e dilate | `alvo_hsv.png` | `07_mascara_bruta_vs_limpa.png` | `utils.py` |
| `08_extrair_roi_imagem.py` | Maior contorno, bounding box, recorte da ROI, overlay e proporção ROI/frame | `alvo_hsv.png` | `08_roi_recortada.png`, `08_roi_overlay.png` | `utils.py` |
| `09_item_b_video.py` | Item B: o pipeline do 08 quadro a quadro em vídeo ou webcam, grava vídeo anotado e mede FPS | `video_alvo.mp4` ou `--fonte 0` | `09_item_b_processado.mp4`, `09_item_b_frame_referencia.png`, `09_item_b_mascara_referencia.png` | `utils.py` |
| `10_item_b_avancado.py` | Item B avançado: trackbars para HSV, abertura morfológica, múltiplas ROIs, FPS na tela | `video_alvo.mp4` ou `--fonte 0` | `10_item_b_avancado.png` (quadro 75) | `utils.py` |
| `11_integrar_roi_e_disparidade.py` | Segmenta o objeto vermelho da imagem esquerda e calcula a proximidade mediana dele no mapa SGBM | par estéreo | `11_roi_mais_disparidade.png` | `utils.py` |

## 5. Passo a passo para executar

Pré-condição: a pasta `aula_05_HSV/recursos/` precisa existir com os quatro arquivos listados na seção 2. Sem ela, todo script falha logo no início com `FileNotFoundError: Execute primeiro: python gerar_recursos.py`.

Abra o terminal dentro de `aula_05_HSV/exemplos_em_python/`. Todos os scripts abaixo aceitam `--sem-janelas` para pular a exibição e apenas gravar em disco.

Parte 1, visão estéreo:

```powershell
python 01_inspecionar_par_estereo.py
```

Imprime shape e dtype das duas imagens e abre uma janela com as duas lado a lado. Confira que as duas têm o mesmo tamanho e que o deslocamento entre elas é horizontal. Qualquer tecla fecha.

```powershell
python 02_disparidade_bm.py
```

Imprime o mínimo e o máximo da disparidade e mostra o mapa JET do StereoBM. Espere um mapa granulado, com regiões pretas ou azuis onde não houve casamento, principalmente na faixa esquerda.

```powershell
python 03_disparidade_sgbm.py
```

Imprime a porcentagem de pixels válidos e mostra o mapa do SGBM, mais suave que o do BM, com os inválidos em preto.

```powershell
python 04_item_a_completo.py
```

Imprime as coordenadas, disparidade, proximidade e profundidade relativa do pixel mais próximo e do mais distante. Na janela, o ponto mais próximo tem uma cruz branca e o mais distante uma cruz preta. Grava também o mapa de profundidade relativa em cinza.

Parte 2, segmentação HSV em imagem:

```powershell
python 05_explorar_hsv.py
python 06_segmentacao_hsv_imagem.py
python 07_morfologia_mascara.py
python 08_extrair_roi_imagem.py
```

O 05 mostra o painel com os canais H, S e V. Observe que o objeto verde tem H homogêneo mesmo com variações de brilho. O 06 abre duas janelas, a máscara bruta e o objeto isolado, e imprime a contagem de pixels. O 07 mostra a máscara bruta ao lado da limpa e imprime a contagem antes e depois. O 08 mostra a imagem com overlay amarelo, retângulo branco e o texto `ROI/frame`, e imprime a bounding box e a proporção.

Parte 3, vídeo e webcam:

```powershell
python 09_item_b_video.py
python 09_item_b_video.py --fonte 0
python 09_item_b_video.py --sem-janelas --limite-frames 100
```

A primeira forma processa `video_alvo.mp4`, a segunda a webcam, a terceira processa só 100 quadros sem abrir janela. A cada quadro imprime `frame=NNNN proporcao_roi=...`. Teclas: `q` ou Esc encerram. Ao final imprime o resumo com o FPS médio e deixa o vídeo anotado em `saidas/09_item_b_processado.mp4`.

```powershell
python 10_item_b_avancado.py
python 10_item_b_avancado.py --fonte 0
```

Abre a janela `Avancado` com quatro trackbars. Mova `H min`, `H max`, `S min` e `V min` enquanto o vídeo roda e veja a máscara amarela e os retângulos mudarem. O texto no canto mostra número de ROIs, área total e FPS. Teclas: `q` ou Esc.

Parte 4, integração:

```powershell
python 11_integrar_roi_e_disparidade.py
```

Imprime a proximidade mediana e a faixa P10 a P90 do objeto vermelho e mostra o painel com a imagem anotada e o mapa de disparidade lado a lado.

Para rodar tudo de uma vez sem janelas:

```powershell
foreach ($s in Get-ChildItem -Filter "*.py" | Where-Object { $_.Name -match '^\d' } | Sort-Object Name) { python $s.Name --sem-janelas }
```

## 6. Exercícios e itens da aula

Os scripts referenciam dois itens.

**Item A, resolvido por `04_item_a_completo.py`.** A partir de um par estéreo, gerar o mapa de disparidade, convertê-lo em profundidade relativa e identificar o ponto mais próximo e o mais distante da câmera. Os scripts 01, 02 e 03 são os passos preparatórios: inspecionar o par, comparar BM com SGBM e entender a máscara de validade. A resposta do item inclui a convenção adotada (proximidade 0 longe e 1 perto, profundidade relativa o contrário), as coordenadas dos dois extremos com seus valores, e o mapa colorido com as marcações. O comentário final do script discute a aplicação em drones e por que o resultado não é métrico.

**Item B, resolvido por `09_item_b_video.py`.** Em vídeo ou webcam, a cada quadro: (1) aplicar um range HSV, (2) limpar a máscara com erode e dilate, (3) desenhar a bounding box e uma máscara colorida semitransparente, e (4) calcular a proporção da área da ROI em relação ao quadro. Os comentários numerados dentro do laço do script 09 mapeiam exatamente esses quatro passos. Os scripts 05 a 08 constroem o pipeline peça por peça em imagem estática antes de aplicá-lo em vídeo. O `10_item_b_avancado.py` é uma extensão do item com trackbars, múltiplas ROIs e FPS na tela.

O script 11 não é nomeado como item, mas integra A e B.

## 7. Armadilhas e dicas

- **A pasta `recursos/` não veio no zip e `gerar_recursos.py` também não.** Nada roda até ela existir. Ela fica em `aula_05_HSV/recursos/`, não em `exemplos_em_python/recursos/`. Se for criar as imagens por conta própria, respeite as cores: alvo verde para o `alvo_hsv.png` e o vídeo, objeto vermelho como o mais próximo no par estéreo.
- **Rode de dentro de `exemplos_em_python/`.** Os scripts fazem `from utils import ...`, que só funciona se essa pasta for o diretório atual ou estiver no `sys.path`.
- **`numDisparities` deve ser múltiplo de 16.** A função `criar_sgbm` valida isso, mas o StereoBM do script 02 não valida. Se mudar o valor lá, mantenha a regra.
- **Divida a disparidade por 16.** `compute` devolve inteiros em ponto fixo. Sem a divisão, os valores ficam 16 vezes maiores e a normalização por percentis fica sem sentido.
- **Ordem `(largura, altura)` no `VideoWriter`** é o contrário do `shape` do NumPy. Se inverter, o vídeo sai vazio ou corrompido sem erro na tela.
- **Chame `writer.release()`.** O script faz isso, mas se for interromper com Ctrl+C no meio, o MP4 pode ficar inválido. Prefira sair com `q` ou usar `--limite-frames`.
- **Vermelho em HSV precisa de duas faixas** (perto de 0 e perto de 179). O script 11 mostra como. O verde do `utils.py` não tem esse problema porque fica no meio da escala.
- **H vai só até 179 no OpenCV.** Se pegar valores de referência de outra ferramenta (0 a 360), divida por 2.
- **Webcam com `--fonte 0`.** Se a câmera não abrir, tente `--fonte 1`. O `cap.get(cv2.CAP_PROP_FPS)` pode devolver 0 em webcams, e o script já cobre isso com `or 30.0`.
- **Contagem no script 10.** O `contador` é incrementado depois do cálculo do FPS, por isso a fórmula usa `contador + 1`. Não é defeito.
- **O script 10 não usa `segmentar_verde` nem `destacar_roi`** do `utils.py`. Ele reimplementa a segmentação com os valores das trackbars e uma abertura em vez de erode mais dilate. Se quiser comparar os dois pipelines, rode 09 e 10 no mesmo vídeo com `--sem-janelas` e compare as proporções impressas.
- **Parâmetros que valem a pena variar.** No SGBM: `blockSize` (3, 5, 7, 9) e `numDisparities` (64, 96, 128). Na normalização: os percentis 10 e 90 e o limiar de 1,5 contra a mediana local. Na morfologia: tamanho do kernel (5, 7, 9) e número de iterações. No contorno: a área mínima (400 no `utils.py`, 500 no script 10).
- **Fidelidade ao professor.** Nada foi alterado nos scripts. O `from __future__ import annotations` e as anotações `tuple[...]` exigem Python 3.9 ou superior, o que não é problema no ambiente atual.

## 8. Glossário

| Termo | Significado |
|---|---|
| Abertura (opening) | Erosão seguida de dilatação com o mesmo kernel. Remove ruído pequeno sem alterar o tamanho das regiões grandes. `cv2.MORPH_OPEN`. |
| `addWeighted` | Mistura duas imagens com pesos, usada para o overlay semitransparente. |
| Baseline (B) | Distância horizontal entre as duas câmeras de um par estéreo. |
| BGR | Ordem de canais que o OpenCV usa por padrão: azul, verde, vermelho. |
| Bounding box | Menor retângulo alinhado aos eixos que contém um contorno. `cv2.boundingRect` devolve `x, y, w, h`. |
| Calibração e retificação | Processo de determinar os parâmetros das câmeras e alinhar as duas imagens para que a busca de correspondência seja linha a linha. Pré-requisito para profundidade métrica. |
| Componente conexo | Grupo de pixels brancos ligados entre si numa máscara binária. `cv2.connectedComponentsWithStats` rotula e mede cada um. |
| Contorno | Curva que delimita uma região branca de uma máscara. `cv2.findContours`. |
| Dilatação (dilate) | Operação morfológica que expande regiões brancas. Fecha furos e conecta partes próximas. |
| Disparidade | Deslocamento horizontal em pixels de um mesmo ponto entre a imagem esquerda e a direita. Maior disparidade significa objeto mais próximo. |
| Elemento estruturante (kernel) | Forma usada nas operações morfológicas. `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))`. |
| Erosão (erode) | Operação morfológica que encolhe regiões brancas. Apaga manchas menores que o kernel. |
| FourCC | Código de 4 caracteres que identifica o codec de vídeo. `mp4v` é MPEG-4 Part 2. |
| FPS | Frames per second, quadros por segundo. Medida de desempenho do pipeline. |
| Frame | Um quadro individual de um vídeo. |
| HSV | Espaço de cor com matiz (H), saturação (S) e valor (V). No OpenCV, H vai de 0 a 179 e S e V de 0 a 255. |
| `inRange` | Função que gera uma máscara binária com os pixels dentro de uma faixa de valores nos três canais. |
| JET | Mapa de cores que vai de azul (baixo) a vermelho (alto). `cv2.COLORMAP_JET`. |
| Máscara | Imagem binária de 8 bits, 255 onde a condição vale e 0 fora. |
| Mediana | Valor central de um conjunto ordenado. Robusta a outliers. |
| Overlay | Camada colorida sobreposta à imagem para destacar uma região. |
| Percentil | Valor abaixo do qual está uma porcentagem dos dados. P10 e P90 delimitam uma faixa robusta. |
| Profundidade relativa | Estimativa adimensional de distância, entre 0 e 1, sem calibração. Proporcional ao inverso da disparidade. |
| Proximidade | Mapa entre 0 (longe) e 1 (perto), proporcional à disparidade normalizada. |
| ROI | Region of Interest, região de interesse. Aqui, a bounding box do objeto segmentado. |
| SGBM | Semi-Global Block Matching. Algoritmo estéreo que adiciona penalidades de suavidade ao casamento de blocos. |
| StereoBM | Block Matching simples: casa blocos de vizinhança linha a linha. |
| Trackbar | Controle deslizante numa janela do OpenCV. `cv2.createTrackbar` e `cv2.getTrackbarPos`. |
| `VideoCapture` | Classe do OpenCV que lê quadros de um arquivo de vídeo ou de uma câmera. |
| `VideoWriter` | Classe do OpenCV que grava quadros num arquivo de vídeo. |
| Z = f·B/d | Relação entre profundidade métrica Z, distância focal f, baseline B e disparidade d. |
