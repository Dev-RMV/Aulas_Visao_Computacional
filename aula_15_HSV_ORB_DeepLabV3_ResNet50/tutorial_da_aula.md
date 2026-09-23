# Aula 15 – Segmentação semântica e pipeline integrativo: HSV, ORB, FCN-ResNet50 e DeepLabV3

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula fecha o ciclo de percepção visual para robótica e veículos autônomos. A pergunta central é: dado um frame de uma rua, como saber o que é pista, calçada, vegetação, veículo e pedestre, pixel a pixel, e como encaixar essa resposta em um pipeline completo que também calibra a câmera, segmenta por cor, extrai pontos de interesse e detecta objetos, tudo dentro de um orçamento de tempo e energia. O professor responde em duas frentes. A primeira é clássica: segmentação por cor no espaço HSV com `cv2.inRange`, limpeza morfológica, ORB e um detector simples com Canny e contornos. A segunda é profunda: redes de segmentação semântica pré-treinadas do `torchvision`, FCN-ResNet50 e DeepLabV3-ResNet50, que classificam cada pixel em uma de 21 classes sem nenhum treinamento local.

O fio condutor é incremental, como nas aulas anteriores. O módulo `seg_utils.py` cria cenas sintéticas de rua com uma máscara ideal de 8 classes, que serve de gabarito. Os scripts 01 a 08 exploram essa máscara: colorir, sobrepor, medir área por classe, segmentar a pista por HSV e comparar com o gabarito. Os scripts 09 a 19 entram no mundo profundo: transformações de entrada, carregamento do FCN, inferência, coloração da saída, lote de 5 imagens, DeepLabV3, latência e memória. Os scripts 20 a 30 integram e medem: tabela comparativa, IoU, diagrama do pipeline, undistort, ORB, detector clássico, frame final anotado, métricas em JSON e tabela em Markdown. Os scripts 31 a 35 são a parte de análise e entrega: orçamento de 5 W, arquitetura de percepção urbana, lacunas de pesquisa, relatório e checklist. Cada script termina com um bloco `DESAFIO DO ALUNO`, que é a parte prática da aula.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas nos scripts. O zip não traz `requirements.txt`; os scripts 09, 10 e 19 indicam as dependências em comentários (`pip install torch torchvision pillow` e `pip install psutil`):

| Biblioteca | Pacote pip | Onde é usada |
|---|---|---|
| OpenCV | `opencv-contrib-python` | `seg_utils.py` e quase todos os scripts |
| NumPy | `numpy` | `seg_utils.py`, 06, 07, 12, 14, 15, 21, 23, 26 |
| PyTorch | `torch` | 10, 11, 15, 17, 18 |
| torchvision | `torchvision` | 09, 10, 11, 15, 17, 18 |
| Pillow | `Pillow` | 11, 15, 18 |
| psutil | `psutil` | 19 |
| tabulate | `tabulate` | 20 |

No repositório, tudo isso está no `requirements.txt` único da raiz, instalado no venv `.venv` com Python 3.13 por `py preparar_ambiente.py`. OpenCV Contrib 4.14, NumPy 2.x, Pillow e psutil 7.2 já estavam lá desde as aulas anteriores. Esta aula acrescentou `torch` 2.14.0 e `torchvision` 0.29.0 (wheel do PyPI, só CPU no Windows) e `tabulate` 0.10.0. O `tabulate` não aparece em nenhum comentário do professor, mas o script 20 importa.

Pasta de execução: todos os caminhos partem de `ROOT = Path(__file__).resolve().parent`, ou seja, da própria pasta dos scripts. Rode sempre de dentro de `aula_15_HSV_ORB_DeepLabV3_ResNet50/exemplos_em_python/`. A função `seg_utils.ensure` cria `imagens/` e `saidas/`, e é chamada por `salvar_cenas`, que quase todos os scripts executam no início. A pasta `relatorio/` nunca é criada pelos scripts, embora 22, 29 e 30 gravem nela; por isso o repositório já traz `relatorio/.gitkeep`. As imagens geradas em `imagens/`, as saídas em `saidas/` e o conteúdo de `relatorio/` ficam fora do git.

Pesos das redes: na primeira execução de 10, 11, 15 ou 18 o `torchvision` baixa o FCN-ResNet50 (cerca de 135 MB) para `~/.cache/torch/hub/checkpoints/`. O 17 baixa o DeepLabV3-ResNet50 (cerca de 160 MB). Depois disso tudo roda offline.

Material faltante: o script 34 espera `relatorio/RELATORIO_INTEGRATIVO_MODELO.md`, um modelo de relatório do professor que não veio no zip. Sem ele o script só imprime um aviso. Todo o resto roda com as cenas sintéticas.

Webcam: nenhum script abre a câmera. A aula inteira funciona com as cinco cenas geradas por `seg_utils.py`.

## 3. Conceitos-chave

### 3.1 Raiz do projeto e pastas de trabalho

Cada script começa definindo `ROOT` como a pasta onde ele mesmo está, com `Path(__file__).resolve().parent`. Todos os caminhos de entrada e saída são construídos a partir dessa raiz, então o script funciona independentemente do diretório de onde o Python foi chamado, desde que `seg_utils.py` esteja na mesma pasta para ser importado. A função `ensure` garante as duas pastas de trabalho com `mkdir(exist_ok=True)`, que não falha se a pasta já existe.

```python
# seg_utils.py
def ensure(root):
    Path(root, "imagens").mkdir(exist_ok=True)
    Path(root, "saidas").mkdir(exist_ok=True)
```

Onde aparece: `seg_utils.py` (`ensure`, chamada dentro de `salvar_cenas`), e `ROOT = Path(__file__).resolve().parent` em todos os scripts de 01 a 30, exceto 09, 10, 17, 19 e 20, que não tocam em arquivos.

### 3.2 Cena sintética de rua com máscara ideal

Para ensinar segmentação sem depender de fotos, o professor desenha a cena e a resposta certa ao mesmo tempo. `criar_cena_externa` monta duas matrizes de 540 por 960: `img`, colorida em BGR, e `mask`, de um canal, onde cada pixel recebe o id da classe. Cada primitiva é desenhada nas duas: o terço superior é céu (id 1), o resto é pista (id 2), os quartos laterais abaixo do horizonte são calçada (id 3), círculos verdes com tronco são vegetação (id 4), retângulos vermelhos com rodas são veículos (id 5), cabeça, corpo e pernas são pedestre (id 6, marcado como retângulo na máscara) e blocos brancos no centro são faixa (id 7). O parâmetro `idx` desloca árvores, carros e pedestres para que as cinco cenas sejam diferentes.

```python
# seg_utils.py
def criar_cena_externa(idx=0, w=960, h=540):
    img = np.zeros((h, w, 3), np.uint8)
    mask = np.zeros((h, w), np.uint8)

    img[:h//3, :] = (210, 180, 120); mask[:h//3, :] = 1
    img[h//3:, :] = (90, 90, 90); mask[h//3:, :] = 2

    cv2.rectangle(img, (0, h//3), (w//4, h), (170, 110, 90), -1); mask[h//3:, 0:w//4] = 3
    cv2.rectangle(img, (3*w//4, h//3), (w, h), (170, 110, 90), -1); mask[h//3:, 3*w//4:w] = 3
```

`salvar_cenas` chama `ensure`, gera `n` cenas e grava `imagens/cena_NN.png` e `imagens/cena_NN_mask.png`. A máscara é salva como PNG de um canal com valores de 0 a 7, que parece toda preta em um visualizador comum.

Onde aparece: `seg_utils.py` (`criar_cena_externa`, `salvar_cenas`) e `01_criar_cenas_sinteticas.py`. Os scripts 02 a 08, 11, 15, 18, 21, 23 a 28 chamam `salvar_cenas(ROOT, 5)` no início, regerando as cinco cenas a cada execução.

### 3.3 Máscara de rótulos por pixel e paleta de cores

Segmentação semântica produz uma imagem de ids: cada pixel guarda um inteiro que diz a que classe ele pertence. O dicionário `CLASSES_DIDATICAS` dá nome aos oito ids, e `PALETTE` associa uma cor BGR a cada um. `colorir_mascara` usa a indexação avançada do NumPy: `PALETTE[mask]` devolve, para cada pixel, a linha da paleta correspondente ao id, produzindo direto uma imagem colorida de três canais. O `np.clip` protege contra ids acima do tamanho da paleta.

```python
# seg_utils.py
CLASSES_DIDATICAS = {
    0: "fundo", 1: "ceu", 2: "pista", 3: "calcada",
    4: "vegetacao", 5: "veiculo", 6: "pedestre", 7: "faixa"
}

PALETTE = np.array([
    [0, 0, 0], [255, 180, 80], [80, 80, 80], [160, 80, 160],
    [80, 180, 80], [70, 70, 220], [220, 60, 60], [255, 255, 255]
], dtype=np.uint8)

def colorir_mascara(mask):
    return PALETTE[np.clip(mask, 0, len(PALETTE)-1)]
```

Onde aparece: `seg_utils.py`, `02_visualizar_mascara_ideal.py` (grava `saidas/02_mascara_colorida.png`), e indiretamente em `sobrepor`, usada por 03, 08, 27 e 28.

### 3.4 Overlay semitransparente com `cv2.addWeighted`

Ver a máscara sozinha não diz se ela bate com a imagem. O overlay mistura as duas: `cv2.addWeighted(a, alpha_a, b, alpha_b, gamma)` calcula `a*alpha_a + b*alpha_b + gamma` pixel a pixel, com saturação em 255. Em `sobrepor`, a imagem entra com peso `1-alpha` e a máscara colorida com peso `alpha` (padrão 0,45), então a cena continua visível por baixo das cores das classes. O mesmo padrão aparece com pesos 0,75 e 0,25 no script 07, 0,7 e 0,3 no 08 e 16, e 0,65 e 0,35 no 13 e 15.

```python
# seg_utils.py
def sobrepor(img, mask, alpha=0.45):
    color = colorir_mascara(mask)
    return cv2.addWeighted(img, 1-alpha, color, alpha, 0)
```

Onde aparece: `seg_utils.py`, `03_sobrepor_mascara_original.py`, `07_overlay_hsv.py`, `08_painel_hsv_vs_mascara_ideal.py`, `13_overlay_fcn.py`, `15_processar_5_imagens_fcn.py`, `16_painel_semantica_vs_hsv.py`, `27_pipeline_com_segmentacao_semantica.py`, `28_frame_final_anotado.py`.

### 3.5 Área por classe

Uma métrica simples e útil em navegação é quanto da imagem cada classe ocupa. `porcentagens` percorre o dicionário de classes e conta os pixels com `np.sum(mask == k)`, uma comparação vetorizada que gera um array booleano somado como inteiros. O resultado é dividido por `mask.size` (total de pixels) e multiplicado por 100. Classes com menos de 0,01 % são omitidas do resultado.

```python
# seg_utils.py
def porcentagens(mask, classes=CLASSES_DIDATICAS):
    total = mask.size
    out = {}
    for k, nome in classes.items():
        pct = 100 * float(np.sum(mask == k)) / total
        if pct > 0.01:
            out[nome] = pct
    return out
```

Onde aparece: `seg_utils.py`, `04_calcular_area_por_classe.py`, `27_pipeline_com_segmentacao_semantica.py`, `28_frame_final_anotado.py`. O script 14 faz a mesma conta de outro jeito, com `np.unique`.

### 3.6 Segmentação da pista por cor em HSV

A primeira forma de segmentar é a clássica: escolher uma faixa de cor. O espaço HSV separa matiz (H, 0 a 179 no OpenCV), saturação (S) e valor (V). A pista sintética é um cinza `(90, 90, 90)`, ou seja, saturação quase zero e brilho médio. `hsv_pista` aceita qualquer matiz (H de 0 a 179), saturação até 60 e valor entre 40 e 150. `cv2.inRange` devolve uma máscara binária com 255 onde os três canais estão dentro dos limites. O problema didático já está plantado: a calçada `(170, 110, 90)` tem saturação maior, mas o céu e as sombras podem escapar pela faixa de brilho.

```python
# seg_utils.py
def hsv_pista(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 40])
    upper = np.array([179, 60, 150])
    return cv2.inRange(hsv, lower, upper)
```

Onde aparece: `seg_utils.py`, `05_segmentacao_hsv_pista.py`, `06_limpeza_morfologica.py`, `07_overlay_hsv.py`, `08_painel_hsv_vs_mascara_ideal.py`, `16_painel_semantica_vs_hsv.py`, `21_iou_mascara_hsv_vs_ideal.py`, `26_integrar_hsv_orb_detector.py`.

### 3.7 Limpeza morfológica: abertura e fechamento

A máscara de `inRange` costuma ter pontos isolados e buracos. O script 06 aplica duas operações morfológicas com um kernel quadrado de 7 por 7 feito com `np.ones`. `cv2.MORPH_OPEN` (erosão seguida de dilatação) apaga manchas menores que o kernel. `cv2.MORPH_CLOSE` (dilatação seguida de erosão) preenche buracos e emenda regiões próximas. A ordem importa: abrir primeiro remove ruído sem crescer as regiões, fechar depois consolida o que sobrou.

```python
# 06_limpeza_morfologica.py
kernel = np.ones((7, 7), np.uint8)
limpa = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, kernel)
```

Onde aparece: `06_limpeza_morfologica.py` (grava `saidas/06_hsv_limpa.png`).

### 3.8 Overlay de máscara binária e painel lado a lado

Para mostrar a máscara HSV sobre a cena, o script 07 cria uma imagem preta do tamanho da cena com `np.zeros_like`, pinta de amarelo `(0, 255, 255)` os pixels onde a máscara é positiva, usando indexação booleana, e mistura com `addWeighted`. O script 08 vai além e monta um painel de três imagens com `painel_lado_a_lado`, que é só um `np.hstack`: original, HSV sobreposto e máscara ideal sobreposta. Como `hstack` exige o mesmo número de canais, a máscara em cinza é convertida com `cv2.COLOR_GRAY2BGR` antes.

```python
# 07_overlay_hsv.py
color = np.zeros_like(img)
color[mask > 0] = (0, 255, 255)
out = cv2.addWeighted(img, 0.75, color, 0.25, 0)
```

```python
# 08_painel_hsv_vs_mascara_ideal.py
painel = painel_lado_a_lado(
    img,
    cv2.addWeighted(img, 0.7, hsv_color, 0.3, 0),
    sobrepor(img, ideal)
)
```

Onde aparece: `07_overlay_hsv.py`, `08_painel_hsv_vs_mascara_ideal.py`, `16_painel_semantica_vs_hsv.py`, `seg_utils.py` (`painel_lado_a_lado`).

### 3.9 Normalização de entrada para redes profundas

Redes pré-treinadas esperam a entrada no mesmo formato usado no treino. O script 09 monta a transformação canônica do `torchvision` com `transforms.Compose`: `ToTensor` converte uma imagem PIL (inteiros 0 a 255, formato altura por largura por canal) em um tensor float de 0 a 1 no formato canal por altura por largura, e `Normalize` subtrai a média e divide pelo desvio padrão de cada canal RGB. Os seis números são as estatísticas do ImageNet, o conjunto em que os troncos ResNet foram pré-treinados. Sem essa normalização a rede recebe valores fora da distribuição que aprendeu e a saída degrada.

```python
# 09_preparar_transform_torchvision.py
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Onde aparece: `09_preparar_transform_torchvision.py`. Nos scripts 11, 15 e 18 a mesma normalização vem embutida em `weights.transforms()`.

### 3.10 Carregar uma rede de segmentação pré-treinada

O `torchvision.models.segmentation` traz arquiteturas prontas com pesos publicados. `fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)` constrói uma FCN (Fully Convolutional Network) com tronco ResNet50 e baixa os pesos na primeira chamada. `.eval()` coloca a rede em modo de inferência, o que desliga dropout e congela as estatísticas de batch normalization. O objeto `weights` carrega metadados: `weights.meta["categories"]` lista as 21 classes (fundo mais as 20 do Pascal VOC, como person, car, bus, bicycle), porque esses pesos foram treinados em um subconjunto do COCO com rótulos VOC.

```python
# 10_carregar_fcn_resnet50.py
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).eval()
classes = weights.meta.get("categories", [])

print("Modelo carregado: FCN-ResNet50")
print("Número de classes:", len(classes))
print("Primeiras classes:", classes[:10])
```

Onde aparece: `10_carregar_fcn_resnet50.py`, `11_inferencia_fcn_uma_imagem.py`, `15_processar_5_imagens_fcn.py`, `18_medir_tempo_segmentacao.py`.

### 3.11 Inferência: transform dos pesos, lote, `no_grad` e `argmax`

O script 11 é o núcleo da parte profunda. `weights.transforms()` devolve a transformação exata que os pesos esperam: redimensiona o lado menor da imagem para 520 pixels, converte em tensor e normaliza com as estatísticas do ImageNet. A imagem é aberta com Pillow e convertida para RGB (o OpenCV usaria BGR, o que trocaria os canais). `unsqueeze(0)` adiciona a dimensão de lote, porque a rede sempre processa um lote, mesmo de uma imagem. `torch.no_grad()` desliga o registro de gradientes, que só serve para treinar e gastaria memória. A saída é um dicionário; `["out"]` tem forma `(1, 21, H, W)`, um mapa de pontuação por classe. `argmax(0)` sobre a dimensão das classes escolhe, para cada pixel, o id de maior pontuação, e `.byte().cpu().numpy()` transforma em uma máscara `uint8` do NumPy.

```python
# 11_inferencia_fcn_uma_imagem.py
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).eval()
preprocess = weights.transforms()

img_pil = Image.open(ROOT / "imagens" / "cena_01.png").convert("RGB")
x = preprocess(img_pil).unsqueeze(0)

with torch.no_grad():
    out = model(x)["out"][0]

mask = out.argmax(0).byte().cpu().numpy()
```

A máscara resultante não tem o tamanho da cena: como o lado menor foi levado a 520, ela sai com 520 pixels de altura e largura proporcional (cerca de 924 para a cena de 960 por 540). Os scripts seguintes precisam redimensionar.

Onde aparece: `11_inferencia_fcn_uma_imagem.py` (grava `saidas/11_mask_fcn.png`), `15_processar_5_imagens_fcn.py`, `18_medir_tempo_segmentacao.py`.

### 3.12 Colorir e sobrepor a saída do FCN

A máscara do FCN tem ids de 0 a 20 e o script 12 não usa a paleta didática, porque as classes são outras. Ele gera uma paleta aleatória de 256 cores com `np.random.default_rng(1)`, semente fixa para que as cores sejam as mesmas em toda execução, e indexa `palette[mask]` como no conceito 3.3. O script 13 lê a máscara colorida, redimensiona para o tamanho da cena com `cv2.resize` (largura e altura na ordem do OpenCV) e sobrepõe com `addWeighted`.

```python
# 12_colorir_saida_fcn.py
rng = np.random.default_rng(1)
palette = rng.integers(0, 255, (256, 3), dtype=np.uint8)
color = palette[mask]
```

```python
# 13_overlay_fcn.py
color = cv2.resize(color, (img.shape[1], img.shape[0]))
out = cv2.addWeighted(img, 0.65, color, 0.35, 0)
```

Onde aparece: `12_colorir_saida_fcn.py` (grava `saidas/12_mask_fcn_colorida.png`), `13_overlay_fcn.py` (grava `saidas/13_overlay_fcn.png`), `15_processar_5_imagens_fcn.py`.

### 3.13 Contagem de classes com `np.unique`

Para a máscara do FCN, cujos ids não estão no dicionário didático, o script 14 conta as classes com `np.unique(mask, return_counts=True)`, que devolve os valores distintos e quantas vezes cada um aparece. Só classes acima de 0,5 % são impressas. Na cena sintética, o resultado costuma ser quase 100 % de `classe_id=0`, o fundo, porque a rede não reconhece os desenhos geométricos como objetos reais.

```python
# 14_area_classes_fcn.py
vals, counts = np.unique(mask, return_counts=True)

for v, c in zip(vals, counts):
    pct = 100 * c / mask.size
    if pct > 0.5:
        print(f"classe_id={int(v):3d}: {pct:5.2f}%")
```

Onde aparece: `14_area_classes_fcn.py`.

### 3.14 Processamento em lote das cinco cenas

O script 15 junta 11, 12 e 13 em um laço: para cada caminho devolvido por `salvar_cenas`, abre com Pillow, pré-processa, infere sem gradiente, colore com a paleta aleatória, redimensiona e sobrepõe. O modelo é carregado uma vez só, fora do laço, o que é a prática certa: carregar a rede é a parte cara.

```python
# 15_processar_5_imagens_fcn.py
for p in paths:
    pil = Image.open(p).convert("RGB")
    x = preprocess(pil).unsqueeze(0)

    with torch.no_grad():
        mask = model(x)["out"][0].argmax(0).byte().cpu().numpy()

    img = cv2.imread(str(p))
    color = cv2.resize(palette[mask], (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 0.65, color, 0.35, 0)

    cv2.imwrite(str(ROOT / "saidas" / f"15_overlay_{p.stem}.png"), overlay)
```

Onde aparece: `15_processar_5_imagens_fcn.py` (grava `saidas/15_overlay_cena_01.png` a `15_overlay_cena_05.png`).

### 3.15 DeepLabV3-ResNet50

DeepLabV3 é uma arquitetura de segmentação mais recente que a FCN. Usa o mesmo tronco ResNet50, mas acrescenta o módulo ASPP (Atrous Spatial Pyramid Pooling), que aplica convoluções dilatadas com várias taxas para capturar contexto em múltiplas escalas sem perder resolução. No `torchvision` a chamada é idêntica à da FCN, com `deeplabv3_resnet50` e `DeepLabV3_ResNet50_Weights.DEFAULT`, e as 21 classes são as mesmas. O script 17 só carrega e imprime; o desafio pede para comparar as duas redes na mesma imagem.

```python
# 17_carregar_deeplabv3.py
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(weights=weights).eval()

print("Modelo carregado: DeepLabV3-ResNet50")
print("Classes:", len(weights.meta.get("categories", [])))
```

Onde aparece: `17_carregar_deeplabv3.py`.

### 3.16 Latência com aquecimento

Medir o tempo da primeira inferência engana: ela inclui alocação de memória e inicialização de kernels. O script 18 roda o modelo duas vezes sem cronometrar (aquecimento) e só então mede cinco execuções seguidas com `time.perf_counter`, dividindo o total por 5. Em CPU, uma FCN-ResNet50 em entrada de 520 pixels leva centenas de milissegundos a alguns segundos, o que já responde à pergunta da aula sobre embarcado.

```python
# 18_medir_tempo_segmentacao.py
for _ in range(2):
    with torch.no_grad():
        model(x)

t0 = time.perf_counter()
n = 5

for _ in range(n):
    with torch.no_grad():
        model(x)

lat = (time.perf_counter() - t0) * 1000 / n
print(f"Latência média FCN: {lat:.2f} ms")
```

Onde aparece: `18_medir_tempo_segmentacao.py`.

### 3.17 Memória do processo com `psutil`

Latência não é o único custo. `psutil.Process(os.getpid())` pega o processo atual e `memory_info().rss` devolve o Resident Set Size, a memória física ocupada, em bytes. O script 19 só imprime o valor do próprio processo vazio; o desafio é medir antes e depois de carregar a FCN, o que mostra o peso da rede em RAM.

```python
# 19_medir_memoria_processo.py
process = psutil.Process(os.getpid())
mem = process.memory_info().rss / (1024 ** 2)

print(f"Memória atual do processo: {mem:.2f} MB")
```

Onde aparece: `19_medir_memoria_processo.py`.

### 3.18 Tabela comparativa com `tabulate`

O script 20 resume o confronto entre as duas abordagens em uma tabela impressa com `tabulate`, usando `tablefmt="github"`, que produz Markdown pronto para colar no relatório. Os valores são qualitativos de propósito: o desafio é substituir velocidade e memória pelas medições dos scripts 18, 19 e 26.

```python
# 20_tabela_comparativa_hsv_fcn.py
linhas = [
    ["HSV por cor", "baixo", "muito alta", "baixo", "falha com iluminação/cor parecida"],
    ["FCN/DeepLab", "alto", "média/baixa em CPU", "alto", "entende contexto e múltiplas classes"],
]

print(tabulate(
    linhas,
    headers=["Técnica", "complexidade", "velocidade", "memória", "observação"],
    tablefmt="github"
))
```

Onde aparece: `20_tabela_comparativa_hsv_fcn.py`.

### 3.19 IoU entre máscaras binárias

Intersection over Union é a métrica padrão de segmentação: área da interseção dividida pela área da união entre a máscara prevista e a máscara verdadeira. Vale 1 quando são idênticas e 0 quando não se tocam. O script 21 compara a máscara HSV (convertida em booleano com `> 0`) com a classe pista da máscara ideal (`ideal == 2`), usando `np.logical_and` e `np.logical_or` e somando os booleanos. É o mesmo IoU que a aula 13 aplica a caixas, só que pixel a pixel.

```python
# 21_iou_mascara_hsv_vs_ideal.py
hsv = hsv_pista(img) > 0
pista = ideal == 2

inter = np.logical_and(hsv, pista).sum()
union = np.logical_or(hsv, pista).sum()

print(f"IoU HSV para classe pista: {inter / union:.3f}")
```

Onde aparece: `21_iou_mascara_hsv_vs_ideal.py`.

### 3.20 Diagrama do pipeline em Mermaid

O script 22 escreve o pipeline completo da disciplina como um `flowchart LR` na sintaxe Mermaid, que o GitHub e vários editores renderizam como diagrama: calibração, pré-processamento, HSV, ORB, detecção clássica, detecção profunda, rastreamento, segmentação semântica e decisão. O texto é gravado com `Path.write_text` em `relatorio/pipeline.mmd`. Como nenhum script cria `relatorio/`, o repositório já traz a pasta com um `.gitkeep`.

```python
# 22_criar_diagrama_pipeline_mermaid.py
diagrama = """flowchart LR
A[Calibração] --> B[Pré-processamento]
B --> C[Segmentação HSV]
B --> D[ORB]
D --> E[Detecção clássica]
E --> F[Detecção profunda]
F --> G[Rastreamento]
G --> H[Segmentação semântica]
H --> I[Decisão de percepção]
"""

Path(ROOT / "relatorio" / "pipeline.mmd").write_text(diagrama, encoding="utf-8")
```

Onde aparece: `22_criar_diagrama_pipeline_mermaid.py`.

### 3.21 Correção de distorção com `cv2.undistort`

O primeiro bloco do pipeline é a calibração da aula 16. O script 23 simula o passo com uma matriz intrínseca didática `K` (focal 800 pixels, centro no meio da imagem) e coeficientes de distorção `dist` com `k1 = -0.10` e `k2 = 0.03`, e aplica `cv2.undistort`. Como a cena sintética não tem distorção real, o resultado é uma leve deformação, apenas para mostrar onde a correção entra. O desafio manda trocar por `K` e `dist` calibrados de verdade.

```python
# 23_simular_undistort.py
K = np.array([[800, 0, w/2], [0, 800, h/2], [0, 0, 1]], dtype=np.float32)
dist = np.array([-0.10, 0.03, 0, 0, 0], dtype=np.float32)

und = cv2.undistort(img, K, dist)
```

Onde aparece: `23_simular_undistort.py` (grava `saidas/23_undistort_didatico.png`).

### 3.22 Pontos de interesse com ORB

ORB (Oriented FAST and Rotated BRIEF) detecta cantos e descreve cada um com um vetor binário, e é o descritor livre e rápido que as aulas 03, 07 e 08 já usaram. `cv2.ORB_create(nfeatures=300)` limita o número de pontos, `detectAndCompute` devolve os keypoints e a matriz de descritores (300 por 32 bytes), e `cv2.drawKeypoints` com `DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` desenha cada ponto com círculo proporcional à escala e traço da orientação. Na cena sintética os pontos se concentram nos cantos dos retângulos e das faixas.

```python
# 24_extrair_orb.py
orb = cv2.ORB_create(nfeatures=300)
kp, des = orb.detectAndCompute(gray, None)

out = cv2.drawKeypoints(
    img, kp, None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
```

Onde aparece: `24_extrair_orb.py` (grava `saidas/24_orb.png`), `26_integrar_hsv_orb_detector.py` (com 250 features).

### 3.23 Detector clássico com Canny e contornos

Antes de um detector profundo, o professor mostra o detector mais simples possível: bordas com `cv2.Canny(gray, 80, 160)` (limiares baixo e alto da histerese), contornos externos com `findContours` e uma caixa `boundingRect` para cada contorno cuja área de caixa passe de 1500 pixels. Ele não sabe o que é cada objeto, só que ali existe uma forma fechada com bordas. O desafio é substituí-lo por Haar ou HOG com SVM em imagens reais.

```python
# 25_detector_classico_simples.py
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 160)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w * h > 1500:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
```

Onde aparece: `25_detector_classico_simples.py` (grava `saidas/25_detector_classico.png`), `26_integrar_hsv_orb_detector.py` (só o Canny, para cronometrar).

### 3.24 Integração das técnicas clássicas com cronômetro

O script 26 executa HSV, ORB e Canny em sequência e mede cada etapa com o par `cronometro` e `ms` de `seg_utils.py`, que embrulha `time.perf_counter`. O resultado visual mistura os keypoints verdes com a máscara HSV pintada de amarelo por indexação booleana, e o dicionário de tempos é impresso. É o esqueleto de medição que o script 29 preenche com valores e o 30 transforma em tabela.

```python
# 26_integrar_hsv_orb_detector.py
t0 = cronometro()
hsv = hsv_pista(img)
tempos["HSV"] = ms(t0)

t0 = cronometro()
kp, _ = cv2.ORB_create(250).detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
tempos["ORB"] = ms(t0)

t0 = cronometro()
_ = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)
tempos["Canny/detector"] = ms(t0)

vis = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
vis[hsv > 0] = (0.6 * vis[hsv > 0] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)
```

Onde aparece: `26_integrar_hsv_orb_detector.py` (grava `saidas/26_integrado_classico.png`), `seg_utils.py` (`cronometro`, `ms`), `27_pipeline_com_segmentacao_semantica.py`.

### 3.25 Pipeline semântico e frame final anotado

Os scripts 27 e 28 fecham a parte visual usando a máscara ideal no lugar da máscara do FCN, para que o pipeline rode rápido e sem rede. O 27 cronometra o overlay e imprime as porcentagens por classe. O 28 escreve as porcentagens sobre o próprio frame com `cv2.putText`, uma linha a cada 25 pixels, em branco com espessura 2. O desafio do 27 é trocar a máscara didática pela prevista pelo FCN, o que exige redimensionar e remapear ids.

```python
# 28_frame_final_anotado.py
out = sobrepor(img, mask)
y = 25

for k, v in porcentagens(mask).items():
    cv2.putText(out, f"{k}: {v:.1f}%", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y += 25
```

Onde aparece: `27_pipeline_com_segmentacao_semantica.py` (grava `saidas/27_pipeline_semantico.png`), `28_frame_final_anotado.py` (grava `saidas/28_frame_final_anotado.png`).

### 3.26 Métricas em JSON e tabela em Markdown

Relatório reprodutível exige dados em arquivo. O script 29 grava um dicionário de tempos por etapa em `relatorio/metricas_demo.json` com `json.dumps(indent=2, ensure_ascii=False)`, e o próprio JSON avisa que os valores são demonstrativos. O script 30 lê esse JSON, seleciona as chaves terminadas em `_ms` e monta uma tabela Markdown linha a linha, gravando `relatorio/tabela_metricas.md`. A coluna de complexidade fica como "preencher" de propósito.

```python
# 30_gerar_tabela_markdown.py
m = json.loads(Path(ROOT / "relatorio" / "metricas_demo.json").read_text(encoding="utf-8"))

linhas = [
    "| Técnica | Tempo médio (ms) | Complexidade | Observação |",
    "|---|---:|---|---|"
]

for k, v in m.items():
    if k.endswith("_ms"):
        linhas.append(f'| {k.replace("_ms", "")} | {v:.1f} | preencher | medir em hardware alvo |')
```

Onde aparece: `29_metricas_json.py`, `30_gerar_tabela_markdown.py`.

### 3.27 Análise para embarcado, arquitetura, lacunas e entrega

Os cinco últimos scripts não processam imagem: são texto e listas que organizam a discussão. O 31 argumenta que, com orçamento de 5 W, HSV, ORB e filtros simples podem rodar continuamente, enquanto redes profundas pedem redução de resolução, quantização, execução intermitente ou acelerador dedicado. O 32 lista as seis etapas da arquitetura de percepção urbana proposta (calibração, HSV, ORB, detector profundo, rastreamento, segmentação semântica). O 33 aponta três lacunas para a "DR4" (fusão de sensores, planejamento com percepção temporal, validação em cenários reais). O 34 imprime os primeiros 1000 caracteres do modelo de relatório, se existir. O 35 imprime o checklist de 11 evidências que a entrega deve conter.

```python
# 32_arquitetura_percepcao_urbana.py
arquitetura = [
    "Calibração e undistort",
    "Segmentação HSV para pistas/faixas simples",
    "ORB para pontos de referência",
    "Detector profundo para objetos dinâmicos",
    "Rastreamento por ID para consistência temporal",
    "Segmentação semântica para área navegável"
]
```

Onde aparece: `31_analise_5w_embarcado.py`, `32_arquitetura_percepcao_urbana.py`, `33_lacunas_para_dr4.py`, `34_gerar_relatorio_base.py`, `35_checklist_final.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `seg_utils.py` | Cena sintética com máscara ideal, paleta, overlay, porcentagens, HSV da pista, painel, cronômetro | importado | nada | cv2, numpy |
| `01_criar_cenas_sinteticas.py` | Gera 5 cenas de rua e suas máscaras | nada | `imagens/cena_01.png` a `cena_05.png` e `_mask.png` | `seg_utils.py` |
| `02_visualizar_mascara_ideal.py` | Colore a máscara ideal da cena 1 com a paleta | cenas (regera) | `saidas/02_mascara_colorida.png` | `seg_utils.py` |
| `03_sobrepor_mascara_original.py` | Overlay da máscara ideal sobre a cena, alpha 0,45 | cenas (regera) | `saidas/03_overlay_semantico.png` | `seg_utils.py` |
| `04_calcular_area_por_classe.py` | Imprime o percentual de cada classe na máscara ideal | cenas (regera) | texto no terminal | `seg_utils.py` |
| `05_segmentacao_hsv_pista.py` | Máscara binária da pista por `inRange` em HSV | cenas (regera) | `saidas/05_hsv_pista.png` | `seg_utils.py` |
| `06_limpeza_morfologica.py` | Abertura e fechamento 7x7 na máscara HSV | cenas (regera) | `saidas/06_hsv_limpa.png` | `seg_utils.py` |
| `07_overlay_hsv.py` | Máscara HSV em amarelo sobre a cena | cenas (regera) | `saidas/07_overlay_hsv.png` | `seg_utils.py` |
| `08_painel_hsv_vs_mascara_ideal.py` | Painel: original, HSV e máscara ideal | cenas (regera) | `saidas/08_comparacao_hsv_semantica.png` | `seg_utils.py` |
| `09_preparar_transform_torchvision.py` | Monta e imprime `Compose(ToTensor, Normalize)` | nada | texto | torchvision |
| `10_carregar_fcn_resnet50.py` | Carrega FCN-ResNet50 e imprime as classes | pesos (baixa na 1ª vez) | texto | torch, torchvision |
| `11_inferencia_fcn_uma_imagem.py` | Segmenta a cena 1 com a FCN e salva a máscara de ids | cenas (regera), pesos | `saidas/11_mask_fcn.png` | torch, torchvision, Pillow, `seg_utils.py` |
| `12_colorir_saida_fcn.py` | Colore a máscara do FCN com paleta aleatória fixa | `saidas/11_mask_fcn.png` | `saidas/12_mask_fcn_colorida.png` | 11 |
| `13_overlay_fcn.py` | Redimensiona a máscara colorida e sobrepõe à cena | cena 1, saída do 12 | `saidas/13_overlay_fcn.png` | 01, 11, 12 |
| `14_area_classes_fcn.py` | Percentual por id na máscara do FCN, acima de 0,5 % | `saidas/11_mask_fcn.png` | texto | 11 |
| `15_processar_5_imagens_fcn.py` | Inferência, coloração e overlay nas 5 cenas | cenas (regera), pesos | `saidas/15_overlay_cena_01.png` a `_05.png` | torch, torchvision, Pillow, `seg_utils.py` |
| `16_painel_semantica_vs_hsv.py` | Painel: original, HSV e overlay do FCN | cena 1, saída do 13 | `saidas/16_painel_semantica_vs_hsv.png` | 01, 11, 12, 13 |
| `17_carregar_deeplabv3.py` | Carrega DeepLabV3-ResNet50 e imprime o número de classes | pesos (baixa na 1ª vez) | texto | torch, torchvision |
| `18_medir_tempo_segmentacao.py` | Latência média da FCN com 2 aquecimentos e 5 medições | cenas (regera), pesos | texto | torch, torchvision, Pillow, `seg_utils.py` |
| `19_medir_memoria_processo.py` | RSS do processo atual em MB | nada | texto | psutil |
| `20_tabela_comparativa_hsv_fcn.py` | Tabela qualitativa HSV contra FCN em formato github | nada | texto | tabulate |
| `21_iou_mascara_hsv_vs_ideal.py` | IoU da máscara HSV contra a classe pista ideal | cenas (regera) | texto | `seg_utils.py` |
| `22_criar_diagrama_pipeline_mermaid.py` | Grava o pipeline como flowchart Mermaid | nada | `relatorio/pipeline.mmd` | pasta `relatorio/` |
| `23_simular_undistort.py` | `cv2.undistort` com K e dist didáticos | cenas (regera) | `saidas/23_undistort_didatico.png` | `seg_utils.py` |
| `24_extrair_orb.py` | ORB com 300 features e keypoints ricos | cenas (regera) | `saidas/24_orb.png` | `seg_utils.py` |
| `25_detector_classico_simples.py` | Canny 80/160, contornos e caixas com área acima de 1500 | cenas (regera) | `saidas/25_detector_classico.png` | `seg_utils.py` |
| `26_integrar_hsv_orb_detector.py` | HSV, ORB e Canny cronometrados, imagem integrada | cenas (regera) | `saidas/26_integrado_classico.png` e tempos | `seg_utils.py` |
| `27_pipeline_com_segmentacao_semantica.py` | Overlay da máscara ideal cronometrado e porcentagens | cenas (regera) | `saidas/27_pipeline_semantico.png` | `seg_utils.py` |
| `28_frame_final_anotado.py` | Overlay com as porcentagens escritas no frame | cenas (regera) | `saidas/28_frame_final_anotado.png` | `seg_utils.py` |
| `29_metricas_json.py` | Grava tempos demonstrativos por etapa em JSON | nada | `relatorio/metricas_demo.json` | pasta `relatorio/` |
| `30_gerar_tabela_markdown.py` | Converte o JSON do 29 em tabela Markdown | `relatorio/metricas_demo.json` | `relatorio/tabela_metricas.md` | 29 |
| `31_analise_5w_embarcado.py` | Imprime a análise de orçamento de 5 W | nada | texto | nada |
| `32_arquitetura_percepcao_urbana.py` | Lista as 6 etapas da arquitetura proposta | nada | texto | nada |
| `33_lacunas_para_dr4.py` | Lista 3 lacunas de pesquisa | nada | texto | nada |
| `34_gerar_relatorio_base.py` | Imprime o início do relatório modelo, se existir | `relatorio/RELATORIO_INTEGRATIVO_MODELO.md` (não veio) | texto | material faltante |
| `35_checklist_final.py` | Imprime o checklist de 11 evidências da entrega | nada | texto | nada |

## 5. Passo a passo para executar

Todos os comandos abaixo são executados de dentro de `aula_15_HSV_ORB_DeepLabV3_ResNet50/exemplos_em_python/`, com o venv ativado (ou usando `..\..\.venv\Scripts\python.exe` no lugar de `python`). Nenhum script abre janela: tudo vai para `saidas/`, `relatorio/` ou o terminal.

Passo 1. Gerar as cinco cenas. Abra `imagens/cena_01.png` e identifique céu, pista, calçadas, árvores, carros, pedestres e faixas. A `cena_01_mask.png` parece preta porque os ids vão só até 7.

```
python 01_criar_cenas_sinteticas.py
```

Passo 2. Ver a máscara ideal colorida e sobreposta, e medir a área por classe. Na cena 1, pista e céu dominam.

```
python 02_visualizar_mascara_ideal.py
python 03_sobrepor_mascara_original.py
python 04_calcular_area_por_classe.py
```

Passo 3. Segmentar a pista por HSV, limpar com morfologia, sobrepor e comparar com o gabarito. Repare no painel do 08 o que o HSV pega a mais ou a menos em relação à máscara ideal.

```
python 05_segmentacao_hsv_pista.py
python 06_limpeza_morfologica.py
python 07_overlay_hsv.py
python 08_painel_hsv_vs_mascara_ideal.py
```

Passo 4. Medir o HSV com IoU contra a classe pista. O valor sai abaixo de 1 porque a máscara HSV inclui pixels que não são pista e recorta objetos sobre ela.

```
python 21_iou_mascara_hsv_vs_ideal.py
```

Passo 5. Preparar a parte profunda. O 09 só imprime a transformação. O 10 baixa os pesos da FCN na primeira vez (cerca de 135 MB, precisa de internet) e lista as 21 classes.

```
python 09_preparar_transform_torchvision.py
python 10_carregar_fcn_resnet50.py
```

Passo 6. Inferência com a FCN na cena 1 e pós-processamento. O 11 grava a máscara de ids; 12, 13 e 14 dependem dela e param com aviso se ela não existir. Na cena sintética a máscara sai quase toda classe 0; para ver classes de verdade, troque `cena_01.png` por uma foto real de rua.

```
python 11_inferencia_fcn_uma_imagem.py
python 12_colorir_saida_fcn.py
python 13_overlay_fcn.py
python 14_area_classes_fcn.py
```

Passo 7. Lote de cinco cenas e painel comparativo. O 16 precisa da saída do 13.

```
python 15_processar_5_imagens_fcn.py
python 16_painel_semantica_vs_hsv.py
```

Passo 8. DeepLabV3 (baixa cerca de 160 MB na primeira vez), latência e memória. Anote os três números: eles entram na tabela do passo 9 e no relatório.

```
python 17_carregar_deeplabv3.py
python 18_medir_tempo_segmentacao.py
python 19_medir_memoria_processo.py
```

Passo 9. Tabela comparativa e diagrama do pipeline. O 22 grava em `relatorio/`, que já existe no repositório.

```
python 20_tabela_comparativa_hsv_fcn.py
python 22_criar_diagrama_pipeline_mermaid.py
```

Passo 10. Blocos clássicos do pipeline: undistort, ORB, detector por bordas e integração cronometrada.

```
python 23_simular_undistort.py
python 24_extrair_orb.py
python 25_detector_classico_simples.py
python 26_integrar_hsv_orb_detector.py
```

Passo 11. Pipeline semântico e frame final anotado, com a máscara ideal.

```
python 27_pipeline_com_segmentacao_semantica.py
python 28_frame_final_anotado.py
```

Passo 12. Métricas em JSON e tabela Markdown. O 30 lê o arquivo do 29.

```
python 29_metricas_json.py
python 30_gerar_tabela_markdown.py
```

Passo 13. Textos de análise e entrega. O 34 avisa que o relatório modelo não está na pasta, porque ele não veio no zip.

```
python 31_analise_5w_embarcado.py
python 32_arquitetura_percepcao_urbana.py
python 33_lacunas_para_dr4.py
python 34_gerar_relatorio_base.py
python 35_checklist_final.py
```

Ao terminar todos os passos, `saidas/` contém os PNGs 02, 03, 05, 06, 07, 08, 11, 12, 13, 15 (cinco), 16, 23, 24, 25, 26, 27 e 28, e `relatorio/` contém `pipeline.mmd`, `metricas_demo.json` e `tabela_metricas.md`.

## 6. Exercícios e itens da aula

Os scripts não numeram itens de prova. Cada um termina com um bloco `# DESAFIO DO ALUNO`, que é a atividade prática correspondente. A lista completa:

1. `01`: abra as imagens e identifique pista, calçada, vegetação, veículos e pedestres.
2. `02`: altere a paleta em `seg_utils.py` e gere novamente a máscara.
3. `03`: teste `alpha=0.25` e `alpha=0.70` e compare a legibilidade.
4. `04`: ordene as classes da maior para a menor área.
5. `05`: ajuste os limites HSV e observe falsos positivos na calçada.
6. `06`: teste kernel 3x3, 7x7 e 15x15.
7. `07`: troque a cor da máscara e justifique a escolha visual.
8. `08`: adicione títulos nas três imagens do painel.
9. `09`: explique por que normalizamos a imagem antes da rede profunda.
10. `10`: descubra em qual conjunto de dados essas classes foram treinadas.
11. `11`: rode com uma foto real de rua e compare a máscara.
12. `12`: use cores fixas para pessoa, carro, ônibus e bicicleta.
13. `13`: altere o peso do overlay e escolha a versão mais clara.
14. `14`: substitua ids numéricos pelos nomes das classes de `weights.meta`.
15. `15`: imprima também as 3 classes com maior área em cada imagem.
16. `16`: escreva no painel os títulos ORIGINAL, HSV e FCN.
17. `17`: compare visualmente FCN e DeepLabV3 na mesma imagem.
18. `18`: rode em CPU e GPU, se disponível, e compare.
19. `19`: meça antes e depois de carregar o modelo FCN.
20. `20`: substitua velocidade e memória por valores medidos na máquina usada.
21. `21`: calcule IoU para calçada criando uma máscara HSV própria.
22. `22`: acrescente uma etapa de fusão temporal antes da decisão.
23. `23`: substitua K e dist pela calibração real da câmera usada.
24. `24`: altere `nfeatures` para 100 e 1000.
25. `25`: substitua por Haar Cascade ou HOG+SVM quando usar imagens reais.
26. `26`: converta o dicionário de tempos em uma tabela formatada.
27. `27`: substitua a máscara didática pela máscara prevista pelo FCN.
28. `28`: adicione tempo de execução total no canto inferior direito.
29. `29`: gere esse JSON automaticamente a partir dos tempos medidos.
30. `30`: acrescente acurácia, memória e consumo estimado.
31. `31`: transforme o texto em três recomendações objetivas de projeto.
32. `32`: indique entrada, saída e métrica de cada etapa.
33. `33`: relacione cada lacuna com uma competência futura em robótica móvel.
34. `34`: substituir os valores demonstrativos por métricas reais coletadas.
35. `35`: marque cada item somente após gerar evidência em imagem, tabela ou texto.

Checklist de entrega impresso pelo script 35, com 11 itens: 5 imagens externas processadas; máscara semântica colorida; overlay semitransparente; percentual de área por classe; comparação HSV vs semântica; tempos por etapa; diagrama do pipeline; tabela de métricas; análise 5 W; arquitetura proposta; 3 lacunas para DR4. Cada item corresponde a um ou mais scripts da seção 4: 01 e 15; 02 e 12; 03 e 13; 04 e 14; 08, 16, 20 e 21; 18, 19 e 26; 22; 29 e 30; 31; 32; 33.

## 7. Armadilhas e dicas

- Rode sempre de dentro de `exemplos_em_python/`. Os caminhos partem de `ROOT`, mas `seg_utils` é importado como módulo local, então o Python precisa estar na pasta.
- `relatorio/` nunca é criada pelos scripts. Os scripts 22, 29 e 30 gravam nela com `write_text` e dariam `FileNotFoundError` sem a pasta. O repositório já traz `relatorio/.gitkeep`; se você apagar a pasta, recrie antes de rodar.
- Quase todos os scripts chamam `salvar_cenas(ROOT, 5)` no início e regravam as dez imagens em `imagens/`. É rápido, mas qualquer edição manual nessas imagens é perdida na próxima execução. Para testar com uma foto real, mude o caminho lido no script, não substitua `cena_01.png`.
- Os scripts 12, 13, 14 e 16 não geram nada sozinhos: dependem de `saidas/11_mask_fcn.png` (e o 13 e o 16 da saída do 12 e do 13). Rode 11, 12 e 13 nessa ordem antes.
- A primeira execução de 10, 11, 15, 17 ou 18 precisa de internet para baixar os pesos (cerca de 135 MB para a FCN e 160 MB para o DeepLabV3), que ficam em `~/.cache/torch/hub/checkpoints/`. Sem rede, esses scripts falham com erro de download.
- Na cena sintética a máscara do FCN sai quase toda classe 0 (fundo). Não é defeito: a rede foi treinada com fotos reais e não reconhece retângulos coloridos como carros ou pessoas. O overlay dos scripts 13 e 15 só fica interessante com fotos de rua.
- Os ids do FCN (0 a 20, rótulos do Pascal VOC) não têm relação com os ids didáticos de `CLASSES_DIDATICAS` (0 a 7). Não use `colorir_mascara` nem `porcentagens` na saída da rede sem remapear; por isso o 12 usa paleta aleatória e o 14 usa `np.unique`.
- A máscara do FCN não tem o tamanho da cena. `weights.transforms()` leva o lado menor para 520 pixels, então `11_mask_fcn.png` sai com 520 de altura. Os scripts 13 e 15 fazem `cv2.resize` para o tamanho da imagem; se você combinar a máscara com a cena por conta própria, faça o mesmo.
- O 11 abre a imagem com Pillow e converte para RGB. Se substituir por `cv2.imread`, lembre que o OpenCV entrega BGR e a rede espera RGB; troque os canais antes de normalizar.
- `torch.no_grad()` é obrigatório na inferência. Sem ele o PyTorch guarda o grafo para retropropagação e o consumo de memória sobe bastante.
- A latência do 18 é em CPU, em uma entrada de 520 pixels. Valores de centenas de milissegundos a alguns segundos são normais; é exatamente o argumento do 31 sobre embarcado. O 18 usa `time.perf_counter`, não o cronômetro de `seg_utils`.
- O 19 mede a memória de um processo que não carregou nenhuma rede. O número interessante é o do desafio: rodar a mesma medição antes e depois de `fcn_resnet50(...)`.
- O 20 precisa de `tabulate`, que não aparece em nenhum comentário do professor. Ele já está no `requirements.txt` da raiz.
- O 25 desenha as caixas na própria `img` lida, não em uma cópia. Como a imagem em disco só é regravada pelo `salvar_cenas`, isso não corrompe nada, mas repare que a variável não serve mais como original depois do laço.
- O 27 e o 28 usam a máscara ideal, não a do FCN. O tempo impresso pelo 27 é só o do overlay, não de uma segmentação de verdade.
- Os valores do 29 são demonstrativos e o próprio JSON diz isso no campo `observacao`. A tabela do 30 herda esses números; substitua pelos tempos dos scripts 18 e 26 antes de colocar em qualquer relatório.
- O 34 espera `relatorio/RELATORIO_INTEGRATIVO_MODELO.md`, que não veio no zip. Sem ele, o script imprime "Relatório modelo está na pasta relatorio." e termina. Se o professor disponibilizar o arquivo, basta colocá-lo em `relatorio/`.
- "DR4" aparece nos scripts 33 e 35 sem definição no material. Trate como a referência do professor a uma etapa ou disciplina futura de robótica e confirme com ele o significado.
- Os limites de `hsv_pista` foram ajustados para o cinza `(90, 90, 90)` da cena sintética. Em fotos reais, asfalto molhado, sombras e concreto claro saem da faixa; o desafio do 05 e o IoU do 21 servem para sentir isso.
- Parâmetros que valem a pena variar: `lower` e `upper` em `hsv_pista`, o tamanho do kernel do 06, `alpha` nos overlays, `nfeatures` do ORB, os limiares do Canny e a área mínima 1500 do 25, e os pesos `K` e `dist` do 23.

## 8. Glossário

| Termo | Significado |
|---|---|
| Segmentação semântica | Atribuir a cada pixel da imagem o rótulo de uma classe (pista, carro, pessoa), sem distinguir instâncias. |
| Máscara de ids | Imagem de um canal em que o valor de cada pixel é o id da classe. |
| Máscara ideal (ground truth) | Gabarito conhecido, aqui gerado junto com a cena sintética. |
| Paleta | Tabela que associa uma cor a cada id de classe, usada para visualizar a máscara. |
| Overlay | Sobreposição semitransparente da máscara colorida sobre a imagem, feita com `cv2.addWeighted`. |
| HSV | Espaço de cor com matiz (H, 0 a 179 no OpenCV), saturação (S) e valor (V). |
| `inRange` | Função que gera uma máscara binária com os pixels dentro de uma faixa de valores por canal. |
| Morfologia | Operações sobre máscaras binárias com um elemento estruturante: erosão, dilatação, abertura, fechamento. |
| Abertura (MORPH_OPEN) | Erosão seguida de dilatação. Remove pontos pequenos. |
| Fechamento (MORPH_CLOSE) | Dilatação seguida de erosão. Preenche buracos e une fragmentos. |
| Kernel | Elemento estruturante, aqui uma matriz 7x7 de uns. |
| IoU | Intersection over Union. Interseção dividida pela união entre máscara prevista e máscara verdadeira; 1 é perfeito. |
| Tensor | Array multidimensional do PyTorch. Uma imagem vira um tensor de forma canal por altura por largura. |
| `ToTensor` | Transformação que converte imagem PIL em tensor float de 0 a 1 e reordena os eixos. |
| `Normalize` | Transformação que subtrai a média e divide pelo desvio padrão por canal. |
| Estatísticas do ImageNet | Média (0,485, 0,456, 0,406) e desvio (0,229, 0,224, 0,225) usadas no pré-treino dos troncos ResNet. |
| FCN | Fully Convolutional Network. Rede de segmentação feita só de convoluções, que produz um mapa de classes do tamanho da entrada. |
| ResNet50 | Tronco (backbone) convolucional de 50 camadas com conexões residuais, pré-treinado no ImageNet. |
| DeepLabV3 | Arquitetura de segmentação que acrescenta ao tronco o módulo ASPP com convoluções dilatadas. |
| ASPP | Atrous Spatial Pyramid Pooling. Convoluções dilatadas em várias taxas para capturar contexto em múltiplas escalas. |
| Pesos pré-treinados | Parâmetros já ajustados em um dataset grande, baixados pelo `torchvision` e usados sem treino local. |
| Pascal VOC | Conjunto de 20 classes de objetos (person, car, bus, bicycle, entre outras) mais fundo, usado nos rótulos das redes desta aula. |
| COCO | Dataset grande de cenas comuns; os pesos usados foram treinados em um subconjunto dele com rótulos VOC. |
| `weights.transforms()` | Pré-processamento oficial dos pesos: resize do lado menor para 520, tensor e normalização. |
| `weights.meta` | Metadados dos pesos, incluindo a lista `categories`. |
| `.eval()` | Modo de inferência do modelo: desliga dropout e congela batch normalization. |
| `torch.no_grad()` | Contexto que desliga o cálculo de gradientes, economizando memória na inferência. |
| `unsqueeze(0)` | Adiciona a dimensão de lote a um tensor de uma imagem. |
| `argmax(0)` | Índice da maior pontuação ao longo da dimensão das classes; produz a máscara de ids. |
| Latência | Tempo de uma inferência, medido aqui com `time.perf_counter` após aquecimento. |
| Aquecimento (warm-up) | Execuções descartadas antes de medir, para excluir custos de inicialização. |
| RSS | Resident Set Size. Memória física ocupada pelo processo, lida com `psutil`. |
| ORB | Oriented FAST and Rotated BRIEF. Detector e descritor binário de pontos de interesse, rápido e livre. |
| Keypoint | Ponto de interesse com posição, escala e orientação. |
| Canny | Detector de bordas com dois limiares (histerese). |
| Contorno | Curva que delimita uma região conectada em uma imagem binária. |
| Bounding box | Retângulo alinhado aos eixos que envolve um contorno, dado por `cv2.boundingRect`. |
| Undistort | Correção da distorção da lente usando a matriz intrínseca K e os coeficientes de distorção. |
| Matriz intrínseca K | Matriz 3x3 com focais e ponto principal da câmera. |
| Coeficientes de distorção | Vetor com k1, k2, p1, p2, k3; k1 e k2 são a distorção radial. |
| Mermaid | Linguagem de texto para diagramas, renderizada pelo GitHub e por vários editores. |
| Pipeline de percepção | Sequência de etapas que transforma frames em decisões: calibração, segmentação, features, detecção, rastreamento. |
| Orçamento de 5 W | Limite de potência elétrica típico de um computador embarcado, que restringe quais técnicas podem rodar continuamente. |
| Quantização | Redução da precisão numérica dos pesos (por exemplo de float32 para int8) para acelerar e economizar memória. |
| DR4 | Sigla usada pelo professor nos scripts 33 e 35 para a etapa seguinte da formação; não definida no material. |
| Checklist de entrega | Lista de 11 evidências (imagens, tabelas e textos) impressa pelo script 35. |
