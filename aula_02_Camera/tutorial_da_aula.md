# Aula 02 – Câmera

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

A aula 02 parte de uma fotografia tirada pela webcam e a usa como matéria-prima para três blocos de processamento de imagem: espaços de cor (HSV e LAB, com manipulação de saturação), realce de nitidez (sharpening por kernel e unsharp masking, com uma métrica objetiva de nitidez) e segmentação clássica (limiarização global, adaptativa e Otsu, detector de bordas Canny, morfologia e contornos classificados por área). Os scripts continuam a numeração da aula 01, indo do Exemplo 09 ao Exemplo 19.

O fio condutor é a imagem-base `imagem_base.png`. O Exemplo 09 a captura e salva na própria pasta dos scripts. Todos os exemplos seguintes, exceto o 18, carregam esse arquivo por meio de uma função `obter_imagem_base` repetida em cada script, que abre a câmera caso o arquivo ainda não exista. Assim, cada exemplo é independente, mas todos trabalham sobre a mesma foto, o que permite comparar os resultados. Os Exemplos 12 e 19 são as soluções integradas dos Exercícios 2 e 3, respectivamente.

## 2. Ambiente e pré-requisitos

- Bibliotecas de terceiros: `cv2` (OpenCV) e `numpy`. A biblioteca padrão fornece `pathlib`.
- Instalação:

```
pip install opencv-python numpy
```

- Não há material faltando para esta aula. Ela é autossuficiente porque a imagem-base é gerada pela webcam.
- Webcam obrigatória ao menos uma vez, no Exemplo 09, para produzir `imagem_base.png`. O Exemplo 18 sempre usa a câmera, sem ler o arquivo. Os demais só abrem a câmera se `imagem_base.png` não existir.
- A imagem-base é gravada e lida ao lado dos scripts, via `Path(__file__).resolve().parent`, portanto o diretório de trabalho não importa. Ainda assim, o mais simples é rodar de dentro de `aula_02_Camera/exemplos_em_python/`.
- Os scripts vieram como documentos Word e foram convertidos para `.py`. O Python usado pelo professor não está registrado para esta aula. Qualquer Python 3 recente com OpenCV serve.
- A imagem capturada é ampliada se tiver menos de 480 x 480 pixels, portanto uma webcam de qualquer resolução funciona.

## 3. Conceitos-chave

### 3.1 Captura de vídeo com `cv2.VideoCapture` e laço de leitura

Uma webcam, para o OpenCV, é uma fonte de vídeo identificada por um índice inteiro. O índice 0 costuma ser a câmera principal. O objeto `cv2.VideoCapture` abre a fonte, `isOpened()` confirma que deu certo, `read()` devolve um par `(ret, frame)` em que `ret` indica sucesso e `frame` é um array NumPy BGR, e `release()` libera o dispositivo. O padrão de uso é um laço `while True` que lê um frame por iteração, exibe e verifica teclas.

O professor coloca o laço dentro de `try/finally` para que `cap.release()` e `cv2.destroyAllWindows()` sejam chamados mesmo em caso de erro ou cancelamento. Isso evita deixar a câmera travada para outros programas.

```python
# 09_capturar_imagem_base.py
cap = cv2.VideoCapture(indice_camera)
if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    return
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler um frame da câmera.")
            break
        ...
finally:
    cap.release()
    cv2.destroyAllWindows()
```

Onde aparece: `09_capturar_imagem_base.py`, `18_contornos_por_area.py` (função `capturar_imagem_camera`) e na função `obter_imagem_base` de `10` a `17` e `19`.

### 3.2 Exibição, teclado e sobreposição de texto

`cv2.imshow(nome_janela, imagem)` desenha uma imagem numa janela. `cv2.waitKey(ms)` espera até `ms` milissegundos por uma tecla e devolve seu código. O `& 0xFF` isola o byte baixo, o que torna o código portátil entre sistemas. A comparação com `ord("s")` e `ord("S")` aceita minúscula e maiúscula, e o valor 27 é a tecla Esc. `waitKey(1)` mantém o vídeo fluido; `waitKey(0)` bloqueia até uma tecla ser pressionada e é usado para painéis estáticos.

`cv2.putText` escreve texto na imagem. Os argumentos são a imagem, o texto, a posição do canto inferior esquerdo do texto, a fonte (`cv2.FONT_HERSHEY_SIMPLEX`), a escala, a cor em BGR, a espessura e o tipo de linha (`cv2.LINE_AA` para antisserrilhado). O professor sempre escreve sobre uma cópia (`frame.copy()`) para não contaminar o frame que será salvo ou processado.

```python
# 09_capturar_imagem_base.py
exibicao = frame.copy()
cv2.putText(exibicao, f"Resolucao atual: {largura}x{altura}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("Captura da imagem-base", exibicao)
tecla = cv2.waitKey(1) & 0xFF
if tecla in (ord("s"), ord("S")):
    frame_capturado = frame.copy()
    break
if tecla in (ord("q"), ord("Q"), 27):
    break
```

Onde aparece: todos os scripts.

### 3.3 Redimensionamento com `cv2.resize` e interpolação

`cv2.resize(imagem, (largura, altura), interpolation=...)` muda o tamanho da imagem. Repare que o tamanho é dado como `(largura, altura)`, ao contrário de `shape`, que devolve `(altura, largura, canais)`. O método de interpolação decide como os novos pixels são calculados: `cv2.INTER_CUBIC` produz ampliações suaves e é usado quando a foto é menor que 480 x 480; `cv2.INTER_AREA` é o recomendado para reduzir, porque faz média das áreas e evita serrilhado. Também é possível informar fatores `fx` e `fy` em vez do tamanho, passando `None` como tamanho, como faz `redimensionar_para_exibicao` no Exemplo 18.

A garantia de 480 x 480 é feita calculando uma escala única `max(480 / altura, 480 / largura)` para ampliar proporcionalmente, sem distorcer.

```python
# 09_capturar_imagem_base.py
altura, largura = frame_capturado.shape[:2]
if altura < 480 or largura < 480:
    escala = max(480 / altura, 480 / largura)
    nova_largura = int(round(largura * escala))
    nova_altura = int(round(altura * escala))
    frame_capturado = cv2.resize(frame_capturado, (nova_largura, nova_altura),
                                 interpolation=cv2.INTER_CUBIC)

# 10_canais_hsv_lab.py
def redimensionar_quadrado(imagem, lado: int = 320):
    return cv2.resize(imagem, (lado, lado), interpolation=cv2.INTER_AREA)
```

Onde aparece: `09` e todos os demais, via `redimensionar_quadrado` ou `redimensionar_para_exibicao`.

### 3.4 Leitura e gravação de imagens: `cv2.imread` e `cv2.imwrite`

`cv2.imwrite(caminho, imagem)` grava um array como arquivo de imagem e devolve `True` ou `False`. `cv2.imread(caminho, cv2.IMREAD_COLOR)` lê o arquivo como BGR de três canais e devolve `None` se ele não existir, sem lançar exceção. É esse `None` que a função `obter_imagem_base` usa para decidir se abre a câmera. O caminho é construído com `pathlib.Path` a partir da pasta do script, o que torna o resultado independente do diretório atual.

```python
# 10_canais_hsv_lab.py, função obter_imagem_base
caminho = Path(__file__).resolve().parent / nome_arquivo
imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
if imagem is not None:
    print(f"Imagem carregada de: {caminho}")
    return imagem
# ... senão captura pela webcam e no fim:
cv2.imwrite(str(caminho), imagem_capturada)
```

Onde aparece: `09_capturar_imagem_base.py` e `obter_imagem_base` em `10` a `17` e `19`.

### 3.5 A imagem como array NumPy: `shape`, `dtype` e cópia

Um frame do OpenCV é um `numpy.ndarray` de forma `(altura, largura, 3)` e tipo `uint8`, ou seja, inteiros de 0 a 255. O professor imprime `shape` e `dtype` após salvar a imagem e após converter para outros espaços de cor, mostrando que a conversão não altera as dimensões. Operações aritméticas de NumPy funcionam diretamente: `imagens[0] * 0` cria uma célula preta do mesmo tamanho para completar uma grade. A cópia com `.copy()` é essencial sempre que uma função vai desenhar sobre a imagem.

```python
# 10_canais_hsv_lab.py
print(f"BGR: {imagem.shape}")
print(f"HSV: {hsv.shape}")
print(f"LAB: {lab.shape}")
celula_vazia = imagens[0] * 0
```

Onde aparece: `09`, `10`, `12`, `18`.

### 3.6 Espaços de cor HSV e LAB com `cv2.cvtColor` e `cv2.split`

O OpenCV carrega imagens coloridas em BGR, não RGB. `cv2.cvtColor(imagem, codigo)` converte entre espaços de cor. Em HSV, o canal H (Hue, matiz) identifica a família da cor e no OpenCV varia de 0 a 179 para caber em um byte, S (Saturation, saturação) mede a pureza da cor de 0 a 255 e V (Value, valor) é o brilho de 0 a 255. Em LAB, L é a luminosidade perceptual, A é o eixo cromático do verde ao vermelho e B é o eixo do azul ao amarelo. HSV é útil para separar cor de brilho; LAB aproxima a percepção humana de diferença entre cores.

`cv2.split` separa os três canais em três arrays de um canal cada. Para exibir um canal ao lado de imagens coloridas é preciso convertê-lo de volta para três canais com `cv2.COLOR_GRAY2BGR`, o que a função `canal_para_bgr` faz. A operação inversa é `cv2.COLOR_HSV2BGR`.

```python
# 10_canais_hsv_lab.py
hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
canal_h, canal_s, canal_v = cv2.split(hsv)
canal_l, canal_a, canal_b = cv2.split(lab)

def canal_para_bgr(canal):
    return cv2.cvtColor(canal, cv2.COLOR_GRAY2BGR)
```

Onde aparece: `10_canais_hsv_lab.py`, `12_exercicio2_item_a_completo.py`. A conversão para cinza (`cv2.COLOR_BGR2GRAY`) aparece de `14` a `19`.

### 3.7 Montagem de painéis com `cv2.hconcat`, `cv2.vconcat` e `cv2.rectangle`

Para comparar resultados lado a lado, o professor concatena imagens: `cv2.hconcat(lista)` junta horizontalmente e `cv2.vconcat(lista)` verticalmente. As imagens precisam ter a mesma altura (hconcat) ou largura (vconcat) e o mesmo número de canais, por isso tudo passa antes por `redimensionar_quadrado` e `canal_para_bgr`. A função `adicionar_titulo` desenha uma faixa preta com `cv2.rectangle(imagem, ponto1, ponto2, cor, espessura)` usando espessura `-1` para preencher, e escreve o título em branco por cima.

```python
# 10_canais_hsv_lab.py
def adicionar_titulo(imagem, titulo: str):
    saida = imagem.copy()
    cv2.rectangle(saida, (0, 0), (saida.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(saida, titulo, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return saida

linha_1 = cv2.hconcat(imagens[0:4])
linha_2 = cv2.hconcat(imagens[4:7] + [celula_vazia])
painel = cv2.vconcat([linha_1, linha_2])
```

Onde aparece: `10` a `19`.

### 3.8 Alteração de saturação com aritmética em ponto flutuante e `np.clip`

Para deixar uma foto mais ou menos colorida sem mexer no brilho, basta multiplicar o canal S do HSV por um fator. Com fator 0 a imagem vira cinza, 0,5 desbota, 1,5 intensifica. Como o array é `uint8`, multiplicar direto por 1,5 estouraria o limite de 255 e daria a volta (por exemplo, 200 x 1,5 = 300 viraria 44). A solução é converter para `float32` com `astype`, multiplicar, limitar ao intervalo 0 a 255 com `np.clip` e só então voltar para `uint8`. A indexação `hsv[:, :, 1]` seleciona todas as linhas, todas as colunas, canal 1, que é o S.

```python
# 11_alterar_saturacao_hsv.py
def alterar_saturacao(imagem_bgr, fator: float):
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    hsv_float = hsv.astype(np.float32)
    hsv_float[:, :, 1] *= fator
    hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1], 0, 255)
    hsv_alterado = hsv_float.astype(np.uint8)
    return cv2.cvtColor(hsv_alterado, cv2.COLOR_HSV2BGR)
```

Onde aparece: `11_alterar_saturacao_hsv.py`, `12_exercicio2_item_a_completo.py` (versão condensada em três linhas).

### 3.9 Sharpening por convolução com kernel manual: `cv2.filter2D`

Uma convolução desliza uma pequena matriz (kernel) sobre a imagem e substitui cada pixel pela soma ponderada de sua vizinhança. O kernel de sharpening 3 x 3 com 9 no centro e -1 ao redor soma 1 no total, então áreas uniformes ficam iguais, mas onde há transição de intensidade o centro é reforçado e os vizinhos subtraídos, o que acentua bordas e detalhes. `cv2.filter2D(src, ddepth, kernel)` aplica o kernel; `ddepth=-1` mantém a profundidade (tipo) da imagem de entrada. O kernel é criado como array `float32` do NumPy.

```python
# 13_sharpening_filter2d.py
kernel_sharpen = np.array([
    [-1, -1, -1],
    [-1,  9, -1],
    [-1, -1, -1],
], dtype=np.float32)
nitida = cv2.filter2D(src=imagem, ddepth=-1, kernel=kernel_sharpen)
```

Onde aparece: `13_sharpening_filter2d.py`, `15_comparar_sharpening_unsharp.py`.

### 3.10 Unsharp masking: `cv2.GaussianBlur` e `cv2.addWeighted`

Unsharp masking realça a nitidez de forma indireta: cria-se uma versão borrada da imagem, que contém apenas as baixas frequências, e subtrai-se parte dela da original, o que sobra reforçado são as altas frequências, ou seja, os detalhes. `cv2.GaussianBlur(imagem, (0, 0), sigmaX=1.2)` faz o borrão; com tamanho de kernel `(0, 0)` o OpenCV calcula o tamanho a partir do desvio padrão `sigmaX`. `cv2.addWeighted(a, alfa, b, beta, gama)` calcula `a * alfa + b * beta + gama` com saturação automática em 255. A combinação `original * 2.0 + blur * (-1.0)` equivale a `original + (original - blur)`, um realce moderado.

```python
# 14_unsharp_masking_laplaciano.py
blur = cv2.GaussianBlur(imagem, (0, 0), sigmaX=1.2)
unsharp = cv2.addWeighted(imagem, 2.0, blur, -1.0, 0)
```

Onde aparece: `14_unsharp_masking_laplaciano.py`, `15_comparar_sharpening_unsharp.py`. `GaussianBlur` com kernel `(5, 5)` e sigma 0 (calculado a partir do tamanho) também aparece como pré-processamento em `16` a `19`.

### 3.11 Métrica de nitidez: variância do Laplaciano

O Laplaciano é a segunda derivada da imagem e responde forte onde a intensidade muda bruscamente. Numa imagem nítida há muitas bordas fortes, então os valores do Laplaciano se espalham bastante e a variância é alta; numa imagem borrada a variância é baixa. `cv2.Laplacian(cinza, cv2.CV_64F)` calcula o operador em ponto flutuante de 64 bits para não perder os valores negativos, e `.var()` do NumPy dá a variância. O professor alerta no Exemplo 15 que ruído também eleva a métrica, portanto ela deve ser lida junto com a inspeção visual.

```python
# 14_unsharp_masking_laplaciano.py
def variancia_laplaciano(imagem_bgr) -> float:
    cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    laplaciano = cv2.Laplacian(cinza, cv2.CV_64F)
    return float(laplaciano.var())

# 15_comparar_sharpening_unsharp.py
metodo_mais_nitido = max(metricas, key=metricas.get)
```

Onde aparece: `14_unsharp_masking_laplaciano.py`, `15_comparar_sharpening_unsharp.py`.

### 3.12 Limiarização global, adaptativa e Otsu

Limiarizar (threshold) é transformar uma imagem em cinza numa imagem binária: pixels acima de um valor viram 255, os demais viram 0. `cv2.threshold(imagem, limiar, valor_max, tipo)` devolve o limiar usado e a imagem binária. Com `cv2.THRESH_BINARY` e limiar 127, o mesmo corte vale para a imagem inteira, o que falha quando a iluminação não é uniforme.

`cv2.adaptiveThreshold(imagem, valor_max, metodo, tipo, tamanho_bloco, C)` calcula um limiar diferente para cada pixel a partir de sua vizinhança. Com `cv2.ADAPTIVE_THRESH_GAUSSIAN_C` o limiar local é a média ponderada gaussiana do bloco (aqui 31 x 31, sempre ímpar) menos a constante C (aqui 5). É o método mais robusto a sombras e gradientes de luz.

O método de Otsu (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`) analisa o histograma e escolhe automaticamente o limiar que melhor separa duas classes de intensidade. O valor de limiar informado (0) é ignorado, e o valor calculado é devolvido no primeiro elemento do retorno. Antes de qualquer limiarização, o professor suaviza com `GaussianBlur (5, 5)` para reduzir ruído.

```python
# 16_limiarizacao_global_adaptativa_otsu.py
cinza_suave = cv2.GaussianBlur(cinza, (5, 5), 0)
_, global_bin = cv2.threshold(cinza_suave, 127, 255, cv2.THRESH_BINARY)
adaptativo = cv2.adaptiveThreshold(cinza_suave, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
limiar_otsu, otsu = cv2.threshold(cinza_suave, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Limiar calculado pelo método de Otsu: {limiar_otsu:.2f}")
```

Onde aparece: `16_limiarizacao_global_adaptativa_otsu.py`, `18_contornos_por_area.py` (só Otsu), `19_exercicio3_solucao_integrada.py`.

### 3.13 Detector de bordas Canny e contagem com `cv2.countNonZero`

`cv2.Canny(imagem, limiar1, limiar2)` encontra bordas em três etapas: gradiente, supressão de não máximos (afina as bordas para um pixel de largura) e histerese com dois limiares. Pixels com gradiente acima de `limiar2` são bordas fortes e entram sempre; pixels entre `limiar1` e `limiar2` são bordas fracas e só entram se estiverem conectados a uma borda forte. Pares menores como (50, 150) detectam mais detalhes e mais ruído; pares maiores como (100, 200) ficam só com as bordas mais marcadas. `cv2.countNonZero` conta os pixels brancos, o que dá uma medida objetiva da quantidade de borda detectada.

```python
# 17_canny_dois_thresholds.py
suave = cv2.GaussianBlur(cinza, (5, 5), 0)
canny_50_150 = cv2.Canny(suave, 50, 150)
canny_100_200 = cv2.Canny(suave, 100, 200)
print(f"Pixels de borda em Canny(50, 150):  {cv2.countNonZero(canny_50_150)}")
print(f"Pixels de borda em Canny(100, 200): {cv2.countNonZero(canny_100_200)}")
```

Onde aparece: `17_canny_dois_thresholds.py`, `18_contornos_por_area.py` (apenas para exibir), `19_exercicio3_solucao_integrada.py` (aplicado à imagem binarizada).

### 3.14 Inversão de máscara com `cv2.bitwise_not`

Uma máscara binária pode ter os objetos em branco sobre fundo preto ou o contrário. `findContours` procura regiões brancas, então a polaridade importa. `cv2.bitwise_not` inverte todos os bits, trocando 0 por 255. O Exemplo 19 inverte o resultado adaptativo para que os objetos fiquem brancos; o Exemplo 18 gera as duas versões da máscara de Otsu e escolhe automaticamente a melhor.

```python
# 18_contornos_por_area.py
binaria_invertida = cv2.bitwise_not(binaria_normal)

# 19_exercicio3_solucao_integrada.py
binaria_escolhida = cv2.bitwise_not(adaptativo)
```

Onde aparece: `18_contornos_por_area.py`, `19_exercicio3_solucao_integrada.py`.

### 3.15 Morfologia matemática: abertura e fechamento com `cv2.morphologyEx`

Operações morfológicas modificam formas numa imagem binária usando um elemento estruturante, criado com `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))`. Erosão encolhe regiões brancas e dilatação as expande. Abertura (`cv2.MORPH_OPEN`) é erosão seguida de dilatação e elimina pontos brancos isolados, ou seja, ruído. Fechamento (`cv2.MORPH_CLOSE`) é dilatação seguida de erosão e tapa pequenos buracos e interrupções, ajudando a formar regiões fechadas. O parâmetro `iterations` repete a operação. No Exemplo 18 a abertura tem 1 iteração e o fechamento 2, para garantir regiões preenchidas; no 19 o fechamento é aplicado às bordas Canny para conectar falhas antes de buscar contornos.

```python
# 18_contornos_por_area.py
def preparar_mascara(mascara):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    limpa = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, kernel, iterations=2)
    return limpa
```

Onde aparece: `18_contornos_por_area.py`, `19_exercicio3_solucao_integrada.py`.

### 3.16 Contornos: `cv2.findContours`, `cv2.contourArea`, `cv2.boundingRect` e `cv2.drawContours`

Um contorno é a curva que delimita uma região branca conectada na imagem binária. `cv2.findContours(mascara, modo, metodo)` devolve a lista de contornos e a hierarquia. `cv2.RETR_EXTERNAL` retorna só os contornos externos, ignorando buracos internos. `cv2.CHAIN_APPROX_SIMPLE` guarda apenas os vértices necessários, comprimindo segmentos retos. `cv2.contourArea` calcula a área interna em pixels quadrados, `cv2.boundingRect` devolve o retângulo alinhado aos eixos `(x, y, w, h)`, `cv2.drawContours(imagem, [contorno], -1, cor, espessura)` desenha o contorno (índice -1 desenha todos da lista) e `cv2.rectangle` desenha a caixa.

O docstring do Exemplo 18 explica um problema importante: linhas finas do Canny não formam regiões fechadas, então `contourArea` devolve zero e o filtro de área descarta tudo. A solução é aplicar `findContours` sobre a máscara preenchida (Otsu com morfologia), e não sobre as bordas. O filtro exigido é área maior que 200 px², e o 18 ainda descarta regiões com 95% ou mais da imagem, que normalmente são o fundo. `max(areas_validas, default=0.0)` evita erro quando a lista está vazia.

```python
# 18_contornos_por_area.py
contornos, _ = cv2.findContours(mascara.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
for contorno in contornos:
    area = cv2.contourArea(contorno)
    if area <= AREA_MINIMA:
        continue
    if area >= 0.95 * area_imagem:
        continue
    cor = cor_por_area(area)
    cv2.drawContours(resultado, [contorno], -1, cor, 3)
    x, y, largura, altura = cv2.boundingRect(contorno)
    cv2.rectangle(resultado, (x, y), (x + largura, y + altura), cor, 2)
```

Onde aparece: `18_contornos_por_area.py`, `19_exercicio3_solucao_integrada.py`.

### 3.17 Classificação por faixa de área

Depois de medir a área, cada contorno recebe uma cor conforme a faixa: verde para pequeno (200 a 999 px²), amarelo para médio (1.000 a 4.999 px²) e vermelho para grande (5.000 px² ou mais). As cores são tuplas BGR, então `(0, 0, 255)` é vermelho. O Exemplo 18 ainda escreve a área ao lado de cada objeto com `putText`, posicionando o texto logo acima da caixa e usando `max(25, y - 8)` para não sair da imagem.

```python
# 18_contornos_por_area.py
def cor_por_area(area: float):
    if area < 1000:
        return (0, 255, 0)       # Verde: contorno pequeno.
    if area < 5000:
        return (0, 255, 255)     # Amarelo: contorno médio.
    return (0, 0, 255)           # Vermelho: contorno grande.
```

Onde aparece: `18_contornos_por_area.py`, `19_exercicio3_solucao_integrada.py`.

### 3.18 Escolha automática de polaridade da máscara

O Exemplo 18 introduz uma heurística para decidir se a máscara de Otsu deve ser usada normal ou invertida. Ambas passam por `preparar_mascara`, depois `avaliar_mascara` conta quantos contornos válidos existem (área acima de 200 px² e abaixo de 95% da imagem) e soma suas áreas. A função devolve a tupla `(quantidade, soma)`, e a comparação `avaliacao_invertida > avaliacao_normal` entre tuplas do Python prioriza a quantidade e desempata pela soma. Vence a máscara com mais regiões úteis.

```python
# 18_contornos_por_area.py
def escolher_melhor_mascara(binaria_normal, binaria_invertida):
    normal_limpa = preparar_mascara(binaria_normal)
    invertida_limpa = preparar_mascara(binaria_invertida)
    avaliacao_normal = avaliar_mascara(normal_limpa)
    avaliacao_invertida = avaliar_mascara(invertida_limpa)
    if avaliacao_invertida > avaliacao_normal:
        return invertida_limpa, "Otsu invertido"
    return normal_limpa, "Otsu normal"
```

Onde aparece: `18_contornos_por_area.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `09_capturar_imagem_base.py` | Abre a webcam, captura um frame com S, amplia para no mínimo 480 x 480 e salva | Webcam | `imagem_base.png` ao lado do script | Nada |
| `10_canais_hsv_lab.py` | Converte para HSV e LAB, separa os canais e monta painel 2 x 4 com 7 imagens | `imagem_base.png` (ou webcam) | Janela com o painel; shapes no terminal | `09` (opcional) |
| `11_alterar_saturacao_hsv.py` | Multiplica o canal S por 0, 0,5 e 1,5 e exibe os três resultados | `imagem_base.png` (ou webcam) | Janela com 3 imagens | `09` (opcional) |
| `12_exercicio2_item_a_completo.py` | Solução do Exercício 2, Item A: grade de canais e painel de saturação em duas janelas sucessivas | `imagem_base.png` (ou webcam) | Duas janelas | `09` (opcional) |
| `13_sharpening_filter2d.py` | Aplica kernel de sharpening 3 x 3 com `filter2D` e compara com a original | `imagem_base.png` (ou webcam) | Janela com 2 imagens; kernel no terminal | `09` (opcional) |
| `14_unsharp_masking_laplaciano.py` | Faz unsharp masking com blur gaussiano e mede nitidez pela variância do Laplaciano | `imagem_base.png` (ou webcam) | Janela com 2 imagens; métricas no terminal | `09` (opcional) |
| `15_comparar_sharpening_unsharp.py` | Compara original, kernel manual e unsharp, e aponta o de maior nitidez | `imagem_base.png` (ou webcam) | Janela com 3 imagens; tabela de métricas | `09` (opcional) |
| `16_limiarizacao_global_adaptativa_otsu.py` | Binariza com limiar global 127, adaptativo gaussiano e Otsu | `imagem_base.png` (ou webcam) | Janela com 3 binárias; limiar de Otsu no terminal | `09` (opcional) |
| `17_canny_dois_thresholds.py` | Aplica Canny com (50, 150) e (100, 200) e conta pixels de borda | `imagem_base.png` (ou webcam) | Janela com 3 imagens; contagens no terminal | `09` (opcional) |
| `18_contornos_por_area.py` | Captura da câmera, Otsu com escolha de polaridade, morfologia, contornos filtrados por área e coloridos | Webcam (sempre) | Painel 2 x 2: original, máscara, Canny, contornos; estatísticas no terminal | Nada (não lê `imagem_base.png`) |
| `19_exercicio3_solucao_integrada.py` | Solução integrada do Exercício 3: três limiarizações, dois Canny sobre a binária adaptativa invertida, fechamento e contornos por área | `imagem_base.png` (ou webcam) | Três janelas sucessivas; estatísticas no terminal | `09` (opcional) |

Nenhum script além do 09 grava arquivos em disco, exceto pela captura de emergência de `obter_imagem_base`, que salva `imagem_base.png` se ele não existir.

## 5. Passo a passo para executar

Entre na pasta dos scripts:

```
cd C:\visao_computacional_PVA\aula_02_Camera\exemplos_em_python
```

Passo 1. Capture a imagem-base. Aponte a câmera para uma cena com objetos de cores variadas e bom contraste com o fundo, pois a mesma foto servirá para cor, nitidez e contornos.

```
python 09_capturar_imagem_base.py
```

Aparece a janela da câmera com a resolução atual escrita em verde. Pressione S para capturar ou Q/Esc para cancelar. O terminal mostra o caminho de `imagem_base.png`, o shape e o dtype.

Passo 2. Espaços de cor (Exercício 2, Item A).

```
python 10_canais_hsv_lab.py
python 11_alterar_saturacao_hsv.py
python 12_exercicio2_item_a_completo.py
```

O 10 abre um painel com BGR original, H, S, V, L, A, B e uma célula preta. Feche com Q ou Esc. O 11 mostra a foto com saturação 0%, 50% e 150%. O 12 mostra as duas coisas em janelas sucessivas: pressione qualquer tecla para avançar da primeira para a segunda.

Passo 3. Nitidez (Exercício 2, Item B).

```
python 13_sharpening_filter2d.py
python 14_unsharp_masking_laplaciano.py
python 15_comparar_sharpening_unsharp.py
```

O 13 imprime o kernel no terminal e mostra original e resultado; feche com Q ou Esc. O 14 imprime a variância do Laplaciano antes e depois e mostra o par; feche com qualquer tecla. O 15 mostra três imagens e imprime qual método deu maior nitidez.

Passo 4. Limiarização e bordas (Exercício 3, Itens A e B).

```
python 16_limiarizacao_global_adaptativa_otsu.py
python 17_canny_dois_thresholds.py
```

O 16 imprime o limiar de Otsu e mostra as três binárias. O 17 imprime a quantidade de pixels de borda em cada par de thresholds e mostra original e os dois Canny. Ambos fecham com qualquer tecla.

Passo 5. Contornos por área.

```
python 18_contornos_por_area.py
```

Este script sempre abre a câmera. Coloque objetos sobre um fundo liso, pressione S. O terminal mostra o limiar de Otsu, a avaliação das duas máscaras, qual foi escolhida, o total de contornos, quantos passaram no filtro de 200 px² e a maior área. O painel 2 x 2 fecha com Q ou Esc. Se nenhum contorno passar, o script imprime cinco sugestões (mais contraste, aproximar o objeto, fundo liso, reduzir `AREA_MINIMA` para 50, evitar sombras).

Passo 6. Solução integrada do Exercício 3.

```
python 19_exercicio3_solucao_integrada.py
```

Três janelas em sequência, cada uma fechada com qualquer tecla: as três limiarizações, os dois Canny aplicados à binária adaptativa invertida, e a imagem original com contornos e caixas coloridas por área. O terminal informa o limiar de Otsu, a escolha do método adaptativo e as estatísticas dos contornos.

Para refazer a foto, apague `imagem_base.png` ou rode o 09 de novo.

## 6. Exercícios e itens da aula

Os scripts fazem referência ao Exercício 2 (Itens A e B) e ao Exercício 3 (Itens A e B), continuando a numeração da aula 01.

**Exercício 2, Item A: espaços de cor.** Pede sete imagens (a original BGR, os canais H, S e V e os canais L, A e B) e a alteração programática da saturação em 0%, 50% e 150%. Os títulos das janelas dos Exemplos 10 e 11 dizem "Exercicio 2A" e a célula vazia do painel diz "7 imagens solicitadas". A solução completa é `12_exercicio2_item_a_completo.py`.

**Exercício 2, Item B: nitidez.** Pede comparar sharpening por kernel manual com unsharp masking e estimar a nitidez com a variância do Laplaciano. Os Exemplos 13 e 14 fazem cada método isoladamente e `15_comparar_sharpening_unsharp.py`, cuja janela se chama "Exercicio 2B", é a comparação final.

**Exercício 3, Item A: limiarização.** Pede comparar limiar global, adaptativo e Otsu e justificar a escolha de um deles. `16_limiarizacao_global_adaptativa_otsu.py` faz a comparação (janela "Exercicio 3A") e o docstring traz a justificativa: Otsu supera um limiar manual porque usa o histograma, mas o adaptativo é mais robusto quando a iluminação varia dentro da imagem.

**Exercício 3, Item B: Canny e contornos.** Pede aplicar Canny com dois pares de thresholds à imagem binarizada escolhida, encontrar contornos, filtrar os de área acima de 200 px² e colorir por faixa de área. `17_canny_dois_thresholds.py` cobre a parte do Canny (janela "Exercicio 3B"), `18_contornos_por_area.py` é a versão corrigida da detecção de contornos, e `19_exercicio3_solucao_integrada.py` junta os Itens A e B.

## 7. Armadilhas e dicas

- **Webcam ocupada ou índice errado.** Se aparecer "Não foi possível abrir a câmera", feche outros programas que usem a webcam ou troque `indice_camera` de 0 para 1 ou 2.
- **Imagem-base ruim compromete tudo.** Como todos os exemplos usam a mesma foto, escolha uma cena com cores variadas (para HSV e saturação), detalhes finos (para nitidez) e objetos destacados sobre fundo liso (para contornos). Para refazer, apague `imagem_base.png`.
- **`import cv2` duplicado e código repetido.** De `10` a `17` e `19`, `import cv2` aparece duas vezes e o bloco `obter_imagem_base`, `adicionar_titulo`, `canal_para_bgr` e `redimensionar_quadrado` é copiado em cada script. É o código do professor, mantido por fidelidade; não causa erro.
- **`destroyWindow` com nome errado no Exemplo 12.** Após o primeiro painel, o script chama `cv2.destroyWindow("Exercício")`, mas a janela se chama "Exercicio 2A - sete imagens". Dependendo da versão do OpenCV isso pode lançar `cv2.error` ou simplesmente não fechar a primeira janela. Mantido como o professor escreveu. Se der erro, o `cv2.destroyAllWindows()` do fim não chega a rodar, então feche as janelas manualmente.
- **Exemplo 18 não usa a imagem-base.** Ele sempre captura da câmera, ao contrário dos demais. Se quiser aplicá-lo à foto salva, seria preciso trocar `capturar_imagem_camera` por `cv2.imread`.
- **Contornos com área zero.** Nunca aplique `findContours` direto sobre o Canny esperando áreas úteis: linhas de um pixel não fecham regiões. Use a máscara preenchida (Otsu ou adaptativa com morfologia), como o 18 e o 19 fazem.
- **Polaridade da máscara.** `findContours` procura regiões brancas. Se os objetos saírem pretos, inverta com `bitwise_not` ou deixe a heurística do 18 escolher.
- **Estouro de `uint8`.** Ao multiplicar canais, converta para `float32` e use `np.clip` antes de voltar para `uint8`, como no 11. Sem isso, valores acima de 255 dão a volta e produzem cores erradas.
- **`resize` recebe `(largura, altura)`** enquanto `shape` devolve `(altura, largura)`. Trocar a ordem distorce a imagem.
- **`hconcat` exige alturas e canais iguais.** Converta canais únicos com `COLOR_GRAY2BGR` e redimensione antes de concatenar, senão o OpenCV lança erro.
- **Parâmetros que valem a pena variar.** O fator de saturação no 11; o valor central do kernel (9) e `sigmaX` (1,2) no 13 e 14; o tamanho do bloco (31) e a constante C (5) do adaptativo no 16; os pares de Canny no 17; o tamanho do elemento estruturante e as iterações de fechamento, e `AREA_MINIMA`, no 18.
- **Métrica de nitidez engana.** Ruído também eleva a variância do Laplaciano. Um kernel de sharpening agressivo pode ganhar na métrica e perder na inspeção visual.

## 8. Glossário

| Termo | Significado |
|---|---|
| BGR | Ordem de canais azul, verde, vermelho usada pelo OpenCV ao carregar imagens coloridas. |
| HSV | Espaço de cor matiz (Hue), saturação (Saturation) e valor (Value, brilho). No OpenCV H vai de 0 a 179. |
| LAB | Espaço de cor perceptual com luminosidade L e dois eixos cromáticos, A (verde a vermelho) e B (azul a amarelo). |
| Canal | Uma das matrizes bidimensionais que compõem a imagem colorida, por exemplo o canal S do HSV. |
| Frame | Um quadro individual do fluxo de vídeo da câmera. |
| Escala de cinza | Imagem de um canal com intensidade de 0 (preto) a 255 (branco). |
| `uint8` | Inteiro sem sinal de 8 bits, de 0 a 255, tipo padrão dos pixels. |
| `float32` | Ponto flutuante de 32 bits, usado para cálculos intermediários sem estouro. |
| Kernel | Pequena matriz que define uma convolução ou o formato de uma operação morfológica. |
| Convolução | Operação que substitui cada pixel pela soma ponderada de sua vizinhança segundo um kernel. |
| Sharpening | Realce de nitidez que acentua bordas e detalhes. |
| Unsharp masking | Técnica de nitidez que subtrai uma versão borrada da imagem original. |
| Blur gaussiano | Suavização em que os vizinhos são ponderados por uma curva gaussiana; controlado por `sigmaX`. |
| Laplaciano | Operador de segunda derivada que responde a mudanças bruscas de intensidade. |
| Variância do Laplaciano | Métrica de nitidez: quanto maior, mais bordas fortes a imagem tem. |
| Alta frequência | Regiões da imagem com variação rápida de intensidade, como bordas e texturas finas. |
| Threshold (limiar) | Valor de corte que separa pixels em duas classes numa binarização. |
| Limiarização global | Um único limiar para a imagem inteira. |
| Limiarização adaptativa | Limiar calculado localmente para cada pixel a partir de sua vizinhança. |
| Otsu | Método que escolhe o limiar automaticamente a partir do histograma. |
| Histograma | Contagem de quantos pixels há em cada nível de intensidade. |
| Binarização | Transformação da imagem em apenas dois valores, 0 e 255. |
| Máscara | Imagem binária que marca em branco as regiões de interesse. |
| Polaridade | Se os objetos da máscara estão em branco sobre preto ou em preto sobre branco. |
| Canny | Detector de bordas com histerese de dois limiares. |
| Histerese | Regra do Canny em que bordas fracas só são aceitas se ligadas a bordas fortes. |
| Morfologia matemática | Operações sobre formas em imagens binárias usando um elemento estruturante. |
| Elemento estruturante | Kernel que define a vizinhança de uma operação morfológica; aqui uma elipse 5 x 5. |
| Erosão | Encolhe regiões brancas. |
| Dilatação | Expande regiões brancas. |
| Abertura | Erosão seguida de dilatação; remove ruído pequeno. |
| Fechamento | Dilatação seguida de erosão; fecha buracos e interrupções. |
| Contorno | Curva que delimita uma região branca conectada. |
| `RETR_EXTERNAL` | Modo de `findContours` que devolve apenas contornos externos. |
| `CHAIN_APPROX_SIMPLE` | Método que guarda apenas os vértices essenciais do contorno. |
| Bounding box | Retângulo alinhado aos eixos que envolve um contorno, obtido com `boundingRect`. |
| px² | Pixels quadrados, unidade de área de um contorno. |
| Interpolação | Forma de calcular novos pixels ao redimensionar; `INTER_CUBIC` para ampliar, `INTER_AREA` para reduzir. |
| Antisserrilhado (`LINE_AA`) | Suavização das bordas de texto e linhas desenhadas. |
| `waitKey` | Espera por tecla durante um tempo em milissegundos; 0 espera indefinidamente. |
| Esc (27) | Código da tecla Escape retornado por `waitKey`. |
| Item A / Item B | Subdivisões dos Exercícios 2 e 3 propostos na aula. |
