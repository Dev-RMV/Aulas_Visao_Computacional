# Aula 16 – Calibração de câmera e realidade aumentada

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula ensina calibração de câmera com OpenCV e, a partir dela, realidade aumentada (AR) simples: estimar a pose de um tabuleiro de xadrez e desenhar eixos e um cubo 3D sobre ele. Calibrar uma câmera é descobrir os parâmetros que ligam o mundo 3D à imagem 2D: a matriz intrínseca K (distâncias focais `fx` e `fy` em pixels e ponto principal `cx`, `cy`) e os coeficientes de distorção da lente. Com esses números, o OpenCV consegue corrigir a distorção de uma imagem, calcular onde a câmera está em relação a um objeto conhecido (`solvePnP`) e projetar qualquer ponto 3D de volta na imagem (`projectPoints`).

A escolha didática do professor é não usar webcam. O módulo `synthetic_utils.py` constrói um laboratório virtual: uma câmera ideal, cuja matriz K verdadeira é conhecida (`TRUE_K`) e cuja distorção é zero, fotografa um tabuleiro de 8 x 7 quadrados a partir de poses escolhidas. Como a verdade é conhecida, cada etapa pode ser conferida contra o valor certo: a K estimada contra a K real, o `tvec` estimado contra o `tvec` usado para gerar a cena, os coeficientes de distorção contra zero. Isso é impossível com uma câmera física, em que só existe a estimativa.

O fio dos 30 scripts é incremental. Os scripts 01 a 03 criam a textura do tabuleiro, uma vista com pose conhecida e o conjunto de 18 imagens de calibração. Os scripts 04 a 07 detectam e refinam os cantos e montam as correspondências entre pontos 3D e 2D. Os scripts 08 a 12 calibram, comparam com a verdade, interpretam a distorção e medem o erro de reprojeção. Os scripts 13 a 16 salvam, carregam e aplicam a calibração para corrigir imagens. Os scripts 17 a 24 fazem a parte de realidade aumentada: pose com `solvePnP`, matriz de rotação, eixos, cubo de arame, faces coloridas, sequência de poses e vídeo com o cubo acompanhando o tabuleiro. Os scripts 25 a 28 são experimentos sobre qualidade da calibração: ruído, pouca variedade de pose, imagem com maior erro e recalibração. O `29_pipeline_completo_imagem_sintetica.py` junta tudo, e o `30_ponte_para_camera_real.py` mostra, em pseudocódigo, o que muda quando a fonte passa a ser `cv2.VideoCapture`.

Cada script termina com um bloco `# DESAFIO DO ALUNO`: a parte principal já funciona e o aluno deve modificar, medir, comparar ou estender. Esses desafios são os exercícios da aula e estão listados na seção 6.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros usadas nos scripts:

| Biblioteca | Pacote pip | Onde é usada |
|---|---|---|
| OpenCV | `opencv-contrib-python` (4.14) | `synthetic_utils.py` e todos os scripts, exceto 06, 09, 10, 11, 12, 13, 14, 25, 27, 28 e 30, que só usam o OpenCV por meio de `synthetic_utils.py` ou não o usam |
| NumPy | `numpy` (2.x) | `synthetic_utils.py`, 09, 12, 13, 14, 16, 19 a 21, 23 a 29 |

Só a biblioteca padrão além disso: `pathlib`. Não há pandas, matplotlib nem TensorFlow nesta aula.

O repositório tem um `requirements.txt` único na raiz, e o venv `.venv` com Python 3.13 já traz tudo o que esta aula precisa. Para criar ou recriar o ambiente, na raiz do repositório:

```
py preparar_ambiente.py
```

Depois, rode os scripts com `.\.venv\Scripts\python.exe` ou ative o venv com `.\.venv\Scripts\Activate.ps1`.

Pasta de execução: todos os scripts definem `ROOT = Path(__file__).resolve().parent`, ou seja, a raiz dos caminhos é a própria pasta `exemplos_em_python/`, independentemente do diretório de trabalho. Ainda assim, rode de dentro de `aula_16_Calibracao/exemplos_em_python/`, porque `synthetic_utils` é importado como módulo local e os comandos da seção 5 assumem isso. As pastas usadas são:

- `dados_sinteticos/`: veio no zip do professor com 20 PNG, `calib_01.png` a `calib_18.png`, `tabuleiro_base.png` e `vista_unica.png`. Todas têm 1280 x 720, exceto o tabuleiro, que tem 800 x 700. Essa pasta fica no git. Os scripts regravam esses arquivos e acrescentam `video_tabuleiro.avi` e `pouca_var_00.png` a `pouca_var_11.png`, que ficam fora do git.
- `saidas/`: o zip trouxe `saidas/04_cantos_detectados.png`, saída do próprio professor. No repositório a pasta existe por um `.gitkeep` e o conteúdo dela fica fora do git.
- `camera_sintetica.npz`: gravado pelo script 13 na pasta dos scripts, também fora do git.

Material faltante: nenhum. A aula é totalmente autossuficiente, porque as imagens são geradas pelo código. Nenhum script abre a webcam; o 30 apenas imprime um pseudocódigo.

Python: o `__pycache__` do zip era de Python 3.12, então o professor rodou a aula com essa versão.

## 3. Conceitos-chave

### 3.1 Laboratório virtual: uma câmera cuja verdade é conhecida

O ponto de partida de tudo está no topo de `synthetic_utils.py`. A câmera virtual tem tamanho de imagem 1280 x 720, matriz intrínseca `TRUE_K` e distorção `TRUE_DIST` igual a zero. O tabuleiro tem `PATTERN_SIZE = (7, 6)`, que conta os cantos internos (colunas, linhas), e cada quadrado mede `SQUARE_SIZE_M = 0.03`, ou seja, 30 mm. Como esses valores são os que geram as imagens, qualquer estimativa feita depois pode ser comparada com eles. O docstring do módulo lista as vantagens: imagens reprodutíveis, K conhecida, poses controláveis, ruído controlado e comparação entre valor verdadeiro e estimado.

```python
# synthetic_utils.py
PATTERN_SIZE = (7, 6)       # cantos internos: colunas, linhas
SQUARE_SIZE_M = 0.03        # 30 mm por quadrado
IMAGE_SIZE = (1280, 720)    # largura, altura

TRUE_K = np.array([
    [900.0,   0.0, 640.0],
    [  0.0, 910.0, 360.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)

TRUE_DIST = np.zeros((5, 1), dtype=np.float64)
```

Onde aparece: `synthetic_utils.py`; `TRUE_K` e `TRUE_DIST` são importados por 09, 17, 19, 20, 21, 22, 24 e 26; `PATTERN_SIZE` por 04 e 05; `SQUARE_SIZE_M` por 06, 19, 20, 21, 24 e 29.

### 3.2 Matriz intrínseca K e o modelo pinhole

No modelo de câmera pinhole, um ponto 3D `(X, Y, Z)` no referencial da câmera cai na imagem em `u = fx * X / Z + cx` e `v = fy * Y / Z + cy`. A matriz K guarda esses quatro números: `fx` e `fy` são as distâncias focais medidas em pixels (podem diferir se o pixel do sensor não for quadrado, e o professor escolheu 900 e 910 de propósito, para que a estimativa possa ser conferida por eixo), e `cx`, `cy` é o ponto principal, aqui exatamente o centro da imagem, 640 e 360. O comentário do script 08 resume o significado físico de cada termo.

```python
# 08_calibrar_camera.py
# Significado físico:
# fx, fy -> focais medidas em pixels
# cx, cy -> ponto principal da imagem
# k1, k2, k3 -> distorção radial
# p1, p2 -> distorção tangencial
```

Onde aparece: `synthetic_utils.py` (`TRUE_K`), `08_calibrar_camera.py`, `09_comparar_K_real_e_estimado.py`, `14_carregar_calibracao.py`.

### 3.3 Textura do tabuleiro: 8 x 7 quadrados dão 7 x 6 cantos internos

O OpenCV procura os cantos internos do tabuleiro, onde quatro quadrados se encontram. Um padrão de 8 colunas por 7 linhas de quadrados tem 7 x 6 desses cantos, e é isso que `PATTERN_SIZE` declara. `make_checkerboard` desenha a textura com `cv2.rectangle` preenchido em preto nos quadrados em que `(x + y) % 2 == 0`, sobre um fundo branco criado por `np.full`. O script 01 usa `square_px=100` e salva `tabuleiro_base.png` com 800 x 700 pixels; já `generate_view` chama `make_checkerboard(120)` internamente, gerando uma textura de 960 x 840 que só existe na memória.

```python
# synthetic_utils.py
def make_checkerboard(square_px=100):
    """
    Cria uma textura com 8 x 7 quadrados para produzir 7 x 6 cantos internos.
    """
    cols = PATTERN_SIZE[0] + 1
    rows = PATTERN_SIZE[1] + 1
    img = np.full((rows*square_px, cols*square_px), 255, np.uint8)
    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 0:
                cv2.rectangle(
                    img,
                    (x*square_px, y*square_px),
                    ((x+1)*square_px, (y+1)*square_px),
                    0, -1
                )
    return img
```

Onde aparece: `synthetic_utils.py`, `01_criar_tabuleiro.py`.

### 3.4 Rotação: ângulos de Euler, matriz R e vetor de Rodrigues

Uma rotação 3D pode ser escrita de vários jeitos. Para o ser humano, três ângulos em graus (`rx`, `ry`, `rz`) são o mais intuitivo. Para a matemática, a matriz de rotação R (3 x 3, ortonormal) é o mais direto. Para o OpenCV, a forma compacta é o vetor de Rodrigues `rvec`, com 3 números: a direção do vetor é o eixo de rotação e o comprimento é o ângulo em radianos. `euler_to_rvec` monta as três matrizes elementares, compõe `R = Rz @ Ry @ Rx` (primeiro gira em X, depois em Y, depois em Z) e converte para `rvec` com `cv2.Rodrigues`. A mesma função faz o caminho inverso, de `rvec` para R, como o script 18 mostra.

```python
# synthetic_utils.py
def euler_to_rvec(rx_deg, ry_deg, rz_deg):
    """
    Converte rotações em graus (Euler XYZ) para vetor de Rodrigues.
    """
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rx = np.array([[1,0,0],
                   [0,np.cos(rx),-np.sin(rx)],
                   [0,np.sin(rx), np.cos(rx)]], dtype=np.float64)
    Ry = np.array([[ np.cos(ry),0,np.sin(ry)],
                   [0,1,0],
                   [-np.sin(ry),0,np.cos(ry)]], dtype=np.float64)
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],
                   [np.sin(rz), np.cos(rz),0],
                   [0,0,1]], dtype=np.float64)

    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    return rvec
```

Onde aparece: `synthetic_utils.py` (usada dentro de `generate_view`), `18_converter_rvec_para_matriz_R.py`.

### 3.5 Gerar uma vista: projeção dos cantos e homografia da textura

`generate_view` é o coração do simulador. Ela recebe a pose do tabuleiro (rotação em graus e translação `tx`, `ty`, `tz` em metros) e devolve a imagem, o `rvec` e o `tvec` usados. O truque é que um plano visto por uma câmera pinhole se relaciona com a imagem por uma homografia, o mesmo objeto da aula 07. Então basta: projetar os quatro cantos externos da placa (`outer_3d`, com largura 8 x 0,03 m e altura 7 x 0,03 m) com `cv2.projectPoints`, calcular a homografia entre os quatro cantos da textura e os quatro pontos projetados com `cv2.getPerspectiveTransform`, e deformar a textura com `cv2.warpPerspective`. O resultado é exatamente o que a câmera de matriz K veria.

```python
# synthetic_utils.py
    outer_3d = np.array([
        [0.0,     0.0,     0.0],
        [board_w, 0.0,     0.0],
        [board_w, board_h, 0.0],
        [0.0,     board_h, 0.0]
    ], dtype=np.float32)

    rvec = euler_to_rvec(rx, ry, rz)
    tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

    dst, _ = cv2.projectPoints(outer_3d, rvec, tvec, K, TRUE_DIST)
    dst = dst.reshape(-1, 2).astype(np.float32)

    src = np.array([
        [0, 0],
        [texture.shape[1]-1, 0],
        [texture.shape[1]-1, texture.shape[0]-1],
        [0, texture.shape[0]-1]
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src, dst)
```

Onde aparece: `synthetic_utils.py`; chamada direta em 02, 17, 19, 20, 21, 22, 23, 24, 26 e 29, e indireta, via `generate_dataset`, em 03, 04, 05, 07 a 13, 15, 16, 25, 27, 28 e 29.

### 3.6 Fundo, máscara, desfoque e ruído controlados

Depois da homografia, `generate_view` compõe a placa sobre um fundo cinza claro (valor 220). Para que a borda da textura não deixe rastro, uma máscara branca do tamanho da textura é deformada pela mesma H, e só os pixels em que `mask > 0` recebem a placa. Dois parâmetros opcionais degradam a imagem de forma controlada: `blur` aplica `cv2.GaussianBlur` com kernel ímpar derivado do próprio valor, e `noise_std` soma ruído gaussiano com `np.random.normal` e recorta para 0 a 255. A imagem é devolvida em BGR (`cv2.COLOR_GRAY2BGR`) para que os scripts possam desenhar em cores por cima. Repare que o ruído não tem semente: cada chamada com `noise_std > 0` produz uma imagem diferente.

```python
# synthetic_utils.py
    bg = np.full((h, w), background, dtype=np.uint8)
    warped = cv2.warpPerspective(texture, H, (w, h), borderValue=background)

    # Máscara do quadrilátero projetado.
    mask_src = np.full(texture.shape, 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_src, H, (w, h), borderValue=0)
    bg[mask > 0] = warped[mask > 0]

    if blur > 0:
        k = int(max(3, round(blur)*2 + 1))
        if k % 2 == 0:
            k += 1
        bg = cv2.GaussianBlur(bg, (k, k), blur)

    if noise_std > 0:
        noise = np.random.normal(0, noise_std, bg.shape)
        bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR), rvec, tvec
```

Onde aparece: `synthetic_utils.py`; o desfoque e o ruído só são ativados por `generate_dataset(noisy=True)` nos scripts 25, 27 e 28.

### 3.7 Conjunto de 18 poses para calibrar

Calibrar exige várias vistas do mesmo padrão em poses diferentes. `generate_dataset` tem uma lista fixa de 18 poses, com inclinações entre -22 e 25 graus, deslocamentos laterais de alguns centímetros e distâncias `tz` de 0,65 a 0,92 m. Cada pose vira `calib_NN.png` em `dados_sinteticos/`. Sem ruído a geração é determinista, então os arquivos regravados são iguais aos que vieram no zip. Com `noisy=True`, o desfoque é 0,8 e o ruído 2,0. A função sempre regrava as 18 imagens, mesmo que já existam.

```python
# synthetic_utils.py
    poses = [
        (  0,   0,   0, -0.12, -0.09, 0.70),
        ( -8,  10,   4, -0.10, -0.07, 0.65),
        ( 10, -12,  -5, -0.08, -0.06, 0.72),
        ( 15,   8,   8, -0.14, -0.08, 0.78),
        ...
    ][:n]

    paths = []
    for i, p in enumerate(poses, start=1):
        image, _, _ = generate_view(
            *p,
            blur=0.8 if noisy else 0.0,
            noise_std=2.0 if noisy else 0.0
        )
        path = out / f"calib_{i:02d}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths
```

Onde aparece: `synthetic_utils.py`, `03_gerar_18_imagens_calibracao.py`, e todos os scripts que calibram: 04, 05, 07 a 13, 15, 16, 25, 27, 28 e 29.

### 3.8 Detecção de cantos com `cv2.findChessboardCorners`

A calibração começa achando, em cada imagem, as posições 2D dos 42 cantos internos. `cv2.findChessboardCorners(gray, PATTERN_SIZE)` recebe a imagem em cinza e o número de cantos esperado, e devolve um booleano `ok` e um array `corners` de forma `(42, 1, 2)`, ordenado linha por linha do padrão. Se o tabuleiro não estiver inteiro na imagem, ou o padrão pedido não bater com o real, `ok` é falso. `cv2.drawChessboardCorners` desenha os cantos coloridos ligados em sequência, o que serve para conferir a ordem. O script 04 usa `paths[4]`, ou seja, `calib_05.png`.

```python
# 04_detectar_cantos.py
img = cv2.imread(str(paths[4]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ok, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)

print("Tabuleiro encontrado?", ok)
if ok:
    vis = img.copy()
    cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, ok)
    cv2.imwrite(str(ROOT / "saidas" / "04_cantos_detectados.png"), vis)
    print("Quantidade de cantos:", len(corners))
```

Onde aparece: `04_detectar_cantos.py`, `05_refinar_cantos_subpixel.py`, e `find_refined_corners` em `synthetic_utils.py`.

### 3.9 Refinamento subpixel com `cv2.cornerSubPix`

`findChessboardCorners` localiza os cantos com precisão de pixel inteiro, o que é pouco para calibração. `cv2.cornerSubPix` refina cada canto ajustando-o ao ponto de sela do gradiente dentro de uma janela de busca. Os argumentos são a imagem em cinza, os cantos iniciais, a meia janela `(11, 11)` (uma região de 23 x 23 pixels), a zona morta `(-1, -1)` (nenhuma) e o critério de parada: 40 iterações ou deslocamento menor que 0,001 pixel. O script 05 imprime o primeiro canto antes e depois; o deslocamento é uma fração de pixel. `find_refined_corners` empacota detecção e refinamento em uma função só, devolvendo `(False, None)` quando o tabuleiro não é achado.

```python
# 05_refinar_cantos_subpixel.py
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    40,
    0.001
)

refined = cv2.cornerSubPix(
    gray, corners, (11, 11), (-1, -1), criteria
)

print("Primeiro canto antes do refinamento:", corners[0,0])
print("Primeiro canto após refinamento:", refined[0,0])
```

```python
# synthetic_utils.py
def find_refined_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)
    if not ok:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001
    )
    refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    return True, refined
```

Onde aparece: `05_refinar_cantos_subpixel.py`, `synthetic_utils.py`; `find_refined_corners` é usada por 07, 17, 19, 20, 21, 22, 24, 29 e por `calibrate_from_paths`.

### 3.10 Pontos 3D do tabuleiro em metros

Para calibrar, cada canto 2D precisa de um par 3D: onde ele está no mundo. O professor adota o referencial do próprio tabuleiro: origem no primeiro canto interno, eixo X ao longo das colunas, eixo Y ao longo das linhas, e Z perpendicular à placa. Assim todos os cantos têm Z = 0, e as coordenadas X e Y são múltiplos do lado do quadrado. `object_points` gera isso com `np.mgrid[0:7, 0:6].T.reshape(-1, 2)`, que percorre as 7 colunas de uma linha e depois passa para a linha seguinte, exatamente a ordem em que `findChessboardCorners` devolve os cantos, e multiplica por `SQUARE_SIZE_M`. É por causa desse produto que `tvec` sai em metros: mude o tamanho do quadrado e a distância estimada muda na mesma proporção.

```python
# synthetic_utils.py
def object_points():
    """
    Coordenadas 3D dos 7 x 6 cantos internos, no plano Z=0.
    """
    obj = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    obj[:, :2] = np.mgrid[
        0:PATTERN_SIZE[0],
        0:PATTERN_SIZE[1]
    ].T.reshape(-1, 2)
    obj *= SQUARE_SIZE_M
    return obj
```

Onde aparece: `synthetic_utils.py`, `06_criar_pontos_3d_tabuleiro.py`, `07_coletar_pontos_2d_3d.py`, e todo script que chama `solvePnP` (17, 19, 20, 21, 22, 24, 29).

### 3.11 Correspondências 2D e 3D: `objpoints` e `imgpoints`

A entrada de `calibrateCamera` são duas listas paralelas: `objpoints`, com uma cópia dos 42 pontos 3D para cada imagem (sempre a mesma, porque o tabuleiro não muda), e `imgpoints`, com os 42 cantos 2D refinados daquela imagem. O script 07 monta as listas na mão, imagem por imagem, e só inclui as imagens em que a detecção deu certo. `calibrate_from_paths`, em `synthetic_utils.py`, faz o mesmo e ainda registra em `used` os caminhos que entraram e o tamanho da imagem.

```python
# 07_coletar_pontos_2d_3d.py
objpoints = []
imgpoints = []
obj = object_points()

for p in paths:
    img = cv2.imread(str(p))
    ok, corners = find_refined_corners(img)

    if ok:
        objpoints.append(obj.copy())
        imgpoints.append(corners)

print("Imagens válidas:", len(imgpoints))
print("Pontos 3D por imagem:", objpoints[0].shape)
print("Pontos 2D por imagem:", imgpoints[0].shape)
```

Onde aparece: `07_coletar_pontos_2d_3d.py`, `calibrate_from_paths` em `synthetic_utils.py`.

### 3.12 Calibração com `cv2.calibrateCamera`

`cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)` resolve, por otimização não linear, a K, os cinco coeficientes de distorção e a pose (`rvec`, `tvec`) de cada imagem que melhor explicam as correspondências. Os dois `None` dizem que não há chute inicial para K nem para a distorção. Ela devolve cinco coisas: o RMS do erro de reprojeção sobre todos os pontos de todas as imagens, `K` (3 x 3), `dist` (1 x 5, na ordem `k1, k2, p1, p2, k3`), e as listas `rvecs` e `tvecs`, uma pose por imagem. A função do professor exige ao menos 3 imagens válidas e devolve uma tupla de 9 elementos, com as listas de pontos, os caminhos usados e o tamanho da imagem.

```python
# synthetic_utils.py
    if len(objpoints) < 3:
        raise RuntimeError("Poucas imagens válidas para calibração.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    return rms, K, dist, rvecs, tvecs, objpoints, imgpoints, used, image_size
```

```python
# 08_calibrar_camera.py
rms, K, dist, rvecs, tvecs, objpoints, imgpoints, used, size = \
    calibrate_from_paths(paths)

print("RMS retornado pelo calibrateCamera:", rms)
print("\nMatriz intrínseca K:")
print(K)
print("\nCoeficientes de distorção:")
print(dist.ravel()[:5])
```

Onde aparece: `synthetic_utils.py`, `08_calibrar_camera.py`, e via `calibrate_from_paths` em 09 a 13, 15, 16, 25 a 29.

### 3.13 Comparar K estimada com K verdadeira

Só o laboratório virtual permite este script. O 09 imprime `TRUE_K`, a K estimada e a diferença elemento a elemento, e calcula o erro relativo de `fx` em porcentagem. Como a câmera é ideal e as imagens não têm ruído, o erro deve ser pequeno, de fração de por cento. A sintaxe `_, K, _, *_ = calibrate_from_paths(paths)` descarta o RMS, a distorção e os outros seis elementos da tupla, ficando só com K.

```python
# 09_comparar_K_real_e_estimado.py
_, K, _, *_ = calibrate_from_paths(paths)

print("K verdadeira usada no simulador:")
print(TRUE_K)

print("\nK estimada pelo OpenCV:")
print(K)

print("\nDiferença K_estimada - K_real:")
print(K - TRUE_K)

erro_rel_fx = abs(K[0,0] - TRUE_K[0,0]) / TRUE_K[0,0] * 100
print(f"\nErro relativo em fx: {erro_rel_fx:.3f}%")
```

Onde aparece: `09_comparar_K_real_e_estimado.py`, `26_pouca_variedade_de_pose.py` (mesma comparação com um conjunto ruim).

### 3.14 Coeficientes de distorção radial e tangencial

Lentes reais deformam a imagem. A distorção radial (`k1`, `k2`, `k3`) depende da distância ao centro óptico e produz o efeito barril ou almofada; a tangencial (`p1`, `p2`) vem do desalinhamento entre lente e sensor. O OpenCV guarda os cinco no vetor `dist` na ordem `k1, k2, p1, p2, k3`. O script 10 desempacota e imprime cada um com oito casas. Como `TRUE_DIST` é zero, o esperado são valores muito próximos de zero; qualquer coisa longe disso indicaria que a otimização compensou outro erro com distorção fictícia.

```python
# 10_interpretar_coeficientes_distorcao.py
_, _, dist, *_ = calibrate_from_paths(paths)

d = dist.ravel()
k1, k2, p1, p2, k3 = d[:5]

print(f"k1 = {k1:.8f}")
print(f"k2 = {k2:.8f}")
print(f"p1 = {p1:.8f}")
print(f"p2 = {p2:.8f}")
print(f"k3 = {k3:.8f}")

print("\nComo a câmera sintética ideal foi criada sem distorção,")
print("esperamos coeficientes próximos de zero.")
```

Onde aparece: `10_interpretar_coeficientes_distorcao.py`, `08_calibrar_camera.py`, `13_salvar_calibracao_real.py`, `14_carregar_calibracao.py`.

### 3.15 Erro de reprojeção por imagem e médio

A medida padrão de qualidade da calibração é o erro de reprojeção: com K, `dist` e a pose de cada imagem, projeta-se os pontos 3D de volta com `cv2.projectPoints` e mede-se a distância até os cantos detectados. `reprojection_errors` calcula, por imagem, `cv2.norm(img, projected, cv2.NORM_L2) / len(projected)`: a norma L2 da diferença (raiz da soma dos quadrados dos 42 pontos) dividida pelo número de pontos. Essa é a fórmula do tutorial oficial do OpenCV, mas note que ela não é o RMS por ponto: para 42 pontos, o valor sai cerca de 6,5 vezes menor que o RMS devolvido por `calibrateCamera`. O script 11 imprime o erro de cada imagem; o 12 tira a média e dá a referência prática do professor.

```python
# synthetic_utils.py
def reprojection_errors(K, dist, rvecs, tvecs, objpoints, imgpoints):
    errors = []
    for obj, img, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = cv2.norm(img, projected, cv2.NORM_L2) / len(projected)
        errors.append(float(err))
    return errors
```

```python
# 12_erro_medio_reprojecao.py
mean_error = np.mean(errors)
print(f"Erro médio de reprojeção: {mean_error:.4f} px")

print("\nReferência prática:")
print("0 a 0,5 px  -> geralmente muito bom")
print("0,5 a 1 px  -> frequentemente utilizável")
print("> 1 px       -> convém investigar")
print("A tolerância real depende da aplicação robótica.")
```

Onde aparece: `synthetic_utils.py`, `11_erro_reprojecao_por_imagem.py`, `12_erro_medio_reprojecao.py`, `25_efeito_do_ruido.py`, `27_identificar_imagem_com_maior_erro.py`, `28_recalibrar_sem_pior_imagem.py`, `29_pipeline_completo_imagem_sintetica.py`.

### 3.16 Salvar e carregar a calibração com `np.savez` e `np.load`

Calibrar é caro e se faz uma vez; usar a calibração acontece em todo frame. Por isso o resultado vai para disco. `np.savez(arquivo, K=K, dist=dist)` grava os dois arrays com nome em um único `.npz`, e `np.load` devolve um objeto indexável por esses nomes. O script 13 grava `camera_sintetica.npz` na pasta dos scripts; o 14 carrega e, se o arquivo não existir, avisa e encerra com `raise SystemExit`. O nome "calibração real" no script 13 significa "estimada pelo OpenCV", em oposição à `TRUE_K` do simulador.

```python
# 13_salvar_calibracao_real.py
arquivo = ROOT / "camera_sintetica.npz"
np.savez(arquivo, K=K, dist=dist)
```

```python
# 14_carregar_calibracao.py
if not arquivo.exists():
    print("Execute primeiro o exemplo 13.")
    raise SystemExit

data = np.load(arquivo)
K = data["K"]
dist = data["dist"]
```

Onde aparece: `13_salvar_calibracao_real.py`, `14_carregar_calibracao.py`; o `30_ponte_para_camera_real.py` cita o arquivo como fonte de K e `dist` para testes sintéticos.

### 3.17 Correção de distorção com `cv2.undistort`

Com K e `dist`, `cv2.undistort(img, K, dist)` remapeia a imagem para o que uma câmera sem distorção veria. Linhas retas do mundo que apareciam curvas voltam a ser retas. O script 15 corrige `paths[5]` e salva original e corrigida em `saidas/`; o 16 monta um painel com `np.hstack` e rotula com `cv2.putText`. Como a câmera virtual não tem distorção e os coeficientes estimados são quase zero, as duas imagens ficam praticamente idênticas, o que é exatamente a lição: a correção só é visível quando há distorção para corrigir. Numa webcam real a diferença aparece nas bordas.

```python
# 16_painel_original_corrigida.py
img = cv2.imread(str(paths[9]))
corrigida = cv2.undistort(img, K, dist)

painel = np.hstack([img, corrigida])

cv2.putText(painel, "ORIGINAL", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
cv2.putText(painel, "CORRIGIDA", (img.shape[1]+30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,120,0), 3)
```

Onde aparece: `15_undistort_em_imagem_sintetica.py`, `16_painel_original_corrigida.py`.

### 3.18 Estimativa de pose com `cv2.solvePnP`

PnP (Perspective-n-Point) é o problema inverso da calibração: K e `dist` já são conhecidos, e o que se quer é a pose de um objeto conhecido a partir de uma única imagem. `cv2.solvePnP(objectPoints, imagePoints, K, dist)` recebe os 42 pontos 3D do tabuleiro, os 42 cantos detectados e a calibração, e devolve `sucesso`, `rvec` e `tvec`: a rotação e a translação que levam o referencial do tabuleiro para o referencial da câmera. `tvec` sai em metros, porque `object_points` está em metros, e `tz` é a distância da câmera ao canto de origem. O script 17 gera a vista com `tz=0.78`, estima com `TRUE_K` e imprime o `tvec` real ao lado para comparação.

```python
# 17_estimar_pose_solvepnp.py
img, rvec_real, tvec_real = generate_view(
    rx=15, ry=-12, rz=7,
    tx=-0.10, ty=-0.06, tz=0.78
)

ok, corners = find_refined_corners(img)
if not ok:
    raise RuntimeError("Tabuleiro não detectado.")

sucesso, rvec_est, tvec_est = cv2.solvePnP(
    object_points(), corners, TRUE_K, TRUE_DIST
)

print("solvePnP funcionou?", sucesso)
print("\nrvec estimado:\n", rvec_est)
print("\ntvec estimado [m]:\n", tvec_est)
```

Onde aparece: `17_estimar_pose_solvepnp.py`, `19_projectpoints_eixos_3d.py`, `20_projetar_cubo_3d.py`, `21_faces_coloridas_do_cubo.py`, `22_imprimir_pose_em_sequencia.py`, `24_ar_em_video_sintetico.py`, `29_pipeline_completo_imagem_sintetica.py`.

### 3.19 De `rvec` para a matriz R e o teste de ortogonalidade

`cv2.Rodrigues(rvec)` devolve a matriz de rotação R (e a jacobiana, descartada). Uma matriz de rotação é ortonormal: suas colunas são vetores unitários perpendiculares entre si, então `R.T @ R` deve dar a identidade, a menos de erro de ponto flutuante. Esse é o teste do script 18, que parte de `euler_to_rvec(15, -10, 5)`. Saber converter nos dois sentidos é útil porque `solvePnP` devolve `rvec`, mas para transformar pontos na mão é R que se usa: `P_camera = R @ P_objeto + tvec`.

```python
# 18_converter_rvec_para_matriz_R.py
rvec = euler_to_rvec(15, -10, 5)
R, _ = cv2.Rodrigues(rvec)

print("Vetor de rotação rvec:")
print(rvec)

print("\nMatriz de rotação R:")
print(R)

print("\nTeste aproximado de ortogonalidade R^T R:")
print(R.T @ R)
```

Onde aparece: `18_converter_rvec_para_matriz_R.py`, `euler_to_rvec` em `synthetic_utils.py`.

### 3.20 Projeção de eixos 3D com `cv2.projectPoints`

Realidade aumentada, no fundo, é projetar pontos 3D definidos no referencial do objeto e desenhá-los na imagem. `cv2.projectPoints(pontos3d, rvec, tvec, K, dist)` faz isso para um array de pontos e devolve as coordenadas 2D (mais uma jacobiana ignorada). O script 19 define quatro pontos: a origem e a ponta de cada eixo a um quadrado de distância. O eixo Z é `-s`, negativo, porque no referencial do tabuleiro (X para a direita, Y para baixo, como na imagem) o Z positivo aponta para dentro da placa; o sinal negativo faz a seta "sair" da mesa em direção à câmera. As cores seguem a convenção BGR do OpenCV: X vermelho `(0,0,255)`, Y verde `(0,255,0)`, Z azul `(255,0,0)`.

```python
# 19_projectpoints_eixos_3d.py
s = SQUARE_SIZE_M
axis3d = np.float32([
    [0,0,0],
    [s,0,0],
    [0,s,0],
    [0,0,-s]
])

pts2d, _ = cv2.projectPoints(axis3d, rvec, tvec, TRUE_K, TRUE_DIST)
p = pts2d.reshape(-1,2).astype(int)

origem = tuple(p[0])
cv2.line(img, origem, tuple(p[1]), (0,0,255), 4)   # X
cv2.line(img, origem, tuple(p[2]), (0,255,0), 4)   # Y
cv2.line(img, origem, tuple(p[3]), (255,0,0), 4)   # Z
```

Onde aparece: `19_projectpoints_eixos_3d.py`; `projectPoints` também em `generate_view` e `reprojection_errors` (`synthetic_utils.py`), 20, 21, 24 e 29.

### 3.21 Cubo de arame sobre o tabuleiro

O cubo é o exemplo clássico de AR. Os oito vértices ficam no referencial do tabuleiro: quatro na base, em Z = 0, e quatro no topo, em Z = -s, com aresta igual a um quadrado. Depois de `projectPoints`, as doze arestas são desenhadas com `cv2.line` em três grupos: base em verde, topo em azul, verticais em vermelho (tudo em BGR). Como a pose vem de `solvePnP` na mesma imagem, o cubo "gruda" no tabuleiro: mude a pose da vista e o cubo acompanha. O `reshape(-1,2).astype(int)` converte a saída de forma `(8, 1, 2)` em inteiros de pixel.

```python
# 20_projetar_cubo_3d.py
s = SQUARE_SIZE_M
cube = np.float32([
    [0,0,0], [s,0,0], [s,s,0], [0,s,0],
    [0,0,-s], [s,0,-s], [s,s,-s], [0,s,-s]
])

pts, _ = cv2.projectPoints(cube, rvec, tvec, TRUE_K, TRUE_DIST)
p = pts.reshape(-1,2).astype(int)

for a,b in [(0,1),(1,2),(2,3),(3,0)]:
    cv2.line(img, tuple(p[a]), tuple(p[b]), (0,255,0), 3)
for a,b in [(4,5),(5,6),(6,7),(7,4)]:
    cv2.line(img, tuple(p[a]), tuple(p[b]), (255,0,0), 3)
for a,b in [(0,4),(1,5),(2,6),(3,7)]:
    cv2.line(img, tuple(p[a]), tuple(p[b]), (0,0,255), 3)
```

Onde aparece: `20_projetar_cubo_3d.py`, `24_ar_em_video_sintetico.py` e `29_pipeline_completo_imagem_sintetica.py` (mesmo cubo, arestas em laranja).

### 3.22 Faces semitransparentes com `fillConvexPoly` e `addWeighted`

O script 21 troca o arame por faces. Cada face é uma lista de quatro índices de vértices e uma cor; `cv2.fillConvexPoly` pinta o polígono em uma cópia da imagem (`overlay`), e `cv2.addWeighted(overlay, 0.30, img, 0.70, 0)` mistura 30% da cópia pintada com 70% da original, dando a transparência. Por fim `cv2.polylines` traça o contorno escuro de cada face. Não há teste de visibilidade: as seis faces são pintadas sempre, inclusive as que estariam atrás; a transparência disfarça isso. Este script é o único que chama `solvePnP` sem verificar o `ok` de `find_refined_corners`, o que quebraria se a detecção falhasse.

```python
# 21_faces_coloridas_do_cubo.py
faces = [
    ([0,1,2,3], (0,255,0)),
    ([4,5,6,7], (255,0,0)),
    ([0,1,5,4], (0,255,255)),
    ([1,2,6,5], (0,0,255)),
    ([2,3,7,6], (255,255,0)),
    ([3,0,4,7], (255,0,255)),
]

for ids, color in faces:
    poly = np.array([p[i] for i in ids], np.int32)
    cv2.fillConvexPoly(overlay, poly, color)

img = cv2.addWeighted(overlay, 0.30, img, 0.70, 0)

for ids, _ in faces:
    poly = np.array([p[i] for i in ids], np.int32)
    cv2.polylines(img, [poly], True, (30,30,30), 2)
```

Onde aparece: `21_faces_coloridas_do_cubo.py`; o desafio do 29 pede a mesma técnica.

### 3.23 Pose em sequência e vídeo sintético com `cv2.VideoWriter`

Antes de chegar ao vídeo, o script 22 mostra o que muda entre frames: quatro poses com inclinação e distância crescentes, e para cada uma `solvePnP` imprime `rvec` e `tvec`. Ler esses números em sequência é o primeiro contato com rastreamento de pose. O script 23 transforma isso em um vídeo de 90 frames a 15 fps, `dados_sinteticos/video_tabuleiro.avi`, com codec MJPG, em que os ângulos e a distância oscilam por senos de `a = i / 89`. É a mesma API `cv2.VideoWriter` das aulas 01 e 10: caminho, fourcc, fps e tamanho `(largura, altura)`.

```python
# 23_simular_video_sem_webcam.py
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(str(saida), fourcc, 15.0, (1280,720))

for i in range(90):
    a = i / 89.0
    rx = 8 + 8*np.sin(a*np.pi*2)
    ry = -12 + 12*np.sin(a*np.pi)
    rz = 6*np.sin(a*np.pi*2)
    tz = 0.75 + 0.05*np.sin(a*np.pi*2)
    tx = -0.10 + 0.02*np.sin(a*np.pi*2)

    frame, _, _ = generate_view(rx, ry, rz, tx, -0.07, tz)
    writer.write(frame)

writer.release()
```

Onde aparece: `22_imprimir_pose_em_sequencia.py`, `23_simular_video_sem_webcam.py`, `24_ar_em_video_sintetico.py`.

### 3.24 Realidade aumentada frame a frame

O script 24 é o laço de AR completo, sem câmera: para cada um dos 90 frames, gera a vista com as mesmas fórmulas do 23, detecta e refina os cantos, estima a pose com `solvePnP`, projeta o cubo e desenha as doze arestas em laranja, gravando `saidas/24_ar_video_sintetico.avi`. Se a detecção falhar em um frame, o frame é gravado sem cubo. Repare que o 24 não lê o vídeo do 23: ele regenera cada frame. Na câmera real, o `30_ponte_para_camera_real.py` mostra que só a primeira linha muda: `frame = generate_view(...)` vira `ret, frame = cap.read()`.

```python
# 24_ar_em_video_sintetico.py
    ok, corners = find_refined_corners(frame)

    if ok:
        _, rvec, tvec = cv2.solvePnP(
            object_points(), corners, TRUE_K, TRUE_DIST
        )
        pts, _ = cv2.projectPoints(
            cube, rvec, tvec, TRUE_K, TRUE_DIST
        )
        p = pts.reshape(-1,2).astype(int)

        for a1,b1 in edges:
            cv2.line(frame, tuple(p[a1]), tuple(p[b1]), (0,120,255), 3)

    writer.write(frame)
```

Onde aparece: `24_ar_em_video_sintetico.py`, `30_ponte_para_camera_real.py` (pseudocódigo do mesmo laço).

### 3.25 Efeito do ruído na calibração

O script 25 gera o conjunto duas vezes, primeiro limpo e depois com `noisy=True` (desfoque 0,8 e ruído gaussiano de desvio 2), calibra cada um e compara o erro médio de reprojeção. O ruído desloca os cantos detectados em frações de pixel e o erro sobe. Como o ruído não tem semente, os números do conjunto ruidoso mudam a cada execução. Efeito colateral importante: a segunda chamada deixa `calib_01.png` a `calib_18.png` sobrescritos com as versões ruidosas, e assim ficam até algum script regerar o conjunto limpo.

```python
# 25_efeito_do_ruido.py
paths_limpos = generate_dataset(ROOT, n=18, noisy=False)
r1, K1, d1, rv1, tv1, op1, ip1, *_ = calibrate_from_paths(paths_limpos)
e1 = np.mean(reprojection_errors(K1,d1,rv1,tv1,op1,ip1))

paths_ruidosos = generate_dataset(ROOT, n=18, noisy=True)
r2, K2, d2, rv2, tv2, op2, ip2, *_ = calibrate_from_paths(paths_ruidosos)
e2 = np.mean(reprojection_errors(K2,d2,rv2,tv2,op2,ip2))

print(f"Erro médio - imagens limpas:   {e1:.4f} px")
print(f"Erro médio - imagens ruidosas: {e2:.4f} px")
```

Onde aparece: `25_efeito_do_ruido.py`; `27_identificar_imagem_com_maior_erro.py` e `28_recalibrar_sem_pior_imagem.py` também usam `noisy=True`.

### 3.26 Pouca variedade de poses degrada K

Este é o experimento mais importante para quem vai calibrar uma câmera de verdade. O script 26 gera 12 vistas quase iguais: rotações menores que 1 grau, `tz` variando 1 mm entre elas. A calibração roda sem erro, mas a K estimada fica longe da `TRUE_K`, sobretudo em `fx` e `fy`. O motivo é geométrico: com um plano visto sempre de frente e à mesma distância, aumentar a distância focal e afastar o tabuleiro produzem quase a mesma imagem, então a otimização não consegue separar `fx` de `tz`, e o ponto principal fica mal determinado. Inclinações variadas quebram essa ambiguidade, e é por isso que os tutoriais mandam fotografar o tabuleiro em muitos ângulos e posições, inclusive nos cantos da imagem.

```python
# 26_pouca_variedade_de_pose.py
for i in range(12):
    # Quase todas as imagens são frontais e muito semelhantes.
    img, _, _ = generate_view(
        rx=1.0*i/12,
        ry=-1.0*i/12,
        rz=0,
        tx=-0.10,
        ty=-0.07,
        tz=0.75 + i*0.001
    )
    p = folder / f"pouca_var_{i:02d}.png"
    cv2.imwrite(str(p), img)
    paths.append(p)

_, K, *_ = calibrate_from_paths(paths)
```

Onde aparece: `26_pouca_variedade_de_pose.py`.

### 3.27 Diagnóstico: pior imagem e recalibração sem ela

Na prática, algumas fotos de calibração saem tremidas ou com cantos mal detectados, e o erro por imagem serve para achá-las. O script 27, sobre o conjunto ruidoso, usa `np.argmax` na lista de erros e imprime o nome da imagem (guardado em `used`) e o valor. O 28 vai além: remove essa imagem da lista e calibra de novo, comparando o erro médio antes e depois. O erro quase sempre cai, mas o desafio do professor alerta que "menor erro" não é o único critério: tirar imagens reduz a variedade de poses, e a seção anterior mostra o preço disso.

```python
# 28_recalibrar_sem_pior_imagem.py
_, K1, d1, rv1, tv1, op1, ip1, used, _ = calibrate_from_paths(paths)
e1 = reprojection_errors(K1,d1,rv1,tv1,op1,ip1)

pior = int(np.argmax(e1))
paths_filtrados = [p for i,p in enumerate(used) if i != pior]

_, K2, d2, rv2, tv2, op2, ip2, _, _ = calibrate_from_paths(paths_filtrados)
e2 = reprojection_errors(K2,d2,rv2,tv2,op2,ip2)
```

Onde aparece: `27_identificar_imagem_com_maior_erro.py`, `28_recalibrar_sem_pior_imagem.py`.

### 3.28 Pipeline completo e a ponte para a câmera real

O script 29 encadeia as cinco etapas em um arquivo só, com comentários `ETAPA 1` a `ETAPA 5`: gerar 18 imagens, calibrar, medir o erro, estimar a pose de uma vista nova e projetar o cubo. A diferença em relação aos scripts 17 a 24 é que aqui `solvePnP` e `projectPoints` usam a K e a `dist` estimadas, não a `TRUE_K`, que é o fluxo de uma aplicação real, em que a verdade não existe. O script 30 não executa nada além de imprimir um pseudocódigo: carregar K e `dist`, abrir a câmera ou um vídeo, e, para cada frame, detectar, refinar, `solvePnP`, `projectPoints`, desenhar e mostrar. O docstring explica que só a origem do frame muda.

```python
# 29_pipeline_completo_imagem_sintetica.py
# ETAPA 1 - gerar pelo menos 15 imagens.
paths = generate_dataset(ROOT, n=18)

# ETAPA 2 - calibrar.
_, K, dist, rvecs, tvecs, objpoints, imgpoints, _, _ = \
    calibrate_from_paths(paths)

# ETAPA 3 - erro de reprojeção.
errors = reprojection_errors(
    K, dist, rvecs, tvecs, objpoints, imgpoints
)
```

```python
# 30_ponte_para_camera_real.py
PSEUDOCODIGO = r"""
carregar K e dist

abrir câmera ou vídeo

enquanto houver frame:
    detectar tabuleiro
    se encontrado:
        refinar cantos
        estimar pose com solvePnP
        projetar cubo com projectPoints
        desenhar cubo
        imprimir rvec e tvec

    mostrar frame

fechar câmera
"""
```

Onde aparece: `29_pipeline_completo_imagem_sintetica.py`, `30_ponte_para_camera_real.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `synthetic_utils.py` | Câmera virtual: constantes, tabuleiro, Euler para Rodrigues, geração de vista e de dataset, pontos 3D, detecção refinada, calibração e erro de reprojeção | importado | `dados_sinteticos/` e `saidas/` via `ensure_dirs` | cv2, numpy |
| `dados_sinteticos/` | Pasta do professor com 20 PNG: `calib_01` a `calib_18`, `tabuleiro_base`, `vista_unica` | veio no zip | regravada pelos scripts 01, 02, 03 e todos que chamam `generate_dataset` | nada |
| `01_criar_tabuleiro.py` | Desenha a textura 8 x 7 com quadrados de 100 px | nada | `dados_sinteticos/tabuleiro_base.png` (800 x 700) | `synthetic_utils.py` |
| `02_gerar_uma_vista_sintetica.py` | Gera uma vista com pose conhecida e imprime `rvec` e `tvec` | pose fixa `rx=12, ry=-15, rz=6, tz=0.75` | `dados_sinteticos/vista_unica.png` | `synthetic_utils.py`, pasta `dados_sinteticos/` |
| `03_gerar_18_imagens_calibracao.py` | Gera as 18 vistas de calibração | nada | `dados_sinteticos/calib_01.png` a `calib_18.png` | `synthetic_utils.py` |
| `04_detectar_cantos.py` | `findChessboardCorners` em `calib_05.png` e desenha os cantos | 18 imagens (regeradas) | `saidas/04_cantos_detectados.png` e texto | `synthetic_utils.py` |
| `05_refinar_cantos_subpixel.py` | `cornerSubPix` em `calib_08.png`, imprime o primeiro canto antes e depois | 18 imagens (regeradas) | texto | `synthetic_utils.py` |
| `06_criar_pontos_3d_tabuleiro.py` | Mostra os 42 pontos 3D em metros | nada | texto | `synthetic_utils.py` |
| `07_coletar_pontos_2d_3d.py` | Monta `objpoints` e `imgpoints` para as 18 imagens | 18 imagens (regeradas) | texto | `synthetic_utils.py` |
| `08_calibrar_camera.py` | `calibrateCamera`: RMS, K e distorção | 18 imagens (regeradas) | texto | `synthetic_utils.py` |
| `09_comparar_K_real_e_estimado.py` | Diferença entre K estimada e `TRUE_K`, erro relativo de `fx` | 18 imagens (regeradas) | texto | `synthetic_utils.py` |
| `10_interpretar_coeficientes_distorcao.py` | Imprime `k1, k2, p1, p2, k3` | 18 imagens (regeradas) | texto | `synthetic_utils.py` |
| `11_erro_reprojecao_por_imagem.py` | Erro de reprojeção de cada imagem | 18 imagens (regeradas) | texto | `synthetic_utils.py` |
| `12_erro_medio_reprojecao.py` | Erro médio e referência prática | 18 imagens (regeradas) | texto | `synthetic_utils.py` |
| `13_salvar_calibracao_real.py` | Calibra e grava K e `dist` | 18 imagens (regeradas) | `camera_sintetica.npz` na pasta dos scripts | `synthetic_utils.py` |
| `14_carregar_calibracao.py` | Lê o `.npz` e imprime K e `dist` | `camera_sintetica.npz` | texto | script 13 |
| `15_undistort_em_imagem_sintetica.py` | `cv2.undistort` em `calib_06.png` | 18 imagens (regeradas) | `saidas/15_original.png`, `saidas/15_corrigida.png` | `synthetic_utils.py` |
| `16_painel_original_corrigida.py` | Painel lado a lado de `calib_10.png` original e corrigida | 18 imagens (regeradas) | `saidas/16_painel_original_corrigida.png` | `synthetic_utils.py` |
| `17_estimar_pose_solvepnp.py` | `solvePnP` numa vista nova, compara `tvec` com o real | vista gerada | texto | `synthetic_utils.py` |
| `18_converter_rvec_para_matriz_R.py` | `Rodrigues` e teste `R.T @ R` | ângulos fixos 15, -10, 5 | texto | `synthetic_utils.py` |
| `19_projectpoints_eixos_3d.py` | Projeta e desenha os eixos X, Y, Z | vista gerada | `saidas/19_eixos_3d.png` | `synthetic_utils.py`, pasta `saidas/` |
| `20_projetar_cubo_3d.py` | Cubo de arame com aresta de 1 quadrado | vista gerada | `saidas/20_cubo_3d.png` | `synthetic_utils.py`, pasta `saidas/` |
| `21_faces_coloridas_do_cubo.py` | Cubo com faces semitransparentes | vista gerada | `saidas/21_faces_coloridas.png` | `synthetic_utils.py`, pasta `saidas/` |
| `22_imprimir_pose_em_sequencia.py` | `rvec` e `tvec` de 4 poses sucessivas | 4 poses fixas | texto | `synthetic_utils.py` |
| `23_simular_video_sem_webcam.py` | Vídeo de 90 frames com o tabuleiro em movimento | nada | `dados_sinteticos/video_tabuleiro.avi` | `synthetic_utils.py` |
| `24_ar_em_video_sintetico.py` | Regenera os 90 frames, estima a pose e desenha o cubo em cada um | nada | `saidas/24_ar_video_sintetico.avi` | `synthetic_utils.py`, pasta `saidas/` |
| `25_efeito_do_ruido.py` | Erro médio com imagens limpas e ruidosas | gera o dataset 2 vezes | texto; deixa `calib_NN.png` ruidosas | `synthetic_utils.py` |
| `26_pouca_variedade_de_pose.py` | Calibra com 12 vistas quase iguais e compara com `TRUE_K` | nada | `dados_sinteticos/pouca_var_00.png` a `pouca_var_11.png` e texto | `synthetic_utils.py` |
| `27_identificar_imagem_com_maior_erro.py` | Acha a imagem de maior erro no conjunto ruidoso | gera o dataset ruidoso | texto; deixa `calib_NN.png` ruidosas | `synthetic_utils.py` |
| `28_recalibrar_sem_pior_imagem.py` | Remove a pior imagem e recalibra | gera o dataset ruidoso | texto; deixa `calib_NN.png` ruidosas | `synthetic_utils.py` |
| `29_pipeline_completo_imagem_sintetica.py` | Gera, calibra, mede o erro, estima a pose e projeta o cubo com a K estimada | nada | `saidas/29_pipeline_completo.png` e texto | `synthetic_utils.py` |
| `30_ponte_para_camera_real.py` | Imprime o pseudocódigo do laço com `VideoCapture` | nada | texto | nada |

## 5. Passo a passo para executar

Todos os comandos abaixo são executados de dentro de `aula_16_Calibracao/exemplos_em_python/`, com o venv ativado. Nenhum script abre janela: tudo vai para o terminal ou para arquivos em `dados_sinteticos/` e `saidas/`, então não há tecla para encerrar. Como os scripts são curtos, a ordem numérica funciona, com um cuidado no passo 15.

Passo 1. Criar a textura do tabuleiro. Regrava `tabuleiro_base.png`, idêntico ao do zip. O terminal confirma 7 x 6 cantos e 8 x 7 quadrados.

```
python 01_criar_tabuleiro.py
```

Passo 2. Gerar uma vista com pose conhecida. Abra `dados_sinteticos/vista_unica.png` e compare com os valores impressos: `tvec` termina em 0,75 m e `rvec` tem três componentes em radianos.

```
python 02_gerar_uma_vista_sintetica.py
```

Passo 3. Gerar as 18 imagens de calibração. Regrava `calib_01.png` a `calib_18.png`, idênticas às do zip. Abra algumas: há vistas quase frontais, inclinadas e mais distantes.

```
python 03_gerar_18_imagens_calibracao.py
```

Passo 4. Detectar e refinar cantos. O 04 deve imprimir `Tabuleiro encontrado? True` e `Quantidade de cantos: 42`, e salvar `saidas/04_cantos_detectados.png` com os cantos coloridos. O 05 imprime o primeiro canto antes e depois do refinamento; a diferença é de fração de pixel.

```
python 04_detectar_cantos.py
python 05_refinar_cantos_subpixel.py
```

Passo 5. Pontos 3D e correspondências. O 06 mostra o array de forma `(42, 3)` com os primeiros pontos em múltiplos de 0,03. O 07 deve reportar 18 imagens válidas, `(42, 3)` pontos 3D e `(42, 1, 2)` pontos 2D por imagem.

```
python 06_criar_pontos_3d_tabuleiro.py
python 07_coletar_pontos_2d_3d.py
```

Passo 6. Calibrar e conferir. O 08 imprime o RMS, a K e os cinco coeficientes. O 09 compara com `TRUE_K`: espere `fx` perto de 900, `fy` perto de 910, `cx` e `cy` perto de 640 e 360, e erro relativo de fração de por cento. O 10 mostra coeficientes de distorção próximos de zero.

```
python 08_calibrar_camera.py
python 09_comparar_K_real_e_estimado.py
python 10_interpretar_coeficientes_distorcao.py
```

Passo 7. Erro de reprojeção. O 11 lista o erro de cada `calib_NN.png`; o 12 imprime a média e a tabela de referência. Com imagens limpas os valores ficam bem abaixo de 0,5 px.

```
python 11_erro_reprojecao_por_imagem.py
python 12_erro_medio_reprojecao.py
```

Passo 8. Salvar e carregar. O 13 grava `camera_sintetica.npz` na pasta dos scripts; o 14 lê e imprime. Se rodar o 14 antes do 13, ele avisa e encerra.

```
python 13_salvar_calibracao_real.py
python 14_carregar_calibracao.py
```

Passo 9. Corrigir distorção. O 15 grava `saidas/15_original.png` e `saidas/15_corrigida.png`; o 16 grava o painel `saidas/16_painel_original_corrigida.png`. As imagens são praticamente iguais, e isso é o esperado.

```
python 15_undistort_em_imagem_sintetica.py
python 16_painel_original_corrigida.py
```

Passo 10. Pose e rotação. O 17 imprime `solvePnP funcionou? True`, o `rvec` e o `tvec` estimados e o `tvec` real; compare os três componentes. O 18 imprime R e `R.T @ R`, que deve ser a identidade com resíduos da ordem de 1e-16.

```
python 17_estimar_pose_solvepnp.py
python 18_converter_rvec_para_matriz_R.py
```

Passo 11. Realidade aumentada em imagem. Três arquivos em `saidas/`: `19_eixos_3d.png` (eixos vermelho, verde e azul saindo do primeiro canto), `20_cubo_3d.png` (cubo de arame) e `21_faces_coloridas.png` (faces translúcidas).

```
python 19_projectpoints_eixos_3d.py
python 20_projetar_cubo_3d.py
python 21_faces_coloridas_do_cubo.py
```

Passo 12. Sequência de poses. O 22 imprime `rvec` e `tvec` de 4 frames com inclinação e distância crescentes; observe `tz` subindo de 0,70 a 0,76.

```
python 22_imprimir_pose_em_sequencia.py
```

Passo 13. Vídeo sintético e AR em vídeo. O 23 grava `dados_sinteticos/video_tabuleiro.avi` (90 frames, 15 fps). O 24 grava `saidas/24_ar_video_sintetico.avi` com o cubo laranja acompanhando o tabuleiro. Abra os dois em qualquer player; cada um dura 6 segundos.

```
python 23_simular_video_sem_webcam.py
python 24_ar_em_video_sintetico.py
```

Passo 14. Efeito do ruído e da pouca variedade. O 25 imprime dois erros médios, o ruidoso maior. O 26 grava 12 `pouca_var_NN.png` e mostra a K estimada longe de `TRUE_K`.

```
python 25_efeito_do_ruido.py
python 26_pouca_variedade_de_pose.py
```

Passo 15. Diagnóstico com imagens ruidosas. O 27 diz qual `calib_NN.png` teve o maior erro; o 28 recalibra sem ela e compara. Os números mudam a cada execução, porque o ruído não tem semente. Atenção: depois dos scripts 25, 27 e 28 as 18 imagens em `dados_sinteticos/` estão ruidosas. Rode o 03 de novo para restaurar as limpas antes de continuar.

```
python 27_identificar_imagem_com_maior_erro.py
python 28_recalibrar_sem_pior_imagem.py
python 03_gerar_18_imagens_calibracao.py
```

Passo 16. Pipeline completo. O 29 regenera as 18 imagens limpas, calibra, imprime K, coeficientes e erro médio, estima a pose de uma vista nova com a K estimada e grava `saidas/29_pipeline_completo.png` com o cubo.

```
python 29_pipeline_completo_imagem_sintetica.py
```

Passo 17. Ponte para a câmera real. O 30 só imprime o pseudocódigo. O programa de verdade é o desafio final da aula (seção 6).

```
python 30_ponte_para_camera_real.py
```

Ao terminar todos os passos, `saidas/` contém `04_cantos_detectados.png`, `15_original.png`, `15_corrigida.png`, `16_painel_original_corrigida.png`, `19_eixos_3d.png`, `20_cubo_3d.png`, `21_faces_coloridas.png`, `24_ar_video_sintetico.avi` e `29_pipeline_completo.png`. `dados_sinteticos/` ganhou `video_tabuleiro.avi` e `pouca_var_00.png` a `pouca_var_11.png`, e a pasta dos scripts ganhou `camera_sintetica.npz`.

## 6. Exercícios e itens da aula

A aula não tem itens A e B nem exercícios numerados à parte. Os exercícios são os blocos `# DESAFIO DO ALUNO` no fim de cada script, e o docstring de todos diz a regra: a parte principal já funciona, e o aluno deve modificar, medir, comparar ou completar. A lista completa, script a script:

1. `01_criar_tabuleiro.py`: alterar `square_px` para 60 e 140; explicar por que um padrão de 7 x 6 cantos internos precisa de 8 x 7 quadrados.
2. `02_gerar_uma_vista_sintetica.py`: gerar uma segunda imagem alterando só `tz` e responder se aumentar `tz` aproxima ou afasta o tabuleiro.
3. `03_gerar_18_imagens_calibracao.py`: abrir as imagens e apontar uma quase frontal, uma inclinada e uma mais distante; explicar por que a variedade de poses ajuda.
4. `04_detectar_cantos.py`: trocar `paths[4]` por outras imagens; todas são detectadas? qual pose parece mais difícil?
5. `05_refinar_cantos_subpixel.py`: calcular a distância em pixels entre o primeiro canto original e o refinado.
6. `06_criar_pontos_3d_tabuleiro.py`: alterar `SQUARE_SIZE_M` para 0,025 em `synthetic_utils.py`; o que muda nas coordenadas 3D e qual grandeza da pose é afetada.
7. `07_coletar_pontos_2d_3d.py`: explicar com as próprias palavras o que `objpoints` e `imgpoints` representam.
8. `08_calibrar_camera.py`: identificar em K os valores `fx`, `fy`, `cx` e `cy` e imprimir cada um separadamente.
9. `09_comparar_K_real_e_estimado.py`: calcular também o erro relativo de `fy`.
10. `10_interpretar_coeficientes_distorcao.py`: pesquisar a diferença física entre distorção radial e tangencial.
11. `11_erro_reprojecao_por_imagem.py`: descobrir qual imagem apresentou o maior erro.
12. `12_erro_medio_reprojecao.py`: calcular também o desvio padrão dos erros.
13. `13_salvar_calibracao_real.py`: gravar no `.npz` também a largura e a altura da imagem.
14. `14_carregar_calibracao.py`: imprimir só `fx`, `fy`, `cx` e `cy`.
15. `15_undistort_em_imagem_sintetica.py`: explicar por que a correção visual é pequena neste experimento.
16. `16_painel_original_corrigida.py`: desenhar uma linha vertical exatamente no centro entre as duas imagens.
17. `17_estimar_pose_solvepnp.py`: calcular o erro euclidiano entre `tvec_est` e `tvec_real`.
18. `18_converter_rvec_para_matriz_R.py`: explicar por que `R.T @ R` deve se aproximar da identidade.
19. `19_projectpoints_eixos_3d.py`: aumentar o comprimento dos eixos para 2 quadrados.
20. `20_projetar_cubo_3d.py`: fazer o cubo ter aresta de 2 quadrados.
21. `21_faces_coloridas_do_cubo.py`: escolher uma face e alterar somente a cor dela.
22. `22_imprimir_pose_em_sequencia.py`: acrescentar mais 4 frames com `tz` crescendo progressivamente.
23. `23_simular_video_sem_webcam.py`: mudar o movimento para que o tabuleiro também varie em `ty`.
24. `24_ar_em_video_sintetico.py`: dar cores diferentes às arestas da base, do topo e verticais.
25. `25_efeito_do_ruido.py`: aumentar `noise_std` dentro de `generate_dataset` e repetir a comparação.
26. `26_pouca_variedade_de_pose.py`: explicar por que muitas imagens quase iguais não substituem um conjunto com ângulos e distâncias variados.
27. `27_identificar_imagem_com_maior_erro.py`: encontrar também a imagem com menor erro.
28. `28_recalibrar_sem_pior_imagem.py`: a remoção sempre melhora a qualidade física da calibração? discutir por que "menor erro" não é o único critério.
29. `29_pipeline_completo_imagem_sintetica.py`: trocar o cubo de arame por faces semitransparentes de cores distintas (técnica do script 21).
30. `30_ponte_para_camera_real.py`: o desafio final da aula. Transformar o pseudocódigo em um programa funcional com `cv2.VideoCapture`. Sugestões do professor: começar com um vídeo gravado, depois trocar pelo índice 0 da webcam; usar `camera_sintetica.npz` somente para testes sintéticos; para a webcam real, fazer uma calibração real da própria câmera, com um tabuleiro impresso e fotos em poses variadas, reaproveitando os scripts 04 a 13.

## 7. Armadilhas e dicas

- Rode de dentro de `exemplos_em_python/`. Os caminhos usam `Path(__file__).resolve().parent`, então funcionam de qualquer diretório, mas `synthetic_utils` precisa estar no caminho de importação, e o Python coloca lá a pasta do script executado, não o diretório de trabalho. Rodar `python aula_16_Calibracao/exemplos_em_python/08_calibrar_camera.py` da raiz também funciona por esse motivo.
- Os scripts 25, 27 e 28 sobrescrevem `calib_01.png` a `calib_18.png` com versões ruidosas (desfoque 0,8 e ruído 2,0). Depois deles, os scripts 08 a 16 calibram com imagens ruidosas e os números pioram sem aviso. Rode o 03, ou qualquer script que chame `generate_dataset` sem `noisy`, para regravar as limpas. O 29 já faz isso sozinho.
- O ruído de `generate_view` usa `np.random.normal` sem semente. Os erros impressos por 25, 27 e 28 mudam a cada execução, e a "pior imagem" do 27 pode não ser a mesma duas vezes. Para resultados reprodutíveis, coloque `np.random.seed(...)` no início desses scripts.
- Todo script que chama `generate_dataset` regera as 18 imagens antes de trabalhar, mesmo que já existam. Isso custa alguns segundos por script e é o que garante que as imagens estejam limpas. Editar um `calib_NN.png` à mão não adianta.
- O erro de `reprojection_errors` é `norm L2 / número de pontos`, a fórmula do tutorial do OpenCV, e não o RMS. Para 42 pontos ele sai cerca de 6,5 vezes menor que o RMS impresso pelo script 08. Não compare os dois números diretamente, e lembre disso ao aplicar a tabela de referência do 12.
- `21_faces_coloridas_do_cubo.py` não testa o `ok` de `find_refined_corners` antes de chamar `solvePnP`. Com a pose fixa do script a detecção sempre funciona, mas ao adaptar para outra imagem, acrescente a verificação, como fazem 17, 19, 20 e 24.
- Os scripts 19, 20, 21 e 24 gravam em `saidas/` sem chamar `ensure_dirs`; o 02 grava em `dados_sinteticos/` sem criar a pasta. No repositório as duas pastas existem (o `.gitkeep` e as imagens do professor), e os scripts 03 a 16 as criam de qualquer forma. Se apagar `saidas/`, `cv2.imwrite` falha em silêncio e nada é salvo: rode o 03 ou crie a pasta.
- `cv2.imwrite` não lança exceção quando falha; devolve `False`. Se um arquivo esperado não aparecer em `saidas/`, o motivo mais comum é a pasta não existir.
- `24_ar_em_video_sintetico.py` não lê `video_tabuleiro.avi`; ele regenera cada frame com as fórmulas do 23. O 23 não é pré-requisito do 24, e alterar o 23 não altera o 24.
- Os vídeos saem em AVI com codec MJPG, 15 fps, 1280 x 720, 6 segundos. Ficam com dezenas de megabytes e estão fora do git.
- A tupla de `calibrate_from_paths` tem 9 elementos, sempre nesta ordem: `rms, K, dist, rvecs, tvecs, objpoints, imgpoints, used, image_size`. Os scripts descartam o que não usam com `_` e `*_`; ao reaproveitar a função, respeite a ordem.
- `calibrate_from_paths` exige ao menos 3 imagens com o tabuleiro detectado, senão lança `RuntimeError`. Na prática use 15 ou mais, como o comentário do 29 recomenda.
- `solvePnP`, `projectPoints` e `calibrateCamera` esperam `objectPoints` em `float32` e devolvem `rvec` e `tvec` de forma `(3, 1)`. As unidades de `tvec` são as de `object_points`, aqui metros. Se calibrar uma câmera real com um tabuleiro impresso, meça o quadrado com régua e ajuste `SQUARE_SIZE_M`, senão a distância sai errada na mesma proporção.
- `PATTERN_SIZE` conta cantos internos, não quadrados. Um tabuleiro impresso de 9 x 6 quadrados tem 8 x 5 cantos. Errar isso faz `findChessboardCorners` devolver `False` em todas as imagens.
- Nos scripts 17 a 24 a pose é estimada com `TRUE_K` e `TRUE_DIST`, a verdade do simulador. Isso isola o erro do `solvePnP`, mas não é o fluxo real. O 29 mostra o fluxo real, com a K e a `dist` estimadas. Numa câmera física só existe a versão do 29.
- O eixo Z dos eixos e do cubo é negativo (`-s`) para apontar da placa em direção à câmera. Se desenhar com `+s`, o cubo "entra" na mesa.
- As cores são BGR: `(0,0,255)` é vermelho, `(255,0,0)` é azul. Os comentários `# X`, `# Y`, `# Z` do script 19 seguem a convenção vermelho, verde, azul.
- No 26, a calibração com 12 vistas quase frontais não falha: ela apenas devolve uma K errada. `calibrateCamera` não avisa quando o conjunto é degenerado; o único sinal é a comparação com a verdade, que só existe aqui. Com uma câmera real, a defesa é a variedade de poses e um erro de reprojeção baixo em muitas imagens diferentes.
- `20_projetar_cubo_3d.py`, `21_faces_coloridas_do_cubo.py` e `24_ar_em_video_sintetico.py` desenham todas as arestas ou faces, sem teste de oclusão. Em `21` isso fica disfarçado pela transparência; para um cubo opaco seria preciso ordenar as faces pela profundidade.
- O `.npz` do script 13 chama-se `camera_sintetica.npz` de propósito: é a calibração da câmera virtual e não serve para nenhuma webcam. O desafio do 30 é explícito quanto a isso.
- Parâmetros que valem a pena variar: `square_px` no 01, `tz` e os ângulos no 02, `(11, 11)` e o critério no 05, `n` em `generate_dataset` (menos imagens, pior K), `noise_std` e `blur` no 25, o número e a abertura angular das poses no 26, e `s` (comprimento dos eixos e aresta do cubo) nos scripts 19 a 21.

## 8. Glossário

| Termo | Significado |
|---|---|
| Calibração de câmera | Estimativa dos parâmetros intrínsecos (K e distorção) a partir de várias imagens de um padrão conhecido. |
| Parâmetros intrínsecos | Propriedades internas da câmera: distâncias focais, ponto principal e coeficientes de distorção. Não dependem de onde a câmera está. |
| Parâmetros extrínsecos | Pose da câmera em relação ao objeto ou ao mundo: rotação `rvec` e translação `tvec`. |
| Modelo pinhole | Modelo de câmera em que cada ponto 3D é projetado por uma reta que passa pelo centro óptico. Base da matriz K. |
| Matriz intrínseca K | Matriz 3 x 3 com `fx`, `fy`, `cx`, `cy`. Converte coordenadas normalizadas em pixels. |
| fx, fy | Distâncias focais em pixels nos eixos horizontal e vertical. |
| cx, cy | Ponto principal: pixel por onde passa o eixo óptico, em geral perto do centro da imagem. |
| Distorção radial (k1, k2, k3) | Deformação que cresce com a distância ao centro; produz efeito barril ou almofada. |
| Distorção tangencial (p1, p2) | Deformação causada pelo desalinhamento entre lente e sensor. |
| dist | Vetor de 5 coeficientes do OpenCV na ordem `k1, k2, p1, p2, k3`. |
| Tabuleiro (chessboard) | Padrão de calibração de quadrados alternados. O OpenCV usa os cantos internos, onde quatro quadrados se tocam. |
| PATTERN_SIZE | Número de cantos internos por linha e por coluna: aqui `(7, 6)`, de um tabuleiro de 8 x 7 quadrados. |
| SQUARE_SIZE_M | Lado físico de cada quadrado, aqui 0,03 m. Define a escala de `tvec`. |
| Câmera virtual | Simulador em `synthetic_utils.py` que renderiza o tabuleiro com K e distorção conhecidas. |
| TRUE_K, TRUE_DIST | Valores verdadeiros do simulador, usados para conferir as estimativas. |
| Homografia | Transformação projetiva 3 x 3 entre dois planos. Usada em `generate_view` para deformar a textura. |
| `getPerspectiveTransform` | Calcula a homografia a partir de 4 pares de pontos. |
| `warpPerspective` | Aplica uma homografia a uma imagem inteira. |
| Ângulos de Euler | Três rotações sucessivas em torno dos eixos X, Y e Z, em graus nos scripts. |
| Matriz de rotação R | Matriz 3 x 3 ortonormal que gira vetores; `R.T @ R` é a identidade. |
| Vetor de Rodrigues (rvec) | Rotação compacta em 3 números: direção é o eixo, comprimento é o ângulo em radianos. |
| `cv2.Rodrigues` | Converte entre `rvec` e R nos dois sentidos. |
| tvec | Vetor de translação `(3, 1)` da origem do objeto até a câmera, em metros aqui. |
| Pose | Par `rvec`, `tvec` que posiciona o objeto no referencial da câmera. |
| Referencial do tabuleiro | Sistema de coordenadas com origem no primeiro canto interno, X nas colunas, Y nas linhas, Z perpendicular à placa. |
| objpoints | Lista com as coordenadas 3D dos cantos, uma cópia por imagem, todas com Z = 0. |
| imgpoints | Lista com as coordenadas 2D refinadas dos cantos em cada imagem. |
| `findChessboardCorners` | Detecta os cantos internos e devolve se achou e o array `(N, 1, 2)` ordenado. |
| `drawChessboardCorners` | Desenha os cantos detectados com cores e linhas de ligação, para conferir a ordem. |
| `cornerSubPix` | Refina a posição dos cantos para precisão subpixel dentro de uma janela de busca. |
| Subpixel | Precisão menor que um pixel, obtida por interpolação do gradiente. |
| Critério de parada | Tupla `(flags, max_iter, eps)`: aqui 40 iterações ou deslocamento menor que 0,001 px. |
| `calibrateCamera` | Função que estima K, distorção e a pose de cada imagem a partir de `objpoints` e `imgpoints`. |
| RMS | Root Mean Square. Raiz da média dos quadrados dos erros de reprojeção, devolvida por `calibrateCamera`. |
| Erro de reprojeção | Distância em pixels entre o canto detectado e a projeção do ponto 3D com os parâmetros estimados. |
| `projectPoints` | Projeta pontos 3D na imagem usando `rvec`, `tvec`, K e `dist`. |
| `undistort` | Remapeia a imagem para remover a distorção da lente. |
| PnP | Perspective-n-Point. Problema de achar a pose a partir de n correspondências 3D e 2D com K conhecida. |
| `solvePnP` | Resolve o PnP e devolve `rvec` e `tvec`. |
| Realidade aumentada (AR) | Desenhar objetos virtuais sobre a imagem real de forma consistente com a pose da câmera. |
| `fillConvexPoly` | Preenche um polígono convexo. Usado para as faces do cubo. |
| `addWeighted` | Mistura ponderada de duas imagens; dá a transparência das faces. |
| `polylines` | Desenha o contorno de um ou mais polígonos. |
| Variedade de pose | Diversidade de ângulos, distâncias e posições do padrão nas imagens de calibração. Sem ela, K fica mal determinada. |
| Conjunto degenerado | Conjunto de imagens que não permite separar os parâmetros, como vistas frontais quase iguais. |
| `np.savez` / `np.load` | Gravação e leitura de vários arrays NumPy em um único arquivo `.npz`. |
| MJPG | Codec Motion JPEG usado nos vídeos AVI gerados pelos scripts 23 e 24. |
| BGR | Ordem dos canais de cor do OpenCV: azul, verde, vermelho. `(0,0,255)` é vermelho. |
| DESAFIO DO ALUNO | Bloco de comentário no fim de cada script com a extensão que o aluno deve fazer. |
