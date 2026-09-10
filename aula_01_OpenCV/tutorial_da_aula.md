# Aula 01 – OpenCV

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula apresenta o OpenCV como ferramenta de entrada e saída de imagens e vídeo, sempre apoiado no NumPy. O fio condutor é a ideia de que uma imagem digital é um array NumPy: uma matriz de altura por largura, com três canais na ordem BGR quando colorida, ou apenas dois eixos quando em escala de cinza. Todos os scripts giram em torno de criar, ler, gravar, exibir e inspecionar esses arrays.

Os oito scripts evoluem em três blocos. Os Exemplos 00 a 02 trabalham sem câmera: validam o ambiente, criam uma imagem sintética, inspecionam shape e dtype e geram um vídeo MP4 de teste. Os Exemplos 03 a 05 usam a webcam para exibir o stream com metadados e FPS (Item A), capturar um frame, salvá-lo e recarregá-lo em cinza (Item B). Os Exemplos 06 e 07 integram tudo em um único programa interativo com captura pela tecla S, média móvel de FPS e painel comparativo colorido e cinza.

## 2. Ambiente e pré-requisitos

- Bibliotecas de terceiros: `cv2` (pacote `opencv-python`) e `numpy`. O resto vem da biblioteca padrão: `pathlib`, `argparse`, `collections`, `typing`.
- Instalação:

```
pip install opencv-python numpy
```

- Pasta de execução: os scripts resolvem caminhos com `Path(__file__).resolve().parent`, então os arquivos gerados ficam sempre em `exemplos_em_python/`, independentemente do diretório de trabalho. Ainda assim, o mais simples é rodar de dentro dessa pasta.
- Material extra: nenhum. A aula é autossuficiente. Segundo `achados_dos_zips.md`, nada falta para esta aula.
- Webcam: obrigatória nos Exemplos 03, 04, 06 e 07. Os Exemplos 00, 01, 02 e 05 rodam sem câmera (o 05 depende do arquivo gerado pelo 04).
- Versão de Python: o professor não informa nesta aula. Nas outras aulas ele usou 3.12 e 3.13. A máquina local tem 3.14.3.
- Os scripts vieram como documentos Word e foram convertidos para `.py`. A única correção feita foi um tab indevido antes do docstring do Exemplo 04.

## 3. Conceitos-chave

### 3.1 Verificação do ambiente e versões

Antes de qualquer processamento, vale confirmar que as bibliotecas estão instaladas e registrar as versões, porque diferenças de versão do OpenCV mudam nomes de constantes e comportamento de codecs. O professor faz isso imprimindo `cv2.__version__` e `np.__version__` e, em seguida, exercitando o módulo de desenho e a gravação em disco, o que testa de ponta a ponta o pacote `imgproc` e `imgcodecs` sem depender de câmera.

Trecho de `00_verificar_ambiente.py`:

```python
print(f"Versao do OpenCV: {cv2.__version__}")
print(f"Versao do NumPy:  {np.__version__}")

imagem = np.zeros((240, 320, 3), dtype=np.uint8)
cv2.rectangle(imagem, (20, 20), (300, 220), (0, 180, 0), thickness=3)
cv2.circle(imagem, (160, 120), 55, (255, 0, 0), thickness=-1)
gravou = cv2.imwrite(str(caminho_saida), imagem)
if not gravou:
    raise RuntimeError("Nao foi possivel salvar a imagem de teste.")
```

Onde aparece: `00_verificar_ambiente.py`.

### 3.2 Imagem como array NumPy: shape, dtype e ordem BGR

Para o OpenCV, uma imagem é um `numpy.ndarray`. Uma imagem colorida tem `shape` igual a `(altura, largura, canais)`, e uma imagem em cinza tem `shape` igual a `(altura, largura)`. O `dtype` mais comum é `uint8`, ou seja, inteiros sem sinal de 0 a 255 por canal. Um detalhe que o professor destaca várias vezes: o OpenCV armazena os canais na ordem BGR (azul, verde, vermelho), e não RGB. Por isso a cor `(255, 0, 0)` desenha um círculo azul no Exemplo 00, e `(0, 255, 0)` é verde nos textos dos demais scripts.

Criar uma imagem do zero é só criar um array de zeros com `np.zeros((altura, largura, 3), dtype=np.uint8)`. Acessar `imagem[0, 0]` devolve o pixel do canto superior esquerdo: um vetor de três valores em BGR na colorida, ou um único inteiro na cinza. A convenção de índices é `[linha, coluna]`, que corresponde a `[y, x]`.

Trecho de `01_imagem_e_array_numpy.py`:

```python
altura, largura, canais = colorida.shape
print(f"Shape: {colorida.shape}")
print(f"Dtype: {colorida.dtype}")
print(f"Canais: {canais} (B, G, R)")
print(f"Pixel [0, 0] em BGR: {colorida[0, 0]}")

cinza = cv2.cvtColor(colorida, cv2.COLOR_BGR2GRAY)
print(f"Shape: {cinza.shape}")            # (H, W), sem eixo de canais
print(f"Pixel [0, 0] (intensidade): {cinza[0, 0]}")
```

Onde aparece: `00_verificar_ambiente.py`, `01_imagem_e_array_numpy.py`, `03_camera_metadados_fps.py`, `04_capturar_e_salvar_frame.py`, `05_recarregar_cinza_hconcat.py`, `06_item_b_completo_camera.py`, `07_pipeline_integrado_a_b.py`.

### 3.3 Primitivas de desenho: rectangle, circle, line e putText

O módulo de desenho do OpenCV altera o array no lugar (in place). As funções usadas na aula são:

- `cv2.rectangle(img, ponto1, ponto2, cor, thickness)`: retângulo pelos cantos opostos.
- `cv2.circle(img, centro, raio, cor, thickness)`: `thickness=-1` preenche a forma.
- `cv2.line(img, ponto1, ponto2, cor, espessura)`: segmento de reta.
- `cv2.putText(img, texto, origem, fonte, escala, cor, espessura, tipo_de_linha)`: escreve texto. A origem é o canto inferior esquerdo da primeira letra, a fonte usada é sempre `cv2.FONT_HERSHEY_SIMPLEX`, e `cv2.LINE_AA` ativa o antialiasing (suavização das bordas).

Pontos são tuplas `(x, y)`, ao contrário da indexação do array, que é `[y, x]`. O professor usa `putText` em todos os scripts para sobrepor rótulos e metadados ao frame, o que serve para demonstração em aula e para depuração.

Trecho de `03_camera_metadados_fps.py`:

```python
cv2.putText(
    frame,                       # Imagem
    texto,                       # Texto
    (10, 30),                    # Posição inicial
    cv2.FONT_HERSHEY_SIMPLEX,    # Fonte
    0.65,                        # Tamanho da fonte
    (0, 255, 0),                 # Cor BGR: verde
    2,                           # Espessura
    cv2.LINE_AA,                 # Suavização das letras
)
```

Onde aparece: `00_verificar_ambiente.py` (rectangle, circle, putText), `02_gerar_video_teste.py` (circle, line, putText), e putText em todos os demais.

### 3.4 Gravação e leitura de imagem: imwrite e imread

`cv2.imwrite(caminho, array)` grava o array em disco no formato indicado pela extensão (aqui sempre PNG) e retorna `True` ou `False`. O professor sempre testa esse retorno, porque um caminho inválido não lança exceção, só devolve `False`.

`cv2.imread(caminho, flag)` faz o caminho inverso. A flag `cv2.IMREAD_COLOR` força três canais BGR. Se o arquivo não existir ou estiver corrompido, a função retorna `None` em vez de lançar erro, então o teste `if imagem is None` é obrigatório. Ambas as funções recebem o caminho como string, por isso os scripts fazem `str(caminho)` ao converter um objeto `Path`.

Trecho de `05_recarregar_cinza_hconcat.py`:

```python
colorida = cv2.imread(str(caminho_imagem), cv2.IMREAD_COLOR)
if colorida is None:
    raise FileNotFoundError(
        f"Nao foi possivel ler {caminho_imagem}. "
        "Execute primeiro 04_capturar_e_salvar_frame.py."
    )
```

Onde aparece: `00_verificar_ambiente.py`, `01_imagem_e_array_numpy.py`, `04_capturar_e_salvar_frame.py`, `05_recarregar_cinza_hconcat.py`, `06_item_b_completo_camera.py`, `07_pipeline_integrado_a_b.py`.

### 3.5 Caminhos relativos ao script com pathlib

Todos os scripts calculam a pasta onde eles próprios estão com `Path(__file__).resolve().parent` e gravam ou leem arquivos ali. Isso evita o erro clássico de rodar o script de outro diretório e não encontrar a imagem. O `resolve()` transforma o caminho em absoluto, e o operador `/` do `pathlib` monta o caminho final.

Trecho de `00_verificar_ambiente.py`:

```python
from pathlib import Path

caminho_saida = Path(__file__).resolve().parent / "teste_ambiente.png"
gravou = cv2.imwrite(str(caminho_saida), imagem)
```

Onde aparece: todos os scripts, exceto `03_camera_metadados_fps.py`, que não grava nada.

### 3.6 Conversão de espaço de cor com cvtColor

`cv2.cvtColor(imagem, codigo)` converte entre representações de cor. A aula usa dois códigos:

- `cv2.COLOR_BGR2GRAY`: reduz os três canais a uma única matriz de intensidade. O resultado perde o eixo de canais e passa a ter shape `(H, W)`.
- `cv2.COLOR_GRAY2BGR`: replica a intensidade nos três canais. A imagem continua visualmente cinza, mas volta a ter shape `(H, W, 3)`. O professor usa isso só para exibição, e deixa claro nos comentários que a variável `cinza` original permanece 2D, porque é ela que responde ao enunciado.

Trecho de `01_imagem_e_array_numpy.py`:

```python
cinza = cv2.cvtColor(colorida, cv2.COLOR_BGR2GRAY)
# hconcat exige arrays com mesma altura, mesmo dtype e mesmo numero de canais.
# A imagem cinza e 2D; por isso criamos apenas uma copia BGR para exibicao.
cinza_para_exibir = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
```

Onde aparece: `01_imagem_e_array_numpy.py`, `05_recarregar_cinza_hconcat.py`, `06_item_b_completo_camera.py`, `07_pipeline_integrado_a_b.py`.

### 3.7 Painel lado a lado com hconcat

`cv2.hconcat([img1, img2, ...])` concatena arrays horizontalmente. A exigência é que todos tenham a mesma altura, o mesmo dtype e o mesmo número de canais. Como a imagem em cinza é 2D e a colorida é 3D, não dá para concatenar direto: é preciso passar a cinza por `COLOR_GRAY2BGR` antes. O professor também posiciona o rótulo da segunda metade do painel usando `colorida.shape[1] + 10`, ou seja, a largura da primeira imagem mais uma margem.

Trecho de `06_item_b_completo_camera.py`:

```python
cinza_bgr = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
painel = cv2.hconcat([colorida, cinza_bgr])
cv2.putText(painel, "Colorida", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(painel, "Escala de cinza", (colorida.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 0), 2, cv2.LINE_AA)
```

Onde aparece: `01_imagem_e_array_numpy.py`, `05_recarregar_cinza_hconcat.py`, `06_item_b_completo_camera.py`, `07_pipeline_integrado_a_b.py`.

### 3.8 Janelas, imshow, waitKey e destroyAllWindows

`cv2.imshow(titulo, array)` abre (ou atualiza) uma janela com o array. A janela só é desenhada de fato quando `cv2.waitKey(ms)` é chamado: essa função processa os eventos da interface, espera até `ms` milissegundos por uma tecla e devolve o código dela, ou -1 se nada foi pressionado. O `& 0xFF` isola o byte baixo do código, o que deixa o valor portável entre sistemas. Comparar com `ord("q")` e com `27` (código do Esc) é o padrão usado em toda a aula. Ao final, `cv2.destroyAllWindows()` fecha as janelas.

Dois ritmos de `waitKey` aparecem nos scripts: `waitKey(1)` quando há um stream de câmera, para não atrasar o próximo `cap.read()`, e `waitKey(30)` dentro de um `while True` quando a imagem é estática e só se espera a tecla de saída.

Trecho de `01_imagem_e_array_numpy.py`:

```python
cv2.imshow("Imagem digital como array NumPy - pressione Q", painel)

while True:
    tecla = cv2.waitKey(30) & 0xFF
    if tecla in (ord("q"), ord("Q"), 27):
        break

cv2.destroyAllWindows()
```

Onde aparece: `01_imagem_e_array_numpy.py`, `03_camera_metadados_fps.py`, `04_capturar_e_salvar_frame.py`, `05_recarregar_cinza_hconcat.py`, `06_item_b_completo_camera.py`, `07_pipeline_integrado_a_b.py`.

### 3.9 Geração de vídeo com VideoWriter e fourcc

Um vídeo é uma sequência de frames gravados a uma taxa nominal de quadros por segundo (FPS). O `cv2.VideoWriter(caminho, fourcc, fps, (largura, altura))` cria o arquivo. O `fourcc` é um código de quatro caracteres que identifica o codec; o professor usa `cv2.VideoWriter_fourcc(*"mp4v")`, geralmente disponível em instalações comuns. É preciso testar `writer.isOpened()` antes de escrever, chamar `writer.write(frame)` para cada frame com exatamente a resolução declarada, e `writer.release()` no fim para fechar o container corretamente. O `try/finally` garante o `release` mesmo se algo falhar.

O Exemplo 02 sintetiza 8 segundos a 30 FPS (240 frames) com um alvo circular que anda na horizontal e oscila na vertical por uma senoide, sobre uma grade de referência. A ideia é ter uma fonte de vídeo reprodutível quando não há webcam.

Trecho de `02_gerar_video_teste.py`:

```python
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(caminho_saida), fourcc, fps, (largura, altura))
if not writer.isOpened():
    raise RuntimeError("Nao foi possivel criar o arquivo de video MP4.")
try:
    for numero_frame in range(total_frames):
        frame = np.zeros((altura, largura, 3), dtype=np.uint8)
        frame[:] = (25, 25, 25)
        x = int((numero_frame / (total_frames - 1)) * (largura - 100)) + 50
        y = int(altura / 2 + 90 * np.sin(numero_frame * 0.08))
        cv2.circle(frame, (x, y), 32, (0, 200, 255), thickness=-1)
        writer.write(frame)
finally:
    writer.release()
```

Onde aparece: `02_gerar_video_teste.py`.

### 3.10 Captura da webcam com VideoCapture

`cv2.VideoCapture(indice)` abre uma câmera pelo índice do dispositivo: 0 costuma ser a principal, e 1 ou 2 são alternativas. O objeto deve ser testado com `cap.isOpened()`. Cada `cap.read()` devolve uma tupla `(ret, frame)`: `ret` é `True` se a captura funcionou e `frame` é o array BGR. Quando `ret` é `False`, o loop deve parar. Ao final, `cap.release()` libera o dispositivo, o que é importante porque a câmera fica bloqueada para outros programas enquanto está aberta. O professor coloca o `release` dentro de um `finally` para que ele rode mesmo em caso de erro.

O Exemplo 03 também mostra `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` e `cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)`, que solicitam uma resolução ao driver. O driver pode aceitar, alterar ou ignorar o pedido, por isso o script lê a resolução real com `frame.shape[:2]`.

Trecho de `03_camera_metadados_fps.py`:

```python
cap = cv2.VideoCapture(indice_camera)
if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    return
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        altura, largura = frame.shape[:2]
finally:
    cap.release()
    cv2.destroyAllWindows()
```

Onde aparece: `03_camera_metadados_fps.py`, `04_capturar_e_salvar_frame.py`, `06_item_b_completo_camera.py`, `07_pipeline_integrado_a_b.py`.

### 3.11 Medição de FPS com getTickCount e getTickFrequency

O FPS real de um pipeline é diferente do FPS nominal da câmera: ele depende do tempo gasto para ler, processar e exibir cada frame. Para medir, o professor usa o relógio de alta precisão do OpenCV. `cv2.getTickCount()` devolve o número de ticks desde um instante de referência, e `cv2.getTickFrequency()` diz quantos ticks cabem em um segundo. A diferença de ticks entre dois frames dividida pela frequência dá o tempo do frame em segundos, e o FPS instantâneo é o inverso desse tempo. O teste `if tempo_frame > 0` evita divisão por zero.

Trecho de `03_camera_metadados_fps.py`:

```python
tick_anterior = cv2.getTickCount()
frequencia_ticks = cv2.getTickFrequency()
while True:
    ret, frame = cap.read()
    tick_atual = cv2.getTickCount()
    tempo_frame = (tick_atual - tick_anterior) / frequencia_ticks
    tick_anterior = tick_atual
    if tempo_frame > 0:
        fps_estimado = 1.0 / tempo_frame
    else:
        fps_estimado = 0.0
```

Onde aparece: `03_camera_metadados_fps.py`, `07_pipeline_integrado_a_b.py`.

### 3.12 Média móvel de FPS com deque

O FPS instantâneo oscila muito de um frame para outro. O Exemplo 07 suaviza o valor exibido guardando as últimas 30 medidas em um `collections.deque(maxlen=30)`. Um deque com `maxlen` descarta automaticamente o item mais antigo quando um novo entra, então ele funciona como uma janela deslizante sem código extra. A função `calcular_media` devolve a média da fila e trata a fila vazia devolvendo zero. O terminal mostra os dois valores, instantâneo e médio, e o texto sobre o frame mostra só o médio.

Trecho de `07_pipeline_integrado_a_b.py`:

```python
from collections import deque

def calcular_media(valores: Deque[float]) -> float:
    if not valores:
        return 0.0
    return sum(valores) / len(valores)

historico_fps: Deque[float] = deque(maxlen=tamanho_janela_fps)
...
historico_fps.append(fps_instantaneo)
fps_medio = calcular_media(historico_fps)
```

Onde aparece: `07_pipeline_integrado_a_b.py`.

### 3.13 Aquecimento da câmera (descarte de frames iniciais)

Logo após abrir, muitas webcams ainda estão ajustando exposição, brilho e foco, e os primeiros frames saem escuros ou desfocados. O Exemplo 04 lê e descarta 10 frames antes de guardar o último como frame válido. Isso é o "aquecimento" da câmera. O script imprime o progresso a cada frame descartado.

Trecho de `04_capturar_e_salvar_frame.py`:

```python
quantidade_aquecimento = 10
frame_capturado = None
for numero in range(quantidade_aquecimento):
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar um frame da câmera.")
        return
    frame_capturado = frame
    print(f"Aquecimento da câmera: {numero + 1}/{quantidade_aquecimento}")
```

Onde aparece: `04_capturar_e_salvar_frame.py`.

### 3.14 Captura interativa por tecla e cópia do frame

Nos Exemplos 06 e 07, o usuário escolhe o momento da captura pressionando S enquanto o stream roda. Dois cuidados aparecem no código. Primeiro, o texto de instrução é desenhado em uma cópia (`frame.copy()`), para que o frame salvo não fique com o texto gravado por cima. Segundo, ao capturar, guarda-se `frame.copy()` e não `frame`, porque o buffer devolvido por `cap.read()` pode ser reaproveitado na próxima leitura. Depois do `break`, o `finally` libera a câmera, e só então o programa segue para gravar, recarregar e comparar. Se o usuário sair com Q ou Esc, a variável do frame continua `None` e o processamento é pulado.

Trecho de `06_item_b_completo_camera.py`:

```python
frame_exibicao = frame.copy()
cv2.putText(frame_exibicao, "S: capturar | Q ou Esc: sair", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("Item B - pressione S para capturar", frame_exibicao)
tecla = cv2.waitKey(1) & 0xFF
if tecla in (ord("s"), ord("S")):
    frame_capturado = frame.copy()
    break
if tecla in (ord("q"), ord("Q"), 27):
    break
```

Onde aparece: `06_item_b_completo_camera.py`, `07_pipeline_integrado_a_b.py`.

### 3.15 Liberação de recursos com try/finally

Câmera aberta, arquivo de vídeo aberto e janelas do OpenCV são recursos do sistema operacional. Se o programa terminar por exceção sem liberá-los, a câmera pode ficar indisponível até o processo morrer, e um MP4 pode ficar corrompido sem o cabeçalho final. O professor envolve os loops em `try/finally` e coloca `cap.release()`, `writer.release()` e `cv2.destroyAllWindows()` no `finally`. Esse padrão se repete em todos os scripts que abrem câmera ou vídeo.

Trecho de `04_capturar_e_salvar_frame.py`:

```python
try:
    ...
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Câmera e janelas liberadas corretamente.")
```

Onde aparece: `02_gerar_video_teste.py`, `03_camera_metadados_fps.py`, `04_capturar_e_salvar_frame.py`, `06_item_b_completo_camera.py`, `07_pipeline_integrado_a_b.py`.

### 3.16 Estrutura de script: main e o guarda __name__

Todos os scripts definem uma função `main() -> None` com docstring e a chamam dentro de `if __name__ == "__main__":`. Isso permite importar o arquivo sem executar nada e é a estrutura padrão que o professor adota em todas as aulas. Os scripts usam type hints (`-> None`, `Deque[float]`, `Optional[Path]`) como documentação.

Trecho de `03_camera_metadados_fps.py`:

```python
def main() -> None:
    """Executa o pipeline básico de captura utilizando a câmera."""
    ...

# Executa a função main somente quando este arquivo é executado diretamente.
if __name__ == "__main__":
    main()
```

Onde aparece: todos os scripts.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `00_verificar_ambiente.py` | Imprime versões do OpenCV e NumPy, cria imagem 320x240 com retângulo, círculo e texto, grava em PNG e imprime shape e dtype | Nenhuma | `teste_ambiente.png` | Nada |
| `01_imagem_e_array_numpy.py` | Lê `teste_ambiente.png`, imprime shape, dtype, altura, largura, canais e pixel [0,0], converte para cinza e mostra painel lado a lado | `teste_ambiente.png` | Janela (nenhum arquivo) | `00_verificar_ambiente.py` |
| `02_gerar_video_teste.py` | Gera MP4 sintético de 8 s a 30 FPS (640x360) com alvo em movimento, grade e metadados desenhados | Nenhuma | `video_teste.mp4` | Nada |
| `03_camera_metadados_fps.py` | Item A: abre a webcam, pede 640x480, exibe stream com resolução, contador de frames e FPS estimado no terminal e na tela; Q encerra | Webcam (índice 0) | Janela (nenhum arquivo) | Webcam |
| `04_capturar_e_salvar_frame.py` | Abre a webcam, descarta 10 frames de aquecimento, salva o último em PNG, imprime shape e dtype e exibe; Q ou Esc fecha | Webcam | `frame_capturado.png` | Webcam |
| `05_recarregar_cinza_hconcat.py` | Segunda parte do Item B: lê `frame_capturado.png`, converte para cinza, imprime shape e dtype de ambas e mostra painel lado a lado | `frame_capturado.png` | Janela (nenhum arquivo) | `04_capturar_e_salvar_frame.py` |
| `06_item_b_completo_camera.py` | Item B completo: stream ao vivo, S captura, salva PNG, recarrega, converte para cinza, imprime metadados e mostra painel colorido e cinza | Webcam | `item_b_frame.png` | Webcam |
| `07_pipeline_integrado_a_b.py` | Itens A e B integrados: stream com resolução, contador, FPS instantâneo e médio (janela de 30), S salva snapshot e abre painel colorido e cinza | Webcam | `snapshot_integrado.png` | Webcam |

## 5. Passo a passo para executar

Abra um terminal dentro de `aula_01_OpenCV/exemplos_em_python/`.

**Passo 1. Validar o ambiente (sem câmera).**

```
python 00_verificar_ambiente.py
```

Espere no terminal as versões do OpenCV e NumPy, `Shape da imagem criada: (240, 320, 3)`, `Dtype da imagem criada: uint8` e `Arquivo gravado: True`. Em disco aparece `teste_ambiente.png`: fundo preto, retângulo verde, círculo azul e o texto "OpenCV OK".

**Passo 2. Inspecionar a imagem como array (sem câmera).**

```
python 01_imagem_e_array_numpy.py
```

O terminal mostra os blocos "IMAGEM COLORIDA" e "IMAGEM EM ESCALA DE CINZA" com shape `(240, 320, 3)` e `(240, 320)`, dtype `uint8` e o pixel `[0, 0]`. Abre uma janela com a colorida à esquerda e a cinza à direita. Tecla Q, q ou Esc fecha.

**Passo 3. Gerar um vídeo de teste (sem câmera).**

```
python 02_gerar_video_teste.py
```

Cria `video_teste.mp4` com 240 frames. O terminal confirma resolução 640x360, FPS 30 e total de frames. Esse vídeo não é consumido por nenhum outro script da aula; ele fica como fonte reprodutível para quem não tem webcam.

**Passo 4. Item A: stream, metadados e FPS (com câmera).**

```
python 03_camera_metadados_fps.py
```

Abre a janela "Camera - pressione Q para encerrar". Cada frame imprime no terminal uma linha `Resolução: 640x480 | Frame: 000001 | FPS estimado: 30.00` (a resolução real pode diferir da pedida). O mesmo texto aparece em verde no canto superior esquerdo do vídeo. Tecla Q ou q encerra. Esc não encerra neste script.

**Passo 5. Item B, primeira parte: capturar um frame (com câmera).**

```
python 04_capturar_e_salvar_frame.py
```

O terminal conta "Aquecimento da câmera: 1/10" até 10/10, depois imprime o caminho, shape e dtype de `frame_capturado.png` e abre a janela com o frame. Q ou Esc fecha.

**Passo 6. Item B, segunda parte: recarregar e comparar (sem câmera).**

```
python 05_recarregar_cinza_hconcat.py
```

Lê o PNG do passo anterior, imprime "VALIDACAO DOS ARRAYS NUMPY" com os dois shapes e abre o painel original e cinza. Q ou Esc fecha. O docstring menciona `--imagem`, mas o argumento não é implementado; o script sempre lê `frame_capturado.png`.

**Passo 7. Item B em um único programa (com câmera).**

```
python 06_item_b_completo_camera.py
```

Stream ao vivo com a instrução "S: capturar | Q ou Esc: sair". Pressione S: o programa salva `item_b_frame.png`, fecha a câmera, recarrega o arquivo, imprime os metadados e abre o painel "Item B - colorida e cinza". Q ou Esc no painel encerra. Se sair com Q antes de capturar, nada é salvo.

**Passo 8. Pipeline integrado A e B (com câmera).**

```
python 07_pipeline_integrado_a_b.py
```

Igual ao passo anterior, mas com FPS instantâneo e médio no terminal e FPS médio sobre o vídeo. S salva `snapshot_integrado.png` e abre o painel "Snapshot colorido e cinza". Q ou Esc encerra em qualquer etapa.

## 6. Exercícios e itens da aula

Os scripts citam dois itens de um exercício prático:

- **Item A: câmera, exibição, metadados e FPS.** Abrir a câmera com `VideoCapture`, exibir os frames em tempo real, mostrar a resolução, contar frames, estimar o FPS com `getTickCount` e encerrar com Q liberando os recursos. Resolvido por `03_camera_metadados_fps.py`.
- **Item B: capturar, salvar, recarregar e comparar.** Capturar um frame da câmera, salvar com `imwrite`, recarregar com `imread`, converter para cinza com `cvtColor`, imprimir shape e dtype das duas representações e exibir colorida e cinza lado a lado com `hconcat`. Resolvido em duas partes por `04_capturar_e_salvar_frame.py` e `05_recarregar_cinza_hconcat.py`, e em um único programa por `06_item_b_completo_camera.py`.
- **Itens A e B integrados.** `07_pipeline_integrado_a_b.py` junta o stream com metadados do Item A e a captura por tecla, o salvamento e o painel do Item B, acrescentando a média móvel do FPS.

Os Exemplos 00, 01 e 02 são preparação: validam o ambiente, apresentam a imagem como array e oferecem uma fonte de vídeo sem câmera.

## 7. Armadilhas e dicas

- **Câmera não abre.** `cv2.VideoCapture(0)` pode falhar por permissão do sistema ou porque o índice é outro. Troque `indice_camera` para 1 ou 2, como o professor sugere nos comentários. Feche outros programas que estejam usando a webcam.
- **Sem webcam.** Rode `02_gerar_video_teste.py` e, como exercício, troque `cv2.VideoCapture(0)` por `cv2.VideoCapture("video_teste.mp4")` nos Exemplos 03, 06 ou 07. Com arquivo, `cap.read()` devolve `ret=False` quando o vídeo acaba, o que encerra o loop sozinho.
- **`imread` devolve `None` em silêncio.** Não lança exceção quando o arquivo não existe. Sempre teste `is None`, como os scripts fazem. O Exemplo 01 exige que o 00 tenha rodado antes, e o 05 exige o 04.
- **`imwrite` pode devolver `False`.** Extensão desconhecida ou pasta sem permissão de escrita fazem a gravação falhar sem exceção. Os scripts testam o retorno.
- **Ordem BGR.** `(255, 0, 0)` é azul, não vermelho. Isso vale para desenho, para `imagem[0, 0]` e para qualquer cálculo por canal.
- **`hconcat` exige mesmo número de canais.** Converta a cinza com `COLOR_GRAY2BGR` só para exibição. A matriz cinza real deve continuar 2D, é ela que responde ao enunciado.
- **`waitKey` é obrigatório para a janela aparecer.** Sem ele, `imshow` não desenha nada. Use `waitKey(1)` em streams e `waitKey(30)` em imagens estáticas. O `& 0xFF` mantém a comparação com `ord("q")` correta em todos os sistemas.
- **Esc não fecha o Exemplo 03.** Só Q ou q. Nos demais scripts, Esc (código 27) também encerra.
- **Resolução pedida não é garantida.** `cap.set(CAP_PROP_FRAME_WIDTH, 640)` é um pedido ao driver. Leia a resolução real de `frame.shape[:2]`.
- **FPS instantâneo é ruidoso.** O valor de `03_camera_metadados_fps.py` oscila bastante. O Exemplo 07 mostra a solução com `deque(maxlen=30)`. Vale variar `tamanho_janela_fps` para ver o efeito.
- **Frames iniciais escuros.** Se a captura única sair escura, aumente `quantidade_aquecimento` no Exemplo 04.
- **Codec `mp4v`.** Se `writer.isOpened()` for `False`, o codec não está disponível na sua build do OpenCV. Alternativas comuns são `"XVID"` com extensão `.avi` ou `"avc1"`.
- **Resolução do VideoWriter deve bater com a do frame.** Se `writer.write` receber um frame com shape diferente do declarado, o vídeo sai vazio ou corrompido.
- **Defeitos mantidos por fidelidade ao professor.** `05_recarregar_cinza_hconcat.py` importa `argparse` sem usar, e o docstring cita `--imagem`, mas o argumento não existe: o caminho é fixo em `frame_capturado.png`. O Exemplo 05 também não tem a linha "Exemplo 05" no docstring.
- **Correção de transcrição já feita.** O Exemplo 04 tinha um tab antes do `"""` que abre o docstring, o que dava erro de indentação. Foi removido na conversão.
- **Parâmetros que valem a pena variar.** Índice da câmera, resolução pedida com `cap.set`, tamanho da janela de FPS, quantidade de aquecimento, duração e FPS do vídeo sintético, e a amplitude `90` e frequência `0.08` da senoide no Exemplo 02.

## 8. Glossário

| Termo | Significado |
|---|---|
| Antialiasing | Suavização das bordas de texto ou linhas para evitar serrilhado. No OpenCV, ativado com `cv2.LINE_AA`. |
| Array NumPy (`ndarray`) | Estrutura de dados multidimensional do NumPy. É o formato de toda imagem no OpenCV. |
| BGR | Ordem dos canais de cor usada pelo OpenCV: azul, verde, vermelho. Inversa do RGB. |
| Canal | Cada uma das matrizes de cor que compõem uma imagem. Colorida tem 3, cinza tem 1 (sem eixo de canal). |
| Codec | Algoritmo de compressão de vídeo. Identificado no OpenCV pelo código fourcc. |
| Container | Formato de arquivo que embala os frames comprimidos (por exemplo MP4). |
| `deque` | Fila de duas pontas do módulo `collections`. Com `maxlen`, descarta o item mais antigo automaticamente. |
| `dtype` | Tipo de dado dos elementos do array. `uint8` significa inteiro sem sinal de 0 a 255. |
| Escala de cinza | Imagem com um único canal de intensidade, sem cor. Shape `(H, W)`. |
| Esc | Tecla Escape, código 27 em `waitKey`. |
| Fourcc | Código de quatro caracteres que identifica o codec de vídeo, por exemplo `mp4v`. |
| FPS | Frames por segundo. Nominal é o declarado pela câmera ou pelo arquivo; estimado é o medido pelo pipeline. |
| FPS instantâneo | Inverso do tempo gasto entre dois frames consecutivos. |
| Frame | Um quadro individual de um vídeo ou de um stream de câmera. |
| `getTickCount` / `getTickFrequency` | Contador de alta precisão do OpenCV e sua frequência em ticks por segundo. Usados para medir tempo. |
| `hconcat` | Concatenação horizontal de imagens com mesma altura, dtype e número de canais. |
| `imread` / `imwrite` | Funções do OpenCV para ler e gravar imagens em disco. |
| `imshow` | Exibe um array em uma janela. Só desenha quando `waitKey` é chamado. |
| In place | Operação que modifica o array original em vez de devolver uma cópia. As funções de desenho são assim. |
| Item A / Item B | Partes do exercício prático da aula: stream com metadados e FPS, e captura com comparação colorida e cinza. |
| Média móvel | Média das últimas N medidas. Usada para suavizar o FPS exibido. |
| Metadados | Informações sobre a imagem ou o stream: resolução, número do frame, FPS. |
| `pathlib.Path` | Classe da biblioteca padrão para manipular caminhos de arquivo de forma portátil. |
| Pixel | Menor elemento da imagem. Em BGR, um vetor de 3 valores; em cinza, um único valor. |
| `release` | Método que libera a câmera (`VideoCapture`) ou fecha o arquivo (`VideoWriter`). |
| Resolução | Largura por altura em pixels, por exemplo 640x480. |
| `ret` | Primeiro valor devolvido por `cap.read()`. `True` se o frame foi lido com sucesso. |
| `shape` | Tupla com as dimensões do array: `(altura, largura, canais)` ou `(altura, largura)`. |
| Snapshot | Frame único salvo em disco a partir do stream. |
| Stream | Fluxo contínuo de frames vindo da câmera. |
| `try/finally` | Estrutura do Python que garante a execução do bloco `finally` mesmo em caso de erro. Usada para liberar recursos. |
| `uint8` | Inteiro sem sinal de 8 bits, faixa 0 a 255. Dtype padrão de imagens. |
| `VideoCapture` | Classe do OpenCV que abre câmera ou arquivo de vídeo e devolve frames. |
| `VideoWriter` | Classe do OpenCV que grava frames em um arquivo de vídeo. |
| `waitKey` | Espera por uma tecla até N milissegundos e processa os eventos das janelas. |
| Webcam | Câmera do computador, acessada pelo índice 0 por padrão. |
