# Aula 09 – Convolução, CNN básica e HOG + SVM

> Tutorial gerado a partir dos scripts do Professor Vitor Amadeu (PVA) em `exemplos_em_python/`.

## 1. Visão geral

Esta aula ficou incompleta na primeira organização do repositório porque só o primeiro dos três zips do professor tinha sido baixado. Com os dois zips restantes (`02_cnn_basica` e `03_hog_svm_atividade`) a aula ficou inteira, e o nome da pasta, `aula_09_HOG_SVM`, passa a fazer sentido: a atividade avaliada da aula é justamente detecção de pedestres com HOG (Histogram of Oriented Gradients) e um classificador SVM (Support Vector Machine). Os três zips são subpastas de uma mesma raiz no material do professor, e o layout foi preservado aqui porque os scripts da terceira parte localizam o modelo, os dados e as saídas a partir da pasta pai.

O fio condutor vai do mais elementar ao mais aplicado, em três blocos. O bloco 1 (`01_fundamentos_convolucao/`, scripts 01 a 14) constrói do zero a intuição de uma camada convolucional: imagem como matriz, convolução à mão, kernels de borda, Sobel, blur, sharpen, stride, padding, múltiplos filtros, ReLU e max pooling. O bloco 2 (`02_cnn_basica/`, scripts 15 a 20) monta essas mesmas peças em redes reais com Keras: uma CNN (Convolutional Neural Network) no MNIST, visualização dos kernels aprendidos e dos feature maps, uma CNN no CIFAR-10, transfer learning com MobileNetV2 e a diferença conceitual entre classificação e detecção. O bloco 3 (`03_hog_svm_atividade/`) é a atividade: no item A, o detector de pedestres pronto do OpenCV (HOG + SVM linear) é medido em dois cenários de velocidade; no item B, o aluno monta o próprio detector, extraindo HOG de recortes 64x128, treinando um SVC e aplicando janela deslizante com pirâmide de escalas e supressão de não máximos.

## 2. Ambiente e pré-requisitos

Bibliotecas de terceiros por bloco:

| Bloco | Bibliotecas | Instalação |
|---|---|---|
| 1 Convolução | `numpy`, `opencv-python` | `pip install numpy opencv-python` |
| 2 CNN básica | `tensorflow`, `matplotlib` (mais `numpy` e `opencv-python` no script 20) | `pip install tensorflow matplotlib` |
| 3 HOG + SVM | `opencv-python`, `numpy`, `scikit-learn`, `joblib` | `pip install scikit-learn joblib` |

`joblib` vem junto com o scikit-learn, mas os scripts fazem `import joblib` direto, então o pacote precisa estar disponível com esse nome.

Layout da pasta `exemplos_em_python/`, espelhando o do professor:

```
aula_09_HOG_SVM/exemplos_em_python/
├── 01_fundamentos_convolucao/    scripts 01 a 14, gravam PNG no diretório de trabalho
├── 02_cnn_basica/                scripts 15 a 20; o 15 salva modelo_mnist.keras, lido por 16 e 17
├── 03_hog_svm_atividade/         item_a_hog_pedestres.py, item_b_preparar_dataset.py,
│                                 item_b_treinar_svm.py, item_b_janela_deslizante.py,
│                                 gerar_dataset_sintetico_demo.py, modelo_hog_svm.joblib
├── dados/positivas/              vazias, a serem preenchidas (veja abaixo)
├── dados/negativas/
└── saidas/                       vazia, recebe deteccoes_janela_deslizante.jpg
```

De onde rodar. Os blocos 1 e 2 usam caminhos relativos ao diretório de trabalho, então é preciso fazer `cd` para dentro da subpasta antes de executar. O bloco 3 usa `Path(__file__).resolve().parents[1]`, que é sempre `exemplos_em_python/` independentemente do diretório atual, e a partir daí monta `dados/`, `saidas/` e `03_hog_svm_atividade/modelo_hog_svm.joblib`. Por isso as pastas `dados/positivas/`, `dados/negativas/` e `saidas/` foram criadas vazias na reconstrução: os scripts gravam nelas sem chamar `mkdir`.

Material que não veio nos zips:

- Item A: um vídeo com pedestres, passado por `--video`. O argumento é obrigatório e o zip não traz vídeo nenhum.
- Item B: um dataset real com no mínimo 100 imagens positivas (pessoas) e 100 negativas (fundo), a ser colocado em `dados/positivas/` e `dados/negativas/`. O script `gerar_dataset_sintetico_demo.py` cria 120 de cada, mas seu docstring avisa que é "somente demonstração do pipeline" e "não substitui um dataset real para a entrega".

Veio no zip 03 o arquivo `modelo_hog_svm.joblib` (2,3 MB), um SVC já treinado pelo professor, o que permite rodar a janela deslizante sem treinar antes.

Webcam: nenhum script usa. O item A abre um arquivo de vídeo. Datasets MNIST, CIFAR-10 e os pesos ImageNet do MobileNetV2 são baixados pelo Keras na primeira execução, o que exige internet.

Python: o material do professor nas outras aulas foi rodado com 3.12 e 3.13. O `.joblib` foi serializado com uma versão específica do scikit-learn e pode avisar ou falhar ao carregar em versão diferente.

## 3. Conceitos-chave

### Bloco 1: fundamentos de convolução

### 3.1 Imagem como matriz NumPy

Para o computador, uma imagem em escala de cinza é uma matriz bidimensional de números inteiros sem sinal de 8 bits (`uint8`), de 0 (preto) a 255 (branco). A forma (`shape`) é `(altura, largura)`, e o acesso a um pixel segue a convenção de matrizes: primeiro a linha (coordenada y), depois a coluna (coordenada x). Isso é o oposto da notação `(x, y)` usada nas funções de desenho do OpenCV, e essa inversão é fonte frequente de erros.

O script cria uma matriz 4x4 à mão e mostra três coisas: a matriz inteira, o formato `(4, 4)` e o valor do pixel na linha 2, coluna 3, que é 160.

```python
# 01_fundamentos_convolucao/01_imagem_como_matriz.py
import numpy as np
imagem = np.array([[0,40,80,120],[20,60,100,140],[40,80,120,160],[60,100,140,200]], dtype=np.uint8)
print(imagem)
print('Formato:', imagem.shape)
print('Pixel [2,3]:', imagem[2,3])
```

Onde aparece: `01_imagem_como_matriz.py`. A ideia se repete em todos os outros scripts do bloco, que criam imagens com `np.zeros`, `np.array` ou `np.arange`.

### 3.2 Canais de cor e a ordem BGR do OpenCV

Uma imagem colorida é um tensor de três dimensões `(altura, largura, 3)`, com um plano por canal. O OpenCV armazena os canais na ordem BGR (azul, verde, vermelho), não RGB. Por isso, quando o script atribui `(255, 0, 0)` ao terço esquerdo da imagem, esse terço fica azul, e `(0, 0, 255)` no terço direito fica vermelho.

`cv2.split(img)` separa o tensor em três matrizes bidimensionais, uma por canal. Cada canal tem shape `(200, 300)` enquanto a imagem tem `(200, 300, 3)`. O script salva a imagem original e cada canal como PNG; nos PNGs de canal, a faixa correspondente à cor daquele canal aparece branca (valor 255) e as outras duas pretas.

```python
# 01_fundamentos_convolucao/02_canais_rgb.py
img = np.zeros((200,300,3), dtype=np.uint8)
img[:,:100] = (255,0,0); img[:,100:200] = (0,255,0); img[:,200:] = (0,0,255)
b, g, r = cv2.split(img)
print('Imagem:', img.shape, 'Canal:', b.shape)
for n, a in [('original',img),('b',b),('g',g),('r',r)]:
    cv2.imwrite(f'02_{n}.png', a)
```

Onde aparece: `02_canais_rgb.py`.

### 3.3 Convolução em um único ponto: produto elemento a elemento e soma

A convolução (no sentido usado em redes neurais, que na verdade é correlação cruzada) é, em cada posição, uma única operação: multiplica-se cada pixel da janela pelo valor correspondente do kernel e soma-se tudo. O resultado é um único número. O kernel (também chamado de filtro ou máscara) é uma matriz pequena, tipicamente 3x3, cujos pesos definem o que o filtro "procura".

O script usa uma imagem 3x3 com uma transição de 10 para 100 na terceira coluna e um kernel que tem `-1` à esquerda, `0` no centro e `+1` à direita. O produto elemento a elemento dá `-10` na coluna esquerda, `0` no meio e `100` na direita, e a soma dá 270. Um valor alto e positivo indica que há uma transição de escuro para claro da esquerda para a direita, exatamente o que esse kernel detecta.

```python
# 01_fundamentos_convolucao/03_convolucao_manual.py
img = np.array([[10,10,100],[10,10,100],[10,10,100]], dtype=np.float32)
k = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=np.float32)
print('Produto:\n', img*k)
print('Soma:', np.sum(img*k))
```

Repare no uso de `float32`: com `uint8` a multiplicação por `-1` estouraria. Este é o motivo de todos os kernels do bloco serem declarados em `np.float32`.

Onde aparece: `03_convolucao_manual.py`.

### 3.4 Kernel deslizante e o tamanho do feature map

A convolução completa repete a operação do conceito anterior para todas as posições possíveis da janela, deslizando o kernel pela imagem. O script implementa isso com dois laços `for`: para cada `(y, x)` de saída, recorta a fatia `img[y:y+3, x:x+3]`, multiplica pelo kernel e soma. O mapa de saída é chamado de feature map (mapa de características).

Sem padding, uma imagem 5x5 convoluída com kernel 3x3 gera saída 3x3, porque a janela cabe em apenas 3 posições em cada eixo. A fórmula geral é `saida = entrada - kernel + 1`. Como a imagem de teste é uma rampa de 1 a 25 e o kernel subtrai a coluna direita da esquerda, todas as nove saídas valem `-6` (três linhas, cada uma contribuindo `-2`).

```python
# 01_fundamentos_convolucao/04_kernel_deslizante.py
img = np.arange(1,26, dtype=np.float32).reshape(5,5)
k = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], np.float32)
out = np.zeros((3,3), np.float32)
for y in range(3):
    for x in range(3):
        out[y,x] = np.sum(img[y:y+3, x:x+3] * k)
```

Onde aparece: `04_kernel_deslizante.py`, e a mesma técnica volta em `10_stride.py` e `14_max_pooling.py`. A janela deslizante do item B (`item_b_janela_deslizante.py`) é a mesma ideia aplicada a recortes 64x128 em vez de janelas 3x3.

### 3.5 Convolução com `cv2.filter2D` e detecção de bordas verticais

`cv2.filter2D(src, ddepth, kernel)` faz a convolução inteira em uma chamada, de forma otimizada. Os parâmetros que a aula usa:

- `src`: a imagem de entrada, aqui `uint8`.
- `ddepth`: a profundidade (tipo) da saída. `cv2.CV_32F` pede saída em ponto flutuante, para que valores negativos e maiores que 255 sejam preservados. `-1` pede a mesma profundidade da entrada.
- `kernel`: a matriz de pesos em `float32`.

Como a saída em `CV_32F` não pode ser salva diretamente como PNG de 8 bits, o script aplica `cv2.convertScaleAbs`, que toma o valor absoluto, arredonda e satura em `[0, 255]`. Isso converte respostas negativas em positivas (bordas de claro para escuro também aparecem) e limita o máximo em 255.

A imagem de teste é preta à esquerda e branca a partir da coluna 150. O kernel `[[-1,0,1]] x 3` responde forte somente na transição, então o PNG resultante é preto com uma linha branca vertical na coluna 150.

```python
# 01_fundamentos_convolucao/05_borda_vertical.py
img = np.zeros((200,300), np.uint8); img[:,150:] = 255
k = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
r = cv2.filter2D(img, cv2.CV_32F, k)
cv2.imwrite('05_borda_vertical.png', cv2.convertScaleAbs(r))
```

Onde aparece: `05_borda_vertical.py`, `06_borda_horizontal.py`, `08_blur.py`, `09_sharpen.py`, `12_multiplos_filtros.py`.

### 3.6 Bordas horizontais: transpondo o kernel

O kernel de borda vertical compara colunas vizinhas. Para detectar bordas horizontais basta comparar linhas vizinhas, ou seja, usar o kernel transposto: `-1` na linha de cima, `0` no meio, `+1` embaixo. A imagem de teste é preta em cima e branca a partir da linha 100, e o resultado é uma linha branca horizontal na linha 100.

```python
# 01_fundamentos_convolucao/06_borda_horizontal.py
img = np.zeros((200,300), np.uint8); img[100:,:] = 255
k = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32)
r = cv2.filter2D(img, cv2.CV_32F, k)
cv2.imwrite('06_borda_horizontal.png', cv2.convertScaleAbs(r))
```

A lição dos scripts 05 e 06 juntos: a orientação dos pesos define a orientação da borda detectada. Uma CNN aprende esses pesos sozinha durante o treino, como o script 16 vai mostrar; aqui eles são escritos à mão.

Onde aparece: `06_borda_horizontal.py`.

### 3.7 Operador de Sobel e gradientes da imagem

O operador de Sobel é um par de kernels 3x3 que estima o gradiente da imagem, isto é, a taxa de variação de intensidade em x (`Gx`) e em y (`Gy`). É parecido com os kernels dos scripts 05 e 06, mas com peso dobrado na linha ou coluna central, o que suaviza o ruído. O gradiente é a base de muitos descritores clássicos, e o script imprime explicitamente que ele é a "base conceitual importante para HOG", que constrói histogramas das orientações desses gradientes. É a ponte direta entre o bloco 1 e o bloco 3.

`cv2.Sobel(src, ddepth, dx, dy, ksize)`:

- `ddepth=cv2.CV_64F`: saída em `float64`, para guardar sinal.
- `dx=1, dy=0`: derivada de primeira ordem em x (bordas verticais).
- `dx=0, dy=1`: derivada em y (bordas horizontais).
- `ksize=3`: tamanho do kernel.

A imagem de teste é um retângulo branco sobre fundo preto. `07_sobel_x.png` mostra as duas arestas verticais do retângulo, e `07_sobel_y.png` mostra as duas arestas horizontais.

```python
# 01_fundamentos_convolucao/07_sobel.py
img = np.zeros((240,320), np.uint8)
cv2.rectangle(img, (80,60), (240,180), 255, -1)
gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
cv2.imwrite('07_sobel_x.png', cv2.convertScaleAbs(gx))
cv2.imwrite('07_sobel_y.png', cv2.convertScaleAbs(gy))
```

Onde aparece: `07_sobel.py`.

### 3.8 Funções de desenho do OpenCV para criar imagens sintéticas

Os scripts 07, 08, 09, 12 e 20, além dos geradores do bloco 3, criam as imagens de teste com primitivas de desenho em vez de fatiamento NumPy. Em todas elas as coordenadas são `(x, y)`, a cor é um escalar quando a imagem é de um canal só, e espessura `-1` significa preenchido.

- `cv2.rectangle(img, (x1,y1), (x2,y2), cor, espessura)`: retângulo pelos cantos opostos.
- `cv2.circle(img, (cx,cy), raio, cor, espessura)`: círculo.
- `cv2.line(img, (x1,y1), (x2,y2), cor, espessura)`: segmento de reta.
- `cv2.putText(img, texto, (x,y), fonte, escala, cor, espessura)`: texto. `(x,y)` é o canto inferior esquerdo da primeira letra. A fonte usada é `cv2.FONT_HERSHEY_SIMPLEX`.

```python
# 01_fundamentos_convolucao/09_sharpen.py
img = np.zeros((240,320), np.uint8)
cv2.putText(img, 'CNN', (65,145), cv2.FONT_HERSHEY_SIMPLEX, 2.6, 200, 7)
```

Onde aparece: `07_sobel.py` (retângulo), `08_blur.py` (círculo), `09_sharpen.py` (texto), `12_multiplos_filtros.py` (retângulo), `20_deteccao_objetos_conceito.py` (retângulos coloridos), `gerar_dataset_sintetico_demo.py` (círculo, retângulo, linhas), `item_a_hog_pedestres.py` e `item_b_janela_deslizante.py` (retângulos e texto sobre as detecções).

### 3.9 Filtro de média (blur)

Um kernel com todos os pesos iguais e somando 1 substitui cada pixel pela média da vizinhança, o que borra a imagem e reduz ruído. O script usa um kernel 5x5 de uns dividido por 25. Note que aqui o `ddepth` é `-1`: como todos os pesos são positivos e somam 1, a saída fica dentro de `[0, 255]` e pode ser gravada diretamente sem `convertScaleAbs`. O círculo branco sai com as bordas suavizadas, em degradê.

```python
# 01_fundamentos_convolucao/08_blur.py
img = np.zeros((240,320), np.uint8)
cv2.circle(img, (160,120), 70, 255, -1)
k = np.ones((5,5), np.float32) / 25
cv2.imwrite('08_blur.png', cv2.filter2D(img, -1, k))
```

Onde aparece: `08_blur.py`.

### 3.10 Filtro de realce (sharpen)

O kernel de sharpen tem `5` no centro e `-1` nos quatro vizinhos diretos, somando 1. Ele mantém o brilho médio, mas amplifica a diferença entre cada pixel e seus vizinhos, o que realça bordas e detalhes. Aplicado ao texto "CNN" desenhado em cinza 200, o resultado tem letras com contorno mais marcado. Com `ddepth=-1` os valores negativos ou acima de 255 são saturados pelo próprio `filter2D`.

```python
# 01_fundamentos_convolucao/09_sharpen.py
k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], np.float32)
cv2.imwrite('09_sharpen.png', cv2.filter2D(img, -1, k))
```

Onde aparece: `09_sharpen.py`.

### 3.11 Stride (passo do deslizamento)

O stride é o número de pixels que o kernel avança a cada passo. Com stride 1 a janela visita todas as posições; com stride 2 ela pula uma posição, e a saída fica cerca de metade do tamanho em cada eixo. A fórmula usada no script é `tamanho_saida = (entrada - kernel) // stride + 1`. Para entrada 7x7 e kernel 3x3: stride 1 gera 5x5, stride 2 gera 3x3.

O script implementa a convolução com stride nos mesmos laços do conceito 3.4, mas indexando por `y*s` e `x*s`. Como o kernel é de uns (soma da janela) e a imagem é a rampa 1 a 49, cada saída vale 9 vezes o pixel central da janela. Assim a saída com stride 1 começa em 81 (9 x 9) e a com stride 2 mostra `[[81, 99, 117], [207, 225, 243], [333, 351, 369]]`.

```python
# 01_fundamentos_convolucao/10_stride.py
img = np.arange(1,50, dtype=np.float32).reshape(7,7); k = np.ones((3,3), np.float32)
def conv(s):
    oh = (7-3)//s + 1; out = np.zeros((oh,oh))
    for y in range(oh):
        for x in range(oh):
            out[y,x] = np.sum(img[y*s:y*s+3, x*s:x*s+3] * k)
    return out
for s in (1,2): print('Stride', s, '\n', conv(s))
```

Onde aparece: `10_stride.py`. O mesmo conceito reaparece como `winStride` no item A e como `--passo` no item B.

### 3.12 Padding: VALID e SAME

Padding é a borda de zeros adicionada em volta da imagem antes da convolução. Sem padding (modo `VALID`, o mesmo dos scripts 04 e 10), a saída encolhe: 5x5 com kernel 3x3 vira 3x3. Com padding de 1 pixel em cada lado (modo `SAME`), a saída mantém o tamanho da entrada, 5x5. Os nomes `VALID` e `SAME` são os usados por TensorFlow e Keras nas camadas `Conv2D`, e o script 18 usa `padding='same'` de fato.

O script não faz nenhuma convolução: apenas imprime os tamanhos esperados nos dois modos, como um lembrete conceitual. Observe que `cv2.filter2D` já usa, por padrão, um padding por reflexão de borda (`BORDER_REFLECT_101`), por isso os PNGs dos scripts 05 a 09 têm o mesmo tamanho da entrada.

```python
# 01_fundamentos_convolucao/11_padding.py
img = np.arange(1,26).reshape(5,5)
print('Entrada:', img.shape)
print('VALID com kernel 3x3 ->', (3,3))
print('SAME ->', img.shape)
```

Onde aparece: `11_padding.py`, `18_cnn_cifar10.py` (`padding='same'`).

### 3.13 Múltiplos filtros e múltiplos feature maps

Uma camada convolucional não tem um único kernel, mas vários. Cada um produz o próprio feature map, e a pilha desses mapas é a saída da camada (é o "número de filtros" de uma `Conv2D`). O script aplica três kernels ao mesmo retângulo branco:

- `vertical`: o detector de bordas verticais do script 05, escrito como `[[-1,0,1]]*3`.
- `horizontal`: o detector de bordas horizontais do script 06.
- `laplaciano`: `[[0,1,0],[1,-4,1],[0,1,0]]`, uma derivada de segunda ordem que responde a bordas em todas as direções ao mesmo tempo, sem distinguir orientação.

O resultado são três PNGs distintos: só as arestas verticais, só as horizontais e o contorno completo do retângulo.

```python
# 01_fundamentos_convolucao/12_multiplos_filtros.py
ks = {'vertical':   np.array([[-1,0,1]]*3, np.float32),
      'horizontal': np.array([[-1,-1,-1],[0,0,0],[1,1,1]], np.float32),
      'laplaciano': np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float32)}
for n, k in ks.items():
    cv2.imwrite(f'12_{n}.png', cv2.convertScaleAbs(cv2.filter2D(img, cv2.CV_32F, k)))
print('Cada filtro gera um feature map diferente.')
```

Onde aparece: `12_multiplos_filtros.py`. A versão aprendida disso é a `Conv2D(32, 3)` do script 15, cujos 32 kernels o script 16 desenha.

### 3.14 Função de ativação ReLU

Depois da convolução, uma CNN aplica uma função de ativação não linear a cada valor do feature map. A mais comum é a ReLU (Rectified Linear Unit): `f(x) = max(0, x)`. Ela zera os valores negativos e mantém os positivos. Sem uma não linearidade, empilhar convoluções equivaleria a uma única convolução maior, e a rede não conseguiria aprender funções complexas.

O script mostra a operação em um vetor: `[-5, -2, 0, 3, 9]` vira `[0, 0, 0, 3, 9]`. A implementação em NumPy é `np.maximum(0, x)`, que funciona igualmente para matrizes inteiras.

```python
# 01_fundamentos_convolucao/13_relu.py
x = np.array([-5,-2,0,3,9], dtype=np.float32)
print('Entrada:', x)
print('ReLU:', np.maximum(0, x))
```

Onde aparece: `13_relu.py`, e como `activation='relu'` em todas as `Conv2D` e `Dense` ocultas dos scripts 15 e 18.

### 3.15 Max pooling

Pooling reduz a resolução do feature map, tornando a rede mais barata e mais tolerante a pequenos deslocamentos. No max pooling com janela 2x2 e stride 2, cada bloco 2x2 da entrada é substituído pelo seu maior valor. Uma matriz 4x4 vira 2x2.

O script implementa com dois laços, fatiando blocos `x[y*2:y*2+2, z*2:z*2+2]` e chamando `.max()`. Para a entrada dada, o resultado é `[[8, 4], [5, 9]]`.

```python
# 01_fundamentos_convolucao/14_max_pooling.py
x = np.array([[1,3,2,0],[2,8,1,4],[5,2,9,1],[0,1,3,7]], dtype=np.float32)
out = np.zeros((2,2))
for y in range(2):
    for z in range(2):
        out[y,z] = x[y*2:y*2+2, z*2:z*2+2].max()
print('MaxPooling:\n', out)
```

Onde aparece: `14_max_pooling.py`, e como `MaxPooling2D(2)` nos scripts 15 e 18.

### 3.16 Juntando tudo: o bloco convolucional de uma CNN

Os scripts 04, 10, 12, 13 e 14, lidos em sequência, formam um bloco convolucional completo: vários kernels deslizam pela entrada com certo stride e padding (`Conv2D`), cada saída passa por ReLU, e o resultado é reduzido por max pooling (`MaxPooling2D`). É exatamente a estrutura que o script 15 monta em Keras. A diferença é que no bloco 1 os pesos dos kernels são escolhidos à mão para fins didáticos, enquanto na CNN eles são aprendidos por retropropagação.

Onde aparece: síntese dos scripts 04, 10, 11, 12, 13 e 14; materializado em `15_cnn_mnist.py`.

### Bloco 2: CNN básica com Keras

### 3.17 CNN no MNIST: da matriz ao modelo treinado

O script 15 é o bloco convolucional do conceito 3.16 escrito em Keras e treinado de verdade. O MNIST é carregado com `tf.keras.datasets.mnist.load_data()`, que devolve imagens `uint8` 28x28 e rótulos inteiros de 0 a 9. Duas transformações preparam o tensor: `xtr[..., None]` acrescenta o eixo de canal (fica `(60000, 28, 28, 1)`), porque `Conv2D` exige entrada com canais, e a divisão por 255 normaliza para `[0, 1]`.

A arquitetura `Sequential` é: `Input((28,28,1))`, `Conv2D(32, 3, relu)`, `MaxPooling2D(2)`, `Conv2D(64, 3, relu)`, `MaxPooling2D(2)`, `Flatten`, `Dense(64, relu)`, `Dense(10, softmax)`. Cada `Conv2D` sem padding encolhe o mapa em 2 pixels (28 para 26, depois 13 para 11), cada pooling divide por 2, e o `Flatten` transforma o tensor `5x5x64` em um vetor de 1600 valores para as camadas densas. A saída `softmax` com 10 unidades dá a probabilidade de cada dígito.

`compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])` escolhe o otimizador Adam e a perda de entropia cruzada na versão "sparse", que aceita rótulos inteiros diretamente, sem one-hot. `fit` treina por 2 épocas com lotes de 128 e separa 10% do treino como validação. `evaluate` devolve `[perda, acurácia]` no conjunto de teste, e `save('modelo_mnist.keras')` grava o modelo inteiro (arquitetura mais pesos) no formato nativo do Keras 3, que os scripts 16 e 17 vão reabrir.

```python
# 02_cnn_basica/15_cnn_mnist.py
(xtr,ytr),(xte,yte) = tf.keras.datasets.mnist.load_data()
xtr = xtr[...,None]/255.0; xte = xte[...,None]/255.0
m = models.Sequential([layers.Input((28,28,1)),
    layers.Conv2D(32,3,activation='relu'), layers.MaxPooling2D(2),
    layers.Conv2D(64,3,activation='relu'), layers.MaxPooling2D(2),
    layers.Flatten(), layers.Dense(64,activation='relu'), layers.Dense(10,activation='softmax')])
m.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy'])
m.fit(xtr,ytr,epochs=2,batch_size=128,validation_split=.1)
print(m.evaluate(xte,yte,verbose=0)); m.save('modelo_mnist.keras')
```

Onde aparece: `15_cnn_mnist.py`. A aula 11 deste repositório aprofunda o mesmo dataset.

### 3.18 Visualizar os kernels aprendidos pela primeira camada

O bloco 1 escreveu kernels à mão. Aqui o script abre o modelo salvo e lê os pesos da primeira camada com `m.layers[0].get_weights()[0]`. Como o modelo foi criado com `Input` explícito, `layers[0]` é a primeira `Conv2D`, e `get_weights()` devolve `[kernels, bias]`. O tensor de kernels tem shape `(3, 3, 1, 32)`: altura, largura, canais de entrada, número de filtros. `w[:, :, 0, i]` extrai o i-ésimo kernel 3x3.

O script desenha os 16 primeiros dos 32 filtros em uma grade 4x4 com `matplotlib`, em escala de cinza, salva em `16_kernels.png` com 150 dpi e abre a janela com `plt.show()`. Muitos deles se parecem com os detectores de borda dos scripts 05 e 06: a rede reinventou sozinha aqueles kernels.

```python
# 02_cnn_basica/16_visualizar_kernels.py
m = tf.keras.models.load_model('modelo_mnist.keras')
w = m.layers[0].get_weights()[0]; print('Shape:', w.shape)   # (3, 3, 1, 32)
fig, axs = plt.subplots(4,4,figsize=(8,8))
for i, ax in enumerate(axs.ravel()):
    ax.axis('off'); ax.imshow(w[:,:,0,i], cmap='gray'); ax.set_title(f'Filtro {i}')
plt.tight_layout(); plt.savefig('16_kernels.png', dpi=150); plt.show()
```

Onde aparece: `16_visualizar_kernels.py`.

### 3.19 Feature maps com um sub-modelo Keras

Para ver o que cada filtro produz em uma imagem real, o script 17 monta um segundo modelo que compartilha as camadas do primeiro mas devolve saídas intermediárias. A lista `outs` reúne o `.output` de todas as camadas que são `Conv2D`, e `tf.keras.Model(m.input, outs)` cria o sub-modelo com a mesma entrada e essas duas saídas. Isso é possível porque no Keras cada camada guarda o tensor simbólico que produz.

A entrada é o primeiro dígito do conjunto de teste, preparado do mesmo jeito do treino: `xt[0][None, ..., None] / 255.0` acrescenta a dimensão de lote na frente e a de canal atrás, ficando `(1, 28, 28, 1)`. `vm.predict(e)` devolve uma lista com dois tensores; o `[0]` pega o da primeira `Conv2D` (`(1, 26, 26, 32)`) e o segundo `[0]` tira o eixo de lote (`(26, 26, 32)`). O script desenha os 16 primeiros mapas em grade 4x4 e salva `17_feature_maps.png`. Cada mapa é o resultado de aplicar um dos kernels do script 16 ao dígito: a versão aprendida dos PNGs do script 12.

```python
# 02_cnn_basica/17_visualizar_feature_maps.py
m = tf.keras.models.load_model('modelo_mnist.keras')
(_, _), (xt, _) = tf.keras.datasets.mnist.load_data()
e = xt[0][None,...,None]/255.0                      # (1, 28, 28, 1)
outs = [l.output for l in m.layers if isinstance(l, tf.keras.layers.Conv2D)]
vm = tf.keras.Model(m.input, outs)
fm = vm.predict(e, verbose=0)[0][0]                 # (26, 26, 32)
for i, ax in enumerate(axs.ravel()):
    ax.axis('off'); ax.imshow(fm[:,:,i], cmap='gray'); ax.set_title(f'Mapa {i}')
```

Onde aparece: `17_visualizar_feature_maps.py`.

### 3.20 CNN no CIFAR-10: padding same e GlobalAveragePooling

O CIFAR-10 tem 60 mil imagens coloridas 32x32 em 10 classes (avião, carro, pássaro, gato e outras). Como já vêm com três canais, basta dividir por 255; o `load_data` devolve os rótulos com shape `(n, 1)`, o que a perda sparse aceita. Duas diferenças de arquitetura em relação ao script 15:

- `padding='same'` em todas as `Conv2D`: o mapa não encolhe a cada convolução (32 continua 32), só os poolings reduzem (32, 16, 8). É o modo SAME do conceito 3.12.
- `GlobalAveragePooling2D()` no lugar de `Flatten`: tira a média espacial de cada um dos 128 mapas finais, gerando um vetor de 128 valores independentemente do tamanho da imagem. Tem muito menos parâmetros que um `Flatten` seguido de `Dense` e serve de ponte direta para a `Dense(10, softmax)`.

O treino é de 3 épocas, lote 128, validação de 10%, e o script imprime `[perda, acurácia]` no teste. Não salva o modelo.

```python
# 02_cnn_basica/18_cnn_cifar10.py
(xtr,ytr),(xte,yte) = tf.keras.datasets.cifar10.load_data(); xtr = xtr/255.0; xte = xte/255.0
m = models.Sequential([layers.Input((32,32,3)),
    layers.Conv2D(32,3,padding='same',activation='relu'), layers.MaxPooling2D(2),
    layers.Conv2D(64,3,padding='same',activation='relu'), layers.MaxPooling2D(2),
    layers.Conv2D(128,3,padding='same',activation='relu'),
    layers.GlobalAveragePooling2D(), layers.Dense(10,activation='softmax')])
m.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy'])
m.fit(xtr,ytr,epochs=3,batch_size=128,validation_split=.1); print(m.evaluate(xte,yte,verbose=0))
```

Onde aparece: `18_cnn_cifar10.py`.

### 3.21 Transfer learning com MobileNetV2 congelada

Treinar uma CNN grande do zero exige muitos dados. Transfer learning reaproveita uma rede já treinada em um dataset enorme (ImageNet, 1,2 milhão de imagens) como extrator de características e troca só a cabeça de classificação. O script 19 usa a API funcional do Keras para isso:

- `tf.keras.applications.MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')` baixa a MobileNetV2 sem a camada final de 1000 classes. `base.trainable = False` congela todos os pesos.
- `preprocess_input` escala os pixels de `[0, 255]` para `[-1, 1]`, que é o que a MobileNetV2 espera. Fica dentro do grafo, então quem usa o modelo passa a imagem crua.
- `base(x, training=False)` roda a rede em modo de inferência, o que mantém as estatísticas de BatchNormalization congeladas.
- `GlobalAveragePooling2D` reduz o tensor `5x5x1280` a um vetor de 1280, e `Dense(2, softmax)` é a cabeça nova para um problema de duas classes.

O script só monta e imprime o `summary()`: não há dataset nem treino. Ele mostra a proporção entre parâmetros congelados (milhões) e treináveis (só a `Dense` final, 2562). A aula 12 deste repositório treina uma cabeça igual a essa para gênero no UTKFace, e a aula 03 faz transfer learning no Fashion-MNIST.

```python
# 02_cnn_basica/19_transfer_learning_mobilenet.py
base = tf.keras.applications.MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')
base.trainable = False
i = layers.Input((160,160,3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
o = layers.Dense(2, activation='softmax')(x)
m = Model(i, o); m.summary()
```

Onde aparece: `19_transfer_learning_mobilenet.py`.

### 3.22 Classificação versus detecção e a ideia do YOLO

O script 20 é conceitual. Seu docstring resume a diferença: classificação é `imagem -> CNN -> classe`; detecção é `imagem -> backbone convolucional -> classe + bounding box`, ou seja, além de dizer o que há na imagem, diz onde, com uma caixa delimitadora por objeto. O script desenha uma imagem clara com dois retângulos coloridos, verde e azul (em BGR `(255,0,0)` é azul), salva `20_deteccao_conceito.png` e imprime o próprio docstring.

A frase central é sobre eficiência: "YOLO compartilha cálculo convolucional na imagem inteira, em vez de classificar milhares de janelas independentes". Essa comparação prepara o item B, que faz exatamente o contrário, rodar HOG e SVM em milhares de janelas, e comenta o custo disso no código.

```python
# 02_cnn_basica/20_deteccao_objetos_conceito.py
"""Classificação: imagem -> CNN -> classe.
Detecção: imagem -> backbone convolucional -> classe + bounding box.
YOLO compartilha cálculo convolucional na imagem inteira, em vez de classificar milhares de janelas independentes."""
img = np.full((420,720,3), 245, np.uint8)
cv2.rectangle(img, (90,120), (250,330), (0,160,0), 3)
cv2.rectangle(img, (390,150), (610,310), (255,0,0), 3)
cv2.imwrite('20_deteccao_conceito.png', img); print(__doc__)
```

Onde aparece: `20_deteccao_objetos_conceito.py`, e os comentários de `item_b_janela_deslizante.py`.

### Bloco 3: HOG + SVM

### 3.23 O que é o descritor HOG

HOG (Histogram of Oriented Gradients) é um descritor clássico para detectar pessoas, proposto por Dalal e Triggs em 2005. A ideia parte do gradiente do conceito 3.7: em cada pixel calcula-se a magnitude e a orientação do gradiente; a imagem é dividida em células (aqui 8x8 pixels) e em cada célula monta-se um histograma de orientações com 9 bins, ponderado pela magnitude; as células são agrupadas em blocos (16x16, ou 2x2 células) que deslizam com passo de 8 pixels e são normalizados para resistir a variações de iluminação; a concatenação de todos os histogramas de todos os blocos é o vetor HOG.

Para a janela padrão de 64x128 pixels: cabem 7 blocos na horizontal (`(64-16)/8+1`) e 15 na vertical (`(128-16)/8+1`), total 105 blocos; cada bloco tem 4 células com 9 bins, 36 valores; o vetor final tem `105 x 36 = 3780` números. É esse vetor de 3780 dimensões que o SVM classifica. Silhuetas humanas em pé produzem padrões de orientação característicos (bordas verticais nas pernas e no tronco, arredondadas na cabeça), e é isso que o classificador aprende.

No OpenCV, `cv2.HOGDescriptor()` sem argumentos já vem com exatamente esses parâmetros. Os scripts do item B os escrevem explicitamente:

```python
# 03_hog_svm_atividade/item_b_treinar_svm.py (igual em item_b_janela_deslizante.py)
hog = cv2.HOGDescriptor(_winSize=(64,128), _blockSize=(16,16), _blockStride=(8,8),
                        _cellSize=(8,8), _nbins=9)
feat = hog.compute(cv2.resize(img, (64,128))).reshape(-1)   # 3780 valores
```

Onde aparece: `item_a_hog_pedestres.py` (parâmetros padrão implícitos), `item_b_treinar_svm.py`, `item_b_janela_deslizante.py`.

### 3.24 Detector de pedestres pronto do OpenCV (item A)

O OpenCV inclui um SVM linear já treinado sobre HOG para pessoas. `hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())` carrega esses pesos no descritor, e a partir daí `hog.detectMultiScale(frame, winStride, padding, scale)` faz tudo: percorre a imagem em várias escalas, extrai HOG em cada janela, aplica o SVM e devolve as caixas `(x, y, w, h)` mais os pesos de confiança. Os parâmetros que a aula varia:

- `winStride`: passo da janela em pixels. Menor é mais preciso e mais lento (o stride do conceito 3.11).
- `padding`: borda acrescentada em volta de cada janela antes de extrair HOG.
- `scale`: fator entre níveis da pirâmide de escalas. `1.03` gera muitos níveis (lento, detecta mais tamanhos); `1.10` gera poucos.

O item A define dois cenários, `rapido` (`winStride=(12,12)`, `scale=1.10`) e `preciso` (`winStride=(4,4)`, `scale=1.03`), roda o vídeo inteiro em cada um, desenha as caixas em verde, mostra o frame com `cv2.imshow('HOG People Detector', f)` e imprime, por frame, o número de detecções e o tempo de inferência em milissegundos medido com `time.perf_counter()`. No fim imprime uma tabela comparando frames processados, detecções por frame, ms por frame e FPS (`1000 / media_ms`). A tecla Esc (`waitKey(1) & 0xFF == 27`) interrompe o cenário atual.

```python
# 03_hog_svm_atividade/item_a_hog_pedestres.py
hog = cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cenarios = {'rapido':  dict(winStride=(12,12), padding=(8,8), scale=1.10),
            'preciso': dict(winStride=(4,4),   padding=(8,8), scale=1.03)}
t = time.perf_counter()
rects, w = hog.detectMultiScale(f, winStride=c['winStride'], padding=c['padding'], scale=c['scale'])
ms = (time.perf_counter()-t)*1000
for x, y, ww, hh in rects: cv2.rectangle(f, (x,y), (x+ww,y+hh), (0,255,0), 2)
```

Onde aparece: `item_a_hog_pedestres.py`.

### 3.25 Preparar o dataset: recortes 64x128 positivos e negativos

Para treinar o próprio detector, o item B precisa de exemplos positivos (recortes com uma pessoa inteira, em pé) e negativos (recortes de fundo sem pessoa). O script de preparação lê `dados/positivas/` e `dados/negativas/`, aceita `.jpg`, `.jpeg`, `.png` e `.bmp`, redimensiona tudo para 64x128 (largura x altura, a janela do HOG) com `cv2.resize` e grava como PNG numerado em `dados/processadas/positivas/` e `dados/processadas/negativas/`. Imprime quantas imagens achou em cada classe e lembra o requisito da entrega: no mínimo 100 de cada.

```python
# 03_hog_svm_atividade/item_b_preparar_dataset.py
base = Path(__file__).resolve().parents[1]; out = base/'dados/processadas'
for nome in ['positivas','negativas']:
    src = base/'dados'/nome; dst = out/nome; dst.mkdir(parents=True, exist_ok=True)
    arqs = [p for p in src.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}]
    for i, p in enumerate(arqs):
        img = cv2.imread(str(p))
        if img is not None: cv2.imwrite(str(dst/f'{i:03d}.png'), cv2.resize(img, (64,128)))
```

Onde aparece: `item_b_preparar_dataset.py`.

### 3.26 Dataset sintético de demonstração

Como o zip não traz fotos, o professor incluiu um gerador que fabrica 120 positivas e 120 negativas com formas simples, só para o pipeline poder ser testado de ponta a ponta. As positivas são um fundo escuro aleatório (`rng.integers(0, 50, (128,64,3))`) com um círculo claro no alto (cabeça) e um retângulo claro vertical embaixo (tronco), com pequeno deslocamento horizontal aleatório. As negativas são fundo aleatório mais claro com 2 a 6 linhas de cores aleatórias. O gerador usa `np.random.default_rng(42)`, então o resultado é reprodutível.

O script também cria `dados/teste/cena_sintetica.png`, uma cena 320x384 com dois "bonecos" nas posições `(95,165)` e `(225,190)`, que serve de entrada para a janela deslizante. Repare que só `dados/teste/` recebe `mkdir`; `dados/positivas/` e `dados/negativas/` precisam existir antes.

```python
# 03_hog_svm_atividade/gerar_dataset_sintetico_demo.py
for i in range(120):
    img = rng.integers(0,50,(128,64,3),dtype=np.uint8)
    cv2.circle(img, (32+int(rng.integers(-3,4)),25), 10, (220,220,220), -1)
    cv2.rectangle(img, (23,38), (41,95), (210,210,210), -1)
    cv2.imwrite(str(pos/f'pos_{i:03d}.png'), img)
```

Onde aparece: `gerar_dataset_sintetico_demo.py`.

### 3.27 Treinar o SVM sobre vetores HOG

O script de treino percorre as duas pastas processadas, lê cada imagem em escala de cinza, redimensiona para 64x128 (garantia extra), extrai o vetor HOG de 3780 valores e monta a matriz `X` (`float32`) e o vetor de rótulos `y` (1 positivo, 0 negativo). Se só houver uma classe, aborta com `RuntimeError`.

`train_test_split(X, y, test_size=.25, random_state=42, stratify=y)` separa 25% para teste mantendo a proporção entre classes (`stratify`) e com semente fixa. O classificador é `SVC(kernel='rbf', C=10.0, gamma='scale', probability=True)`:

- `kernel='rbf'`: kernel gaussiano, que permite fronteiras não lineares (diferente do SVM linear do detector padrão do OpenCV).
- `C=10.0`: penalidade por erro de treino. Maior C ajusta mais aos dados.
- `gamma='scale'`: largura do kernel calculada a partir da variância dos dados.
- `probability=True`: habilita `predict_proba`, necessário para o limiar da janela deslizante. Deixa o treino mais lento porque roda uma validação cruzada interna.

Depois de `fit` e `predict`, o script imprime acurácia, precisão, recall e o `classification_report` com 4 casas decimais. Precisão é a fração das detecções que eram pessoas de fato; recall é a fração das pessoas que foram detectadas. Por fim `joblib.dump` salva o modelo em `03_hog_svm_atividade/modelo_hog_svm.joblib`, sobrescrevendo o do professor.

```python
# 03_hog_svm_atividade/item_b_treinar_svm.py
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, random_state=42, stratify=y)
svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True); svm.fit(Xtr, ytr)
pred = svm.predict(Xte)
print(f'Acurácia : {accuracy_score(yte,pred):.4f}')
print(f'Precisão : {precision_score(yte,pred,zero_division=0):.4f}')
print(f'Recall   : {recall_score(yte,pred,zero_division=0):.4f}')
joblib.dump(svm, base/'03_hog_svm_atividade/modelo_hog_svm.joblib')
```

Onde aparece: `item_b_treinar_svm.py`.

### 3.28 Janela deslizante com pirâmide de escalas

Um classificador de recortes 64x128 vira um detector percorrendo a imagem inteira com uma janela desse tamanho. O script carrega o `.joblib`, lê a imagem de `--imagem` (erro `FileNotFoundError` se não abrir) e entra em um laço de pirâmide: enquanto a imagem reduzida ainda comportar uma janela 64x128, converte para cinza, desliza a janela com passo `--passo` (padrão 16) em y e em x, extrai HOG do recorte, chama `svm.predict_proba(feat)[0, 1]` (probabilidade da classe 1, pessoa) e, se ela for maior ou igual a `--limiar` (padrão 0,80), guarda a caixa e a pontuação.

Como a janela tem tamanho fixo, pessoas maiores que 64x128 só são encontradas em versões reduzidas da imagem. A cada nível a imagem original é reduzida pelo fator `--escala` (padrão 1,25), e `scale_to_orig` acumula esse fator para converter as coordenadas da caixa de volta para a imagem original. Note que os caminhos de arquivo aqui vêm de `Path(__file__).resolve().parents[1]`, ou seja, de `exemplos_em_python/`, e o modelo é procurado em `03_hog_svm_atividade/modelo_hog_svm.joblib`, por isso a subpasta precisa manter esse nome.

```python
# 03_hog_svm_atividade/item_b_janela_deslizante.py
boxes = []; scores = []; scale_to_orig = 1.0; pyr = img0.copy()
while pyr.shape[1] >= 64 and pyr.shape[0] >= 128:
    g = cv2.cvtColor(pyr, cv2.COLOR_BGR2GRAY)
    for y in range(0, g.shape[0]-127, a.passo):
        for x in range(0, g.shape[1]-63, a.passo):
            roi = g[y:y+128, x:x+64]; feat = hog.compute(roi).reshape(1,-1)
            prob = float(svm.predict_proba(feat)[0,1])
            if prob >= a.limiar:
                boxes.append((int(x*scale_to_orig), int(y*scale_to_orig),
                              int((x+64)*scale_to_orig), int((y+128)*scale_to_orig))); scores.append(prob)
    nw = int(pyr.shape[1]/a.escala); nh = int(pyr.shape[0]/a.escala)
    if nw < 64 or nh < 128: break
    scale_to_orig *= a.escala; pyr = cv2.resize(img0, (nw,nh))
```

Onde aparece: `item_b_janela_deslizante.py`.

### 3.29 Supressão de não máximos (NMS) por IoU

A janela deslizante gera dezenas de caixas quase iguais em volta de cada pessoa, uma por posição e escala em que o SVM passou do limiar. A supressão de não máximos (Non-Maximum Suppression, NMS) fica só com a melhor de cada grupo. A função `nms(boxes, scores, t=.35)` ordena as caixas por pontuação decrescente, pega a primeira, calcula a IoU (Intersection over Union, área da interseção dividida pela área da união) dela com todas as outras e descarta as que têm IoU maior que `0,35`, porque são a mesma detecção. Repete com a próxima caixa restante até acabar. Devolve os índices mantidos.

Tudo é vetorizado com NumPy: `b.T` separa `x1, y1, x2, y2`; `np.maximum` e `np.minimum` acham o retângulo de interseção; `+1e-9` evita divisão por zero. A saída final é a imagem com as caixas sobreviventes em verde e a pontuação escrita acima de cada uma, salva em `saidas/deteccoes_janela_deslizante.jpg`, mais três contagens no terminal: janelas candidatas, detecções após NMS e o caminho do arquivo.

```python
# 03_hog_svm_atividade/item_b_janela_deslizante.py
def nms(boxes, scores, t=.35):
    b = np.asarray(boxes, np.float32); s = np.asarray(scores)
    x1, y1, x2, y2 = b.T; area = (x2-x1+1)*(y2-y1+1); order = s.argsort()[::-1]; keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1+1) * np.maximum(0, yy2-yy1+1)
        iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[np.where(iou <= t)[0] + 1]
    return keep
```

Onde aparece: `item_b_janela_deslizante.py`.

### 3.30 Custo computacional: janela deslizante versus YOLO

O comentário no meio de `item_b_janela_deslizante.py` fecha o arco iniciado pelo script 20. A janela deslizante "repete HOG+SVM em muitas posições e escalas": para uma imagem 320x384 com passo 16 e escala 1,25 são centenas de recortes, cada um com extração de HOG e uma chamada de `predict_proba`, e o número cresce com o quadrado da resolução e com o número de níveis da pirâmide. Cada recorte é processado do zero, sem reaproveitar nada dos vizinhos.

Detectores como o YOLO, ao contrário, passam a imagem inteira uma vez por um backbone convolucional e preveem caixas e classes para todas as regiões em uma única passagem, compartilhando as convoluções entre regiões vizinhas. Por isso "costuma ser muito mais eficiente em hardware moderno para detecção geral". O item A já dá uma medida disso: o cenário `preciso`, com `winStride=(4,4)` e `scale=1.03`, é várias vezes mais lento que o `rapido` porque avalia muito mais janelas.

Onde aparece: comentários de `item_b_janela_deslizante.py`, docstring de `20_deteccao_objetos_conceito.py`, tabela final de `item_a_hog_pedestres.py`.

## 4. Roteiro dos scripts

| Script | O que faz | Entrada | Saída gerada | Depende de |
|---|---|---|---|---|
| `01_fundamentos_convolucao/01_imagem_como_matriz.py` | Cria matriz 4x4 `uint8`, imprime shape e um pixel | Nenhuma (matriz fixa) | Texto no terminal | numpy |
| `01_fundamentos_convolucao/02_canais_rgb.py` | Imagem com três faixas coloridas, separa canais com `cv2.split` | Nenhuma (gerada) | `02_original.png`, `02_b.png`, `02_g.png`, `02_r.png` e texto | cv2, numpy |
| `01_fundamentos_convolucao/03_convolucao_manual.py` | Produto elemento a elemento e soma de uma janela 3x3 com um kernel 3x3 | Nenhuma (matrizes fixas) | Texto: imagem, kernel, produto e soma (270) | numpy |
| `01_fundamentos_convolucao/04_kernel_deslizante.py` | Convolução completa com laços `for`, 5x5 para 3x3 | Nenhuma (rampa 1 a 25) | Texto: entrada e saída 3x3 (todos `-6`) | numpy |
| `01_fundamentos_convolucao/05_borda_vertical.py` | `cv2.filter2D` com kernel de borda vertical | Nenhuma (metade preta, metade branca) | `05_borda_vertical.png` | cv2, numpy |
| `01_fundamentos_convolucao/06_borda_horizontal.py` | `cv2.filter2D` com kernel de borda horizontal | Nenhuma (topo preto, base branca) | `06_borda_horizontal.png` | cv2, numpy |
| `01_fundamentos_convolucao/07_sobel.py` | `cv2.Sobel` em x e em y sobre um retângulo | Nenhuma (retângulo desenhado) | `07_sobel_x.png`, `07_sobel_y.png` e frase sobre HOG | cv2, numpy |
| `01_fundamentos_convolucao/08_blur.py` | Filtro de média 5x5 sobre um círculo | Nenhuma (círculo desenhado) | `08_blur.png` | cv2, numpy |
| `01_fundamentos_convolucao/09_sharpen.py` | Kernel de realce sobre o texto "CNN" | Nenhuma (texto desenhado) | `09_sharpen.png` | cv2, numpy |
| `01_fundamentos_convolucao/10_stride.py` | Convolução manual com stride 1 e 2 | Nenhuma (rampa 1 a 49) | Texto: saída 5x5 e 3x3 | numpy |
| `01_fundamentos_convolucao/11_padding.py` | Imprime tamanhos de saída em VALID e SAME | Nenhuma (rampa 1 a 25) | Texto | numpy |
| `01_fundamentos_convolucao/12_multiplos_filtros.py` | Três kernels (vertical, horizontal, laplaciano) no mesmo retângulo | Nenhuma (retângulo desenhado) | `12_vertical.png`, `12_horizontal.png`, `12_laplaciano.png` e texto | cv2, numpy |
| `01_fundamentos_convolucao/13_relu.py` | Aplica `np.maximum(0, x)` a um vetor | Nenhuma (vetor fixo) | Texto | numpy |
| `01_fundamentos_convolucao/14_max_pooling.py` | Max pooling 2x2 com laços `for`, 4x4 para 2x2 | Nenhuma (matriz fixa) | Texto | numpy |
| `02_cnn_basica/15_cnn_mnist.py` | Treina CNN de dois blocos no MNIST por 2 épocas, avalia e salva | MNIST (download pelo Keras) | `modelo_mnist.keras`, summary e `[perda, acurácia]` no terminal | tensorflow, internet |
| `02_cnn_basica/16_visualizar_kernels.py` | Desenha 16 dos 32 kernels 3x3 da primeira `Conv2D` | `modelo_mnist.keras` | `16_kernels.png` e janela matplotlib | tensorflow, matplotlib, script 15 |
| `02_cnn_basica/17_visualizar_feature_maps.py` | Sub-modelo com as saídas das `Conv2D`, desenha 16 feature maps do primeiro dígito de teste | `modelo_mnist.keras`, MNIST | `17_feature_maps.png` e janela matplotlib | tensorflow, matplotlib, script 15 |
| `02_cnn_basica/18_cnn_cifar10.py` | Treina CNN com `padding='same'` e `GlobalAveragePooling2D` no CIFAR-10 por 3 épocas | CIFAR-10 (download pelo Keras) | Summary e `[perda, acurácia]` no terminal | tensorflow, internet |
| `02_cnn_basica/19_transfer_learning_mobilenet.py` | Monta MobileNetV2 congelada com cabeça `Dense(2)` e imprime o summary | Pesos ImageNet (download pelo Keras) | Summary no terminal | tensorflow, internet |
| `02_cnn_basica/20_deteccao_objetos_conceito.py` | Desenha duas caixas e imprime o docstring sobre classificação, detecção e YOLO | Nenhuma (gerada) | `20_deteccao_conceito.png` e texto | cv2, numpy |
| `03_hog_svm_atividade/item_a_hog_pedestres.py` | Detector de pedestres padrão do OpenCV em dois cenários, mede ms e FPS | Vídeo via `--video` (não incluído) | Janela com caixas, log por frame e tabela final no terminal | cv2, numpy, vídeo |
| `03_hog_svm_atividade/gerar_dataset_sintetico_demo.py` | Gera 120 positivas e 120 negativas sintéticas e uma cena de teste | Nenhuma (gerada) | `dados/positivas/pos_*.png`, `dados/negativas/neg_*.png`, `dados/teste/cena_sintetica.png` | cv2, numpy, pastas `dados/positivas` e `dados/negativas` existentes |
| `03_hog_svm_atividade/item_b_preparar_dataset.py` | Redimensiona positivas e negativas para 64x128 | `dados/positivas/`, `dados/negativas/` | `dados/processadas/positivas/*.png`, `dados/processadas/negativas/*.png` | cv2, dataset real ou sintético |
| `03_hog_svm_atividade/item_b_treinar_svm.py` | Extrai HOG, treina `SVC` rbf, imprime métricas e salva o modelo | `dados/processadas/` | `03_hog_svm_atividade/modelo_hog_svm.joblib` (sobrescreve o do professor) e métricas | cv2, numpy, sklearn, joblib, script anterior |
| `03_hog_svm_atividade/item_b_janela_deslizante.py` | Janela deslizante com pirâmide, `predict_proba`, NMS e imagem anotada | `--imagem`, `modelo_hog_svm.joblib` | `saidas/deteccoes_janela_deslizante.jpg` e contagens | cv2, numpy, sklearn, joblib, modelo, pasta `saidas/` existente |

Dependências entre scripts: no bloco 1 nenhuma. No bloco 2, 16 e 17 precisam do modelo salvo pelo 15. No bloco 3, a ordem é gerar ou colocar o dataset, preparar, treinar (opcional, porque o `.joblib` do professor já existe) e depois janela deslizante. O item A é independente, só precisa do vídeo.

## 5. Passo a passo para executar

Bloco 1, fundamentos de convolução. Rode de dentro da subpasta:

```powershell
cd C:\visao_computacional_PVA\aula_09_HOG_SVM\exemplos_em_python\01_fundamentos_convolucao
python 01_imagem_como_matriz.py
python 02_canais_rgb.py
python 03_convolucao_manual.py
python 04_kernel_deslizante.py
```

Espere ver a matriz 4x4, `Formato: (4, 4)` e `Pixel [2,3]: 160`; depois `Imagem: (200, 300, 3) Canal: (200, 300)` e quatro PNGs (a faixa da esquerda de `02_original.png` é azul, ordem BGR); `Soma: 270.0`; e a saída 3x3 toda igual a `-6`.

```powershell
python 05_borda_vertical.py
python 06_borda_horizontal.py
python 07_sobel.py
python 08_blur.py
python 09_sharpen.py
```

Abra os PNGs: linha vertical, linha horizontal, arestas verticais em `07_sobel_x.png` e horizontais em `07_sobel_y.png`, círculo com borda suave, texto "CNN" com contorno reforçado. O 07 também imprime a frase sobre Sobel e HOG.

```powershell
python 10_stride.py
python 11_padding.py
python 12_multiplos_filtros.py
python 13_relu.py
python 14_max_pooling.py
```

O 10 imprime uma saída 5x5 e uma 3x3. O 11 imprime `VALID com kernel 3x3 -> (3, 3)` e `SAME -> (5, 5)`. O 12 gera três PNGs, o do laplaciano com o contorno inteiro. O 13 imprime `ReLU: [0. 0. 0. 3. 9.]`. O 14 imprime `[[8, 4], [5, 9]]`. Nenhum script deste bloco abre janela nem aceita argumentos.

Bloco 2, CNN básica. Rode de dentro da subpasta, na ordem, com internet na primeira vez:

```powershell
cd C:\visao_computacional_PVA\aula_09_HOG_SVM\exemplos_em_python\02_cnn_basica
python 15_cnn_mnist.py
```

O Keras baixa o MNIST (11 MB), imprime o summary, treina 2 épocas (cerca de um minuto em CPU) e imprime algo como `[0.05, 0.98]`. Aparece `modelo_mnist.keras` na pasta.

```powershell
python 16_visualizar_kernels.py
python 17_visualizar_feature_maps.py
```

Cada um imprime um shape (`(3, 3, 1, 32)` no 16), salva um PNG com grade 4x4 e abre uma janela do matplotlib. Feche a janela para o script terminar.

```powershell
python 18_cnn_cifar10.py
python 19_transfer_learning_mobilenet.py
python 20_deteccao_objetos_conceito.py
```

O 18 baixa o CIFAR-10 (170 MB) e treina 3 épocas, alguns minutos em CPU; a acurácia fica em torno de 0,6 com tão poucas épocas. O 19 baixa os pesos ImageNet da MobileNetV2 (9 MB) e imprime um summary longo, com a maioria dos parâmetros como não treináveis. O 20 grava `20_deteccao_conceito.png` e imprime o docstring.

Bloco 3, HOG + SVM. Os scripts localizam tudo a partir de `exemplos_em_python/`, então podem ser chamados de qualquer diretório, mas o exemplo abaixo usa a subpasta.

Item A, com um vídeo de pedestres que você precisa providenciar:

```powershell
cd C:\visao_computacional_PVA\aula_09_HOG_SVM\exemplos_em_python\03_hog_svm_atividade
python item_a_hog_pedestres.py --video C:\caminho\para\pedestres.mp4
```

Abre a janela "HOG People Detector" com as caixas verdes e o texto do cenário, imprime uma linha por frame e, ao fim dos dois cenários, a tabela com Frames, Det/frame, ms/frame e FPS. Esc pula para o próximo cenário.

Item B, caminho rápido com o dataset sintético e o modelo do professor:

```powershell
python gerar_dataset_sintetico_demo.py
python item_b_preparar_dataset.py
python item_b_janela_deslizante.py --imagem ..\dados\teste\cena_sintetica.png
```

O gerador imprime duas mensagens e enche `dados/positivas/`, `dados/negativas/` e `dados/teste/`. A preparação imprime `positivas encontradas: 120` e `negativas encontradas: 120`. A janela deslizante imprime `Janelas candidatas`, `Detecções após NMS` e o caminho `saidas/deteccoes_janela_deslizante.jpg`; abra o arquivo para ver as caixas.

Item B, caminho completo, treinando o próprio modelo (sobrescreve o `.joblib` do professor; faça cópia antes se quiser guardar):

```powershell
Copy-Item modelo_hog_svm.joblib modelo_hog_svm_professor.joblib
python item_b_treinar_svm.py
python item_b_janela_deslizante.py --imagem ..\dados\teste\cena_sintetica.png --passo 8 --escala 1.2 --limiar 0.9
```

O treino imprime acurácia, precisão, recall e o relatório por classe, depois `Modelo salvo.`. Os argumentos opcionais da janela deslizante são `--passo` (padrão 16), `--escala` (padrão 1.25) e `--limiar` (padrão 0.80).

## 6. Exercícios e itens da aula

A atividade avaliada da aula está no bloco 3 e tem dois itens.

Item A, detector de pedestres pronto. Resolvido por `item_a_hog_pedestres.py`. O que pede: rodar o detector HOG + SVM padrão do OpenCV em um vídeo com pedestres, em dois cenários de parâmetros (`rapido` e `preciso`), medindo o tempo de inferência por frame e comparando frames processados, detecções por frame, milissegundos e FPS. O vídeo precisa ser fornecido via `--video`; o zip não traz nenhum.

Item B, detector próprio. Resolvido em três scripts, na ordem `item_b_preparar_dataset.py`, `item_b_treinar_svm.py` e `item_b_janela_deslizante.py`. O que pede: montar um dataset com no mínimo 100 imagens positivas e 100 negativas reais (a mensagem final do script de preparação repete o requisito), redimensionar para 64x128, extrair HOG, treinar um SVM, relatar acurácia, precisão e recall, e aplicar o classificador como detector com janela deslizante, pirâmide de escalas e NMS sobre uma imagem de teste. `gerar_dataset_sintetico_demo.py` existe só para testar o pipeline, e seu docstring é explícito: "Não substitui um dataset real para a entrega".

Os blocos 1 e 2 não referenciam itens ou exercícios numerados. Exercícios propostos para praticá-los:

1. Em `04_kernel_deslizante.py`, troque a rampa por uma imagem 5x5 com uma borda vertical no meio e preveja a saída antes de rodar. Depois transponha o kernel e confira que a resposta some.
2. Escreva uma função `conv2d(img, k, stride, pad)` que junte os scripts 04, 10 e 11 usando `np.pad`, e confirme que com `pad=1` e `stride=1` a saída tem o tamanho da entrada.
3. Substitua o retângulo de `07_sobel.py` por uma foto real em escala de cinza, calcule `cv2.magnitude(gx, gy)` e `cv2.phase(gx, gy, angleInDegrees=True)`, e monte à mão o histograma de 9 bins de orientação de uma célula 8x8. Compare com os 9 primeiros valores de `cv2.HOGDescriptor().compute` na mesma região.
4. Em `15_cnn_mnist.py`, aumente para 5 épocas e adicione `padding='same'` nas `Conv2D`. Rode o 16 de novo e compare os kernels com os da versão original.
5. Em `17_visualizar_feature_maps.py`, mostre os mapas da segunda `Conv2D` (`vm.predict(e)[1][0]`) e observe como ficam mais abstratos e menores (11x11).
6. Complete o `19_transfer_learning_mobilenet.py` com um dataset de duas classes (por exemplo o de formas da aula 08, escolhendo duas), compile com `sparse_categorical_crossentropy` e treine só a cabeça por algumas épocas.
7. No item B, varie `--passo` (8, 16, 32) e `--escala` (1.1, 1.25, 1.5) e anote o número de janelas candidatas e o tempo de execução. Relacione com a discussão de custo do conceito 3.30.
8. Troque o `SVC` rbf por `LinearSVC` (sem `predict_proba`; use `decision_function` como pontuação) e compare acurácia e tempo de treino.

## 7. Armadilhas e dicas

- Aula reconstruída. Só o zip `01_fundamentos_convolucao` tinha sido baixado na primeira organização, por isso o tutorial anterior registrava um "tema divergente". Com os zips `02_cnn_basica` e `03_hog_svm_atividade` a aula está completa e o nome `HOG_SVM` corresponde à atividade avaliada. A dúvida sobre esta aula em `achados_dos_zips.md` está encerrada.
- Layout com subpastas. Diferente das outras aulas, os scripts ficam em três subpastas de `exemplos_em_python/`, e o nome `03_hog_svm_atividade` não pode mudar, porque `item_b_treinar_svm.py` e `item_b_janela_deslizante.py` montam o caminho do `.joblib` com esse nome literal a partir de `Path(__file__).resolve().parents[1]`.
- Pastas que os scripts não criam. `gerar_dataset_sintetico_demo.py` grava em `dados/positivas/` e `dados/negativas/` sem `mkdir` (só `dados/teste/` é criada), e `item_b_janela_deslizante.py` grava em `saidas/` sem `mkdir`. Por isso as três pastas foram criadas vazias na reconstrução. Se apagá-las, `cv2.imwrite` falha em silêncio (devolve `False`) e a pasta fica vazia sem erro visível.
- O modelo do professor é sobrescrito. `item_b_treinar_svm.py` salva em `03_hog_svm_atividade/modelo_hog_svm.joblib`, o mesmo arquivo que veio no zip. Copie o original antes de treinar se quiser comparar.
- Versão do scikit-learn. Um `.joblib` de `SVC` é um pickle e depende da versão do scikit-learn que o gerou. Em versão diferente pode aparecer `InconsistentVersionWarning` ou falha ao carregar. Se falhar, treine de novo com `item_b_treinar_svm.py`.
- Item A sem vídeo. `--video` é obrigatório e nenhum vídeo veio no zip. Com um caminho inválido `cap.read()` devolve `False` no primeiro frame, o script conta 0 frames e a tabela sai com zeros, sem mensagem de erro.
- Item A fecha com Esc, não com q. `waitKey(1) & 0xFF == 27`. E a tecla só pula o cenário atual; o segundo cenário começa em seguida.
- Dataset sintético não vale para a entrega. O docstring é explícito. Com ele o SVM chega perto de 100% porque as classes são trivialmente separáveis, o que não diz nada sobre pedestres reais.
- Caminhos relativos nos blocos 1 e 2. `cv2.imwrite`, `plt.savefig`, `m.save` e `load_model` usam o diretório de trabalho. Rode de dentro de `01_fundamentos_convolucao/` ou `02_cnn_basica/`, senão os PNGs e o `.keras` vão parar em outro lugar e o 16 e o 17 não acham o modelo.
- Downloads na primeira execução. MNIST, CIFAR-10 e pesos ImageNet ficam em `~/.keras/datasets` e `~/.keras/models`. Sem internet os scripts 15, 17, 18 e 19 falham.
- `plt.show()` bloqueia. Os scripts 16 e 17 só terminam quando a janela do matplotlib é fechada.
- Código em uma linha só. Os scripts 15 a 20 e os do item B usam várias instruções por linha separadas por `;`, sem docstrings (exceto 20 e o gerador), no estilo do professor para caber em slides. Mantido por fidelidade. Nenhum defeito de transcrição nesta aula.
- `uint8` estoura. Multiplicar imagem `uint8` por kernel com pesos negativos dá overflow silencioso. O professor declara os kernels em `np.float32` e pede `cv2.CV_32F` ou `cv2.CV_64F` nos filtros de borda; nos de blur e sharpen, cujos pesos somam 1, usa `ddepth=-1`.
- `convertScaleAbs` esconde o sinal. Uma borda de claro para escuro e uma de escuro para claro ficam idênticas no PNG. Para ver o sinal, imprima `r.min()` e `r.max()` do resultado em `CV_32F`.
- Ordem BGR e coordenadas trocadas. `(255, 0, 0)` é azul. NumPy indexa `[y, x]`, as funções de desenho recebem `(x, y)`, e `cv2.resize` recebe `(largura, altura)`: por isso `cv2.resize(img, (64,128))` produz um recorte de 64 de largura por 128 de altura.
- `11_padding.py` não calcula nada. Só imprime tuplas escritas à mão. É intencional.
- Correlação, não convolução. `cv2.filter2D` e os laços dos scripts não espelham o kernel. Para kernels simétricos não muda nada; para os de borda só inverte o sinal. Em redes neurais "convolução" designa exatamente essa operação.
- `probability=True` é lento. Roda validação cruzada interna de 5 dobras no treino do `SVC`. Com o dataset sintético é rápido; com milhares de imagens reais pode levar minutos.
- Parâmetros que valem variar: no bloco 1, tamanho do kernel de blur, peso central do sharpen, `ksize` do Sobel, stride e janela de pooling; no bloco 2, número de épocas, número de filtros e `padding`; no item A, `winStride` e `scale`; no item B, `C` e `gamma` do SVC, `--passo`, `--escala`, `--limiar` e o `t` da NMS.

## 8. Glossário

| Termo | Significado |
|---|---|
| Adam | Otimizador de gradiente com taxa de aprendizado adaptativa, padrão em Keras. |
| Backbone | Parte convolucional de uma rede de detecção que extrai características da imagem inteira. |
| BGR | Ordem dos canais de cor usada pelo OpenCV: azul, verde, vermelho. O inverso do RGB usual. |
| Bin | Intervalo de um histograma. O HOG usa 9 bins de orientação por célula. |
| Bloco (HOG) | Grupo de 2x2 células (16x16 pixels) normalizado em conjunto; desliza com passo de 8 pixels. |
| Blur | Borramento. Filtro que substitui cada pixel pela média da vizinhança. |
| Bounding box | Caixa delimitadora, retângulo `(x, y, w, h)` ou `(x1, y1, x2, y2)` em volta de um objeto detectado. |
| `C` (SVM) | Penalidade por erro de treino. Maior C ajusta mais aos dados, com risco de overfitting. |
| Canal | Um plano bidimensional de uma imagem colorida. Imagens BGR têm três. |
| Célula (HOG) | Região de 8x8 pixels em que se calcula um histograma de orientações. |
| CIFAR-10 | Dataset de 60 mil imagens coloridas 32x32 em 10 classes. |
| CNN | Convolutional Neural Network, rede neural convolucional. |
| `compile` | Método do Keras que define otimizador, função de perda e métricas antes do treino. |
| `Conv2D` | Camada convolucional do Keras: vários kernels aprendidos aplicados à entrada. |
| `convertScaleAbs` | Função do OpenCV que aplica valor absoluto, arredonda e satura em `uint8`. |
| Convolução | Deslizar um kernel pela imagem somando o produto elemento a elemento. Em redes neurais é, a rigor, correlação cruzada. |
| Correlação cruzada | Convolução sem espelhar o kernel. É o que `cv2.filter2D` e as CNNs fazem. |
| `ddepth` | Parâmetro de `filter2D` e `Sobel` que define o tipo numérico da saída. |
| `Dense` | Camada totalmente conectada do Keras. |
| Detecção | Tarefa de localizar objetos com caixa e classe, em vez de só classificar a imagem. |
| `detectMultiScale` | Método do `HOGDescriptor` que percorre a imagem em várias escalas e devolve as caixas onde o SVM disparou. |
| dtype | Tipo de dado de um array NumPy: `uint8`, `float32`, `float64`. |
| Época | Uma passagem completa pelo conjunto de treino. |
| `evaluate` | Método do Keras que calcula perda e métricas em um conjunto de dados. |
| Feature map | Mapa de características. A matriz produzida por um kernel aplicado à imagem. |
| `filter2D` | Função do OpenCV que aplica um kernel arbitrário a uma imagem. |
| `fit` | Método do Keras que treina o modelo. |
| `Flatten` | Camada que transforma um tensor em vetor para as camadas densas. |
| FPS | Frames por segundo. No item A, `1000 / ms_por_frame`. |
| `gamma` (SVM) | Largura do kernel rbf. `'scale'` calcula a partir da variância dos dados. |
| `GlobalAveragePooling2D` | Camada que tira a média espacial de cada feature map, gerando um vetor com um valor por mapa. |
| Gradiente | Vetor das derivadas da intensidade em x e y. Aponta na direção de maior variação. |
| HOG | Histogram of Oriented Gradients. Descritor que concatena histogramas de orientação do gradiente de blocos normalizados. Vetor de 3780 valores na janela 64x128. |
| `HOGDescriptor` | Classe do OpenCV que calcula HOG e, com `setSVMDetector`, detecta objetos. |
| ImageNet | Dataset de 1,2 milhão de imagens em 1000 classes usado para pré-treinar redes como a MobileNetV2. |
| IoU | Intersection over Union. Área da interseção de duas caixas dividida pela área da união. Mede sobreposição. |
| Janela deslizante | Percorrer a imagem com uma janela de tamanho fixo, classificando cada recorte. |
| joblib | Biblioteca de serialização usada pelo scikit-learn para salvar modelos. |
| Kernel | Matriz pequena de pesos que define o que a convolução detecta. No SVM, função que mede similaridade entre vetores. |
| Laplaciano | Kernel de derivada de segunda ordem que responde a bordas em todas as direções. |
| Max pooling | Redução de resolução que mantém o maior valor de cada bloco. |
| MNIST | Dataset de 70 mil dígitos manuscritos 28x28 em escala de cinza. |
| MobileNetV2 | CNN leve pré-treinada no ImageNet, usada como base para transfer learning. |
| NMS | Non-Maximum Suppression, supressão de não máximos. Remove caixas redundantes mantendo a de maior pontuação. |
| Negativa | Imagem de treino sem o objeto de interesse (fundo). |
| Padding | Borda acrescentada à imagem antes da convolução. `VALID` sem borda, `SAME` mantém o tamanho. No `detectMultiScale`, borda em volta de cada janela. |
| Pirâmide de escalas | Sequência de versões reduzidas da imagem para detectar objetos de tamanhos diferentes com uma janela fixa. |
| Positiva | Imagem de treino com o objeto de interesse (pessoa). |
| Precisão | Fração das detecções que eram corretas. |
| `predict_proba` | Método do `SVC` com `probability=True` que devolve a probabilidade de cada classe. |
| `preprocess_input` | Função que escala pixels para a faixa que a rede pré-treinada espera (`[-1, 1]` na MobileNetV2). |
| rbf | Radial basis function, kernel gaussiano do SVM. Permite fronteiras não lineares. |
| Recall | Fração dos objetos reais que foram detectados. |
| ReLU | Rectified Linear Unit, `max(0, x)`. Função de ativação que zera valores negativos. |
| `Sequential` | Modelo Keras formado por uma pilha linear de camadas. |
| Shape | Tupla com as dimensões de um array. |
| Sharpen | Realce. Kernel que amplifica a diferença entre um pixel e seus vizinhos. |
| Sobel | Operador de gradiente com dois kernels 3x3, com peso dobrado no centro. |
| Softmax | Ativação que transforma um vetor em probabilidades que somam 1. |
| `sparse_categorical_crossentropy` | Perda de classificação multiclasse que aceita rótulos inteiros, sem one-hot. |
| Stride | Passo com que o kernel ou a janela avança. `winStride` no HOG, `--passo` no item B. |
| `stratify` | Opção de `train_test_split` que mantém a proporção das classes nas duas partes. |
| `SVC` | Support Vector Classifier do scikit-learn, implementação de SVM com kernels. |
| SVM | Support Vector Machine. Classificador que busca a fronteira de margem máxima entre classes. |
| Transfer learning | Reaproveitar uma rede pré-treinada, congelando a base e treinando só uma cabeça nova. |
| `uint8` | Inteiro sem sinal de 8 bits, de 0 a 255. Tipo padrão de imagens. |
| Validação | Parte do treino separada para medir o modelo a cada época (`validation_split`). |
| YOLO | You Only Look Once. Família de detectores que preveem caixas e classes em uma única passagem convolucional pela imagem. |
