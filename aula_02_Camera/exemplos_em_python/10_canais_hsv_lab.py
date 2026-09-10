"""
Exemplo 10 - Canais dos espaços de cor HSV e LAB.

Saídas:
- BGR original.
- Canais H, S e V.
- Canais L, A e B do espaço LAB.
- Um painel com sete imagens ao todo.
"""

import cv2


from pathlib import Path

import cv2


def obter_imagem_base(nome_arquivo: str = "imagem_base.png"):
    """Carrega a imagem-base ou abre a câmera para capturar uma nova imagem.

    A função torna o exemplo independente: se o arquivo ainda não existir,
    o estudante pode pressionar S para fotografar diretamente pela webcam.
    """

    caminho = Path(__file__).resolve().parent / nome_arquivo

    # Tenta carregar uma fotografia já existente.
    imagem = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if imagem is not None:
        print(f"Imagem carregada de: {caminho}")
        return imagem

    # Caso a imagem não exista, abre a câmera principal.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")

    print("Arquivo imagem_base.png não encontrado.")
    print("Pressione S para capturar uma imagem ou Q para cancelar.")

    imagem_capturada = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Falha ao capturar um frame da câmera.")

            exibicao = frame.copy()
            cv2.putText(
                exibicao,
                "S: capturar | Q: cancelar",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Captura da imagem-base", exibicao)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("s"), ord("S")):
                imagem_capturada = frame.copy()
                break
            if tecla in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if imagem_capturada is None:
        raise RuntimeError("A captura foi cancelada pelo usuário.")

    # Garante dimensões mínimas de 480 x 480 pixels.
    altura, largura = imagem_capturada.shape[:2]
    if altura < 480 or largura < 480:
        escala = max(480 / altura, 480 / largura)
        nova_largura = int(round(largura * escala))
        nova_altura = int(round(altura * escala))
        imagem_capturada = cv2.resize(
            imagem_capturada,
            (nova_largura, nova_altura),
            interpolation=cv2.INTER_CUBIC,
        )

    cv2.imwrite(str(caminho), imagem_capturada)
    print(f"Imagem salva em: {caminho}")
    return imagem_capturada



def adicionar_titulo(imagem, titulo: str):
    """Escreve um título na parte superior de uma cópia da imagem."""

    saida = imagem.copy()
    cv2.rectangle(saida, (0, 0), (saida.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        saida,
        titulo,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return saida


def canal_para_bgr(canal):
    """Converte um canal de uma dimensão para três canais BGR."""

    return cv2.cvtColor(canal, cv2.COLOR_GRAY2BGR)


def redimensionar_quadrado(imagem, lado: int = 320):
    """Redimensiona a imagem para facilitar a montagem de painéis didáticos."""

    return cv2.resize(imagem, (lado, lado), interpolation=cv2.INTER_AREA)



def main() -> None:
    imagem = obter_imagem_base()

    # OpenCV carrega imagens coloridas na ordem BGR.
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)

    # HSV:
    # H (Hue / Matiz): identifica a família de cor. No OpenCV varia de 0 a 179.
    # S (Saturation / Saturação): intensidade ou pureza da cor, de 0 a 255.
    # V (Value / Valor): brilho do pixel, de 0 a 255.
    canal_h, canal_s, canal_v = cv2.split(hsv)

    # LAB:
    # L: luminosidade perceptual, de escuro para claro.
    # A: eixo cromático aproximado entre verde e vermelho.
    # B: eixo cromático aproximado entre azul e amarelo.
    canal_l, canal_a, canal_b = cv2.split(lab)

    original = redimensionar_quadrado(imagem)
    imagens = [
        adicionar_titulo(original, "BGR original"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_h)), "HSV - H (matiz)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_s)), "HSV - S (saturacao)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_v)), "HSV - V (brilho)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_l)), "LAB - L (luminosidade)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_a)), "LAB - A (verde-vermelho)"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canal_b)), "LAB - B (azul-amarelo)"),
    ]

    # Cria uma célula preta para completar uma grade 2 x 4.
    celula_vazia = imagens[0] * 0
    cv2.putText(
        celula_vazia,
        "7 imagens solicitadas",
        (28, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    linha_1 = cv2.hconcat(imagens[0:4])
    linha_2 = cv2.hconcat(imagens[4:7] + [celula_vazia])
    painel = cv2.vconcat([linha_1, linha_2])

    print("Shapes dos espaços de cor:")
    print(f"BGR: {imagem.shape}")
    print(f"HSV: {hsv.shape}")
    print(f"LAB: {lab.shape}")
    print("Pressione Q ou Esc para fechar.")

    cv2.imshow("Exercicio 2A - canais HSV e LAB", painel)
    while True:
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
