"""
Exemplo 11 - Alteração programática da saturação em HSV.

Resultados:
- Saturação a 0% do valor original.
- Saturação a 50% do valor original.
- Saturação a 150% do valor original, com saturação limitada a 255.
"""

import cv2
import numpy as np


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



def alterar_saturacao(imagem_bgr, fator: float):
    """Multiplica somente o canal S do espaço HSV pelo fator informado."""

    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)

    # Converte temporariamente para float32 para evitar estouro de uint8.
    hsv_float = hsv.astype(np.float32)
    hsv_float[:, :, 1] *= fator

    # Saturação deve permanecer no intervalo de 0 a 255.
    hsv_float[:, :, 1] = np.clip(hsv_float[:, :, 1], 0, 255)

    hsv_alterado = hsv_float.astype(np.uint8)
    return cv2.cvtColor(hsv_alterado, cv2.COLOR_HSV2BGR)


def main() -> None:
    imagem = obter_imagem_base()

    sat_0 = alterar_saturacao(imagem, 0.0)
    sat_50 = alterar_saturacao(imagem, 0.5)
    sat_150 = alterar_saturacao(imagem, 1.5)

    imagens = [
        adicionar_titulo(redimensionar_quadrado(sat_0), "Saturacao: 0%"),
        adicionar_titulo(redimensionar_quadrado(sat_50), "Saturacao: 50%"),
        adicionar_titulo(redimensionar_quadrado(sat_150), "Saturacao: 150%"),
    ]

    painel = cv2.hconcat(imagens)

    print("Observe que 0% remove as cores e 150% intensifica as cores.")
    print("Pressione Q ou Esc para fechar.")

    cv2.imshow("Exercicio 2A - variacao da saturacao", painel)
    while True:
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
