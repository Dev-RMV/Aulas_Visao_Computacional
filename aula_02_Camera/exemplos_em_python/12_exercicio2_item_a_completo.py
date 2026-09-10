"""
Exemplo 12 - Solução completa do Exercício 2, Item A.

O programa apresenta:
1. Uma grade com BGR original, três canais HSV e três canais LAB.
2. Um painel com saturação em 0%, 50% e 150%.
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
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * fator, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def main() -> None:
    imagem = obter_imagem_base()
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)

    # HSV: H=matiz, S=saturação e V=brilho.
    h, s, v = cv2.split(hsv)

    # LAB: L=luminosidade, A=eixo verde-vermelho, B=eixo azul-amarelo.
    l, a, b = cv2.split(lab)

    original = redimensionar_quadrado(imagem, 280)
    canais = [
        adicionar_titulo(original, "BGR original"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(h), 280), "HSV - H"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(s), 280), "HSV - S"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(v), 280), "HSV - V"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(l), 280), "LAB - L"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(a), 280), "LAB - A"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(b), 280), "LAB - B"),
    ]

    vazio = canais[0] * 0
    grade_canais = cv2.vconcat([
        cv2.hconcat(canais[:4]),
        cv2.hconcat(canais[4:] + [vazio]),
    ])

    saturacoes = [
        adicionar_titulo(redimensionar_quadrado(alterar_saturacao(imagem, 0.0), 360), "S=0%"),
        adicionar_titulo(redimensionar_quadrado(alterar_saturacao(imagem, 0.5), 360), "S=50%"),
        adicionar_titulo(redimensionar_quadrado(alterar_saturacao(imagem, 1.5), 360), "S=150%"),
    ]
    painel_saturacao = cv2.hconcat(saturacoes)

    print("Pressione qualquer tecla em cada janela para avançar.")
    cv2.imshow("Exercicio 2A - sete imagens", grade_canais)
    cv2.waitKey(0)
    cv2.destroyWindow("Exercício")

    cv2.imshow("Exercicio 2A - saturacao", painel_saturacao)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
