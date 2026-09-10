"""
Exemplo 16 - Limiarização global, adaptativa e Otsu.

Nota técnica importante:
Otsu costuma ser superior a um limiar global escolhido manualmente porque
estima o limiar a partir do histograma. Entretanto, quando a iluminação varia
fortemente dentro da própria imagem, a limiarização adaptativa geralmente é
mais robusta, pois calcula limiares locais.
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
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Suavização leve reduz ruído antes da binarização.
    cinza_suave = cv2.GaussianBlur(cinza, (5, 5), 0)

    # 1) Limiar global fixo: o valor 127 é aplicado a toda a imagem.
    _, global_bin = cv2.threshold(
        cinza_suave,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    # 2) Limiar adaptativo: cada região utiliza um limiar local.
    adaptativo = cv2.adaptiveThreshold(
        cinza_suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,  # Tamanho ímpar da vizinhança.
        5,   # Constante subtraída da média ponderada local.
    )

    # 3) Otsu: o valor inicial de threshold é ignorado e calculado automaticamente.
    limiar_otsu, otsu = cv2.threshold(
        cinza_suave,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    print(f"Limiar calculado pelo método de Otsu: {limiar_otsu:.2f}")

    painel = cv2.hconcat([
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(global_bin), 360), "Global: 127"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(adaptativo), 360), "Adaptativo"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(otsu), 360), f"Otsu: {limiar_otsu:.1f}"),
    ])

    cv2.imshow("Exercicio 3A - limiarizacao", painel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
