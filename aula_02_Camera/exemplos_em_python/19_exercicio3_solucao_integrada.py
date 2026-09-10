"""
Exemplo 19 - Solução integrada do Exercício 3.

Etapas:
1. Compara limiar global, adaptativo e Otsu.
2. Mostra dois resultados do Canny.
3. Escolhe a limiarização adaptativa para uma cena com possível variação local
   de iluminação.
4. Encontra contornos, filtra área > 200 px² e colore por faixa de área.
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



def cor_por_area(area: float):
    if area < 1000:
        return (0, 255, 0)
    if area < 5000:
        return (0, 255, 255)
    return (0, 0, 255)


def main() -> None:
    original = obter_imagem_base()
    cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)

    _, global_bin = cv2.threshold(suave, 127, 255, cv2.THRESH_BINARY)
    adaptativo = cv2.adaptiveThreshold(
        suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    limiar_otsu, otsu = cv2.threshold(
        suave,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    print(f"Limiar de Otsu: {limiar_otsu:.2f}")
    print(
        "Escolha adotada: limiar adaptativo, pois tolera melhor "
        "gradientes locais de iluminação."
    )

    painel_limiares = cv2.hconcat([
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(global_bin), 320), "Global"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(adaptativo), 320), "Adaptativo"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(otsu), 320), "Otsu"),
    ])

    # Escolhemos a imagem adaptativa e a invertemos para deixar os objetos
    # principais brancos sobre fundo preto.
    binaria_escolhida = cv2.bitwise_not(adaptativo)

    # Conforme solicitado, o Canny é aplicado à imagem binarizada escolhida
    # utilizando dois pares de thresholds.
    canny_a = cv2.Canny(binaria_escolhida, 50, 150)
    canny_b = cv2.Canny(binaria_escolhida, 100, 200)

    painel_canny = cv2.hconcat([
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canny_a), 420), "Canny 50-150"),
        adicionar_titulo(redimensionar_quadrado(canal_para_bgr(canny_b), 420), "Canny 100-200"),
    ])

    # O fechamento conecta pequenas interrupções nas bordas do primeiro Canny.
    # O par 50-150 é escolhido porque preserva mais bordas da cena utilizada.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bordas_para_contornos = cv2.morphologyEx(
        canny_a,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    contornos, _ = cv2.findContours(
        bordas_para_contornos,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    resultado = original.copy()
    areas_validas = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area <= 200:
            continue

        areas_validas.append(area)
        cor = cor_por_area(area)
        cv2.drawContours(resultado, [contorno], -1, cor, 2)

        x, y, w, h = cv2.boundingRect(contorno)
        cv2.rectangle(resultado, (x, y), (x + w, y + h), cor, 2)

    print(f"Total de contornos: {len(contornos)}")
    print(f"Contornos válidos: {len(areas_validas)}")
    print(f"Maior área: {max(areas_validas, default=0.0):.2f} px²")

    cv2.imshow("Exercicio 3A - limiares", painel_limiares)
    cv2.waitKey(0)
    cv2.destroyWindow("Exercicio 3A - limiares")

    cv2.imshow("Exercicio 3B - Canny", painel_canny)
    cv2.waitKey(0)
    cv2.destroyWindow("Exercicio 3B - Canny")

    cv2.imshow("Exercicio 3B - contornos", resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
