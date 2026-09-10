"""
Exemplo 01 - Imagem digital como array NumPy.

Objetivo didatico:
- Ler uma imagem do disco.
- Interpretar shape, dtype, altura, largura e canais.
- Converter BGR para escala de cinza.
- Exibir colorida e cinza lado a lado.

Execucao:
    python 01_imagem_e_array_numpy.py

Observacao:
O script usa a imagem criada por 00_verificar_ambiente.py. Caso ela nao
exista, execute primeiro o Exemplo 00.
"""

from pathlib import Path

import cv2


def main() -> None:
    """Carrega uma imagem, inspeciona seus arrays e monta um painel."""

    pasta = Path(__file__).resolve().parent
    caminho_imagem = pasta / "teste_ambiente.png"

    # Carrega a imagem em cores. O OpenCV usa a ordem BGR por padrao.
    colorida = cv2.imread(str(caminho_imagem), cv2.IMREAD_COLOR)

    # Sempre valide o retorno de imread: None indica falha de leitura.
    if colorida is None:
        raise FileNotFoundError(
            f"Imagem nao encontrada em {caminho_imagem}. "
            "Execute primeiro 00_verificar_ambiente.py."
        )

    # Em uma imagem colorida, shape possui: (altura, largura, canais).
    altura, largura, canais = colorida.shape

    print("--- IMAGEM COLORIDA ---")
    print(f"Shape: {colorida.shape}")
    print(f"Dtype: {colorida.dtype}")
    print(f"Altura: {altura} pixels")
    print(f"Largura: {largura} pixels")
    print(f"Canais: {canais} (B, G, R)")
    print(f"Pixel [0, 0] em BGR: {colorida[0, 0]}")

    # Converte a imagem BGR para uma matriz de intensidade em tons de cinza.
    cinza = cv2.cvtColor(colorida, cv2.COLOR_BGR2GRAY)

    print("\n--- IMAGEM EM ESCALA DE CINZA ---")
    print(f"Shape: {cinza.shape}")
    print(f"Dtype: {cinza.dtype}")
    print(f"Pixel [0, 0] (intensidade): {cinza[0, 0]}")

    # hconcat exige arrays com mesma altura, mesmo dtype e mesmo numero de canais.
    # A imagem cinza e 2D; por isso criamos apenas uma copia BGR para exibicao.
    # A variavel 'cinza' continua sendo a imagem realmente monocromatica (H, W).
    cinza_para_exibir = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)

    # Adiciona rotulos no painel para facilitar a demonstracao em aula.
    cv2.putText(
        colorida,
        "Colorida - BGR",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        cinza_para_exibir,
        "Cinza - exibida em BGR",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Concatena horizontalmente as duas imagens.
    painel = cv2.hconcat([colorida, cinza_para_exibir])

    cv2.imshow("Imagem digital como array NumPy - pressione Q", painel)

    # Mantem a janela aberta ate que Q, q ou Esc seja pressionado.
    while True:
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27):
            break

    # Fecha todas as janelas criadas pelo OpenCV.
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
