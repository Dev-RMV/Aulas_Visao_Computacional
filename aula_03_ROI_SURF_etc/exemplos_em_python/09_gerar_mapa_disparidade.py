"""
Gerador de mapa de disparidade sintético para testar
09_disparidade_para_profundidade.py

Considerando:
    focal = 500 px
    baseline = 0.12 m

A relação é:
    Z = f * B / d

Portanto:
    d = f * B / Z
"""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:

    largura = 800
    altura = 500

    focal_px = 500.0
    baseline_m = 0.12

    # Fundo inicialmente inválido
    disparidade = np.zeros(
        (altura, largura),
        dtype=np.float32
    )

    # -------------------------------------------------
    # Objeto 1: profundidade = 1 metro
    #
    # d = 500 * 0.12 / 1
    # d = 60 pixels
    # -------------------------------------------------

    disparidade[
        80:420,
        70:250
    ] = 60.0

    # -------------------------------------------------
    # Objeto 2: profundidade = 2 metros
    #
    # d = 500 * 0.12 / 2
    # d = 30 pixels
    # -------------------------------------------------

    disparidade[
        140:420,
        310:490
    ] = 30.0

    # -------------------------------------------------
    # Objeto 3: profundidade = 4 metros
    #
    # d = 500 * 0.12 / 4
    # d = 15 pixels
    # -------------------------------------------------

    disparidade[
        200:420,
        550:730
    ] = 15.0

    # Cria pasta
    saida = Path("dados/gerados")
    saida.mkdir(parents=True, exist_ok=True)

    # Salva mapa usado pelo exemplo 09
    arquivo = saida / "disparidade_float.npy"

    np.save(
        arquivo,
        disparidade
    )

    # -------------------------------------------------
    # Criar uma visualização da disparidade
    # -------------------------------------------------

    visual = cv2.normalize(
        disparidade,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    visual_colorida = cv2.applyColorMap(
        visual,
        cv2.COLORMAP_TURBO
    )

    # Textos explicativos
    cv2.putText(
        visual_colorida,
        "1 metro",
        (100, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        visual_colorida,
        "2 metros",
        (330, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        visual_colorida,
        "4 metros",
        (570, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imwrite(
        str(saida / "disparidade_teste.png"),
        visual_colorida
    )

    print("Mapa de disparidade criado.")
    print(f"Arquivo: {arquivo}")

    print()
    print("Valores esperados:")
    print("Objeto 1 -> disparidade 60 px -> 1 metro")
    print("Objeto 2 -> disparidade 30 px -> 2 metros")
    print("Objeto 3 -> disparidade 15 px -> 4 metros")

    cv2.imshow(
        "Mapa de disparidade sintetico",
        visual_colorida
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()