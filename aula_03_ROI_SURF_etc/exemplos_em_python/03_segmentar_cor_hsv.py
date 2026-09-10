"""Exemplo 03 — Segmentação por cor no espaço HSV.

O intervalo padrão seleciona o retângulo azul da cena sintética.
Uso: python exemplos/03_segmentar_cor_hsv.py --imagem dados/gerados/cena.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmentação de cor em HSV.")
    parser.add_argument("--imagem", default="dados/gerados/cena.png")
    parser.add_argument("--h-min", type=int, default=90)
    parser.add_argument("--h-max", type=int, default=135)
    parser.add_argument("--s-min", type=int, default=80)
    parser.add_argument("--v-min", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    imagem = cv2.imread(args.imagem)
    if imagem is None:
        raise FileNotFoundError(args.imagem)

    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    limite_inferior = np.array([args.h_min, args.s_min, args.v_min], dtype=np.uint8)
    limite_superior = np.array([args.h_max, 255, 255], dtype=np.uint8)
    mascara = cv2.inRange(hsv, limite_inferior, limite_superior)
    resultado = cv2.bitwise_and(imagem, imagem, mask=mascara)

    proporcao = 100.0 * cv2.countNonZero(mascara) / mascara.size
    print(f"Pixels selecionados: {proporcao:.2f}%")
    print("No OpenCV, H varia de 0 a 179; S e V variam de 0 a 255.")

    saida = Path("dados/gerados")
    cv2.imwrite(str(saida / "mascara_hsv.png"), mascara)
    cv2.imwrite(str(saida / "segmentacao_hsv.png"), resultado)
    cv2.imshow("Original | Máscara | Resultado", np.hstack([
        imagem,
        cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR),
        resultado,
    ]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

