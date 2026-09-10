"""Exemplo 04 — Limiarização global, adaptativa e de Otsu.

Uso: python exemplos/04_limiarizacao_otsu.py --imagem dados/gerados/cena.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagem", default="dados/gerados/cena.png")
    args = parser.parse_args()

    imagem = cv2.imread(args.imagem)
    if imagem is None:
        raise FileNotFoundError(args.imagem)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (5, 5), 0)

    _, global_ = cv2.threshold(suavizada, 127, 255, cv2.THRESH_BINARY)
    limiar_otsu, otsu = cv2.threshold(
        suavizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    adaptativa = cv2.adaptiveThreshold(
        suavizada,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    mosaico = np.hstack([cinza, global_, otsu, adaptativa])
    Path("dados/gerados").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("dados/gerados/comparacao_limiares.png", mosaico)
    print(f"Limiar escolhido automaticamente por Otsu: {limiar_otsu:.1f}")
    print("Ordem: cinza | global 127 | Otsu | adaptativa")
    cv2.imshow("Comparação de limiarização", mosaico)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

