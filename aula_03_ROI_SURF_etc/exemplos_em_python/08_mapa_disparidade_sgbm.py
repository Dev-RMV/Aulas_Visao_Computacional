"""Exemplo 08 — Mapa de disparidade com StereoSGBM.

As imagens devem estar retificadas: pontos correspondentes ficam na mesma linha.
Uso:
  python exemplos/08_mapa_disparidade_sgbm.py \
    --esquerda dados/gerados/estereo_esquerda.png \
    --direita dados/gerados/estereo_direita.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--esquerda", default="dados/gerados/estereo_esquerda.png")
    parser.add_argument("--direita", default="dados/gerados/estereo_direita.png")
    parser.add_argument("--num-disparidades", type=int, default=96)
    parser.add_argument("--bloco", type=int, default=5)
    args = parser.parse_args()

    if args.num_disparidades <= 0 or args.num_disparidades % 16 != 0:
        raise ValueError("--num-disparidades deve ser positivo e múltiplo de 16.")
    if args.bloco < 3 or args.bloco % 2 == 0:
        raise ValueError("--bloco deve ser ímpar e pelo menos 3.")

    esquerda = cv2.imread(args.esquerda, cv2.IMREAD_GRAYSCALE)
    direita = cv2.imread(args.direita, cv2.IMREAD_GRAYSCALE)
    if esquerda is None or direita is None:
        raise FileNotFoundError("Execute o exemplo 07 ou informe um par válido.")
    if esquerda.shape != direita.shape:
        raise ValueError("As imagens esquerda e direita precisam ter a mesma forma.")

    canais = 1
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=args.num_disparidades,
        blockSize=args.bloco,
        P1=8 * canais * args.bloco**2,
        P2=32 * canais * args.bloco**2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparidade = sgbm.compute(esquerda, direita).astype(np.float32) / 16.0
    valida = disparidade > 0

    visual = np.zeros_like(esquerda)
    if np.any(valida):
        minimo, maximo = np.percentile(disparidade[valida], [2, 98])
        visual = np.clip((disparidade - minimo) * 255 / max(maximo - minimo, 1e-6), 0, 255)
        visual = visual.astype(np.uint8)
    colorida = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)
    colorida[~valida] = 0

    saida = Path("dados/gerados")
    np.save(saida / "disparidade_float.npy", disparidade)
    cv2.imwrite(str(saida / "disparidade_colorida.png"), colorida)
    print(f"Pixels com disparidade válida: {100 * valida.mean():.1f}%")
    if np.any(valida):
        print(
            "Faixa válida (p5–p95):",
            np.percentile(disparidade[valida], [5, 95]).round(2),
            "px",
        )
    cv2.imshow("Disparidade SGBM", colorida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

