"""Exemplo 09 — Conversão de disparidade em profundidade métrica.

Relação ideal: Z = f · B / d.
Uso:
  python exemplos/09_disparidade_para_profundidade.py \
    --disparidade dados/gerados/disparidade_float.npy --focal-px 500 --baseline-m 0.12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disparidade", default="dados/gerados/disparidade_float.npy")
    parser.add_argument("--focal-px", type=float, default=500.0)
    parser.add_argument("--baseline-m", type=float, default=0.12)
    parser.add_argument("--max-m", type=float, default=10.0)
    args = parser.parse_args()

    disparidade = np.load(args.disparidade).astype(np.float32)
    valida = disparidade > 0.5
    profundidade = np.full(disparidade.shape, np.nan, dtype=np.float32)
    profundidade[valida] = args.focal_px * args.baseline_m / disparidade[valida]

    # PNG de 16 bits em milímetros: 0 representa inválido.
    profundidade_mm = np.zeros(disparidade.shape, dtype=np.uint16)
    metros_limitados = np.clip(profundidade[valida], 0, 65.535)
    profundidade_mm[valida] = np.round(metros_limitados * 1000).astype(np.uint16)

    escala = np.zeros(disparidade.shape, dtype=np.uint8)
    escala[valida] = np.clip(
        255 * (1.0 - profundidade[valida] / args.max_m), 0, 255
    ).astype(np.uint8)
    colorida = cv2.applyColorMap(escala, cv2.COLORMAP_TURBO)
    colorida[~valida] = 0

    saida = Path("dados/gerados")
    cv2.imwrite(str(saida / "profundidade_mm.png"), profundidade_mm)
    cv2.imwrite(str(saida / "profundidade_colorida.png"), colorida)
    np.save(saida / "profundidade_m.npy", profundidade)

    if np.any(valida):
        p = np.nanpercentile(profundidade, [5, 50, 95])
        print(f"Profundidade p5/p50/p95: {p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f} m")
    print("A escala métrica só é confiável com focal, baseline e retificação calibrados.")
    cv2.imshow("Profundidade aproximada", colorida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

