"""Exemplo 03 — aplica kernels conhecidos antes de aprender os filtros da CNN."""

from __future__ import annotations

import cv2
import numpy as np

from utils import DATASET, OUTPUTS, ensure_dirs, require_dataset


require_dataset(); ensure_dirs()
path = sorted((DATASET / "test" / "quadrado").glob("*.png"))[0]
image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

kernel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
kernel_horizontal = kernel_vertical.T
kernel_blur = np.ones((3, 3), np.float32) / 9.0

outputs = [
    image,
    cv2.convertScaleAbs(cv2.filter2D(image, cv2.CV_32F, kernel_vertical)),
    cv2.convertScaleAbs(cv2.filter2D(image, cv2.CV_32F, kernel_horizontal)),
    cv2.filter2D(image, -1, kernel_blur),
]
labels = ["ORIGINAL", "BORDA VERTICAL", "BORDA HORIZONTAL", "SUAVIZACAO"]
tiles = []
for output, label in zip(outputs, labels):
    tile = cv2.cvtColor(cv2.resize(output, (256, 256), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 230, 255), 2, cv2.LINE_AA)
    tiles.append(tile)
destination = OUTPUTS / "03_convolucoes.png"
cv2.imwrite(str(destination), np.hstack(tiles))

print("Kernel vertical:\n", kernel_vertical)
print("Kernel horizontal:\n", kernel_horizontal)
print("Cada pixel de saída é uma soma ponderada da vizinhança 3×3.")
print("Imagem:", destination)

