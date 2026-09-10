"""Exemplo 02 — mostra que uma imagem RGB é um tensor altura × largura × canais."""

from __future__ import annotations

import cv2
import numpy as np

from utils import DATASET, OUTPUTS, ensure_dirs, require_dataset


require_dataset(); ensure_dirs()
path = sorted((DATASET / "train" / "circulo").glob("*.png"))[0]
bgr = cv2.imread(str(path))
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
normalized = rgb.astype(np.float32) / 255.0

print("Arquivo:", path)
print("shape do tensor:", rgb.shape, "= altura × largura × canais")
print("dtype antes:", rgb.dtype, "intervalo:", (rgb.min(), rgb.max()))
print("dtype depois:", normalized.dtype, "intervalo:", (normalized.min(), normalized.max()))
print("Pixel central RGB:", rgb[32, 32].tolist())

# Ampliação de uma região 8×8 para discutir pixels em sala.
patch = bgr[28:36, 28:36]
patch = cv2.resize(patch, (320, 320), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(OUTPUTS / "02_patch_pixels.png"), patch)
print("Patch ampliado:", OUTPUTS / "02_patch_pixels.png")

