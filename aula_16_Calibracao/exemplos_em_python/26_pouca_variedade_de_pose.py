"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre calibração de câmera
e realidade aumentada com OpenCV.

A proposta é usar primeiro uma câmera virtual e imagens sintéticas. Assim,
é possível aprender cada etapa sem depender de webcam, iluminação, foco ou
qualidade de impressão do tabuleiro.

Ao final existe um "DESAFIO DO ALUNO". A parte principal já funciona, mas
o aluno deve modificar, medir, comparar ou completar uma extensão.
"""

from pathlib import Path
import cv2
import numpy as np
from synthetic_utils import (
    generate_view, calibrate_from_paths, TRUE_K
)

ROOT = Path(__file__).resolve().parent
folder = ROOT / "dados_sinteticos"
folder.mkdir(exist_ok=True)

paths = []
for i in range(12):
    # Quase todas as imagens são frontais e muito semelhantes.
    img, _, _ = generate_view(
        rx=1.0*i/12,
        ry=-1.0*i/12,
        rz=0,
        tx=-0.10,
        ty=-0.07,
        tz=0.75 + i*0.001
    )
    p = folder / f"pouca_var_{i:02d}.png"
    cv2.imwrite(str(p), img)
    paths.append(p)

_, K, *_ = calibrate_from_paths(paths)

print("K verdadeira:\n", TRUE_K)
print("\nK estimada com pouca variedade:\n", K)
print("\nDiferença:\n", K - TRUE_K)

# DESAFIO DO ALUNO:
# Explique por que muitas imagens quase iguais não substituem
# um conjunto com ângulos e distâncias variados.
