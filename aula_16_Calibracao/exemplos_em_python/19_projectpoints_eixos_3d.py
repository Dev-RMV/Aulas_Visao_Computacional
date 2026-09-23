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
    generate_view, find_refined_corners, object_points, TRUE_K, TRUE_DIST,
    SQUARE_SIZE_M
)

ROOT = Path(__file__).resolve().parent
img, _, _ = generate_view(12, -14, 5, -0.10, -0.07, 0.75)

ok, corners = find_refined_corners(img)
if not ok:
    raise RuntimeError("Tabuleiro não encontrado.")

_, rvec, tvec = cv2.solvePnP(
    object_points(), corners, TRUE_K, TRUE_DIST
)

s = SQUARE_SIZE_M
axis3d = np.float32([
    [0,0,0],
    [s,0,0],
    [0,s,0],
    [0,0,-s]
])

pts2d, _ = cv2.projectPoints(axis3d, rvec, tvec, TRUE_K, TRUE_DIST)
p = pts2d.reshape(-1,2).astype(int)

origem = tuple(p[0])
cv2.line(img, origem, tuple(p[1]), (0,0,255), 4)   # X
cv2.line(img, origem, tuple(p[2]), (0,255,0), 4)   # Y
cv2.line(img, origem, tuple(p[3]), (255,0,0), 4)   # Z

cv2.imwrite(str(ROOT / "saidas" / "19_eixos_3d.png"), img)
print("Imagem com eixos 3D salva.")

# DESAFIO DO ALUNO:
# Aumente o comprimento dos eixos para 2 quadrados.
