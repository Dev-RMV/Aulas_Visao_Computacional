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
img, _, _ = generate_view(14, -16, 6, -0.10, -0.07, 0.78)

ok, corners = find_refined_corners(img)
if not ok:
    raise RuntimeError("Tabuleiro não encontrado.")

_, rvec, tvec = cv2.solvePnP(
    object_points(), corners, TRUE_K, TRUE_DIST
)

s = SQUARE_SIZE_M
cube = np.float32([
    [0,0,0], [s,0,0], [s,s,0], [0,s,0],
    [0,0,-s], [s,0,-s], [s,s,-s], [0,s,-s]
])

pts, _ = cv2.projectPoints(cube, rvec, tvec, TRUE_K, TRUE_DIST)
p = pts.reshape(-1,2).astype(int)

for a,b in [(0,1),(1,2),(2,3),(3,0)]:
    cv2.line(img, tuple(p[a]), tuple(p[b]), (0,255,0), 3)
for a,b in [(4,5),(5,6),(6,7),(7,4)]:
    cv2.line(img, tuple(p[a]), tuple(p[b]), (255,0,0), 3)
for a,b in [(0,4),(1,5),(2,6),(3,7)]:
    cv2.line(img, tuple(p[a]), tuple(p[b]), (0,0,255), 3)

cv2.imwrite(str(ROOT / "saidas" / "20_cubo_3d.png"), img)
print("Cubo projetado com aresta igual a 1 quadrado.")

# DESAFIO DO ALUNO:
# Faça o cubo ter aresta de 2 quadrados.
