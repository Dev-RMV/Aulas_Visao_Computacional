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
img, _, _ = generate_view(10, -18, 8, -0.10, -0.07, 0.80)

ok, corners = find_refined_corners(img)
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

overlay = img.copy()

faces = [
    ([0,1,2,3], (0,255,0)),
    ([4,5,6,7], (255,0,0)),
    ([0,1,5,4], (0,255,255)),
    ([1,2,6,5], (0,0,255)),
    ([2,3,7,6], (255,255,0)),
    ([3,0,4,7], (255,0,255)),
]

for ids, color in faces:
    poly = np.array([p[i] for i in ids], np.int32)
    cv2.fillConvexPoly(overlay, poly, color)

img = cv2.addWeighted(overlay, 0.30, img, 0.70, 0)

for ids, _ in faces:
    poly = np.array([p[i] for i in ids], np.int32)
    cv2.polylines(img, [poly], True, (30,30,30), 2)

cv2.imwrite(str(ROOT / "saidas" / "21_faces_coloridas.png"), img)
print("Faces coloridas e semitransparentes salvas.")

# DESAFIO DO ALUNO:
# Escolha uma face e altere somente a cor dela.
