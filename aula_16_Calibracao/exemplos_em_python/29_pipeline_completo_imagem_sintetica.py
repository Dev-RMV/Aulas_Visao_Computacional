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
    generate_dataset, calibrate_from_paths, reprojection_errors,
    generate_view, find_refined_corners, object_points,
    SQUARE_SIZE_M
)

ROOT = Path(__file__).resolve().parent

# ETAPA 1 - gerar pelo menos 15 imagens.
paths = generate_dataset(ROOT, n=18)

# ETAPA 2 - calibrar.
_, K, dist, rvecs, tvecs, objpoints, imgpoints, _, _ = \
    calibrate_from_paths(paths)

# ETAPA 3 - erro de reprojeção.
errors = reprojection_errors(
    K, dist, rvecs, tvecs, objpoints, imgpoints
)

print("K:\n", K)
print("\n5 coeficientes:", dist.ravel()[:5])
print(f"\nErro médio: {np.mean(errors):.4f} px")

# ETAPA 4 - nova imagem para estimativa de pose.
frame, _, _ = generate_view(
    14, -17, 8, -0.10, -0.07, 0.80
)

ok, corners = find_refined_corners(frame)
if not ok:
    raise RuntimeError("Tabuleiro não encontrado.")

_, rvec, tvec = cv2.solvePnP(
    object_points(), corners, K, dist
)

print("\nrvec:", rvec.ravel())
print("tvec [m]:", tvec.ravel())

# ETAPA 5 - cubo com aresta de 1 quadrado.
s = SQUARE_SIZE_M
cube = np.float32([
    [0,0,0],[s,0,0],[s,s,0],[0,s,0],
    [0,0,-s],[s,0,-s],[s,s,-s],[0,s,-s]
])

pts, _ = cv2.projectPoints(cube, rvec, tvec, K, dist)
p = pts.reshape(-1,2).astype(int)

edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

for a,b in edges:
    cv2.line(frame, tuple(p[a]), tuple(p[b]), (0,100,255), 3)

cv2.imwrite(str(ROOT / "saidas" / "29_pipeline_completo.png"), frame)
print("\nResultado salvo em saidas/29_pipeline_completo.png")

# DESAFIO DO ALUNO:
# Troque o cubo de arame por faces semitransparentes de cores distintas.
