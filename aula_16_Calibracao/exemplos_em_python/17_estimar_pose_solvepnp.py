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
from synthetic_utils import (
    generate_view, find_refined_corners, object_points, TRUE_K, TRUE_DIST
)

ROOT = Path(__file__).resolve().parent

img, rvec_real, tvec_real = generate_view(
    rx=15, ry=-12, rz=7,
    tx=-0.10, ty=-0.06, tz=0.78
)

ok, corners = find_refined_corners(img)
if not ok:
    raise RuntimeError("Tabuleiro não detectado.")

sucesso, rvec_est, tvec_est = cv2.solvePnP(
    object_points(), corners, TRUE_K, TRUE_DIST
)

print("solvePnP funcionou?", sucesso)
print("\nrvec estimado:\n", rvec_est)
print("\ntvec estimado [m]:\n", tvec_est)

print("\nComparação com a pose usada no simulador:")
print("tvec real [m]:\n", tvec_real)

# DESAFIO DO ALUNO:
# Calcule o erro Euclidiano entre tvec_est e tvec_real.
