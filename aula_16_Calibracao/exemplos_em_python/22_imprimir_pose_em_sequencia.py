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

import cv2
from synthetic_utils import (
    generate_view, find_refined_corners, object_points, TRUE_K, TRUE_DIST
)

poses = [
    (0, 0, 0, -0.10, -0.07, 0.70),
    (5, -5, 3, -0.10, -0.07, 0.72),
    (10, -10, 6, -0.10, -0.07, 0.74),
    (15, -15, 9, -0.10, -0.07, 0.76),
]

for i, pose in enumerate(poses, start=1):
    img, _, _ = generate_view(*pose)
    ok, corners = find_refined_corners(img)

    if ok:
        _, rvec, tvec = cv2.solvePnP(
            object_points(), corners, TRUE_K, TRUE_DIST
        )
        print(f"\nFRAME {i}")
        print("rvec:", rvec.ravel())
        print("tvec [m]:", tvec.ravel())

# DESAFIO DO ALUNO:
# Acrescente mais 4 frames e faça tz crescer progressivamente.
