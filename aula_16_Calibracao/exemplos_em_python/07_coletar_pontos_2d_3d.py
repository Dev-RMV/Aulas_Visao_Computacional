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
    generate_dataset, object_points, find_refined_corners
)

ROOT = Path(__file__).resolve().parent
paths = generate_dataset(ROOT, n=18)

objpoints = []
imgpoints = []
obj = object_points()

for p in paths:
    img = cv2.imread(str(p))
    ok, corners = find_refined_corners(img)

    if ok:
        objpoints.append(obj.copy())
        imgpoints.append(corners)

print("Imagens válidas:", len(imgpoints))
print("Pontos 3D por imagem:", objpoints[0].shape)
print("Pontos 2D por imagem:", imgpoints[0].shape)

# DESAFIO DO ALUNO:
# Explique com suas palavras:
# - objpoints representa o quê?
# - imgpoints representa o quê?
