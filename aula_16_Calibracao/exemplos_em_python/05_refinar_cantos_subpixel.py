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
from synthetic_utils import generate_dataset, PATTERN_SIZE

ROOT = Path(__file__).resolve().parent
paths = generate_dataset(ROOT, n=18)

img = cv2.imread(str(paths[7]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ok, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)

if not ok:
    raise RuntimeError("O tabuleiro não foi encontrado.")

criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    40,
    0.001
)

refined = cv2.cornerSubPix(
    gray, corners, (11, 11), (-1, -1), criteria
)

print("Primeiro canto antes do refinamento:", corners[0,0])
print("Primeiro canto após refinamento:", refined[0,0])

# DESAFIO DO ALUNO:
# Compare o deslocamento do primeiro canto:
# distancia = ||corner_refinado - corner_original||
# Qual foi o valor em pixels?
