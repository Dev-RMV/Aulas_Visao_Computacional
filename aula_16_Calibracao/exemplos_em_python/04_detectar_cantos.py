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

img = cv2.imread(str(paths[4]))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ok, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE)

print("Tabuleiro encontrado?", ok)
if ok:
    vis = img.copy()
    cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, ok)
    cv2.imwrite(str(ROOT / "saidas" / "04_cantos_detectados.png"), vis)
    print("Quantidade de cantos:", len(corners))

# DESAFIO DO ALUNO:
# Troque paths[4] por outras imagens.
# Todas são detectadas? Qual pose parece mais difícil?
