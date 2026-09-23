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
from synthetic_utils import generate_dataset, calibrate_from_paths

ROOT = Path(__file__).resolve().parent
paths = generate_dataset(ROOT, n=18)

_, K, dist, *_ = calibrate_from_paths(paths)

img = cv2.imread(str(paths[5]))
corrigida = cv2.undistort(img, K, dist)

cv2.imwrite(str(ROOT / "saidas" / "15_original.png"), img)
cv2.imwrite(str(ROOT / "saidas" / "15_corrigida.png"), corrigida)

print("Imagens salvas em 'saidas'.")
print("Como a câmera virtual é quase ideal, a diferença deverá ser pequena.")

# DESAFIO DO ALUNO:
# Explique por que neste experimento a correção visual é pequena.
