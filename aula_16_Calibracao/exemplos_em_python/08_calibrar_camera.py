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
from synthetic_utils import generate_dataset, calibrate_from_paths

ROOT = Path(__file__).resolve().parent
paths = generate_dataset(ROOT, n=18)

rms, K, dist, rvecs, tvecs, objpoints, imgpoints, used, size = \
    calibrate_from_paths(paths)

print("RMS retornado pelo calibrateCamera:", rms)
print("\nMatriz intrínseca K:")
print(K)
print("\nCoeficientes de distorção:")
print(dist.ravel()[:5])

# Significado físico:
# fx, fy -> focais medidas em pixels
# cx, cy -> ponto principal da imagem
# k1, k2, k3 -> distorção radial
# p1, p2 -> distorção tangencial

# DESAFIO DO ALUNO:
# Identifique em K os valores fx, fy, cx e cy e imprima cada um separadamente.
