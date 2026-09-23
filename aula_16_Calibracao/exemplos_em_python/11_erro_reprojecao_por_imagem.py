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
from synthetic_utils import (
    generate_dataset, calibrate_from_paths, reprojection_errors
)

ROOT = Path(__file__).resolve().parent
paths = generate_dataset(ROOT, n=18)

_, K, dist, rvecs, tvecs, objpoints, imgpoints, used, _ = \
    calibrate_from_paths(paths)

errors = reprojection_errors(
    K, dist, rvecs, tvecs, objpoints, imgpoints
)

for p, e in zip(used, errors):
    print(f"{p.name}: {e:.4f} px")

# DESAFIO DO ALUNO:
# Descubra qual imagem apresentou o maior erro.
