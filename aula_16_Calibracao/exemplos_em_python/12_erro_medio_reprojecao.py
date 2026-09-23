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
import numpy as np
from synthetic_utils import (
    generate_dataset, calibrate_from_paths, reprojection_errors
)

ROOT = Path(__file__).resolve().parent
paths = generate_dataset(ROOT, n=18)

_, K, dist, rvecs, tvecs, objpoints, imgpoints, _, _ = \
    calibrate_from_paths(paths)

errors = reprojection_errors(
    K, dist, rvecs, tvecs, objpoints, imgpoints
)

mean_error = np.mean(errors)
print(f"Erro médio de reprojeção: {mean_error:.4f} px")

print("\nReferência prática:")
print("0 a 0,5 px  -> geralmente muito bom")
print("0,5 a 1 px  -> frequentemente utilizável")
print("> 1 px       -> convém investigar")
print("A tolerância real depende da aplicação robótica.")

# DESAFIO DO ALUNO:
# Calcule também o desvio padrão dos erros.
