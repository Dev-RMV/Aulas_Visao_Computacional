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

paths_limpos = generate_dataset(ROOT, n=18, noisy=False)
r1, K1, d1, rv1, tv1, op1, ip1, *_ = calibrate_from_paths(paths_limpos)
e1 = np.mean(reprojection_errors(K1,d1,rv1,tv1,op1,ip1))

paths_ruidosos = generate_dataset(ROOT, n=18, noisy=True)
r2, K2, d2, rv2, tv2, op2, ip2, *_ = calibrate_from_paths(paths_ruidosos)
e2 = np.mean(reprojection_errors(K2,d2,rv2,tv2,op2,ip2))

print(f"Erro médio - imagens limpas:   {e1:.4f} px")
print(f"Erro médio - imagens ruidosas: {e2:.4f} px")

# DESAFIO DO ALUNO:
# Aumente noise_std dentro de generate_dataset e repita a comparação.
