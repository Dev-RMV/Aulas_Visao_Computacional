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
    generate_dataset, calibrate_from_paths, TRUE_K
)

ROOT = Path(__file__).resolve().parent
paths = generate_dataset(ROOT, n=18)

_, K, _, *_ = calibrate_from_paths(paths)

print("K verdadeira usada no simulador:")
print(TRUE_K)

print("\nK estimada pelo OpenCV:")
print(K)

print("\nDiferença K_estimada - K_real:")
print(K - TRUE_K)

erro_rel_fx = abs(K[0,0] - TRUE_K[0,0]) / TRUE_K[0,0] * 100
print(f"\nErro relativo em fx: {erro_rel_fx:.3f}%")

# DESAFIO DO ALUNO:
# Calcule também o erro relativo de fy.
