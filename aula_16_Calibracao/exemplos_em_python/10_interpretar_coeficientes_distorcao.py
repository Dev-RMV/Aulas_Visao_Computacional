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

_, _, dist, *_ = calibrate_from_paths(paths)

d = dist.ravel()
k1, k2, p1, p2, k3 = d[:5]

print(f"k1 = {k1:.8f}")
print(f"k2 = {k2:.8f}")
print(f"p1 = {p1:.8f}")
print(f"p2 = {p2:.8f}")
print(f"k3 = {k3:.8f}")

print("\nComo a câmera sintética ideal foi criada sem distorção,")
print("esperamos coeficientes próximos de zero.")

# DESAFIO DO ALUNO:
# Pesquise/argumente:
# qual diferença física existe entre distorção radial e tangencial?
