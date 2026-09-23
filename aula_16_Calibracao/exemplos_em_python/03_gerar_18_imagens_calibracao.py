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
from synthetic_utils import generate_dataset

ROOT = Path(__file__).resolve().parent

paths = generate_dataset(ROOT, n=18)

print(f"Foram geradas {len(paths)} imagens.")
for p in paths:
    print(" -", p.name)

# DESAFIO DO ALUNO:
# Abra algumas imagens e identifique:
# - uma quase frontal;
# - uma inclinada;
# - uma mais distante.
# Explique por que variedade de poses ajuda a calibração.
