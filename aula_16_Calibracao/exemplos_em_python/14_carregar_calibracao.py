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

ROOT = Path(__file__).resolve().parent
arquivo = ROOT / "camera_sintetica.npz"

if not arquivo.exists():
    print("Execute primeiro o exemplo 13.")
    raise SystemExit

data = np.load(arquivo)
K = data["K"]
dist = data["dist"]

print("K carregada:")
print(K)
print("\nDistorção carregada:")
print(dist.ravel()[:5])

# DESAFIO DO ALUNO:
# Imprima apenas fx, fy, cx e cy.
