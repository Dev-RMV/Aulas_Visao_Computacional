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
from synthetic_utils import generate_dataset, calibrate_from_paths

ROOT = Path(__file__).resolve().parent
paths = generate_dataset(ROOT, n=18)

_, K, dist, *_ = calibrate_from_paths(paths)

arquivo = ROOT / "camera_sintetica.npz"
np.savez(arquivo, K=K, dist=dist)

print("Calibração REAL estimada e salva em:", arquivo)
print("K:\n", K)
print("dist:", dist.ravel()[:5])

# DESAFIO DO ALUNO:
# Adicione ao arquivo .npz também a largura e a altura da imagem.
