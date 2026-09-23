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
paths = generate_dataset(ROOT, n=18, noisy=True)

_, K1, d1, rv1, tv1, op1, ip1, used, _ = calibrate_from_paths(paths)
e1 = reprojection_errors(K1,d1,rv1,tv1,op1,ip1)

pior = int(np.argmax(e1))
paths_filtrados = [p for i,p in enumerate(used) if i != pior]

_, K2, d2, rv2, tv2, op2, ip2, _, _ = calibrate_from_paths(paths_filtrados)
e2 = reprojection_errors(K2,d2,rv2,tv2,op2,ip2)

print("Antes:")
print(f"  erro médio = {np.mean(e1):.4f} px")
print("Depois de retirar a pior imagem:")
print(f"  erro médio = {np.mean(e2):.4f} px")

# DESAFIO DO ALUNO:
# A remoção sempre melhora a qualidade física da calibração?
# Discuta por que "menor erro" não é o único critério.
