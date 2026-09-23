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
import cv2
import numpy as np
from synthetic_utils import generate_dataset, calibrate_from_paths

ROOT = Path(__file__).resolve().parent
paths = generate_dataset(ROOT, n=18)

_, K, dist, *_ = calibrate_from_paths(paths)

img = cv2.imread(str(paths[9]))
corrigida = cv2.undistort(img, K, dist)

painel = np.hstack([img, corrigida])

cv2.putText(painel, "ORIGINAL", (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
cv2.putText(painel, "CORRIGIDA", (img.shape[1]+30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,120,0), 3)

saida = ROOT / "saidas" / "16_painel_original_corrigida.png"
cv2.imwrite(str(saida), painel)
print("Painel salvo em:", saida)

# DESAFIO DO ALUNO:
# Crie uma linha vertical exatamente no centro entre as duas imagens.
