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
from synthetic_utils import generate_view

ROOT = Path(__file__).resolve().parent
saida = ROOT / "dados_sinteticos" / "video_tabuleiro.avi"

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(str(saida), fourcc, 15.0, (1280,720))

for i in range(90):
    a = i / 89.0
    rx = 8 + 8*np.sin(a*np.pi*2)
    ry = -12 + 12*np.sin(a*np.pi)
    rz = 6*np.sin(a*np.pi*2)
    tz = 0.75 + 0.05*np.sin(a*np.pi*2)
    tx = -0.10 + 0.02*np.sin(a*np.pi*2)

    frame, _, _ = generate_view(rx, ry, rz, tx, -0.07, tz)
    writer.write(frame)

writer.release()
print("Vídeo sintético criado em:", saida)

# DESAFIO DO ALUNO:
# Mude o movimento para que o tabuleiro também varie em ty.
