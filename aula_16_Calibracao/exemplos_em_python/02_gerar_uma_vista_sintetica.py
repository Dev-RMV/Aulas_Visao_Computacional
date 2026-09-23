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
from synthetic_utils import generate_view

ROOT = Path(__file__).resolve().parent

img, rvec_real, tvec_real = generate_view(
    rx=12, ry=-15, rz=6,
    tx=-0.10, ty=-0.07, tz=0.75
)

saida = ROOT / "dados_sinteticos" / "vista_unica.png"
cv2.imwrite(str(saida), img)

print("Imagem sintética criada:", saida)
print("rvec usado para gerar a cena:\n", rvec_real)
print("tvec usado para gerar a cena [m]:\n", tvec_real)

# DESAFIO DO ALUNO:
# Gere uma segunda imagem alterando somente tz.
# Observe: aumentar tz aproxima ou afasta o tabuleiro?
