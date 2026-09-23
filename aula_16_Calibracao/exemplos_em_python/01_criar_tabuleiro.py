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
from synthetic_utils import make_checkerboard

ROOT = Path(__file__).resolve().parent
img = make_checkerboard(square_px=100)

saida = ROOT / "dados_sinteticos" / "tabuleiro_base.png"
saida.parent.mkdir(exist_ok=True)

cv2.imwrite(str(saida), img)

print("Imagem criada em:", saida)
print("Cantos internos esperados: 7 x 6")
print("Quadrados desenhados: 8 x 7")

# DESAFIO DO ALUNO:
# 1) altere square_px para 60 e 140;
# 2) explique por que um padrão 7x6 de cantos internos precisa de 8x7 quadrados.
