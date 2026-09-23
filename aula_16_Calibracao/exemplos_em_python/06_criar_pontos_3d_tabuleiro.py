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

from synthetic_utils import object_points, SQUARE_SIZE_M

obj = object_points()

print("Formato:", obj.shape)
print("Primeiros 10 pontos 3D:")
print(obj[:10])
print("Tamanho físico de cada quadrado [m]:", SQUARE_SIZE_M)

# DESAFIO DO ALUNO:
# Altere SQUARE_SIZE_M em synthetic_utils.py para 0.025.
# O que acontece com as coordenadas 3D?
# Qual grandeza da pose será diretamente afetada?
