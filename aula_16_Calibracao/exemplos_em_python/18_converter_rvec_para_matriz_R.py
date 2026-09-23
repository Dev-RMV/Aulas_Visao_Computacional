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

import cv2
from synthetic_utils import euler_to_rvec

rvec = euler_to_rvec(15, -10, 5)
R, _ = cv2.Rodrigues(rvec)

print("Vetor de rotação rvec:")
print(rvec)

print("\nMatriz de rotação R:")
print(R)

print("\nTeste aproximado de ortogonalidade R^T R:")
print(R.T @ R)

# DESAFIO DO ALUNO:
# Explique por que R^T R deve se aproximar da matriz identidade.
