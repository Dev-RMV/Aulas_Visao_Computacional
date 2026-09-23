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
from synthetic_utils import (
    generate_view, find_refined_corners, object_points,
    TRUE_K, TRUE_DIST, SQUARE_SIZE_M
)

ROOT = Path(__file__).resolve().parent
saida = ROOT / "saidas" / "24_ar_video_sintetico.avi"

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(str(saida), fourcc, 15.0, (1280,720))

s = SQUARE_SIZE_M
cube = np.float32([
    [0,0,0],[s,0,0],[s,s,0],[0,s,0],
    [0,0,-s],[s,0,-s],[s,s,-s],[0,s,-s]
])

edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

for i in range(90):
    a = i / 89.0
    frame, _, _ = generate_view(
        8 + 8*np.sin(a*np.pi*2),
        -12 + 12*np.sin(a*np.pi),
        6*np.sin(a*np.pi*2),
        -0.10 + 0.02*np.sin(a*np.pi*2),
        -0.07,
        0.75 + 0.05*np.sin(a*np.pi*2)
    )

    ok, corners = find_refined_corners(frame)

    if ok:
        _, rvec, tvec = cv2.solvePnP(
            object_points(), corners, TRUE_K, TRUE_DIST
        )
        pts, _ = cv2.projectPoints(
            cube, rvec, tvec, TRUE_K, TRUE_DIST
        )
        p = pts.reshape(-1,2).astype(int)

        for a1,b1 in edges:
            cv2.line(frame, tuple(p[a1]), tuple(p[b1]), (0,120,255), 3)

    writer.write(frame)

writer.release()
print("Vídeo AR criado em:", saida)

# DESAFIO DO ALUNO:
# Faça as arestas da base, topo e verticais terem cores diferentes.
