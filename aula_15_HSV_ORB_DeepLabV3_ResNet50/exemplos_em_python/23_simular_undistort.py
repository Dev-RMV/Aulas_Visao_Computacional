"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre segmentação semântica
e integração de técnicas de percepção visual para robótica e veículos autônomos.

A sequência começa com imagens sintéticas e procedimentos clássicos, avança
para modelos profundos pré-treinados e termina com um pipeline integrativo.
Sempre que houver modelo profundo, o código indica a dependência necessária
e a parte que deve ser adaptada para imagens reais.

Ao final há um bloco DESAFIO DO ALUNO para manter uma parte prática da aula.
"""
from pathlib import Path
import cv2
import numpy as np
from seg_utils import salvar_cenas

ROOT = Path(__file__).resolve().parent
salvar_cenas(ROOT, 5)

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))
h, w = img.shape[:2]

K = np.array([[800, 0, w/2], [0, 800, h/2], [0, 0, 1]], dtype=np.float32)
dist = np.array([-0.10, 0.03, 0, 0, 0], dtype=np.float32)

und = cv2.undistort(img, K, dist)
cv2.imwrite(str(ROOT / "saidas" / "23_undistort_didatico.png"), und)

print("Correção didática salva. Para câmera real, use K e dist calibrados anteriormente.")

# DESAFIO DO ALUNO:
# Substitua K e dist pela calibração real da câmera usada.
