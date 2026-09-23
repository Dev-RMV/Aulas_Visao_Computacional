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
from seg_utils import salvar_cenas, hsv_pista

ROOT = Path(__file__).resolve().parent
salvar_cenas(ROOT, 5)

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))
mask = hsv_pista(img)

color = np.zeros_like(img)
color[mask > 0] = (0, 255, 255)
out = cv2.addWeighted(img, 0.75, color, 0.25, 0)

cv2.imwrite(str(ROOT / "saidas" / "07_overlay_hsv.png"), out)
print("Overlay HSV salvo.")

# DESAFIO DO ALUNO:
# Troque a cor da máscara e justifique a escolha visual.
