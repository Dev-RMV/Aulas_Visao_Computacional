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

kernel = np.ones((7, 7), np.uint8)
limpa = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
limpa = cv2.morphologyEx(limpa, cv2.MORPH_CLOSE, kernel)

cv2.imwrite(str(ROOT / "saidas" / "06_hsv_limpa.png"), limpa)
print("Máscara HSV após morfologia salva.")

# DESAFIO DO ALUNO:
# Teste kernel 3x3, 7x7 e 15x15.
