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
ideal = cv2.imread(str(ROOT / "imagens" / "cena_01_mask.png"), cv2.IMREAD_GRAYSCALE)

hsv = hsv_pista(img) > 0
pista = ideal == 2

inter = np.logical_and(hsv, pista).sum()
union = np.logical_or(hsv, pista).sum()

print(f"IoU HSV para classe pista: {inter / union:.3f}")

# DESAFIO DO ALUNO:
# Calcule IoU para calçada criando uma máscara HSV própria.
