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

ROOT = Path(__file__).resolve().parent
mask = cv2.imread(str(ROOT / "saidas" / "11_mask_fcn.png"), cv2.IMREAD_GRAYSCALE)

if mask is None:
    print("Execute primeiro o exemplo 11.")
    raise SystemExit

vals, counts = np.unique(mask, return_counts=True)

for v, c in zip(vals, counts):
    pct = 100 * c / mask.size
    if pct > 0.5:
        print(f"classe_id={int(v):3d}: {pct:5.2f}%")

# DESAFIO DO ALUNO:
# Substitua ids numéricos pelos nomes das classes do weights.meta.
