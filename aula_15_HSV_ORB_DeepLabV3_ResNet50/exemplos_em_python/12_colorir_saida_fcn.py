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

rng = np.random.default_rng(1)
palette = rng.integers(0, 255, (256, 3), dtype=np.uint8)
color = palette[mask]

cv2.imwrite(str(ROOT / "saidas" / "12_mask_fcn_colorida.png"), color)
print("Máscara FCN colorida salva.")

# DESAFIO DO ALUNO:
# Use cores fixas para pessoa, carro, ônibus e bicicleta.
