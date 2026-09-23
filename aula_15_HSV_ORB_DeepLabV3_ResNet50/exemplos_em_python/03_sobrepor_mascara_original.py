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
from seg_utils import salvar_cenas, sobrepor

ROOT = Path(__file__).resolve().parent
salvar_cenas(ROOT, 5)

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))
mask = cv2.imread(str(ROOT / "imagens" / "cena_01_mask.png"), cv2.IMREAD_GRAYSCALE)

out = sobrepor(img, mask, alpha=0.45)
cv2.imwrite(str(ROOT / "saidas" / "03_overlay_semantico.png"), out)
print("Overlay semitransparente salvo.")

# DESAFIO DO ALUNO:
# Teste alpha=0.25 e alpha=0.70 e compare a legibilidade.
