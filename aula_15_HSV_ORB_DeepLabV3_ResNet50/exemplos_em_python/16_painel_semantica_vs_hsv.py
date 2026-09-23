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
from seg_utils import hsv_pista, painel_lado_a_lado

ROOT = Path(__file__).resolve().parent

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))
sem = cv2.imread(str(ROOT / "saidas" / "13_overlay_fcn.png"))

if img is None or sem is None:
    print("Execute 01, 11, 12 e 13 antes.")
    raise SystemExit

hsv = hsv_pista(img)
hsv = cv2.cvtColor(hsv, cv2.COLOR_GRAY2BGR)
hsv = cv2.addWeighted(img, 0.7, hsv, 0.3, 0)

painel = painel_lado_a_lado(img, hsv, sem)
cv2.imwrite(str(ROOT / "saidas" / "16_painel_semantica_vs_hsv.png"), painel)
print("Painel salvo.")

# DESAFIO DO ALUNO:
# Escreva no painel os títulos ORIGINAL, HSV e FCN.
