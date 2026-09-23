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

ROOT = Path(__file__).resolve().parent

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))
color = cv2.imread(str(ROOT / "saidas" / "12_mask_fcn_colorida.png"))

if img is None or color is None:
    print("Execute 01, 11 e 12 antes.")
    raise SystemExit

color = cv2.resize(color, (img.shape[1], img.shape[0]))
out = cv2.addWeighted(img, 0.65, color, 0.35, 0)

cv2.imwrite(str(ROOT / "saidas" / "13_overlay_fcn.png"), out)
print("Overlay do FCN salvo.")

# DESAFIO DO ALUNO:
# Altere o peso do overlay e escolha a versão mais clara.
