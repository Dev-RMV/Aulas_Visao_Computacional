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
from seg_utils import salvar_cenas

ROOT = Path(__file__).resolve().parent
salvar_cenas(ROOT, 5)

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 160)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    if w * h > 1500:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

cv2.imwrite(str(ROOT / "saidas" / "25_detector_classico.png"), img)
print("Detector clássico didático salvo.")

# DESAFIO DO ALUNO:
# Substitua por Haar Cascade ou HOG+SVM quando usar imagens reais.
