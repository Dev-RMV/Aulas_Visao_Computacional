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
from seg_utils import salvar_cenas, hsv_pista, cronometro, ms

ROOT = Path(__file__).resolve().parent
salvar_cenas(ROOT, 5)

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))
tempos = {}

t0 = cronometro()
hsv = hsv_pista(img)
tempos["HSV"] = ms(t0)

t0 = cronometro()
kp, _ = cv2.ORB_create(250).detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
tempos["ORB"] = ms(t0)

t0 = cronometro()
_ = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 80, 160)
tempos["Canny/detector"] = ms(t0)

vis = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
vis[hsv > 0] = (0.6 * vis[hsv > 0] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)

cv2.imwrite(str(ROOT / "saidas" / "26_integrado_classico.png"), vis)
print(tempos)

# DESAFIO DO ALUNO:
# Converta o dicionário de tempos em uma tabela formatada.
