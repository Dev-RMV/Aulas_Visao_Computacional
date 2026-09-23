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
from seg_utils import salvar_cenas, hsv_pista, sobrepor, painel_lado_a_lado

ROOT = Path(__file__).resolve().parent
salvar_cenas(ROOT, 5)

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))
ideal = cv2.imread(str(ROOT / "imagens" / "cena_01_mask.png"), cv2.IMREAD_GRAYSCALE)
hsv = hsv_pista(img)
hsv_color = cv2.cvtColor(hsv, cv2.COLOR_GRAY2BGR)

painel = painel_lado_a_lado(
    img,
    cv2.addWeighted(img, 0.7, hsv_color, 0.3, 0),
    sobrepor(img, ideal)
)

cv2.imwrite(str(ROOT / "saidas" / "08_comparacao_hsv_semantica.png"), painel)
print("Painel comparativo salvo.")

# DESAFIO DO ALUNO:
# Adicione títulos nas três imagens do painel.
