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
from seg_utils import salvar_cenas, sobrepor, porcentagens

ROOT = Path(__file__).resolve().parent
salvar_cenas(ROOT, 5)

img = cv2.imread(str(ROOT / "imagens" / "cena_01.png"))
mask = cv2.imread(str(ROOT / "imagens" / "cena_01_mask.png"), cv2.IMREAD_GRAYSCALE)

out = sobrepor(img, mask)
y = 25

for k, v in porcentagens(mask).items():
    cv2.putText(out, f"{k}: {v:.1f}%", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    y += 25

cv2.imwrite(str(ROOT / "saidas" / "28_frame_final_anotado.png"), out)
print("Frame final anotado salvo.")

# DESAFIO DO ALUNO:
# Adicione tempo de execução total no canto inferior direito.
