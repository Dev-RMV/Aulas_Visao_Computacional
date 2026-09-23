"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import read_pipeline_frame, segment_red_hsv, OUT

frame, _ = read_pipeline_frame()
mask, roi = segment_red_hsv(frame)
vis = frame.copy()

if roi:
    x,y,w,h = roi
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 3)
    cv2.putText(vis, 'ROI HSV', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
else:
    print('Nenhuma ROI encontrada.')

cv2.imwrite(str(OUT/'21_mascara_hsv.jpg'), mask)
cv2.imwrite(str(OUT/'21_roi_hsv.jpg'), vis)
print('Resultados salvos em saidas/. ROI:', roi)

# DESAFIO DO ALUNO:
# Altere os limites HSV para segmentar outra cor.
