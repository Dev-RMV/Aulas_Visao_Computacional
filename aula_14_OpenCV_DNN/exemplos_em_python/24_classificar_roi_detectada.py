"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import read_pipeline_frame, segment_red_hsv, predict_opencv, topk, draw_top3, OUT

frame, _ = read_pipeline_frame()
mask, roi = segment_red_hsv(frame)

if roi:
    x,y,w,h = roi
    crop = frame[y:y+h, x:x+w]
else:
    crop = frame
    x,y,w,h = 0,0,frame.shape[1], frame.shape[0]

prob = predict_opencv(crop)
vis = draw_top3(frame, topk(prob, 3))
cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 3)
cv2.imwrite(str(OUT/'24_classificar_roi.jpg'), vis)
print('ROI classificada e anotada.')

# DESAFIO DO ALUNO:
# Troque a ROI HSV pela caixa retornada pelo HOG ou Haar, quando houver detecção.
