"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import read_pipeline_frame, segment_red_hsv, orb_features, OUT

frame, _ = read_pipeline_frame()
mask, roi = segment_red_hsv(frame)

if roi:
    x,y,w,h = roi
    crop = frame[y:y+h, x:x+w]
else:
    crop = frame

kp, des = orb_features(crop)
vis = cv2.drawKeypoints(crop, kp, None, color=(0,255,0), flags=0)
cv2.imwrite(str(OUT/'22_orb_roi.jpg'), vis)
print('Quantidade de keypoints ORB:', len(kp))
print('Descritores:', None if des is None else des.shape)

# DESAFIO DO ALUNO:
# Aumente nfeatures em dnn_utils.orb_features e compare a quantidade de pontos.
