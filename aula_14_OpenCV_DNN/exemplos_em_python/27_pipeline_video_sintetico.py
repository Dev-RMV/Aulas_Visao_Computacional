"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2, numpy as np, time
from dnn_utils import create_pipeline_frame, read_pipeline_frame, segment_red_hsv, orb_features, OUT

base_path = create_pipeline_frame()
base = cv2.imread(str(base_path))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out_path = OUT / '27_video_pipeline_sintetico.avi'
writer = cv2.VideoWriter(str(out_path), fourcc, 15.0, (640,480))

for i in range(60):
    frame = np.roll(base, shift=i*2, axis=1)
    mask, roi = segment_red_hsv(frame)
    vis = frame.copy()
    if roi:
        x,y,w,h=roi
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 2)
        kp,_ = orb_features(frame[y:y+h, x:x+w])
        cv2.putText(vis, f'ORB: {len(kp)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    writer.write(vis)
writer.release()
print('Vídeo sintético salvo em:', out_path)

# DESAFIO DO ALUNO:
# Acrescente a classificação DNN em cada frame.
