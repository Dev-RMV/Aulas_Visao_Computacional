"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
from dnn_utils import read_pipeline_frame, hog_or_haar_detector, OUT

frame, _ = read_pipeline_frame()
boxes = hog_or_haar_detector(frame)
vis = frame.copy()

if boxes:
    for nome,x,y,w,h in boxes:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(vis, nome, (x,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
else:
    cv2.putText(vis, 'HOG/Haar executado: sem deteccao neste frame', (30,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,180), 2)

cv2.imwrite(str(OUT/'23_detector_hog_haar.jpg'), vis)
print('Caixas detectadas:', boxes)

# DESAFIO DO ALUNO:
# Teste com uma foto contendo pessoa ou rosto em data/pipeline/frame_real.jpg.
