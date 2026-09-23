"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import cv2
import numpy as np
from dnn_utils import read_pipeline_frame, DEFAULT_K, DEFAULT_DIST, OUT

frame, path = read_pipeline_frame()
undist = cv2.undistort(frame, DEFAULT_K, DEFAULT_DIST)
painel = np.hstack([frame, undist])
cv2.putText(painel, 'original', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
cv2.putText(painel, 'undistort', (frame.shape[1]+20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
cv2.imwrite(str(OUT/'20_undistort_painel.jpg'), painel)
print('Painel salvo em saidas/20_undistort_painel.jpg')

# DESAFIO DO ALUNO:
# Substitua DEFAULT_K e DEFAULT_DIST pelos valores da sua câmera calibrada.
