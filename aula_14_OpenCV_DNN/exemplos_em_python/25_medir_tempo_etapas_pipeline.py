"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import time, cv2
from dnn_utils import read_pipeline_frame, DEFAULT_K, DEFAULT_DIST, segment_red_hsv, orb_features, hog_or_haar_detector, predict_opencv

frame, _ = read_pipeline_frame()
tempos = {}

t0=time.perf_counter(); undist=cv2.undistort(frame, DEFAULT_K, DEFAULT_DIST); tempos['undistort']=(time.perf_counter()-t0)*1000

t0=time.perf_counter(); mask, roi=segment_red_hsv(undist); tempos['HSV']=(time.perf_counter()-t0)*1000

t0=time.perf_counter(); kp, des=orb_features(undist); tempos['ORB']=(time.perf_counter()-t0)*1000

t0=time.perf_counter(); boxes=hog_or_haar_detector(undist); tempos['HOG_Haar']=(time.perf_counter()-t0)*1000

t0=time.perf_counter(); prob=predict_opencv(undist); tempos['DNN']=(time.perf_counter()-t0)*1000

print('Tempo por etapa:')
for k,v in tempos.items():
    print(f'{k:12s}: {v:7.2f} ms')
print(f'Total       : {sum(tempos.values()):7.2f} ms')

# DESAFIO DO ALUNO:
# Rode 10 vezes e calcule a média de cada etapa.
