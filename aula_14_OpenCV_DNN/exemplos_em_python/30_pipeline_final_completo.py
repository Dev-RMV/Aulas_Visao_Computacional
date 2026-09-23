"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import time, cv2
from dnn_utils import (
    read_pipeline_frame, DEFAULT_K, DEFAULT_DIST, segment_red_hsv, orb_features,
    hog_or_haar_detector, predict_opencv, topk, draw_top3, load_labels, OUT
)

frame, origem = read_pipeline_frame()
tempos = {}

# 1. Corrigir distorção da câmera.
t0=time.perf_counter()
frame_corrigido = cv2.undistort(frame, DEFAULT_K, DEFAULT_DIST)
tempos['undistort'] = (time.perf_counter()-t0)*1000

# 2. Segmentar ROI por cor HSV.
t0=time.perf_counter()
mask, roi_hsv = segment_red_hsv(frame_corrigido)
tempos['HSV'] = (time.perf_counter()-t0)*1000

vis = frame_corrigido.copy()
if roi_hsv:
    x,y,w,h = roi_hsv
    roi_img = frame_corrigido[y:y+h, x:x+w]
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 3)
    cv2.putText(vis, 'ROI HSV', (x,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,120,120), 2)
else:
    roi_img = frame_corrigido
    x,y,w,h = 0,0,frame_corrigido.shape[1], frame_corrigido.shape[0]

# 3. Extrair features ORB.
t0=time.perf_counter()
kp, des = orb_features(roi_img)
tempos['ORB'] = (time.perf_counter()-t0)*1000
for p in kp[:60]:
    px, py = int(p.pt[0] + x), int(p.pt[1] + y)
    cv2.circle(vis, (px,py), 2, (0,255,0), -1)

# 4. Aplicar HOG+SVM ou Haar Cascade.
t0=time.perf_counter()
boxes = hog_or_haar_detector(frame_corrigido)
tempos['HOG_Haar'] = (time.perf_counter()-t0)*1000
for nome,bx,by,bw,bh in boxes:
    cv2.rectangle(vis, (bx,by), (bx+bw,by+bh), (255,0,0), 2)
    cv2.putText(vis, nome, (bx,by-6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,0,0), 2)

# 5. Classificar ROI com OpenCV DNN.
t0=time.perf_counter()
prob = predict_opencv(roi_img)
tempos['DNN'] = (time.perf_counter()-t0)*1000
vis = draw_top3(vis, topk(prob, 3), load_labels())

# 6. Anotar tempos.
y0 = vis.shape[0] - 125
cv2.rectangle(vis, (10,y0), (370,vis.shape[0]-10), (255,255,255), -1)
cv2.rectangle(vis, (10,y0), (370,vis.shape[0]-10), (0,0,0), 1)
for i,(nome,ms) in enumerate(tempos.items()):
    cv2.putText(vis, f'{nome}: {ms:.1f} ms', (22,y0+24+i*19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 2)
cv2.putText(vis, f'Total: {sum(tempos.values()):.1f} ms', (22,y0+112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,180), 2)

saida = OUT / '30_pipeline_final_completo.jpg'
cv2.imwrite(str(saida), vis)

print('Origem do frame:', origem)
print('Frame final salvo em:', saida)
print('Tempo de cada etapa:')
for nome,ms in tempos.items():
    print(f'{nome:12s}: {ms:7.2f} ms')
print(f'Total       : {sum(tempos.values()):7.2f} ms')

# DESAFIO DO ALUNO:
# Substitua o frame sintético por uma captura real e use a calibração real da câmera.
