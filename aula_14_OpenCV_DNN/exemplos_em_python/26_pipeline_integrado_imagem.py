"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import time, cv2
from dnn_utils import read_pipeline_frame, DEFAULT_K, DEFAULT_DIST, segment_red_hsv, orb_features, hog_or_haar_detector, predict_opencv, topk, draw_top3, load_labels, OUT

frame, _ = read_pipeline_frame()
tempos = {}

# 1) undistort
t0=time.perf_counter(); img=cv2.undistort(frame, DEFAULT_K, DEFAULT_DIST); tempos['undistort']=(time.perf_counter()-t0)*1000

# 2) HSV ROI
t0=time.perf_counter(); mask, roi=segment_red_hsv(img); tempos['HSV']=(time.perf_counter()-t0)*1000
vis = img.copy()
if roi:
    x,y,w,h = roi
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,255), 3)
    roi_img = img[y:y+h, x:x+w]
else:
    roi_img = img

# 3) ORB
t0=time.perf_counter(); kp, des=orb_features(roi_img); tempos['ORB']=(time.perf_counter()-t0)*1000
if roi:
    x,y,w,h = roi
    for p in kp[:40]:
        px,py = int(p.pt[0]+x), int(p.pt[1]+y)
        cv2.circle(vis, (px,py), 2, (0,255,0), -1)

# 4) HOG ou Haar
t0=time.perf_counter(); boxes=hog_or_haar_detector(img); tempos['HOG_Haar']=(time.perf_counter()-t0)*1000
for nome,x,y,w,h in boxes:
    cv2.rectangle(vis, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.putText(vis, nome, (x,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

# 5) DNN na ROI
t0=time.perf_counter(); prob=predict_opencv(roi_img); tempos['DNN']=(time.perf_counter()-t0)*1000
vis = draw_top3(vis, topk(prob, 3), load_labels())

# Anotar tempos
ystart = vis.shape[0]-120
cv2.rectangle(vis, (10, ystart), (330, vis.shape[0]-10), (255,255,255), -1)
for i,(k,v) in enumerate(tempos.items()):
    cv2.putText(vis, f'{k}: {v:.1f} ms', (20, ystart+25+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
cv2.putText(vis, f'Total: {sum(tempos.values()):.1f} ms', (20, ystart+105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,180), 2)

cv2.imwrite(str(OUT/'26_pipeline_integrado.jpg'), vis)
print('Pipeline salvo em saidas/26_pipeline_integrado.jpg')
print('Tempos:')
for k,v in tempos.items(): print(f'{k}: {v:.2f} ms')

# DESAFIO DO ALUNO:
# Use uma imagem real e ajuste a segmentação HSV para uma cor presente nela.
