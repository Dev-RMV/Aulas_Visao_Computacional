import argparse,cv2,numpy as np,joblib
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--imagem',required=True); p.add_argument('--passo',type=int,default=16); p.add_argument('--escala',type=float,default=1.25); p.add_argument('--limiar',type=float,default=.80); a=p.parse_args()
base=Path(__file__).resolve().parents[1]; svm=joblib.load(base/'03_hog_svm_atividade/modelo_hog_svm.joblib'); hog=cv2.HOGDescriptor(_winSize=(64,128),_blockSize=(16,16),_blockStride=(8,8),_cellSize=(8,8),_nbins=9); img0=cv2.imread(a.imagem)
if img0 is None: raise FileNotFoundError(a.imagem)
def nms(boxes,scores,t=.35):
    if not boxes:return []
    b=np.asarray(boxes,np.float32); s=np.asarray(scores); x1,y1,x2,y2=b.T; area=(x2-x1+1)*(y2-y1+1); order=s.argsort()[::-1]; keep=[]
    while order.size:
        i=order[0]; keep.append(i); xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]]); xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]]); w=np.maximum(0,xx2-xx1+1); h=np.maximum(0,yy2-yy1+1); inter=w*h; iou=inter/(area[i]+area[order[1:]]-inter+1e-9); order=order[np.where(iou<=t)[0]+1]
    return keep
# CUSTO COMPUTACIONAL: a janela deslizante repete HOG+SVM em muitas posições e escalas.
# YOLO compartilha cálculos convolucionais sobre a imagem inteira e prevê caixas/classes em uma passagem principal,
# portanto costuma ser muito mais eficiente em hardware moderno para detecção geral.
boxes=[];scores=[]; scale_to_orig=1.0; pyr=img0.copy()
while pyr.shape[1]>=64 and pyr.shape[0]>=128:
    g=cv2.cvtColor(pyr,cv2.COLOR_BGR2GRAY)
    for y in range(0,g.shape[0]-127,a.passo):
        for x in range(0,g.shape[1]-63,a.passo):
            roi=g[y:y+128,x:x+64]; feat=hog.compute(roi).reshape(1,-1); prob=float(svm.predict_proba(feat)[0,1])
            if prob>=a.limiar: boxes.append((int(x*scale_to_orig),int(y*scale_to_orig),int((x+64)*scale_to_orig),int((y+128)*scale_to_orig))); scores.append(prob)
    nw=int(pyr.shape[1]/a.escala); nh=int(pyr.shape[0]/a.escala)
    if nw<64 or nh<128: break
    scale_to_orig*=a.escala; pyr=cv2.resize(img0,(nw,nh))
keep=nms(boxes,scores); vis=img0.copy()
for i in keep:
    x1,y1,x2,y2=boxes[i]; cv2.rectangle(vis,(x1,y1),(x2,y2),(0,255,0),2); cv2.putText(vis,f'{scores[i]:.2f}',(x1,max(20,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
out=base/'saidas/deteccoes_janela_deslizante.jpg'; cv2.imwrite(str(out),vis); print('Janelas candidatas:',len(boxes)); print('Detecções após NMS:',len(keep)); print('Imagem salva em:',out)
