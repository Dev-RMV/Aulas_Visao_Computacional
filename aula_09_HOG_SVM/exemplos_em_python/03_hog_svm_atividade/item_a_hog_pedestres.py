import argparse,time,cv2,numpy as np
p=argparse.ArgumentParser(); p.add_argument('--video',required=True); a=p.parse_args()
hog=cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cenarios={'rapido':dict(winStride=(12,12),padding=(8,8),scale=1.10),'preciso':dict(winStride=(4,4),padding=(8,8),scale=1.03)}
def roda(nome):
    cap=cv2.VideoCapture(a.video); ts=[]; total=frames=0; c=cenarios[nome]
    while True:
        ok,f=cap.read()
        if not ok: break
        t=time.perf_counter(); rects,w=hog.detectMultiScale(f,winStride=c['winStride'],padding=c['padding'],scale=c['scale']); ms=(time.perf_counter()-t)*1000; ts.append(ms); total+=len(rects); frames+=1
        for x,y,ww,hh in rects: cv2.rectangle(f,(x,y),(x+ww,y+hh),(0,255,0),2)
        cv2.putText(f,f'{nome} det={len(rects)} {ms:.1f} ms',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
        print(f'[{nome}] frame={frames:04d} det={len(rects):2d} inferencia={ms:7.2f} ms')
        cv2.imshow('HOG People Detector',f)
        if cv2.waitKey(1)&0xFF==27: break
    cap.release(); cv2.destroyAllWindows(); media=float(np.mean(ts)) if ts else 0; return nome,frames,total/frames if frames else 0,media,1000/media if media else 0
r=[roda('rapido'),roda('preciso')]
print('\n'+'='*72); print(f"{'Cenario':<12}{'Frames':>10}{'Det/frame':>14}{'ms/frame':>14}{'FPS':>12}"); print('-'*72)
for n,f,d,m,fps in r: print(f'{n:<12}{f:>10d}{d:>14.2f}{m:>14.2f}{fps:>12.2f}')
print('='*72)
