from pathlib import Path
import cv2,numpy as np,joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
base=Path(__file__).resolve().parents[1]
hog=cv2.HOGDescriptor(_winSize=(64,128),_blockSize=(16,16),_blockStride=(8,8),_cellSize=(8,8),_nbins=9)
X=[]; y=[]
for pasta,rot in [(base/'dados/processadas/positivas',1),(base/'dados/processadas/negativas',0)]:
    for p in pasta.glob('*'):
        img=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        X.append(hog.compute(cv2.resize(img,(64,128))).reshape(-1)); y.append(rot)
X=np.asarray(X,np.float32); y=np.asarray(y,np.int32)
if len(np.unique(y))<2: raise RuntimeError('É necessário ter amostras positivas e negativas.')
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
svm=SVC(kernel='rbf',C=10.0,gamma='scale',probability=True); svm.fit(Xtr,ytr); pred=svm.predict(Xte)
print('\n=== MÉTRICAS ==='); print(f'Acurácia : {accuracy_score(yte,pred):.4f}'); print(f'Precisão : {precision_score(yte,pred,zero_division=0):.4f}'); print(f'Recall   : {recall_score(yte,pred,zero_division=0):.4f}'); print(classification_report(yte,pred,digits=4,zero_division=0))
joblib.dump(svm,base/'03_hog_svm_atividade/modelo_hog_svm.joblib'); print('Modelo salvo.')
