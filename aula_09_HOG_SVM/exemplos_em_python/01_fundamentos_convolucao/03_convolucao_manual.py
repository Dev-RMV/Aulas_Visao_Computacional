import numpy as np
img=np.array([[10,10,100],[10,10,100],[10,10,100]],dtype=np.float32)
k=np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=np.float32)
print('Imagem:\n',img); print('Kernel:\n',k); print('Produto:\n',img*k); print('Soma:',np.sum(img*k))
