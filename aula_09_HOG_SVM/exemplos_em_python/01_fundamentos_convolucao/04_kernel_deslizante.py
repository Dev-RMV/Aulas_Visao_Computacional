import numpy as np
img=np.arange(1,26,dtype=np.float32).reshape(5,5); k=np.array([[1,0,-1],[1,0,-1],[1,0,-1]],np.float32)
out=np.zeros((3,3),np.float32)
for y in range(3):
    for x in range(3): out[y,x]=np.sum(img[y:y+3,x:x+3]*k)
print('Entrada:\n',img); print('Saída:\n',out)
