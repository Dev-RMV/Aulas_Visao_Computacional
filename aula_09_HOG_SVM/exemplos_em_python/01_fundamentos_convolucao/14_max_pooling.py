import numpy as np
x=np.array([[1,3,2,0],[2,8,1,4],[5,2,9,1],[0,1,3,7]],dtype=np.float32); out=np.zeros((2,2))
for y in range(2):
    for z in range(2): out[y,z]=x[y*2:y*2+2,z*2:z*2+2].max()
print('Entrada:\n',x); print('MaxPooling:\n',out)
