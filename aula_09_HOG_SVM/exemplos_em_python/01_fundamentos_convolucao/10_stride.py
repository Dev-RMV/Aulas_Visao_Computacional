import numpy as np
img=np.arange(1,50,dtype=np.float32).reshape(7,7); k=np.ones((3,3),np.float32)
def conv(s):
    oh=(7-3)//s+1; out=np.zeros((oh,oh))
    for y in range(oh):
        for x in range(oh): out[y,x]=np.sum(img[y*s:y*s+3,x*s:x*s+3]*k)
    return out
for s in (1,2): print('Stride',s,'\n',conv(s))
