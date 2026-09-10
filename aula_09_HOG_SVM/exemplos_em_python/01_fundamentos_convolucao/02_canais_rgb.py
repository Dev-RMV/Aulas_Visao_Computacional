import cv2, numpy as np
img=np.zeros((200,300,3),dtype=np.uint8); img[:,:100]=(255,0,0); img[:,100:200]=(0,255,0); img[:,200:] = (0,0,255)
b,g,r=cv2.split(img); print('Imagem:',img.shape,'Canal:',b.shape)
for n,a in [('original',img),('b',b),('g',g),('r',r)]: cv2.imwrite(f'02_{n}.png',a)
