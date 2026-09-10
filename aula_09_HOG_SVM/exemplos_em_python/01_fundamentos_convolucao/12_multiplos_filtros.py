import cv2, numpy as np
img=np.zeros((200,300),np.uint8); cv2.rectangle(img,(70,50),(230,150),255,-1)
ks={'vertical':np.array([[-1,0,1]]*3,np.float32),'horizontal':np.array([[-1,-1,-1],[0,0,0],[1,1,1]],np.float32),'laplaciano':np.array([[0,1,0],[1,-4,1],[0,1,0]],np.float32)}
for n,k in ks.items(): cv2.imwrite(f'12_{n}.png',cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_32F,k)))
print('Cada filtro gera um feature map diferente.')
