import cv2, numpy as np
img=np.zeros((200,300),np.uint8); img[:,150:]=255
k=np.array([[-1,0,1],[-1,0,1],[-1,0,1]],np.float32)
r=cv2.filter2D(img,cv2.CV_32F,k); cv2.imwrite('05_borda_vertical.png',cv2.convertScaleAbs(r))
