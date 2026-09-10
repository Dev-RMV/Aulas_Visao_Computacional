import cv2, numpy as np
img=np.zeros((200,300),np.uint8); img[100:,:]=255
k=np.array([[-1,-1,-1],[0,0,0],[1,1,1]],np.float32)
r=cv2.filter2D(img,cv2.CV_32F,k); cv2.imwrite('06_borda_horizontal.png',cv2.convertScaleAbs(r))
