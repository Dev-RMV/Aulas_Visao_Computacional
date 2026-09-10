import cv2, numpy as np
img=np.zeros((240,320),np.uint8); cv2.rectangle(img,(80,60),(240,180),255,-1)
gx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
cv2.imwrite('07_sobel_x.png',cv2.convertScaleAbs(gx)); cv2.imwrite('07_sobel_y.png',cv2.convertScaleAbs(gy)); print('Sobel calcula gradientes, base conceitual importante para HOG.')
