import cv2, numpy as np
img=np.zeros((240,320),np.uint8); cv2.circle(img,(160,120),70,255,-1); k=np.ones((5,5),np.float32)/25
cv2.imwrite('08_blur.png',cv2.filter2D(img,-1,k))
