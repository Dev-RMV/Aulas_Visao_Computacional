import cv2, numpy as np
img=np.zeros((240,320),np.uint8); cv2.putText(img,'CNN',(65,145),cv2.FONT_HERSHEY_SIMPLEX,2.6,200,7)
k=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32); cv2.imwrite('09_sharpen.png',cv2.filter2D(img,-1,k))
