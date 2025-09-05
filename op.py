import cv2
import numpy as np  
img1=cv2.imread("f:/python1/New folder/im3.png", cv2.IMREAD_COLOR)
img2=cv2.imread("f:/python1/New folder/im4.png", cv2.IMREAD_COLOR)
dest=cv2.bitwise_not(img1,img2,mask=None)
cv2.imshow("image", dest)     
cv2.waitKey(0)      
cv2.destroyAllWindows()
