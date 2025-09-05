import cv2
import numpy as np
img = cv2.imread("f:/python1/New folder/im1.png")
if img is None:
    print("❌ Failed to load image. Check the path.")
    exit()
img = cv2.resize(img, (150,150))
blur_kernel = np.ones((3, 3), np.float32) / 9
blurred = cv2.filter2D(img, -1, blur_kernel)
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, sharpen_kernel)
edge_kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])
edges = cv2.filter2D(img, -1, edge_kernel)
top_row = np.hstack((img, blurred))
bottom_row = np.hstack((sharpened, edges))
combined = np.vstack((top_row, bottom_row))
cv2.imshow("Original | Blurred\nSharpened | Edges", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
