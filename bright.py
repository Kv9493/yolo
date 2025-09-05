import cv2
import numpy as np

# Load your image (use your path if needed)
img = cv2.imread("f:/python1/New folder/im.png")

# Check if image loaded successfully
if img is None:
    print("❌ Error: Image not found. Check the file path.")
    exit()

# beta > 0 = increase brightness
bright_img = cv2.convertScaleAbs(img, alpha=1.5, beta=40)

# Stack original and processed image horizontally for comparison
combined = np.hstack((img, bright_img))

# Show the result
cv2.imshow("Original vs Brightened", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
