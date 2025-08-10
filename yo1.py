from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load YOLOv8 model (small and fast version)
model = YOLO("yolov8n.pt")  # Change to yolov8s.pt or yolov5s.pt for better accuracy

# Path to the image
img_path = "imput.jpg"
# Run detection
results = model(img_path)
# Plot the result with bounding boxes
result_image = results[0].plot()
# Convert BGR to RGB for matplotlib
result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
# Show result
plt.figure(figsize=(10, 6))
plt.imshow(result_image)
plt.axis('off')
plt.title("YOLO Object Detection")
plt.show() 
