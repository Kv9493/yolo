from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")  # You can also try 'yolov8s.pt' for better accuracy

# Open the webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Run detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the current frame
    results = model(frame)

    # Plot results on frame (draw boxes, labels)
    annotated_frame = results[0].plot()

    # Display the output
    cv2.imshow("YOLOv8 - Webcam Object Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()

# Note: Make sure you have the necessary libraries installed: