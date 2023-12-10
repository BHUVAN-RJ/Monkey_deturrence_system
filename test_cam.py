import cv2
import numpy as np
from yolov5 import detect
import torch

# Load the pre-trained YOLOv5 model.
model = torch.hub.load('BHUVAN-RJ/yolov5-v7','custom','/home/jetson/Downloads/best.pt')
detector = model.detect
# Create a GStreamer pipeline to decode the RTSP stream and send it to the YOLOv5 model.
pipeline = cv2.VideoCapture('rtsp://192.168.1.5:1935/')

# Procss each frame from the GStreamer pipeline.
while True:

    # Capture the next frame from the GStreamer pipeline.
	ret, frame = pipeline.read()

    # If the frame is empty, break out of the loop.
	if not ret:
		break

    # Run the YOLOv5 model on the frame.
	detections = model(frame)
	print(detections)

    # Draw bounding boxes around the detected objects.
	'''if len(detections) > 0:
		for detection in detections:

			bbox = detection['bbox']
			label = detection['class']

			cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
			cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)'''

    # Display the frame.
	cv2.imshow('Frame', frame)

    # If the user presses the 'q' key, break out of the loop.
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the GStreamer pipeline.
pipeline.release()

# Close all windows.
cv2.destroyAllWindows()

