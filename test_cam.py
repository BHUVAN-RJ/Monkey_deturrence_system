import cv2
import numpy as np
from yolov5 import detect
import torch

model = torch.hub.load('BHUVAN-RJ/yolov5-v7','custom','/home/jetson/Downloads/best.pt')
detector = model.detect
pipeline = cv2.VideoCapture('rtsp://192.168.1.5:1935/')

while True:

	ret, frame = pipeline.read()

	if not ret:
		break

	detections = model(frame)
	print(detections)

	'''if len(detections) > 0:
		for detection in detections:

			bbox = detection['bbox']
			label = detection['class']

			cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
			cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)'''

	cv2.imshow('Frame', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

pipeline.release()

cv2.destroyAllWindows()

