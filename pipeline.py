import sys
import torch
from PIL import Image
import cv2

sys.path.append('/home/jetson/yolov5')

model = torch.hub.load('BHUVAN-RJ/yolov5-v7','custom','/home/jetson/Downloads/best.pt')
model.conf=0.5



src = ('rtspsrc location=rtsp://192.168.1.5:1935/ latency=200 ! application/x-rtp,encoding-name=H265 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1')
pipeline = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)

while True:

    ret, frame = pipeline.read()

    if not ret:
        break

    image = Image.fromarray(frame)

    

    results = model(image)

    monk = results.pred[0]
    confidence = monk[:,4].cpu().numpy()
    detections = monk[confidence >0.5]
    if len(detections) >0:
        print("monkey detected")



pipeline.release()

cv2.destroyAllWindows()

