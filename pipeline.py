import sys
import torch
from PIL import Image
import cv2

# Add the local `yolov5` folder to the system path.
sys.path.append('/home/jetson/yolov5')

# Load the YOLOv5 model.
model = torch.hub.load('BHUVAN-RJ/yolov5-v7','custom','/home/jetson/Downloads/best.pt')
model.conf=0.5

# Create a detector object.


# Create a GStreamer pipeline to decode the RTSP stream and send it to the detector.
src = ('rtspsrc location=rtsp://192.168.1.5:1935/ latency=200 ! application/x-rtp,encoding-name=H265 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1')
pipeline = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)

# Process each frame from the GStreamer pipeline.
while True:

    # Capture the next frame from the GStreamer pipeline.
    ret, frame = pipeline.read()

    # If the frame is empty, break out of the loop.
    if not ret:
        break

    # Convert the frame to a PIL image.
    image = Image.fromarray(frame)

    # Run the detector on the image.
    

    # Filter the detections with a confidence threshold of 0.5.
    results = model(image)

    # Count the number of detected monkeys.
    monk = results.pred[0]
    confidence = monk[:,4].cpu().numpy()
    detections = monk[confidence >0.5]
    if len(detections) >0:
        print("monkey detected")

    # Print the results.


# Release the GStreamer pipeline.
pipeline.release()

# Close all windows.
cv2.destroyAllWindows()

