import sys
import torch
from PIL import Image
import cv2
import pandas
import time
import random

from pydub import AudioSegment
from pydub.playback import play

sound1 = AudioSegment.from_file('/home/jetson/Downloads/audio/alarm.wav', format='wav')
sound2 = AudioSegment.from_file('/home/jetson/Downloads/audio/gun.wav', format='wav')
sound3 = AudioSegment.from_file('/home/jetson/Downloads/audio/firecrackers.wav', format='wav')
sound4 = AudioSegment.from_file('/home/jetson/Downloads/audio/tiger.wav', format='wav')

sound_files = [sound1, sound2, sound3, sound4]
# Add the local `yolov5` folder to the system path
sys.path.append('/home/jetson/yolov5')

# Load the YOLOv5 model.
model = torch.hub.load('BHUVAN-RJ/yolov5-v7','custom','/home/jetson/Downloads/best.pt')


# Create a detector object.


# Create a GStreamer pipeline to decode the RTSP stream and send it to the detector.
pipeline = cv2.VideoCapture('rtsp://192.168.1.5:1935/')

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
    #print("image read")

    # Filter the detections with a confidence threshold of 0.5.
    results = model(image)

    # Count the number of detected monkeys.
    monk = results.pred[0]
    confidence = monk[:,4].cpu().numpy()
    #print(confidence)
    #detections = monk[confidence >0.5]
    if len(confidence) >0:
        print("monkey detected")
        random_sound_file = random.choice(sound_files)
        play(random_sound_file)

    # Print the results.


# Release the GStreamer pipeline.
pipeline.release()

# Close all windows.
cv2.destroyAllWindows()

