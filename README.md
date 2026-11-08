# Monkey Deterrence System

Real-time monkey detection for coffee plantations. A custom YOLOv5 detector runs on a Jetson Nano over RTSP streams from the plantation's existing security cameras. On detection it plays a randomly chosen deterrent sound (tiger roars, gunfire) through speakers around the estate and sends an SMS alert to the farmer via Twilio.

Deployed and ran headless on a working coffee plantation in Karnataka that was losing crop to monkey raids.

## Setup

Create a new env and use the YoloV5 that is in this repo, or use https://github.com/BHUVAN-RJ/yolov5-v7. This is the only supported Yolo arch for Jetson Nano deployment (JetPack < 5.0, Python 3.6.9).

Install `requirements.txt` and run `pipeline.py`.
