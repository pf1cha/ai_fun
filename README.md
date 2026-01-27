# CarsDetection

CarsDetection is a Python project, which detects and counts number of cars that have crossed the line.

# Automatic Car Tracking Using YOLOv9 and ByteTrack

This project uses the YOLOv9 model for car detection and tracking in video, along with the `supervision` library for annotating and tracking objects in real time.

## Description

The program performs the following tasks:
1. **Car Detection**: Utilizes the YOLOv9 model to detect cars (class `class_id == 2`).
2. **Object Tracking**: Applies the ByteTrack algorithm for tracking cars across frames.
3. **Annotation**: Adds annotations in the form of bounding boxes around cars, displaying the tracker ID, object class, and model confidence score.
4. **Car Count**: Implements a crossing line to track how many cars pass through a designated line on the video.

## Requirements


## Demonstration
You can view a sample of the output in the video below:

![Example](showcase.mp4)

The video demonstrates car tracking with annotations and the crossing line for counting vehicles.
