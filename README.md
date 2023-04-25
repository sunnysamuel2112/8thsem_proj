```
# Emotion Recognition using DeepFace and Haar Cascade

This project uses DeepFace and Haar Cascade to recognize emotions in real-time video.

## Requirements

- OpenCV
- DeepFace

## Usage

1. Install the required libraries.
2. Load the Haar Cascade classifier by specifying the path to the `haarcascade_frontalface_default.xml` file.
3. Start video capture using OpenCV's `cv2.VideoCapture` function.
4. In a loop, read frames from the video capture and convert them to grayscale.
5. Use the Haar Cascade classifier to detect faces in the grayscale frame.
6. For each detected face, draw a rectangle around it and use DeepFace to analyze the emotions of the cropped face.
7. Display the frame with the detected faces and their emotions.
8. Press 'q' to stop the video capture and destroy the window.
