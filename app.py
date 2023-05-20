from flask import Flask, render_template, request
import cv2
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['image']

        # Read the image file as an array
        image = cv2.imdecode(np.fromstring(
            file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Detect faces in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        # Initialize the results table
        results = []

        # Analyze each face in the image
        for (x, y, w, h) in faces:
            # Crop the face region
            face_cropped = image[y:y+h, x:x+w]

            # Analyze the cropped face using DeepFace
            predictions = DeepFace.analyze(
                face_cropped, actions=['emotion'], enforce_detection=False)

            # Add the dominant emotion to the results table
            dominant_emotion = predictions[0]['dominant_emotion'].upper()
            results.append(dominant_emotion)

        # Render the results page
        return render_template('index.html', results=results)

    # Render the home page
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

