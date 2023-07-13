import cv2
import numpy as np
from flask import Flask, render_template, request, Response

from deepface import DeepFace

app = Flask(__name__)

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define the home page


@app.route('/')
def home():
    return render_template('index.html')


# Define the recognize function
@app.route('/', methods=['POST'])
def recognize():
    if request.method == 'POST':
        # Get the uploaded image from the request
        image = request.files['image'].read()

        # Convert the image to a NumPy array
        npimg = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        # Analyze each face using DeepFace
        results = []
        for (x, y, w, h) in faces:
            face_cropped = img[y:y+h, x:x+w]
            predictions = DeepFace.analyze(
                face_cropped, actions=['emotion'], enforce_detection=False)

            # Add the dominant emotion to the results list
            dominant_emotion = predictions[0]['dominant_emotion'].upper()
            results.append(dominant_emotion)

        # Render the home page with the results
        return render_template('index.html', results=results)

    # If the request method is not POST, redirect to the home page
    return redirect('/')

# Define the live demo page


@app.route('/live_demo')
def live_demo():
    return render_template('live_demo.html')

# Define the video feed function


def generate_frames():
    # Start video capture
    cap = cv2.VideoCapture(0)
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        # Draw a rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_cropped = frame[y:y+h, x: x+w]

            # Analyze the cropped face using DeepFace
            predictions = DeepFace.analyze(
                face_cropped, actions=['emotion'], enforce_detection=False)

            # Add the dominant emotion to the results table
            dominant_emotion = predictions[0]['dominant_emotion'].upper()

            # Adjust the position and font size of the text
            cv2.putText(frame, dominant_emotion, (x+w+10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Encode the frame as a JPEG image
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the video capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()

# Define the video feed route


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
