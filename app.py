from flask import Flask, Response
import cv2
from deepface import DeepFace

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def detect_faces():
    cap = cv2.VideoCapture(0)
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        # Draw a rectangle around each face and detect emotions
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_cropped = frame[y:y+h, x:x+w]
            predictions = DeepFace.analyze(
                face_cropped, actions=['emotion'], enforce_detection=False)

            # Adjust the position and font size of the text
            cv2.putText(frame, predictions[0]['dominant_emotion'], (
                x+w+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Convert the processed frame to a byte array
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Yield the frame as a byte array to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture and destroy the window
    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def video_feed():
    return Response(detect_faces(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
