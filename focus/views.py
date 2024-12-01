
import cv2
import pygame
from django.http import StreamingHttpResponse,HttpResponseServerError
from django.shortcuts import render
from deepface import DeepFace





# Initialize pygame for sound playback
pygame.mixer.init()
ALERT_SOUND = "alert2.mp3"  # Path to your audio file

def play_alert_sound():
    """Plays an alert sound to grab attention."""
    try:
        pygame.mixer.music.load(ALERT_SOUND)
        pygame.mixer.music.play()
    except pygame.error as e:
        print(f"Error playing sound: {e}")

def eye_aspect_ratio(eye_bbox):
    """Calculate a simple aspect ratio for eye closure detection using bounding box."""
    x, y, w, h = eye_bbox
    ear = h / w  # Height-to-width ratio
    return ear

def detect_focus(request):
    """Detects focus and alerts if the user closes their eyes or looks away."""
    # Load Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Eye aspect ratio threshold and frame counters
    EAR_THRESHOLD = 0.3  # Adjust based on testing
    EAR_CONSEC_FRAMES = 15  # Consecutive frames below threshold to trigger alert
    COUNTER = 0
    focus_timer = 0  # For looking away detection

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to access the camera.")
            break

        # Flip frame for a mirror effect
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            focus_timer = 0  # Reset looking away timer
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Detect eyes within the face region
                face_roi = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

                for (ex, ey, ew, eh) in eyes:
                    # Draw rectangle around the eyes
                    eye_bbox = (ex, ey, ew, eh)
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

                    # Calculate eye aspect ratio
                    ear = eye_aspect_ratio(eye_bbox)

                    # Check if EAR is below the threshold
                    if ear < EAR_THRESHOLD:
                        COUNTER += 1
                        if COUNTER > EAR_CONSEC_FRAMES:
                            play_alert_sound()
                            cv2.putText(frame, "FOCUS!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    else:
                        COUNTER = 0
        else:
            focus_timer += 1
            if focus_timer > 30:  # Approx. 1 second at 30 FPS
                play_alert_sound()
                cv2.putText(frame, "LOOK AT THE SCREEN!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Always display "FOCUS" on the screen
        cv2.putText(frame, "FOCUS", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert frame to JPEG format
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()

        # Return the video feed as an HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()
    cv2.destroyAllWindows()

def video_feed(request):
    return StreamingHttpResponse(detect_focus(request),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

#OTHER ONE

# Helper function to process the video stream
def video_stream():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load OpenCV's Haar Cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop the detected face
            face = frame[y:y + h, x:x + w]

            try:
                # Analyze the face for age and emotion
                analysis = DeepFace.analyze(face, actions=['age', 'emotion'], enforce_detection=False)

                # Handle DeepFace output as a list or dict
                if isinstance(analysis, list):
                    analysis = analysis[0]

                age = analysis['age']
                dominant_emotion = analysis['dominant_emotion']

                # Display the results on the video feed
                text = f"Age: {int(age)}, Emotion: {dominant_emotion}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error during analysis: {e}")

        # Encode the frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the frame as bytes for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()

# View to stream video
def live_feed(request):
    try:
        return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')
    except RuntimeError as e:
        return HttpResponseServerError(str(e))
#HOME
def index(request):
    return render(request, 'index.html')


def ageCheck(request):
    return render(request, 'ageCheck.html')