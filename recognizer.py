from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Function to process the video stream and detect faces
def detect_faces_from_stream(ip_camera_url):
    DeepFace.build_model("Facenet512")
    # Capture video from the IP camera
    video_stream = cv2.VideoCapture(0)

    if not video_stream.isOpened():
        print("Error: Unable to open video stream")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = video_stream.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Use DeepFace to detect faces in the frame
        try:
            # Detect face using DeepFace's Face detector
            result = DeepFace.extract_faces(frame, detector_backend='opencv', enforce_detection=False)
            
            # Draw a rectangle around detected faces
            for face in result:
                x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display the frame with detected faces
            cv2.imshow('IP Camera Face Detection', frame)

        except Exception as e:
            print("Error during face detection:", e)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close windows
    video_stream.release()
    cv2.destroyAllWindows()

def verify_face(frame, base_image):
    return

# Replace with your actual IP camera stream URL
ip_camera_url = "http://<username>:<password>@<ip_address>:<port>/video"
detect_faces_from_stream(ip_camera_url)