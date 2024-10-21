from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

DeepFace.build_model("Facenet512")
frame = None

# Function to process the video stream and detect faces
def openCamera(parent, ip_camera_url):
    # Capture video from the IP camera
    video_stream = cv2.VideoCapture(ip_camera_url)

    if not video_stream.isOpened():
        QMessageBox.critical(None, "Recognizer", "Error: Unable to open video stream")
        return 

    while True:
        # Capture frame-by-frame
        ret, frame = video_stream.read()

        if not ret:
            QMessageBox.critical(None, "Recognizer", "Failed to grab frame")
            break

        # Use DeepFace to detect faces in the frame
        try:
            # Detect face using DeepFace's Face detector
            result = DeepFace.extract_faces(frame, detector_backend='yolov8', enforce_detection=False)
            
            # Draw a rectangle around detected faces
            for face in result:
                x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                height, width, channel = frame.shape
                bytes_per_line = channel * width
                
                if channel == 3:
                    q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR888)
                else:
                    q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_BGR8888)

                # Set up QLabel to display the image
                parent.m_lbCameraView.setPixmap(QPixmap.fromImage(q_img))
                parent.m_lbCameraView.setAlignment(Qt.AlignCenter)

            # Display the frame with detected faces
            # cv2.imshow('IP Camera Face Detection', frame)

        except Exception as e:
            QMessageBox.critical(None, "Recognizer", f"Error during face detection: {str(e)}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close windows
    video_stream.release()
    cv2.destroyAllWindows()

def verify_face(parent, file):
   # Read the verification image
    base_img = cv2.imread(file)
    if base_img is None:
        QMessageBox.critical(None, "Recognizer", "Error: Unable to read the verification image")
        return 

    cv2.imshow("base_img", frame)
    # Verify the face
    try:
        obj = DeepFace.verify(frame, img2_path=file, model_name="Facenet512", detector_backend="yolov8")
        if obj['verified']:
            QMessageBox.information(parent, "Verification", "The faces match!")
        else:
            QMessageBox.warning(parent, "Verification", "The faces do not match.")
    except Exception as e:
        QMessageBox.critical(None, "Recognizer", f"Error during face verification: {str(e)}")


    return
