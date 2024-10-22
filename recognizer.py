from deepface import DeepFace
import cv2
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal

class VideoCaptureThread(QThread):
    frameCaptured = pyqtSignal(object)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.running = True

    def run(self):
        video_stream = cv2.VideoCapture(self.camera)

        if not video_stream.isOpened():
            QMessageBox.critical(None, "Recognizer", "Error: Unable to open video stream")
            return 

        while self.running:
            ret, frame = video_stream.read()
            if ret:
                # Use DeepFace to detect faces in the frame
                try:
                    # Detect face using DeepFace's Face detector
                    result = DeepFace.extract_faces(frame, detector_backend='yolov8', enforce_detection=False)
                    
                    # Draw a rectangle around detected faces
                    for face in result:
                        x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                except Exception as e:
                    QMessageBox.critical(None, "Recognizer", f"Error during face detection: {str(e)}")
                    break

                self.frameCaptured.emit(frame)
            
        video_stream.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

DeepFace.build_model("Facenet512")

def verify_face(frame, file):
   # Read the verification image
    base_img = cv2.imread(file)

    if base_img is None:
        return 2    # Verify the face

    try:
        obj = DeepFace.verify(frame, img2_path=file, model_name="Facenet512", detector_backend="yolov8")
        if obj['verified']:
            return 0
        else:
            return 1
    except Exception as e:
        return e