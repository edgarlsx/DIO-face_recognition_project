from mtcnn import MTCNN
import cv2

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb)
        faces = []
        for res in results:
            x, y, w, h = res['box']
            faces.append((x, y, x+w, y+h))
        return faces
