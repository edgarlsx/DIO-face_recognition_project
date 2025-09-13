import cv2
import pandas as pd
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from utils import load_known_faces
from datetime import datetime

def main():
    # Inicializar
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    load_known_faces(recognizer)

    # Logging
    log_df = pd.DataFrame(columns=["timestamp", "name", "confidence"])

    # Webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        for (x1, y1, x2, y2) in faces:
            face_img = frame[y1:y2, x1:x2]
            name, conf = recognizer.recognize(face_img)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Log
            log_df = pd.concat([log_df, pd.DataFrame([[timestamp, name, conf]], columns=log_df.columns)])

            # Mostrar
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Salvar log
    log_df.to_csv("log.csv", index=False)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
