import cv2
import os

def preprocess_face(face, size=(160, 160)):
    face = cv2.resize(face, size)
    face = face.astype('float32') / 255.0
    return face

def load_known_faces(recognizer, base_dir="known_faces"):
    for name in os.listdir(base_dir):
        person_dir = os.path.join(base_dir, name)
        if not os.path.isdir(person_dir):
            continue
        for filename in os.listdir(person_dir):
            path = os.path.join(person_dir, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            recognizer.add_known_face(img, name)
