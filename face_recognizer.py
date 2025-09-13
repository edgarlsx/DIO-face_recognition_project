import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
import os
import cv2
from utils import preprocess_face

class FaceRecognizer:
    def __init__(self, model_path="facenet_keras.h5"):
        self.model = load_model(model_path)
        self.known_embeddings = []
        self.known_names = []

    def get_embedding(self, face):
        face = preprocess_face(face)
        return self.model.predict(np.expand_dims(face, axis=0))[0]

    def add_known_face(self, face, name):
        emb = self.get_embedding(face)
        self.known_embeddings.append(emb)
        self.known_names.append(name)

    def recognize(self, face):
        emb = self.get_embedding(face)
        if not self.known_embeddings:
            return "Unknown", 0.0
        sims = cosine_similarity([emb], self.known_embeddings)[0]
        max_idx = np.argmax(sims)
        if sims[max_idx] > 0.6:
            return self.known_names[max_idx], sims[max_idx]
        return "Unknown", sims[max_idx]
