"""Exemplo 10 — mostra como o limiar troca falso aceite por falso rejeite."""

from __future__ import annotations

import pickle

import face_recognition
import numpy as np

from utils import RESOURCES, ensure_outputs


with (ensure_outputs() / "encodings.pkl").open("rb") as f:
    database = pickle.load(f)

image = face_recognition.load_image_file(str(RESOURCES / "foto_grupo.jpg"))
locations = face_recognition.face_locations(image, model="hog")
vectors = face_recognition.face_encodings(image, locations)

print("Distância menor indica maior similaridade; tolerância menor é mais específica.\n")
for tolerance in [0.40, 0.50, 0.60]:
    labels = []
    for vector in vectors:
        distances = face_recognition.face_distance(database["encodings"], vector)
        best = int(np.argmin(distances))
        label = database["names"][best] if distances[best] <= tolerance else "Desconhecido"
        labels.append(f"{label}(d={distances[best]:.3f})")
    print(f"tolerância={tolerance:.2f}: " + ", ".join(labels))

print("\nNunca escolha o limiar só pela demo: use validação separada e custo do erro da aplicação.")
