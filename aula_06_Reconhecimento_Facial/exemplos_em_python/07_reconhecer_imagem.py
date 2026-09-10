"""Exemplo 7 — identifica três cadastrados e rejeita a quarta pessoa."""

from __future__ import annotations

import pickle

import cv2
import face_recognition
import numpy as np

from utils import RESOURCES, ensure_outputs, put_label


database_path = ensure_outputs() / "encodings.pkl"
if not database_path.exists():
    raise RuntimeError("Execute primeiro: python 06_cadastrar_identidades.py")
with database_path.open("rb") as f:
    database = pickle.load(f)

frame = cv2.imread(str(RESOURCES / "foto_grupo.jpg"))
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
locations = face_recognition.face_locations(rgb, model="hog")
vectors = face_recognition.face_encodings(rgb, locations)

TOLERANCE = 0.50  # menor = comparação mais rígida = menos falsos aceites
for (top, right, bottom, left), vector in zip(locations, vectors):
    distances = face_recognition.face_distance(database["encodings"], vector)
    best = int(np.argmin(distances))
    name = database["names"][best] if distances[best] <= TOLERANCE else "Desconhecido"
    color = (38, 226, 167) if name != "Desconhecido" else (84, 91, 231)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
    put_label(frame, f"{name} | d={distances[best]:.3f}", (left, max(28, top)), color)
    print(f"{name:<12} melhor distância={distances[best]:.3f}")

output = ensure_outputs() / "07_grupo_reconhecido.jpg"
cv2.imwrite(str(output), frame)
print(f"Saída: {output}")
