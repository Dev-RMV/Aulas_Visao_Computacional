"""Exemplo 6 — cadastra três identidades estáticas e salva embeddings 128-D."""

from __future__ import annotations

import pickle

import face_recognition

from utils import RESOURCES, ensure_outputs


KNOWN_NAMES = ["Ana", "Bruno", "Carla"]
encodings = []

for name in KNOWN_NAMES:
    path = RESOURCES / "identidades" / f"{name}.jpg"
    image = face_recognition.load_image_file(str(path))  # RGB
    locations = face_recognition.face_locations(image, model="hog")
    vectors = face_recognition.face_encodings(image, locations)
    if len(vectors) != 1:
        raise RuntimeError(f"Esperado 1 rosto em {path.name}; encontrados {len(vectors)}")
    encodings.append(vectors[0])
    print(f"Cadastrado: {name:<6} | embedding={vectors[0].shape} | {path.name}")

database = {"names": KNOWN_NAMES, "encodings": encodings}
output = ensure_outputs() / "encodings.pkl"
with output.open("wb") as f:
    pickle.dump(database, f)
print(f"Base salva: {output}")
