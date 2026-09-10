"""Exemplo 2 — detectar um rosto em imagem estática e salvar a visualização."""

from pathlib import Path

import cv2

from utils import RESOURCES, ensure_outputs, load_haar, put_label


image_path = RESOURCES / "identidades" / "Ana.jpg"
frame = cv2.imread(str(image_path))
if frame is None:
    raise FileNotFoundError(image_path)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
detector = load_haar()

faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.10,
    minNeighbors=5,
    minSize=(50, 50),
)

for i, (x, y, w, h) in enumerate(faces, start=1):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (38, 226, 167), 3)
    put_label(frame, f"rosto {i}", (x, max(28, y)))

output = ensure_outputs() / "02_deteccao_imagem.jpg"
cv2.imwrite(str(output), frame)
print(f"Rostos detectados: {len(faces)}")
print(f"Saída: {output}")
