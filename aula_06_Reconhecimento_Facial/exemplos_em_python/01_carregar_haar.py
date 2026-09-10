"""Exemplo 1 — carregar o XML e verificar se o classificador está pronto."""

from pathlib import Path

import cv2


xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(str(xml))

print(f"OpenCV: {cv2.__version__}")
print(f"XML: {xml}")
print(f"Classificador carregado: {not detector.empty()}")

if detector.empty():
    raise RuntimeError("O Haar Cascade não foi carregado. Verifique a instalação do OpenCV.")
