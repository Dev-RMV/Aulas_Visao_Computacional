"""Diagnóstico rápido de instalação e dos recursos da aula."""

from pathlib import Path

import cv2
import numpy as np


print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
try:
    import face_recognition
    import dlib
    print(f"face_recognition: OK")
    print(f"dlib: {getattr(dlib, '__version__', 'OK')}")
except Exception as exc:
    print(f"face_recognition/dlib: FALHA — {exc}")

root = Path(__file__).resolve().parents[1]
required = [
    root / "recursos" / "video_teste.mp4",
    root / "recursos" / "foto_grupo.jpg",
    root / "recursos" / "haarcascade_frontalface_default.xml",
]
for path in required:
    print(f"{path.name}: {'OK' if path.exists() else 'AUSENTE'}")
