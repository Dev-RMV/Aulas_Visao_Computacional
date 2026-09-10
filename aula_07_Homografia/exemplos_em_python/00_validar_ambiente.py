"""Exemplo 00 — confirma versões e disponibilidade dos descritores."""

import cv2
import numpy as np


print(f"OpenCV: {cv2.__version__}")
print(f"NumPy: {np.__version__}")
print(f"SIFT disponível: {hasattr(cv2, 'SIFT_create')}")
print(f"ORB disponível: {hasattr(cv2, 'ORB_create')}")
print(f"AKAZE disponível: {hasattr(cv2, 'AKAZE_create')}")
print(f"findHomography disponível: {hasattr(cv2, 'findHomography')}")

if not all(hasattr(cv2, name) for name in ("SIFT_create", "ORB_create", "AKAZE_create")):
    raise RuntimeError("Instalação do OpenCV incompleta para esta aula.")
