"""Exemplo 00 — verifica o ambiente antes da aula."""

from __future__ import annotations

import platform

import cv2
import matplotlib
import numpy as np
import sklearn
import tensorflow as tf


print("=== AMBIENTE CNN / VISÃO COMPUTACIONAL ===")
print("Python:", platform.python_version())
print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__)
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("Matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)
print("GPUs visíveis:", tf.config.list_physical_devices("GPU"))
print("Execução em CPU é suportada; uma GPU apenas reduz o tempo de treino.")

