"""
Exemplo 00 - Verificação do ambiente.

Objetivo:
- Confirmar se TensorFlow, OpenCV, NumPy e Matplotlib estão instalados.
"""

import cv2
import matplotlib
import numpy as np
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Keras:", tf.keras.__version__ if hasattr(tf.keras, "__version__") else "integrado ao TensorFlow")
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("GPU disponível:", tf.config.list_physical_devices("GPU"))
