from _bootstrap import PROJECT_ROOT  # prepara imports e caminhos do projeto
"""
Exemplo 01 - Verificar ambiente.
Objetivo: confirmar OpenCV, TensorFlow, NumPy e Matplotlib.
"""
import cv2, numpy as np, matplotlib
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    print("GPU detectada:", tf.config.list_physical_devices('GPU'))
except Exception as e:
    print("TensorFlow nao carregou:", e)
