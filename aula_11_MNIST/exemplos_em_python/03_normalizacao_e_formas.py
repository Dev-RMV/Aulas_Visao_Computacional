"""
Exemplo 03 - Normalização e ajuste de formas.

Objetivo:
- Converter pixels de 0..255 para 0..1.
- Preparar formato para MLP e CNN.
"""

import numpy as np
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

x_train_norm = x_train.astype("float32") / 255.0
x_mlp = x_train_norm.reshape((-1, 28 * 28))
x_cnn = x_train_norm[..., np.newaxis]

print("Original:", x_train.shape)
print("Para MLP:", x_mlp.shape, "-> vetor com 784 atributos")
print("Para CNN:", x_cnn.shape, "-> altura, largura e canal")
print("Faixa normalizada:", x_train_norm.min(), x_train_norm.max())
