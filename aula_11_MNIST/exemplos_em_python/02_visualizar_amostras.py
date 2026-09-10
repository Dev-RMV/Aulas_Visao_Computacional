"""
Exemplo 02 - Visualização de amostras do MNIST.

Objetivo:
- Mostrar que cada imagem tem 28x28 pixels em escala de cinza.
"""

import matplotlib.pyplot as plt
import tensorflow as tf

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()

plt.figure(figsize=(8, 4))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"rótulo: {y_train[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
