"""
Exemplo 15 - Visualização de mapas de ativação.

Objetivo:
- Mostrar que a CNN transforma a imagem em mapas de características.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from mnist_utils import build_cnn, load_mnist

(x_train, y_train), _ = load_mnist(flatten=False)
model = build_cnn()
model.fit(x_train, y_train, validation_split=0.1, epochs=2, batch_size=128)

# Modelo intermediário: saída da primeira camada Conv2D.
activation_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[0].output)
sample = x_train[0:1]
feature_maps = activation_model.predict(sample, verbose=0)[0]

plt.figure(figsize=(8, 4))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(feature_maps[:, :, i], cmap="viridis")
    plt.axis("off")
plt.suptitle(f"Mapas de ativação para o dígito {y_train[0]}")
plt.tight_layout()
plt.show()
