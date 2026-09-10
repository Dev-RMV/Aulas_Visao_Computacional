"""
Exemplo 14 - Visualização dos filtros aprendidos.

Objetivo:
- Observar pesos da primeira camada convolucional.
"""

import matplotlib.pyplot as plt
import numpy as np
from mnist_utils import build_cnn, load_mnist

(x_train, y_train), _ = load_mnist(flatten=False)
model = build_cnn()
model.fit(x_train, y_train, validation_split=0.1, epochs=2, batch_size=128)

conv_layer = model.layers[0]
weights, biases = conv_layer.get_weights()
# weights: altura, largura, canais_entrada, filtros
filters = weights[:, :, 0, :]

plt.figure(figsize=(8, 4))
for i in range(min(16, filters.shape[-1])):
    f = filters[:, :, i]
    f = (f - f.min()) / (f.max() - f.min() + 1e-8)
    plt.subplot(4, 4, i + 1)
    plt.imshow(f, cmap="gray")
    plt.axis("off")
plt.suptitle("Filtros da primeira Conv2D")
plt.tight_layout()
plt.show()
