"""
Exemplo 26 - Visualização de data augmentation.

Objetivo:
- Mostrar rotações e zooms gerados artificialmente.
"""

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mnist_utils import load_mnist

(x_train, y_train), _ = load_mnist(flatten=False)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.10)

sample = x_train[0:1]
label = y_train[0]

plt.figure(figsize=(8, 4))
plt.subplot(2, 5, 1)
plt.imshow(sample[0].squeeze(), cmap="gray")
plt.title(f"Original {label}")
plt.axis("off")

for i, batch in enumerate(datagen.flow(sample, batch_size=1)):
    plt.subplot(2, 5, i + 2)
    plt.imshow(batch[0].squeeze(), cmap="gray")
    plt.title("Aug")
    plt.axis("off")
    if i == 8:
        break

plt.tight_layout()
plt.show()
