"""
Exemplo 20 - Visualização dos erros.

Objetivo:
- Mostrar imagens nas quais a CNN errou.
"""

import matplotlib.pyplot as plt
import numpy as np
from mnist_utils import build_cnn, load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
model = build_cnn()
model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)

pred = model.predict(x_test, verbose=0).argmax(axis=1)
wrong = np.where(pred != y_test)[0]
print("Total de erros:", len(wrong))

plt.figure(figsize=(8, 4))
for i, idx in enumerate(wrong[:12]):
    plt.subplot(3, 4, i + 1)
    plt.imshow(x_test[idx].squeeze(), cmap="gray")
    plt.title(f"real={y_test[idx]}, pred={pred[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
