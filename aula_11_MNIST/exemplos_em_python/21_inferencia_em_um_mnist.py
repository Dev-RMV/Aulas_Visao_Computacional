"""
Exemplo 21 - Inferência em uma amostra do MNIST.

Objetivo:
- Passar uma única imagem pela CNN e ler a classe predita.
"""

import matplotlib.pyplot as plt
import numpy as np
from mnist_utils import build_cnn, load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
model = build_cnn()
model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)

idx = 0
probs = model.predict(x_test[idx:idx+1], verbose=0)[0]
pred = int(np.argmax(probs))

plt.imshow(x_test[idx].squeeze(), cmap="gray")
plt.title(f"Real: {y_test[idx]} | Predito: {pred} | Confiança: {probs[pred]:.1%}")
plt.axis("off")
plt.show()
