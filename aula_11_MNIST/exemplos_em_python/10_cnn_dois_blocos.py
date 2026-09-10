"""
Exemplo 10 - CNN exigida no Item A.

Objetivo:
- Usar pelo menos 2 blocos Conv2D + MaxPooling.
"""

from mnist_utils import build_cnn, load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
model = build_cnn()
model.summary()

history = model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia de teste da CNN: {acc:.4f}")
