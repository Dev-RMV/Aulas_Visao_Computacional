"""
Exemplo 06 - MLP exigido no Item A.

Objetivo:
- Usar ao menos duas camadas densas antes da saída.
"""

from mnist_utils import build_mlp, load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True)
model = build_mlp()
model.summary()

history = model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia de teste do MLP: {acc:.4f}")
