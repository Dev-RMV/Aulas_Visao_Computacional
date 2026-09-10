"""
Exemplo 07 - Curvas de treino do MLP.

Objetivo:
- Plotar acurácia e loss de treino vs. validação.
"""

from mnist_utils import build_mlp, load_mnist, plot_history

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True)
model = build_mlp()
model.summary()

history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
plot_history(history, "MLP", "resultados/mlp_curvas.png")

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia de teste do MLP: {acc:.4f}")
