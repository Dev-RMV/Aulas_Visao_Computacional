"""
Exemplo 11 - Curvas de treino da CNN.

Objetivo:
- Plotar treino vs. validação e procurar indícios de overfitting.
"""

from mnist_utils import build_cnn, load_mnist, plot_history

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
model = build_cnn()
model.summary()

history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
plot_history(history, "CNN", "resultados/cnn_curvas.png")

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia de teste da CNN: {acc:.4f}")
print("Indício de overfitting: treino melhora continuamente, mas validação para de melhorar ou piora.")
