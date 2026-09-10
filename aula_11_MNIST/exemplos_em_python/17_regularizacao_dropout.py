"""
Exemplo 17 - Regularização com Dropout.

Objetivo:
- Reduzir overfitting desligando neurônios durante o treino.
"""

from mnist_utils import build_cnn, load_mnist, plot_history

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
model = build_cnn(dropout=0.4)
model.summary()

history = model.fit(x_train, y_train, validation_split=0.1, epochs=5, batch_size=128)
plot_history(history, "CNN com Dropout", "resultados/cnn_dropout_curvas.png")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia de teste com Dropout: {acc:.4f}")
