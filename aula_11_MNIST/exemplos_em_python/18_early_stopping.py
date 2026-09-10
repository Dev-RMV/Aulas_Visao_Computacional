"""
Exemplo 18 - EarlyStopping.

Objetivo:
- Parar o treino quando a validação deixa de melhorar.
"""

import tensorflow as tf
from mnist_utils import build_cnn, load_mnist, plot_history

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
model = build_cnn()

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
)

history = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=128,
    callbacks=[callback],
)
plot_history(history, "CNN com EarlyStopping", "resultados/cnn_early_stopping.png")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia de teste: {acc:.4f}")
print("Épocas executadas:", len(history.history["loss"]))
