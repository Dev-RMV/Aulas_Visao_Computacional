"""
Exemplo 05 - Primeiro MLP simples.

Objetivo:
- Criar e treinar uma rede totalmente conectada.
"""

from mnist_utils import load_mnist
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(x_train, y_train, validation_split=0.1, epochs=2, batch_size=128)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia de teste do MLP mínimo: {acc:.4f}")
