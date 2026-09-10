"""
Exemplo 09 - Primeira CNN simples.

Objetivo:
- Mostrar Conv2D, MaxPooling2D, Flatten e Dense.
"""

from mnist_utils import load_mnist
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(x_train, y_train, validation_split=0.1, epochs=2, batch_size=128)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia de teste da CNN mínima: {acc:.4f}")
