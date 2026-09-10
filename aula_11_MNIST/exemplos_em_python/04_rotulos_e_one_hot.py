"""
Exemplo 04 - Rótulos inteiros e one-hot encoding.

Objetivo:
- Mostrar duas formas comuns de representar classes.
"""

import tensorflow as tf

(_, y_train), _ = tf.keras.datasets.mnist.load_data()

y_one_hot = tf.keras.utils.to_categorical(y_train[:5], num_classes=10)

print("Rótulos inteiros:", y_train[:5])
print("Rótulos one-hot:")
print(y_one_hot)

# Neste pacote usamos loss='sparse_categorical_crossentropy',
# portanto os rótulos podem ficar como inteiros de 0 a 9.
