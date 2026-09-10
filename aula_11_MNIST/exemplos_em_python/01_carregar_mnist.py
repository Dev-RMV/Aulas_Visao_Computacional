"""
Exemplo 01 - Carregamento do MNIST.

Objetivo:
- Baixar/carregar o dataset MNIST pelo TensorFlow/Keras.
- Verificar tamanhos de treino e teste.
"""

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Formato x_train:", x_train.shape)
print("Formato y_train:", y_train.shape)
print("Formato x_test:", x_test.shape)
print("Formato y_test:", y_test.shape)
print("Valor mínimo de pixel:", x_train.min())
print("Valor máximo de pixel:", x_train.max())
print("Primeiro rótulo:", y_train[0])
