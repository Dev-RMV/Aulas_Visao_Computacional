"""
Exemplo 13 - Salvando e carregando a CNN.

Objetivo:
- Evitar retreinar o modelo em todos os exemplos do Item B.
"""

from pathlib import Path
import tensorflow as tf
from mnist_utils import build_cnn, load_mnist

Path("resultados").mkdir(exist_ok=True)
model_path = "resultados/cnn_mnist.keras"

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
model = build_cnn()
model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
model.save(model_path)
print("Modelo salvo em:", model_path)

loaded = tf.keras.models.load_model(model_path)
loss, acc = loaded.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia do modelo carregado: {acc:.4f}")
