"""
Exemplo 27 - Treino da CNN com ImageDataGenerator.

Objetivo:
- Aplicar rotação ±15° e zoom ±10% no treino.
"""

from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mnist_utils import build_cnn, load_mnist, plot_history

Path("resultados").mkdir(exist_ok=True)
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)

# Augmentation exigida no enunciado.
datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.10, validation_split=0.1)
train_gen = datagen.flow(x_train, y_train, batch_size=128, subset="training")
val_gen = datagen.flow(x_train, y_train, batch_size=128, subset="validation")

model = build_cnn()
model.summary()
history = model.fit(train_gen, validation_data=val_gen, epochs=5)
plot_history(history, "CNN com augmentation", "resultados/cnn_aug_curvas.png")

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Acurácia no teste MNIST limpo: {acc:.4f}")
model.save("resultados/cnn_mnist_aug.keras")
print("Modelo com augmentation salvo em resultados/cnn_mnist_aug.keras")
