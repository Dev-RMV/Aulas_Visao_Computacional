"""
Exemplo 28 - Comparar acurácia real antes e depois do augmentation.

Objetivo:
- Avaliar se augmentation reduziu o domain gap.
"""

import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mnist_utils import build_cnn, list_real_digit_images, load_mnist, predict_real_digit

parser = argparse.ArgumentParser()
parser.add_argument("--pasta", default="data/real_digits")
args = parser.parse_args()


def ensure_models():
    (x_train, y_train), _ = load_mnist(flatten=False)

    clean_path = Path("resultados/cnn_mnist.keras")
    aug_path = Path("resultados/cnn_mnist_aug.keras")
    clean_path.parent.mkdir(exist_ok=True)

    if not clean_path.exists():
        clean = build_cnn()
        clean.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
        clean.save(clean_path)

    if not aug_path.exists():
        datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.10, validation_split=0.1)
        aug = build_cnn()
        aug.fit(
            datagen.flow(x_train, y_train, batch_size=128, subset="training"),
            validation_data=datagen.flow(x_train, y_train, batch_size=128, subset="validation"),
            epochs=3,
        )
        aug.save(aug_path)

    return tf.keras.models.load_model(clean_path), tf.keras.models.load_model(aug_path)


def real_accuracy(model, samples):
    correct = 0
    for path, label in samples:
        pred = predict_real_digit(model, path)["prediction"]
        correct += int(pred == label)
    return correct / len(samples) if samples else 0.0

samples = list_real_digit_images(args.pasta)
if not samples:
    raise SystemExit("Nenhuma imagem real encontrada. Salve 0.jpg ... 9.jpg em data/real_digits.")

clean_model, aug_model = ensure_models()
acc_clean = real_accuracy(clean_model, samples)
acc_aug = real_accuracy(aug_model, samples)

print(f"Acurácia real antes do augmentation:  {acc_clean:.2%}")
print(f"Acurácia real após augmentation:      {acc_aug:.2%}")
print("Interpretação: augmentation aproxima o treino de variações reais, mas não resolve todo o domain gap.")
