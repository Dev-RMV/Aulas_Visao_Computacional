"""
Exemplo 24 - Inferência em uma pasta com dígitos reais.

Objetivo:
- Ler arquivos 0.jpg, 1.jpg, ..., 9.jpg e exibir predições.
"""

import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from mnist_utils import annotate_prediction, build_cnn, list_real_digit_images, load_mnist, predict_real_digit

parser = argparse.ArgumentParser()
parser.add_argument("--pasta", default="data/real_digits")
parser.add_argument("--modelo", default="resultados/cnn_mnist.keras")
args = parser.parse_args()

if Path(args.modelo).exists():
    model = tf.keras.models.load_model(args.modelo)
else:
    (x_train, y_train), _ = load_mnist(flatten=False)
    model = build_cnn()
    model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)
    Path(args.modelo).parent.mkdir(exist_ok=True)
    model.save(args.modelo)

samples = list_real_digit_images(args.pasta)
if not samples:
    raise SystemExit("Nenhuma imagem encontrada. Salve 0.jpg ... 9.jpg em data/real_digits.")

plt.figure(figsize=(10, 5))
for i, (path, label) in enumerate(samples):
    result = predict_real_digit(model, path)
    annotated = annotate_prediction(result["original"], result["prediction"], result["confidence"])
    plt.subplot(2, 5, i + 1)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title(f"real={label}")
    plt.axis("off")
plt.tight_layout()
plt.show()
