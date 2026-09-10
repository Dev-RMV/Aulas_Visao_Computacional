"""
Exemplo 25 - Acurácia nas 10 imagens reais.

Objetivo:
- Calcular desempenho antes de aplicar data augmentation.
"""

import argparse
from pathlib import Path
import tensorflow as tf
from mnist_utils import build_cnn, list_real_digit_images, load_mnist, predict_real_digit

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
if len(samples) < 10:
    print(f"Aviso: foram encontradas {len(samples)} imagens. O ideal é ter 10 amostras, uma por dígito.")

correct = 0
for path, label in samples:
    result = predict_real_digit(model, path)
    pred = result["prediction"]
    ok = pred == label
    correct += int(ok)
    print(f"{path.name}: real={label}, predito={pred}, confiança={result['confidence']:.2%}, correto={ok}")

acc_real = correct / len(samples) if samples else 0.0
print(f"\nAcurácia nas imagens reais: {acc_real:.2%}")
print("Discussão: queda de desempenho pode ocorrer por iluminação, fundo, espessura do traço, escala, rotação e ruído.")
