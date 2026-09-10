"""
Exemplo 22 - Pré-processamento OpenCV para imagem real.

Objetivo:
- Aplicar escala de cinza, Otsu, resize 28x28 e normalização.

Uso:
python 22_opencv_preprocessar_digito_real.py --imagem data/real_digits/5.jpg
"""

import argparse
import matplotlib.pyplot as plt
from mnist_utils import preprocess_digit_image

parser = argparse.ArgumentParser()
parser.add_argument("--imagem", default="data/real_digits/5.jpg")
args = parser.parse_args()

original, gray, binary, resized, model_input = preprocess_digit_image(args.imagem)

print("Entrada para o modelo:", model_input.shape)
print("Valor mínimo:", model_input.min(), "valor máximo:", model_input.max())

plt.figure(figsize=(10, 3))
for i, (title, img, cmap) in enumerate([
    ("Original", original[:, :, ::-1], None),
    ("Cinza", gray, "gray"),
    ("Otsu", binary, "gray"),
    ("28x28", resized, "gray"),
]):
    plt.subplot(1, 4, i + 1)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
plt.tight_layout()
plt.show()
