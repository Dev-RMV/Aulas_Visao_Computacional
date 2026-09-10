"""Exemplo 11 — classifica uma imagem e imprime as probabilidades softmax."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from utils import CLASS_NAMES, OUTPUTS, PROJECT, ensure_dirs, load_rgb, load_trained_model, read_selection


parser = argparse.ArgumentParser()
parser.add_argument("--imagem", default="", help="caminho para uma imagem PNG/JPG")
args = parser.parse_args()
ensure_dirs()
default_path = PROJECT / read_selection()[0]["path"]
path = PROJECT / args.imagem if args.imagem and not args.imagem.startswith("/") else args.imagem or default_path
path = default_path if path == "" else path
rgb = load_rgb(path)
model = load_trained_model()
probabilities = model.predict(rgb[None, ...], verbose=0)[0]
prediction = int(np.argmax(probabilities))

print("Imagem:", path)
for class_name, probability in zip(CLASS_NAMES, probabilities):
    print(f"  {class_name:<10}: {100*probability:6.2f}%")
print("Predição:", CLASS_NAMES[prediction])

bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
visual = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_NEAREST)
label = f"{CLASS_NAMES[prediction]} | {100*probabilities[prediction]:.1f}%"
cv2.rectangle(visual, (0, 0), (512, 58), (20, 20, 20), -1)
cv2.putText(visual, label, (16, 39), cv2.FONT_HERSHEY_SIMPLEX, 1.05,
            (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(str(OUTPUTS / "11_predicao.png"), visual)
print("Saída:", OUTPUTS / "11_predicao.png")

