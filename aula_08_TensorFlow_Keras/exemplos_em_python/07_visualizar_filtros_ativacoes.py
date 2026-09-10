"""Exemplo 07 — visualiza filtros aprendidos e mapas de ativação."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from utils import OUTPUTS, PROJECT, ensure_dirs, load_rgb, load_trained_model, read_selection


ensure_dirs()
model = load_trained_model()
selection = read_selection()
rgb = load_rgb(PROJECT / selection[0]["path"])
batch = rgb[None, ...]

weights = model.get_layer("conv1a").get_weights()[0]  # 3×3×3×16
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for index, ax in enumerate(axes.ravel()):
    kernel = weights[:, :, :, index].mean(axis=2)
    ax.imshow(kernel, cmap="coolwarm")
    ax.set_title(f"filtro {index + 1}"); ax.axis("off")
fig.suptitle("Filtros aprendidos na primeira Conv2D")
fig.tight_layout(); fig.savefig(OUTPUTS / "07_filtros_conv1.png", dpi=180)
plt.close(fig)

activation_model = keras.Model(model.input, model.get_layer("conv2a").output)
activations = activation_model.predict(batch, verbose=0)[0]
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for index, ax in enumerate(axes.ravel()):
    ax.imshow(activations[:, :, index], cmap="viridis")
    ax.set_title(f"mapa {index + 1}"); ax.axis("off")
fig.suptitle("Mapas de ativação no segundo bloco")
fig.tight_layout(); fig.savefig(OUTPUTS / "07_mapas_ativacao.png", dpi=180)
plt.close(fig)

print("Filtros:", OUTPUTS / "07_filtros_conv1.png")
print("Ativações:", OUTPUTS / "07_mapas_ativacao.png")

