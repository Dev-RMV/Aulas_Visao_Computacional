"""Exemplo 06 — matriz de confusão, relatório e previsões do conjunto de teste."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils import CLASS_NAMES, OUTPUTS, configure, ensure_dirs, load_datasets, load_trained_model


configure(); ensure_dirs()
_, _, test_ds = load_datasets(batch_size=64)
model = load_trained_model()
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
probabilities = model.predict(test_ds, verbose=0)
y_pred = probabilities.argmax(axis=1)
matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

fig, ax = plt.subplots(figsize=(6.3, 5.3))
image = ax.imshow(matrix, cmap="Blues")
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES,
       yticklabels=CLASS_NAMES, xlabel="Predição", ylabel="Classe real",
       title="Matriz de confusão — conjunto de teste")
fig.colorbar(image, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(OUTPUTS / "06_matriz_confusao.png", dpi=180)
plt.close(fig)

with (OUTPUTS / "06_relatorio_classificacao.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("Matriz de confusão:\n", matrix)
print(f"Acurácia global: {100 * np.mean(y_true == y_pred):.2f}%")
print("Imagem:", OUTPUTS / "06_matriz_confusao.png")

