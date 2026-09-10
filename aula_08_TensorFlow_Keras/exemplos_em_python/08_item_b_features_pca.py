"""Item B — extrai 64 features da penúltima Dense e reduz para PCA 2D.

Separabilidade:
Uma boa separação significa que a rede organizou imagens da mesma classe em
regiões próximas e colocou classes diferentes em regiões distintas. O gráfico
é uma projeção 2D de um espaço de 64 dimensões; sobreposição no PCA não prova
que as classes sejam inseparáveis no espaço original.
"""

from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from utils import (
    CLASS_COLORS_RGB, CLASS_NAMES, OUTPUTS, PROJECT, ensure_dirs, load_rgb,
    load_trained_model, select_twenty,
)


ensure_dirs()
selection = select_twenty()
images = np.stack([load_rgb(PROJECT / item["path"]) for item in selection])
labels = np.asarray([item["label"] for item in selection])
model = load_trained_model()

# Remove apenas o classificador final ao criar um novo modelo cuja saída é a
# Dense ``features_dense``. O vetor possui 64 valores por imagem.
extractor = keras.Model(model.input, model.get_layer("features_dense").output)
features = extractor.predict(images, verbose=0)
scaled = StandardScaler().fit_transform(features)
pca = PCA(n_components=2, random_state=2026)
points = pca.fit_transform(scaled)
silhouette = float(silhouette_score(scaled, labels))

fig, ax = plt.subplots(figsize=(8.5, 6.5))
for label, class_name in enumerate(CLASS_NAMES):
    mask = labels == label
    ax.scatter(points[mask, 0], points[mask, 1], s=90,
               color=CLASS_COLORS_RGB[class_name], label=class_name,
               edgecolor="black", linewidth=0.6, alpha=0.9)
ax.axhline(0, color="#BBBBBB", lw=0.8); ax.axvline(0, color="#BBBBBB", lw=0.8)
ax.grid(alpha=0.18)
ax.set(title="Features CNN da Dense de 64 unidades — PCA 2D",
       xlabel=f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% da variância)",
       ylabel=f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% da variância)")
ax.legend(title="Classe")
fig.tight_layout(); fig.savefig(OUTPUTS / "08_pca_features_cnn.png", dpi=190)
plt.close(fig)

csv_path = OUTPUTS / "08_features_cnn.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["arquivo", "classe", "label", "pca_1", "pca_2"] + [f"f{i:02d}" for i in range(64)])
    for item, point, vector in zip(selection, points, features):
        writer.writerow([item["path"], item["classe"], item["label"], *point.tolist(), *vector.tolist()])

result = {
    "imagens": len(selection),
    "dimensao_features": int(features.shape[1]),
    "silhouette_64d": silhouette,
    "variancia_explicada_pca_2d": float(pca.explained_variance_ratio_.sum()),
    "comentario": (
        "Quanto maior o silhouette, mais compactas e separadas estão as classes. "
        "A interpretação deve considerar também o scatter plot e a matriz de confusão."
    ),
}
with (OUTPUTS / "08_resultado_pca_cnn.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\n=== ITEM B — FEATURES CNN ===")
print("Imagens:", len(selection), "(7 círculos, 7 quadrados, 6 triângulos)")
print("Shape das features:", features.shape)
print(f"Silhouette no espaço 64D: {silhouette:.3f}")
print(f"Variância explicada por PC1+PC2: {100*pca.explained_variance_ratio_.sum():.1f}%")
print("Scatter:", OUTPUTS / "08_pca_features_cnn.png")
