"""Item B — compara features CNN com ORB sobre as mesmas 20 imagens.

Comparação justa exige vetores de tamanho fixo. A CNN já entrega 64 valores por
imagem. O ORB entrega uma quantidade variável de descritores 32-byte; por isso
este exemplo cria um Bag of Visual Words (BoVW) com 24 palavras visuais e
representa cada imagem por um histograma de 24 posições.

Interpretação esperada:
- CNN aprende descritores supervisionados para separar as três classes;
- ORB descreve padrões locais genéricos, sem conhecer os rótulos;
- silhouette e PCA ajudam a comparar a organização dos dois espaços, mas não
  substituem a avaliação de classificação.
"""

from __future__ import annotations

import csv
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils import CLASS_COLORS_RGB, CLASS_NAMES, DATASET, OUTPUTS, PROJECT, ensure_dirs, read_selection


VOCABULARY_SIZE = 24
ensure_dirs()
selection = read_selection()
orb = cv2.ORB_create(nfeatures=180, edgeThreshold=7, patchSize=17, fastThreshold=6)


def descriptors(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    _, desc = orb.detectAndCompute(gray, None)
    return desc


# Vocabulário aprendido somente no conjunto de treino.
training_descriptors = []
for class_name in CLASS_NAMES:
    for path in sorted((DATASET / "train" / class_name).glob("*.png"))[:45]:
        desc = descriptors(path)
        if desc is not None:
            training_descriptors.append(desc.astype(np.float32))
all_descriptors = np.vstack(training_descriptors)
kmeans = MiniBatchKMeans(
    n_clusters=VOCABULARY_SIZE, random_state=2026, batch_size=2048,
    n_init=5, max_iter=120,
).fit(all_descriptors)


def bovw(path):
    desc = descriptors(path)
    histogram = np.zeros(VOCABULARY_SIZE, np.float32)
    if desc is not None and len(desc):
        words = kmeans.predict(desc.astype(np.float32))
        histogram = np.bincount(words, minlength=VOCABULARY_SIZE).astype(np.float32)
        histogram /= histogram.sum()
    return histogram


orb_features = np.stack([bovw(PROJECT / item["path"]) for item in selection])
labels = np.asarray([item["label"] for item in selection])
orb_scaled = StandardScaler().fit_transform(orb_features)
orb_points = PCA(n_components=2, random_state=2026).fit_transform(orb_scaled)
orb_silhouette = float(silhouette_score(orb_scaled, labels))

# Reutiliza exatamente os vetores CNN das mesmas 20 imagens.
with (OUTPUTS / "08_features_cnn.csv").open(encoding="utf-8") as f:
    cnn_rows = list(csv.DictReader(f))
cnn_features = np.asarray([[float(row[f"f{i:02d}"]) for i in range(64)] for row in cnn_rows])
cnn_scaled = StandardScaler().fit_transform(cnn_features)
cnn_points = PCA(n_components=2, random_state=2026).fit_transform(cnn_scaled)
cnn_silhouette = float(silhouette_score(cnn_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, points, title in [
    (axes[0], cnn_points, f"CNN — 64 features\nsilhouette = {cnn_silhouette:.3f}"),
    (axes[1], orb_points, f"ORB + BoVW — 24 features\nsilhouette = {orb_silhouette:.3f}"),
]:
    for label, class_name in enumerate(CLASS_NAMES):
        mask = labels == label
        ax.scatter(points[mask, 0], points[mask, 1], s=82,
                   color=CLASS_COLORS_RGB[class_name], label=class_name,
                   edgecolor="black", linewidth=0.55, alpha=0.9)
    ax.axhline(0, color="#BBBBBB", lw=0.8); ax.axvline(0, color="#BBBBBB", lw=0.8)
    ax.grid(alpha=0.18); ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
axes[0].legend(title="Classe")
fig.suptitle("As mesmas 20 imagens em dois espaços de representação")
fig.tight_layout(); fig.savefig(OUTPUTS / "10_comparacao_cnn_orb.png", dpi=190)
plt.close(fig)

with (OUTPUTS / "10_features_orb_bovw.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["arquivo", "classe"] + [f"word_{i:02d}" for i in range(VOCABULARY_SIZE)])
    for item, vector in zip(selection, orb_features):
        writer.writerow([item["path"], item["classe"], *vector.tolist()])

result = {
    "imagens": 20,
    "cnn_dimensao": 64,
    "orb_bovw_dimensao": VOCABULARY_SIZE,
    "cnn_silhouette": cnn_silhouette,
    "orb_silhouette": orb_silhouette,
    "melhor_separacao_neste_ensaio": "CNN" if cnn_silhouette >= orb_silhouette else "ORB+BoVW",
    "comentario": (
        "A CNN foi otimizada com rótulos e tende a organizar o espaço para a tarefa. "
        "ORB é um descritor local clássico, rápido e sem treinamento; o BoVW apenas "
        "transforma seu conjunto variável de descritores em um vetor comparável."
    ),
}
with (OUTPUTS / "10_resultado_comparacao.json").open("w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\n=== COMPARAÇÃO CNN × ORB NAS MESMAS 20 IMAGENS ===")
print(f"CNN: vetor 64D, silhouette={cnn_silhouette:.3f}")
print(f"ORB+BoVW: vetor {VOCABULARY_SIZE}D, silhouette={orb_silhouette:.3f}")
print("Melhor separação neste ensaio:", result["melhor_separacao_neste_ensaio"])
print("Gráfico:", OUTPUTS / "10_comparacao_cnn_orb.png")

