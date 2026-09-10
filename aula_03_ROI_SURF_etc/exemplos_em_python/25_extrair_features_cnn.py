"""Exemplo 25 — Extração de features de uma camada intermediária da CNN.

Projeta vetores de features em 2D com PCA para inspecionar separação entre classes.
Uso:
  python exemplos/25_extrair_features_cnn.py --modelo modelos/cnn_fashion.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tensorflow import keras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="modelos/cnn_fashion.keras")
    parser.add_argument("--camada", default="features")
    parser.add_argument("--amostras", type=int, default=2000)
    args = parser.parse_args()

    if not Path(args.modelo).exists():
        raise FileNotFoundError("Treine o exemplo 23 ou 24 primeiro.")
    modelo = keras.models.load_model(args.modelo)
    try:
        camada = modelo.get_layer(args.camada)
    except ValueError as erro:
        nomes = [c.name for c in modelo.layers]
        raise ValueError(f"Camada ausente. Disponíveis: {nomes}") from erro

    extrator = keras.Model(inputs=modelo.inputs, outputs=camada.output)
    (_, _), (x_teste, y_teste) = keras.datasets.fashion_mnist.load_data()
    n = min(args.amostras, len(x_teste))
    x = (x_teste[:n].astype("float32") / 255.0)[..., None]
    features = extrator.predict(x, batch_size=256, verbose=0)
    features = features.reshape(n, -1)
    np.savez_compressed("modelos/features_fashion.npz", features=features, labels=y_teste[:n])

    pca = PCA(n_components=2, random_state=42)
    projecao = pca.fit_transform(features)
    fig, eixo = plt.subplots(figsize=(8, 6))
    pontos = eixo.scatter(
        projecao[:, 0], projecao[:, 1], c=y_teste[:n], cmap="tab10", s=9, alpha=0.65
    )
    eixo.set(title=f"Features da camada '{args.camada}' projetadas por PCA", xlabel="PC1", ylabel="PC2")
    fig.colorbar(pontos, ax=eixo, ticks=range(10), label="Classe")
    fig.tight_layout()
    fig.savefig("modelos/features_pca.png", dpi=170)
    plt.close(fig)
    print(
        f"Tensor extraído: {features.shape}; "
        f"variância em 2 PCs: {pca.explained_variance_ratio_.sum():.3f}"
    )


if __name__ == "__main__":
    main()
