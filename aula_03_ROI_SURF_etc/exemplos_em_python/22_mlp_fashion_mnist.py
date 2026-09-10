"""Exemplo 22 — MLP para classificação de imagens Fashion-MNIST.

A MLP recebe cada imagem 28×28 como um vetor de 784 valores e perde a topologia 2D.
Uso: python exemplos/22_mlp_fashion_mnist.py --epocas 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers


CLASSES = [
    "camiseta/top",
    "calça",
    "pulôver",
    "vestido",
    "casaco",
    "sandália",
    "camisa",
    "tênis",
    "bolsa",
    "bota",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--semente", type=int, default=42)
    args = parser.parse_args()

    keras.utils.set_random_seed(args.semente)
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.fashion_mnist.load_data()
    x_treino = x_treino.astype("float32") / 255.0
    x_teste = x_teste.astype("float32") / 255.0

    modelo = keras.Sequential(
        [
            keras.Input(shape=(28, 28), name="imagem"),
            layers.Flatten(name="vetor_784"),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(128, activation="relu", name="features"),
            layers.Dense(10, activation="softmax", name="classes"),
        ],
        name="mlp_fashion_mnist",
    )
    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.summary()
    historico = modelo.fit(
        x_treino,
        y_treino,
        validation_split=0.1,
        epochs=args.epocas,
        batch_size=args.batch,
        verbose=2,
    )
    perda, acuracia = modelo.evaluate(x_teste, y_teste, verbose=0)
    print(f"Teste — perda={perda:.4f}; acurácia={acuracia:.4f}")

    pasta = Path("modelos")
    pasta.mkdir(parents=True, exist_ok=True)
    modelo.save(pasta / "mlp_fashion.keras")

    fig, eixos = plt.subplots(1, 2, figsize=(11, 4))
    eixos[0].plot(historico.history["loss"], label="treino")
    eixos[0].plot(historico.history["val_loss"], label="validação")
    eixos[0].set(title="Perda", xlabel="Época")
    eixos[0].legend()
    eixos[1].plot(historico.history["accuracy"], label="treino")
    eixos[1].plot(historico.history["val_accuracy"], label="validação")
    eixos[1].set(title="Acurácia", xlabel="Época")
    eixos[1].legend()
    fig.tight_layout()
    fig.savefig(pasta / "historico_mlp.png", dpi=160)
    plt.close(fig)

    previsoes = modelo.predict(x_teste[:2000], verbose=0).argmax(axis=1)
    matriz = confusion_matrix(y_teste[:2000], previsoes)
    figura, eixo = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matriz, display_labels=CLASSES).plot(
        ax=eixo, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    figura.tight_layout()
    figura.savefig(pasta / "matriz_confusao_mlp.png", dpi=160)
    plt.close(figura)


if __name__ == "__main__":
    main()

