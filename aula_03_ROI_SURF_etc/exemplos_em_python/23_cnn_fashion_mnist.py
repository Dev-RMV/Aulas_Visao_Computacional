"""Exemplo 23 — CNN compacta para Fashion-MNIST.

Compara-se com o exemplo 22: a CNN preserva vizinhança espacial e compartilha pesos.
Uso: python exemplos/23_cnn_fashion_mnist.py --epocas 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


def construir_modelo() -> keras.Model:
    entradas = keras.Input(shape=(28, 28, 1), name="imagem")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(entradas)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="mapas_64")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(96, activation="relu", name="features")(x)
    x = layers.Dropout(0.3)(x)
    saidas = layers.Dense(10, activation="softmax", name="classes")(x)
    return keras.Model(entradas, saidas, name="cnn_fashion_mnist")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.fashion_mnist.load_data()
    x_treino = (x_treino.astype("float32") / 255.0)[..., None]
    x_teste = (x_teste.astype("float32") / 255.0)[..., None]

    modelo = construir_modelo()
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
    modelo.save(pasta / "cnn_fashion.keras")
    fig, eixo = plt.subplots(figsize=(7, 4))
    eixo.plot(historico.history["accuracy"], label="treino")
    eixo.plot(historico.history["val_accuracy"], label="validação")
    eixo.set(title="CNN — evolução da acurácia", xlabel="Época", ylabel="Acurácia")
    eixo.legend()
    fig.tight_layout()
    fig.savefig(pasta / "historico_cnn.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

