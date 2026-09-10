"""Exemplo 24 — CNN com aumento de dados e callbacks.

Demonstra RandomRotation/Translation/Zoom, EarlyStopping, checkpoint e ajuste de LR.
Uso: python exemplos/24_cnn_augmentation_callbacks.py --epocas 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow import keras
from tensorflow.keras import layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epocas", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    keras.utils.set_random_seed(42)
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.fashion_mnist.load_data()
    x_treino = (x_treino.astype("float32") / 255.0)[..., None]
    x_teste = (x_teste.astype("float32") / 255.0)[..., None]

    aumento = keras.Sequential(
        [
            layers.RandomRotation(0.06),
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomZoom(0.08),
        ],
        name="aumento_de_dados",
    )
    entradas = keras.Input(shape=(28, 28, 1), name="imagem")
    x = aumento(entradas)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(96, activation="relu", name="features")(x)
    x = layers.Dropout(0.35)(x)
    saidas = layers.Dense(10, activation="softmax")(x)
    modelo = keras.Model(entradas, saidas, name="cnn_robusta")
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    pasta = Path("modelos")
    pasta.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            pasta / "cnn_fashion_melhor.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
        ),
        keras.callbacks.CSVLogger(pasta / "treinamento_cnn.csv"),
    ]
    modelo.fit(
        x_treino,
        y_treino,
        validation_split=0.1,
        epochs=args.epocas,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=2,
    )
    perda, acuracia = modelo.evaluate(x_teste, y_teste, verbose=0)
    print(f"Teste com pesos restaurados — perda={perda:.4f}; acurácia={acuracia:.4f}")


if __name__ == "__main__":
    main()

