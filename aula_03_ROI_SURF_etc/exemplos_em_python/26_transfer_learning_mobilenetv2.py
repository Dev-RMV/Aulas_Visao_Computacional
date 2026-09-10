"""Exemplo 26 — Transfer learning com MobileNetV2 em dataset por pastas.

Estrutura esperada: dados/dataset_custom/treino/<classe>/ e validacao/<classe>/.
Uso: python exemplos/26_transfer_learning_mobilenetv2.py --dados dados/dataset_custom
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


TAMANHO = (160, 160)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", default="dados/dataset_custom")
    parser.add_argument("--epocas", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    treino_dir = Path(args.dados) / "treino"
    validacao_dir = Path(args.dados) / "validacao"
    if not treino_dir.exists() or not validacao_dir.exists():
        raise FileNotFoundError(
            "Crie treino/<classe>/ e validacao/<classe>/ conforme o README."
        )

    treino = keras.utils.image_dataset_from_directory(
        treino_dir, image_size=TAMANHO, batch_size=args.batch, label_mode="int", seed=42
    )
    validacao = keras.utils.image_dataset_from_directory(
        validacao_dir, image_size=TAMANHO, batch_size=args.batch, label_mode="int", shuffle=False
    )
    classes = treino.class_names
    if len(classes) < 2:
        raise RuntimeError("O dataset precisa ter ao menos duas classes.")

    aumento = keras.Sequential(
        [layers.RandomFlip("horizontal"), layers.RandomRotation(0.08), layers.RandomZoom(0.1)]
    )
    base = keras.applications.MobileNetV2(
        input_shape=(*TAMANHO, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    entradas = keras.Input(shape=(*TAMANHO, 3))
    x = aumento(entradas)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="features")(x)
    x = layers.Dropout(0.25)(x)
    saidas = layers.Dense(len(classes), activation="softmax")(x)
    modelo = keras.Model(entradas, saidas, name="mobilenetv2_transfer")
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    treino = treino.prefetch(buffer_size=tf.data.AUTOTUNE)
    validacao = validacao.prefetch(buffer_size=tf.data.AUTOTUNE)
    modelo.fit(
        treino,
        validation_data=validacao,
        epochs=args.epocas,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=2,
    )
    Path("modelos").mkdir(exist_ok=True)
    modelo.save("modelos/mobilenetv2_custom.keras")
    Path("modelos/mobilenetv2_classes.txt").write_text("\n".join(classes), encoding="utf-8")
    print("Classes:", classes)


if __name__ == "__main__":
    main()
