from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset"
OUTPUTS = PROJECT / "saidas"
MODELS = PROJECT / "modelos"
RESOURCES = PROJECT / "recursos"

CLASS_NAMES = ["circulo", "quadrado", "triangulo"]
CLASS_COLORS_RGB = {
    "circulo": "#EF5350",
    "quadrado": "#3D8DFF",
    "triangulo": "#2FBF71",
}
IMAGE_SIZE = (64, 64)
SEED = 2026


def configure(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def ensure_dirs() -> None:
    for path in (DATASET, OUTPUTS, MODELS, RESOURCES):
        path.mkdir(parents=True, exist_ok=True)


def require_dataset() -> None:
    expected = DATASET / "train" / CLASS_NAMES[0]
    if not expected.exists() or not any(expected.glob("*.png")):
        raise FileNotFoundError(
            "Dataset não encontrado. Execute: python exemplos/01_gerar_dataset.py"
        )


def load_datasets(batch_size: int = 64):
    require_dataset()
    common = dict(
        labels="inferred",
        label_mode="int",
        class_names=CLASS_NAMES,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
    )
    train = keras.utils.image_dataset_from_directory(
        DATASET / "train", shuffle=True, seed=SEED, **common
    )
    val = keras.utils.image_dataset_from_directory(
        DATASET / "val", shuffle=False, **common
    )
    test = keras.utils.image_dataset_from_directory(
        DATASET / "test", shuffle=False, **common
    )
    autotune = tf.data.AUTOTUNE
    return (
        train.prefetch(autotune),
        val.prefetch(autotune),
        test.prefetch(autotune),
    )


def build_cnn() -> keras.Model:
    """CNN do zero exigida no Item A.

    A Dense chamada ``features_dense`` é a penúltima camada Dense. Ela produz
    o vetor de 64 features usado no Item B; a Dense seguinte é o classificador
    softmax de três classes.
    """
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="imagem_rgb")
    x = layers.Rescaling(1.0 / 255.0, name="normalizacao")(inputs)

    # Bloco convolucional 1: detecta padrões locais simples, como bordas.
    x = layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1a")(x)
    x = layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1b")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    # Bloco convolucional 2: combina bordas em partes mais abstratas.
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2a")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2b")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    # Flatten atende ao requisito e converte mapas 2D em um vetor.
    x = layers.Flatten(name="flatten")(x)
    features = layers.Dense(64, activation="relu", name="features_dense")(x)
    x = layers.Dropout(0.15, name="dropout")(features)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax", name="classe")(x)

    model = keras.Model(inputs, outputs, name="cnn_geometrica")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_trained_model() -> keras.Model:
    path = MODELS / "cnn_geometrica.keras"
    if not path.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute: python exemplos/05_item_a_treinar_cnn.py"
        )
    return keras.models.load_model(path)


def image_paths(split: str = "test") -> list[Path]:
    require_dataset()
    return [
        path
        for class_name in CLASS_NAMES
        for path in sorted((DATASET / split / class_name).glob("*.png"))
    ]


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def select_twenty() -> list[dict]:
    """Seleciona exatamente 20 imagens: 7 + 7 + 6, todas do teste."""
    counts = {"circulo": 7, "quadrado": 7, "triangulo": 6}
    selected = []
    for label, class_name in enumerate(CLASS_NAMES):
        paths = sorted((DATASET / "test" / class_name).glob("*.png"))
        for path in paths[: counts[class_name]]:
            selected.append(
                {
                    "path": str(path.relative_to(PROJECT)),
                    "classe": class_name,
                    "label": label,
                }
            )
    if len(selected) != 20:
        raise RuntimeError(f"Esperadas 20 imagens, encontradas {len(selected)}")
    ensure_dirs()
    with (OUTPUTS / "selecao_20_imagens.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    return selected


def read_selection() -> list[dict]:
    path = OUTPUTS / "selecao_20_imagens.json"
    if not path.exists():
        return select_twenty()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def overfitting_diagnosis(history: dict) -> dict:
    train_acc = np.asarray(history["accuracy"], dtype=float)
    val_acc = np.asarray(history["val_accuracy"], dtype=float)
    val_loss = np.asarray(history["val_loss"], dtype=float)
    gap = float(train_acc[-1] - val_acc[-1])
    best_val_loss_epoch = int(np.argmin(val_loss) + 1)
    loss_rising = bool(
        val_loss[-1] > 1.10 * val_loss.min()
        and best_val_loss_epoch <= len(val_loss) - 2
    )
    has_signal = bool(gap > 0.08 or loss_rising)
    if has_signal:
        comment = (
            "Há indício de overfitting: a rede continua favorecendo o treino "
            "enquanto a validação perde qualidade. Considere mais dados, "
            "data augmentation, dropout ou regularização L2."
        )
    else:
        comment = (
            "Não há indício forte de overfitting: treino e validação permanecem "
            "próximos e a loss de validação não apresenta deterioração relevante."
        )
    return {
        "gap_acuracia_final": gap,
        "melhor_epoca_val_loss": best_val_loss_epoch,
        "val_loss_em_alta": loss_rising,
        "indicio_overfitting": has_signal,
        "comentario": comment,
    }

