"""Exemplo 01 — gera um dataset RGB próprio com três classes.

Cada amostra recebe variações de posição, rotação, escala, iluminação, ruído e
oclusão leve. O objetivo é simular a variabilidade observada por uma câmera de
robô sem depender de download ou de uma câmera real.
"""

from __future__ import annotations

import json
import shutil

import cv2
import numpy as np

from utils import CLASS_NAMES, DATASET, IMAGE_SIZE, RESOURCES, SEED, ensure_dirs


SPLITS = {"train": 360, "val": 90, "test": 90}
COLORS_BGR = {
    "circulo": (64, 76, 239),
    "quadrado": (255, 141, 61),
    "triangulo": (113, 191, 47),
}


def polygon_for(class_name: str, center: tuple[int, int], size: int) -> np.ndarray:
    cx, cy = center
    if class_name == "quadrado":
        return np.int32(
            [[cx - size, cy - size], [cx + size, cy - size],
             [cx + size, cy + size], [cx - size, cy + size]]
        )
    return np.int32(
        [[cx, cy - size - 2], [cx + size + 2, cy + size],
         [cx - size - 2, cy + size]]
    )


def make_sample(class_name: str, rng: np.random.Generator) -> np.ndarray:
    height, width = IMAGE_SIZE
    base = int(rng.integers(205, 246))
    image = np.full((height, width, 3), base, np.uint8)

    # Fundo com textura discreta: gera keypoints sem dominar a classe.
    for _ in range(int(rng.integers(5, 12))):
        p = tuple(rng.integers(2, 62, size=2).tolist())
        shade = int(rng.integers(135, 205))
        cv2.circle(image, p, int(rng.integers(1, 3)), (shade,) * 3, -1, cv2.LINE_AA)
    for _ in range(int(rng.integers(1, 4))):
        p1 = tuple(rng.integers(0, 64, size=2).tolist())
        p2 = tuple(rng.integers(0, 64, size=2).tolist())
        shade = int(rng.integers(170, 220))
        cv2.line(image, p1, p2, (shade,) * 3, 1, cv2.LINE_AA)

    layer = np.zeros_like(image)
    mask = np.zeros((height, width), np.uint8)
    center = (32 + int(rng.integers(-4, 5)), 32 + int(rng.integers(-4, 5)))
    size = int(rng.integers(15, 21))
    color = COLORS_BGR[class_name]

    if class_name == "circulo":
        cv2.circle(mask, center, size, 255, -1, cv2.LINE_AA)
        cv2.circle(layer, center, size, color, -1, cv2.LINE_AA)
        cv2.circle(layer, center, size, (25, 25, 25), 2, cv2.LINE_AA)
        cv2.line(layer, (center[0] - size + 5, center[1]),
                 (center[0] + size - 5, center[1]), (245, 245, 245), 3, cv2.LINE_AA)
        cv2.line(layer, (center[0], center[1] - size + 5),
                 (center[0], center[1] + size - 5), (245, 245, 245), 3, cv2.LINE_AA)
    else:
        poly = polygon_for(class_name, center, size)
        cv2.fillConvexPoly(mask, poly, 255, cv2.LINE_AA)
        cv2.fillConvexPoly(layer, poly, color, cv2.LINE_AA)
        cv2.polylines(layer, [poly], True, (25, 25, 25), 2, cv2.LINE_AA)
        if class_name == "quadrado":
            cv2.line(layer, tuple(poly[0]), tuple(poly[2]), (245, 245, 245), 3, cv2.LINE_AA)
            cv2.line(layer, tuple(poly[1]), tuple(poly[3]), (245, 245, 245), 3, cv2.LINE_AA)
        else:
            cv2.circle(layer, center, 5, (245, 245, 245), -1, cv2.LINE_AA)
            cv2.circle(layer, center, 5, (25, 25, 25), 1, cv2.LINE_AA)

    angle = float(rng.uniform(-24, 24))
    scale = float(rng.uniform(0.82, 1.10))
    matrix = cv2.getRotationMatrix2D((32, 32), angle, scale)
    layer = cv2.warpAffine(layer, matrix, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_LINEAR)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    image = np.uint8(image * (1.0 - alpha) + layer * alpha)

    # Oclusão leve em parte das amostras.
    if rng.random() < 0.18:
        x = int(rng.integers(4, 48)); y = int(rng.integers(4, 52))
        w = int(rng.integers(5, 13)); h = int(rng.integers(3, 9))
        shade = int(rng.integers(110, 190))
        cv2.rectangle(image, (x, y), (x + w, y + h), (shade,) * 3, -1)

    # Gradiente de iluminação e ruído de sensor.
    direction = float(rng.choice([-1.0, 1.0]))
    gradient = np.linspace(0.82, 1.18, width, dtype=np.float32)
    if direction < 0:
        gradient = gradient[::-1]
    image = np.clip(image.astype(np.float32) * gradient[None, :, None], 0, 255)
    noise = rng.normal(0, rng.uniform(2.0, 7.0), image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def montage(paths: list, destination) -> None:
    tiles = [cv2.imread(str(path)) for path in paths]
    tiles = [cv2.resize(tile, (128, 128), interpolation=cv2.INTER_NEAREST) for tile in tiles]
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(str(destination), np.vstack(rows))


ensure_dirs()
if DATASET.exists():
    shutil.rmtree(DATASET)
DATASET.mkdir(parents=True)

for split_index, (split, count) in enumerate(SPLITS.items()):
    for class_index, class_name in enumerate(CLASS_NAMES):
        folder = DATASET / split / class_name
        folder.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(SEED + 10000 * split_index + 1000 * class_index)
        for index in range(count):
            image = make_sample(class_name, rng)
            cv2.imwrite(str(folder / f"{class_name}_{index:04d}.png"), image)

sample_paths = []
for class_name in CLASS_NAMES:
    sample_paths.extend(sorted((DATASET / "train" / class_name).glob("*.png"))[:6])
montage(sample_paths, RESOURCES / "amostras_dataset.png")

metadata = {
    "classes": CLASS_NAMES,
    "tamanho_imagem": list(IMAGE_SIZE) + [3],
    "quantidades_por_classe": SPLITS,
    "total": len(CLASS_NAMES) * sum(SPLITS.values()),
    "seed": SEED,
}
with (DATASET / "dataset_info.json").open("w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("Dataset criado:", DATASET)
print("Classes:", CLASS_NAMES)
print("Total de imagens:", metadata["total"])
print("Montagem:", RESOURCES / "amostras_dataset.png")

