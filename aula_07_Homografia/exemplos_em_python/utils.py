from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RESOURCES = PROJECT / "recursos"
OUTPUTS = PROJECT / "saidas"


def ensure_outputs() -> Path:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    return OUTPUTS


def load_pair() -> tuple[np.ndarray, np.ndarray]:
    image_a = cv2.imread(str(RESOURCES / "cena_referencia.png"))
    image_b = cv2.imread(str(RESOURCES / "cena_transformada.png"))
    if image_a is None or image_b is None:
        raise FileNotFoundError("Execute 01_gerar_imagens_teste.py para criar o par.")
    return image_a, image_b


def create_feature(method: str):
    method = method.upper()
    if method == "SIFT":
        return cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.025)
    if method == "ORB":
        return cv2.ORB_create(nfeatures=1800, scaleFactor=1.2, nlevels=8)
    if method == "AKAZE":
        return cv2.AKAZE_create()
    raise ValueError(f"Método desconhecido: {method}")


def descriptor_info(method: str) -> tuple[int, str, str]:
    method = method.upper()
    if method == "SIFT":
        return cv2.NORM_L2, "float32", "128 valores"
    if method == "ORB":
        return cv2.NORM_HAMMING, "uint8", "32 bytes / 256 bits"
    if method == "AKAZE":
        return cv2.NORM_HAMMING, "uint8", "61 bytes / 486 bits (MLDB padrão)"
    raise ValueError(method)


def extract(method: str, image: np.ndarray, repeats: int = 7):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = create_feature(method)
    feature.detectAndCompute(gray, None)  # aquecimento
    times = []
    keypoints = descriptors = None
    for _ in range(repeats):
        start = time.perf_counter()
        keypoints, descriptors = feature.detectAndCompute(gray, None)
        times.append((time.perf_counter() - start) * 1000)
    return keypoints, descriptors, float(np.median(times))


def flann_for(method: str) -> cv2.FlannBasedMatcher:
    method = method.upper()
    if method == "SIFT":
        # KD-tree para descritores contínuos float32.
        return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    # LSH para descritores binários ORB/AKAZE.
    return cv2.FlannBasedMatcher(
        dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2),
        dict(checks=64),
    )


def lowe_filter(knn_matches, ratio: float = 0.75):
    return [m for pair in knn_matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]


def normalized_homography(h: np.ndarray) -> np.ndarray:
    return h / h[2, 2]


def corner_error(h_est: np.ndarray, h_true: np.ndarray, width: int, height: int) -> float:
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]).reshape(-1, 1, 2)
    est = cv2.perspectiveTransform(corners, normalized_homography(h_est))
    true = cv2.perspectiveTransform(corners, normalized_homography(h_true))
    return float(np.mean(np.linalg.norm(est.reshape(-1, 2) - true.reshape(-1, 2), axis=1)))
