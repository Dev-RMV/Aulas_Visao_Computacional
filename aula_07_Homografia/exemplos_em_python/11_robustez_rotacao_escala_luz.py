"""Exemplo 11 — mede inliers sob rotação, escala e iluminação crescentes."""

from __future__ import annotations

import csv

import cv2
import numpy as np

from utils import ensure_outputs, extract, flann_for, load_pair, lowe_filter


reference, _ = load_pair()
h, w = reference.shape[:2]
conditions = [
    ("leve", 5, 0.95, 0.95),
    ("moderada", 15, 0.82, 0.78),
    ("forte", 28, 0.68, 0.62),
]
rows = []

for label, angle, scale, brightness in conditions:
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    variant = cv2.warpAffine(reference, matrix, (w, h), borderValue=(32, 38, 44))
    variant = cv2.convertScaleAbs(variant, alpha=brightness, beta=12)
    kp_a, desc_a, _ = extract("SIFT", reference, repeats=1)
    kp_b, desc_b, _ = extract("SIFT", variant, repeats=1)
    good = lowe_filter(flann_for("SIFT").knnMatch(desc_a, desc_b, k=2), 0.75)
    inliers = 0
    if len(good) >= 4:
        pa = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pb = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
    rows.append({"condicao": label, "angulo_graus": angle, "escala": scale,
                 "brilho": brightness, "matches_lowe": len(good), "inliers": inliers})
    cv2.imwrite(str(ensure_outputs() / f"11_variante_{label}.jpg"), variant)

output = ensure_outputs() / "11_robustez.csv"
with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)

print("condição   ângulo  escala  luz   Lowe  inliers")
for row in rows:
    print(f"{row['condicao']:<10} {row['angulo_graus']:>6} {row['escala']:>7.2f} "
          f"{row['brilho']:>5.2f} {row['matches_lowe']:>6} {row['inliers']:>8}")
print(f"CSV: {output}")
