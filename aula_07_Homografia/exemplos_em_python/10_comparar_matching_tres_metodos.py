"""Exemplo 10 — compara matching e homografia com SIFT, ORB e AKAZE."""

from __future__ import annotations

import csv

import cv2
import numpy as np

from utils import ensure_outputs, extract, flann_for, load_pair, lowe_filter


image_a, image_b = load_pair()
rows = []

for method in ("SIFT", "ORB", "AKAZE"):
    kp_a, desc_a, time_a = extract(method, image_a)
    kp_b, desc_b, time_b = extract(method, image_b)
    knn = flann_for(method).knnMatch(desc_a, desc_b, k=2)
    good = lowe_filter(knn, 0.75)
    inliers = 0
    if len(good) >= 4:
        points_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        points_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
    rows.append({
        "metodo": method,
        "tempo_total_extracao_ms": round(time_a + time_b, 2),
        "keypoints_total": len(kp_a) + len(kp_b),
        "matches_lowe": len(good),
        "inliers_ransac": inliers,
        "taxa_inliers_pct": round(100 * inliers / len(good), 1) if good else 0.0,
    })

output = ensure_outputs() / "10_comparacao_matching.csv"
with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("método   extração(ms)  keypoints  Lowe  inliers  taxa-inliers")
for row in rows:
    print(f"{row['metodo']:<8} {row['tempo_total_extracao_ms']:>11.2f} "
          f"{row['keypoints_total']:>10} {row['matches_lowe']:>6} "
          f"{row['inliers_ransac']:>8} {row['taxa_inliers_pct']:>11.1f}%")
print(f"CSV: {output}")
