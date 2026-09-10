"""Exemplo 08 — estima homografia com RANSAC, inliers e erro nos cantos."""

from __future__ import annotations

import json

import cv2
import numpy as np

from utils import RESOURCES, corner_error, ensure_outputs, extract, flann_for, load_pair, lowe_filter


image_a, image_b = load_pair()
kp_a, desc_a, _ = extract("SIFT", image_a, repeats=1)
kp_b, desc_b, _ = extract("SIFT", image_b, repeats=1)
good = lowe_filter(flann_for("SIFT").knnMatch(desc_a, desc_b, k=2), 0.75)

if len(good) < 4:
    raise RuntimeError("Homografia requer pelo menos quatro correspondências.")

points_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
points_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
homography, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, 4.0)
if homography is None:
    raise RuntimeError("RANSAC não encontrou uma homografia válida")

inlier_mask = mask.ravel().astype(bool)
inlier_matches = [m for m, keep in zip(good, inlier_mask) if keep]
visual_b = image_b.copy()
h, w = image_a.shape[:2]
corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
projected = cv2.perspectiveTransform(corners, homography)
cv2.polylines(visual_b, [np.int32(projected)], True, (38, 226, 167), 5, cv2.LINE_AA)
visual = cv2.drawMatches(image_a, kp_a, visual_b, kp_b, inlier_matches[:140], None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
output = ensure_outputs() / "08_homografia_inliers.jpg"
cv2.imwrite(str(output), visual)

with (RESOURCES / "homografia_real.json").open(encoding="utf-8") as f:
    h_true = np.asarray(json.load(f)["H_referencia_para_transformada"], np.float64)
error = corner_error(homography, h_true, w, h)

print(f"Matches após Lowe: {len(good)}")
print(f"Inliers do RANSAC: {int(mask.sum())}")
print(f"Taxa de inliers: {100 * mask.mean():.1f}%")
print(f"Erro médio nos quatro cantos: {error:.2f} px")
print(f"Saída: {output}")
