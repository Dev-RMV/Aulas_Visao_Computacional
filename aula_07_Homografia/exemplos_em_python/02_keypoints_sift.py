"""Exemplo 02 — keypoints SIFT nas duas imagens."""

import cv2
import numpy as np

from utils import ensure_outputs, extract, load_pair


image_a, image_b = load_pair()
kp_a, desc_a, time_a = extract("SIFT", image_a)
kp_b, desc_b, time_b = extract("SIFT", image_b)

vis_a = cv2.drawKeypoints(image_a, kp_a, None, (61, 141, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
vis_b = cv2.drawKeypoints(image_b, kp_b, None, (61, 141, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
output = ensure_outputs() / "02_keypoints_sift.jpg"
cv2.imwrite(str(output), np.hstack([vis_a, vis_b]))

print(f"SIFT | A: {len(kp_a)} keypoints, {time_a:.2f} ms")
print(f"SIFT | B: {len(kp_b)} keypoints, {time_b:.2f} ms")
print(f"Descritor: shape={desc_a.shape}, dtype={desc_a.dtype}, dimensão={desc_a.shape[1]}")
print(f"Saída: {output}")
