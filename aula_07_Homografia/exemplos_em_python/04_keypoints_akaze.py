"""Exemplo 04 — keypoints AKAZE nas duas imagens."""

import cv2
import numpy as np

from utils import ensure_outputs, extract, load_pair


image_a, image_b = load_pair()
kp_a, desc_a, time_a = extract("AKAZE", image_a)
kp_b, desc_b, time_b = extract("AKAZE", image_b)

vis_a = cv2.drawKeypoints(image_a, kp_a, None, (69, 81, 237), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
vis_b = cv2.drawKeypoints(image_b, kp_b, None, (69, 81, 237), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
output = ensure_outputs() / "04_keypoints_akaze.jpg"
cv2.imwrite(str(output), np.hstack([vis_a, vis_b]))

print(f"AKAZE | A: {len(kp_a)} keypoints, {time_a:.2f} ms")
print(f"AKAZE | B: {len(kp_b)} keypoints, {time_b:.2f} ms")
print(f"Descritor: shape={desc_a.shape}, dtype={desc_a.dtype}, dimensão={desc_a.shape[1]} bytes")
print(f"Saída: {output}")
