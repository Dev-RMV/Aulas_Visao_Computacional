"""Item B completo — BF, FLANN+Lowe, RANSAC, homografia e alinhamento.

Relevância para localização visual em robótica:
Os matches propõem correspondências entre a visão atual e uma referência. A
homografia transforma esses pares em uma hipótese geométrica para uma cena
aproximadamente planar ou uma câmera em rotação. O RANSAC rejeita outliers e o
número/proporção de inliers funciona como evidência de consistência. Um robô não
deve aceitar a pose apenas porque existem muitos matches: deve exigir inliers
suficientes, boa distribuição espacial e erro de reprojeção compatível.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from utils import RESOURCES, corner_error, descriptor_info, ensure_outputs, extract, flann_for, load_pair, lowe_filter


METHOD = "SIFT"
RATIO = 0.75
RANSAC_THRESHOLD = 4.0

image_a, image_b = load_pair()
kp_a, desc_a, time_a = extract(METHOD, image_a, repeats=7)
kp_b, desc_b, time_b = extract(METHOD, image_b, repeats=7)

# 1) BFMatcher + cross-check.
norm, _, _ = descriptor_info(METHOD)
bf_matches = sorted(cv2.BFMatcher(norm, crossCheck=True).match(desc_a, desc_b), key=lambda m: m.distance)
bf_visual = cv2.drawMatches(image_a, kp_a, image_b, kp_b, bf_matches[:120], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(str(ensure_outputs() / "09_bf_crosscheck.jpg"), bf_visual)

# 2) FLANN k-NN + teste de razão de Lowe.
knn = flann_for(METHOD).knnMatch(desc_a, desc_b, k=2)
good = lowe_filter(knn, RATIO)
if len(good) < 4:
    raise RuntimeError("Matches insuficientes para homografia")

points_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
points_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# 3) Homografia referência -> imagem transformada e inliers via RANSAC.
h_ab, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, RANSAC_THRESHOLD)
if h_ab is None:
    raise RuntimeError("Homografia inválida")
inliers = int(mask.sum())
inlier_matches = [m for m, keep in zip(good, mask.ravel().astype(bool)) if keep]
match_visual = cv2.drawMatches(image_a, kp_a, image_b, kp_b, inlier_matches[:160], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(str(ensure_outputs() / "09_flann_lowe_inliers.jpg"), match_visual)

# 4) Inverter H para alinhar B sobre A com warpPerspective.
h_ba = np.linalg.inv(h_ab)
height, width = image_a.shape[:2]
aligned = cv2.warpPerspective(image_b, h_ba, (width, height))
overlay = cv2.addWeighted(image_a, 0.50, aligned, 0.50, 0)
cv2.imwrite(str(ensure_outputs() / "09_imagem_alinhada.png"), aligned)
cv2.imwrite(str(ensure_outputs() / "09_alinhamento_overlay.png"), overlay)

with (RESOURCES / "homografia_real.json").open(encoding="utf-8") as f:
    h_true = np.asarray(json.load(f)["H_referencia_para_transformada"], np.float64)
error = corner_error(h_ab, h_true, width, height)

print("\nITEM B — RESULTADOS")
print(f"Extração SIFT: A={time_a:.2f} ms, B={time_b:.2f} ms")
print(f"BFMatcher cross-check: {len(bf_matches)} matches")
print(f"FLANN k-NN: {len(knn)} pares")
print(f"Após Lowe {RATIO:.2f}: {len(good)} matches")
print(f"Inliers do RANSAC: {inliers}")
print(f"Taxa de inliers: {100 * inliers / len(good):.1f}%")
print(f"Erro médio nos cantos: {error:.2f} px")
print(f"Imagem alinhada: {ensure_outputs() / '09_imagem_alinhada.png'}")
print(f"Overlay: {ensure_outputs() / '09_alinhamento_overlay.png'}")
