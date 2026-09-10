"""Exemplo 07 — FLANN k-NN e teste da razão de Lowe com threshold 0.75."""

from __future__ import annotations

import argparse

import cv2

from utils import ensure_outputs, extract, flann_for, load_pair, lowe_filter


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["SIFT", "ORB", "AKAZE"], default="SIFT")
parser.add_argument("--ratio", type=float, default=0.75)
args = parser.parse_args()

image_a, image_b = load_pair()
kp_a, desc_a, _ = extract(args.method, image_a, repeats=1)
kp_b, desc_b, _ = extract(args.method, image_b, repeats=1)

matcher = flann_for(args.method)
knn = matcher.knnMatch(desc_a, desc_b, k=2)
good = lowe_filter(knn, args.ratio)

visual = cv2.drawMatches(image_a, kp_a, image_b, kp_b, good[:140], None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
output = ensure_outputs() / f"07_flann_lowe_{args.method.lower()}.jpg"
cv2.imwrite(str(output), visual)

print(f"Método: {args.method}")
print(f"Pares k-NN: {len(knn)}")
print(f"Aprovados pelo teste de Lowe ({args.ratio:.2f}): {len(good)}")
print(f"Retenção: {100 * len(good) / len(knn):.1f}%")
print(f"Saída: {output}")
