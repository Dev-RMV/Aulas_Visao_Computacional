"""Exemplo 06 — BFMatcher com cross-check. Padrão: SIFT + distância L2."""

from __future__ import annotations

import argparse

import cv2

from utils import descriptor_info, ensure_outputs, extract, load_pair


parser = argparse.ArgumentParser()
parser.add_argument("--method", choices=["SIFT", "ORB", "AKAZE"], default="SIFT")
args = parser.parse_args()

image_a, image_b = load_pair()
kp_a, desc_a, _ = extract(args.method, image_a, repeats=1)
kp_b, desc_b, _ = extract(args.method, image_b, repeats=1)
norm, _, _ = descriptor_info(args.method)

# crossCheck=True mantém m apenas se A->B e B->A concordarem como melhor par.
matcher = cv2.BFMatcher(norm, crossCheck=True)
matches = sorted(matcher.match(desc_a, desc_b), key=lambda m: m.distance)
shown = matches[:100]
visual = cv2.drawMatches(image_a, kp_a, image_b, kp_b, shown, None,
                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
output = ensure_outputs() / f"06_bf_crosscheck_{args.method.lower()}.jpg"
cv2.imwrite(str(output), visual)

print(f"Método: {args.method}")
print(f"Matches com cross-check: {len(matches)}")
print(f"Exibidos: {len(shown)} melhores")
print(f"Distância mediana: {sorted(m.distance for m in matches)[len(matches)//2]:.3f}")
print(f"Saída: {output}")
