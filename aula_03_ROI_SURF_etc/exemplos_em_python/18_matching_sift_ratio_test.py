"""Exemplo 18 — Matching SIFT com k-NN e ratio test de Lowe.

Uso:
  python exemplos/18_matching_sift_ratio_test.py \
    --imagem-a dados/gerados/objeto_consulta.png \
    --imagem-b dados/gerados/objeto_transformado.png --ratio 0.75
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagem-a", default="dados/gerados/objeto_consulta.png")
    parser.add_argument("--imagem-b", default="dados/gerados/objeto_transformado.png")
    parser.add_argument("--ratio", type=float, default=0.75)
    args = parser.parse_args()

    a = cv2.imread(args.imagem_a, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(args.imagem_b, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise FileNotFoundError("Imagens inválidas.")

    sift = cv2.SIFT_create(nfeatures=1000)
    kp_a, des_a = sift.detectAndCompute(a, None)
    kp_b, des_b = sift.detectAndCompute(b, None)
    if des_a is None or des_b is None:
        raise RuntimeError("Descritores insuficientes.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pares = matcher.knnMatch(des_a, des_b, k=2)
    bons = [[m] for m, n in pares if m.distance < args.ratio * n.distance]
    desenho = cv2.drawMatchesKnn(
        a,
        kp_a,
        b,
        kp_b,
        bons[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    destino = Path("dados/gerados/matching_sift_ratio.png")
    cv2.imwrite(str(destino), desenho)
    print(f"Pares k-NN: {len(pares)}; aprovados no ratio test: {len(bons)}")
    print("Ratio menor é mais seletivo; ratio maior aceita mais ambiguidades.")
    cv2.imshow("SIFT + ratio test", desenho)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

