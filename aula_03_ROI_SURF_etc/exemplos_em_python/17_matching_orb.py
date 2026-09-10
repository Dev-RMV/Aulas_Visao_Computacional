"""Exemplo 17 — Matching ORB com distância de Hamming e cross-check.

Uso:
  python exemplos/17_matching_orb.py \
    --imagem-a dados/gerados/objeto_consulta.png \
    --imagem-b dados/gerados/objeto_transformado.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagem-a", default="dados/gerados/objeto_consulta.png")
    parser.add_argument("--imagem-b", default="dados/gerados/objeto_transformado.png")
    parser.add_argument("--mostrar", type=int, default=40)
    args = parser.parse_args()

    a = cv2.imread(args.imagem_a, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(args.imagem_b, cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise FileNotFoundError("Execute o exemplo 01 ou informe imagens válidas.")

    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    kp_a, des_a = orb.detectAndCompute(a, None)
    kp_b, des_b = orb.detectAndCompute(b, None)
    if des_a is None or des_b is None:
        raise RuntimeError("Não foram encontrados descritores suficientes.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    correspondencias = sorted(matcher.match(des_a, des_b), key=lambda m: m.distance)
    melhores = correspondencias[: args.mostrar]
    desenho = cv2.drawMatches(
        a,
        kp_a,
        b,
        kp_b,
        melhores,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    destino = Path("dados/gerados/matching_orb.png")
    cv2.imwrite(str(destino), desenho)
    media = sum(m.distance for m in melhores) / max(len(melhores), 1)
    print(f"Keypoints A/B: {len(kp_a)}/{len(kp_b)}")
    print(f"Matches: {len(correspondencias)}; distância média dos exibidos: {media:.1f}")
    cv2.imshow("ORB + Hamming", desenho)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

