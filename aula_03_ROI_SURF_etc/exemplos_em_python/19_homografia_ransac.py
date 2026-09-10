"""Exemplo 19 — Localização de objeto por SIFT, homografia e RANSAC.

Uso:
  python exemplos/19_homografia_ransac.py \
    --objeto dados/gerados/objeto_consulta.png --cena dados/gerados/cena.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objeto", default="dados/gerados/objeto_consulta.png")
    parser.add_argument("--cena", default="dados/gerados/cena.png")
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--reprojecao", type=float, default=4.0)
    args = parser.parse_args()

    objeto = cv2.imread(args.objeto, cv2.IMREAD_GRAYSCALE)
    cena = cv2.imread(args.cena, cv2.IMREAD_GRAYSCALE)
    if objeto is None or cena is None:
        raise FileNotFoundError("Execute o exemplo 01 ou informe imagens válidas.")

    sift = cv2.SIFT_create(nfeatures=1500)
    kp_o, des_o = sift.detectAndCompute(objeto, None)
    kp_c, des_c = sift.detectAndCompute(cena, None)
    if des_o is None or des_c is None:
        raise RuntimeError("Descritores insuficientes.")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pares = matcher.knnMatch(des_o, des_c, k=2)
    bons = [m for m, n in pares if m.distance < args.ratio * n.distance]
    if len(bons) < 4:
        raise RuntimeError(f"A homografia exige ao menos 4 matches; obtidos {len(bons)}.")

    origem = np.float32([kp_o[m.queryIdx].pt for m in bons]).reshape(-1, 1, 2)
    destino = np.float32([kp_c[m.trainIdx].pt for m in bons]).reshape(-1, 1, 2)
    H, mascara = cv2.findHomography(origem, destino, cv2.RANSAC, args.reprojecao)
    if H is None or mascara is None:
        raise RuntimeError("Não foi possível estimar uma homografia consistente.")

    h, w = objeto.shape
    cantos = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    projetados = cv2.perspectiveTransform(cantos, H)
    cena_marcada = cv2.polylines(
        cv2.cvtColor(cena, cv2.COLOR_GRAY2BGR),
        [np.int32(projetados)],
        True,
        (40, 210, 40),
        4,
        cv2.LINE_AA,
    )

    inliers = mascara.ravel().astype(bool)
    desenho = cv2.drawMatches(
        objeto,
        kp_o,
        cena_marcada,
        kp_c,
        bons,
        None,
        matchesMask=inliers.astype(int).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    saida = Path("dados/gerados/homografia_ransac.png")
    cv2.imwrite(str(saida), desenho)
    print(f"Matches bons: {len(bons)}; inliers RANSAC: {int(inliers.sum())}")
    print("Matriz H:\n", np.array2string(H, precision=3, suppress_small=True))
    cv2.imshow("Homografia + RANSAC", desenho)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

