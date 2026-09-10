"""Exemplo 16 — Detecção de keypoints e descritores SIFT.

Uso: python exemplos/16_pontos_chave_sift.py --imagem dados/gerados/cena.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagem", default="dados/gerados/cena.png")
    parser.add_argument("--max-pontos", type=int, default=500)
    args = parser.parse_args()

    imagem = cv2.imread(args.imagem)
    if imagem is None:
        raise FileNotFoundError(args.imagem)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=args.max_pontos)
    pontos, descritores = sift.detectAndCompute(cinza, None)

    desenhada = cv2.drawKeypoints(
        imagem,
        pontos,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    destino = Path("dados/gerados/keypoints_sift.png")
    cv2.imwrite(str(destino), desenhada)
    forma = None if descritores is None else descritores.shape
    print(f"Keypoints: {len(pontos)}; descritores: {forma}; dtype: {getattr(descritores, 'dtype', None)}")
    print("O tamanho do círculo representa a escala; a orientação aparece como um raio.")
    cv2.imshow("SIFT", desenhada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

