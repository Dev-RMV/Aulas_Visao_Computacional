"""Exemplo 06 — Segmentação de primeiro plano com GrabCut.

O retângulo inicial deve conter o objeto e uma pequena borda de fundo.
Uso: python exemplos/06_grabcut_roi.py --imagem dados/gerados/cena.png
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagem", default="dados/gerados/cena.png")
    parser.add_argument("--x", type=int, default=45)
    parser.add_argument("--y", type=int, default=65)
    parser.add_argument("--largura", type=int, default=225)
    parser.add_argument("--altura", type=int, default=210)
    parser.add_argument("--iteracoes", type=int, default=5)
    args = parser.parse_args()

    imagem = cv2.imread(args.imagem)
    if imagem is None:
        raise FileNotFoundError(args.imagem)

    mascara = np.zeros(imagem.shape[:2], np.uint8)
    modelo_fundo = np.zeros((1, 65), np.float64)
    modelo_frente = np.zeros((1, 65), np.float64)
    retangulo = (args.x, args.y, args.largura, args.altura)

    cv2.grabCut(
        imagem,
        mascara,
        retangulo,
        modelo_fundo,
        modelo_frente,
        args.iteracoes,
        cv2.GC_INIT_WITH_RECT,
    )
    frente = np.where(
        (mascara == cv2.GC_FGD) | (mascara == cv2.GC_PR_FGD), 1, 0
    ).astype("uint8")
    resultado = imagem * frente[:, :, np.newaxis]

    cv2.imwrite("dados/gerados/mascara_grabcut.png", frente * 255)
    cv2.imwrite("dados/gerados/resultado_grabcut.png", resultado)
    print("Classes da máscara:", dict(zip(*np.unique(mascara, return_counts=True))))
    cv2.imshow("GrabCut", np.hstack([imagem, resultado]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

