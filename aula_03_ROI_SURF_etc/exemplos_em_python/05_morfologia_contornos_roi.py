"""Exemplo 05 — Morfologia, contornos e bounding boxes.

Parte da máscara HSV, remove ruído e produz ROIs dos objetos encontrados.
Uso: python exemplos/05_morfologia_contornos_roi.py --imagem dados/gerados/cena.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagem", default="dados/gerados/cena.png")
    parser.add_argument("--area-minima", type=float, default=1000.0)
    args = parser.parse_args()

    imagem = cv2.imread(args.imagem)
    if imagem is None:
        raise FileNotFoundError(args.imagem)
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, np.array([85, 60, 30]), np.array([140, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    abertura = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
    limpa = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    anotada = imagem.copy()
    saida = Path("dados/gerados/rois")
    saida.mkdir(parents=True, exist_ok=True)

    aceitos = 0
    for contorno in sorted(contornos, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contorno)
        if area < args.area_minima:
            continue
        x, y, largura, altura = cv2.boundingRect(contorno)
        roi = imagem[y : y + altura, x : x + largura]
        cv2.imwrite(str(saida / f"roi_{aceitos:02d}.png"), roi)
        cv2.rectangle(anotada, (x, y), (x + largura, y + altura), (0, 0, 0), 3)
        cv2.putText(
            anotada,
            f"area={area:.0f}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        aceitos += 1

    cv2.imwrite("dados/gerados/mascara_morfologica.png", limpa)
    cv2.imwrite("dados/gerados/contornos_rois.png", anotada)
    print(f"Contornos totais: {len(contornos)}; ROIs aceitas: {aceitos}")
    cv2.imshow("Máscara limpa", limpa)
    cv2.imshow("ROIs", anotada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

