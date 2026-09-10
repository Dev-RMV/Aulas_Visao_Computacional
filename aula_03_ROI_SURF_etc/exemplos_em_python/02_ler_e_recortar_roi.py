"""Exemplo 02 — Leitura, inspeção e recorte de uma ROI.

Uso:
  python exemplos/02_ler_e_recortar_roi.py --imagem dados/gerados/cena.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recorta uma região de interesse.")
    parser.add_argument("--imagem", default="dados/gerados/cena.png")
    parser.add_argument("--x", type=int, default=50)
    parser.add_argument("--y", type=int, default=70)
    parser.add_argument("--largura", type=int, default=210)
    parser.add_argument("--altura", type=int, default=190)
    return parser.parse_args()


def main() -> None:
    args = argumentos()
    imagem = cv2.imread(args.imagem)
    if imagem is None:
        raise FileNotFoundError(
            f"Imagem não encontrada: {args.imagem}. Execute o exemplo 01 primeiro."
        )

    altura_img, largura_img = imagem.shape[:2]
    x1 = max(0, min(args.x, largura_img - 1))
    y1 = max(0, min(args.y, altura_img - 1))
    x2 = max(x1 + 1, min(x1 + args.largura, largura_img))
    y2 = max(y1 + 1, min(y1 + args.altura, altura_img))
    roi = imagem[y1:y2, x1:x2]

    marcada = imagem.copy()
    cv2.rectangle(marcada, (x1, y1), (x2, y2), (0, 0, 0), 3)

    saida = Path("dados/gerados")
    saida.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saida / "roi.png"), roi)
    cv2.imwrite(str(saida / "cena_com_roi.png"), marcada)

    print(f"Imagem: {largura_img}×{altura_img} px, dtype={imagem.dtype}")
    print(f"ROI: x={x1}:{x2}, y={y1}:{y2}, forma={roi.shape}")
    print("Lembrete: NumPy indexa como imagem[y, x], não imagem[x, y].")

    cv2.imshow("Cena com ROI", marcada)
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

