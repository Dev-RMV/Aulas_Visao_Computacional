"""Exemplo 13 — Treinamento de reconhecimento facial LBPH.

Pré-requisito: execute o exemplo 12 para ao menos duas pessoas.
Uso: python exemplos/13_treinar_reconhecedor_lbph.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", default="dados/faces_treino")
    parser.add_argument("--modelo", default="modelos/lbph.yml")
    parser.add_argument("--mapa", default="modelos/lbph_rotulos.json")
    args = parser.parse_args()

    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "cv2.face não está disponível. Instale apenas opencv-contrib-python."
        )

    raiz = Path(args.dados)
    pessoas = sorted(p for p in raiz.iterdir() if p.is_dir()) if raiz.exists() else []
    imagens: list[np.ndarray] = []
    rotulos: list[int] = []
    mapa: dict[int, str] = {}

    for indice, pasta in enumerate(pessoas):
        mapa[indice] = pasta.name
        for caminho in sorted(pasta.glob("*.png")):
            face = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
            if face is None:
                continue
            face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
            imagens.append(face)
            rotulos.append(indice)

    if len(mapa) < 2 or len(imagens) < 10:
        raise RuntimeError("Colete ao menos duas pessoas e várias amostras por pessoa.")

    reconhecedor = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )
    reconhecedor.train(imagens, np.asarray(rotulos, dtype=np.int32))
    Path(args.modelo).parent.mkdir(parents=True, exist_ok=True)
    reconhecedor.write(args.modelo)
    Path(args.mapa).write_text(
        json.dumps(mapa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Modelo salvo: {args.modelo}")
    print(f"Pessoas: {len(mapa)}; amostras: {len(imagens)}")


if __name__ == "__main__":
    main()

