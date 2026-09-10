"""Exemplo 01 — Geração de imagens sintéticas para os exercícios.

Cria uma cena colorida, um objeto de consulta e uma versão transformada.
Assim, os primeiros exemplos não dependem de downloads externos.
Uso: python exemplos/01_gerar_cena_sintetica.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SAIDA = Path("dados/gerados")


def adicionar_textura(imagem: np.ndarray, semente: int = 7) -> np.ndarray:
    rng = np.random.default_rng(semente)
    ruido = rng.normal(0, 10, imagem.shape).astype(np.int16)
    return np.clip(imagem.astype(np.int16) + ruido, 0, 255).astype(np.uint8)


def main() -> None:
    SAIDA.mkdir(parents=True, exist_ok=True)
    cena = np.full((480, 640, 3), (238, 238, 238), dtype=np.uint8)

    # Objetos com diferentes cores, bordas e texturas.
    cv2.rectangle(cena, (60, 80), (250, 250), (255, 80, 20), -1)  # azul em BGR
    cv2.circle(cena, (430, 160), 85, (40, 210, 40), -1)
    triangulo = np.array([[330, 390], [480, 270], [570, 410]], np.int32)
    cv2.fillPoly(cena, [triangulo], (30, 50, 230))
    cv2.rectangle(cena, (70, 330), (235, 430), (20, 190, 230), -1)

    # Detalhes internos criam keypoints para SIFT/ORB.
    for x in range(85, 245, 30):
        cv2.line(cena, (x, 95), (x, 235), (250, 250, 250), 3)
    cv2.putText(
        cena,
        "CV",
        (385, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    cena = adicionar_textura(cena)

    objeto = cena[70:260, 50:260].copy()
    matriz = cv2.getRotationMatrix2D((objeto.shape[1] / 2, objeto.shape[0] / 2), 18, 0.9)
    transformado = cv2.warpAffine(
        objeto,
        matriz,
        (objeto.shape[1], objeto.shape[0]),
        borderValue=(238, 238, 238),
    )

    arquivos = {
        SAIDA / "cena.png": cena,
        SAIDA / "objeto_consulta.png": objeto,
        SAIDA / "objeto_transformado.png": transformado,
    }
    for caminho, imagem in arquivos.items():
        if not cv2.imwrite(str(caminho), imagem):
            raise OSError(f"Não foi possível salvar {caminho}")
        print(f"Salvo: {caminho} — forma {imagem.shape}")


if __name__ == "__main__":
    main()

