"""Exemplo 07 — Criação de um par estéreo sintético e texturizado.

Objetos mais próximos recebem maior deslocamento horizontal (disparidade).
Uso: python exemplos/07_criar_par_estereo_sintetico.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def colar(destino: np.ndarray, objeto: np.ndarray, x: int, y: int) -> None:
    h, w = objeto.shape[:2]
    destino[y : y + h, x : x + w] = objeto


def main() -> None:
    rng = np.random.default_rng(12)
    h, w = 360, 640
    textura = rng.integers(70, 180, size=(h, w, 1), dtype=np.uint8)
    esquerda = np.repeat(textura, 3, axis=2)
    direita = esquerda.copy()

    objetos = []
    for cor, tamanho, texto in [
        ((30, 200, 240), (150, 110), "PERTO"),
        ((220, 110, 30), (130, 95), "MEIO"),
        ((60, 190, 70), (105, 80), "LONGE"),
    ]:
        ow, oh = tamanho
        obj = rng.integers(0, 45, size=(oh, ow, 3), dtype=np.uint8)
        obj = cv2.add(obj, np.full_like(obj, cor))
        cv2.rectangle(obj, (2, 2), (ow - 3, oh - 3), (250, 250, 250), 3)
        cv2.putText(obj, texto, (10, oh // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        objetos.append(obj)

    # (x, y, disparidade): perto desloca mais entre esquerda e direita.
    posicoes = [(90, 190, 48), (300, 120, 28), (500, 55, 12)]
    for obj, (x, y, disparidade) in zip(objetos, posicoes):
        colar(esquerda, obj, x, y)
        colar(direita, obj, x - disparidade, y)

    saida = Path("dados/gerados")
    saida.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saida / "estereo_esquerda.png"), esquerda)
    cv2.imwrite(str(saida / "estereo_direita.png"), direita)
    print("Par estéreo salvo. Disparidades nominais: 48, 28 e 12 px.")
    cv2.imshow("Esquerda | Direita", np.hstack([esquerda, direita]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

