"""Exemplo 10 — Detecção de faces em imagem com Haar Cascade.

Detecção responde "onde há uma face?"; não responde "quem é a pessoa?".
Uso: python exemplos/10_detectar_faces_haar_imagem.py --imagem sua_foto.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagem", required=True)
    parser.add_argument("--escala", type=float, default=1.1)
    parser.add_argument("--vizinhos", type=int, default=5)
    args = parser.parse_args()

    imagem = cv2.imread(args.imagem)
    if imagem is None:
        raise FileNotFoundError(args.imagem)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    cinza = cv2.equalizeHist(cinza)

    caminho = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(caminho)
    if detector.empty():
        raise RuntimeError(f"Não foi possível carregar o cascade: {caminho}")

    faces = detector.detectMultiScale(
        cinza,
        scaleFactor=args.escala,
        minNeighbors=args.vizinhos,
        minSize=(40, 40),
    )
    saida = imagem.copy()
    for indice, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(saida, (x, y), (x + w, y + h), (40, 210, 40), 3)
        cv2.putText(
            saida,
            f"face {indice}",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 210, 40),
            2,
            cv2.LINE_AA,
        )

    destino = Path("dados/gerados/faces_haar.png")
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), saida)
    print(f"Faces encontradas: {len(faces)}. Resultado: {destino}")
    cv2.imshow("Haar Cascade", saida)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

