"""Exemplo 27 — Pipeline integrado OpenCV + CNN Keras em uma ROI da webcam.

Mostre na ROI uma peça/figura semelhante ao Fashion-MNIST. OpenCV recorta e
pré-processa; Keras classifica. Teclas: q encerra; i alterna inversão da máscara.
Uso: python exemplos/27_pipeline_opencv_cnn_webcam.py --modelo modelos/cnn_fashion.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras


CLASSES = [
    "camiseta/top",
    "calça",
    "pulôver",
    "vestido",
    "casaco",
    "sandália",
    "camisa",
    "tênis",
    "bolsa",
    "bota",
]


def preparar(roi: np.ndarray, inverter: bool) -> tuple[np.ndarray, np.ndarray]:
    cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (5, 5), 0)
    tipo = cv2.THRESH_BINARY_INV if inverter else cv2.THRESH_BINARY
    _, binaria = cv2.threshold(cinza, 0, 255, tipo + cv2.THRESH_OTSU)
    imagem_28 = cv2.resize(binaria, (28, 28), interpolation=cv2.INTER_AREA)
    tensor = (imagem_28.astype("float32") / 255.0)[None, ..., None]
    return tensor, imagem_28


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="modelos/cnn_fashion.keras")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.modelo).exists():
        raise FileNotFoundError("Treine o exemplo 23 ou informe um modelo compatível.")
    modelo = keras.models.load_model(args.modelo)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Câmera indisponível.")

    inverter = True
    try:
        while True:
            ok, quadro = camera.read()
            if not ok:
                break
            h, w = quadro.shape[:2]
            lado = int(min(h, w) * 0.55)
            x1, y1 = (w - lado) // 2, (h - lado) // 2
            x2, y2 = x1 + lado, y1 + lado
            roi = quadro[y1:y2, x1:x2]
            tensor, imagem_28 = preparar(roi, inverter)
            probabilidades = modelo.predict(tensor, verbose=0)[0]
            indice = int(np.argmax(probabilidades))
            certeza = float(probabilidades[indice])

            cv2.rectangle(quadro, (x1, y1), (x2, y2), (40, 210, 40), 3)
            texto = f"{CLASSES[indice]} — {100 * certeza:.1f}%"
            cv2.putText(quadro, texto, (x1, max(30, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 210, 40), 2)
            mini = cv2.resize(imagem_28, (140, 140), interpolation=cv2.INTER_NEAREST)
            mini = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
            quadro[10:150, 10:150] = mini
            cv2.imshow("ROI + CNN — q sair | i inverter", quadro)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("i"):
                inverter = not inverter
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

