"""Exemplo 11 — Haar Cascade em vídeo.

Teclas: q encerra; e alterna equalização de histograma.
Uso: python exemplos/11_detectar_faces_haar_webcam.py --camera 0
"""

from __future__ import annotations

import argparse
import time

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--largura", type=int, default=640)
    args = parser.parse_args()

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Câmera indisponível ou sem permissão.")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.largura)

    equalizar = True
    instante_anterior = time.perf_counter()
    fps_suave = 0.0
    try:
        while True:
            ok, quadro = camera.read()
            if not ok:
                break
            cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
            entrada = cv2.equalizeHist(cinza) if equalizar else cinza
            faces = detector.detectMultiScale(
                entrada, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            for x, y, w, h in faces:
                cv2.rectangle(quadro, (x, y), (x + w, y + h), (40, 210, 40), 2)

            agora = time.perf_counter()
            fps = 1.0 / max(agora - instante_anterior, 1e-6)
            instante_anterior = agora
            fps_suave = fps if fps_suave == 0 else 0.9 * fps_suave + 0.1 * fps
            cv2.putText(
                quadro,
                f"faces={len(faces)}  FPS={fps_suave:.1f}  equalizar={equalizar}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Haar em tempo real — q: sair | e: equalizar", quadro)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("e"):
                equalizar = not equalizar
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

