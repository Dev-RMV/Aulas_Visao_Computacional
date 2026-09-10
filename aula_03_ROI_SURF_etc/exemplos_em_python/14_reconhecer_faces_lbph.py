"""Exemplo 14 — Reconhecimento facial LBPH em tempo real.

No LBPH, a "confiança" retornada funciona como distância: menor é melhor.
Uso: python exemplos/14_reconhecer_faces_lbph.py --limiar 65
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="modelos/lbph.yml")
    parser.add_argument("--mapa", default="modelos/lbph_rotulos.json")
    parser.add_argument("--limiar", type=float, default=65.0)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not hasattr(cv2, "face"):
        raise RuntimeError("Instale opencv-contrib-python para usar cv2.face.")
    if not Path(args.modelo).exists() or not Path(args.mapa).exists():
        raise FileNotFoundError("Execute o exemplo 13 antes.")

    mapa = json.loads(Path(args.mapa).read_text(encoding="utf-8"))
    reconhecedor = cv2.face.LBPHFaceRecognizer_create()
    reconhecedor.read(args.modelo)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Câmera indisponível.")

    try:
        while True:
            ok, quadro = camera.read()
            if not ok:
                break
            cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(cinza, 1.1, 5, minSize=(70, 70))
            for x, y, w, h in faces:
                face = cv2.resize(cinza[y : y + h, x : x + w], (160, 160))
                face = cv2.equalizeHist(face)
                rotulo, distancia = reconhecedor.predict(face)
                nome = mapa.get(str(rotulo), "desconhecido")
                aceito = distancia <= args.limiar
                texto = f"{nome if aceito else 'desconhecido'} d={distancia:.1f}"
                cor = (40, 210, 40) if aceito else (40, 40, 230)
                cv2.rectangle(quadro, (x, y), (x + w, y + h), cor, 2)
                cv2.putText(quadro, texto, (x, max(25, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
            cv2.imshow("LBPH — q encerra", quadro)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

