"""Exemplo 12 — Coleta consentida de faces para o LBPH.

Salva recortes alinhados apenas por bounding box; procure variar expressão e pose.
Teclas: c captura; q encerra.
Uso: python exemplos/12_coletar_faces_webcam.py --nome aluno_01 --quantidade 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def normalizar_nome(nome: str) -> str:
    seguro = "".join(c for c in nome.strip().lower() if c.isalnum() or c in "_-")
    if not seguro:
        raise ValueError("Use um nome não vazio com letras, números, '_' ou '-'.")
    return seguro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nome", required=True, help="Identificador consentido da pessoa.")
    parser.add_argument("--quantidade", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    nome = normalizar_nome(args.nome)
    pasta = Path("dados/faces_treino") / nome
    pasta.mkdir(parents=True, exist_ok=True)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Câmera indisponível.")

    contador = len(list(pasta.glob("*.png")))
    try:
        while contador < args.quantidade:
            ok, quadro = camera.read()
            if not ok:
                break
            cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                cv2.equalizeHist(cinza), 1.1, 5, minSize=(80, 80)
            )
            # Para a coleta, usamos a maior face do quadro.
            maior = max(faces, key=lambda f: f[2] * f[3]) if len(faces) else None
            if maior is not None:
                x, y, w, h = maior
                cv2.rectangle(quadro, (x, y), (x + w, y + h), (40, 210, 40), 2)

            cv2.putText(
                quadro,
                f"{nome}: {contador}/{args.quantidade} — c captura | q encerra",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Coleta consentida", quadro)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("c") and maior is not None:
                x, y, w, h = maior
                face = cinza[y : y + h, x : x + w]
                face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
                face = cv2.equalizeHist(face)
                cv2.imwrite(str(pasta / f"{contador:03d}.png"), face)
                contador += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Amostras salvas em {pasta}: {contador}")


if __name__ == "__main__":
    main()

