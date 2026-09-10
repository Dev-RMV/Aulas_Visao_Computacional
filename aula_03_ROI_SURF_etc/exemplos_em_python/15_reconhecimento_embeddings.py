"""Exemplo 15 — Reconhecimento facial por embeddings com face_recognition.

Cada imagem em --conhecidas deve conter uma face; o nome do arquivo vira o rótulo.
Uso:
  python exemplos/15_reconhecimento_embeddings.py \
    --conhecidas dados/faces_referencia --consulta dados/consulta_faces.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    try:
        import face_recognition
    except ImportError as erro:
        raise RuntimeError(
            "Instale requirements-opcionais.txt para executar este exemplo."
        ) from erro

    parser = argparse.ArgumentParser()
    parser.add_argument("--conhecidas", required=True)
    parser.add_argument("--consulta", required=True)
    parser.add_argument("--tolerancia", type=float, default=0.55)
    args = parser.parse_args()

    nomes: list[str] = []
    vetores: list[np.ndarray] = []
    extensoes = {".jpg", ".jpeg", ".png", ".bmp"}
    for caminho in sorted(Path(args.conhecidas).iterdir()):
        if caminho.suffix.lower() not in extensoes:
            continue
        rgb = face_recognition.load_image_file(str(caminho))
        codigos = face_recognition.face_encodings(rgb)
        if len(codigos) != 1:
            print(f"Ignorada {caminho.name}: esperada 1 face, encontradas {len(codigos)}.")
            continue
        nomes.append(caminho.stem)
        vetores.append(codigos[0])

    if not vetores:
        raise RuntimeError("Nenhuma referência válida foi carregada.")

    bgr = cv2.imread(args.consulta)
    if bgr is None:
        raise FileNotFoundError(args.consulta)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    localizacoes = face_recognition.face_locations(rgb, model="hog")
    consultas = face_recognition.face_encodings(rgb, localizacoes)

    for (topo, direita, base, esquerda), vetor in zip(localizacoes, consultas):
        distancias = face_recognition.face_distance(vetores, vetor)
        melhor = int(np.argmin(distancias))
        distancia = float(distancias[melhor])
        aceito = distancia <= args.tolerancia
        nome = nomes[melhor] if aceito else "desconhecido"
        cor = (40, 210, 40) if aceito else (40, 40, 230)
        cv2.rectangle(bgr, (esquerda, topo), (direita, base), cor, 3)
        cv2.putText(
            bgr,
            f"{nome} d={distancia:.3f}",
            (esquerda, max(25, topo - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            cor,
            2,
            cv2.LINE_AA,
        )

    destino = Path("dados/gerados/reconhecimento_embeddings.png")
    destino.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), bgr)
    print(f"Faces consultadas: {len(consultas)}; saída: {destino}")
    cv2.imshow("Embeddings faciais", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

