"""Exemplo 21 — Recuperação de imagens por similaridade local com ORB.

O score é a fração de descritores da consulta aprovados no ratio test.
Uso:
  python exemplos/21_recuperacao_imagens_orb.py \
    --consulta dados/gerados/objeto_consulta.png --banco dados/gerados
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def descritores_orb(imagem, orb):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) if imagem.ndim == 3 else imagem
    return orb.detectAndCompute(cinza, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--consulta", default="dados/gerados/objeto_consulta.png")
    parser.add_argument("--banco", default="dados/gerados")
    parser.add_argument("--ratio", type=float, default=0.78)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    consulta = cv2.imread(args.consulta)
    if consulta is None:
        raise FileNotFoundError(args.consulta)
    orb = cv2.ORB_create(nfeatures=1200)
    kp_q, des_q = descritores_orb(consulta, orb)
    if des_q is None:
        raise RuntimeError("A consulta não produziu descritores ORB.")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    resultados = []
    consulta_resolvida = Path(args.consulta).resolve()
    for caminho in sorted(Path(args.banco).iterdir()):
        if caminho.suffix.lower() not in EXTENSOES or caminho.resolve() == consulta_resolvida:
            continue
        candidata = cv2.imread(str(caminho))
        if candidata is None:
            continue
        kp_c, des_c = descritores_orb(candidata, orb)
        if des_c is None or len(des_c) < 2:
            continue
        pares = matcher.knnMatch(des_q, des_c, k=2)
        bons = [m for m, n in pares if m.distance < args.ratio * n.distance]
        score = len(bons) / max(len(des_q), 1)
        resultados.append((score, len(bons), caminho.name, candidata, len(kp_c)))

    resultados.sort(key=lambda item: (item[0], item[1]), reverse=True)
    print(f"Consulta: {len(kp_q)} keypoints, {len(des_q)} descritores")
    for posicao, (score, bons, nome, _, n_kp) in enumerate(resultados[: args.top_k], 1):
        print(f"{posicao:02d}. {nome:32s} score={score:.3f} bons={bons:3d} kp={n_kp:4d}")

    if resultados:
        melhor = resultados[0]
        visual = melhor[3].copy()
        cv2.putText(
            visual,
            f"1º {melhor[2]} | score={melhor[0]:.3f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 210, 40),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite("dados/gerados/melhor_recuperacao_orb.png", visual)
        cv2.imshow("Melhor resultado", visual)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nenhuma candidata válida encontrada no banco.")


if __name__ == "__main__":
    main()

