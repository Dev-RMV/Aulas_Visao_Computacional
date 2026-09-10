"""Exemplo 20 — SURF com tratamento didático de indisponibilidade.

SURF costuma não estar nos wheels oficiais do OpenCV por questões de distribuição.
O script tenta criar SURF e, se necessário, demonstra SIFT como alternativa.
Uso: python exemplos/20_surf_opcional.py --imagem dados/gerados/cena.png
"""

from __future__ import annotations

import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagem", default="dados/gerados/cena.png")
    parser.add_argument("--hessian", type=float, default=400.0)
    args = parser.parse_args()

    imagem = cv2.imread(args.imagem)
    if imagem is None:
        raise FileNotFoundError(args.imagem)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    algoritmo = "SURF"
    try:
        if not hasattr(cv2, "xfeatures2d"):
            raise AttributeError("cv2.xfeatures2d ausente")
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=args.hessian)
        pontos, descritores = detector.detectAndCompute(cinza, None)
    except (AttributeError, cv2.error) as erro:
        print("SURF não está disponível neste build:", erro)
        print("Alternativa prática para a aula: cv2.SIFT_create() ou cv2.ORB_create().")
        algoritmo = "SIFT (fallback)"
        detector = cv2.SIFT_create(nfeatures=600)
        pontos, descritores = detector.detectAndCompute(cinza, None)

    desenho = cv2.drawKeypoints(
        imagem, pontos, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.putText(desenho, algoritmo, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite("dados/gerados/keypoints_surf_ou_fallback.png", desenho)
    print(f"Algoritmo executado: {algoritmo}; pontos: {len(pontos)}; descritores: {None if descritores is None else descritores.shape}")
    cv2.imshow("SURF opcional", desenho)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

