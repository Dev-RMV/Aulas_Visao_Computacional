"""Exemplo 00 — Verificação do ambiente.

Objetivo: confirmar versões, câmera e recursos opcionais antes da aula.
Uso: python exemplos/00_verificar_ambiente.py
"""

from __future__ import annotations

import importlib
import platform
import sys


def versao(modulo: str) -> str:
    """Importa um módulo e devolve sua versão ou uma mensagem de ausência."""
    try:
        pacote = importlib.import_module(modulo)
        return str(getattr(pacote, "__version__", "instalado"))
    except Exception as erro:  # diagnóstico: queremos exibir qualquer falha
        return f"não disponível ({type(erro).__name__})"


def main() -> None:
    print("=== Ambiente de visão computacional ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Sistema: {platform.system()} {platform.release()}")
    for nome in ("numpy", "cv2", "tensorflow", "matplotlib", "sklearn"):
        print(f"{nome:12s}: {versao(nome)}")

    try:
        import cv2

        print("\n=== Recursos do OpenCV ===")
        print("SIFT:", hasattr(cv2, "SIFT_create"))
        print("ORB:", hasattr(cv2, "ORB_create"))
        print("Módulo cv2.face:", hasattr(cv2, "face"))
        print(
            "SURF:",
            hasattr(cv2, "xfeatures2d")
            and hasattr(cv2.xfeatures2d, "SURF_create"),
        )

        camera = cv2.VideoCapture(0)
        aberta = camera.isOpened()
        ok, _ = camera.read() if aberta else (False, None)
        camera.release()
        print("Câmera 0:", "pronta" if ok else "não detectada ou sem permissão")
    except ImportError:
        print("\nInstale requirements-base.txt para testar os recursos do OpenCV.")


if __name__ == "__main__":
    main()

