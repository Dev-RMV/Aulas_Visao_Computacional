"""
Exemplo 00 - Verificacao do ambiente Python, OpenCV e NumPy.

Objetivo didatico:
- Confirmar que as bibliotecas foram instaladas corretamente.
- Exibir as versoes utilizadas.
- Criar e salvar uma imagem simples sem depender de uma camera.

Execucao:
    python 00_verificar_ambiente.py
"""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    """Executa uma verificacao minima do ambiente."""

    # Exibe as versoes para facilitar a reproducao e o diagnostico.
    print(f"Versao do OpenCV: {cv2.__version__}")
    print(f"Versao do NumPy:  {np.__version__}")

    # Cria uma imagem preta com 240 linhas, 320 colunas e 3 canais BGR.
    # O dtype uint8 representa cada canal com valores inteiros de 0 a 255.
    imagem = np.zeros((240, 320, 3), dtype=np.uint8)

    # Desenha elementos simples para confirmar que o modulo imgproc funciona.
    cv2.rectangle(imagem, (20, 20), (300, 220), (0, 180, 0), thickness=3)
    cv2.circle(imagem, (160, 120), 55, (255, 0, 0), thickness=-1)
    cv2.putText(
        imagem,
        "OpenCV OK",
        (75, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Define o caminho de saida na mesma pasta do script.
    caminho_saida = Path(__file__).resolve().parent / "teste_ambiente.png"

    # cv2.imwrite retorna True quando a gravacao foi concluida.
    gravou = cv2.imwrite(str(caminho_saida), imagem)

    print(f"Shape da imagem criada: {imagem.shape}")
    print(f"Dtype da imagem criada: {imagem.dtype}")
    print(f"Arquivo gravado: {gravou} -> {caminho_saida}")

    if not gravou:
        raise RuntimeError("Nao foi possivel salvar a imagem de teste.")

    print("Ambiente basico validado com sucesso.")


if __name__ == "__main__":
    main()
