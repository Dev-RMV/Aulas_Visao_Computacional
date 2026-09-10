"""
Exemplo 02 - Geracao de um arquivo MP4 sintetico para testes.

Objetivo didatico:
- Criar uma fonte de video reproduzivel quando nao houver webcam disponivel.
- Introduzir VideoWriter, FPS, resolucao e sequencia de frames.

Execucao:
    python 02_gerar_video_teste.py

Saida:
    video_teste.mp4
"""

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    """Gera um video curto com formas geometricas em movimento."""

    # Parametros do video de teste.
    largura = 640
    altura = 360
    fps = 30.0
    duracao_segundos = 8
    total_frames = int(fps * duracao_segundos)

    pasta = Path(__file__).resolve().parent
    caminho_saida = pasta / "video_teste.mp4"

    # Define o codec MP4V, geralmente disponivel em instalacoes comuns.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Cria o objeto escritor de video.
    writer = cv2.VideoWriter(
        str(caminho_saida),
        fourcc,
        fps,
        (largura, altura),
    )

    # Verifica se o arquivo de video pode ser criado.
    if not writer.isOpened():
        raise RuntimeError("Nao foi possivel criar o arquivo de video MP4.")

    try:
        for numero_frame in range(total_frames):
            # Cria um fundo escuro BGR.
            frame = np.zeros((altura, largura, 3), dtype=np.uint8)
            frame[:] = (25, 25, 25)

            # Calcula uma posicao horizontal que varia com o numero do frame.
            x = int((numero_frame / (total_frames - 1)) * (largura - 100)) + 50

            # Calcula uma oscilacao vertical simples.
            y = int(altura / 2 + 90 * np.sin(numero_frame * 0.08))

            # Desenha um alvo em movimento, simulando um objeto observado pelo robo.
            cv2.circle(frame, (x, y), 32, (0, 200, 255), thickness=-1)
            cv2.circle(frame, (x, y), 12, (0, 0, 0), thickness=-1)

            # Desenha uma grade para dar referencia espacial.
            for gx in range(0, largura, 80):
                cv2.line(frame, (gx, 0), (gx, altura), (50, 50, 50), 1)
            for gy in range(0, altura, 60):
                cv2.line(frame, (0, gy), (largura, gy), (50, 50, 50), 1)

            # Insere metadados visuais no proprio frame.
            cv2.putText(
                frame,
                f"Frame: {numero_frame + 1}/{total_frames}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Resolucao: {largura}x{altura} | FPS nominal: {fps:.1f}",
                (15, altura - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Adiciona o frame ao arquivo de video.
            writer.write(frame)

    finally:
        # Garante a finalizacao correta do container de video.
        writer.release()

    print(f"Video de teste criado em: {caminho_saida}")
    print(f"Resolucao: {largura}x{altura}")
    print(f"FPS nominal: {fps}")
    print(f"Total de frames: {total_frames}")


if __name__ == "__main__":
    main()
