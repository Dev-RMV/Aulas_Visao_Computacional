"""
Exemplo 09 - Captura de uma imagem-base com a câmera.

Objetivos didáticos:
- Abrir a webcam com cv2.VideoCapture(0).
- Exibir o vídeo em tempo real.
- Capturar um frame ao pressionar S.
- Garantir que a imagem tenha pelo menos 480 x 480 pixels.
- Salvar a imagem em disco para os exemplos seguintes.

Controles:
- S: capturar e salvar.
- Q ou Esc: cancelar.
"""

from pathlib import Path

import cv2


def main() -> None:
    """Captura e salva a imagem-base utilizada nos exercícios 2 e 3."""

    indice_camera = 0
    caminho_saida = Path(__file__).resolve().parent / "imagem_base.png"

    cap = cv2.VideoCapture(indice_camera)
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    print("Pressione S para capturar uma imagem.")
    print("Pressione Q ou Esc para cancelar.")

    frame_capturado = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Falha ao ler um frame da câmera.")
                break

            exibicao = frame.copy()
            altura, largura = exibicao.shape[:2]

            cv2.putText(
                exibicao,
                f"Resolucao atual: {largura}x{altura}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                exibicao,
                "S: capturar | Q: cancelar",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Captura da imagem-base", exibicao)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("s"), ord("S")):
                frame_capturado = frame.copy()
                break
            if tecla in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if frame_capturado is None:
        print("Nenhuma imagem foi capturada.")
        return

    # Se algum dos lados for menor que 480 pixels, ampliamos proporcionalmente.
    altura, largura = frame_capturado.shape[:2]
    if altura < 480 or largura < 480:
        escala = max(480 / altura, 480 / largura)
        nova_largura = int(round(largura * escala))
        nova_altura = int(round(altura * escala))
        frame_capturado = cv2.resize(
            frame_capturado,
            (nova_largura, nova_altura),
            interpolation=cv2.INTER_CUBIC,
        )

    gravou = cv2.imwrite(str(caminho_saida), frame_capturado)
    if not gravou:
        print("Falha ao salvar a imagem.")
        return

    print(f"Imagem salva em: {caminho_saida}")
    print(f"Shape: {frame_capturado.shape}")
    print(f"Dtype: {frame_capturado.dtype}")


if __name__ == "__main__":
    main()
