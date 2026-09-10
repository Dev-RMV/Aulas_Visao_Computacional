"""
Exemplo 04 - Captura e salvamento de um único frame usando somente a câmera.

Objetivos didáticos:
- Abrir a webcam com cv2.VideoCapture.
- Descartar alguns frames iniciais para estabilizar exposição e foco.
- Capturar um único frame.
- Salvar a imagem em disco com cv2.imwrite.
- Exibir shape e dtype do array NumPy.
- Mostrar a imagem capturada em uma janela.
- Encerrar com Q ou Esc e liberar todos os recursos.
"""

from pathlib import Path

import cv2


def main() -> None:
    """Captura um frame da câmera, salva a imagem e exibe seus metadados."""

    # Índice da câmera principal do computador.
    # Caso a câmera não abra, teste os índices 1 ou 2.
    indice_camera = 0

    # Quantidade de frames iniciais descartados.
    # Isso dá tempo para a câmera ajustar foco, brilho e exposição.
    quantidade_aquecimento = 10

    # Nome do arquivo que será criado.
    nome_arquivo = "frame_capturado.png"

    # Define o caminho de saída na mesma pasta deste script.
    caminho_saida = Path(__file__).resolve().parent / nome_arquivo

    # Abre a câmera selecionada.
    cap = cv2.VideoCapture(indice_camera)

    # Verifica se a câmera foi aberta corretamente.
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        print("Verifique as permissões ou tente outro índice de câmera.")
        return

    # Variável que armazenará o frame escolhido.
    frame_capturado = None

    try:
        # Lê e descarta alguns frames iniciais.
        for numero in range(quantidade_aquecimento):
            ret, frame = cap.read()

            # ret será False se ocorrer uma falha de captura.
            if not ret:
                print("Falha ao capturar um frame da câmera.")
                return

            # Mantém o frame mais recente.
            frame_capturado = frame

            print(
                f"Aquecimento da câmera: "
                f"{numero + 1}/{quantidade_aquecimento}"
            )

        # Verifica se algum frame válido foi obtido.
        if frame_capturado is None:
            print("Nenhum frame foi capturado.")
            return

        # Salva o frame em disco.
        # cv2.imwrite retorna True quando a gravação é bem-sucedida.
        gravou = cv2.imwrite(
            str(caminho_saida),
            frame_capturado
        )

        # Verifica o resultado da gravação.
        if not gravou:
            print(f"Falha ao salvar a imagem em: {caminho_saida}")
            return

        # Exibe informações sobre o array NumPy da imagem.
        print()
        print("Imagem salva com sucesso.")
        print(f"Arquivo: {caminho_saida}")
        print(f"Shape: {frame_capturado.shape}")
        print(f"Dtype: {frame_capturado.dtype}")
        print("Pressione Q ou Esc para fechar a janela.")

        # Mostra o frame capturado em uma janela.
        cv2.imshow(
            "Frame capturado - pressione Q ou Esc",
            frame_capturado
        )

        # Mantém a janela aberta até o usuário pressionar Q ou Esc.
        while True:
            tecla = cv2.waitKey(30) & 0xFF

            if tecla in (ord("q"), ord("Q"), 27):
                break

    finally:
        # Libera o acesso à câmera.
        cap.release()

        # Fecha todas as janelas abertas pelo OpenCV.
        cv2.destroyAllWindows()

        print("Câmera e janelas liberadas corretamente.")


# Executa main somente quando este arquivo é executado diretamente.
if __name__ == "__main__":
    main()
