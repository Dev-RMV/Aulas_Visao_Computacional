"""
Exemplo 06 - Item B completo usando somente a câmera.

Objetivos didáticos:
1. Abrir a webcam com cv2.VideoCapture(0).
2. Exibir o vídeo da câmera em tempo real.
3. Capturar o frame exibido ao pressionar S.
4. Salvar o frame com cv2.imwrite.
5. Recarregar a imagem com cv2.imread.
6. Converter a imagem para escala de cinza.
7. Imprimir shape e dtype das imagens.
8. Exibir as versões colorida e cinza lado a lado com cv2.hconcat.

Comandos:
- Pressione S para capturar e salvar a imagem.
- Pressione Q ou Esc para encerrar sem capturar.
"""

from pathlib import Path

import cv2


def main() -> None:
    """Executa todas as etapas do Item B usando apenas a webcam."""

    # Índice 0 normalmente representa a câmera principal do computador.
    # Caso ela não abra, teste os índices 1 ou 2.
    indice_camera = 0

    # Nome e caminho do arquivo que será salvo.
    caminho_saida = Path(__file__).resolve().parent / "item_b_frame.png"

    # Abre a câmera.
    cap = cv2.VideoCapture(indice_camera)

    # Verifica se a câmera foi aberta corretamente.
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        print("Verifique as permissões ou tente outro índice de câmera.")
        return

    # Variável que receberá uma cópia do frame escolhido pelo usuário.
    frame_capturado = None

    print("Câmera aberta corretamente.")
    print("Pressione S para capturar a imagem.")
    print("Pressione Q ou Esc para sair sem capturar.")

    try:
        # Mantém a câmera em funcionamento até o usuário escolher uma ação.
        while True:
            # Lê um frame da câmera.
            ret, frame = cap.read()

            # ret será False se ocorrer uma falha na captura.
            if not ret:
                print("Falha ao obter um frame da câmera.")
                break

            # Cria uma cópia para desenhar as instruções sem alterar
            # o frame original que poderá ser salvo.
            frame_exibicao = frame.copy()

            # Desenha as instruções sobre a imagem exibida.
            cv2.putText(
                frame_exibicao,
                "S: capturar | Q ou Esc: sair",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Mostra o vídeo da câmera em tempo real.
            cv2.imshow(
                "Item B - pressione S para capturar",
                frame_exibicao
            )

            # Aguarda 1 milissegundo e verifica a tecla pressionada.
            tecla = cv2.waitKey(1) & 0xFF

            # Ao pressionar S, copia o frame atual da câmera.
            if tecla in (ord("s"), ord("S")):
                frame_capturado = frame.copy()
                print("Frame capturado.")
                break

            # Q ou Esc encerra o programa sem realizar a captura.
            if tecla in (ord("q"), ord("Q"), 27):
                print("Programa encerrado sem capturar uma imagem.")
                break

    finally:
        # Libera a câmera mesmo se ocorrer algum erro.
        cap.release()

        # Fecha a janela de visualização da câmera.
        cv2.destroyAllWindows()

        print("Câmera liberada.")

    # Se o usuário saiu sem pressionar S, não há imagem para processar.
    if frame_capturado is None:
        return

    # Salva o frame capturado em disco.
    gravou = cv2.imwrite(
        str(caminho_saida),
        frame_capturado
    )

    # Verifica se a gravação foi realizada corretamente.
    if not gravou:
        print(f"Não foi possível salvar a imagem em: {caminho_saida}")
        return

    print(f"Imagem salva em: {caminho_saida}")

    # Recarrega explicitamente a imagem salva em modo colorido.
    colorida = cv2.imread(
        str(caminho_saida),
        cv2.IMREAD_COLOR
    )

    # Verifica se a imagem foi recarregada corretamente.
    if colorida is None:
        print("A imagem foi salva, mas não pôde ser recarregada.")
        return

    # Converte a imagem colorida de BGR para escala de cinza.
    cinza = cv2.cvtColor(
        colorida,
        cv2.COLOR_BGR2GRAY
    )

    # Exibe os metadados dos arrays NumPy.
    print()
    print("Metadados dos arrays NumPy:")
    print(
        f"Colorida -> shape: {colorida.shape}, "
        f"dtype: {colorida.dtype}"
    )
    print(
        f"Cinza    -> shape: {cinza.shape}, "
        f"dtype: {cinza.dtype}"
    )

    # A imagem colorida possui três canais: (altura, largura, 3).
    # A imagem cinza possui apenas dois eixos: (altura, largura).
    #
    # Para usar cv2.hconcat, as duas imagens precisam ter a mesma
    # quantidade de canais. Por isso, convertemos temporariamente
    # a imagem cinza para BGR apenas para montar o painel.
    cinza_bgr = cv2.cvtColor(
        cinza,
        cv2.COLOR_GRAY2BGR
    )

    # Junta horizontalmente a imagem colorida e a versão cinza.
    painel = cv2.hconcat(
        [colorida, cinza_bgr]
    )

    # Adiciona identificações ao painel.
    cv2.putText(
        painel,
        "Colorida",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        painel,
        "Escala de cinza",
        (colorida.shape[1] + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    print()
    print("Pressione Q ou Esc para fechar o painel.")

    # Exibe as duas versões lado a lado.
    cv2.imshow(
        "Item B - colorida e cinza",
        painel
    )

    # Mantém o painel aberto até Q ou Esc ser pressionado.
    while True:
        tecla = cv2.waitKey(30) & 0xFF

        if tecla in (ord("q"), ord("Q"), 27):
            break

    # Fecha todas as janelas restantes do OpenCV.
    cv2.destroyAllWindows()

    print("Janelas fechadas corretamente.")


# Executa main somente quando o arquivo for executado diretamente.
if __name__ == "__main__":
    main()
