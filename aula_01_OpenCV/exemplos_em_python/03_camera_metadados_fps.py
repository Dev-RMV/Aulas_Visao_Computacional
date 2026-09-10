"""
Exemplo 03 - Item A: câmera, exibição, metadados e FPS.

Objetivos:
- Abrir a câmera com cv2.VideoCapture.
- Ler e exibir os frames em tempo real.
- Mostrar a resolução atual.
- Contar o número de frames.
- Estimar a taxa de FPS com cv2.getTickCount().
- Encerrar ao pressionar Q.
- Liberar corretamente os recursos utilizados.
"""

import cv2


def main() -> None:
    """Executa o pipeline básico de captura utilizando a câmera."""

    # Índice da câmera:
    # 0 representa normalmente a câmera principal do computador.
    # Caso existam outras câmeras, podem ser utilizados os índices 1, 2 etc.
    indice_camera = 0

    # Cria o objeto responsável pela captura da câmera.
    cap = cv2.VideoCapture(indice_camera)

    # Verifica se a câmera foi aberta corretamente.
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        print("Verifique as permissões ou tente outro índice de câmera.")
        return

    # Estas linhas solicitam uma resolução para a câmera.
    # O driver pode aceitar, alterar ou ignorar os valores.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Variável utilizada para contar os frames capturados.
    numero_frame = 0

    # getTickCount retorna o valor atual de um contador de alta precisão.
    tick_anterior = cv2.getTickCount()

    # getTickFrequency informa quantos ticks ocorrem por segundo.
    frequencia_ticks = cv2.getTickFrequency()

    print("Câmera aberta corretamente.")
    print("Pressione Q na janela para encerrar.")
    print()
    print("Resolução atual | número do frame | FPS estimado")

    try:
        # O loop permanece ativo enquanto a câmera estiver funcionando.
        while True:

            # Lê um frame da câmera.
            #
            # ret:
            #   True  -> o frame foi capturado corretamente.
            #   False -> ocorreu uma falha na captura.
            #
            # frame:
            #   array NumPy que contém a imagem capturada.
            ret, frame = cap.read()

            # Verifica se o frame foi capturado corretamente.
            if not ret:
                print("Falha na leitura do frame.")
                break

            # Incrementa o contador de frames.
            numero_frame += 1

            # Obtém o instante atual utilizando o contador de alta precisão.
            tick_atual = cv2.getTickCount()

            # Calcula o tempo transcorrido desde o frame anterior.
            tempo_frame = (
                tick_atual - tick_anterior
            ) / frequencia_ticks

            # Atualiza o instante anterior para o próximo cálculo.
            tick_anterior = tick_atual

            # Calcula o FPS instantâneo.
            #
            # FPS = 1 / tempo necessário para processar um frame
            if tempo_frame > 0:
                fps_estimado = 1.0 / tempo_frame
            else:
                fps_estimado = 0.0

            # frame.shape normalmente possui:
            #
            # (altura, largura, canais)
            #
            # frame.shape[:2] retorna apenas altura e largura.
            altura, largura = frame.shape[:2]

            # Exibe os metadados no terminal.
            print(
                f"Resolução: {largura}x{altura} | "
                f"Frame: {numero_frame:06d} | "
                f"FPS estimado: {fps_estimado:7.2f}"
            )

            # Cria o texto que será desenhado sobre a imagem.
            texto = (
                f"{largura}x{altura} | "
                f"Frame {numero_frame} | "
                f"FPS {fps_estimado:.1f}"
            )

            # Desenha o texto sobre o frame.
            cv2.putText(
                frame,                       # Imagem
                texto,                       # Texto
                (10, 30),                    # Posição inicial
                cv2.FONT_HERSHEY_SIMPLEX,     # Fonte
                0.65,                        # Tamanho da fonte
                (0, 255, 0),                 # Cor BGR: verde
                2,                           # Espessura
                cv2.LINE_AA,                  # Suavização das letras
            )

            # Exibe o frame em uma janela.
            cv2.imshow(
                "Camera - pressione Q para encerrar",
                frame,
            )

            # Aguarda 1 milissegundo e verifica se alguma tecla foi pressionada.
            tecla = cv2.waitKey(1) & 0xFF

            # Encerra o programa quando Q ou q for pressionado.
            if tecla == ord("q") or tecla == ord("Q"):
                print("Encerramento solicitado pelo usuário.")
                break

    finally:
        # Libera o acesso à câmera.
        cap.release()

        # Fecha todas as janelas criadas pelo OpenCV.
        cv2.destroyAllWindows()

        print("Câmera liberada.")
        print("Janelas fechadas corretamente.")


# Executa a função main somente quando este arquivo é executado diretamente.
if __name__ == "__main__":
    main()
