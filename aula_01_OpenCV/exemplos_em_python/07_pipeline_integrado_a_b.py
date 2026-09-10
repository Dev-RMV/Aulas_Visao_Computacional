"""
Exemplo 07 - Pipeline integrado dos Itens A e B usando somente a câmera.

Objetivos didáticos:
- Abrir diretamente a webcam com cv2.VideoCapture(0).
- Exibir o stream da câmera em tempo real.
- Mostrar resolução, número do frame e FPS estimado.
- Salvar o frame exibido ao pressionar S.
- Encerrar o stream ao pressionar Q ou Esc.
- Recarregar o último frame salvo.
- Converter a imagem para escala de cinza.
- Exibir as versões colorida e cinza lado a lado.

Controles:
- S: captura e salva o frame atual.
- Q ou Esc: encerra o stream.
"""

from collections import deque
from pathlib import Path
from typing import Deque, Optional

import cv2


def calcular_media(valores: Deque[float]) -> float:
    """Calcula a média dos valores armazenados na fila."""

    # Quando a fila estiver vazia, retorna zero para evitar divisão por zero.
    if not valores:
        return 0.0

    return sum(valores) / len(valores)


def main() -> None:
    """Executa o pipeline integrado utilizando somente a webcam."""

    # Índice da câmera principal.
    # Caso ela não abra, teste os índices 1 ou 2.
    indice_camera = 0

    # Quantidade de medidas utilizadas na média móvel do FPS.
    tamanho_janela_fps = 30

    # Caminho da imagem que será salva.
    caminho_saida = (
        Path(__file__).resolve().parent
        / "snapshot_integrado.png"
    )

    # Abre diretamente a câmera.
    cap = cv2.VideoCapture(indice_camera)

    # Verifica se a câmera foi aberta corretamente.
    if not cap.isOpened():
        print("Não foi possível abrir a câmera.")
        print("Verifique as permissões ou tente outro índice.")
        return

    # Contador dos frames capturados.
    numero_frame = 0

    # Armazena o caminho do último snapshot salvo.
    ultimo_snapshot: Optional[Path] = None

    # A fila guarda as medidas recentes de FPS.
    # A média móvel reduz oscilações muito rápidas no valor exibido.
    historico_fps: Deque[float] = deque(
        maxlen=tamanho_janela_fps
    )

    # Inicia o contador de alta precisão.
    tick_anterior = cv2.getTickCount()

    # Informa quantos ticks ocorrem por segundo.
    frequencia_ticks = cv2.getTickFrequency()

    print("Câmera aberta corretamente.")
    print("Pressione S para capturar o frame atual.")
    print("Pressione Q ou Esc para encerrar.")

    try:
        while True:
            # Captura um frame da câmera.
            ret, frame = cap.read()

            # ret será False se houver falha na captura.
            if not ret:
                print("Falha ao capturar um frame da câmera.")
                break

            # Incrementa o contador de frames.
            numero_frame += 1

            # Obtém o valor atual do contador de alta precisão.
            tick_atual = cv2.getTickCount()

            # Calcula o tempo transcorrido desde o frame anterior.
            tempo_frame = (
                tick_atual - tick_anterior
            ) / frequencia_ticks

            # Atualiza o instante anterior.
            tick_anterior = tick_atual

            # Calcula o FPS instantâneo.
            if tempo_frame > 0:
                fps_instantaneo = 1.0 / tempo_frame
            else:
                fps_instantaneo = 0.0

            # Adiciona a medida atual ao histórico.
            historico_fps.append(fps_instantaneo)

            # Calcula a média móvel do FPS.
            fps_medio = calcular_media(historico_fps)

            # Obtém altura e largura diretamente do array NumPy.
            altura, largura = frame.shape[:2]

            # Exibe os metadados no terminal.
            print(
                f"Resolução: {largura}x{altura} | "
                f"Frame: {numero_frame:06d} | "
                f"FPS instantâneo: {fps_instantaneo:7.2f} | "
                f"FPS médio: {fps_medio:7.2f}"
            )

            # Cria uma cópia para desenhar textos.
            # O frame original será preservado para o salvamento.
            exibicao = frame.copy()

            # Mostra os metadados sobre a imagem.
            cv2.putText(
                exibicao,
                (
                    f"{largura}x{altura} | "
                    f"Frame {numero_frame} | "
                    f"FPS {fps_medio:.1f}"
                ),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Mostra os controles na parte inferior da imagem.
            cv2.putText(
                exibicao,
                "S: capturar | Q ou Esc: sair",
                (10, altura - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Exibe a câmera em tempo real.
            cv2.imshow(
                "Pipeline integrado A+B",
                exibicao
            )

            # Aguarda brevemente por uma tecla.
            tecla = cv2.waitKey(1) & 0xFF

            # Ao pressionar S, salva o frame original.
            if tecla in (ord("s"), ord("S")):
                gravou = cv2.imwrite(
                    str(caminho_saida),
                    frame
                )

                if gravou:
                    ultimo_snapshot = caminho_saida
                    print(f"Snapshot salvo em: {caminho_saida}")
                    break
                else:
                    print(
                        "Não foi possível salvar o snapshot."
                    )

            # Q ou Esc encerra sem salvar.
            if tecla in (ord("q"), ord("Q"), 27):
                print("Encerramento solicitado pelo usuário.")
                break

    finally:
        # Libera o acesso à câmera.
        cap.release()

        # Fecha a janela do stream.
        cv2.destroyAllWindows()

        print("Câmera liberada.")

    # Se nenhum frame foi salvo, não há imagem para processar.
    if ultimo_snapshot is None:
        print(
            "Nenhum snapshot foi salvo. "
            "A etapa de comparação não será executada."
        )
        return

    # Recarrega o snapshot salvo em modo colorido.
    colorida = cv2.imread(
        str(ultimo_snapshot),
        cv2.IMREAD_COLOR
    )

    # Verifica se a imagem foi recarregada.
    if colorida is None:
        print("Não foi possível recarregar o snapshot.")
        return

    # Converte a imagem de BGR para escala de cinza.
    cinza = cv2.cvtColor(
        colorida,
        cv2.COLOR_BGR2GRAY
    )

    # Exibe shape e dtype no terminal.
    print()
    print("Arrays do snapshot:")
    print(
        f"Colorida -> shape: {colorida.shape}, "
        f"dtype: {colorida.dtype}"
    )
    print(
        f"Cinza    -> shape: {cinza.shape}, "
        f"dtype: {cinza.dtype}"
    )

    # O cv2.hconcat exige imagens com o mesmo número de canais.
    # A imagem cinza é convertida temporariamente para três canais.
    cinza_bgr = cv2.cvtColor(
        cinza,
        cv2.COLOR_GRAY2BGR
    )

    # Junta as duas imagens horizontalmente.
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

    print("Pressione Q ou Esc para fechar o painel.")

    # Exibe a comparação lado a lado.
    cv2.imshow(
        "Snapshot colorido e cinza",
        painel
    )

    # Mantém o painel aberto até Q ou Esc.
    while True:
        tecla = cv2.waitKey(30) & 0xFF

        if tecla in (ord("q"), ord("Q"), 27):
            break

    # Fecha todas as janelas restantes.
    cv2.destroyAllWindows()

    print("Janelas fechadas corretamente.")


# Executa main somente quando o arquivo é executado diretamente.
if __name__ == "__main__":
    main()
