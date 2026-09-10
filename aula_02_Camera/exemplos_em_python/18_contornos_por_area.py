"""
Exemplo 18 corrigido - Detecção e classificação de contornos por área.

Problema corrigido:
O detector Canny produz linhas finas de borda. Quando essas linhas não formam
regiões perfeitamente fechadas, cv2.contourArea() pode retornar zero ou uma
área muito pequena. Nesse caso, todos os contornos podem ser descartados pelo
filtro de área superior a 200 px².

Solução adotada:
1. Capturar uma imagem diretamente da câmera.
2. Converter para escala de cinza.
3. Aplicar suavização gaussiana.
4. Criar uma máscara binária com o método de Otsu.
5. Escolher automaticamente entre a máscara normal e a invertida.
6. Aplicar operações morfológicas para remover ruídos e fechar falhas.
7. Usar cv2.findContours() sobre a máscara binária preenchida.
8. Filtrar contornos com área superior a 200 px².
9. Desenhar os contornos com cores diferentes conforme a área.

Faixas de área:
- Pequeno: de 200 até 999 px², em verde.
- Médio: de 1.000 até 4.999 px², em amarelo.
- Grande: 5.000 px² ou mais, em vermelho.

Controles:
- S: captura a imagem da câmera.
- Q ou Esc: encerra sem capturar.
"""

import cv2
import numpy as np


# Área mínima exigida no exercício.
AREA_MINIMA = 200.0


def capturar_imagem_camera(indice_camera: int = 0):
    """Abre a câmera e retorna o frame capturado ao pressionar S."""

    # Abre a câmera principal do computador.
    cap = cv2.VideoCapture(indice_camera)

    # Verifica se a câmera foi aberta corretamente.
    if not cap.isOpened():
        raise RuntimeError(
            "Não foi possível abrir a câmera. "
            "Verifique as permissões ou teste outro índice."
        )

    print("Câmera aberta.")
    print("Pressione S para capturar a imagem.")
    print("Pressione Q ou Esc para cancelar.")

    imagem_capturada = None

    try:
        while True:
            # Lê um frame da câmera.
            ret, frame = cap.read()

            if not ret:
                raise RuntimeError("Falha ao capturar um frame da câmera.")

            # Cria uma cópia para inserir instruções sem modificar
            # o frame original que será processado.
            exibicao = frame.copy()

            cv2.putText(
                exibicao,
                "S: capturar | Q ou Esc: cancelar",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Captura para deteccao de contornos", exibicao)

            # Aguarda uma tecla durante 1 milissegundo.
            tecla = cv2.waitKey(1) & 0xFF

            # S salva uma cópia do frame atual.
            if tecla in (ord("s"), ord("S")):
                imagem_capturada = frame.copy()
                print("Imagem capturada.")
                break

            # Q ou Esc encerra sem capturar.
            if tecla in (ord("q"), ord("Q"), 27):
                print("Captura cancelada.")
                break

    finally:
        # Libera a câmera e fecha a janela, mesmo em caso de erro.
        cap.release()
        cv2.destroyAllWindows()

    return imagem_capturada


def cor_por_area(area: float):
    """Retorna uma cor BGR conforme a faixa de área do contorno."""

    if area < 1000:
        return (0, 255, 0)       # Verde: contorno pequeno.

    if area < 5000:
        return (0, 255, 255)     # Amarelo: contorno médio.

    return (0, 0, 255)           # Vermelho: contorno grande.


def preparar_mascara(mascara: np.ndarray) -> np.ndarray:
    """Remove pequenos ruídos e fecha falhas da máscara binária."""

    # Elemento estruturante utilizado nas operações morfológicas.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    # Abertura:
    # erosão seguida de dilatação.
    # Remove pequenos pontos brancos isolados.
    limpa = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Fechamento:
    # dilatação seguida de erosão.
    # Fecha pequenas falhas e ajuda a formar regiões preenchidas.
    limpa = cv2.morphologyEx(
        limpa,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return limpa


def avaliar_mascara(mascara: np.ndarray) -> tuple[int, float]:
    """Avalia quantos contornos úteis existem em uma máscara.

    A avaliação serve para escolher automaticamente entre a máscara de Otsu
    normal e sua versão invertida.
    """

    contornos, _ = cv2.findContours(
        mascara.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    altura, largura = mascara.shape
    area_imagem = float(altura * largura)

    quantidade_validos = 0
    soma_areas = 0.0

    for contorno in contornos:
        area = cv2.contourArea(contorno)

        # Ignora ruídos pequenos.
        if area <= AREA_MINIMA:
            continue

        # Ignora uma região que ocupe praticamente a imagem inteira,
        # pois normalmente ela corresponde ao fundo.
        if area >= 0.95 * area_imagem:
            continue

        quantidade_validos += 1
        soma_areas += area

    return quantidade_validos, soma_areas


def escolher_melhor_mascara(
    binaria_normal: np.ndarray,
    binaria_invertida: np.ndarray
) -> tuple[np.ndarray, str]:
    """Escolhe a máscara que contém mais regiões úteis."""

    normal_limpa = preparar_mascara(binaria_normal)
    invertida_limpa = preparar_mascara(binaria_invertida)

    avaliacao_normal = avaliar_mascara(normal_limpa)
    avaliacao_invertida = avaliar_mascara(invertida_limpa)

    print()
    print("Avaliação automática das máscaras:")
    print(
        "Otsu normal   -> "
        f"{avaliacao_normal[0]} contornos válidos, "
        f"soma das áreas = {avaliacao_normal[1]:.2f} px²"
    )
    print(
        "Otsu invertido -> "
        f"{avaliacao_invertida[0]} contornos válidos, "
        f"soma das áreas = {avaliacao_invertida[1]:.2f} px²"
    )

    # Prioriza a máscara com maior quantidade de contornos úteis.
    # Em caso de empate, utiliza a que possui maior soma de áreas.
    if avaliacao_invertida > avaliacao_normal:
        return invertida_limpa, "Otsu invertido"

    return normal_limpa, "Otsu normal"


def redimensionar_para_exibicao(
    imagem: np.ndarray,
    largura_maxima: int = 500
) -> np.ndarray:
    """Redimensiona uma imagem apenas para facilitar sua exibição."""

    altura, largura = imagem.shape[:2]

    if largura <= largura_maxima:
        return imagem

    escala = largura_maxima / largura

    return cv2.resize(
        imagem,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_AREA
    )


def main() -> None:
    """Captura a imagem, encontra contornos e classifica-os por área."""

    # Captura uma imagem diretamente da câmera.
    original = capturar_imagem_camera(indice_camera=0)

    # Encerra caso o usuário tenha cancelado.
    if original is None:
        return

    # Converte a imagem BGR para escala de cinza.
    cinza = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )

    # Reduz ruídos de alta frequência antes da limiarização.
    suave = cv2.GaussianBlur(
        cinza,
        (5, 5),
        0
    )

    # O método de Otsu calcula automaticamente o valor de limiar.
    # THRESH_BINARY cria objetos claros sobre fundo escuro.
    limiar_otsu, binaria_normal = cv2.threshold(
        suave,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # A versão invertida troca branco por preto e preto por branco.
    binaria_invertida = cv2.bitwise_not(binaria_normal)

    print(f"Valor de limiar calculado por Otsu: {limiar_otsu:.2f}")

    # Escolhe automaticamente a polaridade mais adequada.
    mascara, nome_mascara = escolher_melhor_mascara(
        binaria_normal,
        binaria_invertida
    )

    print(f"Máscara selecionada: {nome_mascara}")

    # Canny é calculado para mostrar as bordas encontradas.
    # Entretanto, findContours será aplicado à máscara preenchida,
    # pois ela produz áreas fechadas mais adequadas a contourArea().
    bordas = cv2.Canny(
        mascara,
        50,
        150
    )

    # Localiza os contornos externos na máscara binária preenchida.
    contornos, _ = cv2.findContours(
        mascara.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    resultado = original.copy()
    areas_validas = []

    altura_imagem, largura_imagem = original.shape[:2]
    area_imagem = float(altura_imagem * largura_imagem)

    for contorno in contornos:
        # Calcula a área interna do contorno.
        area = cv2.contourArea(contorno)

        # Elimina ruídos abaixo do limite exigido.
        if area <= AREA_MINIMA:
            continue

        # Evita classificar o fundo inteiro como objeto.
        if area >= 0.95 * area_imagem:
            continue

        areas_validas.append(area)

        # Escolhe a cor conforme a faixa de área.
        cor = cor_por_area(area)

        # Desenha o contorno encontrado.
        cv2.drawContours(
            resultado,
            [contorno],
            -1,
            cor,
            3
        )

        # Calcula o retângulo delimitador do objeto.
        x, y, largura, altura = cv2.boundingRect(contorno)

        # Desenha o bounding box.
        cv2.rectangle(
            resultado,
            (x, y),
            (x + largura, y + altura),
            cor,
            2
        )

        # Escreve a área próxima ao objeto.
        cv2.putText(
            resultado,
            f"{area:.0f} px2",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cor,
            2,
            cv2.LINE_AA
        )

    # Calcula a maior área entre os contornos aprovados.
    maior_area = max(areas_validas, default=0.0)

    print()
    print("Resultado da detecção:")
    print(f"Total de contornos encontrados: {len(contornos)}")
    print(
        f"Contornos com área superior a {AREA_MINIMA:.0f} px²: "
        f"{len(areas_validas)}"
    )
    print(f"Área do maior contorno válido: {maior_area:.2f} px²")

    if not areas_validas:
        print()
        print("Nenhum contorno passou pelo filtro de área.")
        print("Tente estas ações:")
        print("- use objetos com maior contraste em relação ao fundo;")
        print("- aproxime o objeto da câmera;")
        print("- utilize um fundo liso;")
        print("- reduza temporariamente AREA_MINIMA para 50;")
        print("- evite sombras fortes e reflexos.")

    # Converte as imagens de um canal para BGR antes de concatenar.
    mascara_bgr = cv2.cvtColor(
        mascara,
        cv2.COLOR_GRAY2BGR
    )

    bordas_bgr = cv2.cvtColor(
        bordas,
        cv2.COLOR_GRAY2BGR
    )

    # Redimensiona apenas para caber melhor na tela.
    original_exibicao = redimensionar_para_exibicao(original)
    mascara_exibicao = redimensionar_para_exibicao(mascara_bgr)
    bordas_exibicao = redimensionar_para_exibicao(bordas_bgr)
    resultado_exibicao = redimensionar_para_exibicao(resultado)

    # Garante dimensões iguais antes de concatenar.
    altura_painel, largura_painel = original_exibicao.shape[:2]

    mascara_exibicao = cv2.resize(
        mascara_exibicao,
        (largura_painel, altura_painel)
    )

    bordas_exibicao = cv2.resize(
        bordas_exibicao,
        (largura_painel, altura_painel)
    )

    resultado_exibicao = cv2.resize(
        resultado_exibicao,
        (largura_painel, altura_painel)
    )

    # Monta dois painéis horizontais.
    linha_superior = cv2.hconcat(
        [original_exibicao, mascara_exibicao]
    )

    linha_inferior = cv2.hconcat(
        [bordas_exibicao, resultado_exibicao]
    )

    # Junta as duas linhas verticalmente.
    painel = cv2.vconcat(
        [linha_superior, linha_inferior]
    )

    # Adiciona títulos às quatro imagens.
    cv2.putText(
        painel,
        "Original",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        painel,
        "Mascara binaria",
        (largura_painel + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        painel,
        "Bordas Canny",
        (10, altura_painel + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        painel,
        "Contornos por area",
        (largura_painel + 10, altura_painel + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    print()
    print("Pressione Q ou Esc para fechar o painel.")

    cv2.imshow(
        "Exemplo 18 corrigido - contornos por area",
        painel
    )

    while True:
        tecla = cv2.waitKey(30) & 0xFF

        if tecla in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
