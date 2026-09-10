"""
Objetivo didatico:
- Ler com cv2.imread uma imagem salva anteriormente.
- Converter BGR para escala de cinza com cv2.cvtColor.
- Imprimir shape e dtype das duas representacoes.
- Exibir original e cinza lado a lado com cv2.hconcat.

Execucao:
    python 05_recarregar_cinza_hconcat.py
    python 05_recarregar_cinza_hconcat.py --imagem frame_capturado.png
"""

import argparse
from pathlib import Path

import cv2

def main() -> None:
    """Executa a segunda parte do Item B."""

    caminho_imagem = Path("frame_capturado.png")
    if not caminho_imagem.is_absolute():
        caminho_imagem = Path(__file__).resolve().parent / caminho_imagem

    # Recarrega a imagem colorida do disco.
    colorida = cv2.imread(str(caminho_imagem), cv2.IMREAD_COLOR)
    if colorida is None:
        raise FileNotFoundError(
            f"Nao foi possivel ler {caminho_imagem}. "
            "Execute primeiro 04_capturar_e_salvar_frame.py."
        )

    # Converte a matriz BGR de tres canais para uma matriz 2D de intensidades.
    cinza = cv2.cvtColor(colorida, cv2.COLOR_BGR2GRAY)

    # Saidas esperadas pelo enunciado.
    print("--- VALIDACAO DOS ARRAYS NUMPY ---")
    print(f"Colorida -> shape: {colorida.shape}, dtype: {colorida.dtype}")
    print(f"Cinza    -> shape: {cinza.shape}, dtype: {cinza.dtype}")

    # hconcat nao concatena diretamente uma imagem 3D BGR com uma imagem 2D.
    # A conversao abaixo serve apenas para a exibicao. A matriz 'cinza' original
    # continua com shape (H, W), conforme a validacao solicitada.
    cinza_para_exibir = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)

    # Cria copias para que os rotulos nao alterem os arrays usados na validacao.
    colorida_rotulada = colorida.copy()
    cinza_rotulada = cinza_para_exibir.copy()

    cv2.putText(
        colorida_rotulada,
        "Original colorida (BGR)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        cinza_rotulada,
        "Escala de cinza",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Monta um unico painel horizontal.
    painel = cv2.hconcat([colorida_rotulada, cinza_rotulada])

    cv2.imshow("Item B - original e cinza - pressione Q", painel)

    while True:
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
