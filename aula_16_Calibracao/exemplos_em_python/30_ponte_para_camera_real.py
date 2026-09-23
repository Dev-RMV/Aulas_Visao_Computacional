"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre calibração de câmera
e realidade aumentada com OpenCV.

A proposta é usar primeiro uma câmera virtual e imagens sintéticas. Assim,
é possível aprender cada etapa sem depender de webcam, iluminação, foco ou
qualidade de impressão do tabuleiro.

Ao final existe um "DESAFIO DO ALUNO". A parte principal já funciona, mas
o aluno deve modificar, medir, comparar ou completar uma extensão.
"""

"""
Neste último exemplo NÃO abrimos a webcam.

A ideia é deixar clara a ponte entre o laboratório sintético e a aplicação real.

No ambiente sintético:
    frame = generate_view(...)

Na aplicação real:
    ret, frame = cap.read()

Todo o restante do pipeline permanece praticamente igual:
1. converter/detectar cantos;
2. refinar;
3. usar K e dist já calibrados;
4. solvePnP;
5. projectPoints;
6. desenhar o cubo;
7. exibir rvec e tvec.

Assim, a câmera real pode ser deixada como exercício final do aluno.
"""

PSEUDOCODIGO = r"""
carregar K e dist

abrir câmera ou vídeo

enquanto houver frame:
    detectar tabuleiro
    se encontrado:
        refinar cantos
        estimar pose com solvePnP
        projetar cubo com projectPoints
        desenhar cubo
        imprimir rvec e tvec

    mostrar frame

fechar câmera
"""

print(PSEUDOCODIGO)

# DESAFIO DO ALUNO:
# Transforme este pseudocódigo em um programa funcional com cv2.VideoCapture.
#
# Sugestões:
# - comece com um vídeo gravado;
# - depois substitua pelo índice 0 da webcam;
# - use a calibração salva em camera_sintetica.npz SOMENTE para testes sintéticos;
# - para a webcam real, faça uma calibração real da própria câmera.
