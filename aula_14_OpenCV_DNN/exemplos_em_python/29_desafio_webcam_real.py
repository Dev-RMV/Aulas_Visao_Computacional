"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

PSEUDOCODIGO = r"""
carregar K e dist da calibração real
carregar modelo DNN
abrir cv2.VideoCapture(0)

para cada frame:
    corrigir distorção com cv2.undistort
    segmentar ROI por HSV
    extrair ORB na ROI
    aplicar HOG ou Haar
    classificar ROI com DNN
    desenhar top-3, caixas, keypoints e tempos
    mostrar frame

fechar câmera
"""

print(PSEUDOCODIGO)

# DESAFIO DO ALUNO:
# Transforme este pseudocódigo em código funcional.
# Essa é a parte que deve ficar para o aluno resolver em laboratório.
