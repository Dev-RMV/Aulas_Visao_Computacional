# Aulas de Visão Computacional

Repositório de estudo da disciplina de Visão Computacional. Reúne, organizados por aula, os códigos apresentados em sala pelo **Professor Vitor Amadeu (PVA)**, prontos para rodar em Python. Para cada aula há também um tutorial de estudo, escrito por mim a partir da leitura desses códigos.

## Créditos

**Do Professor Vitor Amadeu (PVA):** os scripts de exemplo de todas as aulas, os zips originais em `referencia_de_aula/`, os enunciados dos exercícios e os recursos de apoio. Ele disponibiliza esse material aos alunos pelo Google Drive da disciplina. Aqui os scripts estão exatamente como ele os escreveu, apenas organizados em pastas e, quando vieram em documentos Word, transcritos para `.py`.

**Deste repositório (material de estudo, não do professor):** os tutoriais `tutorial_da_aula.md` de cada aula, o índice `tutoriais.md`, o relatório `achados_dos_zips.md`, o `requirements.txt` e o script `preparar_ambiente.py`. Os tutoriais explicam os conceitos com base no que os scripts fazem e citam trechos deles, mas foram escritos por mim com auxílio de IA (Claude), e não pelo professor. Qualquer erro de interpretação nos tutoriais é meu, não dele.

## As 12 aulas

| Aula | Tema | Tutorial |
|---|---|---|
| 01 | OpenCV: imagem como array, câmera, vídeo | [tutorial](aula_01_OpenCV/tutorial_da_aula.md) |
| 02 | Câmera: espaços de cor, filtros, limiarização, contornos | [tutorial](aula_02_Camera/tutorial_da_aula.md) |
| 03 | ROI, segmentação, estéreo, faces, descritores, CNN | [tutorial](aula_03_ROI_SURF_etc/tutorial_da_aula.md) |
| 04 | Tensores com TensorFlow | [tutorial](aula_04_Tensor/tutorial_da_aula.md) |
| 05 | HSV, segmentação por cor e visão estéreo | [tutorial](aula_05_HSV/tutorial_da_aula.md) |
| 06 | Reconhecimento facial: Haar e embeddings | [tutorial](aula_06_Reconhecimento_Facial/tutorial_da_aula.md) |
| 07 | Homografia e pontos-chave: SIFT, ORB, AKAZE, RANSAC | [tutorial](aula_07_Homografia/tutorial_da_aula.md) |
| 08 | TensorFlow e Keras: CNN em dataset sintético | [tutorial](aula_08_TensorFlow_Keras/tutorial_da_aula.md) |
| 09 | Convolução, CNN básica e HOG + SVM | [tutorial](aula_09_HOG_SVM/tutorial_da_aula.md) |
| 10 | Rastreamento: MOG2, KNN, CamShift, Kalman | [tutorial](aula_10_Rastreamento/tutorial_da_aula.md) |
| 11 | MNIST: MLP, CNN, regularização, dígitos reais | [tutorial](aula_11_MNIST/tutorial_da_aula.md) |
| 12 | DNN e Caffe: idade e gênero, UTKFace, MobileNetV2, TFLite | [tutorial](aula_12_DNN_Caffe/tutorial_da_aula.md) |

O índice completo, com os conceitos de cada aula e os detalhes encontrados na leitura dos scripts, está em [tutoriais.md](tutoriais.md).

## Estrutura

```
aula_NN_Tema/
├── referencia_de_aula/      zip original baixado do Drive do professor, intocado
├── exemplos_em_python/      scripts .py prontos para uso
└── tutorial_da_aula.md      tutorial da aula em 8 seções
achados_dos_zips.md          o que veio em cada zip, o que falta e o ambiente
tutoriais.md                 índice dos tutoriais
requirements.txt             dependências unificadas (Python 3.13)
preparar_ambiente.py         prepara o ambiente no Windows ou no Linux
```

## Como preparar o ambiente

Um único ambiente roda todas as aulas. O script detecta o sistema, obtém um Python 3.13 se necessário, cria o venv, instala tudo e verifica.

```powershell
py preparar_ambiente.py            # Windows
```

```bash
python3 preparar_ambiente.py       # Linux ou WSL
```

As versões escolhidas e o porquê de cada uma estão na seção 6 de [achados_dos_zips.md](achados_dos_zips.md). Em resumo: OpenCV Contrib 4.14, TensorFlow 2.21 com Keras 3.15, scikit-learn 1.9, dlib e face_recognition.

## Material que não veio nos zips

Alguns recursos de apoio ficam fora dos zips por tamanho ou licença e precisam ser obtidos à parte. A lista está na seção 4 de [achados_dos_zips.md](achados_dos_zips.md).
