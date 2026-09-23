"""
COMENTÁRIO GERAL

Este exemplo faz parte de uma sequência didática sobre OpenCV DNN.
A ideia é construir o pipeline aos poucos: imagem -> blob -> modelo -> forward -> top-3 -> comparação com Keras -> pipeline integrado.

A parte principal do código já funciona. Ao final, o bloco "DESAFIO DO ALUNO" indica o que deve ser modificado, calculado ou completado para transformar a demonstração em atividade prática.
"""

import csv, cv2
from dnn_utils import ROOT, DATA_CLASS, list_classification_images, predict_opencv, topk, load_labels

csv_path = ROOT / 'data' / 'labels_top1.csv'
if not csv_path.exists():
    print('Arquivo labels_top1.csv não encontrado. Execute 03 ou crie manualmente.')
    raise SystemExit

labels = load_labels()
net_labels = []
corretos = 0
total = 0

with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        expected = row['label_esperado'].strip().lower()
        if not expected:
            continue
        img_path = DATA_CLASS / row['arquivo']
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        prob = predict_opencv(img)
        pred_idx = topk(prob, 1)[0][0]
        pred_label = labels[pred_idx].lower().replace(' ', '_')
        ok = expected in pred_label or pred_label in expected
        corretos += int(ok)
        total += 1
        print(row['arquivo'], 'esperado=', expected, 'predito=', pred_label, 'ok=', ok)

acc = corretos / total if total else 0.0
print(f'Acurácia top-1: {acc*100:.1f}% ({corretos}/{total})')

# DESAFIO DO ALUNO:
# Preencha labels_top1.csv com as 10 imagens reais e rode novamente.
