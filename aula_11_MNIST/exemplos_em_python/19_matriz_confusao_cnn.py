"""
Exemplo 19 - Matriz de confusão da CNN.

Objetivo:
- Diagnosticar erros por classe.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from mnist_utils import build_cnn, load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
model = build_cnn()
model.fit(x_train, y_train, validation_split=0.1, epochs=3, batch_size=128)

pred = model.predict(x_test, verbose=0).argmax(axis=1)
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm).plot(values_format="d")
plt.title("Matriz de confusão - CNN")
plt.show()
