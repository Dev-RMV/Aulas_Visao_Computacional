import tensorflow as tf
from tensorflow.keras import layers,models
(xtr,ytr),(xte,yte)=tf.keras.datasets.mnist.load_data(); xtr=xtr[...,None]/255.0; xte=xte[...,None]/255.0
m=models.Sequential([layers.Input((28,28,1)),layers.Conv2D(32,3,activation='relu'),layers.MaxPooling2D(2),layers.Conv2D(64,3,activation='relu'),layers.MaxPooling2D(2),layers.Flatten(),layers.Dense(64,activation='relu'),layers.Dense(10,activation='softmax')])
m.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy']); m.summary(); m.fit(xtr,ytr,epochs=2,batch_size=128,validation_split=.1); print(m.evaluate(xte,yte,verbose=0)); m.save('modelo_mnist.keras')
