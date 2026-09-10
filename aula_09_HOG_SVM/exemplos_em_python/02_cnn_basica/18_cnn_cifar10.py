import tensorflow as tf
from tensorflow.keras import layers,models
(xtr,ytr),(xte,yte)=tf.keras.datasets.cifar10.load_data(); xtr=xtr/255.0; xte=xte/255.0
m=models.Sequential([layers.Input((32,32,3)),layers.Conv2D(32,3,padding='same',activation='relu'),layers.MaxPooling2D(2),layers.Conv2D(64,3,padding='same',activation='relu'),layers.MaxPooling2D(2),layers.Conv2D(128,3,padding='same',activation='relu'),layers.GlobalAveragePooling2D(),layers.Dense(10,activation='softmax')])
m.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy']); m.summary(); m.fit(xtr,ytr,epochs=3,batch_size=128,validation_split=.1); print(m.evaluate(xte,yte,verbose=0))
