import numpy as np
import tensorflow as tf

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
from keras.callbacks import EarlyStopping

max_fetrs = 50000
(X_train,y_train),(X_test, y_test) = imdb.load_data(num_words = max_fetrs)

max_len =500
X_train = sequence.pad_sequences(X_train,maxlen=max_len)
X_test = sequence.pad_sequences(X_test,maxlen=max_len)

model = Sequential()
model.add(Embedding(max_fetrs, 128, input_length=500))
model.add(SimpleRNN(128,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile('adam','binary_crossentropy',metrics=['accuracy'])
model.summary()


earlystopper = EarlyStopping(monitor='val_loss', patience=8,restore_best_weights=True)
history = model.fit(X_train,y_train,epochs=10,batch_size=10,validation_split=0.2,callbacks=[earlystopper] )

model.save('rnn_model.h5')


