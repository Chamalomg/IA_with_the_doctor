import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#preprocessing utiler ça 
#https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

#votre matrice de confusion metrics


from sklearn import model_selection

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#Const
vocab_size = 10000
max_length = 100

# Head : idx,text,type,details,episodeid,doctorid
df = pd.read_csv("dataset/all-scripts.csv")
df = df.dropna()
data = df[df['type'] == 'talk']
data = shuffle(data)

X = data['text'].tolist()
y = data['details'].tolist()

#Extract labels
number_doctor = 0
number_other = 0
for i in range(len(y)):
    if(y[i]=='DOCTOR'):
        y[i]=1
        number_doctor += 1
    else:
        y[i]=0
        number_other += 1

print("Database of "+str(len(X))+ " subtitles, with : "+str(number_doctor)+" doctor's sentences and "+str(number_other)+ " others's sentences")

#split train test 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

#preprossessing tokenization + padding(to fix )
tokenizer = Tokenizer(num_words=vocab_size ,oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_length, padding="post", truncating="post")

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")

import numpy as np
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

#Model
model = keras.Sequential([
    keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))














