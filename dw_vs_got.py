import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing utiler ça
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

# votre matrice de confusion metrics

from sklearn import model_selection

import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Const
vocab_size = 10000
max_length = 100

# Extraction GOT
# Head : idx,text,type,details,episodeid,doctorid
df_got = pd.read_csv("dataset/Game_of_Thrones_Script.csv")
df_got = df_got.dropna()

X_got = df_got['Sentence'].tolist()
y_got = [1] * len(X_got)

# Extraction dw
df_dw = pd.read_csv("dataset/all-scripts.csv")
df_dw = df_dw.dropna()
df_dw = df_dw[df_dw['type'] == 'talk']

X_dw = df_dw['text'].tolist()
y_dw = [0] * len(X_dw)

X = X_got + X_dw
y = y_got + y_dw

# split train test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)

# preprossessing tokenization + padding(to fix )
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_length, padding="post")

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length, padding="post")

import numpy as np

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Model
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
    layers.GlobalAveragePooling1D(),

    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


def confusion(conf):
    sns.heatmap(conf, square=True, annot=True, cbar=False
                , xticklabels=list(['Tyrion', 'Other'])
                , yticklabels=list(['Tyrion', 'Other']))
    plt.show()


Y_pred = model.predict_generator(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['DW', 'Got']
print(classification_report(y_test, y_pred, target_names=target_names))

conf = confusion_matrix(y_test, y_pred)
