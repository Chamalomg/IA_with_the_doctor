import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# preprocessing utiler ça
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

# votre matrice de confusion metrics

from sklearn import model_selection
import os
import matplotlib.pylab as plt
from matplotlib.pyplot import figure
from tensorflow.python.keras.initializers.initializers_v2 import Constant
from tensorflow.python.keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix

from utils.glove import readGloveFile, createPretrainedEmbeddingLayer
from utils.plot import plot_loss, confusion

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Const
vocab_size = 10000
max_length = 100
main_character = 'tyrion lannister'

# Head : idx,text,type,details,episodeid,doctorid
df = pd.read_csv("dataset/Game_of_Thrones_Script.csv")
df = df.dropna()
print(df.head)
data = df
data = shuffle(data)

X = data['Sentence'].tolist()
y = data['Name'].tolist()

# Extract labels
number_main_character = 0
number_other = 0
for i in range(len(y)):
    if y[i] == main_character:
        y[i] = 1
        number_main_character += 1
    else:
        y[i] = 0
        number_other += 1

print("Database of " + str(len(X)) + " subtitles, with : " + str(
    number_main_character) + " main character's sentences and " + str(number_other) +
      " others's sentences : {} %".format(100 * number_main_character / (number_main_character + number_other)))

# split train test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# preprossessing tokenization + padding(to fix )
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
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


def define_model(layer):
    """Create model : embedding and MLP"""
    model = keras.Sequential([
        layer,
        # layers.Conv1D(32, 7, padding="valid", activation="relu", strides=3),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dropout(.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def define_model_conv(layer):
    """Create model : embedding and MLP"""
    model = keras.Sequential([
        layer,
        # layers.Conv1D(32, 7, padding="valid", activation="relu", strides=3),
        layers.Conv1D(128, 5, activation="relu"),
        layers.MaxPooling1D(5),
        layers.Conv1D(128, 5, activation="relu"),
        layers.MaxPooling1D(5),
        layers.Conv1D(128, 5, activation="relu"),
        layers.MaxPooling1D(5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# usage
wordToIndex, indexToWord, wordToGlove = readGloveFile('glove.6B.50d/glove.6B.50d.txt')
pretrainedEmbeddingLayer = createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, False)

# Model
model = define_model(pretrainedEmbeddingLayer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)
model.summary()

history = model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test))
print(history.history)
# model.save('./saved_model/lastest.keras')

results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)
predictions = model.predict(X_test[:5])
print("predictions shape:", predictions.shape)

plot_loss(history)
Y_pred = model.predict_generator(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
conf = confusion_matrix(y_test, y_pred)

print('Classification Report')
target_names = ['Other', 'Tyrion']
print(classification_report(y_test, y_pred, target_names=target_names))
print(y_test, Y_pred)
confusion(conf)

