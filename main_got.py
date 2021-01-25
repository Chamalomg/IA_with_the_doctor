import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# preprocessing utiler ça
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer

# votre matrice de confusion metrics

from sklearn import model_selection

import os
import matplotlib.pylab as plt
from matplotlib.pyplot import figure
from tensorflow.python.keras.initializers.initializers_v2 import Constant
from tensorflow.python.keras.utils.vis_utils import plot_model

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
      " others's sentences : {} %".format(number_main_character / (number_main_character + number_other)))

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


def define_model():
    """Create model : embedding and MLP"""
    model = keras.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=8, input_length=max_length),
        # layers.Conv1D(32, 7, padding="valid", activation="relu", strides=3),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dropout(.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


model = define_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# history = model.fit(X_train, y_train, epochs=50, batch_size=20, validation_data=(X_test, y_test))
# print(history.history)





# Prepare Glove File
# noinspection PyBroadException
def readGloveFile(gloveFile):
    with open(gloveFile, 'r', encoding="utf8") as f:
        wordToGlove = {}  # map from a token (word) to a Glove embedding vector
        wordToIndex = {}  # map from a token to an index
        indexToWord = {}  # map from an index to a token
        count_fail = 0
        for line in f:
            record = line.strip().split()
            token = record[0]  # take the token (word) from the text line
            try:
                wordToGlove[token] = np.array(record[1:],
                                              dtype=np.float32)  # associate the Glove embedding vector to a that token (word)
            except:
                print('line skipped : ', count_fail)
                count_fail += 1
                pass

        tokens = sorted(wordToGlove.keys())
        for idx, tok in enumerate(tokens):
            kerasIdx = idx + 1  # 0 is reserved for masking in Keras (see above)
            wordToIndex[tok] = kerasIdx  # associate an index to a token (word)
            indexToWord[kerasIdx] = tok  # associate a word to a token (word). Note: inverse of dictionary above

    return wordToIndex, indexToWord, wordToGlove


# Create Pretrained Keras Embedding Layer
def createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, isTrainable=False):
    vocabLen = len(wordToIndex)  # adding 1 to account for masking
    embDim = next(iter(wordToGlove.values())).shape[0]  # 300 in our case
    print('createPretrainedEmbeddingLayer \nembDim : {}, wordToIndex : {}'.format(len(wordToIndex), embDim))
    embeddingMatrix = np.zeros((vocabLen, embDim))  # initialize with zeros
    for word, index in wordToIndex.items():
        try:
            embeddingMatrix[index, :] = wordToGlove[word]  # create embedding: word index to Glove word embedding
        except IndexError:
            pass
    print('for out')

    embeddingLayer = keras.layers.Embedding(len(wordToIndex), embDim, embeddings_initializer=Constant(embeddingMatrix),
                                            trainable=isTrainable)
    return embeddingLayer


# usage
wordToIndex, indexToWord, wordToGlove = readGloveFile('glove.6B.50d/glove.6B.50d.txt')
pretrainedEmbeddingLayer = createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, False)

# Model
model = keras.Sequential([
    # keras.layers.Embedding(embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    #                        input_dim=vocab_size, output_dim=16, input_length=max_length),
    pretrainedEmbeddingLayer,
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))
print(history.history)
# model.save('./saved_model/lastest.keras')

results = model.evaluate(X_test, y_test, batch_size=128)
print("test loss, test acc:", results)
predictions = model.predict(X_test[:5])
print("predictions shape:", predictions.shape)


def plot_loss(history=history):
    figure(figsize=(8, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


plot_loss(history)