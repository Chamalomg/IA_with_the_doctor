import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import model_selection
import os
import numpy as np
from utils.glove import createPretrainedEmbeddingLayer, readGloveFile
from utils.plot import plot_confusion
from utils.utils import define_model

# Root file
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Const
vocab_size = 20000
max_length = 200
main_character = 'DOCTOR'

# Head : idx,text,type,details,episodeid,doctorid
df = pd.read_csv("dataset/all-scripts.csv")
df = df.dropna()
data = df[df['type'] == 'talk']
data = shuffle(data)

X = data['text'].tolist()
y = data['details'].tolist()

# Extract labels
number_doctor = 0
number_other = 0
for i in range(len(y)):
    if y[i] == main_character:
        y[i] = 1
        number_doctor += 1
    else:
        y[i] = 0
        number_other += 1

print("Database of " + str(len(X)) + " subtitles, with : " + str(number_doctor) + " doctor's sentences and " + str(
    number_other) + " others's sentences : {} %".format(100 * number_doctor / (number_doctor + number_other)))

# split train test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# preprossessing tokenization + padding(to fix )
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=max_length, padding="post", truncating="post")

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=max_length, padding="post", truncating="post")

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# usage
wordToIndex, indexToWord, wordToGlove = readGloveFile('glove.6B.50d/glove.6B.50d.txt')
pretrainedEmbeddingLayer = createPretrainedEmbeddingLayer(wordToGlove, wordToIndex, False)

# Model
model = define_model(pretrainedEmbeddingLayer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

Y_pred = model.predict_generator(X_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['Other', 'Doctor']
print(classification_report(y_test, y_pred, target_names=target_names))

conf = confusion_matrix(y_test, y_pred)
plot_confusion(conf)
