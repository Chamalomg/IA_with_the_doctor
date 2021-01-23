import pandas as pd
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

# Head : idx,text,type,details,episodeid,doctorid
df = pd.read_csv("dataset/all-scripts.csv")
df = df.dropna()
text = df[df['type'] == 'talk']

# Personnages étudiés
OUR_PERS = []
with open('dataset/personnages.txt', 'r') as f:
    OUR_PERS = f.readlines()
OUR_PERS = [x.strip('\n') for x in OUR_PERS]

# script d'un personnage en particulier
def pers_text(df, pers) -> pd.DataFrame:
    return text[text['details'] == pers]

data = pers_text(df,'')
for pers in OUR_PERS:
    #print('Text from {} :'.format(pers))
    #print(pers_text(df, pers).head())
    data = data.append(pers_text(df, pers))

data = shuffle(data)
#extract data to list --> X : subtitle / y : Speaker
X = data['text'].tolist()
y = data['details'].tolist()
#transforme string labels into binary vector
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

#extract labels 
labels = data.details.unique()
print("Database of "+str(len(X))+ " subtitles, with "+str(len(labels))+" speakers")

#preprossecing
tokenizer = keras.preprocessing.text.Tokenizer(
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
)
#Updates internal vocabulary based on a list of texts
tokenizer.fit_on_texts(X)
print("number of words : " + str(len(tokenizer.word_counts)))

X = tokenizer.texts_to_sequences(X)
#fix input length
X = pad_sequences(X, padding='post')


#add Normalisation !!!
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train,y_train, test_size=0.2)


model = keras.Sequential(
    [
    #layers.Input(shape=(None,)),
    layers.Embedding(input_dim=240,output_dim=32,input_length=X.shape[1]),
    layers.Dense(32),
    #layers.LSTM(units=32,return_sequences=True),
    #layers.LSTM(units=64),
    layers.Dense(len(OUR_PERS),activation="softmax")
    ]
)


model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()

#model.fit(X_train,y_train,batch_size=20,epochs=40,validation_data=(X_val,y_val))














