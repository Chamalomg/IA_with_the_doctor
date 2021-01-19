import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

from utils.preprocessing import tokenization

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

#tokenization 
X = tokenization(X)


X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2)

X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train,y_train, test_size=0.2)











