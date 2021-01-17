import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import OneHotEncoder

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Head : idx,text,type,details,episodeid,doctorid
df = pd.read_csv("dataset/all-scripts.csv")

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

print(X[:10])
print(y[:10])






