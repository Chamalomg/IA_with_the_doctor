import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import model_selection
import os

# TODO: Gérer le docteur !
from utils.preprocessing import get_words

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


def pers_text(df, pers) -> pd.DataFrame:
    """script d'un personnage en particulier"""
    return text[text['details'] == pers]


data = pers_text(df, '')
for pers in OUR_PERS:
    # print('Text from {} :'.format(pers))
    # print(pers_text(df, pers).head())
    data = data.append(pers_text(df, pers))

data = shuffle(data)
# extract data to list --> X : subtitle / y : Speaker
X = data['text'].tolist()
y = data['details'].tolist()
# transforme string labels into binary vector
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

# extract labels
labels = data.details.unique()
print("Database of " + str(len(X)) + " subtitles, with " + str(len(labels)) + " speakers")

# print('X : \n', X[0:5])
# print('y : \n', y[0:5])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=0.2)


##################################################
# TEST
##################################################
# values = np.array(get_words(X[1]))
# print(values)
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(values)
# print(integer_encoded)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#
# # binary encode
# onehot_encoder = OneHotEncoder(sparse=False)
#
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)

def dictionary(total_mot):
    return LabelEncoder().fit_transform(total_mot)


def one_hot(liste_mots) -> list:
    """
    Prend une liste de mot, renvoie une liste de vecteur [0, 0, .., 1, .., 0]
    """
    integer_encoded = LabelEncoder().fit_transform(liste_mots)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return OneHotEncoder(sparse=False).fit_transform(integer_encoded)


print(dictionary(['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']))
