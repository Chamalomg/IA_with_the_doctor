import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def get_words(str) -> list:
    return re.split(r'\W+', str)


def tokenization(X) -> list:
    vectorizer = CountVectorizer()
    tokenizer = vectorizer.build_tokenizer()
    for i in range(len(X)):
        X[i] = clean_text(X[i])
        X[i] = tokenizer(X[i])
    return X


def one_hot(liste_mots) -> list:
    """
    Prend une liste de mot, renvoie une liste de vecteur [0, 0, .., 1, .., 0]
    """
    integer_encoded = LabelEncoder().fit_transform(liste_mots)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return OneHotEncoder(sparse=False).fit_transform(integer_encoded)


def clean_text(str) -> str:
    """
    :param str: Str à clean
    :return: str sans espaces et ponctuation
    """
    return re.sub(r'[^\w\s^]', '', str).lower()