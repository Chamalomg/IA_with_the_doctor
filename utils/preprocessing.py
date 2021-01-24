import pandas as pd
import re

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def process(df) -> pd.DataFrame:
    pass


def get_words(str) -> list:
    return re.split(r'\W+', str)


def clean_text(str) -> str:
    """
    :param str: Str à clean
    :return: str sans espaces et ponctuation
    """
    return re.sub(r'[^\w\s^]', '', str).lower().replace(" ", "")


def one_hot(liste_mots) -> list:
    """
    Prend une liste de mot, renvoie une liste de vecteur [0, 0, .., 1, .., 0]
    """
    integer_encoded = LabelEncoder().fit_transform(liste_mots)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return OneHotEncoder(sparse=False).fit_transform(integer_encoded)
