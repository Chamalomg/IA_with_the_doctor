import pandas as pd
import re


def process(df) -> pd.DataFrame:
    pass


def clean_txt(str) -> str:
    pass


def get_words(str) -> list:
    return re.split(r'\W+', str)


def clean_text(str) -> str:
    """
    :param str: Str à clean
    :return: str sans espaces et ponctuation
    """
    return re.sub(r'[^\w\s^]', '', str).lower().replace(" ", "")