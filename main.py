import pandas as pd

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


for pers in OUR_PERS:
    print('Text from {} :'.format(pers))
    print(pers_text(df, pers))

    
