import pandas as pd

df = pd.read_csv("dataset/all-scripts.csv")
text = df[len(df['text']) > 10 & df['type'] == 'talk' & df['details'] == 'ROSE')
a = dfObj[(dfObj['Sale'] > 30) & (dfObj['Sale'] < 33) ]
print(text[0:10])