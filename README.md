# Objectif

Projet visant à pouvoir prédire l'attribution d'une réplique à un personnage.
Ce projet utilise deux datasets qui proposent l'ensemble des dialogues présents dans chaque série.
Chaque dataset à un intéret : DrWho est immense (240 000 string) avec certaines défaults et le Got est plus restreint (22 300 strings) mais d'excelente qualité.

Base de données Drwho utilisée : https://www.kaggle.com/jeanmidev/doctor-who?select=all-scripts.csv

Base de données Got utilisée https://www.kaggle.com/albenft/game-of-thrones-script-all-seasons

## Groupe de projet : 

SOLDE Fabien - Raphaël DELLA SETTA - Evrard DE PARCEVAUX - Hugo LERONDEL

EPF, 5ème année

## Usage

####Installation des dépendances

```python
pip install -r requirement.txt
```
#### Structure du fichier :

Télécharger le dossier glove.6B.zip : https://nlp.stanford.edu/projects/glove/
extraire le fichier glove.6B.50d.txt, et le placer dans ./glove.6B.50d/glove.6B.50d.txt

## License
[MIT](https://choosealicense.com/licenses/mit/)
