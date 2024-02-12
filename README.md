# TABR: TABULAR DEEP LEARNING MEETSNEAREST NEIGHBORS IN 2023
Réimplémentation du modèle TabR

Dans le travail des auteurs, ils présentent "TabR", un modèle de Deep Learning (DL) à base récupération dédié aux données tabulaires, se démarquant des modèles basés sur les arbres de décision boostés par gradient (GBDT) prédominants dans les tâches de classification et de régression.

TabR intègre une composante inspirée des k-Plus Proches Voisins (k-NN). Cette caractéristique permet à TabR non seulement de surpasser les modèles DL tabulaires existants, mais aussi de devancer les GBDT dans des benchmarks spécifiques, notamment ceux favorables aux GBDT.

## Architecture du code
`data.py `: contient nos méthodes pour traitement des données tabulaire
`deep.py` : contient les traitements nécessaire utiles pour l'apprentissage et évaluation
`exp1.py` : Fait l'entrainement de l'exp1 et sauvegarde les sauvegardes les résultats dans dossier log, les résultats sont  utilisés par `result.py`
`exp2.py` : Fait l'entrainement de l'exp2 et sauvegarde les sauvegardes les résultats dans dossier log, les résultats sont  utilisés par `result.py`
`model.py `: contient la ré-implémentation du modèle TabR-S
`result.py `: affiche les résultats pour les expériences 1 et expériences 2 
`log/` : Contient les résultats des expériences 1 et 2 