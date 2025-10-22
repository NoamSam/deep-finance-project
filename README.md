# Deep Finance Project — CAC 40 & Portfolio Prediction

## 🎯 Objectif
Développer un outil Python basé sur le Deep Learning (LSTM, CNN, GRU) pour :
- Prédire les rendements d'actions du CAC 40
- Compléter les données manquantes (time-series completion)
- Optimiser un portefeuille basé sur les prévisions




# Workflow pour un nouveau membre du projet

## 1. Clonage du dépôt

Le nouveau membre récupère le projet depuis GitHub :

```bash
git clone https://github.com/NoamSam/deep-finance-project.git
cd deep-finance-project
```
Cela télécharge le dépôt localement et configure le lien vers GitHub (origin).

⸻

2. Installation de l’environnement Python

Depuis le dossier du projet (vous pouvez copier coller le chemin du fichier "deep-finance-project" ou simplement drag and drop le fichier dans votre terminal: 
```
bash setup.sh
source venv/bin/activate
```
Cette étape crée un environnement virtuel Python et installe automatiquement toutes les dépendances nécessaires :
TensorFlow, PyTorch, Pandas, Scikit-learn, YFinance, etc.

⸻

3. Création de la branche personnelle

Chaque membre dispose d’une branche de travail permanente.

Exemple : le nouveau membre s’appelle Lucas
```
git checkout develop
git pull origin develop
git checkout -b Lucas
git push -u origin Lucas
```
Une branche locale et distante nommée Lucas est ainsi créée et suivie automatiquement.

⸻

4. Configuration Git recommandée (optionnelle mais conseillée)

Pour éviter le message “no upstream branch” :
```
git config --global push.autoSetupRemote true
```
Cette configuration permet à Git de relier automatiquement une nouvelle branche à son équivalent distant lors du premier push.

⸻

5. Organisation du travail

Lucas peut travailler librement sur sa branche :
```
git add .
git commit -m "Ajout du module de preprocessing CAC40"
git push
```
Les modifications restent isolées dans la branche Lucas sans impacter develop ni main.

⸻

6. Synchronisation avec la branche commune develop

Avant de commencer une nouvelle session de travail :
```
git pull origin develop
```
Cette commande permet de récupérer les dernières mises à jour issues des autres branches intégrées dans develop.

⸻

7. Création d’une Pull Request

Lorsque la fonctionnalité développée est stable :
	1.	Aller sur GitHub Desktop 
	2.	Cliquer sur Compare & pull request
	3.	Vérifier la base et la comparaison :
	•	Base : develop
	•	Compare : Lucas
	4.	Ajouter un titre et une description, par exemple :
[Preprocessing] Ajout du nettoyage des données CAC40
	5.	Créer la Pull Request (PR)

La PR sera relue, commentée et validée avant fusion dans develop.

⸻

8. Fusion dans develop

Une fois la PR validée, elle est fusionnée dans la branche develop par le responsable du projet (Noam ou un reviewer désigné).
La branche personnelle reste inchangée pour les futures évolutions.

⸻

9. Mise à jour locale après fusion

Pour récupérer la dernière version du projet :
```
git checkout develop
git pull origin develop
```
Puis mettre à jour sa propre branche :
```
git checkout Lucas
git pull origin develop
```
La branche Lucas est maintenant synchronisée avec develop.

⸻

10. Gestion des conflits

En cas de conflit entre branches :

Git indique les fichiers concernés.
Le membre peut les corriger manuellement, puis exécuter :
```
git add .
git commit -m "Résolution de conflit avec develop"
git push
```

⸻

Résumé des branches du projet

Branche	Utilisation	Accès

main	Version stable finale	Restreinte à Noam

develop	Branche d’intégration commune	Libre

Noam	Branche personnelle	Noam

Jade	Branche personnelle	Jade

Matteo	Branche personnelle	Matteo

Lucas	Branche personnelle (nouveau membre)	Lucas

⸻

Checklist pour un nouveau membre

Étape	Commande	Objectif
1.	git clone ...	Télécharger le projet
2.	bash setup.sh && source venv/bin/activate	Créer l’environnement
3.	git checkout -b Lucas && git push -u origin Lucas	Créer la branche personnelle
4.	git pull origin develop	Récupérer les nouveautés
5.	git add . && git commit && git push	Sauvegarder le code
6.	PR Lucas → develop	Soumettre le travail
7.	Merge et synchronisation	Maintenir le code à jour


⸻

Organisation Git globale
```
main        ← stable (Noam uniquement)
 └── develop ← branche commune d’intégration
      ├── Noam
      ├── Jade
      ├── Matteo
      └── Lucas
```
Ce modèle permet à chaque membre de travailler de manière autonome tout en assurant la stabilité du projet global.

---


## Exemple concret: 

🔹 Contexte
	•	Noam a poussé sur develop un fichier : fichier1.py
	•	Matteo travaille localement sur fichier2.py, dans sa branche Matteo
	•	Matteo veut mettre à jour sa branche pour avoir les nouveautés de Noam
sans perdre ou écraser son propre code.

⸻

🧭 Étapes à suivre (solution propre)

1️⃣ Matteo s’assure d’avoir tout sauvegardé et commité

Avant toute manipulation, il doit enregistrer son travail local :
```
git add .
git commit -m "Travail en cours sur fichier2"
```
Cela garde ses modifications en sécurité dans l’historique Git.

⸻

2️⃣ Récupérer la dernière version de develop depuis GitHub
```
git pull origin develop --rebase
```



