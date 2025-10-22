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
<<<<<<< HEAD
=======
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
1	git clone ...	Télécharger le projet
2	bash setup.sh && source venv/bin/activate	Créer l’environnement
3	git checkout -b Lucas && git push -u origin Lucas	Créer la branche personnelle
4	git pull origin develop	Récupérer les nouveautés
5	git add . && git commit && git push	Sauvegarder le code
6	PR Lucas → develop	Soumettre le travail
7	Merge et synchronisation	Maintenir le code à jour


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

Parfait — voici ton exemple concret réécrit et adapté à ton choix d’utiliser git pull origin develop --rebase, dans un style clair, cohérent avec le reste de ton README et sans emojis.

⸻


## Exemple concret : mise à jour de sa branche sans écraser son travail

### Contexte
- Noam a poussé sur `develop` un fichier : `fichier1.py`
- Matteo travaille localement sur `fichier2.py`, dans sa branche `Matteo`
- Matteo souhaite mettre à jour sa branche pour récupérer les modifications de Noam
  sans perdre ni écraser son propre code.

---

### Étapes à suivre (méthode recommandée)

#### 1. Sauvegarder le travail local

Avant toute mise à jour, Matteo doit enregistrer son travail dans un commit :

```bash
git add .
git commit -m "Travail en cours sur fichier2"
```
Cette étape garantit que ses modifications sont conservées dans l’historique Git
avant d’intégrer les nouveautés de develop.

⸻

2. Mettre à jour la branche à l’aide d’un rebase

Depuis sa branche Matteo, il exécute :
```
git pull origin develop --rebase
```
Git va :
	•	télécharger les nouveaux commits de develop (par exemple, le fichier ajouté par Noam) ;
	•	rejouer les commits de Matteo au-dessus de ces modifications récentes.

Cela évite les “merge commits” et maintient un historique linéaire et propre.

⸻

3. Résolution de conflits éventuels

Si Git détecte un conflit, il indiquera les fichiers concernés :

CONFLICT (content): Merge conflict in fichierX.py

Matteo ouvre les fichiers concernés, corrige les zones marquées par Git, puis :
```
git add fichierX.py
git rebase --continue
```
S’il souhaite abandonner l’opération :
```
git rebase --abort
```

⸻

4. Finaliser et pousser les modifications

Une fois le rebase terminé avec succès :
```
git push -f
```
Le paramètre -f (force push) est nécessaire car l’historique local de Matteo
a été réécrit lors du rebase.
Sa branche sur GitHub est alors mise à jour, intégrant :
	•	les changements récents de develop (fichier1.py de Noam) ;
	•	son propre travail (fichier2.py de Matteo).

⸻

Résumé

Étape	Commande	Objectif
```
1	git add . && git commit -m "Travail en cours"	Sauvegarder son code local
2	git pull origin develop --rebase	Intégrer les nouveautés sans merge commit
3	git rebase --continue ou --abort	Gérer un conflit si nécessaire
4	git push -f	Mettre à jour la branche distante
```

⸻

Bonnes pratiques
	•	Toujours commiter avant de lancer un rebase.
	•	Ne jamais interrompre un rebase sans savoir où il en est.
	•	Utiliser le rebase uniquement sur sa branche personnelle, jamais sur develop ou main.

Ce processus garantit une intégration fluide des nouveautés du projet
sans écrasement ni pollution de l’historique Git.

---
