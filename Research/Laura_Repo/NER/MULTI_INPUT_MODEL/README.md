# Modèles simples et Multi Input Modèles (en cours de construction)

## Script  

importer le fichier "malaisianReceipts.hf" dans ':/content' dans google Colab

https://colab.research.google.com/drive/17bLMi7Qo4vloJxfuRFhKVNfDH08BXwwT?usp=sharing

## Desciption des données
Voici un apercu de la répartition du texte sur les tickets de caisse en fonction de leur label (bleu= date, rose= company, jaune= total, blanc= adresse...)  
<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/bbox.png" width="250" height="250"/>  
  
  
On remarque bien qu'il y a une correlation entre la position du texte est son label.  
D'ou la possiblienécéssité d'unmodel multi-Input (modèles 4 et 5)  

Une features d'entrée pourra donc être la position du texte.  
La deuxième feature d'entrée sera le texte en lui-même.  
Voici un apercu du dataFrame  
![alt text](https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/dataFrame.PNG?raw=true)

## Description des models
1) TFIDF +Logistic Regression  
- vectorization TFIDF avec keras puis Naive Bayes multinomial regression avec scikit-learn

2) Embedding layer +Bidirectional LSTM

3) Embedding layer + CNN + LSTM

4) Multi Input model avec 2)+ coordonees de position du texte

5) multi-input model avec 3)+ coordonees de position du texte

6) les modèles précédents avec une "pretrained embedding layer" (pas encore fait) 

modèle2:    

<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model2.png" width="250" height="250"/>  

modèle3:    

<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model3.png" width="250" height="250"/>   
modèle 3 (bis): Embedding+ Conv1D+LSTM  
  
  
<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model3(1).png" width="250" height="250"/>   


 
modèle4:  

<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model4.png" width="250" height="250"/>   
modèle5:  

<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model5(1).png" width="250" height="250"/>   

## Résultats  
-10 epochs  
-batch-size=64
### Modèle 2  
<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model_2_10ep_resultats.PNG"/> 

On retrouve de bons resultats globalement surtout pour les classes 2 et 3 ('company' et 'adresse' respectivement.)  

Les resultats sont moins satisfaisant sur les classes 1 et 4 (respectivement 'total' et 'date'):  

L'erreur principale commise est l'attribution du label 0 ('pas d'intéret') à la place de 'date ou'total'. 
2 raisons:  
1) les classes sont "imbalenced": C'est à dire qu'il y a beaucoup plus (énormément) de phrases labélisées 0 dans le dataset. Le modèle attribue donc 0 dans la majorité des cas étant donné qu'il est presque sur d'augmenter son accuracy en faisant cela. La représentation de 'date' et 'total' n'étant pas claire pour le modèle, on tombe facilement dans cette erreur lorsqu'il s'agit de ces deux labels.  

2) La définition de 'date et'total' n'est pas claire: Dans le dataset, le total apparait plusieurs fois ainsi que des sous-totaux. Seulement les totaux principaux sont etiquetés. Le modèle a donc du mal à se représenter ce qu'est un total.  
C'est la même idée pour les dates: plusieurs dates sont présentes dans les recus, mais seulement la date d'emission est etiquetée( par ex). 
  
  
Les deux problèmes sont donc liés  
  
  
Solutions proposées:  

Problème 1: SMOTE pour equilibrer le nombre de données par classe  
Problème2: Etiqueter les dataset plus précisément: total TVA, sous total, total après remise etc... + faire passer le dataset en mode chevauchement (voir methode mathieu), pour pouvoir voir les mots avant et après.  

On s'attend à régler le problème du total grâce au modèle 4 (inclue les données de position) puisque les totaux sont situés dans une zone bien précise des tickets comme on peut le voir ici. Les dates ne présentent pas la même particularité. Cependant, les dates sont facilement extractable grâce à du simple Regex(Regular expression. Ce résultat n'est donc pas trop inquétant.


### Modèle 3    

Modèle 3: Embedding layer+Conv1D+Maxpooling  

<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model_3_conv_10_ep_resultats.PNG "/> 
   
Bons résultats 

Modèle 3 bis: Embedding layer+Conv1D+LSTM

<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model_3_10ep_resultats.PNG"/>   
  
Très mauvais résultats en comparaison avec le modèle 2 ou 3.


### Modèle 4  
  
<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model4_10ep_resultats.PNG"/>  
La deuxième entrée de features (coordonée geographiques) semblent perturber la labelisation des totaux alors que c'est celle qui devrait être corrigé logiquement... étonnant

### Modèle 5  
  Modèle: modèle 3 conv + coordonees geo  
  
<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model_5_conv_10ep_resultats.PNG"/>  
  
  idem que pour 4  
  
  Modèle 5( Bis)
  
<img src="https://github.com/LauraBreton-leonard/PRD/blob/main/NER/MULTI_INPUT_MODEL/IMAGES/model_5_10ep_resultats.PNG"/>  

## Conclusion et Problèmes rencontrés

Plusieurs models fonctionnels qui semblent donner de bons résultats (85% accuracy apres 2 epochs pour certains) sur le dataset "AG news classification dataset" (classement de titres d'articles selon quatre domaines : sport, business, world....)  

Problème: Pas de dataset correct pour tester les modèles sur notre type de donnés, à savoir les tickets de caisse.  

-On ne peut pas tester  
-on ne peut pas faire varier les parametres de nos modèles  qui sont liés au dataset utilisé pour l'entrainement final (nb couches, dimensions, fonctions d'activation...) 
-Il faudrait tester l'influence des pretrained embedding layers. Cette influence est étroitement liée au dataset: taille du dataset, type de language utilisé...  


## Améliorations possibles
1) Intégrer une instance d'un model Bert sur la branche NLP ou au moins la couche d'embedding de BERT pré-entrainé ou autre embedding pré entrainé   

2) Essayer SMOTE ou cost sensitive learning pour imbalanced multiclass classification problem
