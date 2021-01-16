# Pretrained BERT Cased model

## Script
"BERT_cased_Laura" sur Google Colab:  
Avant d'utiliser le script:https://colab.research.google.com/drive/1GRgupPihbt34MCFdOeZp51nICD0yYkqv?usp=sharing  
Importer le dataset "lauraDataset2.csv "dans :/content

## Infos BERT model:  
https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270
## Infos Pretrained Bert model cased: 
https://towardsml.com/2019/09/17/bert-explained-a-complete-guide-with-theory-and-tutorial/
## Le Dataset

Le dataset provient du dataset ICDAR   
résultat du préprocessing donne un dataset organisé de la manière suivante:  
-Séparé en "sentence" (equivalent à un reçu)  
-Labélisé en BIO B: Begining, I: Inside, O: Outside  
-POS-taggé: Ajout d'un tag de position gramaticale : adjectif, verbe, prépositions...  

![alt text](https://github.com/LauraBreton-leonard/PRD/blob/main/NER/BERT/IMAGES/dataImg.PNG?raw=true)

## Résultats


![alt text](https://github.com/LauraBreton-leonard/PRD/blob/main/NER/BERT/IMAGES/learning_curve.png?raw=true)

![alt text](https://github.com/LauraBreton-leonard/PRD/blob/main/NER/BERT/IMAGES/f1_score.png?raw=true)

![alt text](https://github.com/LauraBreton-leonard/PRD/blob/main/NER/BERT/IMAGES/recall.png?raw=true)

![alt text](https://github.com/LauraBreton-leonard/PRD/blob/main/NER/BERT/IMAGES/precision.png?raw=true)

### Interprétation des résultats

Les résultats obtenus sont inexploitables pour plusieurs raisons:  
-L'overfit est inévitable compte tenu du dataset redondant: en effet, nous nous sommes rendus compte que le dataset était redondant:beaucoup de tickets des memes magasins avec mêmes totaux et dates
-Les bons résultats de precision, recall et F1 score proviennent de l'inégalité de la quantité de données pour chaque labels (beaucoup plus de 0)  
![alt text](https://github.com/LauraBreton-leonard/PRD/blob/main/NER/BERT/IMAGES/prpLabels.png?raw=true)  
-les données d'entrainement sont en vietnamien alors que le model est en anglais, les bons résultats (f1 score) proviennent de l'overfit. Un essai sur un ticket anglais montre rapidement l'inefficacité du modèle.

### Conclusion
Même si ces résultats sont inexploitables, nous disposons d'un bon pipeline de donées, ainsi que d'un modèle cohérent avec la tâche à accomplir. nous espérons donc pouvoir tester l'entrainement de ce modèle avec un meilleur dataset.  
Nous continuons à travailler sur d'autres pistes en parallèle.  

### Pistes travaillées en parallèle:  

2 pistes:  
-Travail sur du Reg_ex pour postTraitement des résultats  
-Travail sur un modèle personalisé multi input

