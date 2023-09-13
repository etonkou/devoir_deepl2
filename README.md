## **README** :

***bert_main.py*** :
contient les fonctions : 
- de chargement du dataset, de preprocessing,
- d'encodage des labels, 
- de separation des données en train et test,
- d'entrainement du modele
Une fois le modele entrainé, il est sauvergadé sur le Drive de Google


***bert_push_to_huggingface.py*** :
permet de se connecter a huggingface et de pusher le modele sauvegardé

***le lien huggingface est :***
[https://huggingface.co/etonkou/bert-classification](https://huggingface.co/etonkou/bert-classification)

***app.py***: 
Ce fichier contient le code permet de deployer notre modele sur gradio

***fast_api.py*** :
Ce fichier contient le code permet de deployer notre modele sur fastapi

***Dockerfile*** :
Dockerfile contient les commandes pour construire l'image et de deployer notre modele accessible via fastapi sur Docker
