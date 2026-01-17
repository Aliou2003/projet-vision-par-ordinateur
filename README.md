# 🚀 MobileNet Segmentation d'images avec MobileNetV2 - Projet Vision par Ordinateur 
**Projet réalisé dans le cadre du module Vision par ordinateur** - Démonstration d'un pipeline complet de Machine Learning opérationnel, de l'entraînement jusqu'à l'application Streamlit. 
Dévelopé par DIALLO Mamadou Aliou, DIALLO Mamadou Dian & CHARKANI EL HASSANI Mohammed 

## 📋 Résumé du Projet
Déploiement d'un modèle **MobileNetV2** pour la segmentation sémantique sur le dataset **Oxford-IIIT Pets**, avec une interface Streamlit.

**Période :** 27/12/2025 au 18/01/2026  

**Encadrement :** Module Vision par ordinateur 

**Niveau :** 5IIIA

## 🏗️ Architecture Technique
(voir architecture.png)


🎨 Dataset & Modèle
Dataset : Oxford-IIIT Pets (37 catégories, 7,349 images)

Tâche : Segmentation sémantique (pixels -> classes animaux)

Modèle : MobileNetV2 + U-Net decoder

🛠️ Stack Technologique

Catégorie	Technologies

ML/DL	TensorFlow 2.x, MobileNetV2

Backend	Streamlit, Python 3.9, NumPy


🚀 Guide de Déploiement Rapide

1. Local Development
   
bash

cd projet-vision-par ordinateur

pip install -r requirements.txt

streamlit run app.py
