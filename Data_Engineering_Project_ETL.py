# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:17:52 2024

@author: cedri
"""
##Librairie 
import requests
import pandas as pd
import numpy as np
import re
import logging  # Import du module logging

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

from flask import Flask, jsonify, request
from datetime import datetime, timedelta



from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer


# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.operators.bash import BashOperator
from nltk.stem import SnowballStemmer
from urllib.parse import quote


class DataEngineeringProject:
    def __init__(self):
        """
        
        Initialisation des variables URL et des données
        
        """
        # Initialisation des variables URL et des données
        self.url = "https://data.ademe.fr/data-fair/api/v1/datasets/les-aides-financieres-de-l%27ademe/data-files"
        self.data = None
        self.columns_remoove = ['Nom de l attribuant', 'idAttribuant']
        self.time_columns = ['dateConvention', 'datesPeriodeVersement']
        self.colonne_remoove = ["objet", "Activité Principale Libelle", "Activité Principale Libelle 4",
                                "Activité Principale Libelle 3", "Activité Principale Libelle 2",
                                "Activité Principale Libelle 1"]
        self.number_keywords = 80
        self.list_keyword_extracted = []
        self.liste_kw_retirer = ['autre', 'autres', 'adhésion', 'action', 'activités', 'conditionné', 'courte',
                                 'durée', 'gros', 'similaire', 'faisabilité', 'détail', 'générale', 'élimination',
                                 'dangereux', 'biens', 'exception', 'fonctionnant', 'obligatoire', 'projet',
                                 'volontaire', 'supérieur', 'faisabilité', 'exception', 'générale']
        self.dico_replace = {'sociaux': 'sociale', 'produits': 'production', 'sciences': 'scientifiques',
                             'économique': 'économiques', 'spécialisé': 'spécialisées'}
        
        # Enregistrement fichier log
        # Configuration du système de journalisation
        logging.basicConfig(level=logging.INFO)

        # Création du gestionnaire de fichier pour enregistrer dans un fichier 'data_engineering.log'
        file_handler = logging.FileHandler('data_engineering.log')

        # Configuration du format du message
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # Ajout du gestionnaire de fichier au logger de la classe
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(file_handler)

        self.logger.info("Application lancée")  # Enregistrement d'un message au lancement


        print("Etape 1 : Application se lance")
        print("....................")
        
    def fetch_data_from_api(self):
        
        """
        Appel de l'API' via l'URL de l'ADEME
        
        Args : 
            String : URL ADEME
        Returns : 
            Dictionnaire : Dictionnaire comportant les URL des différents fichiers de L'ADEME'
        Raises :
            ValueError
        
        """
        try:
            self.logger.info("Appel de l'API via l'URL de l'ADEME")
            print("Etape 2 : Appel de l'API via l'URL de l'ADEME")
            print("....................")
            response = requests.get(self.url)
            response.raise_for_status()
            data_ = response.json()
            print("Pas de problème lors de la requête", response.status_code)
            return data_
        except requests.exceptions.RequestException as e:
            print(f'Erreur lors de la requête à l\'API : {e}')
            return None

    def load_data_from_csv(self, url_csv):
        
        """
        Récupération du lien csv choisi
        
        Args : 
            String : URL ADEME
            
        Returns : 
            DataFrame : Dataframe crée à partir du fichier CSV
            
        Raises :
            ValueError
        
        """
        self.logger.info("Chargement des données depuis le fichier CSV")
        print("Etape 3 : Chargement des données depuis le fichier CSV")
        print("....................")
        url_encoded = quote(url_csv, safe=':/')
        data = pd.read_csv(url_encoded, sep=',', header=0, encoding="utf-8", low_memory=False)
        return data
    
    def rename_columns (self, data) :
        
        """
        Récupération du lien csv choisi
        
        Args : 
            DataFrame : Data
            
        Returns : 
            DataFrame : Dataframe avec les colonnes renommées
            
        """
        self.logger.info("Renommage des colonnes")
        rename_columns = {
                                        'ï»¿Nom de l attribuant' : 'Attribuant',  
                                        'dateConvention':'dateConvention',
                                       '_siret_infos.activitePrincipaleEtablissementNAFRev2Libelle' : 'Activité Principale Libelle',
                                       '_siret_infos.activitePrincipaleEtablissementNAFRev2LibelleNiv4': 'Activité Principale Libelle 4',
                                       '_siret_infos.activitePrincipaleEtablissementNAFRev2LibelleNiv3': 'Activité Principale Libelle 3',
                                       '_siret_infos.activitePrincipaleEtablissementNAFRev2LibelleNiv2': 'Activité Principale Libelle 2',
                                       '_siret_infos.activitePrincipaleEtablissementNAFRev2LibelleNiv1': 'Activité Principale Libelle 1',
                                       '_siret_infos._siret_coords.y_latitude' : 'y_latitude',
                                       '_siret_infos._siret_coords.x_longitude': 'x_longitude',
                                       '_siret_infos._infos_commune.code_departement' :'code_departement' ,
                                       '_siret_infos._infos_commune.nom_departement' : 'nom_departement'

                            }
      
        data.rename(columns = rename_columns, inplace=True)
        # Utilise la méthode drop pour enlever les colonnes spécifiées
        colonnes_a_enlever = ['Nom de l attribuant', 'idAttribuant']
        data = data.drop(columns=colonnes_a_enlever)
    
        return data
       
    def missing_values_treatment (self, data) :
        
        """
        Récupération du lien csv choisi
        
        Args : 
            DataFrame : Data
            
        Returns : 
            DataFrame : Dataframe avec les valeurs manquantes traitées
            
        """
        self.logger.info("Traitement des valeurs Manquantes")
        # Sélectionner les colonnes catégorielles
        colonnes_categorielles = data.select_dtypes(include=['object'])
        
        # Traitements des valeurs numériques
        noms_colonnes_categorielles = colonnes_categorielles.columns.tolist()
        data[noms_colonnes_categorielles] = data[noms_colonnes_categorielles].fillna("Autres")
        # Convertir toutes les valeurs en minuscules
        data[noms_colonnes_categorielles] = data[noms_colonnes_categorielles].apply(lambda x: x.lower() if isinstance(x, str) else x)

        # Traitements des valeurs numériques
        data["notificationUE"] = data["notificationUE"].fillna(0)
        data["idBeneficiaire"] = data["idBeneficiaire"].fillna(0)
        data = data.dropna()
        
        return data
    
    def time_treatment (self, data) :
        
        
        """
        Récupération du lien csv choisi
        
        Args : 
            DataFrame : Data
            
        Returns : 
            DataFrame : Dataframe avec les colonnes de types "dates" retravaillées
            
        """
        data_= data.copy()
        # Divise la colonne "Periode" en deux colonnes distinctes
        data_[['DateDebut Versement', 'DateFin Versement']] = data_['datesPeriodeVersement'].str.split('_', expand=True)
        data_['DateFin Versement'] = data_['DateFin Versement'].fillna(data_['DateDebut Versement'])
        
        # Convertit les nouvelles colonnes en objets datetime
        data_['DateDebut Versement'] = pd.to_datetime(data_['DateDebut Versement'], format='%Y-%m-%d', errors='coerce')
        data_['DateFin Versement' ] = pd.to_datetime(data_['DateFin Versement'], format='%Y-%m-%d', errors='coerce')
        
        # Utilise .loc[] pour éviter le SettingWithCopyWarning
        data_.loc[:, 'DateDebut Versement'] = pd.to_datetime(data_['DateDebut Versement'])
        data_.loc[:, 'DateFin Versement' ] = pd.to_datetime(data_['DateFin Versement'])
        
        # Calculer la différence en jours et créer une nouvelle colonne
        data_.loc[:, 'Nombre Jours versement'] = (data_['DateFin Versement'] - data_['DateDebut Versement']).dt.days + 1
        
        return data_ 

    
    @staticmethod
    def stop_word_init () :
        
        """
        Initialisation de la liste avec les stop words. Plus ajout de stop words supplémentaires.
        
        """
        # Liste de stopwords personnalisés
        custom_stop_words = {
                        'au', 'aux', 'avec', 'ce', 'ces', 'dans', 'de', 'des', 'du', 'elle', 'en',
                        'et', 'eux', 'il', 'je', 'la', 'le', 'leur', 'lui', 'ma', 'mais', 'me',
                        'même', 'mes', 'moi', 'mon', 'ne', 'nos', 'notre', 'nous', 'on', 'ou',
                        'par', 'pas', 'pour', 'qu', 'que', 'qui', 'sa', 'se', 'ses', 'son', 'sur',
                        'ta', 'te', 'tes', 'toi', 'ton', 'tu', 'un', 'une', 'vos', 'votre', 'vous',
                        'c', 'd', 'j', 'l', 'à', 'm', 'n', 's', 't', 'y', 'été', 'étée', 'étées',
                        'étés', 'étant', 'étante', 'étants', 'étantes', 'suis', 'es', 'est', 'sommes',
                        'êtes', 'sont', 'serai', 'seras', 'sera', 'serons', 'serez', 'seront', 'serais',
                        'serait', 'serions', 'seriez', 'seraient', 'étais', 'était', 'étions', 'étiez',
                        'étaient', 'fus', 'fut', 'fûmes', 'fûtes', 'furent', 'sois', 'soit', 'soyons',
                        'soyez', 'soient', 'fusse', 'fusses', 'fût', 'fussions', 'fussiez', 'fussent',
                        'pour',"d'","l'",'0','1','2','3','4','5','6','7','8','9'
                    }
        
        # Charger les stopwords en français depuis NLTK
        stop_words = set(stopwords.words('french'))
        
        # Étendre la liste de stopwords avec les mots personnalisés
        stop_words.update(custom_stop_words)
        
        return stop_words
    
    @staticmethod
    def remove_stopwords_and_digits(sentence):
        
        """
        Fonction qui enlève les stop words de chaque ligne
        
        Args : 
            String : Ligne de chaque dataframe
            
        Returns : 
            String : Ligne de chaque dataframe avec les stop words retirés
        """
        stop_words = DataEngineeringProject.stop_word_init()
        
        if isinstance(sentence, str):  # Vérifier si la valeur est une chaîne
            words = word_tokenize(sentence)
            filtered_words = [word for word in words if word.lower() not in stop_words and not word.isdigit()]
            return ' '.join(filtered_words)
        else:
            return sentence
        
    def apply_remove_stop_word(self, data):
        
        """
        Application de la fonction stopword au dataframe
        """
        data_copy = data.copy()  # Créer une copie explicite du DataFrame
    
        for col in self.colonne_remoove:
            data_copy.loc[:, col] = data_copy[col].apply(self.remove_stopwords_and_digits)
    
        return data_copy  

    def text_cleaning(self, text):
        """
        Remove figures, punctuation, words shorter than two letters (excepted C or R) in a lowered text. 
        
        Args:
            text(String): Row text to clean
            
        Returns:
           res(string): Cleaned text
        """
        pattern = re.compile(r'[^\w]|[\d_]')
        
        try: 
            res = re.sub(pattern, " ", text).lower()
        except TypeError:
            return text
        
        res = res.split(" ")
        res = list(filter(lambda x: len(x) > 3, res))
        res = " ".join(res)
        return res

    def apply_text_cleaning_and_strip(self, data):
        for col in self.colonne_remoove:
            data[col] = [self.text_cleaning(text).strip() for text in data[col]]

        return data

    @staticmethod
    def tokenize_text(sentence):
        return word_tokenize(sentence)
    
    @staticmethod
    def reverse_tokenize(tokens):
        return ' '.join(tokens)

    def apply_tokenization(self, data):
        for col in self.colonne_remoove:
            data[col] = data[col].apply(self.tokenize_text)
        
        return data

    def lemmatize_text_fr(self,sentence):
        if isinstance(sentence, str):  
            tokens = word_tokenize(sentence)
            stemmer = SnowballStemmer('french')  
            lemmatized_tokens = [stemmer.stem(token) for token in tokens]
            return ' '.join(lemmatized_tokens)
        else:
            return sentence
        
    def apply_lemmatization(self, data):
        for col in self.colonne_remoove:
            data[col] = data[col].apply(self.lemmatize_text_fr)
        
        return data
    
    def apply_reverse_tokenization(self, data):
        for col in self.colonne_remoove:
            data[col] = data[col].apply(self.reverse_tokenize)
        return data
    
    def keyword_extraction (self, data) :
        
        """
        Fonction qui enlève les stop words de chaque ligne
        
        Args : 
            String : Ligne de chaque dataframe
            
        Returns : 
            String : Ligne de chaque dataframe avec les stop words retirés
        """
        data["Activité"] = ""
        for col in self.colonne_remoove:
            data["Activité"] = data["Activité"] + " " + data[col].astype(str)
        # Appliquez TF-IDF pour extraire des mots-clés
        vectorizer = TfidfVectorizer(max_features=self.number_keywords)  # Choisissez le nombre maximum de mots-clés à extraire
        keywords_matrix = vectorizer.fit_transform(data["Activité"])
        # Obtenez les noms des mots-clés extraits
        keywords = vectorizer.get_feature_names_out()
        liste_keyword = [mot for mot in keywords if mot not in self.liste_kw_retirer]
        self.list_keyword_extracted = liste_keyword
        return liste_keyword
    
    
    def extract_keywords(self, sentence):
        keywords_found = [word for word in self.list_keyword_extracted if word in sentence]
        return keywords_found if keywords_found else ['autre']   
    
    def apply_keywords_transformation(self, data):
        
        for col in self.colonne_remoove:
            data[col] = data[col].replace(self.dico_replace)
            data[col] = data[col].apply(self.extract_keywords)
            
        return data
    
    def process_data(self, data):
        """
        Applique toutes les transformations du DataFrame
        """
        self.logger.info("Processus de transformation de la Data")
        print("Etape 4 : Processus de transformation de la Data")
        print("....................")
        # 2. Transformation des données
        # 2.1 Renommage des colonnes
        data_col_rename = self.rename_columns(data)

        # 2.2 Traitement des valeurs manquantes
        data_missing_values = self.missing_values_treatment(data_col_rename)

        # 2.3 Traitement des valeurs de dates
        data_time_treatment = self.time_treatment(data_missing_values)

        # 2.4 Suppression des stopwords dans le dataframe
        data_remove = self.apply_remove_stop_word(data_time_treatment)

        # 2.5 Appliquer la fonction de nettoyage de texte et strip
        data_cleaned = self.apply_text_cleaning_and_strip(data_remove)

        # 2.6 Appliquer la tokenization
        data_tokenized = self.apply_tokenization(data_cleaned)

        # 2.7 Appliquer la lemmatization
        data_lemmatized = self.apply_lemmatization(data_tokenized)

        # 2.8 Extraire la liste de mots-clés
        list_kw = self.keyword_extraction(data_lemmatized)

        # 2.9 Appliquer la transformation des mots-clés
        data_keyword = self.apply_keywords_transformation(data_lemmatized)

        # 2.10 Appliquer l'inversion de la tokenization
        data_reverse_tokenization = self.apply_reverse_tokenization(data_keyword)

        return data_reverse_tokenization
            
    def save_cleaned_data(self, data):
        self.logger.info("Sauvegarde de la Data")
        print("Etape 5 : Sauvegarde de la Data")
        print("....................")
        data.to_csv('data_clean_ademe.csv', index=False)
        data.to_excel("data_clean_ademe.xlsx")