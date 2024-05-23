# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:41:12 2024

@author: cedri
"""

import unittest
from unittest.mock import patch
import requests
from Data_Engineering_Project_ETL import DataEngineeringProject

class TestDataEngineeringProject(unittest.TestCase):

    @patch('requests.get')
    def test_fetch_data_from_api_success(self, mock_get):
        # Créer une instance de la classe DataEngineeringProject
        project = DataEngineeringProject()
        
        # Définir le comportement attendu de mock_get
        mock_get.return_value.json.return_value = {"key": "value"}
        
        # Appeler la méthode fetch_data_from_api
        result = project.fetch_data_from_api()
        
        # Vérifier que la méthode a retourné les données attendues
        self.assertEqual(result, {"key": "value"})

    @patch('requests.get')
    def test_fetch_data_from_api_failure(self, mock_get):
        # Créer une instance de la classe DataEngineeringProject
        project = DataEngineeringProject()
        
        # Définir le comportement attendu de mock_get pour lever une exception
        mock_get.side_effect = requests.exceptions.RequestException("Error")
        
        # Appeler la méthode fetch_data_from_api
        result = project.fetch_data_from_api()
        
        # Vérifier que la méthode a retourné None en cas d'échec
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
