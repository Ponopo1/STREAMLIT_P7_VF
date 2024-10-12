import unittest
from unittest.mock import patch, MagicMock
import os
from Streamlit_function import prediction

# Obtenir le répertoire du fichier actuel 
current_file_directory = os.path.dirname(__file__)

class TestApi(unittest.TestCase):
    API_URL = "https://api-projet7-open-bd8c05735794.herokuapp.com"
    @patch('requests.get')
    @patch('streamlit.markdown')
    def test_prediction_green(self, mock_markdown, mock_requests_get):
        # Simulation de l'api qui renvoit 0.9
        mock_response = MagicMock()
        mock_response.json.return_value = {'prediction': 0.9}
        mock_requests_get.return_value = mock_response

        # On applique la fonction
        prediction(12345)  

        # Check si on a bien le vert qui ressort
        mock_markdown.assert_any_call(''':green[CREDIT VALIDE]''')
        print('La validation fonctionne')
    
    @patch('requests.get')
    @patch('streamlit.markdown')
    def test_prediction_red(self, mock_markdown, mock_requests_get):
        # Simulation de l'api qui renvoit 0.5
        mock_response = MagicMock()
        mock_response.json.return_value = {'prediction': 0.5}
        mock_requests_get.return_value = mock_response

        # On applique la fonction
        prediction(12345)  

        # Check si on a bien le vert qui ressort
        mock_markdown.assert_any_call(''':red[CREDIT REFUSE]''')
        print('Le refus fonctionne')

if __name__ == '__main__':
    unittest.main()
    

