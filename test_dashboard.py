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
    @patch('streamlit.write')
    @patch('streamviz.gauge')
    def test_prediction_green(self, mock_gauge, mock_write, mock_markdown, mock_requests_get):
        # Simulation de l'api qui renvoit 0.9
        mock_response = MagicMock()
        mock_response.json.return_value = {'prediction': 0.9}
        mock_requests_get.return_value = mock_response

        # On applique la fonction
        prediction(12345)  

        # Check si on a bien le rouge qui ressort
        mock_markdown.assert_any_call(f"<h1 style='color: red; font-size: 36px;'>CREDIT REFUSE</h1>",unsafe_allow_html=True)
        print('La refus fonctionne')
    
    @patch('requests.get')
    @patch('streamlit.markdown')
    def test_prediction_red(self, mock_gauge, mock_write, mock_markdown, mock_requests_get):
        # Simulation de l'api qui renvoit 0.5
        mock_response = MagicMock()
        mock_response.json.return_value = {'prediction': 0.5}
        mock_requests_get.return_value = mock_response

        # On applique la fonction
        prediction(12345)  

        # Check si on a bien le vert qui ressort
        mock_markdown.assert_any_call(f"<h1 style='color: green; font-size: 36px;'>CREDIT VALIDE</h1>",unsafe_allow_html=True)
        print('Le validation fonctionne')

if __name__ == '__main__':
    unittest.main()
