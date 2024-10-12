import streamlit as st
import pandas as pd
import numpy as np
import plost
import pandas as pd
import joblib
from pydantic import BaseModel
import plotly.graph_objects as go
import streamviz
from PIL import Image
import shap
import requests
import json
import matplotlib.pyplot as plt
from Streamlit_function import logo,liste_client,prediction,info_client,influence_valeur,benchmark,scatter

# URL de l'API FastAPI déployée sur Heroku
API_URL = "https://api-projet7-open-bd8c05735794.herokuapp.com"

# 4. Mise en page, graphe streamlit
st.set_page_config(layout='wide', initial_sidebar_state='expanded') # Permet d'avoir la fenetre qui s'étend avec la largeur de la page

# Centrage de l'image du logo dans la sidebar

logo()

# SELECTION DU CLIENT
st.sidebar.subheader('CLIENT')

selected_client = liste_client()

# Sélection des informations à visualiser
info_desc = st.sidebar.checkbox('Information descriptives du client')
fact_influ = st.sidebar.checkbox('Facteur d\'influence')
Benchmark = st.sidebar.checkbox('Comparaison avec les autres clients')

width = st.sidebar.slider("plot width", 1, 25, 3)
height = st.sidebar.slider("plot height", 1, 25, 1)


# Ligne 1
prediction(selected_client)


# Ligne 2
if info_desc == True :
   info_client(selected_client)
else :
   st.markdown('')

# Ligne 3
if fact_influ == True :
   influence_valeur(selected_client,width,height)

# Ligne 4
if Benchmark == True :
   #benchmark(selected_client,width,height)
   _, _, numeric_cols = benchmark(selected_client, width, height)

   # SCATTER PLOT
   st.markdown('### :blue[Comparaison avec autres clients via Scatter]')
   col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
   with col1:
      st.sidebar.write("")
   with col2:
      Variable_1 = st.selectbox('SELECT Varible 1',numeric_cols)
   with col3:
      st.sidebar.write("")
   with col4:
      Variable_2 = st.selectbox('SELECT Varible 2',numeric_cols)
   with col5:
      st.sidebar.write("")

   scatter(Variable_1,Variable_2,selected_client,width,height)
   
else :
   st.markdown('')
