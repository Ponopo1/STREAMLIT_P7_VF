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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# URL de l'API FastAPI déployée sur Heroku
API_URL = "https://api-projet7-open-bd8c05735794.herokuapp.com"

# Centrage de l'image du logo dans la sidebar
def logo() :
   col1, col2, col3 = st.columns([1,1,1])
   with col1:
      st.sidebar.write("")
   with col2:
      # Pousser une image dans le serveur pour la mettre
      image = Image.open('Logo.png')
      st.sidebar.image(image, use_column_width="always")
   with col3:
      st.sidebar.write("")
   st.sidebar.header('Dashboard `Crédit`')
   return col1, col2, col3

def liste_client() :
   # Faire la requête GET
   response = requests.get(f"{API_URL}/CLIENTS")
   # Récupérer les données au format JSON
   Client = response.json()
   # Extraire la liste des ID_CLIENT
   id_clients = Client.get("ID_CLIENT", [])
   # Créer la selectbox dans la barre latérale avec les ID_CLIENT
   selected_client = st.sidebar.selectbox('SELECT', id_clients)
   # Afficher l'ID_CLIENT sélectionné
   st.write(f"ID_CLIENT sélectionné : {selected_client}")
   return selected_client

# Ligne 1
def prediction(selected_client) :
   responsepred = requests.get(f"{API_URL}/predict",params={"id_client": selected_client})
   prediction = responsepred.json()
   prediction_value = prediction.get('prediction')
   # Visuel 
   st.markdown('### Indicateur prêt')
   if prediction_value<0.65 :
      return st.markdown(''':green[CREDIT VALIDE]'''), streamviz.gauge(prediction_value,gcLow="#00FF00",gcMid="#FFA500",gcHigh="#FFA500",grLow=0.7,grMid=0.65)
   else :
      return st.markdown(''':red[CREDIT REFUSE]'''), streamviz.gauge(prediction_value,gcLow="#00FF00",gcMid="#FFA500",gcHigh="#FFA500",grLow=0.,grMid=0.65)

# Ligne 2
def info_client(selected_client) :
   responseclient = requests.get(f"{API_URL}/INFO_CLIENTS",params={"ID_CLIENT": selected_client})
   information_client = responseclient.json()
   return st.markdown('### :blue[Information sur le client]'), st.table(information_client)


# Ligne 3
def influence_valeur(selected_client,width,height):
   shap_value_select = st.slider(label='Nombre de variable à visualiser',min_value=1,max_value=20)
   st.markdown('### :blue[Facteur d\'influence locale du résultat]')
   responseclient_shap = requests.get(f"{API_URL}/shap_individual",params={"ID_CLIENT": selected_client, # Pour sélectionner le client dans GET
                                                                           "shap_values_class_1": selected_client,
                                                                           "observation": selected_client,
                                                                           "columns": selected_client})
   information_client_shap = responseclient_shap.json()
   shap_values_class_1 = information_client_shap["shap_values_class_1"]
   observation_shap_ind = information_client_shap["observation"]
   columns_shap_ind = information_client_shap["columns"]

   # Générer le SHAP summary plot
   fig, ax = plt.subplots(figsize=(width, height))
   # Now convert to Explanation object
   shap_values_feat_ind = shap.Explanation(shap_values_class_1, feature_names=columns_shap_ind)
   shap.plots.bar(shap_values_feat_ind, max_display=shap_value_select)

   response_importance_feat = requests.get(f"{API_URL}/SHAP_GLOBAL")
   information_importance_feat = response_importance_feat.json()
   features = information_importance_feat.get('features')
   importances = information_importance_feat.get('importances')
   #std_dev = information_importance_feat.get('std_dev')
   # Graphique de l'importance des features
   #std_series = pd.Series(std_dev, index=features)
   forest_importances = pd.Series(importances, index=features)
   # Sélectionnez les X caractéristiques les plus importantes
   top_importances = forest_importances.nlargest(shap_value_select)
   #top_std = std_series[top_importances.index]  # Sélectionner les écarts-types correspondants
   # Créer le graphique à barres
   fig_shap_glob, ax_shap_glob = plt.subplots(figsize=(width, height))
   top_importances.plot.bar(ax=ax_shap_glob)#yerr=top_std
   ax_shap_glob.set_title("Feature importances générales")
   ax_shap_glob.set_ylabel("Mean decrease")
   ax_shap_glob.set_xticklabels(top_importances.index, rotation=45, ha='right')
   fig_shap_glob.tight_layout()
   return st.pyplot(fig),st.markdown('### :blue[Facteur d\'influence globaux]'),st.pyplot(fig_shap_glob)

# Ligne 4
def benchmark(selected_client,width,height) :
   data = requests.post(f"{API_URL}/INFO_CLIENTS_GLOBAL")
   df = pd.DataFrame(data.json())
   # BOXPLOT
   # ID de l'individu à mettre en évidence
   highlight_id = selected_client
   # Filtrer uniquement les variables numériques
   numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
   # Vérifier si 'ID' est dans les colonnes et le supprimer pour les données du boxplot
   if 'ID' in numeric_cols:
      numeric_cols = numeric_cols.drop('ID')
   # Exclure 'ID' des données utilisées pour le boxplot
   numeric_data = df[numeric_cols]
   # Normalisation Min-Max
   normalized_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
   # Création du boxplot avec matplotlib
   plt.figure(figsize=(width, height))
   boxplot = plt.boxplot(normalized_data.values, labels=numeric_cols, patch_artist=True)
   # Mettre en évidence l'individu sélectionné
   highlight_index = df[df['ID'] == highlight_id].index[0]
   highlight_data = normalized_data.iloc[highlight_index]
   # Correction pour le placement des étoiles
   positions = list(range(1, len(numeric_cols) + 1))  # Assurez-vous que positions correspond à la longueur de highlight_data
   plt.scatter(positions, highlight_data, color='yellow', edgecolor='black', marker='*', s=100, zorder=5, label=f'Client {highlight_id}')
   # Ajouter un titre et faire pivoter les étiquettes des axes
   plt.title("Boxplot des variables avec client")
   plt.xticks(rotation=45)
   plt.ylabel("Valeurs normalisées")
   plt.grid(True)
   plt.legend(loc='upper left')

   # Afficher le graphique
   return st.markdown('### :blue[Comparaison avec autres clients via Boxplot]'),st.pyplot(plt),numeric_cols

def scatter(Variable_1, Variable_2, selected_client):
    # Récupérer les données via l'API
    data = requests.post(f"{API_URL}/INFO_CLIENTS_GLOBAL")
    df = pd.DataFrame(data.json())

    # ID de l'individu à mettre en évidence
    highlight_id = selected_client

    # Récupérer les données du client à mettre en évidence
    highlight_row = df[df['ID'] == highlight_id]
    highlight_x = highlight_row[Variable_1].values[0]
    highlight_y = highlight_row[Variable_2].values[0]

    # Masque pour les clients refusés (selon un seuil proba > 0.65)
    refus = df['proba'] > 0.65

    # Création de la figure Plotly
    fig = go.Figure()

    # Ajouter les autres clients (validés)
    fig.add_trace(go.Scatter(
        x=df[Variable_1][~refus],
        y=df[Variable_2][~refus],
        mode='markers',
        marker=dict(color='blue'),
        name='Clients validés'
    ))

    # Ajouter les clients refusés
    fig.add_trace(go.Scatter(
        x=df[Variable_1][refus],
        y=df[Variable_2][refus],
        mode='markers',
        marker=dict(color='yellow', line=dict(width=1, color='black')),
        name='Clients refusés'
    ))

    # Ajouter le client à mettre en évidence
    fig.add_trace(go.Scatter(
        x=[highlight_x],
        y=[highlight_y],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='black')),
        name=f'ID {highlight_id}'
    ))

    # Mise en forme du graphique
    fig.update_layout(
        title=f'Scatter Plot de {Variable_1} vs {Variable_2}',
        xaxis_title=Variable_1,
        yaxis_title=Variable_2,
        autosize=True,  # Taille automatique
        height=600,     # Hauteur fixe, mais largeur dynamique
        legend=dict(x=0, y=1.1, orientation="h")
    )

    # Afficher le graphique dans Streamlit avec largeur dynamique
    st.plotly_chart(fig, use_container_width=True)

def distribution(Variable_1, Variable_2, selected_client):
    # Récupérer les données via l'API
    data = requests.post(f"{API_URL}/INFO_CLIENTS_GLOBAL")
    df = pd.DataFrame(data.json())

    # ID de l'individu à mettre en évidence
    highlight_id = selected_client

    # Récupérer les données du client à mettre en évidence
    highlight_row = df[df['ID'] == highlight_id]
    highlight_x = highlight_row[Variable_1].values[0]
    highlight_y = highlight_row[Variable_2].values[0]

    # Masque pour les clients refusés (selon un seuil proba > 0.65)
    refus = df['proba'] > 0.65

    # Création de la figure avec deux sous-graphiques côte à côte
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Distribution de {Variable_1}", f"Distribution de {Variable_2}"))

    # Histogramme pour Variable_1
    fig.add_trace(go.Histogram(
        x=df[Variable_1][~refus],
        name=f'{Variable_1} (Clients validés)',
        marker=dict(color='blue'),
        opacity=0.75
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=df[Variable_1][refus],
        name=f'{Variable_1} (Clients refusés)',
        marker=dict(color='yellow', line=dict(width=1, color='black')),
        opacity=0.75
    ), row=1, col=1)

    # Marqueur pour le client sélectionné (highlight) pour Variable_1
    fig.add_trace(go.Scatter(
        x=[highlight_x],
        y=[0],  # On ne se sert pas des y pour un histogramme, mais on met 0
        mode='markers',
        marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='black')),
        name=f'ID {highlight_id} ({Variable_1})'
    ), row=1, col=1)

    # Histogramme pour Variable_2
    fig.add_trace(go.Histogram(
        x=df[Variable_2][~refus],
        name=f'{Variable_2} (Clients validés)',
        marker=dict(color='blue'),
        opacity=0.75
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=df[Variable_2][refus],
        name=f'{Variable_2} (Clients refusés)',
        marker=dict(color='yellow', line=dict(width=1, color='black')),
        opacity=0.75
    ), row=1, col=2)

    # Marqueur pour le client sélectionné (highlight) pour Variable_2
    fig.add_trace(go.Scatter(
        x=[highlight_y],
        y=[0],
        mode='markers',
        marker=dict(color='red', size=15, symbol='star', line=dict(width=2, color='black')),
        name=f'ID {highlight_id} ({Variable_2})'
    ), row=1, col=2)

    # Mise en forme du graphique
    fig.update_layout(
        title_text=f"Distributions de {Variable_1} et {Variable_2}",
        autosize=True,
        height=600,  # Hauteur du graphique
        barmode='overlay',  # Superpose les barres d'histogrammes
        showlegend=True
    )

    # Ajustement des axes et titres
    fig.update_xaxes(title_text=Variable_1, row=1, col=1)
    fig.update_xaxes(title_text=Variable_2, row=1, col=2)
    fig.update_yaxes(title_text="Fréquence", row=1, col=1)
    fig.update_yaxes(title_text="Fréquence", row=1, col=2)

    # Afficher le graphique dans Streamlit avec largeur dynamique
    st.plotly_chart(fig, use_container_width=True)



