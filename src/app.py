"""
app.py
Aplicación web que implementa los modelos de identificación de entidades biomédicas.

@authors: Melissa Pérez, Sara Echeverría, Ricardo Méndez, Adrián Fulladolsa
"""

import streamlit as st
import joblib
import torch
from transformers import BertTokenizer, BertForTokenClassification
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx

# Descargar recursos de NLTK si no están disponibles
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocesamiento del texto
def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(tokens)

#Creación de grafo para GCN
def create_graph_from_input(abstract, entities):
    G = nx.Graph()
    
    # Add nodes for the user input abstract
    for idx, entity in enumerate(entities):
        unique_entity_id = f"user_{idx}_{entity}"
        G.add_node(unique_entity_id, abstract_id="user", entity=entity, x=np.random.rand(128))
    
    # Add edges between entities in the user input abstract
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            node_id_1 = f"user_{i}_{entities[i]}"
            node_id_2 = f"user_{j}_{entities[j]}"
            G.add_edge(node_id_1, node_id_2, relation_type="related")
    
    # Ensure all nodes have standard attributes
    standard_attributes = {'abstract_id': None, 'entity': None}
    for node in G.nodes:
        for attr, default_value in standard_attributes.items():
            if attr not in G.nodes[node]:
                G.nodes[node][attr] = default_value
    
    data = from_networkx(G)
    return data

# Cargar modelos y recursos necesarios
@st.cache_resource
def load_models():
    # Carga el modelo SVM y su vectorizador
    svm_model = joblib.load("./models/svm_model.pkl")
    
    # Carga el modelo GCN
    gcn_model, gcn_vectorizer = joblib.load("./models/gcn_model.pkl")
    
    # Carga el modelo BERT y su tokenizer
    bert_model, tokenizer = joblib.load("./models/bert_model.pkl")
    
    return svm_model, gcn_model, gcn_vectorizer, bert_model, tokenizer

# Cargar los modelos
svm_model,  gcn_model, gcn_vectorizer, bert_model, tokenizer = load_models()
vectorizer = TfidfVectorizer()

# Interfaz de la aplicación
st.title("Identificación de Entidades Biomédicas")
st.write("Ingrese un resumen biomédico y seleccione el modelo para predecir entidades.")

# Entrada del usuario
user_input = st.text_area("Resumen Biomédico", "Escribe aquí el texto del resumen...")
model_option = st.selectbox("Selecciona el modelo:", ["SVM", "GCN", "BERT"])

# Botón de predicción
if st.button("Predecir"):
    if not user_input.strip():
        st.error("Por favor ingrese un resumen válido.")
    else:
        # Preprocesar el texto
        preprocessed_text = preprocess_text(user_input)

        # Predicciones basadas en el modelo seleccionado
        if model_option == "SVM":
            # Vectorización del texto para SVM
            input_vector = vectorizer.transform([preprocessed_text])
            predictions = svm_model.predict(input_vector)
            st.write("Entidades Predichas:", predictions)
        
        elif model_option == "GCN":
            # Vectorización y conversión a tensor para GCN
            input_vector = create_graph_from_input(preprocessed_text, preprocessed_text.split())
            input_tensor = torch.tensor(input_vector, dtype=torch.float32)
            predictions = gcn_model(input_tensor)
            st.write("Entidades Predichas:", predictions.detach().numpy())
        
        elif model_option == "BERT":
            # Tokenización y predicción con BERT
            inputs = tokenizer(preprocessed_text, return_tensors="pt", truncation=True, padding=True)
            outputs = bert_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            entity_results = {token: prediction.item() for token, prediction in zip(tokens, predictions[0])}
            st.write("Entidades Predichas:", entity_results)

        # Confirmación de predicción exitosa
        st.success("Predicción completada con éxito.")
