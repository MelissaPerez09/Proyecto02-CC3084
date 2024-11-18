import streamlit as st
import joblib
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
from torch_geometric.data import Data
import networkx as nx
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

ENTITY_CLASSES = ["Protein", "Disease", "Chemical", "Gene", "Organ", "Pathway"]

# Define the GCN model architecture
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Path to the GCN model
gcn_model_path = "./models/gcn_model.pkl"
bert_model_path = "./models/bert_model.pkl"

# Load and reconstruct the GCN model
@st.cache_resource
@st.cache_resource
def load_gcn_model(model_path):
    try:
        gcn_state_dict = joblib.load(model_path)
        out_channels = len(gcn_state_dict["conv2.bias"])  # Inferir canales de salida
        model = GCN(in_channels=128, hidden_channels=16, out_channels=out_channels)
        model.load_state_dict(gcn_state_dict)
        model.eval()  # Configurar en modo evaluación
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo GCN: {e}")
        return None

@st.cache_resource
def load_bert_model(model_path):
    try:
        # Cargar state_dict
        bert_state_dict = joblib.load(model_path)
        # Inicializar modelo y tokenizador BERT predefinido
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(ENTITY_CLASSES))
        model.load_state_dict(bert_state_dict)
        model.eval()  # Modo evaluación
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error al cargar el modelo BERT: {e}")
        return None, None

# Load models
gcn_model = load_gcn_model(gcn_model_path)
bert_model, bert_tokenizer = load_bert_model(bert_model_path)

# Preprocess text for GCN input
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = wordpunct_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Create graph data for GCN
def create_graph_from_input(tokens):
    G = nx.Graph()
    for idx, token in enumerate(tokens):
        G.add_node(idx, token=token, x=np.random.rand(128))  # Random embeddings as example

    # Sequential edges
    edge_index = torch.tensor([[i, i + 1] for i in range(len(tokens) - 1)] +
                              [[i + 1, i] for i in range(len(tokens) - 1)], dtype=torch.long).T

    # Optimize tensor conversion
    x = torch.tensor(np.array([G.nodes[node]["x"] for node in G.nodes]), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index), tokens

# Streamlit App
st.title("Biomedical Entity Identification ")
st.write("Select a model to predict entities in a biomedical abstract.")

# Model selection
model_choice = st.selectbox("Model:", options=["GCN", "BERT"])

# User input
user_input = st.text_area("Biomedical Abstract:", "")

# Predict button
if st.button("Predict"):
    if not user_input.strip():
        st.error("Please enter a valid abstract.")
    else:
        # Preprocess the input text
        tokens = preprocess_text(user_input)

        # Perform prediction with GCN
        if model_choice == "GCN" and gcn_model:
            graph_data, token_list = create_graph_from_input(tokens)
            try:
                gcn_prediction = gcn_model(graph_data.x, graph_data.edge_index)
                probabilities = torch.sigmoid(gcn_prediction).detach().numpy()

                # Map predictions to tokens and entity classes
                grouped_predictions = {entity: [] for entity in ENTITY_CLASSES}
                for idx, token in enumerate(token_list):
                    for i, prob in enumerate(probabilities[idx]):
                        if prob > 0.5:
                            grouped_predictions[ENTITY_CLASSES[i]].append((token, prob))

                # Display predictions grouped by entities
                st.subheader("GCN Model Predictions (Grouped by Entity)")
                for entity, token_confidences in grouped_predictions.items():
                    if token_confidences:
                        st.markdown(f"### **{entity}:**")
                        token_confidences.sort(key=lambda x: x[1], reverse=True)
                        st.write(", ".join([f"{token} ({conf:.2f})" for token, conf in token_confidences]))
                        
                        df_predictions = pd.DataFrame(probabilities, columns=ENTITY_CLASSES, index=token_list)
                st.subheader("Prediction Probabilities for Each Token")
                st.dataframe(df_predictions)
            except Exception as e:
                st.error(f"Error durante la predicción con GCN: {e}")

        elif model_choice == "BERT" and bert_model:
            try:
                # tokenize input
                inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = bert_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy()

                # show BERT predictions
                st.subheader("BERT Model Predictions")
                for idx, entity in enumerate(ENTITY_CLASSES):
                    st.write(f"**{entity}:** {probabilities[0][idx]:.2f}")
            except Exception as e:
                st.error(f"Error durante la predicción con BERT: {e}")
        else:
            st.error(f"Modelo {model_choice} no disponible.")
