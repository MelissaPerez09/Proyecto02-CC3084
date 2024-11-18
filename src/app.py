from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from nltk.corpus import stopwords
import torch.nn.functional as F
from visualizations import *
import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import joblib
import torch
import nltk
import time
import os

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

# Paths to the models
gcn_model_path = "./models/gcn_model.pkl"
bert_model_path = "./models/bert_model.pkl"
svm_model_path = "./models/svm_model.pkl"
vectorizer_path = "./models/tfidf_vectorizer.pkl"

# Load and reconstruct the GCN model
@st.cache_resource
def load_gcn_model(model_path):
    try:
        gcn_state_dict = joblib.load(model_path)
        out_channels = len(gcn_state_dict["conv2.bias"])  # Infer output channels
        model = GCN(in_channels=128, hidden_channels=16, out_channels=out_channels)
        model.load_state_dict(gcn_state_dict)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading GCN model: {e}")
        return None

# Load and reconstruct the BERT model
@st.cache_resource
def load_bert_model(model_path):
    try:
        bert_state_dict = joblib.load(model_path)
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(ENTITY_CLASSES))
        model.load_state_dict(bert_state_dict)
        model.eval()  # Set to evaluation mode
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading BERT model: {e}")
        return None, None

# Load the SVM model and vectorizer
@st.cache_resource
def load_svm_model_and_vectorizer():
    try:
        svm_model = joblib.load(svm_model_path)
        vectorizer = joblib.load(vectorizer_path)
        return svm_model, vectorizer
    except Exception as e:
        st.error(f"Error loading SVM model or vectorizer: {e}")
        return None, None

# Load models
gcn_model = load_gcn_model(gcn_model_path)
bert_model, bert_tokenizer = load_bert_model(bert_model_path)
svm_model, vectorizer = load_svm_model_and_vectorizer()

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

# Vectorize the text for SVM using the pre-fitted vectorizer
def vectorize_text_for_svm(text):
    return vectorizer.transform([text])

# top 3 entities
def get_top_entities(predictions, entity_classes, top_n=3):
    top_indices = np.argsort(predictions)[-top_n:][::-1]
    top_entities = [(entity_classes[idx], predictions[idx]) for idx in top_indices]
    return top_entities

# Streamlit App
st.title("Biomedical Entity Identification with GCN, BERT, and SVM")
st.write("Enter a biomedical abstract to predict entities.")

# Model selection
model_choice = st.selectbox("Model:", options=["GCN", "BERT", "SVM", "Compare Models"])

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
                
                # Generate and display the interactive graph
                st.subheader("Interactive Graph of Tokens and Entities")
                graph_fig = create_interactive_graph(token_list, probabilities, graph_data.edge_index.numpy(), ENTITY_CLASSES)
                st.plotly_chart(graph_fig)
                
                # Add legend below the graph
                st.markdown("""
                ### How to Interpret the Graph:
                - **Nodes (points):** Represent the words (tokens) from the abstract.
                - **Size of the node:** Indicates the relevance of the token based on the highest probability assigned by the GCN model.
                - **Color of the node:** Represents the probability of association with a specific entity.
                    - **Yellow:** High probability of being relevant to an entity.
                    - **Green/Blue:** Low probability.
                - **Edges (lines):** Show contextual relationships between words, such as proximity in the text or connections learned by the model.
                - **Color Scale (right bar):** Maps the probability associated with the nodes:

                Use this graph to identify key tokens (large, yellow nodes) and analyze their relationships to validate or interpret the classification of biomedical entities.
                """)
                
                # Display predictions for each token
                df_predictions = pd.DataFrame(probabilities, columns=ENTITY_CLASSES, index=token_list)
                st.subheader("Detailed Predictions by Token")
                st.dataframe(df_predictions)
                
            except Exception as e:
                st.error(f"Error during prediction with GCN: {e}")

        # Perform prediction with BERT
        elif model_choice == "BERT" and bert_model:
            try:
                # Tokenize input
                inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = bert_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

                # Show BERT predictions (overall summary)
                bert_predictions = {ENTITY_CLASSES[idx]: prob for idx, prob in enumerate(probabilities)}
                st.subheader("BERT Model Predictions")
                for entity, prob in bert_predictions.items():
                    st.write(f"**{entity}:** {prob:.2f}")

                # Generate and display the word cloud
                st.subheader("Word Cloud")
                wordcloud_fig = generate_wordcloud(bert_predictions)
                st.pyplot(wordcloud_fig)

                # Token-level probabilities
                token_probabilities = []
                for token in tokens:
                    token_inputs = bert_tokenizer(token, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    token_outputs = bert_model(**token_inputs)
                    token_probs = torch.softmax(token_outputs.logits, dim=1).detach().numpy()[0]
                    token_probabilities.append(token_probs)

                # Create a DataFrame for detailed token predictions
                df_bert = pd.DataFrame(token_probabilities, columns=ENTITY_CLASSES, index=tokens)
                st.subheader("Detailed Predictions by Token")
                st.dataframe(df_bert)

            except Exception as e:
                st.error(f"Error during prediction with BERT: {e}")

        # Perform prediction with SVM
        elif model_choice == "SVM" and svm_model and vectorizer:
            try:
                # Vectorize input and predict
                svm_vectorized_input = vectorize_text_for_svm(user_input)
                svm_prediction_prob = svm_model.predict_proba(svm_vectorized_input)
                prediction_summary = {entity: svm_prediction_prob[0][i] for i, entity in enumerate(ENTITY_CLASSES)}

                # Display prediction summary
                st.subheader("SVM Model Predictions (Summary)")
                for entity, prob in prediction_summary.items():
                    st.write(f"**{entity}:** {prob:.2f}")

                # Bar Chart
                st.subheader("SVM Prediction Probabilities (Bar Chart)")
                bar_chart_fig = go.Figure([go.Bar(x=list(prediction_summary.keys()), y=list(prediction_summary.values()))])
                bar_chart_fig.update_layout(title="Probabilities for Each Entity", xaxis_title="Entities", yaxis_title="Probability")
                st.plotly_chart(bar_chart_fig)

                # Token-level probabilities
                svm_probabilities = svm_model.predict_proba(vectorizer.transform(tokens))
                df_svm = pd.DataFrame(svm_probabilities, columns=ENTITY_CLASSES, index=tokens)
                st.subheader("Detailed Predictions by Token")
                st.dataframe(df_svm)
            except Exception as e:
                st.error(f"Error during prediction with SVM: {e}")
        
        # compare models
        elif model_choice == "Compare Models":
            metrics = pd.DataFrame({
                "Model": ["SVM", "GCN", "BERT"],
                "F1 Score (Micro)": [0.901716, 0.627119, 0.841060],
                "Sequence Label Accuracy": [0.3875, 0.0125, 0.2250],
                "ROC-AUC": [0.841907, 0.695623, 0.791120]
            })
            
            st.subheader("Model Comparison")
            
            # metrics table
            st.subheader("Performance Metrics")
            st.dataframe(metrics)
            
            # bar chart metrics comparison
            st.subheader("Metric Comparison")
            plot_model_metrics(metrics)

            # classify abstract
            if user_input.strip():
                tokens = preprocess_text(user_input)
                
                # GCN predictions
                if gcn_model:
                    graph_data, token_list = create_graph_from_input(tokens)
                    gcn_predictions = torch.sigmoid(gcn_model(graph_data.x, graph_data.edge_index)).detach().numpy().mean(axis=0)
                    gcn_top_entities = get_top_entities(gcn_predictions, ENTITY_CLASSES)
                else:
                    gcn_top_entities = None

                # BERT predictions
                if bert_model:
                    inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
                    bert_outputs = bert_model(**inputs)
                    bert_predictions = torch.softmax(bert_outputs.logits, dim=1).detach().numpy()[0]
                    bert_top_entities = get_top_entities(bert_predictions, ENTITY_CLASSES)
                else:
                    bert_top_entities = None

                # SVM predictions
                if svm_model and vectorizer:
                    svm_vectorized_input = vectorize_text_for_svm(user_input)
                    svm_predictions = svm_model.predict_proba(svm_vectorized_input).flatten()
                    svm_top_entities = get_top_entities(svm_predictions, ENTITY_CLASSES)
                else:
                    svm_top_entities = None

                # show top 3 entities predicted by each model
                st.subheader("Top 3 Entities Predicted by Each Model")
                result_data = {
                    "Model": ["GCN", "BERT", "SVM"],
                    "Top 3 Entities": [
                        ", ".join([f"{entity} ({prob:.2f})" for entity, prob in gcn_top_entities]) if gcn_top_entities else "N/A",
                        ", ".join([f"{entity} ({prob:.2f})" for entity, prob in bert_top_entities]) if bert_top_entities else "N/A",
                        ", ".join([f"{entity} ({prob:.2f})" for entity, prob in svm_top_entities]) if svm_top_entities else "N/A"
                    ]
                }
                st.table(result_data)
                        
        else:
            st.error(f"Model {model_choice} not available.")
