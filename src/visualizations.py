import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import networkx as nx
import plotly.graph_objects as go
import numpy as np

def generate_wordcloud(predictions):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis"
    ).generate_from_frequencies(predictions)

    # Convert the wordcloud into an image for Streamlit
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    return fig

def create_interactive_graph(tokens, probabilities, edge_index, entity_classes):
    # Create the graph using NetworkX
    G = nx.Graph()
    for idx, token in enumerate(tokens):
        G.add_node(idx, label=token, entity=entity_classes[np.argmax(probabilities[idx])],
                   probability=np.max(probabilities[idx]))

    # Add edges based on edge_index
    for i, j in edge_index.T:
        G.add_edge(i, j)

    # Node positions
    pos = nx.spring_layout(G, seed=42)

    # Prepare data for Plotly
    node_x = []
    node_y = []
    node_labels = []
    node_colors = []
    node_sizes = []
    
    for node in G.nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(f"{G.nodes[node]['label']} ({G.nodes[node]['entity']} - {G.nodes[node]['probability']:.2f})")
        node_colors.append(G.nodes[node]['probability'])  # Use probabilities for color
        node_sizes.append(G.nodes[node]['probability'] * 20)  # Scale size by probability

    edge_x = []
    edge_y = []
    for edge in G.edges:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Create the Plotly figure
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_sizes,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title='Probability',
                xanchor='left',
                titleside='right'
            )
        ),
        text=node_labels)

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))
    return fig
