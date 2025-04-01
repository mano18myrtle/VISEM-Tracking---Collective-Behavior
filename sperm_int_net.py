#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:03:31 2025

@author: Mano
"""

import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import os

# ========== Batch IDs ==========
batch_ids = [11, 12, 13, 14, 15, 19, 21, 22, 23, 24, 29, 30, 35, 36, 38, 47, 52, 54, 60, 82]
base_path = '/home/user/Mano/ml_env/Interaction'

# ========== Function to Process Each Batch ==========
def process_batch(batch_id):
    interaction_csv_path = f'{base_path}/{batch_id}/Sperm_Interactions.csv'
    output_html_path = f'{base_path}/{batch_id}/Sperm_Interaction_Network.html'
    
    if not os.path.exists(interaction_csv_path):
        print(f"Batch {batch_id}: CSV not found, skipping.")
        return
    
    # Load Interaction Data
    interaction_df = pd.read_csv(interaction_csv_path)
    print(f"Batch {batch_id}: Interactions Loaded - {len(interaction_df)}")

    # Create Network Graph
    G = nx.Graph()
    for _, row in interaction_df.iterrows():
        G.add_edge(
            f"Sperm_{row['sperm_A']}", 
            f"Sperm_{row['sperm_B']}", 
            weight=row['distance']
        )
    print(f"Batch {batch_id}: Graph Created | Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        print(f"Batch {batch_id}: No interactions, skipping plot.")
        return

    # Spring Layout
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Edge Trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Node Trace
    node_x, node_y, node_text = [], [], []
    node_degrees = [G.degree(n) for n in G.nodes()]
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f'{node}<br>Degree: {G.degree(node)}')

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=[n for n in G.nodes()],
        hovertext=node_text,
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            color=node_degrees,
            size=[5 + d * 3 for d in node_degrees],
            colorbar=dict(
                thickness=15,
                title='Sperm Interactions (Degree)',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    # Layout and Figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title={'text': f'Batch {batch_id} - Interactive Sperm Interaction Network', 'font': {'size': 20}},
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    # Save Interactive HTML
    fig.write_html(output_html_path)
    print(f"Batch {batch_id}: Interactive Network saved at {output_html_path}")

# ========== Run Batch Processing ==========
for batch_id in batch_ids:
    process_batch(batch_id)

print("\n All batches processed.")
