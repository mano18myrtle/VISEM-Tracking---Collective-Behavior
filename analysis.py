import pandas as pd
import numpy as np
import re
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import os

# ========== Parameters ==========
batch_ids = [11, 12, 13, 14, 15, 19, 21, 22, 23, 24, 29, 30, 35, 36, 38, 47, 52, 54, 60, 82]
base_velocity_path = '/home/user/Mano/ml_env/Velocity'
base_interaction_path = '/home/user/Mano/ml_env/Interaction'

# ========== Batch Processing ==========
for batch_id in batch_ids:
    print(f"\n=== Processing Batch ID: {batch_id} ===")

    try:
        # File paths
        excel_file = f"{base_velocity_path}/{batch_id}/{batch_id}.xlsx"
        interaction_plot_path = f"{base_interaction_path}/{batch_id}/Interactions_Per_Frame.png"
        network_plot_path = f"{base_interaction_path}/{batch_id}/Sperm_Interaction_Network.png"
        interaction_csv_path = f"{base_interaction_path}/{batch_id}/Sperm_Interactions.csv"

        # Ensure interaction folder exists
        os.makedirs(f"{base_interaction_path}/{batch_id}", exist_ok=True)

        # ========== Step 1: Parse sheet names ==========
        xls = pd.ExcelFile(excel_file)
        sperm_data = []
        for sheet in xls.sheet_names:
            match = re.match(r"ID_(\d+)_(\w+)", sheet)
            if match:
                sperm_data.append({
                    'sperm_id': int(match.group(1)),
                    'class': match.group(2),
                    'sheet': sheet
                })

        print(f"Found {len(sperm_data)} sperm entries in the dataset.")

        # ========== Step 2: Compile Data ==========
        all_data = []
        for entry in sperm_data:
            df = pd.read_excel(excel_file, sheet_name=entry['sheet'])
            df['sperm_id'] = entry['sperm_id']
            df['class'] = entry['class']
            all_data.append(df)

        master_df = pd.concat(all_data, ignore_index=True)

        # Standardize frame column
        if 'Time' in master_df.columns and 'frame' not in master_df.columns:
            master_df.rename(columns={'Time': 'frame'}, inplace=True)

        print(f"Master Data Compiled. Columns: {master_df.columns.tolist()}")

        # ========== Step 3: Compute Interactions ==========
        def compute_interactions(df, distance_threshold=5):
            interaction_results = []
            for frame in df['frame'].unique():
                frame_data = df[df['frame'] == frame]
                if len(frame_data) < 2:
                    continue
                positions = frame_data[['X', 'Y']].values
                sperm_ids = frame_data['sperm_id'].values
                classes = frame_data['class'].values
                distance_matrix = cdist(positions, positions)

                # Record pairwise interactions within threshold
                for i in range(len(sperm_ids)):
                    for j in range(i + 1, len(sperm_ids)):
                        if distance_matrix[i, j] < distance_threshold:
                            interaction_results.append({
                                'frame': frame,
                                'sperm_A': sperm_ids[i],
                                'class_A': classes[i],
                                'sperm_B': sperm_ids[j],
                                'class_B': classes[j],
                                'distance': distance_matrix[i, j]
                            })
            return pd.DataFrame(interaction_results)

        interaction_df = compute_interactions(master_df)
        print(f"Total Interactions Found: {len(interaction_df)}")

        # ========== Step 4: Plot - Interactions per Frame ==========
        plt.figure(figsize=(12, 6))
        frame_counts = interaction_df['frame'].value_counts().sort_index()
        sns.lineplot(x=frame_counts.index, y=frame_counts.values, marker='o', color='darkblue')
        plt.xlabel('Frame')
        plt.ylabel('Number of Interactions')
        plt.title('Interactions per Frame')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(interaction_plot_path, dpi=300)
        plt.close()
        print(f"Interaction plot saved: {interaction_plot_path}")

        # ========== Step 5: Network Graph ==========
        G = nx.Graph()
        for _, row in interaction_df.iterrows():
            G.add_edge(row['sperm_A'], row['sperm_B'], weight=row['distance'])

        print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

        if G.number_of_edges() > 0:
            # Generate spring layout
            pos = nx.spring_layout(G, k=0.6, iterations=50, seed=42)

            # Node size based on degree
            node_sizes = [300 + 50 * G.degree(n) for n in G.nodes()]

            # Color mapping for nodes
            cmap_nodes = plt.get_cmap('tab20')
            node_colors = [cmap_nodes(i % cmap_nodes.N) for i, node in enumerate(G.nodes())]

            # Edge weight-based coloring
            weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
            edge_widths = 1 + 4 * (weights - weights.min()) / (np.ptp(weights) + 1e-5)

            fig, ax = plt.subplots(figsize=(14, 10))
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black', linewidths=0.7)
            nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

            # Draw edges with color gradient based on distance
            edges = nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color=weights,
                edge_cmap=plt.cm.Blues,
                width=edge_widths
            )

            ax.set_title(f'Sperm Interaction Network - {batch_id}', fontsize=16)
            ax.axis('off')

            # Colorbar for interaction distance
            sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(
                vmin=weights.min(),
                vmax=weights.max()
            ))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.01)
            cbar.set_label('Interaction Distance', rotation=270, labelpad=15)

            plt.tight_layout()
            plt.savefig(network_plot_path, dpi=300)
            plt.close()
            print(f"Network graph saved: {network_plot_path}")
        else:
            print("No interactions to plot in the network graph.")

        # ========== Step 6: Save Interaction Data ==========
        interaction_df.to_csv(interaction_csv_path, index=False)
        print(f"Interaction data saved: {interaction_csv_path}")

    except Exception as e:
        print(f"Error processing Batch ID {batch_id}: {e}")

