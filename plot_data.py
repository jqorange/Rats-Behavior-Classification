import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler



pca = PCA(n_components=3)
data_pca = pca.fit_transform(data)[:]
data_final = data_pca



# 获取信道名
channel_names = jump_labels_df.columns[1:]
channel_names_new = channel_names.append(pd.Index(["NAN"]))
colors = plt.cm.get_cmap('tab10', len(channel_names_new))
# Define a custom color list (for example 5 unique colors for 5 categories)
custom_colors = [
    'rgb(255, 0, 0)',      # Red
    'rgb(0, 255, 0)',      # Green
    'rgb(0, 0, 255)',      # Blue
    'rgb(255, 255, 0)',    # Yellow
    'rgb(0, 255, 255)',    # Cyan
    'rgb(255, 0, 255)',    # Magenta
    'rgb(128, 128, 128)',  # Grey for NAN
    'rgb(255, 165, 0)',    # Orange
    'rgb(75, 0, 130)',     # Indigo
    'rgb(255, 105, 180)',  # Hot Pink
    'rgb(34, 139, 34)',    # Forest Green
    'rgb(255, 20, 147)',   # Deep Pink
    'rgb(238, 130, 238)',  # Violet
    'rgb(255, 69, 0)',     # Red-Orange
    'rgb(60, 179, 113)',   # Medium Sea Green
    'rgb(186, 85, 211)',   # Medium Orchid
    'rgb(128, 0, 128)',    # Purple
    'rgb(0, 191, 255)',    # Deep Sky Blue
    'rgb(139, 69, 19)',    # Saddle Brown
    'rgb(255, 222, 173)'   # Navajo White
]

# Create a dictionary mapping channel names to custom colors
color_map = {name: custom_colors[i % len(custom_colors)] for i, name in enumerate(channel_names_new)}


# 创建一个字典来存储每个类别的数据点
scatter_data = {name: ([], [], []) for name in channel_names_new}

for i in tqdm(range(len(data_final))):
    NAN = 0
    for name in channel_names:
        if jump_labels_df[name][i] == 1:
            scatter_data[name][0].append(data_final[i, 0])
            scatter_data[name][1].append(data_final[i, 1])
            scatter_data[name][2].append(data_final[i, 2])  # Add the third component
            NAN = 1
    if NAN == 0:
        scatter_data["NAN"][0].append(data_final[i, 0])
        scatter_data["NAN"][1].append(data_final[i, 1])
        scatter_data["NAN"][2].append(data_final[i, 2])  # Add the third component

# Create the Plotly figure
fig = go.Figure()

# Plot the NAN category points
fig.add_trace(go.Scatter3d(
    x=scatter_data["NAN"][0],
    y=scatter_data["NAN"][1],
    z=scatter_data["NAN"][2],
    mode='markers',
    marker=dict(size=2, color="grey"),
    name="NAN"
))

# Plot the other categories
for name in channel_names:
    if len(scatter_data[name][0])!=0:

        fig.add_trace(go.Scatter3d(
            x=scatter_data[name][0],
            y=scatter_data[name][1],
            z=scatter_data[name][2],
            mode='markers',
            marker=dict(size=2, color=color_map[name]),
            name=name
        ))

# Set labels for axes
fig.update_layout(
    scene=dict(
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        zaxis_title='PCA Component 3'
    ),
    title="3D PCA of Data with Labels",
    legend=dict(x=1, y=1)
)

# Show the plot
fig.show()