# -*- coding: utf-8 -*-
"""
@author: Iurii Savvateev
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Used data, in case can not be retrieved from the csv file

# data = {
#     "Cluster_3": [287, 259, 299, 282, 472, 127, 251, 331, 271, 243, 285, 179, 9, 259, 267],
#     "Cluster_1": [337, 372, 446, 378, 500, 215, 296, 230, 325, 357, 372, 506, 428, 332, 386],
#     "Cluster_4": [6, 3, 3, 6, 16, 0, 6, 6, 0, 0, 0, 0, 0, 0, 0],
#     "Cluster_2": [653, 705, 751, 685, 837, 587, 561, 621, 590, 647, 646, 585, 685, 665, 677],
# }


# Import Data and convert to a pandas DataFrame
# Data  file is stored in the same directory
df = pd.read_csv("Fig3A_data_violin.csv", sep=";") 
data = df.to_dict(orient="list")
df = pd.DataFrame(data)

# Set the figure size to be incorporated in Fig 3A.
plt.figure(figsize=(10, 6))

# Create the violin plot
clusters = ["Cluster_1", "Cluster_2", "Cluster_3", "Cluster_4"]
values = [df[cluster].values for cluster in clusters]

violin_parts = plt.violinplot(values, showmeans=True, showmedians=True, showextrema=False)

# Add scatter points for individual data
x_positions = np.arange(1, len(clusters) + 1)

for x, cluster_values in zip(x_positions, values):
    plt.scatter([x] * len(cluster_values), cluster_values, color="black", alpha=0.7, zorder=2)

# Connect the dots from the same position across clusters with dashed lines

for datapoints in zip(*values):
    plt.plot(x_positions, datapoints, linestyle="--", color="gray", alpha=0.7, zorder=1)

# Customizing x-axis labels
plt.xticks(range(1, len(clusters) + 1), ["C1", "C2", "C3", "C4"],fontsize=35)
plt.yticks(fontsize=35)

# Adding title and labels
plt.title("Virus Expression", fontsize=50)
plt.xlabel("Clusters", fontsize=40)
plt.ylabel("Count (voxels)", fontsize=40)

# Remove the figure box
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["left"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)

# Adding grid lines for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()
