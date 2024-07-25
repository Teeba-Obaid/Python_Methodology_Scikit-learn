# Python Methodology Scikit-learn

This repository contains data, code, and visuals for several research projects focusing on educational simulations and genetics units.

## Projects and Data Description

### Genetics Unit Project
- **Data**: Contains sequential data for participants numbered from 1 to 248, each participating in a study using a genetics unit of instruction.
- **Visuals**: Visuals include k-means clusters identification and Silhouette Coefficient analysis for two experimental conditions: Revisit and Critique. Also features diagrams of the string analysis model used by the researcher.

### Simulation Explanatory Inferencing Project
- **Data**: Comprises 12 folders, each representing a participant engaged in a research study on physics simulations. The data is sequential.
- **Visuals**: Visualizations specific to this project are included.

### Simulation Rule Formation Project
- **Data**: This project explores a physics simulation combined with a rule formation tool. It includes `data8` and `data10` files. `data8` covers participants' engineering approaches, and `data10` details participants' search strategies. The repository includes a `codelable` file for sequential analysis using Levenshtein edit distance.
- **Visuals**: Features multidimensional scaling (MDS), k-means clusters, Silhouette Coefficient analysis for k-means, and k-means spider radar visuals for three experimental conditions: Control, Decision Table and Inductive Rules, and Inductive Rules alone. Also includes schematics of the coding schema and hierarchical structures of participants' engineering approaches.

## Visuals

The visuals directory contains comprehensive graphics and diagrams for each project, aiding in the interpretation and analysis of the data:

### Genetics Unit Project
- **K-means Clusters**: Identified k-means clusters for two experimental conditions: Revisit and Critique.
- **Silhouette Coefficient Analysis**: Analysis of k-means clusters using Silhouette Coefficients to evaluate the consistency within each cluster.
- **Sting Analysis Model**: Diagrams illustrating the sting analysis model used by the researcher, providing insights into the methodology.

### Simulation Explanatory Inferencing Project
- Visuals for this project help in understanding the sequential inferencing processes used in physics simulations.

### Simulation Rule Formation Project
- **Multidimensional Scaling (MDS)**: Visuals that showcase the MDS used to analyze participant data, reflecting various dimensions of data similarity.
- **K-means Clusters and Silhouette Coefficient Analysis**: Identifies k-means clusters and evaluates them using Silhouette Coefficient analysis for three experimental conditions: Control, Decision Table and Inductive Rules, and Inductive Rules alone.
- **K-means Spider Radar**: Graphical representations of k-means clustering results displayed in a radar chart to highlight the attributes of each cluster.
- **Coding Schema and Hierarchical Structure**: Schematics of the coding schema and the hierarchical structure of participants' engineering approaches, facilitating a deeper understanding of how rules and decisions were formulated during the simulations.

## Requirements

To run the scripts in this repository, you will need Python 3.x and the following libraries:

- pandas
- numpy
- scipy
- matplotlib
- scikit-learn

The median computed is approximate (finding median is a hard problem)

## Code
The repository includes a `run.py` script that facilitates the loading, visualization, and statistical analysis of the data:
```python
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster include KMeans
from sklearn.metrics include silhouette_samples, silhouette_score
from sklearn.manifold include MDS
import src.util.lib as lib

class LevenshteinClustering:
    # Code setup and methods for running the analyses
