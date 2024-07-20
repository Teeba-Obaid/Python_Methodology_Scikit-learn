"""
This code loads the students graph, and help
to visualize an get important stats from the data.
Example to run:
>> python run.py -p ../data/
"""
from __future__ import division

import argparse
# from networkx.drawing.nx_agraph import graphviz_layout
import os
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_samples, silhouette_score

import src.util.lib as lib


class LevenshteinClustering:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to input graphs directory.")
    parser.add_argument("-o", "--output", help="Path to input graphs directory.")
    args = parser.parse_args()

    # def draw_graph(G):
    #     pos = nx.circular_layout(G)
    #     # nx.draw(G, pos=graphviz_layout(G))
    #     nx.draw(G, pos, font_size=8, with_labels = True)
    #     plt.show()

    path = os.getcwd()
    # get parent directory
    # PARENT_DIR = os.path.abspath(os.path.join(path, os.pardir))
    PARENT_DIR = path
    GRAPH_DIR = os.path.join('data', 'graphs')

    # general fields
    df = None
    gn_list = None
    data = None
    dimensions = None
    num_clusters = None
    insertion_dict = None
    deletion_dict = None
    substitution_dict = None
    k_interval = None
    graph_cluster_range = None

    def set_data(self):
        while True:
            try:
                data_file_name = input("enter the name of input csv data file (default is data): ")
                if data_file_name == '':
                    data_file_name = 'data'
                self.df = pd.read_csv(self.PARENT_DIR + '/data/' + data_file_name + '.csv')
                break
            except FileNotFoundError:
                print("Please enter valid data file name that is in the data directory ...")
                continue
        while True:
            graph_names = input(
                "enter index numbers (a,b,c,...) or an interval (a-b) or 'all' for predefined data (default all): ")
            if graph_names == 'all' or graph_names == '':
                graph_names = range(1, len(self.df.values) + 1)
                self.data = self.df
                self.gn_list = [*graph_names]
                break
            elif graph_names.count('-') == 1:
                try:
                    interval_arr = graph_names.split('-')
                    interval = list(map(int, interval_arr))
                    if len(interval) != 2:
                        print('invalid input, interval should be indicated with only one hyphen (-)')
                        continue
                    else:
                        first_index = interval[0]
                        end_index = interval[1]
                        if first_index > end_index:
                            print('invalid input, first index should be lower than second one!')
                            continue
                        if first_index <= 0:
                            print('invalid input, first index should be greater than zero!')
                            continue
                        if end_index > len(self.df.values):
                            print('invalid input, end index should be smaller than ' + str(len(self.df.values) + 1))
                            continue
                        self.gn_list = [*range(first_index, end_index + 1)]
                        self.data = self.df.iloc[first_index - 1: end_index]
                        break
                except ValueError:
                    print("invalid input, interval should be indicated with only one hyphen (-) between two integers")
                    continue
            elif ',' in graph_names:
                try:
                    self.gn_list = graph_names.split(',')
                    indices = np.array(list(map(int, self.gn_list))) - 1
                    self.data = self.df.iloc[indices]
                    break
                except ValueError:
                    print(
                        "invalid input, specific input data should only contains integers that are separated with comma")
                    continue
            else:
                print('invalid input')
                continue

        self.data = self.data.fillna('')
        # self.gn_list = [*graph_names]  # graph_names.split(',')
        if self.k_interval is None and len(self.data.values) > 1:
            self.k_interval = range(2, len(self.data.values))
        # self.gn_list = list(map(int, gn_list))

    def set_clustering_parameters(self):
        while True:
            try:
                dimensions = input("enter number of dimensions - n_component of Manifold-MDS (default 2): ")
                if dimensions == '':
                    dimensions = '2'
                self.dimensions = int(dimensions)
                break
            except ValueError:
                print("Please enter integer only ...")
                continue

        while True:
            try:
                num_clusters = input("enter number of clusters (default 2): ")
                if num_clusters == '':
                    num_clusters = '2'
                self.num_clusters = int(num_clusters)
                break
            except ValueError:
                print("Please input integer only...")
                continue
        while True:
            try:
                interval_str = input(
                    "enter k_interval, range of cluster numbers (a-b) for stress curves (default = wide): ")
                if interval_str.lower() == 'wide' or interval_str == '':
                    if self.data is None:
                        print('data is not specisied, you should choose an interval')
                        continue
                    else:
                        self.k_interval = range(2, len(self.data.values))
                        break
                else:
                    interval_arr = interval_str.split('-')
                    interval = list(map(int, interval_arr))
                    if len(interval) != 2:
                        print('invalid input, interval should be indicated with only one hyphen (-)')
                        continue
                    else:
                        first_index = interval[0]
                        end_index = interval[1]
                        if first_index > end_index:
                            print('invalid input, first index should be lower than second one!')
                            continue
                        if first_index <= 0:
                            print('invalid input, first index should be greater than zero!')
                            continue
                        self.k_interval = range(first_index, end_index + 1)
                        break
            except ValueError:
                print("Please input integer only...")
                continue

        while True:
            try:
                graph_cluster_range = input(
                    "enter n-cluster graphs, range of GSM cluster numbers (a-b) (default = (2-20)): ")
                if graph_cluster_range == '':
                    self.graph_cluster_range = range(2, 21)
                    break
                else:
                    interval_arr = graph_cluster_range.split('-')
                    interval = list(map(int, interval_arr))
                    if len(interval) != 2:
                        print('invalid input, interval should be indicated with only one hyphen (-)')
                        continue
                    else:
                        first_index = interval[0]
                        end_index = interval[1]
                        if first_index > end_index:
                            print('invalid input, first index should be lower than second one!')
                            continue
                        if first_index <= 0:
                            print('invalid input, first index should be greater than zero!')
                            continue
                        self.graph_cluster_range = range(first_index, end_index + 1)
                        break
            except ValueError:
                print("Please input integer only...")
                continue

    def set_weights(self):
        while True:
            try:
                weight_file_name = input("enter the name of weight excel file (default is weights): ")
                if weight_file_name == '':
                    weight_file_name = 'weights'
                weights_num = input("enter the number of weights parameters (default is 120): ")
                if weights_num == '':
                    weights_num = '120'
                print("please wait while reading weights ...")
                self.insertion_dict, self.deletion_dict, self.substitution_dict = lib.read_integrated_weights(
                    self.PARENT_DIR + '/data/' + weight_file_name + '.xlsx', int(weights_num))
                break
            except FileNotFoundError:
                print("File Not Found, enter the weight file name that is in the data directory")
                continue
            except ValueError:
                print("enter integer for number of weights")
                continue

        while True:
            manual_weights = input("Do you want to enter weights manually (y/n) (default is no)? ")
            if manual_weights.lower() not in ('y', 'n', 'yes', 'no', ''):
                print('enter yes/no or y/n')
                continue
            if manual_weights.lower() == 'y' or manual_weights.lower() == 'yes':
                try:
                    weights = input(
                        "enter 3 numbers for 'insertion', 'deletion', and 'substitution' weights separated by comma: ")
                    weights = [float(w.strip()) for w in weights.split(',')]
                    self.insertion_dict = dict.fromkeys(self.insertion_dict, weights[0])
                    self.deletion_dict = dict.fromkeys(self.deletion_dict, weights[1])
                    self.substitution_dict = dict.fromkeys(self.substitution_dict, weights[2])
                    break
                except ValueError:
                    print('enter 3 numbers separated by commas')
                    continue
            elif manual_weights.lower() == 'n' or manual_weights.lower() == 'no' or manual_weights == '':
                break

    def set_user_input(self):
        # get current directory
        # path = os.getcwd()
        # get parent directory
        # PARENT_DIR = os.path.abspath(os.path.join(path, os.pardir))
        # PARENT_DIR = path
        self.set_data()
        self.set_clustering_parameters()
        self.set_weights()

    def main(self):
        help_prompt = "These are the valid commands:\n"
        help_prompt += " q\tExit anytime.\n"
        help_prompt += " h\thelp.\n"
        help_prompt += " sa\tSet all inputs includes Data, Weights, and Clustering Parameters.\n"
        help_prompt += " sd\tSet only Data.\n"
        help_prompt += " sw\tSet only Weights.\n"
        help_prompt += " sp\tSet only Clustering Parameters.\n"
        help_prompt += " gs\tGraph stats.\n"
        help_prompt += " ld\tLevenshtein distance.\n"
        help_prompt += " ld_ovsa\tLevenshtein distance: one vs all\n"
        help_prompt += " wld\tWeighted Levenshtein distance.\n"
        help_prompt += " wld_ovsa\tWeighted Levenshtein distance: one vs all\n"
        help_prompt += " ld_edop\tLevenshtein edit operations.\n"
        help_prompt += " ld_edop_ovsa\tLevenshtein edit operations: one vs all\n"
        help_prompt += " lm\tLevenshtein median (approximate).\n"
        help_prompt += " lm_dev\tLevenshtein distance to the Levenshtein median.\n"
        help_prompt += " vis\tVisualize a 2-dimensional embedding.\n"
        help_prompt += " kmeans\tKMeans clustering.\n"
        help_prompt += " kmeans_vis\tKMeans clustering and colorful visualization.\n"
        help_prompt += " in\tIsolated nodes.\n"
        help_prompt += " mds\tDetermining the number of components in Manifold-MDS.\n"
        help_prompt += " sm\tSilhouette Method.\n"
        help_prompt += " gsm\tGraphical Silhouette Method.\n"
        help_prompt += " em\tElbow Method.\n"
        help_prompt += " cwl\tCustom Weighted Levenshtein.\n"
        print(help_prompt)

        while True:
            command = input("Enter your command: ")
            command = command.lower()
            if command not in (
                    'q', 'h', 'sa', 'sd', 'sw', 'sp', 'gs', 'ld', 'ld_ovsa', 'wld', 'wld_ovsa', 'ld_edop',
                    'ld_edop_ovsa', 'lm', 'lm_dev', 'vis', 'kmeans', 'kmeans_vis', 'in', 'mds', 'sm', 'gsm', 'em',
                    'cwl'):
                print("Enter a valid command")
                continue
            elif command == 'h':
                print(help_prompt)
            elif command == 'q':
                break
            elif (self.data is None or self.insertion_dict is None or self.num_clusters is None) and \
                    (command != 'sa' and command != 'sd' and command != 'sw' and command != 'sp'):
                print("you should first determine the input data values using 'sa, sd, sw, or sp' commands")
                continue
            elif command == 'sa':
                self.set_user_input()
            elif command == 'sd':
                self.set_data()
            elif command == 'sw':
                self.set_weights()
            elif command == 'sp':
                self.set_clustering_parameters()
            elif command == 'gs':
                graph_name = input("enter valid graph name: ")
                graph_file = os.path.join(self.GRAPH_DIR, graph_name, 'adj.txt')
                if not os.path.exists(graph_file):
                    print("FILE NOT FOUND!")
                    continue
                graph = lib.load_graph(graph_file)
                print("\n>> Graph Stats:\n{}\n".format(graph_name, lib.get_stat(graph)))
            elif command == 'ld':
                ord_list = self.data.values.tolist()
                dist_matrix = []
                dist_matrix.append(['*'] + self.gn_list)
                for i, ord_i in enumerate(ord_list):
                    row = []
                    row.append(self.gn_list[i])
                    for ord_j in ord_list:
                        i_j_dist = lib.get_custom_weighted_levenshtein_dist(ord_i, ord_j, self.insertion_dict,
                                                                            self.deletion_dict, self.substitution_dict)
                        row.append(i_j_dist)
                    dist_matrix.append(row)
                lib.print_df(pd.DataFrame(dist_matrix))
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'prints', 'ld_distance' + date_time)
                lib.save_print_as_text(dist_matrix, results_dir)
            elif command == 'ld_ovsa':
                print("comparing the first data with all others")
                ord_single = self.data.values.tolist()[0]
                ord_list = self.data.values.tolist()[1:]
                dist_matrix = []
                dist_matrix.append(['*', self.gn_list[0]])
                for i, ord_i in enumerate(ord_list):
                    row = []
                    row.append(self.gn_list[i + 1])
                    i_dist = lib.get_custom_weighted_levenshtein_dist(ord_i, ord_single, self.insertion_dict,
                                                                      self.deletion_dict, self.substitution_dict)
                    row.append(i_dist)

                    dist_matrix.append(row)
                lib.print_df(pd.DataFrame(dist_matrix))
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'prints', 'ld_ovsa_distance' + date_time)
                lib.save_print_as_text(dist_matrix, results_dir)
            elif command == 'wld':
                print("comparing all data with each other")
                # weights = input("enter the weights for 'insertion', 'deletion', and 'substitution' separated by comma: ")
                # weights = [float(w.strip()) for w in weights.split(',')]
                ord_list = self.data.values.tolist()
                dist_matrix = []
                dist_matrix.append(['*'] + self.gn_list)
                for i, ord_i in enumerate(ord_list):
                    row = []
                    row.append(self.gn_list[i])
                    for ord_j in ord_list:
                        i_j_dist = lib.get_custom_weighted_levenshtein_dist(ord_i, ord_j, self.insertion_dict,
                                                                            self.deletion_dict, self.substitution_dict)
                        row.append(i_j_dist)
                    dist_matrix.append(row)
                lib.print_df(pd.DataFrame(dist_matrix))
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'prints', 'wld_distance' + date_time)
                lib.save_print_as_text(dist_matrix, results_dir)
            elif command == 'wld_ovsa':
                print("comparing first data with all others")
                # weights = input("enter the weights for 'insertion', 'deletion', and 'substitution' separated by comma: ")
                # weights = [float(w.strip()) for w in weights.split(',')]
                ord_single = self.data.values.tolist()[0]
                ord_list = self.data.values.tolist()[1:]
                dist_matrix = []
                dist_matrix.append(['*', self.gn_list[0]])
                for i, ord_i in enumerate(ord_list):
                    row = []
                    row.append(self.gn_list[i + 1])
                    i_dist = lib.get_custom_weighted_levenshtein_dist(ord_i, ord_single, self.insertion_dict,
                                                                      self.deletion_dict, self.substitution_dict)
                    row.append(i_dist)
                    dist_matrix.append(row)
                lib.print_df(pd.DataFrame(dist_matrix))
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'prints', 'wld_ovsa_distance' + date_time)
                lib.save_print_as_text(dist_matrix, results_dir)
            elif command == 'ld_edop':
                ord_1 = self.data.values.tolist()[0]
                ord_2 = self.data.values.tolist()[1]
                edop = lib.get_edit_operations(ord_1, ord_2)
                print("\n>> Levenshtein Edit operations for two first data: {}\n".format(edop))
            elif command == 'ld_edop_ovsa':
                print("comparing first data with others")
                ord_list = self.data.values.tolist()[1:]
                ord_single = self.data.values.tolist()[0]
                for i, ord_i in enumerate(ord_list):
                    print('{}->{}: {}'.format(self.gn_list[i + 1], self.gn_list[0],
                                              lib.get_edit_operations(ord_i, ord_single)))
            elif command == 'lm':
                median = lib.get_levenshtein_median(self.data.values.tolist())
                print("\n>> Levenshtein Median:\n{}\n".format(','.join(median)))
            elif command == 'lm_dev':
                distances, edits = lib.get_weighted_levenshtein_median_deviation(self.data.values.tolist(),
                                                                                 self.gn_list, self.insertion_dict,
                                                                                 self.deletion_dict,
                                                                                 self.substitution_dict)
                print("\n>> Deviation From Levenshtein Median:")
                for n, d in distances:
                    print("** {}: {}".format(n, d))
                print("\n>> Edit Operations to Median:")
                for n, e in edits:
                    print("** {}: {}".format(n, e))
                print()
            elif command == 'in':
                graph_name = input("enter valid graph name: ")
                graph_names = graph_names.replace("'", "")
                graph_names = graph_names.replace(" ", "")
                graph_file = os.path.join(self.GRAPH_DIR, graph_name, 'adj.txt')
                nodes_file = os.path.join(self.GRAPH_DIR, graph_name, 'nodes.txt')
                if not (os.path.exists(graph_file) and os.path.join(nodes_file)):
                    print("FILE NOT FOUND!")
                    continue
                graph = lib.load_graph(graph_file)
                inference_nodes = lib.get_inference_nodes(nodes_file)
                isolated_nodes = lib.get_isolated_nodes(graph, inference_nodes)
                print(">> Isolated Nodes: {}\n".format(isolated_nodes))
            elif command == 'vis':
                distances = lib.get_levenshtein_dist_matrix3(self.data.values.tolist(), self.insertion_dict,
                                                             self.deletion_dict, self.substitution_dict)
                symmetric_distances = np.add(np.array(distances), np.array(distances).transpose()) / 2
                lib.visualize_embedding(self.gn_list, symmetric_distances)
            elif command == 'kmeans':
                distances = lib.get_levenshtein_dist_matrix3(self.data.values.tolist(), self.insertion_dict,
                                                             self.deletion_dict, self.substitution_dict)
                symmetric_distances = np.add(np.array(distances), np.array(distances).transpose()) / 2
                clusters = lib.kmeans_cluster(self.gn_list, symmetric_distances, n_clusters=self.num_clusters,
                                              dimensions=self.dimensions)
                print(">> Clusters: {}\n".format(clusters))
                accumulated_attrs = lib.attribute_accumulation(self.data.values.tolist())
                # Save Data as csv file:
                # lib.put_data_into_csv(raw_data)
                df = pd.DataFrame(dict(
                    r=accumulated_attrs.values(),
                    theta=accumulated_attrs.keys()))
                fig = px.line_polar(df, r='r', theta='theta', line_close=True)
                fig.update_traces(fill='toself')
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'kmeans_spider_radar', date_time)
                fig.write_image(results_dir + ".png", format='png')
                fig.show()
            elif command == 'kmeans_vis':
                distances = lib.get_levenshtein_dist_matrix3(self.data.values.tolist(), self.insertion_dict,
                                                             self.deletion_dict, self.substitution_dict)
                symmetric_distances = np.add(np.array(distances), np.array(distances).transpose()) / 2
                clusters = lib.kmeans_cluster(self.gn_list, symmetric_distances, n_clusters=self.num_clusters)
                colors = {gn: None for gn in self.gn_list}
                for i, c in enumerate(clusters):
                    for j in c:
                        colors[j] = i
                print('clusters: ', clusters)
                lib.visualize_embedding_color(self.gn_list, symmetric_distances, self.dimensions, colors)
            elif command == "mds":
                distances = lib.get_levenshtein_dist_matrix3(self.data.values.tolist(), self.insertion_dict,
                                                             self.deletion_dict, self.substitution_dict)
                symmetric_distances = np.add(np.array(distances), np.array(distances).transpose()) / 2
                # k_range = range(1, len(self.gn_list))
                stress = [MDS(dissimilarity='precomputed', n_components=k,
                              random_state=42, max_iter=300, eps=1e-9).fit(symmetric_distances).stress_ for k in
                          self.k_interval]
                print(stress)
                plt.plot(self.k_interval, stress, 'bx-')
                plt.xlabel("k")
                plt.ylabel("stress")
                plt.title('Determining the number of components of MDS')
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'mds', date_time)
                plt.savefig(results_dir)
                plt.show()
            elif command == 'sm':
                silhouette_avg = []
                distances = lib.get_levenshtein_dist_matrix3(self.data.values.tolist(), self.insertion_dict,
                                                             self.deletion_dict, self.substitution_dict)
                symmetric_distances = np.add(np.array(distances), np.array(distances).transpose()) / 2
                # max_k = len(symmetric_distances)
                # range_n_clusters = range(2, max_k)  # [2, 3, 4, 5]
                coords = lib.embed(names=self.gn_list, distances=symmetric_distances, n_components=self.dimensions)
                for num_clusters in self.k_interval:
                    # initialize kmeans
                    kmeans = KMeans(n_clusters=num_clusters)
                    kmeans.fit(coords)
                    cluster_labels = kmeans.labels_
                    # silhouette score
                    silhouette_avg.append(silhouette_score(coords, cluster_labels))
                plt.plot(self.k_interval, silhouette_avg, 'bx-')
                plt.xlabel('Values of K')
                plt.ylabel('Silhouette score')
                plt.title('Silhouette analysis For Optimal k')
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'sm', date_time)
                plt.savefig(results_dir)
                plt.show()
            elif command == "gsm":
                distances = lib.get_levenshtein_dist_matrix3(self.data.values.tolist(), self.insertion_dict,
                                                             self.deletion_dict, self.substitution_dict)
                symmetric_distances = np.add(np.array(distances), np.array(distances).transpose()) / 2
                # max_k = len(symmetric_distances)
                # range_n_clusters = range(2, min(max_k, 10))  # [2, 3, 4, 5]
                X = lib.embed(names=self.gn_list, distances=symmetric_distances, n_components=self.dimensions)
                # temp_range = self.k_interval
                # if self.k_interval.start == 1:
                #     temp_range = range(2, self.k_interval.stop)
                # if self.k_interval.stop > 20:
                #     temp_range = range(self.k_interval.start, 20)
                for n_clusters in self.graph_cluster_range:
                    # Create a subplot with 1 row and 2 columns
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    fig.set_size_inches(18, 7)
                    # The 1st subplot is the silhouette plot
                    # The silhouette coefficient can range from -1, 1 but in this example all
                    # lie within [-0.1, 1]
                    ax1.set_xlim([-0.1, 1])
                    # The (n_clusters+1)*10 is for inserting blank space between silhouette
                    # plots of individual clusters, to demarcate them clearly.
                    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

                    # Initialize the clusterer with n_clusters value and a random generator
                    # seed of 10 for reproducibility.
                    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                    cluster_labels = clusterer.fit_predict(X)

                    # The silhouette_score gives the average value for all the samples.
                    # This gives a perspective into the density and separation of the formed
                    # clusters
                    silhouette_avg = silhouette_score(X, cluster_labels)
                    print("For n_clusters =", n_clusters,
                          "The average silhouette_score is :", silhouette_avg)

                    # Compute the silhouette scores for each sample
                    sample_silhouette_values = silhouette_samples(X, cluster_labels)

                    y_lower = 10
                    for i in range(n_clusters):
                        # Aggregate the silhouette scores for samples belonging to
                        # cluster i, and sort them
                        ith_cluster_silhouette_values = \
                            sample_silhouette_values[cluster_labels == i]

                        ith_cluster_silhouette_values.sort()

                        size_cluster_i = ith_cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i

                        color = cm.nipy_spectral(float(i) / n_clusters)
                        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                          0, ith_cluster_silhouette_values,
                                          facecolor=color, edgecolor=color, alpha=0.7)

                        # Label the silhouette plots with their cluster numbers at the middle
                        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                        # Compute the new y_lower for next plot
                        y_lower = y_upper + 10  # 10 for the 0 samples

                    ax1.set_title("The silhouette plot for the various clusters.")
                    ax1.set_xlabel("The silhouette coefficient values")
                    ax1.set_ylabel("Cluster label")

                    # The vertical line for average silhouette score of all the values
                    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                    ax1.set_yticks([])  # Clear the yaxis labels / ticks
                    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                    # 2nd Plot showing the actual clusters formed
                    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                                c=colors, edgecolor='k')

                    # Labeling the clusters
                    centers = clusterer.cluster_centers_
                    # Draw white circles at cluster centers
                    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                                c="white", alpha=1, s=200, edgecolor='k')

                    for i, c in enumerate(centers):
                        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                    s=50, edgecolor='k')

                    ax2.set_title("The visualization of the clustered data.")
                    ax2.set_xlabel("Feature space for the 1st feature")
                    ax2.set_ylabel("Feature space for the 2nd feature")

                    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                                  "with n_clusters = %d" % n_clusters),
                                 fontsize=14, fontweight='bold')
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'gsm', date_time)
                plt.savefig(results_dir)
                plt.show()
            elif command == 'em':
                distortions = []
                inertias = []
                mapping1 = {}
                mapping2 = {}
                if self.data is None:
                    print("you should first determine the input data values using 'ds' command")
                    continue
                distances = lib.get_levenshtein_dist_matrix3(self.data.values.tolist(), self.insertion_dict,
                                                             self.deletion_dict, self.substitution_dict)
                symmetric_distances = np.add(np.array(distances), np.array(distances).transpose()) / 2
                # max_k = len(symmetric_distances)
                # range_n_clusters = range(2, max_k)
                coords = lib.embed(names=self.gn_list, distances=symmetric_distances, n_components=self.dimensions)
                for k in self.k_interval:
                    # Building and fitting the model
                    kmeanModel = KMeans(n_clusters=k).fit(coords)
                    kmeanModel.fit(coords)
                    distortions.append(sum(np.min(cdist(coords, kmeanModel.cluster_centers_,
                                                        'euclidean'), axis=1)) / coords.shape[0])
                    inertias.append(kmeanModel.inertia_)
                    mapping1[k] = sum(np.min(cdist(coords, kmeanModel.cluster_centers_,
                                                   'euclidean'), axis=1)) / coords.shape[0]
                    mapping2[k] = kmeanModel.inertia_
                # Tabulating and Visualizing the results
                # a) Using the different values of Distortion
                for key, val in mapping1.items():
                    print(f'{key} : {val}')
                plt.plot(self.k_interval, distortions, 'bx-')
                plt.xlabel('Values of K')
                plt.ylabel('Distortion')
                plt.title('The Elbow Method using Distortion')
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'em-distortion', date_time)
                plt.savefig(results_dir)
                plt.show()
                # b) Using the different values of Inertia:
                for key, val in mapping2.items():
                    print(f'{key} : {val}')
                plt.plot(self.k_interval, inertias, 'bx-')
                plt.xlabel('Values of K')
                plt.ylabel('Inertia')
                plt.title('The Elbow Method using Inertia')
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'em-inertia', date_time)
                plt.savefig(results_dir)
                plt.show()
            elif command == 'cwl':
                dist_matrix = []
                for i, ord_i in enumerate(self.data.values.tolist()):  # ord_list):
                    row = []
                    # row.append(gn_list[i])
                    for ord_j in self.data.values.tolist():
                        i_j_dist = lib.get_custom_weighted_levenshtein_dist(ord_i, ord_j, self.insertion_dict,
                                                                            self.deletion_dict, self.substitution_dict)
                        row.append(i_j_dist)
                    dist_matrix.append(row)
                print('distance matrix:')
                lib.print_df(pd.DataFrame(dist_matrix))
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'prints', 'cwl_orig_distance' + date_time)
                lib.save_print_as_text(dist_matrix, results_dir)
                symmetric_distances = np.add(np.array(dist_matrix), np.array(dist_matrix).transpose()) / 2
                print('symmetric distance matrix:')
                lib.print_df(pd.DataFrame(symmetric_distances))
                now = datetime.now()
                date_time = now.strftime("%Y%m%d-%H%M%S")
                results_dir = os.path.join(self.PARENT_DIR, 'results', 'prints', 'cwl_symmetric_distance' + date_time)
                lib.save_print_as_text(symmetric_distances, results_dir)
                clusters = lib.kmeans_cluster(self.gn_list, symmetric_distances, n_clusters=self.num_clusters,
                                              dimensions=self.dimensions)
                print(">> Clusters: {}\n".format(clusters))
            else:
                print("Enter valid command.")
                # print(help_prompt)


if __name__ == '__main__':
    lvn_clstr = LevenshteinClustering()
    lvn_clstr.main()
