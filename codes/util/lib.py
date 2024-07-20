"""To keep the code for getting stats of graphs"""

import csv
import datetime
import math
import re
import string
import sys
from collections import Counter

import Levenshtein
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.cluster import KMeans
from weighted_levenshtein import lev


####################################################################
# Graph based methods
####################################################################


def get_stat(g):
    stats_report = "    {}\t{}\n".format("Number of nodes", g.number_of_nodes())
    stats_report += "    {}\t{}\n".format("Number of links", g.number_of_edges())
    return stats_report


def get_inference_nodes(graph_file):
    inference_nodes = []
    with open(graph_file, 'r') as f:
        line = f.readline()
        while line:
            info = line.strip().split()
            node_id = info[0]
            node_type = info[1]

            if node_type.lower() == "i":
                inference_nodes.append(node_id)
            line = f.readline()
    return inference_nodes


def get_isolated_nodes(graph, inference_nodes):
    isolated_nodes = []
    for n in graph.nodes():
        has_path = False
        for ni in inference_nodes:
            if nx.has_path(graph, n, ni):
                has_path = True
                break

        if not has_path:
            isolated_nodes.append(n)
    return isolated_nodes


####################################################################
# Sequence based methods
####################################################################

def get_levenshtein_median(gf_list):
    # g_rank_list = [get_nodes_ranks(gf) for gf in gf_list]
    # g_rank_list = [get_seq(gf) for gf in gf_list]

    coded_g_rank_list, char_node_mp = _convert_nodes_chars(gf_list)
    coded_median = Levenshtein.median(coded_g_rank_list)
    median = [char_node_mp[c] for c in coded_median]
    return median


def get_levenshtein_median_deviation(gf_list, gn_list):
    # g_rank_list = [get_nodes_ranks(gf) for gf in gf_list]
    # g_rank_list = [get_seq(gf) for gf in gf_list]

    median = get_levenshtein_median(gf_list)
    distances = [(gn_list[i], get_levenshtein_dist(median, gf_list[i])) for i in range(len(gn_list))]
    edits = [(gn_list[i], get_edit_operations(median, gf_list[i])) for i in range(len(gn_list))]
    return distances, edits


def get_weighted_levenshtein_median_deviation(data, gn_list, insertion_dict, deletion_dict, substitution_dict):
    # g_rank_list = [get_nodes_ranks(gf) for gf in gf_list]
    # g_rank_list = [get_seq(gf) for gf in gf_list]

    median = get_levenshtein_median(data)
    distances = [(gn_list[i], get_custom_weighted_levenshtein_dist(median, data[i], insertion_dict, deletion_dict,
                                                                   substitution_dict)) for i in range(len(gn_list))]
    edits = [(gn_list[i], get_edit_operations(median, data[i])) for i in range(len(gn_list))]
    return distances, edits


def get_levenshtein_dist(ord_1, ord_2):
    coded_g_rank_list, char_node_mp = _convert_nodes_chars([ord_1, ord_2])
    return float(Levenshtein.distance(coded_g_rank_list[0], coded_g_rank_list[1]))


def get_weighted_levenshtein_dist(ord_1, ord_2, insert_costs, delete_costs, substitute_costs):
    coded_g_rank_list, char_node_mp = _convert_nodes_chars([ord_1, ord_2])
    # insert_costs = weights[0] * np.ones(128, dtype=np.float64)
    # delete_costs = weights[1] * np.ones(128, dtype=np.float64)
    # substitute_costs = weights[2] * np.ones((128, 128), dtype=np.float64)
    return lev(coded_g_rank_list[0],
               coded_g_rank_list[1],
               insert_costs=insert_costs,
               delete_costs=delete_costs,
               substitute_costs=substitute_costs)


def get_custom_weighted_levenshtein_dist(ord_1, ord_2, insertion_dict, deletion_dict, substitution_dict):
    coded_g_rank_list, char_node_mp = _convert_nodes_chars([ord_1, ord_2])

    # coding_tuple_list, insert_costs, delete_costs, substitute_costs = set_static_custom_weight()
    # coded1 = [item[2] for item in coding_tuple_list for label in ord_1 if item[1] == label]
    # coded2 = [item[2] for item in coding_tuple_list for label in ord_2 if item[1] == label]

    insert_costs, delete_costs, substitute_costs = set_dynamic_custom_weight(char_node_mp, insertion_dict,
                                                                             deletion_dict, substitution_dict)
    if not check_symmetric(substitute_costs):
        print("Conflict")
    return lev(coded_g_rank_list[0],
               coded_g_rank_list[1],
               insert_costs=insert_costs,
               delete_costs=delete_costs,
               substitute_costs=substitute_costs)


def read_labeled_weights(path):
    # workbook = xlrd.open_workbook('weights.xlsx')
    workbook = pd.read_excel(path, engine='openpyxl', sheet_name=None)
    insertion_dict = {}
    deletion_dict = {}
    substitution_dict = {}
    labels = {'Label1': 7, 'Label2': 11, 'Label3': 9, 'Label4': 3, 'Label5': 12, 'Label6': 10, 'Label7': 9,
              'Label8': 27, 'Label9': 23, 'Label10': 9}

    for key, value in labels.items():
        worksheet = workbook.get(key)
        h_index = 0
        for h_item in range(value):
            h_index += 1
            v_index = 0
            horizontal_label = worksheet.values[h_item + 1, 0]
            insertion_dict[horizontal_label] = 1 if math.isnan(
                float(worksheet.values[h_item + 1, 1])) else worksheet.values[h_item + 1, 1]
            deletion_dict[horizontal_label] = 1 if math.isnan(
                float(worksheet.values[h_item + 1, 2])) else worksheet.values[h_item + 1, 2]
            for v_item in range(value):
                vertical_label = worksheet.values[0, v_item + 3]
                cell_value = worksheet.values[h_item + 1, v_item + 3]
                v_index += 1
                if h_index == v_index:
                    substitution_dict[(horizontal_label, vertical_label)] = 0
                elif h_index < v_index:
                    diagonal_symmetric_cell_value = worksheet.values[v_item + 1, h_item + 3]
                    substitution_dict[(horizontal_label, vertical_label)] = 1 \
                        if math.isnan(float(diagonal_symmetric_cell_value)) else diagonal_symmetric_cell_value
                else:
                    substitution_dict[(horizontal_label, vertical_label)] = 1 \
                        if math.isnan(float(cell_value)) else cell_value
    for k1, v1 in substitution_dict.items():
        rk1 = reverse_tuple(k1)
        for k2, v2 in substitution_dict.items():
            rk2 = reverse_tuple(k2)
            if k1 == k2 and v1 != v2:
                print("conflict in key " + k1 + " with different values: " + v1 + " and " + v2)
            if rk1 == k2 and v1 != v2:
                print("conflict in symmetric key " + k2 + " with different values: " + v1 + " and " + v2)

    return insertion_dict, deletion_dict, substitution_dict


def read_integrated_weights(path, weights_num):
    # workbook = xlrd.open_workbook('weights.xlsx')
    workbook = pd.read_excel(path, engine='openpyxl', sheet_name=None)
    insertion_dict = {}
    deletion_dict = {}
    substitution_dict = {}
    worksheet = workbook.get('Integrated')
    h_index = 0
    for h_item in range(weights_num):
            h_index += 1
            v_index = 0
            horizontal_label = worksheet.values[h_item + 3, 2]
            insertion_dict[horizontal_label] = 1 if math.isnan(
                float(worksheet.values[h_item + 3, 3])) else worksheet.values[h_item + 3, 3]
            deletion_dict[horizontal_label] = 1 if math.isnan(
                float(worksheet.values[h_item + 3, 4])) else worksheet.values[h_item + 3, 4]
            for v_item in range(weights_num):
                vertical_label = worksheet.values[2, v_item + 5]
                cell_value = worksheet.values[h_item + 3, v_item + 5]
                v_index += 1
                if h_index == v_index:
                    substitution_dict[(horizontal_label, vertical_label)] = 0
                elif h_index < v_index:
                    diagonal_symmetric_cell_value = worksheet.values[v_item + 3, h_item + 5]
                    substitution_dict[(horizontal_label, vertical_label)] = 1 \
                        if math.isnan(float(diagonal_symmetric_cell_value)) else diagonal_symmetric_cell_value
                else:
                    substitution_dict[(horizontal_label, vertical_label)] = 1 \
                        if math.isnan(float(cell_value)) else cell_value


    return insertion_dict, deletion_dict, substitution_dict


def reverse_tuple(tuples):
    new_tup = ()
    for k in reversed(tuples):
        new_tup = new_tup + (k,)
    return new_tup


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_symmetric2(matrix):
    A = np.array(matrix)
    i_index = 0
    j_index = 0
    for row in A:
        i_index += 1
        for column in A.T:
            j_index += 1
            comparison = row == column
            equal_arrays = comparison.all()
            if not equal_arrays:
                print("Conflict in row " + str(i_index) + " and column " + str(
                    j_index) + "\nNumber of Conflictions: " + str(len(row) - np.count_nonzero(comparison)))


def set_dynamic_custom_weight(char_node_mp, insertion_dict, deletion_dict, substitution_dict):
    insert_costs = np.ones(128, dtype=np.float64)
    delete_costs = np.ones(128, dtype=np.float64)
    substitute_costs = np.ones((128, 128), dtype=np.float64)

    for h_code, h_label in char_node_mp.items():
        if h_label == "":
            continue
        insert_costs[ord(h_code)] = insertion_dict[h_label]
        delete_costs[ord(h_code)] = deletion_dict[h_label]
        for v_code, v_label in char_node_mp.items():
            if v_label == "":
                continue
            if (h_label, v_label) not in substitution_dict:
                substitute_costs[ord(h_code), ord(v_code)] = 2
            else:
                substitute_costs[ord(h_code), ord(v_code)] = substitution_dict[(h_label, v_label)]
    return insert_costs, delete_costs, substitute_costs


def create_coding_tuple(weight_path, coded_label_path):
    workbook = pd.read_excel(weight_path, engine='openpyxl', sheet_name=None)
    insertion_dict = {}
    deletion_dict = {}
    substitution_dict = {}
    labels = {'Label1': 7, 'Label2': 11, 'Label3': 9, 'Label4': 3, 'Label5': 12, 'Label6': 10, 'Label7': 9,
              'Label8': 27, 'Label9': 23, 'Label10': 9}

    for key, value in labels.items():
        worksheet = workbook.get(key)
        for h_item in range(value):
            insertion_dict[worksheet.values[h_item + 1, 0]] = worksheet.values[h_item + 1, 1]
            deletion_dict[worksheet.values[h_item + 1, 0]] = worksheet.values[h_item + 1, 2]
            for v_item in range(value):
                substitution_dict[(worksheet.values[h_item + 1, 0], worksheet.values[0, v_item + 3])] = \
                    worksheet.values[
                        h_item + 1, v_item + 3]

    generate_coded_label()
    codedLabel_workbook = pd.read_excel(coded_label_path, engine='openpyxl', sheet_name=None)
    codedLabel_worksheet = codedLabel_workbook.get('Sheet1')
    coded_label = {}
    coding_tuple_list = []
    for row_index, row in codedLabel_worksheet.iterrows():
        index = codedLabel_worksheet.at[row_index, 'Index']
        label = codedLabel_worksheet.at[row_index, 'Label']
        code = codedLabel_worksheet.at[row_index, 'CodedLabel']
        coded_label[label] = codedLabel_worksheet.at[row_index, 'Index']
        my_tuple = (index, label, code)
        coding_tuple_list.append(my_tuple)
    return coding_tuple_list


def set_static_custom_weight(weight_path, coded_label_path):
    coding_tuple_list = create_coding_tuple(weight_path, coded_label_path)

    insertion_dict = {}
    deletion_dict = {}
    substitution_dict = {}

    insert_costs = np.ones(128, dtype=np.float64)
    delete_costs = np.ones(128, dtype=np.float64)
    substitute_costs = np.ones((128, 128), dtype=np.float64)

    for key, value in insertion_dict.items():
        my_index = [item[0] for item in coding_tuple_list if item[1] == key].pop()
        if math.isnan(insertion_dict.get(key)):
            insert_costs[my_index] = 1
        else:
            insert_costs[my_index] = insertion_dict.get(key)
    for key, value in deletion_dict.items():
        my_index = [item[0] for item in coding_tuple_list if item[1] == key].pop()
        if math.isnan(deletion_dict.get(key)):
            delete_costs[my_index] = 1
        else:
            delete_costs[my_index] = deletion_dict.get(key)
    for key_tuple, value in substitution_dict.items():
        i_index = [item[0] for item in coding_tuple_list if item[1] == key_tuple[0]].pop()
        j_index = [item[0] for item in coding_tuple_list if item[1] == key_tuple[1]].pop()
        if math.isnan(substitution_dict.get(key_tuple)):
            substitute_costs[i_index, j_index] = 1
        else:
            substitute_costs[i_index, j_index] = substitution_dict.get(key_tuple)

    return coding_tuple_list, insert_costs, delete_costs, substitute_costs


def generate_coded_label():
    workbook = pd.read_excel('weights.xlsx', engine='openpyxl', sheet_name=None)
    worksheet = workbook.get('CodedLabels')
    worksheet.fillna('', inplace=True)
    for i in range(0, 120):
        # worksheet.values[i, 3] = chr(worksheet.values[i, 0] + 32)
        # worksheet.at[i, 'CodedLabel'] = chr(int(worksheet.values[i, 0]) + 32)
        # print(bytes([i + 33]).decode('cp437'))
        # worksheet.at[i, 'CodedLabel'] = bytes([i + 33]).decode('cp437')
        worksheet.at[i, 'CodedLabel'] = chr(i + 33)
    with pd.ExcelWriter('CodedLabel.xlsx', engine='xlsxwriter', options={'strings_to_urls': False,
                                                                         'strings_to_formulas': False}) as writer:
        worksheet.to_excel(writer, sheet_name='Sheet1', index=False, encoding='utf-8')


# editops('spam', 'park')
def get_edit_operations(ord_1, ord_2):
    coded_g_rank_list, char_node_mp = _convert_nodes_chars([ord_1, ord_2])
    return Levenshtein.editops(coded_g_rank_list[0], coded_g_rank_list[1])


def get_levenshtein_dist_matrix(gf_list):
    num_graphs = len(gf_list)
    # g_rank_list = [get_nodes_ranks(gf) for gf in gf_list]
    g_rank_list = [get_seq(gf) for gf in gf_list]
    distances = [[get_levenshtein_dist(g_rank_list[i], g_rank_list[j]) for j in range(num_graphs)] for i in
                 range(num_graphs)]
    return distances


def get_levenshtein_dist_matrix2(data_list):
    num_graphs = len(data_list)
    # g_rank_list = [get_nodes_ranks(gf) for gf in gf_list]
    # g_rank_list = [get_seq(gf) for gf in data]
    distances = [[get_levenshtein_dist(data_list[i], data_list[j]) for j in range(num_graphs)] for i in
                 range(num_graphs)]
    return distances


def get_levenshtein_dist_matrix3(data_list, insertion_cost, deletion_cost, substitution_cost):
    num_graphs = len(data_list)
    # g_rank_list = [get_nodes_ranks(gf) for gf in gf_list]
    # g_rank_list = [get_seq(gf) for gf in data]
    distances = [[get_custom_weighted_levenshtein_dist(data_list[i], data_list[j], insertion_cost, deletion_cost,
                                            substitution_cost) for j in range(num_graphs)] for i in range(num_graphs)]
    return distances


def get_data_from_files(gf_list):
    num_graphs = len(gf_list)
    # g_rank_list = [get_nodes_ranks(gf) for gf in gf_list]
    g_rank_list = [get_seq(gf) for gf in gf_list]
    return g_rank_list


####################################################################
# Clustering
####################################################################
# Obained from here: https://baoilleach.blogspot.com/2014/01/convert-distance-matrix-to-2d.html

def embed(names, distances, n_components):
    adist = np.array(distances, dtype=float)
    amax = np.amax(adist)
    adist /= amax
    mds = manifold.MDS(n_components=n_components, dissimilarity="precomputed", random_state=6)
    results = mds.fit(adist)
    coords = results.embedding_
    return coords


def visualize_embedding(names, distances):
    coords = embed(names=names, distances=distances, n_components=2)
    plt.subplots_adjust(bottom=0.1)
    plt.scatter(coords[:, 0], coords[:, 1], marker='o')
    for label, x, y in zip(names, coords[:, 0], coords[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.show()


def visualize_embedding_color(names, distances, dimensions, colors):
    coords = embed(names=names, distances=distances, n_components=dimensions)
    col = [colors[n] for n in names]
    table = pd.DataFrame({'col': col, 'x': coords[:, 0], 'y': coords[:, 1]})
    print(table)
    cmap = plt.cm.get_cmap('jet', table.col.nunique())
    # cmap = plt.cm.get_cmap('nipy_spectral', table.col.nunique())
    # print('cmap: ', cmap)
    # print('cmap: ', cmap[0])
    # ax = table.plot.scatter(
    #     x='x',y='y', c='col',
    #     cmap=cmap
    # )
    fig, ax = plt.subplots()
    ax.scatter(coords[:, 0], coords[:, 1])
    for label, x, y, c in zip(names, coords[:, 0], coords[:, 1], col):
        print('>>>', label, c, table.values[c])
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(-20, 20),
            xycoords='data',
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc=cmap(c), alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    plt.show()


def kmeans_cluster(names, distances, n_clusters=2, dimensions=2):
    coords = embed(names=names, distances=distances, n_components=dimensions)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(coords)
    labels = kmeans.predict(coords)
    print(">> labels: ", labels)
    group_by = {l: set([]) for l in labels}
    for i in range(len(names)):
        group_by[labels[i]].add(names[i])
    return [list(s) for s in group_by.values()]


####################################################################
# Helper Functions
####################################################################
def get_nodes_ranks(graph_file):
    rank_node = []
    with open(graph_file, 'r') as f:
        line = f.readline()
        while line:
            info = line.strip().split()
            node_id = info[0]
            node_ranks = info[1]

            for r in node_ranks.split(','):
                rank_node.append((int(r), node_id))
            line = f.readline()
    rank_node.sort()
    return [n_id for r, n_id in rank_node]


def get_seq(graph_file):
    rank_node = []
    with open(graph_file, 'r') as f:
        line = f.readline()
        while line:
            rank_node.append(line.strip())
            line = f.readline()
    return rank_node


def _convert_nodes_chars(g_rank_list):
    nodes_list = sorted(set().union(*g_rank_list))
    n_c = {}
    c_n = {}
    for i in range(len(nodes_list)):
        n_c[nodes_list[i]] = string.printable[i]
        c_n[string.printable[i]] = nodes_list[i]
    coded_g_rank_list = [''.join([n_c[node] for node in l]) for l in g_rank_list]
    return coded_g_rank_list, c_n


def load_graph(graph_file):
    return nx.read_adjlist(graph_file, create_using=nx.DiGraph())


def print_df(df):
    # This code delete index and columns names
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    df_ = df.copy()
    df_.columns = ['' for _ in range(len(df_.columns))]
    df_.index = ['' for _ in range(len(df))]
    print(df_)


def save_print_as_text(df, file_name):
    a = np.array(df)
    mat = np.matrix(a)
    max_len = 5 # max(max(map(len, l)) for l in zip(mat))
    fmt = ('{:^%d}' % max_len).format('%s')

    with open(file_name+'.txt', 'wb') as f:
        for row in mat:
            np.savetxt(f, row, fmt=fmt, delimiter='\t')


def attribute_accumulation(matrix):
    accumulated_dict = {}
    for array in matrix:
        new_array = []
        for element in array:
            # temp_element = re.split(r'\t+', element)
            if element != '':
                new_array.append(element)
        unique, counts = np.unique(new_array, return_counts=True)
        temp_dict = dict(zip(unique, counts))
        a_counter = Counter(accumulated_dict)
        b_counter = Counter(temp_dict)
        accumulated_dict = dict(a_counter + b_counter)
    return accumulated_dict


def put_data_into_csv(matrix):
    df = pd.DataFrame(
        columns=['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10'])
    for array in matrix:
        new_array = []
        for element in array:
            temp_element = re.split(r'\t+', element)
            new_array.append(temp_element[0])
        df = df.append(pd.Series(new_array, index=df.columns[:len(new_array)]), ignore_index=True)
    today = datetime.datetime.now()
    file_name = 'data_' + today.strftime("%Y-%m-%d_%H-%M-%S") + '.csv'
    df.to_csv(file_name, index=False)
