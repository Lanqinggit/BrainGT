import os
import argparse
import scipy.io as sio
import torch
from torch_geometric.data import Data
import numpy as np
from networkx.convert_matrix import from_numpy_array
import networkx as nx
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
import DataUtil
from sklearn.metrics.pairwise import cosine_similarity


data_folder = "FC_R"


def read_sigle_data(matrix, sub_x):
    pcorr = np.abs(matrix)
    pcorr[pcorr < 0.5] = 0
    num_nodes = pcorr.shape[0]
    G = from_numpy_array(pcorr)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)
    att = sub_x
    att[att == float('inf')] = 0
    att_torch = torch.from_numpy(att).float()
    graph = Data(x=att_torch, edge_index=edge_index.long(), edge_attr=edge_att)
    return graph


def get_networks(subject_list, variable='ROICorrelation'):
    x_list = []
    structural_roi_feature = np.load("structural_roi_feature.npy")
    function_roi_feature = np.load("function_roi_feature.npy")
    allSub_X = np.concatenate((structural_roi_feature, function_roi_feature), axis=2)
    for i in range(allSub_X.shape[0]):
        sub_array = allSub_X[i, :, :]
        x_list.append(sub_array)
    graphs = []
    for sub_x, subject in zip(x_list, subject_list):
        standardized_data = (sub_x - np.mean(sub_x, axis=0)) / np.std(sub_x, axis=0)
        fl = os.path.join(data_folder,
                           "ROICorrelation_" + subject + ".mat")
        fc_matrix = sio.loadmat(fl)[variable]
        stru_data = standardized_data[:, :5]
        ss_matrix = cosine_similarity(stru_data)
        mc_matrix = (fc_matrix+ss_matrix)/2
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_mc_matrix = np.arctanh(mc_matrix)
        graph = read_sigle_data(norm_mc_matrix, standardized_data)
        graphs.append(graph)
    return graphs


def get_all_networks():
    
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset')
    parser.add_argument('--atlas', default='Destrieux')
    parser.add_argument('--seed', default=42, type=int, help='Seed for random initialisation. default: 42.')
    parser.add_argument('--nclass', default=2, type=int, help='Number of classes. default:2')
    args = parser.parse_args()
    params = dict()
    params['seed'] = args.seed  
    params['atlas'] = args.atlas
    subject_IDs = DataUtil.get_ids()
    num_subjects = len(subject_IDs)
    params['n_subjects'] = num_subjects
    all_graphs = get_networks(subject_IDs)
    return all_graphs





