import numpy as np
from sklearn.model_selection import StratifiedKFold
from BrainNetwork import get_all_networks
import csv
import os
from scipy.spatial import distance


class dataloader():
    def __init__(self): 
        self.pd_dict = {}

    def load_data(self):
        subject_IDs = get_ids()
        labels = get_subject_score(subject_IDs, score='DX_GROUP')
        num_nodes = len(subject_IDs)
        ages = get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        genders = get_subject_score(subject_IDs, score='SEX')
        y = np.zeros([num_nodes], dtype=int)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=int)
        for i in range(num_nodes):
            y[i] = int(labels[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]
        self.y = y 
        self.all_graphs = get_all_networks()
        phonetic_data = np.zeros([num_nodes, 2], dtype=np.float32)
        phonetic_data[:,0] = gender 
        phonetic_data[:,1] = age 
        self.pd_dict['SEX'] = np.copy(phonetic_data[:,0])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(phonetic_data[:,1])
        phonetic_score = self.pd_dict
        return self.all_graphs, self.y, phonetic_data, phonetic_score
    

    def data_split(self, n_folds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_splits = list(skf.split(self.all_graphs, self.y))
        return cv_splits 


    def get_inputs(self, nonimg, embeddings, thershold):
        self.node_ftr  = np.array(embeddings.detach().cpu().numpy())
        n = self.node_ftr.shape[0]
        num_edge = n*(1+n)//2 - n
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        aff_adj = get_static_affinity_adj(self.node_ftr)
        flatten_ind = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:,flatten_ind] = [i,j]
                edgenet_input[flatten_ind]  = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind +=1

        assert flatten_ind == num_edge, "Error in computing edge input"

        sorted_indices = np.argsort(aff_score)[::-1]
        num_edges_to_keep = int(num_edge * thershold)
        keep_ind = sorted_indices[:num_edges_to_keep]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input

def get_subject_score(subject_list, score):
    scores_dict = {}
    phenotype = "Phenotypic_V1_0b_preprocessed1.csv"
    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['FILE_ID'] in subject_list:
                scores_dict[row['FILE_ID']] = row[score]
    return scores_dict

def get_ids(num_subjects=None):

    subject_IDs = np.genfromtxt(os.path.join("Subjects_ID.txt"), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs

def get_static_affinity_adj(features):
    distv = distance.pdist(features, metric='correlation') 
    dist = distance.squareform(distv)  
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = feature_sim
    return adj
