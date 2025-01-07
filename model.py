import numpy as np
import igraph as ig
from torch_geometric.nn import GATConv
from DataUtil import dataloader
from torch_geometric.utils import to_networkx
from edge_weight import EDGE_Weight
from opt import *
from torch import nn 
import torch
import math
from torch.nn import Linear, ReLU, Sequential, Dropout
import torch.nn.functional as F
from torch_geometric.utils import degree


opt = OptInit().initialize()

class MultiHeadNeuroAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, key_size, value_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_head_dim = key_size // num_heads
        self.k_head_dim = key_size // num_heads
        self.v_head_dim = value_size // num_heads

        self.W_q = nn.Linear(embed_dim, key_size)

        self.W_k = nn.Linear(embed_dim, key_size)
        self.W_v = nn.Linear(embed_dim, value_size)

        self.q_proj = nn.Linear(key_size, key_size)
        self.k_proj = nn.Linear(key_size, key_size)
        self.v_proj = nn.Linear(value_size, value_size)
        self.out_proj = nn.Linear(value_size, embed_dim)

    def forward(self, x, shortest_path):
        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        L, value_size = v.size()

        q = q.reshape(L, self.num_heads, self.q_head_dim).transpose(0, 1)
        k = k.reshape(L, self.num_heads, self.k_head_dim).transpose(0, 1)
        v = v.reshape(L, self.num_heads, self.v_head_dim).transpose(0, 1)

        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.size(-1))
        shortest_path = F.softmax(shortest_path, dim=-1) / 2
        shortest_path = shortest_path.unsqueeze(0).repeat(self.num_heads, 1, 1)
        att = F.softmax(att + shortest_path, dim=-1)
        output = torch.matmul(att.float(), v)
        output = output.transpose(0, 1).reshape(L, value_size)
        output = self.out_proj(output)

        return output

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate=0.3):
        super(Encoder, self).__init__()
        self.multihead_neurattn = MultiHeadNeuroAttention(embed_dim=input_dim, num_heads=num_heads, key_size=9, value_size=9)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, graph_feat, shortest_path):
        residual = graph_feat
        attn_output = self.multihead_neurattn(graph_feat, shortest_path)
        graph_features = residual + self.dropout(attn_output)
        graph_features = self.layer_norm1(graph_features)
        residual = graph_features
        graph_features = F.relu(self.linear2(F.relu(self.linear1(graph_features))))
        graph_features = residual + self.dropout(graph_features)
        graph_features = self.layer_norm2(graph_features)
        return graph_features

class TBS_Former(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, dropout_rate=0.3):
        super(TBS_Former, self).__init__()
        self.encoder1 = Encoder(input_dim, hidden_dim, num_heads, dropout_rate)
        self.encoder2 = Encoder(input_dim, hidden_dim, num_heads, dropout_rate)
        self.encoder3 = Encoder(input_dim, hidden_dim, num_heads, dropout_rate)
        self.linear3 = nn.Linear(input_dim, 8)
        self.stru_linear = nn.Linear(2, 1)
        self.stru_norm = nn.LayerNorm(2)
    def forward(self, graph):

        G = to_networkx(graph)
        i_g = ig.Graph.from_networkx(G)

        shortest_path = i_g.shortest_paths()
        shortest_path = np.array(shortest_path)
        diagonal_indices = np.diag_indices(148)
        shortest_path[diagonal_indices] += 1

        shortest_path = np.divide(1, shortest_path, out=np.zeros_like(shortest_path, dtype=np.float64),
                                  where=shortest_path != 0)
        shortest_path = torch.tensor(shortest_path, device=opt.device)

        pagerank_values = i_g.pagerank()
        pagerank_values = torch.tensor(pagerank_values, device=opt.device).view(-1, 1)

        Degree = degree(graph.edge_index[0], num_nodes=graph.num_nodes)
        total_degree = Degree.to(opt.device).view(-1, 1)

        structural_inf = torch.cat((total_degree, pagerank_values), dim=1).float()
        structural_inf = F.normalize(structural_inf, p=2, dim=0)
        Node_scores = F.relu(self.stru_linear(structural_inf))
        Node_scores = Node_scores.expand(-1, 9)
        graph_features = graph.x.to(opt.device) * Node_scores
        graph_features = graph_features.to(opt.device).float()
        graph_features = self.encoder1(graph_features, shortest_path)
        graph_features = self.encoder2(graph_features, shortest_path)
        graph_features = self.encoder3(graph_features, shortest_path)
        graph_features = F.relu(self.linear3(graph_features))
        graph_features = graph_features.view(1, -1)
        return graph_features



class MLP(torch.nn.Module):
    def __init__(self, input_n, output_n, num_layer=2, layer_list=[4, 8], dropout=0.5):
        """
        :param input_n: int 输入神经元个数
        :param output_n: int 输出神经元个数
        :param num_layer: int 隐藏层层数
        :param layer_list: list(int) 每层隐藏层神经元个数
        :param dropout: float
        """
        super(MLP, self).__init__()
        self.input_n = input_n
        self.output_n = output_n
        self.num_layer = num_layer
        self.layer_list = layer_list

        self.input_layer = Sequential(
            Linear(input_n, layer_list[0], bias=False),
            ReLU()
        )

        self.hidden_layer = Sequential()

        for index in range(num_layer-1):
            self.hidden_layer.append(Linear(layer_list[index], layer_list[index+1], bias=False))
            self.hidden_layer.append(ReLU())
        self.dropout = Dropout(dropout)

        # 输出层
        self.output_layer = Sequential(
            Linear(layer_list[-1], output_n, bias=False),
            ReLU(),
        )

    def forward(self, x):
        input = self.input_layer(x)
        hidden = self.hidden_layer(input)
        hidden = self.dropout(hidden)
        output = self.output_layer(hidden)
        return output


class MVGR_GNN(nn.Module):
    def __init__(self):
        super(MVGR_GNN, self).__init__()
        self.num_layers = 4
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(1190, 20, heads=1, dropout=0.3))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(GATConv(20, 20, heads=1, dropout=0.3))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(GATConv(20, 20, heads=1, dropout=0.3))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(GATConv(20, 20, heads=1, dropout=0.3))
        self.bns.append(nn.BatchNorm1d(20))
        self.out_fc = nn.Linear(20, 2)

        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_fc.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, features, edges, edges_weight):

        edges_weight = edges_weight.unsqueeze(1)
        x = features 
        layer_out = []
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[0](x, edges, edge_attr=edges_weight)
        x = self.bns[0](x)
        x = F.relu(x, inplace=True)

        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, edges, edge_attr=edges_weight)
        x = self.bns[1](x)
        x = F.relu(x, inplace=True)

        x = x + 0.7 * layer_out[0]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[2](x, edges, edge_attr=edges_weight)
        x = self.bns[2](x)
        x = F.relu(x, inplace=True)
        x = x + 0.7 * layer_out[1]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[3](x, edges, edge_attr=edges_weight)
        x = self.bns[3](x)
        x = F.relu(x, inplace=True)
        x = x + 0.7 * layer_out[2]
        layer_out.append(x)
        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]
        emb = sum(layer_out)
        x = self.out_fc(emb)

        return x
class BrainGT(torch.nn.Module):

    def __init__(self,nonimg, phonetic_score):
        super(BrainGT, self).__init__()

        self.nonimg = nonimg
        self.phonetic_score = phonetic_score
        self.weights = torch.nn.Parameter(torch.randn(3))
        self._setup()

    def _setup(self):
        self.edge_weight = EDGE_Weight(2, dropout=0.5)
        self.TBS_Former = TBS_Former(9, 9, 3)
        total_params = sum(p.numel() for p in self.TBS_Former.parameters() if p.requires_grad)
        print(f"TBS-Former Total number of trainable parameters: {total_params}")
        self.MVGR_GNN = MVGR_GNN()
        params = sum(p.numel() for p in self.MVGR_GNN.parameters() if p.requires_grad)
        print(f"MVGR-GNN Total number of trainable parameters: {params}")
        self.nonimg_mlp = MLP(2, 6)
    def forward(self, graphs):
        dl = dataloader()
        embeddings = []
        for graph in graphs:
            embedding = self.TBS_Former(graph)
            embeddings.append(embedding)
        embeddings = torch.cat(tuple(embeddings))
        non_embeddings = self.nonimg_mlp(torch.tensor(self.nonimg).to(opt.device))
        node_features = torch.cat((embeddings, non_embeddings), dim=1)

        edge_index, edge_input = dl.get_inputs(self.nonimg, embeddings, thershold=0.8)
        edge_index2, edge_input2 = dl.get_inputs(self.nonimg, embeddings, thershold=0.5)
        edge_index3, edge_input3 = dl.get_inputs(self.nonimg, embeddings, thershold=0.2)

        edge_input = (edge_input - edge_input.mean(axis=0)) / edge_input.std(axis=0)
        edge_input2 = (edge_input2 - edge_input2.mean(axis=0)) / edge_input2.std(axis=0)
        edge_input3 = (edge_input3 - edge_input3.mean(axis=0)) / edge_input3.std(axis=0)

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edge_index2 = torch.tensor(edge_index2, dtype=torch.long).to(opt.device)
        edge_index3 = torch.tensor(edge_index3, dtype=torch.long).to(opt.device)
        edge_input = torch.tensor(edge_input, dtype=torch.float32).to(opt.device)
        edge_input2 = torch.tensor(edge_input2, dtype=torch.float32).to(opt.device)
        edge_input3 = torch.tensor(edge_input3, dtype=torch.float32).to(opt.device)

        edge_weight = torch.squeeze(self.edge_weight(edge_input))
        edge_weight2 = torch.squeeze(self.edge_weight(edge_input2))
        edge_weight3 = torch.squeeze(self.edge_weight(edge_input3))

        predictions = self.MVGR_GNN(node_features, edge_index, edge_weight)
        predictions2 = self.MVGR_GNN(node_features, edge_index2, edge_weight2)
        predictions3 = self.MVGR_GNN(node_features, edge_index3, edge_weight3)
        weight = F.softmax(self.weights, dim=0)
        predictions = predictions * weight[0]
        predictions2 = predictions2 * weight[1]
        predictions3 = predictions3 * weight[2]
        pre = predictions + predictions2 + predictions3
        return pre