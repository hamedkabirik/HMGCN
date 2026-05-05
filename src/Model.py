import numpy
import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
import numpy as np
from src.Decoupling_matrix_aggregation import adj_matrix_weight_merge, adj_matrix_merge
from src.Decoupling_matrix_aggregation import coototensor
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
#from torch_geometric.nn.pool.topk_pool import TopKPooling,filter_adj
from torch_geometric.nn.pool import TopKPooling
from torch.nn import Parameter
from torch_geometric.nn.pool import SAGPooling




class Net2(torch.nn.Module):
    def __init__(self, nfeat, nhid, out, attn_vec_dim, pooling_ratio, dropout_ratio):
        super(Net2, self).__init__()
        #self.args = args
        self.nfeat = nfeat
        self.nhid = nhid
        self.out = out
        self.pooling_ratio = pooling_ratio
        self.dropout_ratio = dropout_ratio
        self.attn_vec_dim = attn_vec_dim

        self.conv1 = GCNConv(self.nfeat, self.out, add_self_loops=False, normalize=False)


    def forward(self, feature, edge_index):


        x = self.conv1(feature, edge_index)

        h = x.mean(dim=0)

        return x, h






class LHGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, pooling_ratio, dropout_ratio, attn_vec_dim):
        super(LHGCN, self).__init__()

        self.dropout_ratio = dropout_ratio
        self.nfeat = nfeat
        self.nhid = nhid
        self.out = out
        self.pooling_ratio = pooling_ratio
        self.attn_vec_dim = attn_vec_dim
        self.sig = torch.nn.Sigmoid()

        self.net1 = Net2(self.nfeat, self.nhid, self.out, self.attn_vec_dim, self.pooling_ratio, self.dropout_ratio)
        self.net2 = Net2(self.nfeat, self.nhid, self.out, self.attn_vec_dim, self.pooling_ratio, self.dropout_ratio)
        self.net3 = Net2(self.nfeat, self.nhid, self.out, self.attn_vec_dim, self.pooling_ratio, self.dropout_ratio)
        self.net4 = Net2(self.nfeat, self.nhid, self.out, self.attn_vec_dim, self.pooling_ratio, self.dropout_ratio)

        self.fc0 = nn.Linear(self.attn_vec_dim, 1, bias=False)
        self.fcfc0 = nn.Linear(self.out, self.attn_vec_dim, bias=True)
        self.fc1 = nn.Linear(self.attn_vec_dim, 1, bias=False)
        self.fcfc1 = nn.Linear(self.out, self.attn_vec_dim, bias=True)



    def forward(self, feature, A, node_types, use_relu=True):


        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        feature = feature.to(device='cuda')
        feature = feature.float()


        embs0 = []
        embs1 = []
        beta0 = []
        beta1 = []

        a = A[0][0]
        a = a.tocoo()

        edge_index = torch.tensor(numpy.array([a.row, a.col]))
        try:
            edge_index = torch.tensor(edge_index.toarray())
        except:
            try:
                edge_index = torch.from_numpy(edge_index.toarray())
            except:
                pass

        edge_index = edge_index.to(device='cuda')
        edge_index =edge_index.long()


        b = A[1][0]
        b = b.tocoo()
        edge_index1 = torch.tensor(numpy.array([b.row, b.col]))
        try:
            edge_index1 = torch.tensor(edge_index1.toarray())
        except:
            try:
                edge_index1 = torch.from_numpy(edge_index1.toarray())
            except:
                pass

        edge_index1 = edge_index1.to(device='cuda')
        edge_index1 =edge_index1.long()

        c = A[2][0]
        c = c.tocoo()

        edge_index2 = torch.tensor(numpy.array([c.row, c.col]))
        try:
            edge_index2 = torch.tensor(edge_index2.toarray())
        except:
            try:
                edge_index2 = torch.from_numpy(edge_index2.toarray())
            except:
                pass

        edge_index2 = edge_index2.to(device='cuda')
        edge_index2 = edge_index2.long()


        d = A[3][0]
        d = d.tocoo()

        edge_index3 = torch.tensor(numpy.array([d.row, d.col]))
        try:
            edge_index3 = torch.tensor(edge_index3.toarray())
        except:
            try:
                edge_index3 = torch.from_numpy(edge_index3.toarray())
            except:
                pass

        edge_index3 = edge_index3.to(device='cuda')
        edge_index3 =edge_index3.long()


        node_indices0 = np.where(node_types == 0)[0]
        node_indices1 = np.where(node_types == 1)[0]

        embeds, x = self.net1(feature, edge_index)

        dd0 = embeds[node_indices0, :]
        embs0.append(dd0)


        dd1 = embeds[node_indices1, :]
        embs1.append(dd1)


        dd0 = self.fcfc0(dd0)
        dd0 = F.tanh(dd0)
        x = dd0.mean(dim=0)


        x = self.fc0(x)
        beta0.append(x)


        dd1 = self.fcfc1(dd1)
        dd1 = F.tanh(dd1)
        x = dd1.mean(dim=0)

        x = self.fc1(x)
        beta1.append(x)

        embeds, x = self.net2(feature, edge_index1)

        dd0 = embeds[node_indices0, :]
        embs0.append(dd0)


        dd1 = embeds[node_indices1, :]
        embs1.append(dd1)


        dd0 = self.fcfc0(dd0)
        dd0 = F.tanh(dd0)
        x = dd0.mean(dim=0)

        x = self.fc0(x)
        beta0.append(x)


        dd1 = self.fcfc1(dd1)
        dd1 = F.tanh(dd1)
        x = dd1.mean(dim=0)

        x = self.fc1(x)
        beta1.append(x)

        embeds, x = self.net3(feature, edge_index2)

        dd0 = embeds[node_indices0, :]
        embs0.append(dd0)

        dd1 = embeds[node_indices1, :]
        embs1.append(dd1)


        dd0 = self.fcfc0(dd0)
        dd0 = F.tanh(dd0)
        x = dd0.mean(dim=0)

        x = self.fc0(x)
        beta0.append(x)


        dd1 = self.fcfc1(dd1)
        dd1 = F.tanh(dd1)
        x = dd1.mean(dim=0)

        x = self.fc1(x)
        beta1.append(x)


        embeds, x = self.net4(feature, edge_index3)

        dd0 = embeds[node_indices0, :]
        embs0.append(dd0)


        dd1 = embeds[node_indices1, :]
        embs1.append(dd1)


        dd0 = self.fcfc0(dd0)
        dd0 = F.tanh(dd0)
        x = dd0.mean(dim=0)

        x = self.fc0(x)
        beta0.append(x)


        dd1 = self.fcfc1(dd1)
        dd1 = F.tanh(dd1)
        x = dd1.mean(dim=0)

        x = self.fc1(x)
        beta1.append(x)

        beta0 = torch.cat(beta0, dim=0)
        beta0 = F.softmax(beta0, dim=0)

        beta1 = torch.cat(beta1, dim=0)
        beta1 = F.softmax(beta1, dim=0)

        beta0 = beta0.tolist()
        beta1 = beta1.tolist()


        multiplied = []
        for s in range(0, len(embs0)):
            multiplied.append(beta0[s] * embs0[s])

        h0 = sum(multiplied)

        multiplied = []
        for s in range(0, len(embs1)):
            multiplied.append(beta1[s] * embs1[s])

        h1 = sum(multiplied)
        h = torch.cat((h0, h1), 0)


        final_emb = h

        return final_emb





class MLHGCN(nn.Module):
    def __init__(self, nfeat, nhid, out, pooling_ratio, dropout_ratio, attn_vec_dim):
        super(MLHGCN, self).__init__()

        self.dropout_ratio = dropout_ratio
        self.nfeat = nfeat
        self.nhid = nhid
        self.out = out
        self.pooling_ratio = pooling_ratio
        self.attn_vec_dim = attn_vec_dim
        self.sig = torch.nn.Sigmoid()

        self.lhgcn1 = LHGCN(self.nfeat, self.nhid, self.out, self.pooling_ratio, self.dropout_ratio, self.attn_vec_dim)
        self.lhgcn2 = LHGCN(self.out, self.nhid, self.out, self.pooling_ratio, self.dropout_ratio, self.attn_vec_dim)


    def forward(self, feature, A, node_types, use_relu=True):

        h1 = self.lhgcn1(feature, A, node_types)
        h2 = self.lhgcn2(h1, A, node_types)

        final_emb = (h1 + h2)/2

        final_emb = F.relu(final_emb)

        return final_emb

