import sys
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from dgcnn.lib.gnn_lib import GNNLIB
from dgcnn.lib.util import weights_init, gnn_spmm


class Classifier(nn.Module):
    def __init__(self, args, regression=False):
        super(Classifier, self).__init__()
        self.args = args
        self.regression = regression
        if args.gm == 'DGCNN':
            model = DGCNN
        else:
            print('unknown gm %s' % args.gm)
            sys.exit()

        if args.gm == 'DGCNN':
            self.gnn = model(latent_dim=args.latent_dim,
                             output_dim=args.output_dim,
                             num_node_feats=args.feat_dim+args.embed_dim+args.attr_dim,
                             num_edge_feats=args.edge_feat_dim,
                             k=args.sortpooling_k,
                             conv1d_activation=args.conv1d_activation)
        out_dim = args.output_dim
        if out_dim == 0:
            if args.gm == 'DGCNN':
                out_dim = self.gnn.dense_dim
            else:
                out_dim = args.latent_dim
        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=args.hidden_dim, num_class=args.num_class, with_dropout=args.dropout)
        if regression:
            self.mlp = MLPRegression(input_size=out_dim, hidden_size=args.hidden_dim, with_dropout=args.dropout)

    def PrepareFeatureLabel(self, batch_graph):
        if self.regression:
            labels = torch.FloatTensor(len(batch_graph))
        else:
            labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0

        if batch_graph[0].node_tags is not None:
            node_tag_flag = True
            concat_tag = []
        else:
            node_tag_flag = False

        if batch_graph[0].node_embeds is not None:
            node_embed_flag = True
            concat_embed = []
        else:
            node_embed_flag = False

        if batch_graph[0].node_attrs is not None:
            node_attr_flag = True
            concat_attr = []
        else:
            node_attr_flag = False

        if self.args.edge_feat_dim > 0:
            edge_feat_flag = True
            concat_edge_feat = []
        else:
            edge_feat_flag = False

        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            if node_tag_flag == True:
                concat_tag += batch_graph[i].node_tags
            if node_embed_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_embeds).type('torch.FloatTensor')
                concat_embed.append(tmp)
            if node_attr_flag == True:
                tmp = torch.from_numpy(batch_graph[i].node_attrs).type('torch.FloatTensor')
                concat_attr.append(tmp)
            if edge_feat_flag == True:
                if batch_graph[i].edge_features is not None:  # in case no edge in graph[i]
                    tmp = torch.from_numpy(batch_graph[i].edge_features).type('torch.FloatTensor')
                    concat_edge_feat.append(tmp)

        if node_tag_flag == True:
            concat_tag = torch.LongTensor(concat_tag).view(-1, 1)
            node_tag = torch.zeros(n_nodes, self.args.feat_dim)
            node_tag.scatter_(1, concat_tag, 1)

        if node_embed_flag == True:
            node_embed = torch.cat(concat_embed, 0)
        
        if node_attr_flag == True:
            node_attr = torch.cat(concat_attr, 0)

        node_feat_flag = True
        if node_embed_flag and node_attr_flag:
            # concatenate embeddings and attributes as continuous node features
            node_feat = torch.cat((node_embed, node_attr), 1)
        elif node_embed_flag == True and node_attr_flag == False:
            node_feat = node_embed
        elif node_embed_flag == False and node_attr_flag == True:
            node_feat = node_attr
        else:
            node_feat_flag = False

        if node_feat_flag and node_tag_flag:
            # concatenate one-hot embedding of node tags with continuous node features
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
        elif node_feat_flag == False and node_tag_flag == True:
            node_feat = node_tag
        elif node_feat_flag == True and node_tag_flag == False:
            pass
        else:
            node_feat = torch.ones(n_nodes, 1)  # use all-one vector as node features
        
        if edge_feat_flag == True:
            edge_feat = torch.cat(concat_edge_feat, 0)

        if self.args.mode == 'gpu':
            node_feat = node_feat.cuda()
            labels = labels.cuda()
            if edge_feat_flag == True:
                edge_feat = edge_feat.cuda()

        if edge_feat_flag == True:
            return node_feat, edge_feat, labels

        return node_feat, labels

    def forward(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return self.mlp(embed, labels)

    def output_features(self, batch_graph):
        feature_label = self.PrepareFeatureLabel(batch_graph)
        if len(feature_label) == 2:
            node_feat, labels = feature_label
            edge_feat = None
        elif len(feature_label) == 3:
            node_feat, edge_feat, labels = feature_label
        embed = self.gnn(batch_graph, node_feat, edge_feat)
        return embed, labels
        

class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30, conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU'):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats + num_edge_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k - 2) / 2 + 1)
        self.dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        self.conv1d_activation = eval('nn.{}()'.format(conv1d_activation))

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs) + 1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = GNNLIB.PrepareSparseMatrices(graph_list)

        if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            node_degs = node_degs.cuda()
        node_feat = Variable(node_feat)
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
            if torch.cuda.is_available() and isinstance(node_feat, torch.cuda.FloatTensor):
                edge_feat = edge_feat.cuda()
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)

        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):
        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            input_edge_linear = edge_feat
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        lv = 0
        cur_message_layer = node_feat
        cat_message_layers = []
        while lv < len(self.latent_dim):
            n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer  # Y = (A + I) * X
            node_linear = self.conv_params[lv](n2npool)  # Y = Y * W
            normalized_linear = node_linear.div(node_degs)  # Y = D^-1 * Y
            cur_message_layer = torch.tanh(normalized_linear)
            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, 1)

        ''' sortpooling layer '''
        sort_channel = cur_message_layer[:, -1]
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim)
        if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
            batch_sortpooling_graphs = batch_sortpooling_graphs.cuda()

        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        accum_count = 0
        for i in range(subg_sp.size()[0]):
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.total_latent_dim)
                if torch.cuda.is_available() and isinstance(node_feat.data, torch.cuda.FloatTensor):
                    to_pad = to_pad.cuda()

                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]

        ''' traditional 1d convlution and dense layers '''
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))
        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = self.conv1d_activation(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = self.conv1d_activation(conv1d_res)

        to_dense = conv1d_res.view(len(graph_sizes), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = self.conv1d_activation(out_linear)
        else:
            reluact_fp = to_dense

        return self.conv1d_activation(reluact_fp)


class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, with_dropout=False):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)
        pred = self.h2_weights(h1)[:, 0]

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            mae = mae.cpu().detach()
            return pred, mae, mse
        else:
            return pred


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)
            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits


class GNNGraph(object):
    def __init__(self, nodes, g, label, node_tags=None, node_embeds=None, node_attrs=None):
        ''' nodes: indices in net for nodes of g
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_embeds: a numpy array of continuous node embeddings
            node_attrs: a numpy array of node attributes (here a.k.a. node classes)
        '''
        self.nodes = nodes
        self.degs = list(dict(g.degree).values())
        self.label = label
        self.node_tags = node_tags
        self.num_nodes = len(node_tags)
        self.node_embeds = node_embeds  # numpy array (node_num * embed_dim)
        self.node_attrs = node_attrs  # numpy array (node_num * attr_dim)
        self.adj = nx.to_scipy_sparse_matrix(g)
        self.add_mask = False

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)        
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(edge_features.values()[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

    def update(self, masked_adj, masked_x=None):
        self.adj = masked_adj.squeeze()
        self.add_mask = True
        if masked_x is not None:
            self.node_attrs = masked_x.squeeze().detach().numpy()