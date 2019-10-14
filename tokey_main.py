import os, csv
import pickle
import torch
from explain_main import arg_parse
from utils import io_utils

args = arg_parse()
filename = 'masked_graph.pkl' if args.graph_idx != -1 else 'masked_graphs.pkl'
filepath = os.path.join(args.logdir, io_utils.gen_explainer_prefix(args), filename)
data = pickle.load(open(filepath,'rb'))

edge_dict = {}
with open('data/'+args.data_name+'/edgeclass', 'r') as f:
    for row in f.readlines():
        row = row.strip().split('\t')
        edge_dict[(int(row[0]),int(row[1]))] = (row[2], row[4],row[3])

if args.graph_idx != -1:
    key = []
    edges = list(data.key.edges)
    for edge in edges:
        class0 = data.node_attrs[edge[0]].nonzero()[0][0]
        class1 = data.node_attrs[edge[1]].nonzero()[0][0]
        if class0 > class1:
            key.append(edge_dict[(class1,class0)])
        else:
            key.append(edge_dict[(class0,class1)])
    with open(os.path.join(args.logdir, io_utils.gen_explainer_prefix(args), 'key'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow([edge[1] for edge in key])
    print('pred_loss of last epoch is', data.pred_loss)
else:
    cands = {}
    lCands = []  # [{(,),(,)...},{...},...]
    graphs = [graph for graph in data if graph.pred_loss < 0.69]
    print('Remove {}/{} graphs whose pred_loss are more than 0.69.'.format(len(data)-len(graphs), len(data)))
    for graph in graphs:
        cand = set()
        edges = list(graph.key.edges)
        for edge in edges:
            class0 = graph.node_attrs[edge[0]].nonzero()[0][0]
            class1 = graph.node_attrs[edge[1]].nonzero()[0][0]
            if class0 > class1:
                cand.update([(class1,class0)])
            else:
                cand.update([(class0,class1)])
        if cand not in lCands:
            cands[len(lCands)] = 1
            lCands.append(cand)
        else:
            idx = lCands.index(cand)
            cands[idx] += 1
    ikeys, keys = [], [] 
    for i in cands.keys():
        if cands[i] > 0.1 * len(graphs):
            ikeys.append(lCands[i])
    for key in ikeys:
        keys.append([edge_dict[edge] for edge in key])
    with open(os.path.join(args.logdir, io_utils.gen_explainer_prefix(args), 'keys'), 'w') as f:
        writer = csv.writer(f)
        for key in keys:
            writer.writerow([edge[1] for edge in key])