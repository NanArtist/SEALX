import os
import pickle
import torch
from explain_main import arg_parse
from utils.io_utils import denoise_graph, log_graph

args = arg_parse()
filename = os.path.join(args.logdir, 'dbac_e50_explain_RANDOM30_005/masked_graphs.pkl')
data = pickle.load(open(filename,'rb'))
graph = data[13]
adj = graph.adj
nodes = graph.nodes
G = denoise_graph(adj.cpu().detach().numpy(), torch.tensor(graph.node_features), 
        threshold=args.threshold, threshold_num=args.threshold_num, tokey=True, max_component=True)
log_graph(None, G, '1', identify_self=False, nodecolor='feat', label_node_feat=True, args=args)