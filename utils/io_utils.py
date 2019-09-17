import os
import statistics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import networkx as nx
import tensorboardX


def gen_prefix(args):
    name = args.data_name
    if len(args.name_suffix) > 0:
        name += '_' + args.name_suffix
    return name


def gen_explainer_prefix(args):
    name = gen_prefix(args)
    if args.train_num_epochs > 0:
        name += '_e' + str(args.train_num_epochs)
    name += '_explain' 
    if len(args.explainer_suffix) > 0:
        name += '_' + args.explainer_suffix
    return name


def create_filename(save_dir, args, num_epochs=-1):
    '''  create checkpoint name
    Args: 
        args: the arguments parsed in the parser
        num_epochs: epoch number of the model
    '''
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, gen_prefix(args))

    if num_epochs > 0:
        filename += '_e' + str(num_epochs)

    return filename + '.pth.tar'


def save_checkpoint(args, model, cg_dict):
    filename = create_filename(args.ckptdir, args, args.num_epochs)
    torch.save({'args': args,
                'model_state': model.state_dict(),
                'cg': cg_dict}, 
               filename)


def load_ckpt(args, num_epochs=-1):
    filename = create_filename(args.ckptdir, args, num_epochs)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    return ckpt


def log_matrix(writer, mat, name, epoch, fig_size=(8,6), dpi=200):
    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    mat = mat.cpu().detach().numpy()
    if mat.ndim == 1:
        mat = mat[:, np.newaxis]
    plt.imshow(mat, cmap=plt.get_cmap('BuPu'))
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")

    plt.tight_layout()
    fig.canvas.draw()
    writer.add_image(name, tensorboardX.utils.figure_to_image(fig), epoch)


def denoise_graph(adj, feat=None, label=None, threshold=0.1, threshold_num=None,
        max_component=True):
    num_nodes = adj.shape[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.node[0]['self'] = 1
    G.node[1]['self'] = 1

    if feat is not None:
        for node in G.nodes():
            G.node[node]['feat'] = feat[node]
    if label is not None:
        for node in G.nodes():
            G.node[node]['label'] = label[node] 

    if threshold_num is not None:
        adj += np.random.rand(adj.shape[0],adj.shape[1])*1e-4
        threshold = np.sort(adj[adj>0])[-threshold_num]
    if threshold is not None:
        weighted_edge_list = [(i, j, adj[i,j]) for i in range(num_nodes) for j in range(num_nodes) if
                adj[i,j] >= threshold]
    else:
        weighted_edge_list = [(i, j, adj[i,j]) for i in range(num_nodes) for j in range(num_nodes) if
                adj[i,j] > 1e-6]
    G.add_weighted_edges_from(weighted_edge_list)
    
    if max_component:
        G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len) 
    else:
        # remove zero degree nodes
        G.remove_nodes_from(list(nx.isolates(G)))
    
    return G


def log_graph(writer, Gc, name, epoch=-1, identify_self=True, nodecolor='label', fig_size=(4,3),
        dpi=400, label_node_feat=False, edge_vmax=None, args=None):
    '''
    Args:
        nodecolor: the color of node, can be determined by 'label', or 'feat'. 
        For feat, it needs to be one-hot.
    '''
    cmap = plt.get_cmap('Set1')
    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    node_colors = []
    edge_colors = [w for (u,v,w) in Gc.edges.data('weight', default=1)]

    # maximum value for node color
    vmax = 8
    for i in Gc.nodes():
        if nodecolor == 'feat' and 'feat' in Gc.node[i]:
            num_classes = Gc.node[i]['feat'].size()[0]
            if num_classes >= 10:
                cmap = plt.get_cmap('tab20')
                vmax = 19
            elif num_classes >= 8:
                cmap = plt.get_cmap('tab10')
                vmax = 9
            break
    
    feat_labels={}
    for i in Gc.nodes():
        if identify_self and 'self' in Gc.node[i]:
            node_colors.append(0)
        elif nodecolor == 'label' and 'label' in Gc.node[i]:
            node_colors.append(Gc.node[i]['label'] + 1)
        elif nodecolor == 'feat' and 'feat' in Gc.node[i]:
            feat = Gc.node[i]['feat'].detach().numpy()
            # idx with pos val in 1D array
            for j in range(len(feat)):
                if feat[j] == 1:
                    feat_class = j
                    break
            node_colors.append(feat_class)
            feat_labels[i] = feat_class
        else:
            node_colors.append(1)
    if not label_node_feat:
        feat_labels=None

    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    if Gc.number_of_nodes() == 0:
        raise Exception('empty graph')
    if Gc.number_of_edges() == 0:
        raise Exception('empty edge')

    pos_layout = nx.kamada_kawai_layout(Gc)
    # pos_layout = nx.spring_layout(Gc)

    weights = [d for (u,v,d) in Gc.edges(data='weight', default=1)]
    if edge_vmax is None:
        edge_vmax = statistics.median_high(weights)
    edge_vmin = min(weights) / 1.1
    nx.draw(Gc, pos=pos_layout, with_labels=label_node_feat, labels=feat_labels, font_size=4,
            node_color=node_colors, cmap=cmap, vmin=0, vmax=vmax, node_size=50,
            edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'), 
            edge_vmin=edge_vmin, edge_vmax=edge_vmax, width=1.0,
            alpha=0.8)
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

    if args is None:
        save_path = os.path.join('log', name + '.pdf')
    else:
        epoch_str = '_' + str(epoch) if epoch!=-1 else '' 
        save_path = os.path.join(args.logdir, gen_explainer_prefix(args), name + epoch_str + '.pdf')
        print(args.logdir + '/' + gen_explainer_prefix(args) + '/' + name + epoch_str + '.pdf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf')

    img = tensorboardX.utils.figure_to_image(fig)
    writer.add_image(name, img, epoch)