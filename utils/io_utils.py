import os, csv
import itertools
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
        if not args.gpu:
            ckpt = torch.load(filename, map_location=torch.device('cpu'))
        else:
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


def denoise_graph(adj, feat=None, label=None, threshold=None, threshold_ratio=None, threshold_num=None, tokey=False, max_component=False):
    num_nodes = adj.shape[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.nodes[0]['self'] = 1
    G.nodes[1]['self'] = 1

    if feat is not None:
        for node in G.nodes():
            G.nodes[node]['feat'] = feat[node]
    if label is not None:
        for node in G.nodes():
            G.nodes[node]['label'] = label[node] 

    if tokey and feat is not None:
        # dFeat: [[node, node_class],...]; dClass: {node_class:node(s),...}
        # dMax: {max_val for node_class:node,...}; lMax: [max_val for node_class,...]
        # th: the threshold_num maximum values in lMax
        dFeat=feat.nonzero().data.numpy()
        dClass0, dMax0, lMax0, dClass1, dMax1, lMax1 = {}, {}, [], {}, {}, []
        for i in range(2):
            for node in adj[i].nonzero()[0]:
                if dFeat[node,1] not in eval('dClass'+str(i)).keys():
                    eval('dClass'+str(i))[dFeat[node,1]] = [node]
                else:
                    eval('dClass'+str(i))[dFeat[node,1]].append(node)
            for node_class in eval('dClass'+str(i)).keys(): 
                tmp_array = adj[[i]*len(eval('dClass'+str(i))[node_class]),[eval('dClass'+str(i))[node_class]]]
                max_val = np.max(tmp_array)
                eval('lMax'+str(i)).append(max_val)
                eval('dMax'+str(i))[max_val] = eval('dClass'+str(i))[node_class][np.argmax(tmp_array)]
        lMax0, lMax1 = np.sort(lMax0), np.sort(lMax1)
        if threshold_num is not None:
            th0, th1 = lMax0[-threshold_num:], lMax1[-threshold_num:]
        elif threshold_ratio is not None:
            thr0, thr1 = lMax0[-1]*threshold_ratio, lMax1[-1]*threshold_ratio
            th0, th1 = lMax0[lMax0>=thr0], lMax1[lMax1>=thr1]
        elif threshold is not None:
            th0, th1 = lMax0[lMax0>=threshold], lMax1[lMax1>=threshold]
        if threshold==None and threshold_ratio==None and threshold_num==None:
            weighted_edge_list = [(0, j, adj[0,j]) for j in range(num_nodes) if adj[0,j] > 1e-6]
        else:
            ind = np.argmax([sum(th0),sum(th1)])
            dMax, th = eval('dMax'+str(ind)), eval('th'+str(ind))
            if threshold_num is not None:  # remove small values relative to th[-1] 
                th = [val for val in th if 10*val >= th[-1]]
            weighted_edge_list = [(ind, dMax[val], val) for val in th]
            attr2attr = list(itertools.combinations(th,2))  # add big edges among attrs selected
            if attr2attr != []:
                for (i,j) in attr2attr:
                    if adj[dMax[i],dMax[j]] >= 0.9*th[0]:
                        weighted_edge_list.append((dMax[i],dMax[j],adj[dMax[i],dMax[j]]))
    else:
        if threshold_num is not None:
            threshold = np.sort(adj[np.triu(adj>0)])[-threshold_num]
        if threshold is not None:
            weighted_edge_list = [(i, j, adj[i,j]) for i in range(num_nodes) for j in range(num_nodes) if
                    adj[i,j] >= threshold]
        else:
            weighted_edge_list = [(i, j, adj[i,j]) for i in range(num_nodes) for j in range(num_nodes) if
                    adj[i,j] > 1e-6]

    G.add_weighted_edges_from(weighted_edge_list)
    
    if max_component:
        def weight(Gc):
            return sum([d for (u,v,d) in Gc.edges(data='weight')]) if len(Gc)!=1 else 0
        G = max((G.subgraph(c) for c in nx.connected_components(G)), key=weight)
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
    edge_colors = [w for (u,v,w) in Gc.edges.data('weight')]

    # maximum value for node color
    vmax = 8
    for i in Gc.nodes():
        if nodecolor == 'feat' and 'feat' in Gc.nodes[i]:
            num_classes = Gc.nodes[i]['feat'].size()[0]
            if num_classes >= 10:
                cmap = plt.get_cmap('tab20')
                vmax = 19
            elif num_classes >= 8:
                cmap = plt.get_cmap('tab10')
                vmax = 9
            break
    
    feat_labels={}
    for i in Gc.nodes():
        if identify_self and 'self' in Gc.nodes[i]:
            node_colors.append(0)
        elif nodecolor == 'label' and 'label' in Gc.nodes[i]:
            node_colors.append(Gc.nodes[i]['label'] + 1)
        elif nodecolor == 'feat' and 'feat' in Gc.nodes[i]:
            feat = Gc.nodes[i]['feat'].detach().numpy()
            # idx with pos val in 1D array
            for j in range(len(feat)):
                if feat[j] != 0:
                    feat_class = j
                    node_colors.append(feat_class)
                    feat_labels[i] = feat_class
                    break
        else:
            node_colors.append(1)
    if not label_node_feat:
        feat_labels=None

    plt.switch_backend('agg')
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    if Gc.number_of_nodes() == 0:
        return None
        # raise Exception('empty graph')
    if Gc.number_of_edges() == 0:
        return None
        # raise Exception('empty edge')

    pos_layout = nx.kamada_kawai_layout(Gc)
    # pos_layout = nx.spring_layout(Gc)

    weights = [d for (u,v,d) in Gc.edges(data='weight')]
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


def subgraph2key(args, denoise_graph, pred_loss):
    key = []
    edges = list(denoise_graph.edges)
    for edge in edges:
        class0 = denoise_graph.nodes[edge[0]]['feat'].data.numpy().nonzero()[0][0]
        class1 = denoise_graph.nodes[edge[1]]['feat'].data.numpy().nonzero()[0][0]
        if class0 > class1:
            key.append(args.edge_dict[(class1,class0)])
        else:
            key.append(args.edge_dict[(class0,class1)])
    
    filename = 'keylog'
    if args.mc_idx != -1:
        filename += '_' + str(args.mc_idx) + '+'
    elif args.mc_sidx != -1 and args.mc_eidx != -1:
        filename += '_' + str(args.mc_sidx) + '-' + str(args.mc_eidx)
    
    with open(os.path.join(args.logdir, gen_explainer_prefix(args), filename), 'a+') as f:
        writer = csv.writer(f)
        if pred_loss < 0.69:
            writer.writerow(key)
        else:
            writer.writerow(['False']+key)
    return key


def keylog2keys(args):
    filepath = os.path.join(args.logdir, gen_explainer_prefix(args))
    data = list(csv.reader(open(filepath+'/keylog', 'r')))

    cands = {}
    lCands = []  # [{(,),(,)...},{...},...]
    candkeys = [row for row in data if row[0]!='False']
    print('Remove {}/{} keys whose pred_loss are more than 0.69.'.format(len(data)-len(candkeys), len(data)))
    for candk in candkeys:
        cand = set(candk)
        if cand not in lCands:
            cands[len(lCands)] = 1
            lCands.append(cand)
        else:
            idx = lCands.index(cand)
            cands[idx] += 1
    cands = sorted(cands.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
    with open(filepath+'/keys', 'w', newline="") as f:
        writer = csv.writer(f)
        for key in cands:
            writer.writerow(list(lCands[key[0]])+[key[1]])
        writer.writerow(['Remove {}/{} keys whose pred_loss are more than 0.69.'.format(len(data)-len(candkeys), len(data))])
