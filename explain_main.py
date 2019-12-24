import os
import random
import pickle
import argparse
from tensorboardX import SummaryWriter

from dgcnn import models
from explain import explain
from utils import io_utils


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments for SEAL Explainer')
    # io settings
    parser.add_argument('--ckptdir', dest='ckptdir',
            help='Model checkpoint directory')
    parser.add_argument('--data-name', 
            help='Network name')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='Suffix added to filename')
    parser.add_argument('--train-num-epochs', dest='train_num_epochs', type=int,
            help='Number of epochs for the checkpoint')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--explainer-suffix', dest='explainer_suffix',
            help='Suffix added to the explainer log')
    parser.add_argument('--no-writer', dest='writer', action='store_const',
            const=False, default=True,
            help='Whether to add writer. Default to True.')
    parser.add_argument('--rm-log', dest='rm_log', action='store_const',
            const=True, default=False,
            help='Whether to remove existing log dir. Default to False.')
    # explain settings
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--gpu', dest='gpu', action='store_const',
            const=True, default=False,
            help='Whether to use GPU. Default to False.')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int,
            help='Number of epochs to learn for explain')
    parser.add_argument('--mask-act', dest='mask_act', type=str,
            help='sigmoid, ReLU')
    parser.add_argument('--mask-bias', dest='mask_bias', action='store_const',
            const=True, default=False,
            help='Whether to add bias. Default to False.')
    parser.add_argument('--threshold-num', dest='threshold_num', type=int,
            help='Threshold number of masked edges to remain')
    parser.add_argument('--threshold-ratio', dest='threshold_ratio', type=float,
            help='Threshold ratio to remain masked edges')
    parser.add_argument('--threshold', dest='threshold', type=float,
            help='Threshold to remain masked edges')
    parser.add_argument('--graph-idx', dest='graph_idx', type=int,
            help='Graph to explain')
    parser.add_argument('--multigraph-class', dest='multigraph_class', type=int,
            help='Graph class to explain')
    parser.add_argument('--mc-idx', dest='mc_idx', type=int,
            help='index of start for multigraph-class explanation')
    parser.add_argument('--graph-indices', dest='graph_indices',
            help='Graphs to explain')
    # optimizaion
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
            help='Type of optimizer scheduler (by default none)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
            help='Learning rate decay ratio')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate')
    # defaults
    parser.set_defaults(ckptdir='test/ckpt',    # io settings
                        data_name='dbac',
                        name_suffix='',
                        train_num_epochs=50,
                        logdir='test/log',
                        explainer_suffix='',
                        cuda='0',    # explain settings
                        num_epochs=300,
                        mask_act='sigmoid',
                        threshold_num=None,
                        threshold_ratio=0.9,
                        threshold=None,
                        graph_idx=-1,
                        multigraph_class=-1,
                        mc_idx = -1,
                        graph_indices='RANDOM30',
                        opt='adam',    # optimization
                        opt_scheduler='none',
                        lr=0.1
                       )
    return parser.parse_args()


def main():
    # args
    prog_args = arg_parse()

    if prog_args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = prog_args.cuda
        print('CUDA', prog_args.cuda)
    else:
        print('Using CPU')

    if prog_args.writer:
        path = os.path.join(prog_args.logdir, io_utils.gen_explainer_prefix(prog_args))
        if prog_args.rm_log and os.path.isdir(path) and os.listdir(path):
            print('Remove existing log dir:', path)
            import shutil
            shutil.rmtree(path)
        writer = SummaryWriter(path)
    else:
        writer = None

    prog_args.edge_dict = {}
    with open('data/'+prog_args.data_name+'/edgeclass', 'r') as f:
        for row in f.readlines():
            row = row.strip().split(',')
            prog_args.edge_dict[(int(row[0]),int(row[1]))] = row[-1]

    ckpt = io_utils.load_ckpt(prog_args, prog_args.train_num_epochs)
    cg_dict = ckpt['cg']

    # build model
    model_args = ckpt['args']
    model_args.mode = 'gpu' if prog_args.gpu else 'cpu'
    model = models.Classifier(model_args)
    if prog_args.gpu:
        model = model.cuda()
    model.load_state_dict(ckpt['model_state'])

    # build explainer
    explainer = explain.Explainer(model, cg_dict['graph'], cg_dict['adj'], cg_dict['feat'], cg_dict['label'],
                                  prog_args, writer=writer, print_training=True, graph_idx=prog_args.graph_idx)

    # explain graph classification
    if prog_args.graph_idx != -1:
        # explain a single graph
        masked_graph = explainer.explain(graph_idx=prog_args.graph_idx)             
    elif prog_args.multigraph_class >= 0:
        # only run for graphs with label specified by multigraph_class
        graph_indices = []
        if prog_args.mc_idx == -1:
            for i, l in enumerate(cg_dict['label']):
                if l == prog_args.multigraph_class:
                    graph_indices.append(i)
        else:
            for i in range(prog_args.mc_idx, cg_dict['label'].shape[0]):
                if cg_dict['label'][i] == prog_args.multigraph_class:
                    graph_indices.append(i)
            
        print('Graph indices for label', prog_args.multigraph_class, ':', graph_indices)
        masked_graphs = explainer.explain_graphs(graph_indices=graph_indices)
    else:
        # explain a customized set of indices
        if prog_args.graph_indices == 'ALL':
            graph_indices = range(cg_dict['label'].shape[0])
        elif prog_args.graph_indices == 'RANDOM30':
            graph_indices = sorted(random.sample(range(cg_dict['label'].shape[0]),30))
        else:
            graph_indices = [int(i) for i in prog_args.graph_indices.split()]
        print('Graph indices to explain', ':', graph_indices)
        masked_graphs = explainer.explain_graphs(graph_indices=graph_indices)
    
    if writer is not None:
        writer.close()

    # save masked_graph(s)
    filename = 'masked_graph.pkl' if prog_args.graph_idx != -1 else 'masked_graphs.pkl'
    file_graph = masked_graph if prog_args.graph_idx != -1 else masked_graphs
    pickle.dump(file_graph, open(os.path.join(prog_args.logdir,io_utils.gen_explainer_prefix(prog_args),filename),'wb'))

    # # save keys
    # if prog_args.graph_idx == -1:
    #     io_utils.keylog2keys(prog_args)


if __name__ == "__main__":
    main()