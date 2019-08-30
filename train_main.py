import os, math, random, argparse, time
import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
from dgcnn.models import *
from utils.train_utils import *
from utils.io_utils import save_checkpoint


def args_parse():
    parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
    # general settings
    parser.add_argument('--data-name', help='network name')
    parser.add_argument('--train-name', help='train name')
    parser.add_argument('--test-name', help='test name')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, metavar='S', help='random seed')
    parser.add_argument('--name-suffix', dest='name_suffix', help='suffix added to the output filename')
    parser.add_argument('--logdir', dest='logdir', help='Tensorboard log directory')
    parser.add_argument('--ckptdir', dest='ckptdir', help='Model checkpoint directory')
    # settings for stage 1 and 2
    parser.add_argument('--max-train-num', type=int, help='set maximum number of train links (to fit into memory)')
    parser.add_argument('--test-ratio', type=float, help='ratio of test links')
    parser.add_argument('--hop', metavar='S', help='enclosing subgraph hop number, options: 1, 2,..., "auto"')
    parser.add_argument('--max-nodes-per-hop', help='if > 0, upper bound the # nodes per hop by subsampling')
    parser.add_argument('--use-embedding', action='store_true', default=False, help='whether to use node2vec node embeddings')
    parser.add_argument('--use-attribute', action='store_true', default=False, help='whether to use node attributes')
    # DGCNN configurations (primary)
    cmd_args = parser.add_argument_group(description='Arguments for DGCNN')
    cmd_args.add_argument('--mode', help='cpu/gpu')
    cmd_args.add_argument('--gm', help='gnn model to use')
    cmd_args.add_argument('--feat-dim', type=int, help='dimension of discrete node feature (maximum node tag)')
    cmd_args.add_argument('--edge-feat-dim', type=int, help='dimension of edge features')
    cmd_args.add_argument('--attr-dim', type=int, help='dimension(s) of node attributes')
    cmd_args.add_argument('--num-class', type=int, help='#classes')
    cmd_args.add_argument('--num-epochs', type=int, help='number of epochs')
    cmd_args.add_argument('--batch-size', type=int, help='minibatch size')
    cmd_args.add_argument('--latent-dim', help='dimension(s) of latent layers')
    cmd_args.add_argument('--sortpooling-k', type=float, help='number of nodes kept after SortPooling')
    cmd_args.add_argument('--conv1d-activation', type=str, help='which nn activation layer to use')
    cmd_args.add_argument('--output-dim', type=int, help='graph embedding output size')
    cmd_args.add_argument('--hidden-dim', type=int, help='dimension of mlp hidden layer')
    cmd_args.add_argument('--dropout', type=bool, help='whether add dropout after dense layer')
    cmd_args.add_argument('--printAUC', type=bool, help='whether to print AUC (for binary classification only)')
    # optimization
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str, help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str, help='Type of optimizer scheduler (by default none)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int, help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float, help='Learning rate decay ratio')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int, help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--lr', dest='lr', type=float, help='Learning rate')
    # defaults
    parser.set_defaults(data_name='dbac',  # general settings
                        train_name=None,
                        test_name=None,
                        seed=1,
                        name_suffix='',
                        logdir='log',
                        ckptdir='ckpt',
                        max_train_num=100000,  # settings for stage 1 and 2
                        test_ratio=0.1,
                        hop=1,
                        max_nodes_per_hop=None,
                        mode='cpu',  # DGCNN configurations (primary)
                        gm='DGCNN',
                        feat_dim=0,
                        edge_feat_dim=0,
                        attr_dim = 0,
                        num_class=2,
                        num_epochs=50,
                        batch_size=50,
                        latent_dim=[32,32,32,1],
                        sortpooling_k=0.6,
                        conv1d_activation='ReLU',
                        output_dim=0,
                        hidden_dim=128,
                        dropout=True,
                        printAUC=True,
                        opt='adam',  # optimization
                        opt_scheduler='none',
                        lr=1e-4
                       )
    return parser.parse_args()


def loop_dataset(g_list, classifier, sample_idxes, bsize, optimizer=None, scheduler=None):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            pred, mae, loss = classifier(batch_graph)
            all_scores.append(pred.cpu().detach())  # for binary classification
        else:
            logits, loss, acc = classifier(batch_graph)
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                    scheduler.step()

        loss = loss.data.cpu().detach().numpy()
        if classifier.regression:
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mae) )
            total_loss.append( np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )
            total_loss.append( np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()
    
    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    
    if not classifier.regression:
        all_targets = np.array(all_targets)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc]))
    
    return avg_loss


def main():
    '''argument settings'''
    args = args_parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))
    if args.hop != 'auto':
        args.hop = int(args.hop)
    if args.max_nodes_per_hop is not None:
        args.max_nodes_per_hop = int(args.max_nodes_per_hop)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    ''' stage 1: node information matrix construction
    and stage 2: enclosing subgraph extraction'''
    # load net and sample train_pos, train_neg, test_pos, test_neg links
    if args.train_name is None:
        args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
        data = sio.loadmat(args.data_dir)
        net = data['net']
        if 'group' in data.keys():
        # load node attributes (here a.k.a. node classes)
            attributes = data['group'].toarray().astype('float32')
        else:
            attributes = None
        if 'dbac' in data.keys():
        # load same_as links
            dbac = data['dbac']
        if 'iden' in data.keys():
        # load vid_entity mapping (sparse matrix in scipy is different from that in matlab, while general matrix is equal.)
            iden = data['iden']
            for i in range(iden.shape[0]):
                iden[i,0] = iden[i,0] - 1
                iden[i,1] = iden[i,1] - 1
        if False:
        # check whether net is symmetric (for small nets only)
            net_ = net.toarray()
            assert(np.allclose(net_, net_.T, atol=1e-8))
        train_pos, train_neg, test_pos, test_neg = sample_neg(mat=dbac, test_ratio=args.test_ratio, max_train_num=args.max_train_num)
        if args.data_name == 'dbac':
            train_pos, train_neg, test_pos, test_neg = entity2vid(train_pos, iden), entity2vid(train_neg, iden), entity2vid(test_pos, iden), entity2vid(test_neg, iden)
        verify_sample(net, train_pos, train_neg, test_pos, test_neg)
    else:
        args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
        args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
        train_idx = np.loadtxt(args.train_dir, dtype=int)
        test_idx = np.loadtxt(args.test_dir, dtype=int)
        max_idx = max(np.max(train_idx), np.max(test_idx))
        net = ssp.csc_matrix((np.ones(len(train_idx)),(train_idx[:,0],train_idx[:,1])), shape=(max_idx+1, max_idx+1))
        net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
        net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
        # sample negative links (resp. for train and test)
        train_pos = (train_idx[:, 0], train_idx[:, 1])
        test_pos = (test_idx[:, 0], test_idx[:, 1])
        train_pos, train_neg, test_pos, test_neg = sample_neg(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)

    A = net.copy()  # the observed network
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links
    
    # node information matrix construction
    node_information = None
    if args.use_embedding:
        embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
        node_information = embeddings
    if args.use_attribute and attributes is not None:
        if node_information is not None:
            node_information = np.concatenate([node_information, attributes], axis=1)
        else:
            node_information = attributes
    # enclosing subgraph extraction
    train_graphs, test_graphs, max_n_label = links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, args.hop, args.max_nodes_per_hop, node_information)
    print('#train: %d, #test: %d' % (len(train_graphs), len(test_graphs)))

    '''stage 3: GNN Learning'''
    # DGCNN configurations
    args.mode = 'gpu' if args.cuda else 'cpu'
    args.feat_dim = max_n_label + 1
    if node_information is not None:
        args.attr_dim = node_information.shape[1]
    if args.sortpooling_k <= 1:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs+test_graphs])
        args.sortpooling_k = num_nodes_list[int(math.ceil(args.sortpooling_k*len(num_nodes_list)))-1]
        args.sortpooling_k = max(10, args.sortpooling_k)
        print('k used in SortPooling is: ' + str(args.sortpooling_k))

    # model and optimizer
    classifier = Classifier(args)
    if args.mode == 'gpu':
        classifier = classifier.cuda()
    scheduler, optimizer = build_optimizer(args, classifier.parameters())

    # train and test
    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(args.num_epochs):
        random.shuffle(train_idxes)
        classifier.train()
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, args.batch_size, optimizer=optimizer, scheduler=scheduler)
        if not args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

        classifier.eval()
        test_loss = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))), args.batch_size)
        if not args.printAUC:
            test_loss[2] = 0.0
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1], test_loss[2]))

    # save evaluation results
    os.makedirs(args.logdir+'/train', exist_ok=True)
    with open(args.logdir+'/train/acc_results.txt', 'a+') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())+'\t'+args.data_name+'\t'+str(test_loss[1])+'\n')

    if args.printAUC:
        with open(args.logdir+'/train/auc_results.txt', 'a+') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())+'\t'+args.data_name+'\t'+str(test_loss[2])+'\n')

    # save checkpoint
    adj = train_graphs + test_graphs
    cg_dict = { 'adj': adj,
                'feat': 0,
                'label': [graph.label for graph in adj],
                'pred': 0, 
                'train_idx': train_idxes}
    save_checkpoint(args, classifier, cg_dict)


if __name__ == "__main__":
    main()