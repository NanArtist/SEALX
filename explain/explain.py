import math, time
import numpy as np
import torch
import torch.nn as nn

from utils import io_utils
from utils.train_utils import build_optimizer


class Explainer:
    def __init__(self, model, graph, adj, feat, label, pred, train_idx, args, writer=None,
            print_training=True, graph_idx=0):
        print('Initializing Explainer')
        self.model = model
        self.graph = graph
        self.adj = adj
        self.feat = feat
        self.label = label
        self.pred = pred
        self.train_idx = train_idx
        self.args = args
        self.writer = writer
        self.print_training = print_training
        self.graph_idx = graph_idx

    def explain(self, graph_idx=0, unconstrained=False):
        # prefix for filenames
        gidx = 'gidx_'+str(graph_idx)+'_' if graph_idx!=0 else ''

        # index of the query node in the new adj
        graph = self.graph[graph_idx]

        sub_adj = self.adj[graph_idx]
        sub_adj = np.expand_dims(sub_adj.toarray(), axis=0)
        adj = torch.tensor(sub_adj, dtype=torch.float)

        sub_feat = self.feat[graph_idx]
        if sub_feat is not None:
            sub_feat = np.expand_dims(sub_feat, axis=0)
            sub_feat = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        x = sub_feat

        sub_label = self.label[graph_idx]
        label = torch.tensor(sub_label, dtype=torch.long)

        pred_label = np.argmax(self.pred[0][graph_idx], axis=0)
        print('pred label:', pred_label)

        explainer = ExplainModule(self.model, graph, adj, x, label, self.args, writer=self.writer, graph_idx=graph_idx)
        if self.args.gpu:
            explainer = explainer.cuda()

        self.model.eval()
        explainer.train()
        begin_time = time.time()
        for epoch in range(self.args.num_epochs):
            explainer.zero_grad()
            explainer.optimizer.zero_grad()
            ypred = explainer(unconstrained=unconstrained)
            loss = explainer.loss(ypred, pred_label, epoch)
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            mask_density = explainer.mask_density()
            if self.print_training:
                print('epoch:', epoch, '; loss:', loss.item(),
                        '; mask density:', mask_density.item(),
                        '; pred:', ypred) 
            single_subgraph_label = sub_label.squeeze()

            if self.writer is not None:
                self.writer.add_scalar(gidx+'mask/density', mask_density, epoch)
                self.writer.add_scalar(gidx+'optimization/lr', explainer.optimizer.param_groups[0]['lr'], epoch)
                if epoch % 25 == 0:
                    explainer.log_mask(epoch)
                    explainer.log_masked_adj(epoch, label=single_subgraph_label)
                    explainer.log_adj_grad(pred_label, epoch, label=single_subgraph_label)

        print('finished training in', time.time()-begin_time)
        masked_adj = explainer.masked_adj[0].cpu().detach().numpy()*sub_adj.squeeze()
        
        if self.args.graph_idx != -1:
            G_orig = io_utils.denoise_graph(self.adj[graph_idx], 0, feat=self.feat[graph_idx],
                threshold=None, max_component=False) 
            io_utils.log_graph(self.writer, G_orig, 'explain/gidx_{}'.format(graph_idx),
                identify_self=False, nodecolor='feat', args=self.args)
            G_denoised = io_utils.denoise_graph(masked_adj, 0, threshold_num=20, feat=self.feat[graph_idx], 
                max_component=False)
            io_utils.log_graph(self.writer, G_denoised, 
                'explain/gidx_{}_label_{}'.format(graph_idx, self.label[graph_idx]),
                identify_self=False, nodecolor='feat', args=self.args)
            io_utils.plot_cmap_tb(self.writer, 'tab20', 20, 'tab20_cmap')

        return masked_adj

    def explain_graphs(self, graph_indices):
        masked_adjs = []

        for graph_idx in graph_indices:
            masked_adj = self.explain(graph_idx=graph_idx)
            masked_adjs.append(masked_adj)
            
            # import networkx as nx
            # G_orig = nx.from_numpy_matrix(self.adj[graph_idx].cpu().detach().numpy())
            G_orig = io_utils.denoise_graph(self.adj[graph_idx], 0, feat=self.feat[graph_idx],
                threshold=None, max_component=False)
            io_utils.log_graph(self.writer, G_orig, 'explain/gidx_{}'.format(graph_idx),
                identify_self=False, nodecolor='feat', args=self.args)

            G_denoised = io_utils.denoise_graph(masked_adj, 0, threshold_num=20, feat=self.feat[graph_idx],
                max_component=False)
            io_utils.log_graph(self.writer, G_denoised, 
                'explain/gidx_{}_label_{}'.format(graph_idx, self.label[graph_idx]), 
                identify_self=False, nodecolor='feat', args=self.args)

        # plot cmap for graphs' node features
        io_utils.plot_cmap_tb(self.writer, 'tab20', 20, 'tab20_cmap')

        return masked_adjs


class ExplainModule(nn.Module):
    def __init__(self, model, graph, adj, x, label, args, writer=None, graph_idx=0, use_sigmoid=True):
        super(ExplainModule, self).__init__()
        self.model = model
        self.graph = graph
        self.adj = adj
        self.x = x
        self.label = label
        self.args = args
        self.mask_act = args.mask_act
        self.writer = writer
        self.graph_idx = graph_idx
        self.use_sigmoid = use_sigmoid

        num_nodes = adj.size()[1]
        self.mask, self.mask_bias = self.construct_edge_mask(num_nodes, init_strategy='normal')
        self.feat_mask = self.construct_feat_mask(x.size(-1), init_strategy='constant') \
            if x is not None else None
        params = [self.mask]
        if self.feat_mask is not None:
            params.append(self.feat_mask)
        if self.mask_bias is not None:
            params.append(self.mask_bias)
        
        # For masking diagonal entries
        self.diag_mask = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
        if args.gpu:
            self.diag_mask = self.diag_mask.cuda()

        self.scheduler, self.optimizer = build_optimizer(args, params)

        self.coeffs = {'size': 0.005, 'feat_size': 1.0, 'ent': 1.0, 'feat_ent':0.1, 'lap': 1.0}

    def construct_edge_mask(self, num_nodes, init_strategy='normal', const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == 'normal':
            std = nn.init.calculate_gain('relu') * math.sqrt(2.0/(num_nodes+num_nodes))
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == 'const':
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None
       
        return mask, mask_bias

    def construct_feat_mask(self, feat_dim, init_strategy='normal'):
        mask = nn.Parameter(torch.FloatTensor(feat_dim))
        if init_strategy == 'normal':
            std = 0.1
            with torch.no_grad():
                mask.normal_(1.0, std)
        elif init_strategy == 'constant':
            with torch.no_grad():
                nn.init.constant_(mask, 0.0)
        return mask

    def forward(self, unconstrained=False, mask_features=True):
        if unconstrained:
            sym_mask = torch.sigmoid(self.mask) if self.use_sigmoid else self.mask
            self.masked_adj = torch.unsqueeze((sym_mask + sym_mask.t()) / 2, 0) * self.diag_mask
            #TODO self.graph.update(self.masked_adj)
        else:
            self.masked_adj = self._masked_adj()
            if mask_features and self.x is not None:
                x = self.x.cuda() if self.args.gpu else self.x
                feat_mask = torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
                marginalize = False
                if marginalize:
                    std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                    mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                    z = torch.normal(mean=mean_tensor, std=std_tensor)
                    x = x + z * (1 - feat_mask)
                else:
                    x = x * feat_mask
            #TODO self.graph.update(self.masked_adj, x)

        logits, loss, acc = self.model([self.graph])
        res = nn.Softmax(dim=0)(np.expand_dims(logits.data.numpy(), axis=0)) 

        return res

    def _masked_adj(self):
        sym_mask = self.mask
        if self.mask_act == 'sigmoid':
            sym_mask = torch.sigmoid(self.mask)
        elif self.mask_act == 'ReLU':
            sym_mask = nn.ReLU()(self.mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj
        masked_adj = adj * sym_mask
        if self.args.mask_bias:
            bias = (self.mask_bias + self.mask_bias.t()) / 2
            bias = nn.ReLU6()(bias * 6) / 6
            masked_adj += (bias + bias.t()) / 2
        return masked_adj * self.diag_mask

    def loss(self, pred, pred_label, epoch, grad=False):
        '''
        Args:
            pred: prediction made by current model
            pred_label: the label predicted by the original model.
        '''
        # prefix for names
        gidx = 'gidx_'+str(self.graph_idx)+'_' if self.graph_idx!=0 else ''

        # pred
        mi_obj = False
        if mi_obj:
            pred_loss = - torch.sum(pred * torch.log(pred))
        else:
            pred_label_node = pred_label
            gt_label_node = self.label
            logit = pred[gt_label_node] 
            pred_loss = -torch.log(logit)

        # size
        mask = self.mask
        if self.mask_act == 'sigmoid':
            mask = torch.sigmoid(self.mask)
        elif self.mask_act == 'ReLU':
            mask = nn.ReLU()(self.mask)
        size_loss = self.coeffs['size'] * torch.sum(mask)

        # pre_mask_sum = torch.sum(self.feat_mask)
        feat_mask = torch.sigmoid(self.feat_mask) if self.use_sigmoid else self.feat_mask
        feat_size_loss = self.coeffs['feat_size'] * torch.mean(feat_mask) 

        # entropy
        mask_ent = -mask * torch.log(mask) - (1-mask) * torch.log(1-mask)
        mask_ent_loss = self.coeffs['ent'] * torch.mean(mask_ent)

        feat_mask_ent = -feat_mask * torch.log(feat_mask) - (1-feat_mask) * torch.log(1-feat_mask)
        feat_mask_ent_loss = self.coeffs['feat_ent'] * torch.mean(feat_mask_ent)

        # laplacian
        D = torch.diag(torch.sum(self.masked_adj[0], 0))
        m_adj = self.masked_adj
        L = D - m_adj 
        pred_label_t = torch.tensor(pred_label, dtype=torch.float)
        if self.args.gpu:
            pred_label_t = pred_label_t.cuda()
            L = L.cuda()
        lap_loss = 0

        loss = pred_loss + size_loss + feat_size_loss + mask_ent_loss + lap_loss
        
        if self.writer is not None:
            self.writer.add_scalar(gidx+'optimization/pred_loss', pred_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/size_loss', size_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/feat_size_loss', feat_size_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/mask_ent_loss', mask_ent_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/feat_mask_ent_loss', feat_mask_ent_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/lap_loss', lap_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/overall_loss', loss, epoch)
        return loss

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def log_mask(self, epoch):
        # prefix for names
        gidx = 'gidx_'+str(self.graph_idx)+'_' if self.graph_idx!=0 else ''

        io_utils.log_matrix(self.writer, self.mask, gidx+'mask/mask', epoch, fig_size=(4, 3), dpi=400)
        
        io_utils.log_matrix(self.writer, torch.sigmoid(self.feat_mask), gidx+'mask/feat_mask', epoch)

        # use [0] to remove the batch dim
        io_utils.log_matrix(self.writer, self.masked_adj[0], gidx+'mask/adj', epoch, fig_size=(4, 3), dpi=400)

        if self.args.mask_bias:
            io_utils.log_matrix(self.writer, self.mask_bias, gidx+'mask/bias', epoch, fig_size=(4, 3), dpi=400)

    def log_masked_adj(self, epoch, label=None):
        # prefix for names
        gidx = 'gidx_'+str(self.graph_idx)+'_' if self.graph_idx!=0 else ''
        name = gidx+'mask/graph'
        
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        G = io_utils.denoise_graph(masked_adj, 0, feat=self.x[0], threshold=0.2, # threshold_num=20,
                max_component=True)
        io_utils.log_graph(self.writer, G, name=name, identify_self=False,
                    nodecolor='feat', epoch=epoch, label_node_feat=True, edge_vmax=None, args=self.args)

    def log_adj_grad(self, pred_label, epoch, label=None):
        # prefix for names
        gidx = 'gidx_'+str(self.graph_idx)+'_' if self.graph_idx!=0 else ''

        predicted_label = pred_label
        adj_grad, x_grad = self.adj_feat_grad(predicted_label)
        adj_grad = torch.abs(adj_grad)[0]
        x_grad = torch.sum(x_grad[0], 0, keepdim=True).t()
        
        io_utils.log_matrix(self.writer, x_grad, gidx+'grad/feat', epoch)

        adj_grad = (adj_grad + adj_grad.t()) / 2
        io_utils.log_matrix(self.writer, adj_grad, gidx+'grad/adj', epoch)
        adj_grad = (adj_grad * self.adj).squeeze()
        io_utils.log_matrix(self.writer, adj_grad, gidx+'grad/adj1', epoch)
        io_utils.log_matrix(self.writer, self.adj.squeeze(), gidx+'grad/adj_orig', epoch)

        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        G = io_utils.denoise_graph(masked_adj, 0, feat=self.x[0], threshold=None, max_component=False)
        io_utils.log_graph(self.writer, G, name=gidx+'grad/graph_orig', epoch=epoch, identify_self=False,
                label_node_feat=True, nodecolor='feat', edge_vmax=None, args=self.args)

        adj_grad = adj_grad.cpu().detach().numpy()
        print('GRAPH model')
        G = io_utils.denoise_graph(adj_grad, 0, feat=self.x[0], threshold=0.0003, # threshold_num=20,
                max_component=True)
        io_utils.log_graph(self.writer, G, name=gidx+'grad/graph', epoch=epoch, identify_self=False,
                    label_node_feat=True, nodecolor='feat', edge_vmax=None, args=self.args)

    def adj_feat_grad(self, pred_label_node):
        self.model.zero_grad()
        self.adj.requires_grad = True
        self.x.requires_grad = True
        if self.adj.grad is not None:
            self.adj.grad.zero_()
            self.x.grad.zero_()
        if self.args.gpu:
            adj = self.adj.cuda()
            x = self.x.cuda()
            label = self.label.cuda()
        else:
            x, adj = self.x, self.adj
        ypred, _ = self.model(x, adj)
        logit = nn.Softmax(dim=0)(ypred[0])
        
        logit = logit[pred_label_node]
        loss = -torch.log(logit)
        loss.backward()
        return self.adj.grad, self.x.grad