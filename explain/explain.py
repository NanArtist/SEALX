import math, time
import numpy as np
import torch
import torch.nn as nn

from utils import io_utils
from utils.train_utils import build_optimizer


class Explainer:
    def __init__(self, model, graph, adj, feat, label, args, writer=None,
            print_training=True, graph_idx=0):
        print('Initializing Explainer')
        self.model = model
        self.graph = graph
        self.adj = adj
        self.feat = feat
        self.label = label
        self.args = args
        self.writer = writer
        self.print_training = print_training
        self.graph_idx = graph_idx

    def explain(self, graph_idx=0, unconstrained=False):
        # prefix for filenames
        gidx = 'g'+str(graph_idx)+'_'

        # index of the query node in the new adj
        graph = self.graph[graph_idx]
        graph.gidx = graph_idx

        sub_adj = self.adj[graph_idx]
        sub_adj = np.expand_dims(sub_adj.toarray(), axis=0)
        adj = torch.tensor(sub_adj, dtype=torch.float)

        sub_feat = self.feat[graph_idx] if self.feat is not None else None
        if sub_feat is not None:
            sub_feat = np.expand_dims(sub_feat, axis=0)
            sub_feat = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float)
        x = sub_feat

        sub_label = self.label[graph_idx]
        label = torch.tensor(sub_label, dtype=torch.long)

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
            loss, pred_loss = explainer.loss(ypred, epoch)
            loss.backward()

            explainer.optimizer.step()
            if explainer.scheduler is not None:
                explainer.scheduler.step()

            mask_density = explainer.mask_density()
            if self.print_training:
                print('epoch:', epoch, ';\tloss:', loss.item(),
                        ';\tmask density:', mask_density.item(),
                        ';\tpred:', ypred)

            if self.writer is not None:
                self.writer.add_scalar(gidx+'mask/density', mask_density, epoch)
                self.writer.add_scalar(gidx+'optimization/lr', explainer.optimizer.param_groups[0]['lr'], epoch)
                if epoch % 25 == 0:
                    explainer.log_mask(epoch)
                    explainer.log_masked_adj(epoch)
        print('finished training in', time.time()-begin_time)

        feat = x[0] if x is not None else None

        G_orig = io_utils.denoise_graph(self.adj[graph_idx], feat=feat)
        io_utils.log_graph(self.writer, G_orig, 'explain/gidx_{}'.format(graph_idx),
            identify_self=False, nodecolor='feat', label_node_feat=True, args=self.args)
        
        G_denoised = io_utils.denoise_graph(graph.adj.cpu().detach().numpy(), feat=feat, 
            threshold=self.args.threshold, threshold_ratio=self.args.threshold_ratio, threshold_num=self.args.threshold_num,
            tokey=True, max_component=True)
        io_utils.log_graph(self.writer, G_denoised,
            'explain/gidx_{}_label_{}'.format(graph_idx, self.label[graph_idx]),
            identify_self=False, nodecolor='feat', label_node_feat=True, args=self.args)
        
        graph.key = io_utils.subgraph2key(self.args, G_denoised, pred_loss)
        graph.pred_loss = pred_loss

        return graph

    def explain_graphs(self, graph_indices):
        masked_graphs = []

        for graph_idx in graph_indices:
            masked_graph = self.explain(graph_idx=graph_idx)
            masked_graphs.append(masked_graph)

        return masked_graphs


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
        self.feat_mask = None # self.construct_feat_mask(num_nodes, x.size(-1), init_strategy='constant', bidim=False) \
            # if x is not None else None
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

        self.coeffs = {'pred': 1, 'adj_size': 1, 'feat_size': 1, 'adj_ent': 1, 'feat_ent': 1}

    def construct_edge_mask(self, num_nodes, init_strategy='normal', const_val=1.0):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        if init_strategy == 'normal':
            std = nn.init.calculate_gain('relu') * math.sqrt(2.0/(num_nodes+num_nodes))
            with torch.no_grad():
                mask.normal_(1.0, std)
                # mask.clamp_(0.0, 1.0)
        elif init_strategy == 'constant':
            nn.init.constant_(mask, const_val)

        if self.args.mask_bias:
            mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.constant_(mask_bias, 0.0)
        else:
            mask_bias = None
       
        return mask, mask_bias

    def construct_feat_mask(self, num_nodes, feat_dim, init_strategy='normal', bidim=True):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, feat_dim)) if bidim else nn.Parameter(torch.FloatTensor(feat_dim))
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
            self.graph.update(self.masked_adj)
        else:
            self.masked_adj = self._masked_adj()
            x = self.x.cuda() if (self.args.gpu and self.x is not None) else self.x
            self.masked_x = None
            if mask_features and x is not None:
                feat_mask = torch.sigmoid(self.feat_mask) if (self.use_sigmoid and self.feat_mask is not None) else self.feat_mask
                marginalize = False
                if feat_mask is not None:
                    if marginalize:
                        std_tensor = torch.ones_like(x, dtype=torch.float) / 2
                        mean_tensor = torch.zeros_like(x, dtype=torch.float) - x
                        z = torch.normal(mean=mean_tensor, std=std_tensor)
                        self.masked_x = x + z * (1 - feat_mask)
                    else:
                        self.masked_x = x * feat_mask
            self.graph.update(self.masked_adj, self.masked_x)

        logits, _, _ = self.model([self.graph])
        res = nn.Softmax(dim=0)(logits.data[0]) 

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

    def loss(self, pred, epoch):
        '''
        Args:
            pred: prediction made by current model
        '''
        # prefix for names
        gidx = 'g'+str(self.graph_idx)+'_'

        # pred
        mi_obj = False
        if mi_obj:
            pred_loss = self.coeffs['pred'] * (-torch.sum(pred * torch.log(pred))).requires_grad_()
        else:
            logit = pred[self.label]
            pred_loss = self.coeffs['pred'] * (-torch.log(logit)).requires_grad_()

        # entropy
        adj_ent = -self.masked_adj * torch.log(self.masked_adj+(self.masked_adj==0).float()) \
            -(1-self.masked_adj) * torch.log(1-self.masked_adj+(self.masked_adj==1).float())
        adj_ent_loss = self.coeffs['adj_ent'] * torch.mean(adj_ent)

        feat_ent = -self.masked_x * torch.log(self.masked_x+(self.masked_x==0).float()) \
            -(1-self.masked_x) * torch.log(1-self.masked_x+(self.masked_x==1).float()) if self.masked_x is not None else torch.zeros(1)
        feat_ent_loss = self.coeffs['feat_ent'] * torch.mean(feat_ent)

        # size
        adj_size_loss = self.coeffs['adj_size'] * torch.mean(self.masked_adj).cpu()
        feat_size_loss = self.coeffs['feat_size'] * torch.mean(self.masked_x).cpu() if self.masked_x is not None else torch.zeros(1)

        loss = pred_loss + adj_ent_loss + feat_ent_loss + adj_size_loss + feat_size_loss

        if self.writer is not None:
            self.writer.add_scalar(gidx+'optimization/1_overall_loss', loss, epoch)
            self.writer.add_scalar(gidx+'optimization/2_adj_size_loss', adj_size_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/3_feat_size_loss', feat_size_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/4_pred_loss', pred_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/5_adj_ent_loss', adj_ent_loss, epoch)
            self.writer.add_scalar(gidx+'optimization/6_feat_ent_loss', feat_ent_loss, epoch)
        
        return loss, pred_loss

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / adj_sum

    def log_mask(self, epoch):
        # prefix for names
        gidx = 'g'+str(self.graph_idx)+'_'

        io_utils.log_matrix(self.writer, self.mask, gidx+'mask/mask', epoch, fig_size=(4, 3), dpi=400)
        if self.feat_mask is not None:        
            io_utils.log_matrix(self.writer, torch.sigmoid(self.feat_mask), gidx+'mask/feat_mask', epoch)

        # use [0] to remove the batch dim
        io_utils.log_matrix(self.writer, self.masked_adj[0], gidx+'mask/adj', epoch, fig_size=(4, 3), dpi=400)
        if self.masked_x is not None:
            io_utils.log_matrix(self.writer, self.masked_x[0], gidx+'mask/feat', epoch, fig_size=(4, 3), dpi=400)
        if self.args.mask_bias:
            io_utils.log_matrix(self.writer, self.mask_bias, gidx+'mask/bias', epoch, fig_size=(4, 3), dpi=400)

    def log_masked_adj(self, epoch):
        # prefix for names
        gidx = 'g'+str(self.graph_idx)+'_'
        
        # use [0] to remove the batch dim
        masked_adj = self.masked_adj[0].cpu().detach().numpy()
        feat = self.x[0] if self.x is not None else None
        G = io_utils.denoise_graph(masked_adj, feat=feat,
                threshold=self.args.threshold, threshold_ratio=self.args.threshold_ratio, threshold_num=self.args.threshold_num,
                tokey=True, max_component=True)
        io_utils.log_graph(self.writer, G, name=gidx+'mask/graph', epoch=epoch,
                identify_self=False, nodecolor='feat', label_node_feat=True, args=self.args)