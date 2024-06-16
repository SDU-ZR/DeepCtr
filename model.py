import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from net import ESCMLayer  # 假设 ESCMLayer 是一个已经定义的 PyTorch 模块


class StaticModel(nn.Module):
    def __init__(self, config):
        super(StaticModel, self).__init__()
        self.config = config
        self._init_hyper_parameters()

        self.escm_model = ESCMLayer(
            self.sparse_feature_number, self.sparse_feature_dim,
            self.num_field, self.ctr_fc_sizes, self.cvr_fc_sizes,
            self.expert_num, self.expert_size, self.tower_size,
            self.counterfact_mode, self.feature_size
        )

    def _init_hyper_parameters(self):
        self.max_len = self.config.get("hyper_parameters.max_len", 3)
        self.global_w = self.config.get("hyper_parameters.global_w", 0.5)
        self.counterfactual_w = self.config.get("hyper_parameters.counterfactual_w", 0.5)
        self.sparse_feature_number = self.config.get("hyper_parameters.sparse_feature_number")
        self.sparse_feature_dim = self.config.get("hyper_parameters.sparse_feature_dim")
        self.num_field = self.config.get("hyper_parameters.num_field")
        self.learning_rate = self.config.get("hyper_parameters.optimizer.learning_rate")
        self.ctr_fc_sizes = self.config.get("hyper_parameters.ctr_fc_sizes")
        self.cvr_fc_sizes = self.config.get("hyper_parameters.cvr_fc_sizes")
        self.expert_num = self.config.get("hyper_parameters.expert_num")
        self.counterfact_mode = self.config.get("runner.counterfact_mode")
        self.expert_size = self.config.get("hyper_parameters.expert_size")
        self.tower_size = self.config.get("hyper_parameters.tower_size")
        self.feature_size = self.config.get("hyper_parameters.feature_size")

    def create_feeds(self, is_infer=False):
        sparse_input_ids = [torch.zeros((None, self.max_len), dtype=torch.int64) for i in range(23)]
        label_ctr = torch.zeros((None, 1), dtype=torch.int64)
        label_cvr = torch.zeros((None, 1), dtype=torch.int64)
        inputs = sparse_input_ids + [label_ctr, label_cvr]
        return inputs

    def counterfact_ipw(self, loss_cvr, ctr_num, O, ctr_out_one):
        PS = ctr_out_one * ctr_num.float()
        min_v = torch.full_like(PS, 0.000001)
        PS = torch.max(PS, min_v)
        IPS = 1.0 / PS
        batch_size = O.size(0)
        IPS = torch.clamp(IPS, min=-15, max=15)  # online trick
        IPS = IPS * batch_size
        IPS = IPS.detach()
        loss_cvr = loss_cvr * IPS * O
        return torch.mean(loss_cvr)

    def counterfact_dr(self, loss_cvr, O, ctr_out_one, imp_out):
        e = loss_cvr - imp_out
        min_v = torch.full_like(ctr_out_one, 0.000001)
        ctr_out_one = torch.max(ctr_out_one, min_v)
        IPS = O.float() / ctr_out_one
        IPS = torch.clamp(IPS, min=-15, max=15)  # online trick
        IPS = IPS.detach()
        loss_error_second = e * IPS
        loss_error = imp_out + loss_error_second
        loss_imp = torch.square(e) * IPS
        loss_dr = loss_error + loss_imp
        return torch.mean(loss_dr)

    def forward(self, inputs, is_infer=False):
        out_list = self.escm_model(inputs[:-2])

        ctr_out, ctr_out_one, cvr_out, cvr_out_one, ctcvr_prop, ctcvr_prop_one = out_list[:6]
        ctr_clk = inputs[-2]
        ctcvr_buy = inputs[-1]
        ctr_num = torch.sum(ctr_clk.float(), dim=0)
        O = ctr_clk.float()

        auc_ctr = self.compute_auc(ctr_out, ctr_clk)
        auc_ctcvr = self.compute_auc(ctcvr_prop, ctcvr_buy)
        auc_cvr = self.compute_auc(cvr_out, ctcvr_buy)

        if is_infer:
            fetch_dict = {
                'auc_ctr': auc_ctr,
                'auc_cvr': auc_cvr,
                'auc_ctcvr': auc_ctcvr
            }
            return fetch_dict

        loss_ctr = F.binary_cross_entropy_with_logits(ctr_out_one, ctr_clk.float())
        loss_cvr = F.binary_cross_entropy_with_logits(cvr_out_one, ctcvr_buy.float())

        if self.counterfact_mode == "DR":
            loss_cvr = self.counterfact_dr(loss_cvr, O, ctr_out_one, out_list[6])
        else:
            loss_cvr = self.counterfact_ipw(loss_cvr, ctr_num, O, ctr_out_one)

        loss_ctcvr = F.binary_cross_entropy_with_logits(ctcvr_prop_one, ctcvr_buy.float())
        cost = loss_ctr + loss_cvr * self.counterfactual_w + loss_ctcvr * self.global_w
        avg_cost = torch.mean(cost)

        fetch_dict = {
            'cost': avg_cost,
            'auc_ctr': auc_ctr,
            'auc_cvr': auc_cvr,
            'auc_ctcvr': auc_ctcvr
        }
        return fetch_dict

    # def compute_auc(self, preds, labels):
    #     preds = torch.sigmoid(preds)
    #     auc = roc_auc_score(labels.cpu().numpy(), preds.detach().cpu().numpy())
    #     return auc

    def create_optimizer(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def infer_net(self, inputs):
        return self.forward(inputs, is_infer=True)


