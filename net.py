import torch
import torch.nn as nn
import torch.nn.functional as F


class ESCMLayer(nn.Module):
    def __init__(self, sparse_feature_number, sparse_feature_dim, num_field,
                 ctr_layer_sizes, cvr_layer_sizes, expert_num, expert_size,
                 tower_size, counterfact_mode, feature_size):
        super(ESCMLayer, self).__init__()
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim
        self.num_field = num_field
        self.ctr_layer_sizes = ctr_layer_sizes
        self.cvr_layer_sizes = cvr_layer_sizes
        self.counterfact_mode = counterfact_mode
        self.expert_num = expert_num
        self.expert_size = expert_size
        self.tower_size = tower_size
        self.feature_size = feature_size
        self.gate_num = 3 if counterfact_mode == "DR" else 2

        self.embedding = nn.Embedding(
            self.sparse_feature_number,
            self.sparse_feature_dim,
            padding_idx=0
        )

        self.experts = nn.ModuleList([
            nn.Linear(self.feature_size, self.expert_size) for _ in range(self.expert_num)
        ])

        self.gates = nn.ModuleList([
            nn.Linear(self.feature_size, self.expert_num) for _ in range(self.gate_num)
        ])

        self.towers = nn.ModuleList([
            nn.Linear(self.expert_size, self.tower_size) for _ in range(self.gate_num)
        ])

        self.tower_outs = nn.ModuleList([
            nn.Linear(self.tower_size, 2) for _ in range(self.gate_num)
        ])

    def forward(self, inputs):
        emb = []
        for data in inputs:
            feat_emb = self.embedding(data)
            feat_emb = torch.sum(feat_emb, dim=1)
            emb.append(feat_emb)
        concat_emb = torch.cat(emb, dim=1)

        expert_outputs = [F.relu(expert(concat_emb)) for expert in self.experts]
        expert_concat = torch.stack(expert_outputs, dim=1)

        output_layers = []
        for i in range(self.gate_num):
            cur_gate = F.softmax(self.gates[i](concat_emb), dim=1)
            cur_gate_expert = torch.sum(expert_concat * cur_gate.unsqueeze(2), dim=1)
            cur_tower = F.relu(self.towers[i](cur_gate_expert))
            out = F.softmax(self.tower_outs[i](cur_tower), dim=1)
            out = torch.clamp(out, min=1e-15, max=1.0 - 1e-15)
            output_layers.append(out)

        ctr_out = output_layers[0]
        cvr_out = output_layers[1]

        ctr_prop_one = ctr_out[:, 1:2]
        cvr_prop_one = cvr_out[:, 1:2]
        ctcvr_prop_one = ctr_prop_one * cvr_prop_one
        ctcvr_prop = torch.cat([1 - ctcvr_prop_one, ctcvr_prop_one], dim=1)

        out_list = [ctr_out, ctr_prop_one, cvr_out, cvr_prop_one, ctcvr_prop, ctcvr_prop_one]
        if self.counterfact_mode == "DR":
            imp_out = output_layers[2]
            imp_prop_one = imp_out[:, 1:2]
            out_list.append(imp_prop_one)

        return out_list
