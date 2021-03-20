import copy
import torch
import numpy as np
from config import cfg
from collections import OrderedDict
from utils import make_optimizer

class Federation:
    def __init__(self, model_state_dict, global_optimizer_state_dict, data_split, target_split):
        self.model_state_dict = model_state_dict
        self.global_optimizer_state_dict = global_optimizer_state_dict
        self.data_split = data_split
        self.target_split = target_split
        self.local_weight = self.make_local_weight(data_split)

    def make_local_weight(self, data_split):
        local_weight = {}
        for split in data_split:
            local_weight[split] = []
            for m in range(len(data_split[split])):
                local_weight[split].append(len(data_split[split][m]))
            local_weight[split] = torch.tensor(local_weight[split])
            local_weight[split] = local_weight[split] / local_weight[split].sum()
        return local_weight

    def distribute(self, user_idx):
        local_parameters = [copy.deepcopy(self.model_state_dict) for _ in range(len(user_idx))]
        return local_parameters

    def update(self, model, global_optimizer, local_parameters):
        global_optimizer.zero_grad()
        for k, v in model.named_parameters():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
                for m in range(len(local_parameters)):
                    tmp_v += self.local_weight['train'][m] * local_parameters[m][k]
                v.grad = (v - tmp_v.to(v.dtype)).detach()
                # v.data = tmp_v.data
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        global_optimizer.step()
        self.model_state_dict = model.state_dict()
        self.global_optimizer_state_dict = global_optimizer.state_dict()
        return
