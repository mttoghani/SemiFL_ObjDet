import copy
import datetime
import numpy as np
import sys
import time
import random
import torch
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, make_batchnorm_stats, FixTransform, MixDataset
from utils import to_device, make_optimizer, collate, to_device
from metrics import Accuracy
from vision.nn.multibox_loss import MultiboxLoss


class Server:
    def __init__(self, model, create_net=None, priors=None):
        self.create_net = create_net
        self.priors = priors
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
            global_optimizer = make_optimizer(model.parameters(), 'global')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())

    def distribute(self, client, batchnorm_dataset=None):
        if self.create_net:
            model = self.create_net(cfg['target_size']).to(cfg["device"])
        else:
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(self.model_state_dict)
        if batchnorm_dataset is not None:
            model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        model_state_dict = save_model_state_dict(model.state_dict())
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
        return

    def update(self, client):
        if 'fmatch' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    if self.create_net:
                        model = self.create_net(cfg['target_size'])
                    else:
                        model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        elif 'fmatch' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False
        return

    def update_parallel(self, client):
        if 'frgd' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server))
                    weight = weight / (2 * (weight.sum() - 1))
                    weight[0] = 1 / 2 if len(valid_client_server) > 1 else 1
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client_server)):
                                tmp_v += weight[m] * valid_client_server[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        elif 'frgd' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                num_valid_client = len(valid_client_server) - 1
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server)) / (num_valid_client // 2 + 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v_1 = v.data.new_zeros(v.size())
                            tmp_v_1 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(1, num_valid_client // 2 + 1):
                                tmp_v_1 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v_2 = v.data.new_zeros(v.size())
                            tmp_v_2 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(num_valid_client // 2 + 1, len(valid_client_server)):
                                tmp_v_2 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v = (tmp_v_1 + tmp_v_2) / 2
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False
        return

    def train(self, dataset, lr, metric, logger):
        if 'fmatch' not in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            if self.create_net:
                model = self.create_net(cfg['target_size']).to(cfg["device"])
            else:
                model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            if self.create_net:
                criterion = MultiboxLoss(self.priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=cfg['device'])
                for epoch in range(1, cfg['server']['num_epochs'] + 1):
                    for i, data_ in enumerate(data_loader):
                        data_size = data_[0].size(0)
                        if data_size<2:
                            continue
                        data_ = to_device(data_, cfg['device'])
                        output_target, output_box = model(data_[0])
                        loss_reg, loss_cls =  criterion(output_target, output_box, data_[2], data_[1]) 
                        loss = loss_reg + loss_cls
                        # print(f'Server Loss:{round(loss.item(),4)}')
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        ss1, ss2 = data_[2].shape
                        idd = data_[2].reshape(-1)>0
                        evaluation = metric.evaluate(['Loss', 'PAccuracy'], {'target': data_[2].reshape(-1)[idd]-1}, {'target': F.softmax(output_target, dim=-1).reshape(ss1*ss2, -1)[idd][:,1:], 'loss': loss})
                        # evaluation = metric.evaluate(['Loss', 'PAccuracy'], {'target': data_[2].reshape(-1)}, {'target': F.softmax(output_target, dim=-1).reshape(ss1*ss2, -1), 'loss': loss})
                        logger.append(evaluation, 'train', n=data_size)
                        if num_batches is not None and i == num_batches - 1:
                            break
            else:
                for epoch in range(1, cfg['server']['num_epochs'] + 1):
                    for i, input in enumerate(data_loader):
                        input = collate(input)
                        input_size = input['data'].size(0)
                        input = to_device(input, cfg['device'])
                        optimizer.zero_grad()
                        output = model(input)
                        output['loss'].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        evaluation = metric.evaluate(['Loss', 'PAccuracy'], input, output)
                        logger.append(evaluation, 'train', n=input_size)
                        if num_batches is not None and i == num_batches - 1:
                            break
        else:
            if self.create_net:
                raise #### Not Implemented ####
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            v.grad[(v.grad.size(0) // 2):] = 0
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'PAccuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return


class Client:
    def __init__(self, client_id, model, data_split, create_net=None, priors=None):
        self.create_net = create_net
        self.priors = priors
        self.client_id = client_id
        self.data_split = data_split
        self.model_state_dict = save_model_state_dict(model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_phi_parameters(), 'local')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
        self.verbose = cfg['verbose']

    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg['threshold'])
        return hard_pseudo_label, mask

    def make_dataset(self, dataset, metric, logger):
        if 'sup' in cfg['loss_mode']:
            return dataset
        elif 'fix' in cfg['loss_mode']:
            with torch.no_grad():
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                if self.create_net:
                    model = self.create_net(cfg['target_size']).to(cfg["device"])
                    criterion = MultiboxLoss(self.priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=cfg['device'])
                else:
                    model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
                model.load_state_dict(self.model_state_dict)
                model.train(False)
                output = []
                target = []
                data = [] if self.create_net else None
                box = [] if self.create_net else None
                box_o = [] if self.create_net else None
                for i, input in enumerate(data_loader):
                    if self.create_net:
                        input = to_device(input, cfg['device'])
                        output_ = model(input[0])
                        data.append(input[0].cpu())
                        box_o.append(input[1].cpu())
                        output.append(output_[0].cpu())
                        box.append(output_[1].cpu())
                        target.append(input[2].cpu())
                    else:
                        input = collate(input)
                        input = to_device(input, cfg['device'])
                        output_ = model(input)
                        output_i = output_['target']
                        target_i = input['target']
                        output.append(output_i.cpu())
                        target.append(target_i.cpu())
                output_, input_ = {}, {}
                output_['target'] = torch.cat(output, dim=0)
                input_['target'] = torch.cat(target, dim=0)
                new_data = torch.cat(data, dim=0) if self.create_net else None
                new_box = torch.cat(box, dim=0) if self.create_net else None
                output_['target'] = F.softmax(output_['target'], dim=-1)
                new_target, mask = self.make_hard_pseudo_label(output_['target'])
                output_['mask'] = mask
                if self.create_net:
                    ss1, ss2 = input_['target'].shape
                    output_['loss_reg'], output_['loss_cls'] = criterion(output_['target'], new_box, input_['target'], torch.cat(box_o, dim=0))
                    output_['loss'] = output_['loss_reg'] + output_['loss_cls']
                    idd = input_['target'].reshape(-1)>0
                    evaluation = metric.evaluate(['Loss', 'PAccuracy'], {'target': input_['target'].reshape(-1)[idd]-1}, {'target': output_['target'].reshape(ss1*ss2, -1)[idd][:,1:], 'loss': output_['loss']})
                    # evaluation = metric.evaluate(['Loss', 'PAccuracy'], {'target': input_['target'].reshape(-1)}, {'target': output_['target'].reshape(ss1*ss2, -1), 'loss': output_['loss']})
                    logger.append(evaluation, 'train', n=ss1*ss2)
                    #####################################################################################################################################################################################################################
                    # tmp_input_, tmp_output_ = {}, {}
                    # tmp_input_['target'] = input_['target'][input_['target']>0]-1
                    # tmp_output_['target'], tmp_output_['mask'] = (output_['target'][:,:,1:][(input_['target']>0).unsqueeze(-1).repeat(1,1,cfg['target_size']-1)]).reshape(-1, cfg['target_size']-1), output_['mask'][input_['target']>0]
                    # tmp_output_['loss_reg'], tmp_output_['loss_cls'] = criterion(output_['target'], new_box, input_['target'], torch.cat(box_o, dim=0))
                    # tmp_output_['loss'] = tmp_output_['loss_reg'] + tmp_output_['loss_cls']
                    # evaluation = metric.evaluate(['Loss', 'PAccuracy'], tmp_input_, tmp_output_)
                    # logger.append(evaluation, 'train', n=tmp_input_['target'].shape[0])
                    #####################################################################################################################################################################################################################
                    # mask[0,0], mask[1,0] = True, True ####### Modify ####### Remove
                    mask = mask.sum(1)>5
                    # mask = mask.any(1) ##### This may be modified to ratio #####
                    # print(mask)
                else:
                    evaluation = metric.evaluate(['Loss', 'PAccuracy'], input_, output_)
                    logger.append(evaluation, 'train', n=len(input_['target']))
                    # mask[0], mask[1] = True, True ####### Modify ####### Remove
                if torch.any(mask):
                    if(mask.sum()<2): ##### This may be modified to repeating the single data two times #####
                        return None
                    print(f'Client {self.client_id}: {mask.sum()} samples')
                    # print(mask) ####### Modify ####### Remove
                    if self.create_net:
                        dataset.data = list(new_data) # new_data.tolist()
                        dataset.box = list(new_box) # new_box.tolist()
                        dataset.target = list(new_target)
                    fix_dataset = copy.deepcopy(dataset)
                    mask = mask.tolist()
                    if self.create_net:
                        fix_dataset.box = list(compress(fix_dataset.box, mask))
                        fix_dataset.ids = list(compress(fix_dataset.ids, mask))
                    else:
                        fix_dataset.target = new_target.tolist()
                    fix_dataset.data = list(compress(fix_dataset.data, mask))
                    fix_dataset.target = list(compress(fix_dataset.target, mask))
                    fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}
                    if 'mix' in cfg['loss_mode']:
                        mix_dataset = copy.deepcopy(dataset)
                        if self.create_net:
                            idx = torch.randint(len(mix_dataset.data), (len(fix_dataset.data),))
                            mix_dataset.data = [mix_dataset.data[i] for i in idx]
                            mix_dataset.box = [mix_dataset.box[i] for i in idx]
                            mix_dataset.target = [mix_dataset.target[i] for i in idx]
                            mix_dataset.ids = [mix_dataset.ids[i] for i in idx]
                            mix_dataset.other = {'id': list(range(len(mix_dataset.data)))}
                        else:
                            mix_dataset.target = new_target.tolist()
                            mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                    else:
                        mix_dataset = None
                    return fix_dataset, mix_dataset
                else:
                    return None
        else:
            raise ValueError('Not valid client loss mode')

    def train(self, dataset, lr, metric, logger):
        if cfg['loss_mode'] == 'sup':
            if self.create_net:
                raise #### Not Implemented ####
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' not in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            if self.create_net:
                raise #### Not Implemented ####
            fix_dataset, _ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(fix_data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'PAccuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, mix_dataset = dataset
            if self.create_net:
                model = self.create_net(cfg['target_size']).to(cfg["device"])
            else:
                fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
                mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
                model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))

            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            if self.create_net:
                criterion = MultiboxLoss(self.priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=cfg['device'])
                for epoch in range(1, cfg['client']['num_epochs'] + 1):
                    sz = len(fix_dataset)
                    cbs  = cfg['client']['batch_size']['train']
                    for i in range((sz//cbs)+1):
                        i_s, i_e = cbs*i, min(cbs*(i+1),sz)
                        if (i_e-i_s<2):
                            continue
                        data_ = {'f_data': fix_dataset.data[i_s:i_e], 'f_box': fix_dataset.box[i_s:i_e], 'f_target': fix_dataset.target[i_s:i_e], 
                                'm_data': mix_dataset.data[i_s:i_e], 'm_box': mix_dataset.box[i_s:i_e], 'm_target': mix_dataset.target[i_s:i_e]}
                        data_ = collate(data_)
                        data_size = data_['f_data'].size(0)
                        data_['lam'] = self.beta.sample()[0]
                        data_['m_data'] = (data_['lam'] * data_['f_data'] + (1 - data_['lam']) * data_['m_data']).detach()
                        data_ = to_device(data_, cfg['device'])
                        data_['f_output_target'], data_['f_output_box'] = model(data_['f_data'])
                        data_['m_output_target'], data_['m_output_box'] = model(data_['m_data'])
                        losses = {}
                        losses['ff_reg'], losses['ff_cls'] =  criterion(data_['f_output_target'], data_['f_output_box'], data_['f_target'], data_['f_box']) 
                        losses['mf_reg'], losses['mf_cls'] =  criterion(data_['m_output_target'], data_['m_output_box'], data_['f_target'], data_['f_box']) 
                        losses['mm_reg'], losses['mm_cls'] =  criterion(data_['m_output_target'], data_['m_output_box'], data_['m_target'], data_['m_box']) 
                        loss = (losses['ff_reg']+losses['ff_cls']) + data_['lam']*(losses['mf_reg']+losses['mf_cls']) + (1-data_['lam'])*(losses['mm_reg']+losses['mm_cls'])
                        # print(f'Client {self.client_id} Loss:{round(loss.item(),4)}')
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        ss1, ss2 = data_['f_target'].shape
                        idd = data_['f_target'].reshape(-1)>0
                        evaluation = metric.evaluate(['Loss', 'PAccuracy'], {'target': data_['f_target'].reshape(-1)[idd]-1}, {'target': F.softmax(data_['f_output_target'], dim=-1).reshape(ss1*ss2, -1)[idd][:,1:], 'loss': loss/2})
                        # evaluation = metric.evaluate(['Loss', 'PAccuracy'], {'target': data_['f_target'].reshape(-1)}, {'target': F.softmax(data_['f_output_target'], dim=-1).reshape(ss1*ss2, -1), 'loss': loss/2})
                        logger.append(evaluation, 'train', n=data_size)
            else:
                for epoch in range(1, cfg['client']['num_epochs'] + 1):
                    for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                        input = {'data': fix_input['data'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                                'mix_data': mix_input['data'], 'mix_target': mix_input['target']}
                        input = collate(input)
                        input_size = input['data'].size(0)
                        input['lam'] = self.beta.sample()[0]
                        input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
                        input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                        input['loss_mode'] = cfg['loss_mode']
                        input = to_device(input, cfg['device'])
                        optimizer.zero_grad()
                        output = model(input)
                        output['loss'].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                        evaluation = metric.evaluate(['Loss', 'PAccuracy'], input, output)
                        logger.append(evaluation, 'train', n=input_size)
                        if num_batches is not None and i == num_batches - 1:
                            break
        elif 'batch' in cfg['loss_mode'] or 'frgd' in cfg['loss_mode'] or 'fmatch' in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            if 'fmatch' in cfg['loss_mode']:
                optimizer = make_optimizer(model.make_phi_parameters(), 'local')
            else:
                optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    with torch.no_grad():
                        model.train(False)
                        input_ = collate(input)
                        input_ = to_device(input_, cfg['device'])
                        output_ = model(input_)
                        output_i = output_['target']
                        output_['target'] = F.softmax(output_i, dim=-1)
                        new_target, mask = self.make_hard_pseudo_label(output_['target'])
                        output_['mask'] = mask
                        evaluation = metric.evaluate(['Loss', 'PAccuracy'], input_, output_)
                        logger.append(evaluation, 'train', n=len(input_['target']))
                    if torch.all(~mask):
                        continue
                    model.train(True)
                    input = {'data': input['data'][mask], 'aug': input['aug'][mask], 'target': new_target[mask]}
                    input = to_device(input, cfg['device'])
                    input_size = input['data'].size(0)
                    input['loss_mode'] = 'fix'
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'PAccuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            raise ValueError('Not valid client loss mode')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return


def save_model_state_dict(model_state_dict):
    return {k: v.cpu() for k, v in model_state_dict.items()}


def save_optimizer_state_dict(optimizer_state_dict):
    optimizer_state_dict_ = {}
    for k, v in optimizer_state_dict.items():
        if k == 'state':
            optimizer_state_dict_[k] = to_device(optimizer_state_dict[k], 'cpu')
        else:
            optimizer_state_dict_[k] = copy.deepcopy(optimizer_state_dict[k])
    return optimizer_state_dict_
