import argparse
import datetime
import warnings
warnings.filterwarnings("ignore") ####### Modify ####### Remove This Line and Check Warnings For Versions
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset, separate_dataset_su, \
    make_batchnorm_dataset_su, make_batchnorm_stats
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger
from vision.datasets.voc_dataset import VOCDataset
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.nn.multibox_loss import MultiboxLoss

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
ObjDet = True
if args['model_name'] == 'vgg16-ssd':
    create_net = create_vgg_ssd
    conf = vgg_ssd_config
elif args['model_name'] == 'mb1-ssd':
    create_net = create_mobilenetv1_ssd
    conf = mobilenetv1_ssd_config
elif args['model_name'] == 'mb1-ssd-lite':
    create_net = create_mobilenetv1_ssd_lite
    conf = mobilenetv1_ssd_config
elif args['model_name'] == 'sq-ssd-lite':
    create_net = create_squeezenet_ssd_lite
    conf = squeezenet_ssd_config
elif args['model_name'] == 'mb2-ssd-lite':
    # create_net = lambda num: create_mobilenetv2_ssd_lite(num)
    create_net = create_mobilenetv2_ssd_lite
    conf = mobilenetv1_ssd_config
else:
    create_net = None
    conf = None
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    epc = cfg['global']['num_epochs']
    print(f'Number of Epochs {epc}')
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    set_conf = {"image_size": conf.image_size, "image_mean":conf.image_mean,
            "image_std":conf.image_std, "priors": conf.priors,
            "center_variance": conf.center_variance, "size_variance":conf.size_variance} if conf else None
    server_dataset = fetch_dataset(cfg['data_name'], set_conf)
    client_dataset = fetch_dataset(cfg['data_name'], set_conf)
    process_dataset(server_dataset, True if conf else False)
    server_dataset['train'], client_dataset['train'], supervised_idx = separate_dataset_su(server_dataset['train'], client_dataset['train'])
    if conf:
        server_dataset['test'].ids, client_dataset['test'].ids = server_dataset['test'].ids[:1024], client_dataset['test'].ids[-1280:]
    data_loader = make_data_loader(server_dataset, 'global')
    if conf:
        model = create_net(cfg['target_size']).to(cfg["device"])
    else:
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    if cfg['sbn'] == 1:
        batchnorm_dataset = make_batchnorm_dataset_su(server_dataset['train'], client_dataset['train'])
    elif cfg['sbn'] == 0:
        if cfg['data_name'] in ['BDD100K']:
            batchnorm_dataset = None
        else:
            batchnorm_dataset = server_dataset['train']
    else:
        raise ValueError('Not valid sbn')
    data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    if cfg['loss_mode'] != 'sup':
        # metric = Metric({'train': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
        #                  'test': ['Loss', 'Accuracy']})
        metric = Metric({'train': ['Loss', 'PAccuracy'],
                         'test': ['Loss', 'PAccuracy']})
    else:
        metric = Metric({'train': ['Loss', 'PAccuracy'], 'test': ['Loss', 'PAccuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split = result['data_split']
            supervised_idx = result['supervised_idx']
            server = result['server']
            client = result['client']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            server = make_server(model)
            client = make_client(model, data_split)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        server = make_server(model, create_net, conf.priors)
        client = make_client(model, data_split, create_net, conf.priors)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        train_client(batchnorm_dataset, client_dataset['train'], server, client, optimizer, metric, logger, epoch)
        if 'ft' in cfg and cfg['ft'] == 0:
            train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
            logger.reset()
            server.update_parallel(client)
        else:
            logger.reset()
            server.update(client)
            train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        scheduler.step()
        model.load_state_dict(server.model_state_dict)
        if conf:
            test(data_loader['test'], model, metric, logger, epoch)
        else:
            test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
            test(data_loader['test'], test_model, metric, logger, epoch)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def make_server(model, create_net=None, priors=None):
    server = Server(model, create_net, priors)
    return server


def make_client(model, data_split, create_net=None, priors=None):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {'train': data_split['train'][m], 'test': data_split['test'][m]}, create_net, priors)
    return client


def train_client(batchnorm_dataset, client_dataset, server, client, optimizer, metric, logger, epoch):
    logger.safe(True)
    num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    for i in range(num_active_clients):
        client[client_id[i]].active = True
    server.distribute(client, batchnorm_dataset)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        dataset_m = separate_dataset(client_dataset, client[m].data_split['train'])
        if 'batch' not in cfg['loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            dataset_m = client[m].make_dataset(dataset_m, metric, logger)
        if dataset_m is not None:
            client[m].active = True
            client[m].train(dataset_m, lr, metric, logger)
        else:
            client[m].active = False
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            # info = {'info': ['Model: {}'.format(cfg['model_tag']),
            #                  'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
            #                  'Learning rate: {:.6f}'.format(lr),
            #                  'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
            #                  'Epoch Finished Time: {}'.format(epoch_finished_time),
            #                  'Experiment Finished Time: {}'.format(exp_finished_time)]}
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                             'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def train_server(dataset, server, optimizer, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    server.train(dataset, lr, metric, logger)
    _time = (time.time() - start_time)
    epoch_finished_time = datetime.timedelta(seconds=round((cfg['global']['num_epochs'] - epoch) * _time))
    # info = {'info': ['Model: {}'.format(cfg['model_tag']),
    #                  'Train Epoch (S): {}({:.0f}%)'.format(epoch, 100.),
    #                  'Epoch Finished Time: {}'.format(epoch_finished_time)]}
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch (S): {}({:.0f}%)'.format(epoch, 100.)]}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        if conf:
            criterion = MultiboxLoss(conf.priors, iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=cfg['device'])
            for i, data_ in enumerate(data_loader):
                data_size = data_[0].size(0)
                if data_size<2:
                    continue
                data_ = to_device(data_, cfg['device'])
                output_target, output_box = model(data_[0])
                loss_reg, loss_cls =  criterion(output_target, output_box, data_[2], data_[1]) 
                loss = loss_reg + loss_cls
                ss1, ss2 = data_[2].shape
                idd = data_[2].reshape(-1)>0
                evaluation = metric.evaluate(metric.metric_name['test'], {'target': data_[2].reshape(-1)[idd]-1}, {'target': F.softmax(output_target, dim=-1).reshape(ss1*ss2, -1)[idd][:,1:], 'loss': loss})
                # evaluation = metric.evaluate(metric.metric_name['test'], {'target': data_[2].reshape(-1)}, {'target': F.softmax(output_target, dim=-1).reshape(ss1*ss2, -1), 'loss': loss})
                logger.append(evaluation, 'test', n=data_size)
        else:
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()